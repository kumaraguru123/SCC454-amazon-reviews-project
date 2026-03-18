import time
import random
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error


REVIEWS_PATH = Path("cleaned_data/reviews_clean.parquet")
PRODUCTS_PATH = Path("cleaned_data/products_clean.parquet")

TOPK = 10

# Hyperparameters
MIN_USER_REVIEWS = 3
MIN_ITEM_REVIEWS = 3
KNN_K = 50


# -------------------------------------------------
# Load data
# -------------------------------------------------
def load_data():
    reviews = pd.read_parquet(REVIEWS_PATH)
    products = pd.read_parquet(PRODUCTS_PATH)

    reviews = reviews.dropna(subset=["user_id", "parent_asin", "rating"]).copy()
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce").fillna(0).astype(float)

    if "timestamp" in reviews.columns:
        reviews["timestamp"] = pd.to_datetime(reviews["timestamp"], errors="coerce")

    products["title"] = products.get("title", "").fillna("").astype(str)

    def safe_text(x):
        if isinstance(x, list):
            return " ".join(map(str, x))
        if x is None:
            return ""
        if isinstance(x, float) and pd.isna(x):
            return ""
        return str(x)

    products["description"] = products.get("description", "").apply(safe_text)
    products["text"] = (products["title"] + " " + products["description"]).str.strip()

    return reviews, products


# -------------------------------------------------
# Train/test split
# -------------------------------------------------
def train_test_split_by_time(reviews):
    df = reviews.copy()

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values(["user_id", "timestamp"])
    else:
        df = df.sample(frac=1, random_state=42).sort_values(["user_id"])

    idx_test = df.groupby("user_id").tail(1).index
    test = df.loc[idx_test].copy()
    train = df.drop(idx_test).copy()

    train_users = set(train["user_id"].unique())
    test = test[test["user_id"].isin(train_users)].copy()

    return train, test


# -------------------------------------------------
# Build mappings + sparse matrix
# -------------------------------------------------
def build_mappings(train):
    users = train["user_id"].unique()
    items = train["parent_asin"].unique()

    u2i = {u: i for i, u in enumerate(users)}
    i2i = {a: i for i, a in enumerate(items)}

    return u2i, i2i, users, items


def build_user_item_matrix(train, u2i, i2i):
    rows = train["user_id"].map(u2i).to_numpy()
    cols = train["parent_asin"].map(i2i).to_numpy()
    vals = train["rating"].to_numpy()

    mat = csr_matrix((vals, (rows, cols)), shape=(len(u2i), len(i2i)))
    return mat


# -------------------------------------------------
# Baselines
# -------------------------------------------------
def rec_popularity(train, user_seen, topk=TOPK):
    pop = (
        train.groupby("parent_asin")
        .agg(n=("rating", "size"), avg=("rating", "mean"))
        .reset_index()
    )

    pop["score"] = pop["avg"] * np.log1p(pop["n"])
    pop = pop.sort_values("score", ascending=False)

    recs = [a for a in pop["parent_asin"].tolist() if a not in user_seen]
    return recs[:topk]


def rec_random(train, user_seen, topk=TOPK):
    all_items = [a for a in train["parent_asin"].unique().tolist() if a not in user_seen]
    random.shuffle(all_items)
    return all_items[:topk]


# -------------------------------------------------
# Item-based CF
# -------------------------------------------------
class ItemKNN:
    def __init__(self, mat, items, k=50):
        self.items = items
        self.k = k
        self.item_mat = normalize(mat.T, norm="l2")
        self.nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(k + 1, self.item_mat.shape[0])
        )
        self.nn.fit(self.item_mat)

    def recommend(self, user_vector, user_seen, topk=TOPK):
        liked_idx = user_vector.indices

        if len(liked_idx) == 0:
            return []

        scores = {}

        for it in liked_idx:
            dist, nbrs = self.nn.kneighbors(self.item_mat[it], return_distance=True)
            dist = dist.flatten()
            nbrs = nbrs.flatten()

            for d, nb in zip(dist, nbrs):
                if nb == it:
                    continue
                asin = self.items[nb]
                if asin in user_seen:
                    continue
                sim = 1.0 - float(d)
                scores[asin] = scores.get(asin, 0.0) + sim

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [a for a, _ in ranked[:topk]]

    def predict_rating(self, user_vector, item_idx):
        liked_idx = user_vector.indices
        liked_vals = user_vector.data

        if len(liked_idx) == 0:
            return 3.0

        preds = []
        weights = []

        for it, rating in zip(liked_idx, liked_vals):
            dist, nbrs = self.nn.kneighbors(self.item_mat[it], return_distance=True)
            dist = dist.flatten()
            nbrs = nbrs.flatten()

            for d, nb in zip(dist, nbrs):
                if nb == item_idx:
                    sim = 1.0 - float(d)
                    preds.append(rating * sim)
                    weights.append(sim)

        if len(weights) == 0 or sum(weights) == 0:
            return 3.0

        return sum(preds) / sum(weights)


# -------------------------------------------------
# User-based CF
# -------------------------------------------------
class UserKNN:
    def __init__(self, mat, users, k=50):
        self.users = users
        self.k = k
        self.user_mat = normalize(mat, norm="l2")
        self.nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(k + 1, self.user_mat.shape[0])
        )
        self.nn.fit(self.user_mat)

    def recommend(self, user_index, mat, items, user_seen, topk=TOPK):
        dist, nbrs = self.nn.kneighbors(self.user_mat[user_index], return_distance=True)
        dist = dist.flatten()
        nbrs = nbrs.flatten()

        scores = {}

        for d, nb in zip(dist, nbrs):
            if nb == user_index:
                continue

            sim = 1.0 - float(d)
            row = mat[nb]

            for it, rating in zip(row.indices, row.data):
                asin = items[it]
                if asin in user_seen:
                    continue
                scores[asin] = scores.get(asin, 0.0) + sim * float(rating)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [a for a, _ in ranked[:topk]]

    def predict_rating(self, user_index, mat, item_idx):
        dist, nbrs = self.nn.kneighbors(self.user_mat[user_index], return_distance=True)
        dist = dist.flatten()
        nbrs = nbrs.flatten()

        preds = []
        weights = []

        for d, nb in zip(dist, nbrs):
            if nb == user_index:
                continue

            sim = 1.0 - float(d)
            row = mat[nb]

            if item_idx in row.indices:
                pos = np.where(row.indices == item_idx)[0][0]
                rating = row.data[pos]
                preds.append(rating * sim)
                weights.append(sim)

        if len(weights) == 0 or sum(weights) == 0:
            return 3.0

        return sum(preds) / sum(weights)


# -------------------------------------------------
# Content-based TF-IDF
# -------------------------------------------------
class ContentTFIDF:
    def __init__(self, products):
        self.products = products.copy()
        self.asins = self.products["parent_asin"].astype(str).tolist()
        self.a2idx = {a: i for i, a in enumerate(self.asins)}

        vect = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words="english"
        )

        self.X = vect.fit_transform(self.products["text"].fillna("").astype(str))
        self.nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(200, self.X.shape[0])
        )
        self.nn.fit(self.X)

    def similar_items(self, asin, topn=50):
        if asin not in self.a2idx:
            return []

        idx = self.a2idx[asin]
        dist, nbrs = self.nn.kneighbors(self.X[idx], return_distance=True)
        nbrs = nbrs.flatten().tolist()

        out = [self.asins[i] for i in nbrs if i != idx]
        return out[:topn]

    def recommend_for_user(self, user_seen, exclude, topk=TOPK):
        scores = {}

        for asin in user_seen[-5:]:
            for cand in self.similar_items(asin, topn=100):
                if cand in exclude:
                    continue
                scores[cand] = scores.get(cand, 0.0) + 1.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [a for a, _ in ranked[:topk]]

    def recommend_cold_item(self, cold_items, topk=TOPK):
        # for truly new products, fallback to showing them directly
        return cold_items[:topk]


# -------------------------------------------------
# Metrics
# -------------------------------------------------
def hit_rate_at_k(recs, true_item):
    return 1.0 if true_item in recs else 0.0


def precision_at_k(recs, true_item, k):
    return (1.0 / k) if true_item in recs[:k] else 0.0


def ndcg_at_k(recs, true_item, k):
    if true_item not in recs[:k]:
        return 0.0
    rank = recs.index(true_item) + 1
    return 1.0 / np.log2(rank + 1)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate_methods(train, test, products, knn_k=KNN_K):
    u2i, i2i, users, items = build_mappings(train)
    mat = build_user_item_matrix(train, u2i, i2i)

    user_counts = train.groupby("user_id").size().to_dict()
    item_counts = train.groupby("parent_asin").size().to_dict()

    item_knn = ItemKNN(mat, items, k=knn_k)
    user_knn = UserKNN(mat, users, k=knn_k)
    content = ContentTFIDF(products.copy())

    user_hist = (
        train.sort_values(["user_id", "timestamp"]) if "timestamp" in train.columns else train
    ).groupby("user_id")["parent_asin"].apply(list).to_dict()

    results = []
    rating_preds = []

    sample_test = test.sample(n=min(2000, len(test)), random_state=42)

    all_train_items = train["parent_asin"].unique().tolist()
    cold_items_list = [a for a, c in item_counts.items() if c < MIN_ITEM_REVIEWS]

    for _, row in sample_test.iterrows():
        uid = row["user_id"]
        true_item = row["parent_asin"]
        true_rating = row["rating"]

        if uid not in u2i:
            continue

        seen = set(user_hist.get(uid, []))
        seen_plus_true = set(seen) | {true_item}

        cold_user = user_counts.get(uid, 0) < MIN_USER_REVIEWS
        cold_item = item_counts.get(true_item, 0) < MIN_ITEM_REVIEWS

        # popularity
        t0 = time.perf_counter()
        rec0 = rec_popularity(train, user_seen=seen_plus_true, topk=TOPK)
        t1 = time.perf_counter()

        # random baseline
        t2 = time.perf_counter()
        rec_rand = rec_random(train, user_seen=seen_plus_true, topk=TOPK)
        t3 = time.perf_counter()

        # item knn
        t4 = time.perf_counter()
        if cold_user:
            rec1 = rec0
        else:
            uidx = u2i[uid]
            rec1 = item_knn.recommend(mat[uidx], user_seen=seen_plus_true, topk=TOPK)
            if len(rec1) < TOPK:
                rec1 += [x for x in rec0 if x not in set(rec1)][:TOPK - len(rec1)]
        t5 = time.perf_counter()

        # user knn
        t6 = time.perf_counter()
        if cold_user:
            rec2 = rec0
        else:
            uidx = u2i[uid]
            rec2 = user_knn.recommend(uidx, mat, items, user_seen=seen_plus_true, topk=TOPK)
            if len(rec2) < TOPK:
                rec2 += [x for x in rec0 if x not in set(rec2)][:TOPK - len(rec2)]
        t7 = time.perf_counter()

        # content tfidf
        t8 = time.perf_counter()
        hist = user_hist.get(uid, [])
        if cold_user:
            rec3 = rec0
        elif cold_item:
            rec3 = content.recommend_cold_item(cold_items_list, topk=TOPK)
        else:
            rec3 = content.recommend_for_user(hist, exclude=seen_plus_true, topk=TOPK)
            if len(rec3) < TOPK:
                rec3 += [x for x in rec0 if x not in set(rec3)][:TOPK - len(rec3)]
        t9 = time.perf_counter()

        for name, recs, dt in [
            ("popularity", rec0, t1 - t0),
            ("random", rec_rand, t3 - t2),
            ("item_knn", rec1, t5 - t4),
            ("user_knn", rec2, t7 - t6),
            ("content_tfidf", rec3, t9 - t8),
        ]:
            results.append({
                "method": name,
                "hit@10": hit_rate_at_k(recs, true_item),
                "precision@10": precision_at_k(recs, true_item, 10),
                "ndcg@10": ndcg_at_k(recs, true_item, 10),
                "time_sec": dt,
                "cold_user": cold_user,
                "cold_item": cold_item
            })

        # rating prediction
        if true_item in i2i:
            item_idx = i2i[true_item]
            uidx = u2i[uid]

            pred_item = item_knn.predict_rating(mat[uidx], item_idx)
            pred_user = user_knn.predict_rating(uidx, mat, item_idx)

            rating_preds.append(("item_knn", true_rating, pred_item))
            rating_preds.append(("user_knn", true_rating, pred_user))

    res = pd.DataFrame(results)

    summary = res.groupby("method").agg(
        hit10=("hit@10", "mean"),
        p10=("precision@10", "mean"),
        ndcg10=("ndcg@10", "mean"),
        avg_time=("time_sec", "mean")
    ).reset_index()

    cold_user_summary = res[res["cold_user"]].groupby("method").agg(
        hit10=("hit@10", "mean"),
        ndcg10=("ndcg@10", "mean"),
        avg_time=("time_sec", "mean"),
        n=("method", "size")
    ).reset_index()

    cold_item_summary = res[res["cold_item"]].groupby("method").agg(
        hit10=("hit@10", "mean"),
        ndcg10=("ndcg@10", "mean"),
        avg_time=("time_sec", "mean"),
        n=("method", "size")
    ).reset_index()

    pred_df = pd.DataFrame(rating_preds, columns=["method", "true", "pred"])
    rating_summary = pred_df.groupby("method").apply(
        lambda x: pd.Series({
            "rmse": np.sqrt(mean_squared_error(x["true"], x["pred"])),
            "mae": mean_absolute_error(x["true"], x["pred"])
        })
    ).reset_index()

    return summary, cold_user_summary, cold_item_summary, rating_summary


# -------------------------------------------------
# Hyperparameter tuning
# -------------------------------------------------
def hyperparameter_tuning(train, test, products):
    rows = []

    for k in [20, 50, 100]:
        print(f"\nTuning KNN_K={k}")
        summary, _, _, _ = evaluate_methods(train, test, products, knn_k=k)

        for _, row in summary.iterrows():
            if row["method"] in ["item_knn", "user_knn"]:
                rows.append({
                    "KNN_K": k,
                    "method": row["method"],
                    "hit10": row["hit10"],
                    "ndcg10": row["ndcg10"],
                    "avg_time": row["avg_time"]
                })

    tune_df = pd.DataFrame(rows)
    return tune_df


# -------------------------------------------------
# Top-10 for 5 users
# -------------------------------------------------
def show_top10_for_5_users(train, products, knn_k=KNN_K):
    counts = train.groupby("user_id").size().sort_values(ascending=False)
    sample_users = counts.head(5).index.tolist()

    u2i, i2i, users, items = build_mappings(train)
    mat = build_user_item_matrix(train, u2i, i2i)

    item_knn = ItemKNN(mat, items, k=knn_k)
    user_knn = UserKNN(mat, users, k=knn_k)
    content = ContentTFIDF(products.copy())

    user_hist = (
        train.sort_values(["user_id", "timestamp"]) if "timestamp" in train.columns else train
    ).groupby("user_id")["parent_asin"].apply(list).to_dict()

    out = {}

    for uid in sample_users:
        seen = set(user_hist.get(uid, []))
        rec0 = rec_popularity(train, seen, topk=TOPK)
        rec_rand = rec_random(train, seen, topk=TOPK)

        uidx = u2i[uid]
        rec1 = item_knn.recommend(mat[uidx], user_seen=seen, topk=TOPK)
        rec2 = user_knn.recommend(uidx, mat, items, user_seen=seen, topk=TOPK)
        rec3 = content.recommend_for_user(user_hist[uid], exclude=seen, topk=TOPK)

        out[uid] = {
            "popularity": rec0,
            "random": rec_rand,
            "item_knn": rec1,
            "user_knn": rec2,
            "content_tfidf": rec3
        }

    return out


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    reviews, products = load_data()
    train, test = train_test_split_by_time(reviews)

    print("Train rows:", len(train), "Test rows:", len(test))
    print("Unique users:", train["user_id"].nunique(), "Unique items:", train["parent_asin"].nunique())

    print("\n--- Evaluation ---")
    summary, cold_user_summary, cold_item_summary, rating_summary = evaluate_methods(train, test, products)
    print(summary)

    print("\n--- Cold-start users ---")
    print(cold_user_summary)

    print("\n--- Cold-start items ---")
    print(cold_item_summary)

    print("\n--- Rating prediction metrics ---")
    print(rating_summary)

    print("\n--- Hyperparameter tuning ---")
    tuning = hyperparameter_tuning(train, test, products)
    print(tuning)

    print("\n--- Top-10 recommendations for 5 sample users ---")
    top10 = show_top10_for_5_users(train, products)
    for uid, recs in top10.items():
        print("\nUser:", uid)
        for m, lst in recs.items():
            print(f"  {m}: {lst}")

    print("\nDONE")


if __name__ == "__main__":
    main()
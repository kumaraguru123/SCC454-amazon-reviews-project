import time
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from datasketch import MinHash, MinHashLSH

import faiss


REVIEWS_PATH = "cleaned_data/reviews_clean.parquet"
PRODUCTS_PATH = "cleaned_data/products_clean.parquet"
USERS_FEAT_PATH = "cleaned_data/users_features.parquet"


def timer(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def pick_5_samples(series):
    vals = series.dropna().astype(str).unique()
    return vals[:5].tolist() if len(vals) >= 5 else vals.tolist()


def ensure_str(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join([str(v) for v in x])
    return str(x)


# -------------------------
# PRODUCT SIMILARITY (3)
# -------------------------
def build_product_text(products: pd.DataFrame) -> pd.Series:
    title = products.get("title", "").map(ensure_str)
    desc = products.get("description", "").map(ensure_str)
    feats = products.get("features", "").map(ensure_str)
    return (title + " " + desc + " " + feats).fillna("")


# A1) Exact cosine on TF-IDF
def prod_exact_tfidf(products: pd.DataFrame, query_asin: str, topn: int = 10):
    ids = products["parent_asin"].astype(str).tolist()
    text = build_product_text(products)

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(text)

    if str(query_asin) not in ids:
        return []
    q_idx = ids.index(str(query_asin))

    sims = cosine_similarity(X[q_idx], X).ravel()
    sims[q_idx] = -1
    top_idx = np.argsort(-sims)[:topn]
    return [(ids[i], float(sims[i])) for i in top_idx]


# A2) FAISS exact search on SVD-reduced TF-IDF embeddings
def prod_faiss_svd(products: pd.DataFrame, query_asin: str, topn: int = 10):
    ids = products["parent_asin"].astype(str).tolist()
    text = build_product_text(products)

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(text)

    svd = TruncatedSVD(n_components=256, random_state=42)
    emb = svd.fit_transform(X).astype("float32")
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    if str(query_asin) not in ids:
        return []
    q_idx = ids.index(str(query_asin))

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    D, I = index.search(emb[q_idx:q_idx+1], topn + 1)

    out = []
    for sim, idx in zip(D[0], I[0]):
        if idx == q_idx:
            continue
        out.append((ids[int(idx)], float(sim)))
        if len(out) == topn:
            break
    return out


# A3) Approximate-ish: HashingVectorizer + NearestNeighbors
def prod_hash_nn(products: pd.DataFrame, query_asin: str, topn: int = 10):
    ids = products["parent_asin"].astype(str).tolist()
    text = build_product_text(products)

    hv = HashingVectorizer(n_features=2**18, alternate_sign=False, norm="l2")
    X = hv.transform(text)

    svd = TruncatedSVD(n_components=128, random_state=42)
    emb = svd.fit_transform(X).astype("float32")
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    if str(query_asin) not in ids:
        return []
    q_idx = ids.index(str(query_asin))

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=topn + 1)
    nn.fit(emb)
    dists, idxs = nn.kneighbors(emb[q_idx:q_idx+1])

    out = []
    for dist, idx in zip(dists[0], idxs[0]):
        if idx == q_idx:
            continue
        sim = 1.0 - float(dist)
        out.append((ids[int(idx)], sim))
        if len(out) == topn:
            break
    return out


# -------------------------
# USER SIMILARITY (3)
# -------------------------
# B1) Exact cosine on behaviour features
def user_exact_behaviour(users_feat: pd.DataFrame, query_user: str, topn: int = 10):
    users_feat = users_feat.copy()
    users_feat["user_id"] = users_feat["user_id"].astype(str)

    features = ["n_reviews", "avg_rating", "pct_verified", "avg_helpful"]
    X = users_feat[features].fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    ids = users_feat["user_id"].tolist()
    if str(query_user) not in ids:
        return []
    q_idx = ids.index(str(query_user))

    sims = cosine_similarity(Xs[q_idx:q_idx+1], Xs).ravel()
    sims[q_idx] = -1
    top_idx = np.argsort(-sims)[:topn]
    return [(ids[i], float(sims[i])) for i in top_idx]


# B2) SVD latent embedding from interactions + FAISS
def user_svd_faiss(reviews: pd.DataFrame, query_user: str, topn: int = 10):
    df = reviews[["user_id", "parent_asin", "rating"]].dropna().copy()
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)

    users = df["user_id"].unique()
    u2i = {u: i for i, u in enumerate(users)}

    dim = 4096
    U = np.zeros((len(users), dim), dtype="float32")
    for u, p, r in df.itertuples(index=False):
        U[u2i[u], hash(p) % dim] += float(r)

    svd = TruncatedSVD(n_components=128, random_state=42)
    emb = svd.fit_transform(U).astype("float32")
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    q = str(query_user)
    if q not in u2i:
        return []
    q_idx = u2i[q]

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    D, I = index.search(emb[q_idx:q_idx+1], topn + 1)

    out = []
    for sim, idx in zip(D[0], I[0]):
        if idx == q_idx:
            continue
        out.append((users[int(idx)], float(sim)))
        if len(out) == topn:
            break
    return out


# B3) MinHash LSH on sets of reviewed products
def user_minhash_lsh(reviews: pd.DataFrame, query_user: str, topn: int = 10):
    df = reviews[["user_id", "parent_asin"]].dropna().copy()
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)

    user_items = df.groupby("user_id")["parent_asin"].apply(lambda x: set(x.tolist()))
    q = str(query_user)
    if q not in user_items:
        return []

    lsh = MinHashLSH(threshold=0.2, num_perm=128)
    mhs = {}

    for uid, items in user_items.items():
        mh = MinHash(num_perm=128)
        for it in items:
            mh.update(it.encode("utf-8"))
        lsh.insert(uid, mh)
        mhs[uid] = mh

    candidates = [c for c in lsh.query(mhs[q]) if c != q]

    qset = user_items[q]
    scored = []
    for uid in candidates:
        j = len(qset & user_items[uid]) / max(1, len(qset | user_items[uid]))
        scored.append((uid, float(j)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topn]


def main():
    print("Loading parquet...")
    reviews = pd.read_parquet(REVIEWS_PATH)
    products = pd.read_parquet(PRODUCTS_PATH)
    users_feat = pd.read_parquet(USERS_FEAT_PATH)

    sample_products = pick_5_samples(products["parent_asin"])
    sample_users = pick_5_samples(users_feat["user_id"])
    N = 10

    print("\n========================")
    print("PART A: PRODUCT SIMILARITY")
    print("========================")
    for asin in sample_products:
        print(f"\nProduct: {asin}")

        res, dt = timer(prod_exact_tfidf, products, asin, N)
        print(f"1) Exact TF-IDF cosine time={dt:.4f}s")
        print(res)

        res, dt = timer(prod_faiss_svd, products, asin, N)
        print(f"2) FAISS (SVD+IP) time={dt:.4f}s")
        print(res)

        res, dt = timer(prod_hash_nn, products, asin, N)
        print(f"3) Hash+SVD+NN time={dt:.4f}s")
        print(res)

    print("\n========================")
    print("PART B: USER SIMILARITY")
    print("========================")
    for uid in sample_users:
        print(f"\nUser: {uid}")

        res, dt = timer(user_exact_behaviour, users_feat, uid, N)
        print(f"1) Exact behaviour cosine time={dt:.4f}s")
        print(res)

        res, dt = timer(user_svd_faiss, reviews, uid, N)
        print(f"2) SVD latent + FAISS time={dt:.4f}s")
        print(res)

        res, dt = timer(user_minhash_lsh, reviews, uid, N)
        print(f"3) MinHash LSH time={dt:.4f}s")
        print(res)

    print("\nDONE ")

if __name__ == "__main__":
    main()

"""
Task 2 – Similarity Computation Benchmark
Product–Product and User–User Similarity
Includes dataset size + parameter benchmarking
"""

import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import faiss
from datasketch import MinHash, MinHashLSH


REVIEWS_PATH = "cleaned_data/reviews_clean.parquet"
PRODUCTS_PATH = "cleaned_data/products_clean.parquet"
USERS_PATH = "cleaned_data/users_features.parquet"

RESULTS_FILE = "task2_benchmark_results.csv"

N = 10

def timer():
    return time.perf_counter()


def pick_5_samples(series):
    return series.dropna().sample(min(5, len(series)), random_state=42).tolist()


def safe_text(x):
    """Convert list/array/string/None to clean text"""

    if x is None:
        return ""

    # handle lists or numpy arrays
    if isinstance(x, (list, np.ndarray)):
        return " ".join([str(i) for i in x])

    # handle NaN
    if isinstance(x, float) and pd.isna(x):
        return ""

    return str(x)

def product_tfidf_cosine(products):

    products["text"] = (
        products["title"].apply(safe_text)
        + " "
        + products["description"].apply(safe_text)
    )

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")

    X = tfidf.fit_transform(products["text"])

    return X


def product_svd_faiss(X, dim):

    svd = TruncatedSVD(n_components=dim, random_state=42)
    X_reduced = svd.fit_transform(X)

    index = faiss.IndexFlatL2(dim)

    index.add(X_reduced.astype("float32"))

    return X_reduced, index


def product_hash_knn(products):

    text = (
        products["title"].apply(safe_text)
        + " "
        + products["description"].apply(safe_text)
    )

    hv = HashingVectorizer(n_features=4096)

    X = hv.transform(text)

    svd = TruncatedSVD(n_components=64, random_state=42)

    X_red = svd.fit_transform(X)

    knn = NearestNeighbors(metric="cosine")

    knn.fit(X_red)

    return X_red, knn

def user_cosine(users):

    features = users[
        ["n_reviews", "avg_rating", "pct_verified", "avg_helpful"]
    ].fillna(0)

    X = features.values

    sim = cosine_similarity(X)

    return X, sim


def user_faiss_embeddings(reviews):

    pivot = reviews.pivot_table(
        index="user_id",
        columns="parent_asin",
        values="rating",
        fill_value=0
    )

    svd = TruncatedSVD(n_components=64, random_state=42)

    X = svd.fit_transform(pivot)

    index = faiss.IndexFlatL2(64)

    index.add(X.astype("float32"))

    return pivot.index.tolist(), X, index


def user_lsh(reviews):

    user_sets = reviews.groupby("user_id")["parent_asin"].apply(set)

    lsh = MinHashLSH(threshold=0.2, num_perm=64)

    minhashes = {}

    for user, items in user_sets.items():

        m = MinHash(num_perm=64)

        for i in items:
            m.update(i.encode("utf8"))

        lsh.insert(user, m)

        minhashes[user] = m

    return user_sets.index.tolist(), minhashes, lsh

def benchmark():

    reviews = pd.read_parquet(REVIEWS_PATH)
    products = pd.read_parquet(PRODUCTS_PATH)
    users = pd.read_parquet(USERS_PATH)

    dataset_sizes = [10000, 30000, 60000]
    svd_dims = [64, 128, 256]

    rows = []

    for size in dataset_sizes:

        print("\nDataset size:", size)

        p = products.head(size).copy()

        start = timer()

        X = product_tfidf_cosine(p)

        t = timer() - start

        rows.append(["product_tfidf", size, "-", t])

        for dim in svd_dims:

            start = timer()

            Xr, index = product_svd_faiss(X, dim)

            t = timer() - start

            rows.append(["product_faiss", size, dim, t])

        start = timer()

        Xr, knn = product_hash_knn(p)

        t = timer() - start

        rows.append(["product_hash_knn", size, "-", t])

    print("\nUser similarity benchmarking")

    start = timer()

    X, sim = user_cosine(users)

    rows.append(["user_cosine", len(users), "-", timer() - start])

    start = timer()

    ids, X, index = user_faiss_embeddings(reviews.head(50000))

    rows.append(["user_faiss", 50000, 64, timer() - start])

    start = timer()

    ids, hashes, lsh = user_lsh(reviews.head(50000))

    rows.append(["user_lsh", 50000, "-", timer() - start])

    df = pd.DataFrame(rows, columns=["method", "dataset_size", "parameter", "time_sec"])

    df.to_csv(RESULTS_FILE, index=False)

    print("\nBenchmark saved:", RESULTS_FILE)

    print(df)

def demo_queries():

    reviews = pd.read_parquet(REVIEWS_PATH)
    products = pd.read_parquet(PRODUCTS_PATH)
    users = pd.read_parquet(USERS_PATH)

    print("\nRunning similarity examples")

    sample_products = pick_5_samples(products["parent_asin"])

    sample_users = pick_5_samples(users["user_id"])

    print("\nSample products:", sample_products)

    print("Sample users:", sample_users)

    X = product_tfidf_cosine(products)

    sim = cosine_similarity(X)

    for pid in sample_products:

        idx = products.index[products["parent_asin"] == pid][0]

        scores = sim[idx]

        top = np.argsort(-scores)[1:11]

        print("\nProduct:", pid)

        for i in top:
            print(products.iloc[i]["parent_asin"], scores[i])

    features = users[
        ["n_reviews", "avg_rating", "pct_verified", "avg_helpful"]
    ].fillna(0).values

    sim = cosine_similarity(features)

    for uid in sample_users:

        idx = users.index[users["user_id"] == uid][0]

        scores = sim[idx]

        top = np.argsort(-scores)[1:11]

        print("\nUser:", uid)

        for i in top:
            print(users.iloc[i]["user_id"], scores[i])

if __name__ == "__main__":

    benchmark()

    demo_queries()
"""
Task 3 – Product Clustering using TF-IDF Text Features

Second feature representation approach for products.
Clusters products based on title + description text.
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


DATA_PATH = "cleaned_data/products_clean.parquet"

OUTPUT_ASSIGN = "products_clustered_tfidf.parquet"

RANDOM_STATE = 42


def safe_text(x):
    """Convert list/array/string to clean text"""

    if x is None:
        return ""

    if isinstance(x, (list, np.ndarray)):
        return " ".join([str(i) for i in x])

    if isinstance(x, float) and pd.isna(x):
        return ""

    return str(x)


def build_text_features(products):

    text = (
        products["title"].apply(safe_text)
        + " "
        + products["description"].apply(safe_text)
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X = vectorizer.fit_transform(text)

    return X


def find_best_k(X):

    print("\nSearching for best number of clusters")

    best_k = None
    best_score = -1

    for k in range(3, 11):

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)

        print(f"k={k} silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print("\nBest k =", best_k)
    print("Best silhouette =", best_score)

    return best_k


def run_clustering(products):

    X = build_text_features(products)

    best_k = find_best_k(X)

    model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(X)

    products["cluster_tfidf"] = labels

    return products, best_k


def cluster_profiles(products, k):

    print("\nCluster sizes")

    sizes = products["cluster_tfidf"].value_counts().sort_index()

    print(sizes)

    print("\nRepresentative examples")

    for c in range(k):

        sample = products[products["cluster_tfidf"] == c].head(3)

        print("\nCluster", c)

        for _, row in sample.iterrows():
            print(row["parent_asin"], "-", row["title"])


def main():

    print("Loading products...")

    products = pd.read_parquet(DATA_PATH)

    products, k = run_clustering(products)

    products.to_parquet(OUTPUT_ASSIGN, index=False)

    print("\nCluster assignments saved:", OUTPUT_ASSIGN)

    cluster_profiles(products, k)


if __name__ == "__main__":
    main()
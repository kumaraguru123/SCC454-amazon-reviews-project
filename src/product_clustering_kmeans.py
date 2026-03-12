import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

IN_PRODUCTS = "cleaned_data/products_features.parquet"
OUT_PRODUCTS = "cleaned_data/products_clustered_kmeans.parquet"

FEATURES = ["n_reviews", "avg_rating", "avg_helpful", "star_1", "star_2", "star_3", "star_4", "star_5"]

def main():
    df = pd.read_parquet(IN_PRODUCTS)
    X = df[FEATURES].fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Parameter selection: try k and choose best silhouette
    best_k, best_score, best_model = None, -1, None
    for k in [3, 4, 5, 6, 8, 10]:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        print(f"k={k} silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, labels

    df["cluster"] = best_model
    df.to_parquet(OUT_PRODUCTS, index=False)

    print("\nSaved:", OUT_PRODUCTS)
    print("Best k:", best_k, "Silhouette:", best_score)
    print("\nCluster sizes:")
    print(df["cluster"].value_counts())
    print("\nCluster profile (mean features):")
    print(df.groupby("cluster")[FEATURES].mean())

    print("\nRepresentative examples (top 3 per cluster by n_reviews):")
    for c in sorted(df["cluster"].unique()):
        ex = df[df["cluster"] == c].sort_values("n_reviews", ascending=False).head(3)
        print("\nCluster", c)
        print(ex[["parent_asin", "title", "main_category", "n_reviews", "avg_rating"]])

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

IN_PRODUCTS = "cleaned_data/products_features.parquet"
OUT_PRODUCTS = "cleaned_data/products_clustered_dbscan.parquet"

FEATURES = ["n_reviews", "avg_rating", "avg_helpful", "star_1", "star_2", "star_3", "star_4", "star_5"]

def main():
    df = pd.read_parquet(IN_PRODUCTS)
    X = df[FEATURES].fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Try a few settings (simple “parameter selection evidence”)
    best = None
    for eps in [0.6, 0.8, 1.0, 1.2]:
        model = DBSCAN(eps=eps, min_samples=10)
        labels = model.fit_predict(Xs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = (labels == -1).mean()

        # silhouette only valid if >=2 clusters and not all noise
        sil = None
        if n_clusters >= 2 and noise < 0.95:
            sil = silhouette_score(Xs[labels != -1], labels[labels != -1])

        print(f"eps={eps} clusters={n_clusters} noise={noise:.2%} silhouette={sil}")
        best = (eps, labels, n_clusters, noise, sil)

    # just save the last tried; you can choose best manually based on above output
    df["cluster"] = best[1]
    df.to_parquet(OUT_PRODUCTS, index=False)

    print("\nSaved:", OUT_PRODUCTS)
    print("Clusters (including noise=-1):")
    print(df["cluster"].value_counts().head(10))

if __name__ == "__main__":
    main()

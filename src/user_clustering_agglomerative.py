import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

IN_USERS = "cleaned_data/users_features.parquet"
OUT_USERS = "cleaned_data/users_clustered_agglo.parquet"

FEATURES = ["n_reviews", "avg_rating", "pct_verified", "avg_helpful"]

def main():
    df = pd.read_parquet(IN_USERS)
    X = df[FEATURES].fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Pick K by trying a few values and choosing best silhouette
    best_k, best_score, best_labels = None, -1, None
    for k in [3, 4, 5, 6, 8]:
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        print(f"k={k} silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    df["cluster"] = best_labels
    df.to_parquet(OUT_USERS, index=False)

    print("\nSaved:", OUT_USERS)
    print("Best k:", best_k, "Silhouette:", best_score)
    print("\nCluster sizes:")
    print(df["cluster"].value_counts())
    print("\nCluster profile (mean features):")
    print(df.groupby("cluster")[FEATURES].mean())

if __name__ == "__main__":
    main()

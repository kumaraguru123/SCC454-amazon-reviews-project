import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

users = pd.read_parquet("cleaned_data/users_features.parquet")

X = users[["n_reviews","avg_rating","pct_verified","avg_helpful"]]

pca = PCA(n_components=2)
coords = pca.fit_transform(X)

plt.figure(figsize=(7,6))
plt.scatter(coords[:,0], coords[:,1], alpha=0.5)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("User Behaviour Clusters")

plt.grid()
plt.savefig("figure_user_clusters.png", dpi=300)
plt.show()

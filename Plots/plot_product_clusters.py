import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

products = pd.read_parquet("cleaned_data/products_features.parquet")

features = [
    "n_reviews",
    "avg_rating",
    "avg_helpful",
    "star_1",
    "star_2",
    "star_3",
    "star_4",
    "star_5"
]

X = products[features]

pca = PCA(n_components=2)
coords = pca.fit_transform(X)

plt.figure(figsize=(7,6))
plt.scatter(coords[:,0], coords[:,1], alpha=0.5)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Product Clusters")

plt.grid()
plt.savefig("figure_product_clusters.png", dpi=300)
plt.show()
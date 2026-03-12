import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

X = products[features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ks = range(1,11)
inertia = []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(ks, inertia, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")

plt.grid()
plt.savefig("figure_elbow_method.png", dpi=300)
plt.show()
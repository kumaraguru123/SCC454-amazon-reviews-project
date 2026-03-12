import pandas as pd
import matplotlib.pyplot as plt

reviews = pd.read_parquet("cleaned_data/reviews_clean.parquet")

rating_counts = reviews["rating"].value_counts().sort_index()

plt.figure(figsize=(8,5))
rating_counts.plot(kind="bar")

plt.xlabel("Rating")
plt.ylabel("Count")
plt.title("Distribution of Product Ratings")

plt.grid(axis='y')
plt.savefig("figure_rating_distribution.png", dpi=300)
plt.show()
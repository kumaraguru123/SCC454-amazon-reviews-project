import pandas as pd
import matplotlib.pyplot as plt

reviews = pd.read_parquet("cleaned_data/reviews_clean.parquet")

reviews["timestamp"] = pd.to_datetime(reviews["timestamp"])
reviews["year"] = reviews["timestamp"].dt.year

reviews_per_year = reviews.groupby("year").size()

plt.figure(figsize=(8,5))
reviews_per_year.plot()

plt.title("Reviews Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Reviews")

plt.grid()
plt.savefig("figure_reviews_over_time.png", dpi=300)
plt.show()
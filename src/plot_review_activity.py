import pandas as pd
import matplotlib.pyplot as plt

reviews = pd.read_parquet("cleaned_data/reviews_clean.parquet")

reviews_per_user = reviews.groupby("user_id").size()
reviews_per_product = reviews.groupby("parent_asin").size()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(reviews_per_user, bins=50)
plt.title("Reviews per User")
plt.xlabel("Number of Reviews")
plt.ylabel("Users")

plt.subplot(1,2,2)
plt.hist(reviews_per_product, bins=50)
plt.title("Reviews per Product")
plt.xlabel("Number of Reviews")
plt.ylabel("Products")

plt.tight_layout()
plt.savefig("figure_review_activity.png", dpi=300)
plt.show()
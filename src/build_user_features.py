import pandas as pd
from pathlib import Path

IN_REVIEWS = Path("cleaned_data/reviews_clean.parquet")
OUT_USERS = Path("cleaned_data/users_features.parquet")

print("Loading reviews...")
df = pd.read_parquet(IN_REVIEWS)

# Basic cleaning safety
df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)
df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0)

print("Building user features...")
users = (
    df.groupby("user_id")
      .agg(
          n_reviews=("rating", "count"),
          avg_rating=("rating", "mean"),
          pct_verified=("verified_purchase", "mean"),
          avg_helpful=("helpful_vote", "mean"),
      )
      .reset_index()
)

OUT_USERS.parent.mkdir(parents=True, exist_ok=True)
users.to_parquet(OUT_USERS, index=False)

print("Saved:", OUT_USERS)
print("Users:", len(users))
print(users.head())

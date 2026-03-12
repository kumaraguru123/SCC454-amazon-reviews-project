import pandas as pd

REVIEWS_PATH = "cleaned_data/reviews_clean.parquet"
PRODUCTS_PATH = "cleaned_data/products_clean.parquet"
OUT_PRODUCTS = "cleaned_data/products_features.parquet"

def main():
    reviews = pd.read_parquet(REVIEWS_PATH)
    products = pd.read_parquet(PRODUCTS_PATH)

    # Basic review-based product features
    agg = (
        reviews.groupby("parent_asin")
        .agg(
            n_reviews=("rating", "count"),
            avg_rating=("rating", "mean"),
            avg_helpful=("helpful_vote", "mean"),
        )
        .reset_index()
    )

    # Rating distribution (counts per star)
    dist = (
        reviews.pivot_table(index="parent_asin", columns="rating", values="title", aggfunc="count", fill_value=0)
        .reset_index()
    )
    # rename columns like star_1, star_2...
    dist.columns = ["parent_asin"] + [f"star_{int(c)}" for c in dist.columns[1:]]

    feat = products[["parent_asin", "main_category", "title"]].merge(agg, on="parent_asin", how="left").merge(dist, on="parent_asin", how="left")

    # Fill missing
    for c in ["n_reviews", "avg_rating", "avg_helpful"]:
        feat[c] = feat[c].fillna(0)

    # star columns might be missing if no such rating exists in sample
    for s in [1, 2, 3, 4, 5]:
        col = f"star_{s}"
        if col not in feat.columns:
            feat[col] = 0
        feat[col] = feat[col].fillna(0)

    feat.to_parquet(OUT_PRODUCTS, index=False)
    print("Saved:", OUT_PRODUCTS)
    print("Rows:", len(feat))
    print(feat.head())

if __name__ == "__main__":
    main()

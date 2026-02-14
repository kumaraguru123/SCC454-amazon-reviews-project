import pandas as pd
# Settings (edit if you want)
N_REVIEWS = 100_000

REVIEWS_PATH = "data/Electronics.jsonl.gz"
META_PATH = "data/meta_Electronics.jsonl.gz"

OUT_REVIEWS = "cleaned_data/reviews_clean.parquet"
OUT_PRODUCTS = "cleaned_data/products_clean.parquet"

print(f"Loading first {N_REVIEWS:,} reviews...")
# Load reviews subset
reviews = pd.read_json(
    REVIEWS_PATH,
    lines=True,
    compression="gzip",
    nrows=N_REVIEWS
)

print("Reviews loaded:", len(reviews))
print("Review columns:", list(reviews.columns))

# Keep only important columns (these exist in your sample output)
reviews_clean = reviews[
    ["parent_asin", "rating", "title", "text", "timestamp", "helpful_vote", "verified_purchase"]
].copy()

# Basic cleaning
reviews_clean["title"] = reviews_clean["title"].fillna("")
reviews_clean["text"] = reviews_clean["text"].fillna("")
reviews_clean["timestamp"] = pd.to_datetime(reviews_clean["timestamp"], errors="coerce")

# Drop rows missing core IDs
reviews_clean = reviews_clean.dropna(subset=["parent_asin"])

print("Cleaned reviews shape:", reviews_clean.shape)
# Load metadata (full file)
print("Loading metadata (this can take a bit)...")

meta = pd.read_json(
    META_PATH,
    lines=True,
    compression="gzip"
)

print("Metadata loaded:", len(meta))
print("Metadata columns:", list(meta.columns))

# Filter metadata only for products seen in reviews subset
product_ids = set(reviews_clean["parent_asin"].dropna().unique())
products = meta[meta["parent_asin"].isin(product_ids)].copy()

print("Filtered products:", len(products))

# Keep only useful columns (these exist in your metadata output)
products_clean = products[
    ["parent_asin", "title", "description", "main_category", "features", "store", "price", "bought_together"]
].copy()
# Clean product fields
products_clean["title"] = products_clean["title"].fillna("")
products_clean["description"] = products_clean["description"].fillna("")
products_clean["main_category"] = products_clean["main_category"].fillna("")
products_clean["store"] = products_clean["store"].fillna("")

# Robust price cleaning:
# - handles: "", "—", "-", "$1,299.00", text, None
# - converts to float or NaN
price_str = (
    products_clean["price"]
    .astype(str)
    .str.strip()
    .replace({"": None, "—": None, "-": None, "None": None, "nan": None})
)

price_str = price_str.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
products_clean["price"] = pd.to_numeric(price_str, errors="coerce")

# Remove duplicates by product id (keep first)
products_clean = products_clean.drop_duplicates(subset=["parent_asin"])

print("Final products shape:", products_clean.shape)

# ----------------------------
# Save to Parquet
# ----------------------------
print("Saving parquet files...")

reviews_clean.to_parquet(OUT_REVIEWS, index=False)
products_clean.to_parquet(OUT_PRODUCTS, index=False)

print("✅ Clean datasets saved:")
print(" -", OUT_REVIEWS)
print(" -", OUT_PRODUCTS)

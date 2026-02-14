import pandas as pd
# --------- Load small sample from reviews ----------
reviews_path = "data/Electronics.jsonl.gz"

print("Reading sample from reviews...")

reviews_sample = pd.read_json(
    reviews_path,
    lines=True,
    compression="gzip",
    nrows=5  # only read first 5 rows
)

print("\nReviews Sample:")
print(reviews_sample.head())

print("\nReview Columns:")
print(reviews_sample.columns)
# --------- Load small sample from metadata ----------
meta_path = "data/meta_Electronics.jsonl.gz"

print("\nReading sample from metadata...")

meta_sample = pd.read_json(
    meta_path,
    lines=True,
    compression="gzip",
    nrows=5
)

print("\nMetadata Sample:")
print(meta_sample.head())

print("\nMetadata Columns:")
print(meta_sample.columns)

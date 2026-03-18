import os
import pandas as pd
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import ASCENDING, DESCENDING

uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi("1"))
db = client["amazon"]

client.admin.command("ping")
print("Connected")

print("Reading parquet...")
reviews = pd.read_parquet("cleaned_data/reviews_clean.parquet")
products = pd.read_parquet("cleaned_data/products_clean.parquet")

# MongoDB-safe converter
def mongo_safe(x):
    # pandas missing values
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    # numpy scalars -> python scalars
    if isinstance(x, (np.integer, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float64)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    # numpy array -> list
    if isinstance(x, np.ndarray):
        return x.tolist()
    # pandas Timestamp -> string
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    # dict/list: recursively clean
    if isinstance(x, dict):
        return {k: mongo_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [mongo_safe(v) for v in x]
    return x

# Clean types
reviews["timestamp"] = reviews["timestamp"].astype(str)

print("Converting products to MongoDB-safe types...")
products = products.map(mongo_safe)

print("Converting reviews to MongoDB-safe types...")
reviews = reviews.map(mongo_safe)

print("Dropping old collections (if any)...")
db.reviews.drop()
db.products.drop()

# Upload in batches (safer than one huge insert)
def insert_in_batches(collection, df, batch_size=5000):
    records = df.to_dict("records")
    for i in range(0, len(records), batch_size):
        collection.insert_many(records[i : i + batch_size])
        print(f"Inserted {min(i+batch_size, len(records))}/{len(records)} into {collection.name}")

print("Uploading products...")
insert_in_batches(db.products, products, batch_size=2000)

print("Uploading reviews...")
insert_in_batches(db.reviews, reviews, batch_size=5000)

print("Creating indexes...")
db.products.create_index([("parent_asin", ASCENDING)])
db.reviews.create_index([("parent_asin", ASCENDING), ("timestamp", DESCENDING)])
db.reviews.create_index([("user_id", ASCENDING)])

print("Upload complete!")
print("Products:", db.products.count_documents({}))
print("Reviews:", db.reviews.count_documents({}))

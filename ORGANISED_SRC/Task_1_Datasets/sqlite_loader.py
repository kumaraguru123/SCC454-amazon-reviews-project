import pandas as pd
import sqlite3
import time

# Paths
REVIEWS_PATH = "cleaned_data/reviews_clean.parquet"
PRODUCTS_PATH = "cleaned_data/products_clean.parquet"
DB_PATH = "amazon_sqlite.db"

print("Loading cleaned parquet files...")
reviews = pd.read_parquet(REVIEWS_PATH)
products = pd.read_parquet(PRODUCTS_PATH)

print("Connecting to SQLite database...")
conn = sqlite3.connect(DB_PATH)

print("Writing tables to SQLite...")
reviews.to_sql("reviews", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)

print("Creating indexes...")

conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_parent ON reviews(parent_asin)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_timestamp ON reviews(timestamp)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_products_parent ON products(parent_asin)")

# Create user index only if user_id exists
cols = [r[1] for r in conn.execute("PRAGMA table_info(reviews)").fetchall()]
if "user_id" in cols:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_user ON reviews(user_id)")
else:
    print("⚠ user_id not found in reviews table, skipping user index.")

conn.commit()

print("Database setup complete!")

conn.close()

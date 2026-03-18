import duckdb
import time

DB_PATH = "amazon_duckdb.duckdb"


def timed(con, sql, params=None):
    t0 = time.perf_counter()

    if params is None:
        rows = con.execute(sql).fetchall()
    else:
        rows = con.execute(sql, params).fetchall()

    t1 = time.perf_counter()
    return rows, (t1 - t0)


def main():

    print("Connecting to DuckDB...")
    con = duckdb.connect(DB_PATH)

    print("Creating tables from parquet files...")

    # Create tables from parquet
    con.execute("""
        CREATE OR REPLACE TABLE reviews AS
        SELECT * FROM 'cleaned_data/reviews_clean.parquet'
    """)

    con.execute("""
        CREATE OR REPLACE TABLE products AS
        SELECT * FROM 'cleaned_data/products_clean.parquet'
    """)

    print("Tables ready!")

    # Get sample IDs
    parent_asin = con.execute(
        "SELECT parent_asin FROM reviews LIMIT 1"
    ).fetchone()[0]

    user_id = con.execute(
        "SELECT user_id FROM reviews LIMIT 1"
    ).fetchone()[0]

    keyword = "headphone"
    N = 5

    print("Using parent_asin:", parent_asin)
    print("Using user_id:", user_id)
    print("Keyword:", keyword)

    print("-" * 40)

    # Q1 Product info
    rows, t = timed(con, """
        SELECT title, description
        FROM products
        WHERE parent_asin = ?
    """, [parent_asin])

    print("Q1 time:", t, "rows:", len(rows))

    # Q2 Recent reviews
    rows, t = timed(con, """
        SELECT rating, title, text, timestamp
        FROM reviews
        WHERE parent_asin = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, [parent_asin, N])

    print("Q2 time:", t, "rows:", len(rows))

    # Q3 Keyword search
    rows, t = timed(con, """
        SELECT parent_asin, title
        FROM products
        WHERE lower(title) LIKE ?
    """, [f"%{keyword}%"])

    print("Q3 time:", t, "rows:", len(rows))

    # Q4 User history
    rows, t = timed(con, """
        SELECT parent_asin, rating, timestamp
        FROM reviews
        WHERE user_id = ?
    """, [user_id])

    print("Q4 time:", t, "rows:", len(rows))

    # Q5 Product stats
    rows, t = timed(con, """
        SELECT AVG(rating), COUNT(*)
        FROM reviews
        WHERE parent_asin = ?
    """, [parent_asin])

    print("Q5 time:", t, "result:", rows[0])


if __name__ == "__main__":
    main()

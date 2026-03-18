import sqlite3
import time

DB_PATH = "amazon_sqlite.db"

def timed(cur, sql, params=()):
    t0 = time.perf_counter()
    rows = cur.execute(sql, params).fetchall()
    t1 = time.perf_counter()
    return rows, (t1 - t0)

def pick_sample_parent_asin(cur):
    # get a product that definitely has reviews
    row = cur.execute("SELECT parent_asin FROM reviews LIMIT 1").fetchone()
    return row[0]

def pick_sample_user(cur):
    row = cur.execute("SELECT user_id FROM reviews LIMIT 1").fetchone()
    return row[0]

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    parent_asin = pick_sample_parent_asin(cur)
    user_id = pick_sample_user(cur)
    keyword = "headphone"
    N = 5

    print("Using parent_asin:", parent_asin)
    print("Using user_id:", user_id)
    print("Keyword:", keyword)
    print("N:", N)
    print("-" * 50)

    # 1) Product Information Retrieval
    sql1 = """
    SELECT title, description
    FROM products
    WHERE parent_asin = ?
    """
    rows, dt = timed(cur, sql1, (parent_asin,))
    print("Q1 time:", dt, "rows:", len(rows))

    # 2) Recent Reviews
    sql2 = """
    SELECT rating, title, text, timestamp
    FROM reviews
    WHERE parent_asin = ?
    ORDER BY timestamp DESC
    LIMIT ?
    """
    rows, dt = timed(cur, sql2, (parent_asin, N))
    print("Q2 time:", dt, "rows:", len(rows))

    # 3) Keyword Search (simple LIKE baseline)
    sql3 = """
    SELECT parent_asin, title
    FROM products
    WHERE lower(title) LIKE '%' || lower(?) || '%'
       OR lower(description) LIKE '%' || lower(?) || '%'
    """
    rows, dt = timed(cur, sql3, (keyword, keyword))
    print("Q3 time:", dt, "rows:", len(rows))

    # 4) User Review History
    sql4 = """
    SELECT parent_asin, rating, timestamp
    FROM reviews
    WHERE user_id = ?
    ORDER BY timestamp DESC
    """
    rows, dt = timed(cur, sql4, (user_id,))
    print("Q4 time:", dt, "rows:", len(rows))

    # 5) Product Statistics
    sql5a = """
    SELECT AVG(rating), COUNT(*)
    FROM reviews
    WHERE parent_asin = ?
    """
    rows_a, dt_a = timed(cur, sql5a, (parent_asin,))

    sql5b = """
    SELECT rating, COUNT(*) as cnt
    FROM reviews
    WHERE parent_asin = ?
    GROUP BY rating
    ORDER BY rating
    """
    rows_b, dt_b = timed(cur, sql5b, (parent_asin,))

    print("Q5a (avg+count) time:", dt_a, "result:", rows_a[0])
    print("Q5b (distribution) time:", dt_b, "rows:", len(rows_b))
    print("Distribution:", rows_b)

    conn.close()

if __name__ == "__main__":
    main()

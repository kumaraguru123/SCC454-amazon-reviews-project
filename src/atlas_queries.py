import os
import time
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = os.getenv("MONGO_URI")

client = MongoClient(uri, server_api=ServerApi('1'))
db = client["amazon"]

reviews = db.reviews
products = db.products

print("Connected to MongoDB Atlas")
parent_asin = reviews.find_one({}, {"parent_asin": 1})["parent_asin"]
user_id = reviews.find_one({}, {"user_id": 1})["user_id"]

keyword = "headphone"
N = 5

print("Using parent_asin:", parent_asin)
print("Using user_id:", user_id)
print("Keyword:", keyword)
print("------------------------------------")

def timed(func):
    t0 = time.perf_counter()
    result = func()
    t1 = time.perf_counter()
    return result, (t1 - t0)

def q1():
    return list(products.find(
        {"parent_asin": parent_asin},
        {"_id": 0, "title": 1, "description": 1}
    ))


rows, t = timed(q1)
print("Q1 time:", t, "rows:", len(rows))

def q2():
    return list(reviews.find(
        {"parent_asin": parent_asin},
        {"_id": 0, "rating": 1, "title": 1, "text": 1, "timestamp": 1}
    ).sort("timestamp", -1).limit(N))


rows, t = timed(q2)
print("Q2 time:", t, "rows:", len(rows))

def q3():
    return list(products.find(
        {
            "$or": [
                {"title": {"$regex": keyword, "$options": "i"}},
                {"description": {"$regex": keyword, "$options": "i"}}
            ]
        },
        {"_id": 0, "parent_asin": 1, "title": 1}
    ))


rows, t = timed(q3)
print("Q3 time:", t, "rows:", len(rows))

def q4():
    return list(reviews.find(
        {"user_id": user_id},
        {"_id": 0, "parent_asin": 1, "rating": 1, "timestamp": 1}
    ))


rows, t = timed(q4)
print("Q4 time:", t, "rows:", len(rows))

def q5():
    pipeline = [
        {"$match": {"parent_asin": parent_asin}},
        {
            "$group": {
                "_id": "$rating",
                "count": {"$sum": 1},
                "avg_rating": {"$avg": "$rating"}
            }
        }
    ]
    return list(reviews.aggregate(pipeline))


rows, t = timed(q5)
print("Q5 time:", t, "rows:", len(rows))

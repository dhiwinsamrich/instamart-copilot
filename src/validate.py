"""Assert the Part A reference checkpoints against data/clean and data/instamart.db."""
import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "clean")

customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

assert customers["city"].value_counts().to_dict() == {
    "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert customers["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert customers["age"].isna().sum() == 0
assert customers["age"].between(18, 60).all()
assert len(orders) == 12309 and orders["order_id"].is_unique
assert len(completed) == 11110 and len(cancelled) == 1199
assert (orders["status"] == "pending").sum() == 0
assert orders["order_amount"].isna().sum() == 0
assert completed["delivery_minutes"].isna().sum() == 0
assert cancelled["delivery_minutes"].isna().sum() == 1199
assert round(completed["order_amount"].sum(), 2) == 2998668.80

with sqlite3.connect(os.path.join(ROOT, "data", "instamart.db")) as con:
    tables = set(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)["name"])
    assert {"customers", "products", "orders"} <= tables
    db_rev = pd.read_sql_query("SELECT ROUND(SUM(order_amount), 2) AS r FROM orders WHERE status='completed'", con)["r"][0]
    assert db_rev == 2998668.80
    ts = pd.read_sql_query("SELECT order_ts FROM orders LIMIT 1", con)["order_ts"][0]
    assert len(ts) == 19 and ts[10] == " "

print("ALL CHECKPOINTS PASSED")

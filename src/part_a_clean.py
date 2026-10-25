"""Part A: audit the raw exports, clean them, store CSVs and the SQLite database."""
import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CLEAN = os.path.join(ROOT, "data", "clean")
DB = os.path.join(ROOT, "data", "instamart.db")


def profile(name, df):
    print(f"\n== {name}: shape {df.shape}")
    blank = df.isna() | df.astype(str).apply(lambda s: s.str.strip() == "")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": blank.sum(),
                        "missing_pct": (blank.mean() * 100).round(2)}).to_string())


# ---------------- audit ----------------
orders_raw = pd.read_csv(os.path.join(RAW, "orders.csv"), keep_default_na=False)
customers_raw = pd.read_csv(os.path.join(RAW, "customers.csv"), keep_default_na=False)
products_raw = pd.read_csv(os.path.join(RAW, "products.csv"), keep_default_na=False)
tickets_raw = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)

for name, df in [("orders", orders_raw), ("customers", customers_raw), ("products", products_raw)]:
    profile(name, df)

amount_num = pd.to_numeric(orders_raw["order_amount"], errors="coerce")
deliv_num = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(customers_raw["age"], errors="coerce")
gender_raw = customers_raw["gender"].str.strip()
audit = pd.DataFrame([
    ("raw orders.csv rows", len(orders_raw), 13084),
    ("exact duplicate rows", int(orders_raw.duplicated().sum()), 119),
    ("rows with status pending", int((orders_raw["status"] == "pending").sum()), 656),
    ("distinct order_id", orders_raw["order_id"].nunique(), 12309),
    ("non-numeric order_amount", int((amount_num.isna() & (orders_raw["order_amount"] != "")).sum()), 120),
    ("negative delivery_minutes", int((deliv_num < 0).sum()), 106),
    ("blank delivery_minutes", int((orders_raw["delivery_minutes"].str.strip() == "").sum()), 2541),
    ("blank age", int((customers_raw["age"].str.strip() == "").sum()), 61),
    ("age outside 10-100", int(((age_num < 10) | (age_num > 100)).sum()), 6),
    ("distinct raw city spellings", customers_raw["city"].nunique(), 24),
    ("distinct non-blank gender spellings", gender_raw[gender_raw != ""].nunique(), 10),
    ("blank gender", int((gender_raw == "").sum()), 30),
], columns=["figure", "value", "reference"])
audit["match"] = audit["value"] == audit["reference"]
print("\n== audit figures\n" + audit.to_string(index=False))
assert audit["match"].all(), "audit differs from the reference; fix before cleaning"
print(f"tickets.csv: {len(tickets_raw)} rows, {int((tickets_raw['text'].str.strip() == '').sum())} empty texts")

# ---------------- clean customers ----------------
changes = []
customers = customers_raw.copy()
city_clean = customers["city"].str.strip().str.title()
changes.append(("customers", "strip + title-case city", int((city_clean != customers["city"]).sum())))
customers["city"] = city_clean

GENDER = {"male": "male", "m": "male", "female": "female", "f": "female", "": "unknown"}
gender_clean = customers["gender"].str.strip().str.lower().map(GENDER)
assert gender_clean.notna().all()
changes.append(("customers", "map gender spellings to male/female/unknown",
                int((gender_clean != customers["gender"]).sum())))
customers["gender"] = gender_clean

customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
invalid_age = (customers["age"] < 10) | (customers["age"] > 100)
changes.append(("customers", "age outside 10-100 set to missing", int(invalid_age.sum())))
customers.loc[invalid_age, "age"] = np.nan
missing_age = customers["age"].isna()
customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))
changes.append(("customers", "missing age filled with city median", int(missing_age.sum())))
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ---------------- clean orders ----------------
products = products_raw.copy()
orders = orders_raw.copy()
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

n0 = len(orders)
orders = orders.drop_duplicates()
changes.append(("orders", "1. drop exact duplicate rows", n0 - len(orders)))

n1 = len(orders)
orders = (orders.sort_values("order_ts", kind="stable")
          .drop_duplicates("order_id", keep="last")
          .sort_values("order_id").reset_index(drop=True))
changes.append(("orders", "2. keep latest order_ts per order_id", n1 - len(orders)))

orders["quantity"] = orders["quantity"].astype(int)
orders["discount_pct"] = orders["discount_pct"].astype(int)
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
missing_amt = orders["order_amount"].isna()
formula = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(formula)
changes.append(("orders", "3. unparseable order_amount filled from price formula", int(missing_amt.sum())))

orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
negative = orders["delivery_minutes"] < 0
changes.append(("orders", "4. negative delivery_minutes set to missing", int(negative.sum())))
orders.loc[negative, "delivery_minutes"] = np.nan
completed = orders["status"] == "completed"
cat_median = orders["delivery_minutes"].where(completed).groupby(orders["category"]).transform("median")
fill = completed & orders["delivery_minutes"].isna()
orders.loc[fill, "delivery_minutes"] = cat_median[fill]
changes.append(("orders", "5. completed missing delivery_minutes filled with category median", int(fill.sum())))
print("\ncategory medians of delivery_minutes (completed):")
print(orders[completed].groupby("category")["delivery_minutes"].median().to_string())

orders = orders.drop(columns=["category", "price_inr"])

print("\n== cleaning rules and rows/values changed\n" +
      pd.DataFrame(changes, columns=["table", "rule", "changed"]).to_string(index=False))

# ---------------- post-clean summary ----------------
print("\ncities:", customers["city"].value_counts().to_dict())
print("genders:", customers["gender"].value_counts().to_dict())
print("missing ages:", int(customers["age"].isna().sum()), "age range:", customers["age"].min(), "-", customers["age"].max())
print("orders:", len(orders), orders["status"].value_counts().to_dict())
print("missing order_amount:", int(orders["order_amount"].isna().sum()))
print("missing delivery_minutes by status:", orders.groupby("status")["delivery_minutes"].apply(lambda s: int(s.isna().sum())).to_dict())
print("completed revenue (INR): %.2f" % orders.loc[orders["status"] == "completed", "order_amount"].sum())

# ---------------- store ----------------
os.makedirs(CLEAN, exist_ok=True)
customers_out = customers.assign(signup_date=customers["signup_date"].dt.strftime("%Y-%m-%d"))
orders_out = orders.assign(order_ts=orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
customers_out.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
orders_out.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)

if os.path.exists(DB):
    os.remove(DB)
with sqlite3.connect(DB) as con:
    customers_out.to_sql("customers", con, index=False)
    products.to_sql("products", con, index=False)
    orders_out.to_sql("orders", con, index=False)
print(f"\nwrote {CLEAN}/*.csv and {DB}")

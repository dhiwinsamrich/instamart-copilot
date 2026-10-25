"""Part A: run sql/queries.sql against data/instamart.db, then the pandas cross-check and EDA."""
import os
import re
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option("display.width", 160)


def load_queries(path):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- (Q\d+):.*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


queries = load_queries(os.path.join(ROOT, "sql", "queries.sql"))
with sqlite3.connect(os.path.join(ROOT, "data", "instamart.db")) as con:
    res = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}

q1, q2, q3, q4, q5, q6, q7 = (res[f"Q{i}"] for i in range(1, 8))
print("== Q1 customers with > INR 8000 in completed orders above INR 500:", len(q1))
print(q1.head().to_string(index=False))
print(f"\n== Q2 rows: {len(q2)}, revenue total INR {q2['revenue'].sum():,.2f}; largest cell:")
print(q2.head(1).to_string(index=False))
print("\n== Q3 products above the average price:", len(q3))
print(q3.to_string(index=False))
print("\n== Q4 customers with no completed order:", len(q4))
print(f"\n== Q5 customers above average spend: {len(q5)}; average INR {q5['avg_total'].iloc[0]:,.2f}")
print("\n== Q6 top customer per city")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
ties = q6[q6["rnk"] != q6["row_num"]]
print(f"rows where RANK() != ROW_NUMBER(): {len(ties)}")
print(ties.to_string(index=False))
print(f"\n== Q7 daily rows: {len(q7)}; last cumulative INR {q7['cumulative_revenue'].iloc[-1]:,.2f}")
print(q7.tail(3).to_string(index=False))

assert len(q1) == 30 and len(q2) == 30 and len(q3) == 11 and len(q4) == 25 and len(q5) == 550
assert round(q2["revenue"].sum(), 2) == 2998668.80 and len(ties) == 8 and len(q7) == 183
assert q7["cumulative_revenue"].iloc[-1] == 2998668.80

# ---------------- pandas cross-check ----------------
clean = os.path.join(ROOT, "data", "clean")
orders = pd.read_csv(os.path.join(clean, "orders.csv"))
customers = pd.read_csv(os.path.join(clean, "customers.csv"))
products = pd.read_csv(os.path.join(clean, "products.csv"))
comp = (orders[orders["status"] == "completed"]
        .merge(customers[["customer_id", "city"]], on="customer_id")
        .merge(products[["product_id", "category"]], on="product_id"))

pivot = pd.pivot_table(comp, values="order_amount", index="city", columns="category",
                       aggfunc="sum", margins=True, margins_name="Total").round(2)
print("\n== pivot_table revenue (INR) by city x category\n" + pivot.to_string())
grand = round(pivot.loc["Total", "Total"], 2)
print(f"pivot grand total INR {grand:,.2f} == SQL Q2 total INR {q2['revenue'].sum():,.2f}: "
      f"{grand == round(q2['revenue'].sum(), 2)}")

by_city = comp.groupby("city").agg(revenue_sum=("order_amount", "sum"), revenue_mean=("order_amount", "mean"),
                                   orders=("order_amount", "count")).round(2)
print("\n== completed order_amount per city\n" + by_city.to_string())

amt = comp["order_amount"]
q1_, q3_ = amt.quantile(0.25), amt.quantile(0.75)
iqr = q3_ - q1_
lo, hi = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
stats = pd.Series({"mean": amt.mean(), "median": amt.median(), "skewness": amt.skew(), "Q1": q1_,
                   "Q3": q3_, "IQR": iqr, "lower_fence": lo, "upper_fence": hi,
                   "outliers_below": int((amt < lo).sum()), "outliers_above": int((amt > hi).sum())})
print("\n== completed order_amount statistics\n" + stats.round(3).to_string())

corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3)
print("\n== correlation matrix (completed orders)\n" + corr.to_string())
pairs = corr.where(~pd.DataFrame(
    [[i <= j for j in range(4)] for i in range(4)], index=corr.index, columns=corr.columns)).stack()
top = pairs.abs().idxmax()
print(f"largest |correlation|: {top[1]} vs {top[0]} = {pairs[top]:.3f}")

"""Part B: RFM table, K-means model selection, segment profiles, PCA, outputs/customer_segments.csv."""
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
pd.set_option("display.width", 160)

orders = pd.read_csv(os.path.join(ROOT, "data", "clean", "orders.csv"), parse_dates=["order_ts"])
comp = orders[orders["status"] == "completed"]
rfm = comp.groupby("customer_id").agg(last_ts=("order_ts", "max"), frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum"))
rfm["recency"] = (SNAPSHOT - rfm["last_ts"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].round({"monetary": 2})
print(f"== RFM: {len(rfm)} customers")
print(rfm.describe().round(2).to_string())

X = StandardScaler().fit_transform(rfm)
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3)))
ktable = pd.DataFrame(rows, columns=["K", "inertia", "silhouette"])
ktable["inertia_drop"] = (-ktable["inertia"].diff()).round(1)
print("\n== K-means model selection\n" + ktable.to_string(index=False))

km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
profile = rfm.groupby("segment_id").agg(size=("recency", "size"), recency=("recency", "mean"),
                                        frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)

# Cluster numbers carry no order, so names are assigned from each profile, not from the id.
premium_id = profile["monetary"].idxmax()
rest = profile.drop(premium_id)
lapsing_id = rest["recency"].idxmax()
rest = rest.drop(lapsing_id)
loyal_id = rest["frequency"].idxmax()
NAMES = {premium_id: "Premium Big Spenders", lapsing_id: "Dormant Low-Value",
         loyal_id: "Loyal Regulars", rest.drop(loyal_id).index[0]: "Occasional Shoppers"}
ACTIONS = {"Premium Big Spenders": "priority desk and early access to new electronics launches",
           "Dormant Low-Value": "low-cost win-back coupon, then stop spending if no response",
           "Loyal Regulars": "grocery subscription / loyalty points to protect habit",
           "Occasional Shoppers": "basket-building offers to raise frequency"}
profile["segment_name"] = profile.index.map(NAMES)
profile["is_premium"] = profile.index == premium_id
profile["action"] = profile["segment_name"].map(ACTIONS)
print(f"\n== segment profile (K={FINAL_K})\n" + profile.to_string())

pca = PCA(n_components=2).fit(X)
ratios = pca.explained_variance_ratio_
print(f"\n== PCA explained variance ratio: PC1 {ratios[0]:.3f}, PC2 {ratios[1]:.3f}, total {ratios.sum():.3f}")
pcs = pd.DataFrame(pca.transform(X), columns=["pc1", "pc2"], index=rfm.index)
print("segment centroids in PCA space\n" + pcs.groupby(rfm["segment_id"]).mean().round(2).to_string())

rfm["segment_name"] = rfm["segment_id"].map(NAMES)
rfm["is_premium"] = rfm["segment_id"] == premium_id
os.makedirs(OUT, exist_ok=True)
rfm.reset_index()[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv(os.path.join(OUT, "customer_segments.csv"), index=False)
print(f"\nwrote outputs/customer_segments.csv ({len(rfm)} rows, {int(rfm['is_premium'].sum())} premium)")

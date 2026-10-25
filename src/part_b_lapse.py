"""Part B: lapse label, leakage-free features, tuned models, balancing, persistence, retention list."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")
pd.set_option("display.width", 160)

orders = pd.read_csv(os.path.join(ROOT, "data", "clean", "orders.csv"), parse_dates=["order_ts"])
customers = pd.read_csv(os.path.join(ROOT, "data", "clean", "customers.csv"), parse_dates=["signup_date"])
products = pd.read_csv(os.path.join(ROOT, "data", "clean", "products.csv"))
orders = orders.merge(products[["product_id", "category"]], on="product_id")

# ---------------- label and features ----------------
before = orders[orders["order_ts"] < CUTOFF]
comp_before = before[before["status"] == "completed"]
sept = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END) & (orders["status"] == "completed")]

feat = comp_before.groupby("customer_id").agg(
    last_ts=("order_ts", "max"), frequency=("order_id", "nunique"), monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"), avg_delivery=("delivery_minutes", "mean"))
feat["recency"] = (CUTOFF - feat["last_ts"]).dt.days
feat["cancel_rate"] = before.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())
feat["fav_category"] = (comp_before.groupby(["customer_id", "category"]).size().reset_index(name="n")
                        .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
                        .drop_duplicates("customer_id").set_index("customer_id")["category"])
feat = feat.join(customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat["lapsed"] = (~feat.index.isin(sept["customer_id"])).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[NUM + CAT], feat["lapsed"]
assert X.isna().sum().sum() == 0, "a feature column contains missing values"
assert before["order_ts"].max() < CUTOFF
print(f"== cohort: {len(feat)} customers, lapsed {y.sum()} ({y.mean():.2%}); no missing feature values")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train {len(X_tr)} rows, test {len(X_te)} rows")


def lapsed_metrics(name, y_true, y_pred):
    return {"model": name, "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


baseline = np.zeros(len(y_te), dtype=int)
results = [lapsed_metrics("baseline: always not-lapsed", y_te, baseline)]


def make_pipe(model):
    prep = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
    return Pipeline([("prep", prep), ("model", model)])


lr_grid = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)), {"model__C": [0.01, 0.1, 1, 10, 100]},
                       cv=5, scoring="f1").fit(X_tr, y_tr)
rf_grid = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                       {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
                       cv=5, scoring="f1").fit(X_tr, y_tr)
print(f"\n== LogisticRegression best {lr_grid.best_params_}, CV F1 {lr_grid.best_score_:.3f}")
print(f"== RandomForest best {rf_grid.best_params_}, CV F1 {rf_grid.best_score_:.3f} "
      f"({len(rf_grid.cv_results_['params']) * 5} fits)")

scaler = lr_grid.best_estimator_.named_steps["prep"].named_transformers_["num"]
print(f"\nscaler mean of frequency (train rows only): {scaler.mean_[NUM.index('frequency')]:.4f}; "
      f"full-cohort mean: {X['frequency'].mean():.4f}")

# The test set is touched only from here, after both searches are finished.
for name, grid in [("logistic regression", lr_grid), ("random forest", rf_grid)]:
    pred = grid.predict(X_te)
    results.append(lapsed_metrics(name, y_te, pred))
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"\n{name} confusion matrix (rows actual 0/1, cols predicted 0/1):\n[[{tn} {fp}]\n [{fn} {tp}]]")

# ---------------- class-balance experiment (training rows only) ----------------
train = X_tr.assign(lapsed=y_tr)
minority, majority = train[train["lapsed"] == 1], train[train["lapsed"] == 0]
extra = resample(minority, replace=True, n_samples=len(majority) - len(minority), random_state=42)
balanced = pd.concat([train, extra])
print(f"\nbalanced training set: {balanced['lapsed'].value_counts().to_dict()}")
lr_bal = make_pipe(LogisticRegression(max_iter=1000, C=lr_grid.best_params_["model__C"]))
lr_bal.fit(balanced[NUM + CAT], balanced["lapsed"])
results.append(lapsed_metrics("logistic regression, over-sampled", y_te, lr_bal.predict(X_te)))

metrics = pd.DataFrame(results).set_index("model").round(3)
print("\n== test-set metrics for the lapsed class\n" + metrics.to_string())

# ---------------- interpretation ----------------
rf_best = rf_grid.best_estimator_
names = rf_best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
print("\n== random forest top 5 importances\n" + imp.head(5).round(4).to_string())
lr_best = lr_grid.best_estimator_
coef = pd.Series(lr_best.named_steps["model"].coef_[0], index=lr_best.named_steps["prep"].get_feature_names_out())
print("\n== logistic regression coefficients (scaled features)\n" + coef.sort_values().round(3).to_string())

# ---------------- persist ----------------
selected_name, selected = ("random forest", rf_best) if metrics.loc["random forest", "f1"] >= \
    metrics.loc["logistic regression", "f1"] else ("logistic regression", lr_best)
assert metrics.loc[selected_name, "f1"] >= 0.55
path = os.path.join(OUT, "lapse_model.joblib")
joblib.dump(selected, path)
reloaded = joblib.load(path)
same = np.array_equal(reloaded.predict(X_te), selected.predict(X_te)) and \
    np.allclose(reloaded.predict_proba(X_te), selected.predict_proba(X_te))
print(f"\nselected model: {selected_name}; reloaded model reproduces test predictions exactly: {same}")
assert same

# ---------------- retention list ----------------
seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv")).set_index("customer_id")
test_scored = pd.DataFrame({"lapse_probability": reloaded.predict_proba(X_te)[:, 1].round(4),
                            "actually_lapsed": y_te}, index=X_te.index)
premium_test = test_scored[test_scored.index.isin(seg.index[seg["is_premium"]])]
retention = premium_test.sort_values("lapse_probability", ascending=False).head(10)
retention.rename_axis("customer_id").reset_index().to_csv(os.path.join(OUT, "retention_list.csv"), index=False)
print(f"\n== retention list: {len(premium_test)} premium customers in the test set, top {len(retention)} written; "
      f"{int(retention['actually_lapsed'].sum())} of them truly lapsed")
print(retention.to_string())

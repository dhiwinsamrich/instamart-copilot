# Instamart Customer Intelligence & Support Copilot

## Overview

One repository that turns Instamart's messy synthetic exports into a trusted SQLite database (Part A), segments customers and predicts who will lapse (Part B), triages support tickets safely with an offline mock LLM (Part C), and answers policy questions and decides refund requests with a retrieval-grounded tool-using agent (Part D). Every Part reads what the previous Part wrote, and every reference checkpoint in the brief is asserted in code. All data is synthetic, all amounts are in INR, and no network call or API key is used anywhere (every model call goes through the supplied `mock_llm.py`).

| Script | Part | Writes |
|---|---|---|
| `generate_data.py` (supplied, unmodified) | setup | `data/raw/*.csv`, `data/policies/*.txt` |
| `src/part_a_clean.py` | A | audit tables, `data/clean/*.csv`, `data/instamart.db` |
| `src/validate.py` | A | prints `ALL CHECKPOINTS PASSED` |
| `src/part_a_analysis.py` | A | runs `sql/queries.sql`, pandas cross-check and EDA |
| `src/part_b_segments.py` | B | `outputs/customer_segments.csv` |
| `src/part_b_lapse.py` | B | `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| `src/part_c_triage.py` | C | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| `src/rag.py` | D | chunking, `retrieve`, `answer`, retrieval evaluation |
| `src/agent.py` | D | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/*.json` |

## Reproduce

Python 3.11 or newer. Run from the repository root, in order:

1. `git clone https://github.com/dhiwinsamrich/instamart-copilot.git`
2. `cd instamart-copilot`
3. `python -m venv .venv`
4. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
5. `pip install -r requirements.txt`
6. `python generate_data.py`
7. `python src/part_a_clean.py`
8. `python src/validate.py`
9. `python src/part_a_analysis.py`
10. `python src/part_b_segments.py`
11. `python src/part_b_lapse.py`
12. `python src/part_c_triage.py`
13. `python src/rag.py`
14. `python src/agent.py`

Step 6 ends with `generated: customers=1200 products=40 orders_raw=13084 tickets=164`; step 8 prints `ALL CHECKPOINTS PASSED`. Every other script asserts its own reference checkpoints and stops with an `AssertionError` if one is missed.

## Part A

### Raw data profile

All raw tables are read with `keep_default_na=False`, so `N/A` stays a string and is counted as unparseable. A value counts as missing when it is blank or whitespace only. `order_amount`, `delivery_minutes` and `age` load as text (`str`) because of the dirty values.

`orders.csv`, shape 13,084 × 9:

| Column | Dtype | Missing | Missing % |
|---|---|---|---|
| order_id | str | 0 | 0.00 |
| customer_id | str | 0 | 0.00 |
| product_id | str | 0 | 0.00 |
| quantity | int64 | 0 | 0.00 |
| order_ts | str | 0 | 0.00 |
| status | str | 0 | 0.00 |
| discount_pct | int64 | 0 | 0.00 |
| order_amount | str | 0 | 0.00 |
| delivery_minutes | str | 2,541 | 19.42 |

`customers.csv`, shape 1,200 × 5:

| Column | Dtype | Missing | Missing % |
|---|---|---|---|
| customer_id | str | 0 | 0.00 |
| city | str | 0 | 0.00 |
| signup_date | str | 0 | 0.00 |
| age | str | 61 | 5.08 |
| gender | str | 30 | 2.50 |

`products.csv`, shape 40 × 4:

| Column | Dtype | Missing | Missing % |
|---|---|---|---|
| product_id | str | 0 | 0.00 |
| product_name | str | 0 | 0.00 |
| category | str | 0 | 0.00 |
| price_inr | int64 | 0 | 0.00 |

`tickets.csv` (164 rows, 4 empty texts) is also read with `keep_default_na=False` everywhere it is used.

### Audit figures

| Figure | Value | Reference |
|---|---|---|
| raw `orders.csv` rows | 13,084 | 13,084 |
| exact duplicate rows | 119 | 119 |
| rows with status `pending` | 656 | 656 |
| distinct `order_id` | 12,309 | 12,309 |
| `order_amount` not convertible to a number | 120 | 120 |
| negative `delivery_minutes` | 106 | 106 |
| blank `delivery_minutes` | 2,541 | 2,541 |
| blank `age` | 61 | 61 |
| `age` outside 10–100 | 6 | 6 |
| distinct raw `city` spellings | 24 | 24 |
| distinct non-blank raw `gender` spellings | 10 | 10 |
| blank `gender` | 30 | 30 |

### Cleaning rules (in the order applied) and what each changed

| Table | Rule | Rows / values changed |
|---|---|---|
| customers | strip spaces and title-case `city` → exactly 6 cities | 476 values |
| customers | map `male/Male/M/m/MALE` → `male`, `female/Female/F/f/FEMALE` → `female`, blank → `unknown` | 950 values |
| customers | `age` outside 10–100 set to missing | 6 values |
| customers | every missing `age` (61 blank + 6 invalid) filled with the median age of the customer's city (`groupby("city").transform("median")`) | 67 values |
| customers | `signup_date` converted to datetime | all rows |
| orders | 1. `order_ts` converted to datetime | all rows |
| orders | 2. drop exact duplicate rows | 119 rows removed |
| orders | 3. sort by `order_ts`, keep the latest record per `order_id` (drops the earlier `pending` copies) | 656 rows removed |
| orders | 4. `pd.to_numeric(errors="coerce")` on `order_amount`, then fill from `round(price_inr × quantity × (1 − discount_pct/100), 2)` | 112 values filled (8 of the 120 bad amounts were on removed duplicate rows) |
| orders | 5a. negative `delivery_minutes` set to missing | 105 values (1 was on a removed row) |
| orders | 5b. completed orders: missing `delivery_minutes` filled with the median of the product category over completed orders (`groupby().transform("median")`; Dairy, Electronics, Grocery, Snacks 25.0, Personal Care 25.5) | 774 values |
| orders | 5c. cancelled orders: `delivery_minutes` left missing (never delivered, so imputing would invent data) | 1,199 values stay missing |

After cleaning: cities Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; genders female 610, male 560, unknown 30; 0 missing ages, all between 18 and 60. Orders: 12,309 rows (one per `order_id`), 11,110 completed, 1,199 cancelled, 0 pending, 0 missing `order_amount`, 0 missing `delivery_minutes` among completed and 1,199 among cancelled. **Completed revenue: INR 2,998,668.80.**

### Storage and validation

The three clean tables are written to `data/clean/{customers,products,orders}.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`) with `DataFrame.to_sql`; `order_ts` is stored as text `YYYY-MM-DD HH:MM:SS` and `signup_date` as `YYYY-MM-DD`. `src/validate.py` makes 14 assertions (city and gender counts, ages, row counts, status counts, missing values, revenue in CSV and in SQLite, table names, timestamp format) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE `order_amount > 500` then HAVING total > 8000 | 30 customers (top: C0693, INR 13,776.55) |
| Q2 | three-table join, revenue by city × category | 30 rows summing to INR 2,998,668.80; largest Bengaluru / Electronics INR 233,909.45 |
| Q3 | scalar subquery, price above the catalogue average | 11 products |
| Q4 | LEFT JOIN, customers with no completed order | 25 customers |
| Q5 | two CTEs, spend above the average customer total | 550 customers; average INR 2,552.06 |
| Q6 | `RANK`, `DENSE_RANK`, `ROW_NUMBER` by city | top per city below; 8 rows where `RANK() ≠ ROW_NUMBER()` |
| Q7 | running total `SUM(revenue) OVER (ORDER BY day)` | 183 daily rows; last cumulative INR 2,998,668.80 |

Q6 top customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

**Why `RANK()` differs from `ROW_NUMBER()` on ties:** customers with equal `total_spend` share the same `RANK()` (and the next rank skips), while `ROW_NUMBER()` still gives every row a distinct number, so every row at or after a tie in a city gets a rank lower than its row number — for example Pune C0057 and C0936 both spend INR 285.00 and share rank 189 but have row numbers 190 and 191.

### pandas cross-check

`pd.pivot_table(..., margins=True)` over completed orders reproduces Q2: grand total INR 2,998,668.80, equal to the SQL total. Category totals: Electronics INR 1,219,248.10, Grocery INR 614,293.30, Dairy INR 531,938.95, Snacks INR 464,754.50, Personal Care INR 168,433.95.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00, IQR 232.20, lower fence −283.50 (0 outliers below), upper fence INR 645.30 (897 outliers above).

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | −0.095 | 0.087 | −0.003 |
| discount_pct | −0.095 | 1.000 | −0.000 | −0.010 |
| quantity | 0.087 | −0.000 | 1.000 | −0.013 |
| delivery_minutes | −0.003 | −0.010 | −0.013 | 1.000 |

**Exploratory write-up.** *What is the data about?* 12,309 synthetic Instamart orders from April–September 2025 placed by 1,200 customers in six cities for 40 products in five categories; 11,110 completed orders earn INR 2,998,668.80. *Distributions and statistics:* completed order value is strongly right-skewed (mean INR 269.91 versus median INR 135.00, skewness 4.61, IQR INR 64.80–297.00); Electronics brings 40.7% of revenue; delivery times centre on 25 minutes. *Outliers or anomalies:* 897 orders exceed the INR 645.30 upper fence and none fall below the lower fence; 635 are Electronics and the rest are multi-unit baskets, so they are genuine high-value purchases and are kept. The true anomalies — duplicates, pending copies, unparseable amounts, negative durations — were fixed during cleaning. *Relationships:* every pairwise correlation is weak; the largest in absolute value is `order_amount` with `discount_pct` at −0.095, just ahead of `order_amount` with `quantity` at +0.087. A correlation this small explains under 1% of variance, and correlation does not establish causation.

## Part B

### RFM (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency median 19 days (min 0, max 180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

### K-means on standardised RFM (`n_init=10`, `random_state=42`)

| K | Inertia | Silhouette | Inertia drop from K−1 |
|---|---|---|---|
| 2 | 1820.8 | 0.465 | – |
| 3 | 1268.2 | 0.445 | 552.6 |
| 4 | 743.1 | **0.526** | 525.1 |
| 5 | 617.4 | 0.480 | 125.7 |
| 6 | 503.2 | 0.482 | 114.2 |
| 7 | 408.5 | 0.416 | 94.7 |
| 8 | 366.5 | 0.412 | 42.0 |

**Chosen K = 4.** The elbow is at K=4: inertia falls by about 525 from 3 to 4 but only about 126 from 4 to 5. K=4 also has the best silhouette (0.526), and four segments are few enough for the retention team to run a distinct campaign for each. Inertia of K=8 is always lower than K=2 because every extra centroid can only move points closer to their nearest centre (in the limit, one centroid per customer gives inertia 0). So inertia keeps falling regardless of whether the new clusters mean anything, and on its own it cannot pick K; you need the elbow, silhouette and interpretability.

### Segment profile (K = 4)

Cluster numbers are only labels with no ordering; names are assigned in code from each profile (highest mean monetary → Premium; then highest recency; then highest frequency), so no cluster number is hard-coded.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium | Recommended action |
|---|---|---|---|---|---|---|---|
| 1 | Premium Big Spenders | 138 | 24.12 | 6.77 | 7,129.70 | **yes** | priority desk and early access to new electronics launches |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | no | grocery subscription / loyalty points to protect the habit |
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | no | basket-building offers to raise frequency |
| 3 | Dormant Low-Value | 182 | 109.04 | 2.20 | 685.99 | no | low-cost win-back coupon, stop spending if no response |

PCA(2) on the scaled RFM matrix: explained variance ratio **0.634** and **0.218** (total 0.852). `outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium` (1,175 rows, 138 premium).

### Lapse problem

Cutoff 2025-09-01. Cohort = customers with at least one completed order before the cutoff: **1,156 customers, 443 lapsed (38.32%)** (no completed order 2025-09-01 to 2025-09-30). Features, all computed from orders dated **before** the cutoff plus the clean customer table: recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure days. **No order dated on or after 2025-09-01 enters any feature (asserted in code), and the segment labels from the RFM step are not used as features**, because both would leak the label. No feature column contains a missing value (asserted). Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: **924 train / 232 test**.

### Preprocessing without leakage

A `ColumnTransformer` inside a `Pipeline`: `StandardScaler` on the 8 numeric features and `OneHotEncoder(drop="first")` (k−1 columns) on city, gender and favourite category. The pipeline is fitted inside `GridSearchCV` on training rows only, so **no encoder, scaler or resampler ever saw a test row**. The fitted scaler's mean of `frequency` is **8.1147** versus the full-cohort mean of **8.0285**; they differ because the scaler learned its mean from the 924 training rows only and never saw the 232 test customers.

### Tuning and test evaluation (lapsed class = positive)

| Model | Best params | Best CV F1 |
|---|---|---|
| LogisticRegression(max_iter=1000), C ∈ {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 |
| RandomForest(n_estimators=100, random_state=42), 9 combos × 5 folds = 45 fits | max_depth = 8, min_samples_leaf = 10 | 0.681 |

| Model (test set, 232 rows) | Confusion [[TN FP] [FN TP]] | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline: always "not lapsed" | [[143 0] [89 0]] | 0.616 | 0.000 | 0.000 | 0.000 |
| Logistic regression | [[111 32] [31 58]] | 0.728 | 0.644 | 0.652 | 0.648 |
| Random forest | [[113 30] [31 58]] | 0.737 | 0.659 | 0.652 | **0.655** |

The logistic regression chose C=100 rather than the reference C=1 because of this feature set; its test F1 is of the reference order.

**Selected model: random forest.** Missing a lapsing customer costs the whole future basket (about INR 2,550 of completed spend per customer over six months), while wrongly targeting a loyal customer only costs one coupon. Both tuned models catch the same 58 of 89 lapsers (recall 0.652). The forest does it with fewer wasted offers (30 versus 32 false positives), so its precision (0.659) and F1 (0.655) are higher. It also captures the non-linear effect of frequency and recency without hand-made interactions, and if the team wants more recall it can lower the decision threshold. Both models were scored on the test set only after both grid searches had finished.

### Class-balance experiment (training rows only)

Every original training row was kept, and 216 lapsed training rows were added, drawn with replacement by `resample(..., replace=True, random_state=42)`, giving **570 / 570**. The best logistic regression (C=100) was refitted and scored on the unchanged test set:

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Unbalanced | 0.644 | 0.652 | 0.648 |
| Over-sampled | 0.589 | **0.854** | 0.697 |

Recall moved most (0.652 → 0.854): the balanced model catches 76 instead of 58 of the 89 lapsers. Precision did not rise; it slipped to 0.589. So the retention team reaches far more at-risk customers, at the cost of roughly 4 in 10 offers going to customers who would have stayed. That trade is worth it when the offer is cheap.

### Interpretation

Random forest top 5 importances: `frequency` 0.280, `monetary` 0.271, `recency` 0.115, `avg_delivery` 0.061, `cancel_rate` 0.052. Customers who order often, spend a lot and ordered recently are the ones who stay.

Logistic regression coefficient signs (scaled features):
- `frequency` **negative** (−1.70): each extra habitual order strongly lowers lapse risk, so protect the order habit.
- `monetary` **positive** (+0.29): at equal frequency, bigger total spend means a few large, occasional (electronics) baskets, a pattern that lapses more than small regular baskets.
- `avg_delivery` **positive** (+0.08): slower average deliveries raise lapse risk, so delivery speed is a retention lever.

### Persistence and retention list

The whole selected pipeline (preprocessing plus random forest) is saved with `joblib.dump` to `outputs/lapse_model.joblib`. Reloaded with `joblib.load`, it reproduces the original test predictions and probabilities exactly (asserted). Of the 232 test customers, 40 are in the Premium segment. The top 10 by predicted lapse probability (0.452–0.491) are in `outputs/retention_list.csv` (`customer_id, lapse_probability, actually_lapsed`); **1 of the 10 truly lapsed** (C0525). The Premium flag is a post-hoc whole-window label that includes the September window defining the lapse label, so this retention list is illustrative and not a leakage-free evaluation. The model's own features still use only pre-cutoff orders.

## Part C

Tickets are processed in file order. Each ticket goes through: input guard → redact → budget check → model call with retry → validate → route.

### Input guardrails and redaction

| Guard / redaction | Count |
|---|---|
| empty or whitespace-only → `skipped_empty`, never sent | 4 |
| `est_tokens(text) > 400` → `flagged_long`, never sent | 4 (each 734 tokens) |
| emails `[\w.+-]+@[\w-]+\.[\w.]+` → `[EMAIL]` | 55 |
| phones `\b[6-9]\d{9}\b` → `[PHONE]` | 91 |
| order ids `ORD-\d{6}` → `[ORDER_ID]` | 132 |
| total substitutions (all 164 tickets) | **278** |
| matches left after re-applying the three patterns | 0 |

The patterns are applied in that order. The user message is exactly the redacted text, and the code asserts that no pattern matches any prompt actually sent.

### Prompts

Zero-shot (26 estimated tokens): `CLASSIFY` instruction, the four allowed categories, JSON-only reply `{"category": "...", "reason": "..."}`, with no `EXAMPLES:`. Few-shot (107 estimated tokens): the same text plus an `EXAMPLES:` line and four labelled examples (fraud, technical, billing, delivery).

### Resilient calls

There are at most 3 retries per ticket. The wait before retry n is `0.05 × 2**n × random.uniform(0.9, 1.15)` (`random.seed(42)`), with one log line per retry giving the ticket id, retry number and wait. `mock_llm.reset()` is called before every full run. A full run over the 156 eligible tickets hits **19 `TransientLLMError`s, all recovered on retry 0** (waits 0.0453–0.0562 s, inside the 0.045–0.0575 s range), and no ticket ends with a permanent error.

### Validation and comparison

Replies are parsed as JSON first. If parsing fails, the reply falls back to `re.findall(r"[a-z]+", text.lower())` and is accepted only when exactly one allowed category word appears. Anything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to human review without guessing.

| Run (no budget) | classified_json | classified_keyword | flagged_invalid_output | Classified | Accuracy | Total tokens | Tokens / classified ticket |
|---|---|---|---|---|---|---|---|
| Zero-shot | 135 | 12 | 9 | 147 | **85.7%** (126/147) | 8,464 | 57.6 |
| Few-shot | 135 | 12 | 9 | 147 | **100.0%** (147/147) | 21,100 | 143.5 |

Zero-shot confusion (rows = true, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

All 21 zero-shot errors are fraud (11) or technical (10) tickets labelled billing, because those tickets also mention charges or payment. The few-shot prompt costs 143.5 versus 57.6 tokens per classified ticket (about 2.5×, 12,636 extra tokens for this batch, about 600 per corrected ticket) and fixes all 21 errors. It is worth paying here because each of the 11 fraud tickets sent to the billing queue would miss the 2-hour security deadline, which costs far more than about 86 extra tokens per ticket. If every category were low-risk, the zero-shot prompt at 85.7% would be the cheaper choice.

### Budget demo and final routing

`TOKEN_BUDGET = 6000`, zero-shot, with the budget checked before each call: **45 tickets end `budget_exhausted`** (6,027 tokens spent before the stop). Saved as `outputs/triage_budget_demo.csv`.

The final run is few-shot with `TOKEN_BUDGET = 100000` and writes `outputs/triage_results.csv`: 164 rows, columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`. Routing reads `outputs/customer_segments.csv`; customers missing from it count as not premium.

| Route | Priority | Tickets |
|---|---|---|
| `human_review` (4 empty + 4 long + 9 invalid output) | NORMAL | 17 |
| `security_team` (fraud) | HIGH | 34 |
| `premium_desk` (premium, non-fraud) | HIGH | 11 |
| `support_queue` | NORMAL | 102 |

**Rule-validated vs human:** the 135 `classified_json` answers passed the JSON-plus-allowed-category check, and the 12 `classified_keyword` answers passed the single-category word check; those 147 are routed automatically. The 17 tickets that are empty, too long or have invalid model output always go to a human, and no model output is ever guessed.

## Part D

### Chunking, retrieval and grounded answers

The five policy files are split into 40-word chunks with a 10-word overlap (step 30), stopping after the chunk that reaches the last word. That gives **15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2**, each tagged with its source file. `TfidfVectorizer()` is fitted on the chunk texts, and `retrieve(query, k=3)` ranks chunks by `cosine_similarity` and returns `source`, `score`, `text`. For `refund policy late delivery` the top three are all `refund_policy.txt` (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word occurs in a chunk, and IDF (inverse document frequency) down-weights words that appear in many chunks, so rare, specific words like "wallet" or "escalation" count more than "order". The limitation is that TF-IDF only matches exact surface words: "money back" does not match "refund" and "parcel" does not match "delivery", so a paraphrased question can retrieve the wrong chunks even when the answer exists.

`answer(question)` retrieves 3 chunks and sends two messages to `mock_llm.chat`. The system message contains `POLICY ASSISTANT`, says to answer only from context, and gives the fallback wording. The user message is `CONTEXT:`, the three chunks each prefixed with `[<source file>]` on its own line, then `QUESTION:` and the question. It returns the reply and the sources.

### Evaluation (`data/raw/policy_eval.csv`, 14 questions)

| Setting | Chunks | Expected doc in top 3 (12 answerable) | Answer contains expected fact (12) | Fallback on unanswerable (2) |
|---|---|---|---|---|
| 40 words, overlap 10 | 15 | **12/12** | **12/12** | **2/2** |
| 40 words, overlap 0 | 13 | 12/12 | 10/12 | 2/2 |

With zero overlap, two questions break even though the right document is retrieved. In both, the answer sentence is split across a chunk boundary:
- *"What delivery time makes a completed order eligible for a full refund?"*: the refund chunks split as "…delivered in more than 45 minutes, the customer | receives a full refund…". The best-matching sentence fragment in context is "receives a full refund of the order amount.", which lacks "more than 45 minutes".
- *"How soon is money returned when a payment fails but is deducted?"*: the payment chunks split as "…the amount is returned automatically | within 48 hours.", so the returned fragment lacks "48 hours".

The 10-word overlap repeats the end of each chunk at the start of the next, so every one of these sentences survives intact in at least one chunk.

### Tools

`TOOLS_SPEC` lists five tools, each with a name, a one-sentence description and a JSON-schema parameter object. `TOOL_MAP` maps each name to its Python function, and the loop calls `TOOL_MAP[name](**arguments)`.

| Tool | Behaviour |
|---|---|
| `get_order(order_id)` | reads `data/instamart.db`; returns `order_id, status, order_amount, delivery_minutes` or `{"error": "order not found"}` |
| `search_policy(query)` | `{"results": retrieve(query, k=3)}` |
| `calculate_refund(order_amount, delivery_minutes)` | 100% of the amount if delivered in more than 45 minutes, else 0 |
| `issue_refund(order_id, refund_inr)` | appends a JSON line (timestamp, tool, order, amount) to `outputs/refund_ledger.jsonl` |
| `escalate_to_human(order_id, refund_inr)` | same ledger line, for refunds above INR 1000 |

### ReAct loop: seven scenarios (`run_agent`, defaults `max_iterations=6, token_limit=6000, time_limit_seconds=10`)

| Scenario | Expected | Status | Refund (INR) | Model calls | Tokens used | Tools executed |
|---|---|---|---|---|---|---|
| S1 | REFUND | REFUND | 128.25 | 5 | 1,042 | get_order, search_policy, calculate_refund, issue_refund |
| S2 | REFUND | REFUND | 135.00 | 5 | 1,042 | get_order, search_policy, calculate_refund, issue_refund |
| S3 | ESCALATE | ESCALATE | 1,999.00 | 5 | 1,029 | get_order, search_policy, calculate_refund, escalate_to_human |
| S4 | DENIED | DENIED | 0.00 | 4 | 721 | get_order, search_policy, calculate_refund |
| S5 | DENIED | DENIED | 0.00 | 3 | 424 | get_order, search_policy |
| S6 | NOT_FOUND | NOT_FOUND | 0.00 | 2 | 130 | get_order |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,174 | search_policy × 6 |

Every run is one line in `outputs/agent_runs.jsonl`, with `scenario_id`, `status`, `final_answer`, `llm_calls`, `tokens_used` and every step's tool, arguments and observation. The token-limit, timeout and crash-and-resume demos are logged there too, tagged by `run_type`. The ledger holds the issued refunds (S1, S2, and S1 again from the resume demo) and the S3 escalation.

### Stop safely

| Demonstration | Status | Detail |
|---|---|---|
| S7 (`[LOOP]`) with defaults | `FALLBACK_MAX_ITERATIONS` | 6 model calls, 3,174 tokens, not stopped early by the 6,000 token limit |
| S1 with `token_limit=200` | **`FALLBACK_TOKEN_LIMIT`** | stopped after model call 3 (441 tokens), without executing the requested `calculate_refund` |
| S1 with `time_limit_seconds=0` | **`FALLBACK_TIMEOUT`** | stopped at the top of the first iteration, 0 model calls |

Each check sits where the brief puts it. Timeout (`time.monotonic()` elapsed `>=` limit) is checked at the top of every iteration. The token limit is checked right after each model call, before any tool runs. Max iterations is checked once `max_iterations` model calls have produced no final answer. Every fallback returns the steps completed so far and a human-handoff summary in `final_answer`.

### Crash and resume

After every tool observation the loop writes `state/<run_id>.json` containing the run id, scenario id, limits, full message list, `tokens_used`, `llm_calls`, step number, tools executed and steps. The S1 demo raises a simulated crash right after the 2nd tool's checkpoint, then `resume_agent("S1-crash-resume-demo")` reloads it and continues.

| Phase | Tools executed, in order |
|---|---|
| before the crash | 1. `get_order` 2. `search_policy` |
| after `resume_agent` | 3. `calculate_refund` 4. `issue_refund` |

That is four executions with none repeated. Final status **REFUND, INR 128.25**, 5 model calls in total.

### ReAct versus plan-and-execute

My ReAct agent makes one model call per step and decides each step from the last observation. I measured 5 calls for a refund or escalation (S1–S3, about 1,040 tokens), 4 for a late-check denial (S4), 3 for a cancelled order (S5), 2 for a missing order (S6), and 6 before the max-iteration stop on the looping S7. A plan-and-execute agent would ask once for the whole plan (`get_order → search_policy → calculate_refund → issue_refund/escalate`), run the tools without the model, and make one final call to summarise: about 2 calls and fewer tokens, because the growing message history is not re-sent at every step. That saving only holds when the plan survives. For S6 the plan breaks at step 1 (`NOT_FOUND`), and for S5 the cancelled order makes the refund steps meaningless, so the executor must detect the unexpected observation and replan. That adds calls and moves branching logic into hand-written checks, and ReAct already handles these cases naturally: it stopped after 2 and 3 calls. The branch between issuing and escalating also depends on a computed amount, which a fixed plan cannot know in advance. For this workflow I would keep ReAct. Its paths are short (at most 5 calls), each decision is grounded in the latest observation, which money-moving actions need, and the three safety limits already bound its worst case. Plan-and-execute pays off for long, predictable pipelines where replanning is rare.

## Limitations and Next Steps

- **The retention list is illustrative.** The Premium flag is a post-hoc whole-window label (its RFM includes the September window that defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation. Next step: recompute segments on pre-cutoff orders only and score the list on a later window.
- **The mock LLM is not a real model.** Classification uses keyword rules and the few-shot gain comes from a reordered keyword list, so the accuracies say nothing about a real LLM. Re-run the same pipeline against a hosted model (key kept in `.env`) before trusting the token/accuracy trade-off.
- **TF-IDF retrieval is lexical.** Paraphrased questions can miss; sentence-aware chunking or embeddings would make retrieval robust to chunk boundaries and synonyms.
- **The lapse model is moderate.** F1 is about 0.65 with a fixed 0.5 threshold, and the label is a single 30-day window. Next steps: threshold tuning against campaign cost, more behavioural features (inter-order gaps, trend), and time-based validation over several cutoffs.
- **Cleaning rules are fixed.** Imputed ages and delivery minutes (67 and 774 values) are medians and shrink variance. Flag imputed values so downstream models can tell them apart.
- **Refund rules are simplified.** The agent does not check the 7-day request window or call-back rules and treats every request as in-window, as the brief specifies.

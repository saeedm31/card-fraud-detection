# Data card

## 1. Simulated card payments (main data)

| | |
|---|---|
| **Source** | `fraud_detection/simulate.py`, fixed seed |
| **Size** | 3,000 customers, 6 months, 343,039 payments, 330 fraud cases |
| **Fraud rate** | under 1 in 200 payments |
| **Personal data** | None. Customers, shops and devices are invented |

**Normal life is made to look suspicious too:** trips abroad, big one-off purchases, new phones, shared family
tablets and in-app bursts, so a model cannot just flag anything unusual.

**Four kinds of fraud cases:**

| Kind | What it looks like |
|---|---|
| Card testing | tiny online payments, then big ones |
| Stolen card | in-store payments, often in another city |
| Account takeover | big online purchases from a new device, often at night |
| Subtle fraud | ordinary-looking online purchases, half from the customer's own device |

**Split by time:** train January to April, choose the alert threshold on May, report June (56,416 payments, 180 of
them fraud in 51 cases).

## 2. Real card payments (benchmark)

| | |
|---|---|
| **Source** | ULB Machine Learning Group, [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Database Contents License) |
| **Size** | 284,807 payments by European cardholders over two days, 492 frauds |
| **Features** | 28 anonymised PCA components (V1-V28), log amount and hour of day |
| **In the repo** | No (150 MB). Download it and run `python benchmark_ulb.py creditcard.csv` |
| **Split** | by time; the test window has 56,962 payments and 75 frauds |

The real data has no customer IDs, so the behavioural features cannot be built there; the benchmark checks that
the model ranking holds on real payments, not the full pipeline. Each real fraud payment counts as its own case.

## Limits
Simulated fraud is cleaner than real fraud. Labels here are known at once; real labels arrive with chargebacks.

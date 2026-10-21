"""Run the same evaluation on real card data: the ULB credit card fraud dataset.

284,807 real card payments by European cardholders over two days in September
2013, 492 of them fraud. Features V1-V28 are anonymised (PCA), so there are no
readable reasons here; this is a check that the modelling holds up on real data.

Usage:
    python benchmark_ulb.py path/to/creditcard.csv

Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
(or OpenML dataset 1597). It is not included in this repo because of its size.
Writes reports/benchmark_ulb.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection import evaluate as ev

FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour"]


def load(path):
    df = pd.read_csv(path).sort_values("Time", kind="mergesort").reset_index(drop=True)
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour"] = (df["Time"] / 3600) % 24
    # The shape the evaluation code expects: each fraud payment is its own case.
    tx = pd.DataFrame({"time": df["Time"].astype(int), "amount": df["Amount"],
                       "is_fraud": df["Class"].astype(bool)})
    tx["case_id"] = np.where(tx["is_fraud"], np.arange(len(tx)), -1)
    return df, tx


def split(n, train=0.6, valid=0.2):
    """By time: first 60% to train, next 20% to choose the threshold, last 20% to report."""
    a, b = int(n * train), int(n * (train + valid))
    idx = np.arange(n)
    return idx < a, (idx >= a) & (idx < b), idx >= b


def models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False, random_state=42),
    }


def main(path, out="reports/benchmark_ulb.json"):
    df, tx = load(path)
    y = df["Class"].to_numpy()
    tr, va, te = split(len(df))
    report = {"data": {"rows": int(len(df)), "frauds": int(y.sum()),
                       "test_rows": int(te.sum()), "test_frauds": int(y[te].sum()),
                       "split": "by time: 60% train, 20% validation (threshold), 20% test"},
              "review_cost_eur": ev.REVIEW_COST, "models": {}}
    for name, model in models().items():
        model.fit(df.loc[tr, FEATURES], y[tr])
        s_va = model.predict_proba(df.loc[va, FEATURES])[:, 1]
        s_te = model.predict_proba(df.loc[te, FEATURES])[:, 1]
        thr = ev.best_threshold(tx[va], s_va)
        res = {**ev.ranking_scores(y[te], s_te), "at_threshold": ev.outcome(tx[te], s_te, thr),
               "pr_curve": ev.pr_curve(y[te], s_te)}
        report["models"][name] = res
        o = res["at_threshold"]
        print(f"{name:18s} PR-AUC {res['pr_auc']:.3f}  ROC-AUC {res['roc_auc']:.3f}  "
              f"recall {o['recall']:.2f}  precision {o['precision']:.2f}  alerts/day {o['alerts_per_day']:.0f}  "
              f"fraud stopped {o['stopped_pct']:.0f}%  net €{o['net_saving_eur']:,.0f}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, separators=(",", ":"))
    print("Saved", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

"""Simulate the data, compare the three models, and save the results.

Usage:
    python train.py

Trains on January-April, picks each model's alert threshold on May (the one
that saves the most money), and reports on June, which no model has seen.
Writes reports/metrics.json, which the app and the README charts use.
"""

import json
import os

import numpy as np

from fraud_detection import evaluate as ev
from fraud_detection import models
from fraud_detection.features import build_features
from fraud_detection.simulate import DAYS, Settings, simulate


def threshold_curve(tx, scores, points=40):
    """Money and workload for a range of alert volumes (for the README chart)."""
    days = (tx["time"].max() - tx["time"].min()) / 86400
    out = []
    for per_day in np.geomspace(1, 200, points):
        share = min(per_day * days / len(tx), 1.0)
        o = ev.outcome(tx, scores, ev.budget_threshold(scores, share))
        out.append({k: round(float(o[k]), 3) for k in
                    ("alerts_per_day", "precision", "recall", "stopped_pct", "net_saving_eur", "review_cost_eur")})
    return out


def main(out="reports/metrics.json"):
    settings = Settings()
    tx, customers, merchants = simulate(settings)
    X = build_features(tx, customers)
    y = tx["is_fraud"].to_numpy()
    tr, va, te = models.time_split(tx)
    print(f"{len(tx):,} payments, {customers.shape[0]:,} customers, {y.mean():.3%} fraud")

    report = {
        "data": {"payments": int(len(tx)), "customers": int(len(customers)), "merchants": int(len(merchants)),
                 "days": DAYS, "fraud_payments": int(y.sum()), "fraud_rate_pct": round(100 * y.mean(), 3),
                 "fraud_cases": int(tx.loc[y, "case_id"].nunique()),
                 "fraud_eur": round(float(tx.loc[y, "amount"].sum()), 2),
                 "fraud_by_type": tx.loc[y, "fraud_type"].value_counts().to_dict(),
                 "seed": settings.seed},
        "split": {"train": "Jan-Apr 2026", "threshold": "May 2026", "test": "Jun 2026",
                  "test_payments": int(te.sum()), "test_frauds": int(y[te].sum()),
                  "test_cases": int(tx.loc[te & y, "case_id"].nunique())},
        "costs": {"review_eur_per_alert": ev.REVIEW_COST, "alert_budget_share": ev.ALERT_BUDGET},
        "models": {},
    }
    for model in models.all_models():
        model.fit(X[tr], y[tr])
        s_va, s_te = model.predict_proba(X[va]), model.predict_proba(X[te])
        thr = ev.best_threshold(tx[va], s_va)
        at = ev.outcome(tx[te], s_te, thr)
        flagged = s_te >= thr
        test = tx[te].assign(flag=flagged)
        by_type = test[test["is_fraud"]].groupby("fraud_type")["flag"].mean()
        res = {**ev.ranking_scores(y[te], s_te), "params": model.params,
               "at_threshold": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in at.items()},
               "recall_at_budget": float((s_te >= ev.budget_threshold(s_te))[y[te]].mean()),
               "recall_by_type": {k: round(float(v), 3) for k, v in by_type.items()},
               "pr_curve": ev.pr_curve(y[te], s_te)}
        if model.name == "gradient_boosting":
            res["threshold_curve"] = threshold_curve(tx[te], s_te)
        report["models"][model.name] = res
        print(f"{model.name:18s} PR-AUC {res['pr_auc']:.3f}  recall {at['recall']:.2f}  "
              f"precision {at['precision']:.2f}  alerts/day {at['alerts_per_day']:.1f}  "
              f"stopped {at['stopped_pct']:.0f}% of €{at['fraud_eur']:,.0f}  net €{at['net_saving_eur']:,.0f}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, separators=(",", ":"))
    print("Saved", out)


if __name__ == "__main__":
    main()

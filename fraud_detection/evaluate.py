"""Measure a fraud model the way a fraud team would.

Accuracy is useless here (99.6% of payments are fine). What matters:
- PR-AUC: how well the model ranks fraud above normal payments.
- Recall at a fixed review budget: analysts can only check so many alerts a day.
- Money: an alert costs analyst time; a missed fraud costs its amount. When an
  alert fires, the payment is declined and the card is blocked, so the rest of
  that fraud case is stopped too.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

REVIEW_COST = 5.0          # euros of analyst time per alert
ALERT_BUDGET = 0.005       # analysts can review 0.5% of payments


def money(tx, flagged):
    """Euros of fraud stopped if each flagged payment is declined and its card blocked.

    tx needs columns amount, is_fraud, case_id (fraud case number, -1 for normal) and time.
    """
    fraud = tx[tx["is_fraud"].to_numpy()].assign(flag=np.asarray(flagged)[tx["is_fraud"].to_numpy()])
    total = fraud["amount"].sum()
    stopped = 0.0
    caught_cases = 0
    for _, case in fraud.sort_values("time").groupby("case_id"):
        hit = np.flatnonzero(case["flag"].to_numpy())
        if len(hit):
            caught_cases += 1
            stopped += case["amount"].to_numpy()[hit[0]:].sum()
    return {"fraud_eur": total, "stopped_eur": stopped, "cases": fraud["case_id"].nunique(),
            "cases_caught": caught_cases}


def outcome(tx, scores, threshold):
    """Everything a fraud manager wants to know at one threshold."""
    flagged = scores >= threshold
    y = tx["is_fraud"].to_numpy()
    days = max(1, (tx["time"].max() - tx["time"].min()) / 86400)
    m = money(tx, flagged)
    alerts = int(flagged.sum())
    review = alerts * REVIEW_COST
    return {
        "threshold": float(threshold),
        "alerts": alerts,
        "alerts_per_day": alerts / days,
        "precision": float(y[flagged].mean()) if alerts else 0.0,
        "recall": float(flagged[y].mean()),
        "cases_caught_pct": 100 * m["cases_caught"] / max(m["cases"], 1),
        "fraud_eur": m["fraud_eur"],
        "stopped_eur": m["stopped_eur"],
        "stopped_pct": 100 * m["stopped_eur"] / max(m["fraud_eur"], 1e-9),
        "review_cost_eur": review,
        "net_saving_eur": m["stopped_eur"] - review,
    }


def budget_threshold(scores, share=ALERT_BUDGET):
    """Score above which the top `share` of payments fall."""
    return float(np.quantile(scores, 1 - share))


def best_threshold(tx, scores, candidates=200):
    """Threshold with the largest net saving (fraud stopped minus review cost)."""
    qs = np.unique(np.quantile(scores, np.linspace(0.95, 0.99995, candidates)))
    results = [(outcome(tx, scores, q)["net_saving_eur"], q) for q in qs]
    return float(max(results)[1])


def ranking_scores(y, scores):
    return {"pr_auc": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "fraud_rate": float(np.mean(y))}


def pr_curve(y, scores, points=101):
    """Precision at evenly spaced recall levels (the best precision reachable at that recall or more)."""
    precision, recall, _ = precision_recall_curve(y, scores)
    grid = np.linspace(0, 1, points)
    best = [float(precision[recall >= r].max()) for r in grid]
    return {"recall": grid.round(4).tolist(), "precision": np.round(best, 4).tolist()}

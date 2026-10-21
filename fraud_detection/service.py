"""What the app needs: build everything once, then score payments on demand."""

import numpy as np
import pandas as pd

from . import evaluate as ev
from .explain import reasons
from .features import build_features
from .models import BoostingModel, time_split
from .simulate import CITIES, Settings, simulate

PRESETS = {
    "Weekly groceries": dict(amount=48.0, category="grocery", online=False, city="home", hour=18.5,
                             device="own", shop="usual", burst=0),
    "Card testing": dict(amount=420.0, category="gift_cards", online=True, city="home", hour=3.2,
                         device="new", shop="new", burst=5),
    "Stolen card abroad": dict(amount=780.0, category="electronics", online=False, city="Bucharest", hour=15.0,
                               device="own", shop="new", burst=1),
    "Account takeover at night": dict(amount=950.0, category="travel", online=True, city="home", hour=3.5,
                                      device="shared", shop="new", burst=0),
}


def build(settings=None):
    """Simulate, train on Jan-Apr, choose the threshold on May, score June."""
    tx, customers, merchants = simulate(settings or Settings())
    X = build_features(tx, customers)
    y = tx["is_fraud"].to_numpy()
    tr, va, te = time_split(tx)
    model = BoostingModel().fit(X[tr], y[tr])
    threshold = ev.best_threshold(tx[va], model.predict_proba(X[va]))
    test = tx[te].copy()
    test["risk"] = model.predict_proba(X[te])
    return {"tx": tx, "X": X, "customers": customers, "merchants": merchants, "model": model,
            "threshold": threshold, "test": test, "X_test": X[te]}


def alerts(state, threshold=None):
    """June payments at or above the threshold, riskiest first, with reasons."""
    thr = state["threshold"] if threshold is None else threshold
    test = state["test"]
    hit = test[test["risk"] >= thr].sort_values("risk", ascending=False)
    why = reasons(state["model"], state["X_test"].loc[hit.index], hit["city"])
    return hit.assign(reasons=why)


def history(state, customer_id, before=None, n=10):
    tx = state["tx"]
    h = tx[tx["customer_id"] == customer_id]
    if before is not None:
        h = h[h["time"] < before]
    return h.tail(n)


def score_payment(state, customer_id, amount, category, online, city, hour, device, shop, burst):
    """Score a made-up payment for a customer, on 1 July right after their real history.

    device: "own" (a device they used before), "new" (never seen) or "shared" (seen on other cards).
    shop: "usual" (a shop they paid before) or "new". burst: small payments in the hour before.
    """
    tx, customers = state["tx"], state["customers"]
    past = tx[tx["customer_id"] == customer_id]
    home = customers.loc[customers["customer_id"] == customer_id, "home"].iloc[0]
    city = home if city == "home" else city
    t = int(tx["time"].max()) // 86400 * 86400 + 86400 + int(hour * 3600)   # 1 July, at `hour`
    own_devices = past["device_id"].dropna()
    dev_id = (own_devices.iloc[-1] if device == "own" and len(own_devices) else
              "NEW-DEVICE" if device in ("new", "shared") else "NEW-DEVICE")
    usual = past[(past["merchant_category"] == category) & (past["online"] == online)]["merchant_id"]
    merchant = int(usual.mode().iloc[0]) if shop == "usual" and len(usual) else -1

    rows = [dict(time=t - 300 * (burst - i), amount=1.99, merchant_category="digital_goods",
                 merchant_id=-2, city=None, online=True, device_id=dev_id) for i in range(burst)]
    rows.append(dict(time=t, amount=float(amount), merchant_category=category, merchant_id=merchant,
                     city=None if online else city, online=online, device_id=dev_id if online else None))
    new = pd.DataFrame(rows).assign(customer_id=customer_id, is_fraud=False)
    seq = pd.concat([past[new.columns.tolist()], new], ignore_index=True).sort_values("time", kind="mergesort")
    seq = seq.reset_index(drop=True)
    X = build_features(seq, customers).tail(1).copy()
    if online and device == "shared":
        X["device_customers"] = 3
    model = state["model"]
    risk = float(model.predict_proba(X)[0])
    why = reasons(model, X, [city if not online else None])[0]
    return risk, why, X.iloc[0]


def city_names():
    return list(CITIES)

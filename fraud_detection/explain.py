"""Plain-language reasons for an alert.

For each group of signals (amount, location, device, ...) the transaction is
scored again with that group set back to what is normal. The bigger the drop
in risk, the more that group explains the alert. This works with any model and
needs no extra library.
"""

import numpy as np
import pandas as pd

MIN_DROP = 0.05            # ignore groups that move the risk by less than 5 points

# Typical values of a normal payment (medians in the training data).
NORMAL = {"amount": 26.0, "n_24h": 1, "amt_24h": 9.0, "mins_since_prev": 1350.0, "n_prior": 60, "hour": 14.5}


def _reset(X, group):
    Z = X.copy()
    if group == "amount":
        ratio = Z["amount_ratio"]
        Z["amount"] = np.where(ratio.notna() & (ratio > 0), Z["amount"] / ratio, NORMAL["amount"])
        Z["amount_ratio"] = 1.0
    elif group == "time":
        Z["hour_dev"] = 0.0
        Z["hour"] = NORMAL["hour"]
    elif group == "location":
        Z["dist_home_km"] = np.where(Z["dist_home_km"].notna(), 0.0, np.nan)
        Z["speed_kmh"] = np.where(Z["speed_kmh"].notna(), 0.0, np.nan)
    elif group == "new_shop":
        Z["new_merchant"] = 0
    elif group == "new_device":
        Z["new_device"] = 0
    elif group == "shared_device":
        Z["device_customers"] = 0
    elif group == "burst":
        Z["n_1h"] = 0
        Z["n_small_1h"] = 0
        Z["n_24h"] = NORMAL["n_24h"]
        Z["amt_24h"] = NORMAL["amt_24h"]
        Z["mins_since_prev"] = NORMAL["mins_since_prev"]
    elif group == "history":
        Z["n_prior"] = NORMAL["n_prior"]
    elif group == "shop_type":
        Z["category"] = pd.Categorical(np.where(Z["online"] == 1, "marketplace", "grocery"),
                                       categories=Z["category"].cat.categories)
    return Z


GROUPS = ["amount", "time", "location", "new_shop", "new_device", "shared_device", "burst", "history",
          "shop_type"]
RISKY_TYPES = {"electronics", "gift_cards", "jewelry", "travel", "digital_goods"}


def _unusual(group, row):
    """Only explain with a signal that is actually out of the ordinary."""
    if group == "amount":
        r = row["amount_ratio"]
        return (pd.notna(r) and r >= 2) or (pd.isna(r) and row["amount"] >= 100)
    if group == "time":
        return pd.notna(row["hour_dev"]) and row["hour_dev"] >= 4
    if group == "location":
        return (pd.notna(row["dist_home_km"]) and row["dist_home_km"] >= 50) or \
            (pd.notna(row["speed_kmh"]) and row["speed_kmh"] > 900)
    if group == "new_shop":
        return row["new_merchant"] == 1
    if group == "new_device":
        return row["new_device"] == 1
    if group == "shared_device":
        return row["device_customers"] > 0
    if group == "burst":
        return row["n_1h"] >= 2 or row["n_24h"] >= 4
    if group == "history":
        return row["n_prior"] < 10
    if group == "shop_type":
        return str(row["category"]) in RISKY_TYPES
    return True


def _text(group, row, city):
    if group == "amount":
        r = row["amount_ratio"]
        if pd.notna(r):
            return f"€{row['amount']:,.0f}, {r:.1f}× this customer's usual"
        return f"€{row['amount']:,.0f}, and the card has little history"
    if group == "time":
        h = int(row["hour"])
        m = int(round((row["hour"] - h) * 60)) % 60
        dev = row["hour_dev"]
        extra = f", {dev:.0f} h from this customer's usual time" if pd.notna(dev) else ""
        return f"At {h:02d}:{m:02d}{extra}"
    if group == "location":
        if pd.notna(row["speed_kmh"]) and row["speed_kmh"] > 900:
            return f"In {city}, {row['dist_home_km']:,.0f} km from home: impossible travel since the last shop payment"
        return f"In {city}, {row['dist_home_km']:,.0f} km from home"
    if group == "new_shop":
        return "First payment at this shop"
    if group == "new_device":
        return "First payment from this device"
    if group == "shared_device":
        n = int(row["device_customers"])
        return f"Device already used on {n} other card{'s' if n != 1 else ''}"
    if group == "burst":
        if row["n_small_1h"] >= 2:
            return f"{int(row['n_1h'])} payments in the last hour, {int(row['n_small_1h'])} of them under €5 (card testing)"
        if row["n_1h"] >= 2:
            return f"{int(row['n_1h'])} payments in the last hour"
        return f"{int(row['n_24h'])} payments in the last 24 h, €{row['amt_24h']:,.0f} in total"
    if group == "history":
        return f"New card: only {int(row['n_prior'])} earlier payments"
    if group == "shop_type":
        return f"Risky shop type: {str(row['category']).replace('_', ' ')}"
    return group


def reasons(model, X, cities=None, top=3):
    """Return a list (one per row of X) of [(reason text, risk drop in points), ...]."""
    base = model.predict_proba(X)
    drops = {g: base - model.predict_proba(_reset(X, g)) for g in GROUPS}
    cities = list(cities) if cities is not None else [None] * len(X)
    out = []
    for i in range(len(X)):
        row = X.iloc[i]
        ranked = sorted(((drops[g][i], g) for g in GROUPS if _unusual(g, row)), reverse=True)
        items = [(_text(g, row, cities[i]), round(100 * d)) for d, g in ranked if d >= MIN_DROP][:top]
        out.append(items)
    return out

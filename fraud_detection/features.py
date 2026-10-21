"""Behavioural features for each transaction, using only what was known at that moment.

A fraud analyst asks: is this normal *for this customer*? So most features
compare the transaction with the customer's own history (earlier transactions
only), plus a few signals across customers, such as a device seen on other
people's cards.
"""

import numpy as np
import pandas as pd

from .simulate import CAT_NAMES, CITIES, haversine_km

FEATURES = [
    "amount", "amount_ratio", "hour", "hour_dev", "online", "category",
    "dist_home_km", "speed_kmh", "new_merchant", "new_device", "device_customers",
    "n_prior", "mins_since_prev", "n_1h", "n_24h", "amt_24h", "n_small_1h",
]
SMALL_AMOUNT = 5.0       # euros; card testers use tiny payments
HISTORY = 50             # transactions used for the customer's usual amount
MIN_HISTORY = 3


def _circular_dev(hours, prior_sin, prior_cos):
    """Hours between this transaction and the customer's usual time of day (0 to 12)."""
    usual = (np.degrees(np.arctan2(prior_sin, prior_cos)) / 15) % 24
    d = np.abs(hours - usual) % 24
    return np.minimum(d, 24 - d)


def build_features(tx, customers):
    """tx: transactions sorted by time. customers: table with customer_id and home city.

    Returns a DataFrame aligned with tx, containing FEATURES.
    """
    if not tx["time"].is_monotonic_increasing:
        raise ValueError("transactions must be sorted by time")
    out = pd.DataFrame(index=tx.index)
    t = tx["time"].to_numpy(np.int64)
    amount = tx["amount"].to_numpy(float)
    online = tx["online"].to_numpy(bool)
    cust = tx["customer_id"].to_numpy()
    hour = (t % 86400) / 3600.0

    out["amount"] = amount
    out["hour"] = hour
    out["online"] = online.astype(int)
    out["category"] = pd.Categorical(tx["merchant_category"], categories=CAT_NAMES)

    # Distance from the card holder's home city (in-store payments only).
    home = customers.set_index("customer_id")["home"].reindex(cust).to_numpy()
    lat = np.array([CITIES[c][0] if isinstance(c, str) else np.nan for c in tx["city"]])
    lon = np.array([CITIES[c][1] if isinstance(c, str) else np.nan for c in tx["city"]])
    hlat = np.array([CITIES[c][0] for c in home])
    hlon = np.array([CITIES[c][1] for c in home])
    out["dist_home_km"] = np.where(online, np.nan, haversine_km(lat, lon, hlat, hlon))

    n = len(tx)
    cols = {k: np.full(n, np.nan) for k in ("amount_ratio", "hour_dev", "speed_kmh", "mins_since_prev")}
    for k in ("n_prior", "n_1h", "n_24h", "amt_24h", "n_small_1h"):
        cols[k] = np.zeros(n)

    order = np.argsort(cust, kind="stable")            # rows grouped by customer, time order kept
    bounds = np.flatnonzero(np.diff(cust[order])) + 1
    for rows in np.split(order, bounds):
        tc, ac = t[rows], amount[rows]
        k = len(rows)
        idx = np.arange(k)
        cols["n_prior"][rows] = idx
        gaps = np.diff(tc, prepend=np.nan) / 60
        cols["mins_since_prev"][rows] = gaps
        csum = np.concatenate([[0.0], np.cumsum(ac)])
        small = np.concatenate([[0], np.cumsum(ac < SMALL_AMOUNT)])
        start_1h = np.searchsorted(tc, tc - 3600, side="left")
        start_24h = np.searchsorted(tc, tc - 86400, side="left")
        cols["n_1h"][rows] = idx - start_1h
        cols["n_24h"][rows] = idx - start_24h
        cols["amt_24h"][rows] = csum[idx] - csum[start_24h]
        cols["n_small_1h"][rows] = small[idx] - small[start_1h]

        usual = pd.Series(ac).shift(1).rolling(HISTORY, min_periods=MIN_HISTORY).median().to_numpy()
        cols["amount_ratio"][rows] = ac / usual

        hc = hour[rows]
        ang = np.radians(hc * 15)
        ps = np.concatenate([[0.0], np.cumsum(np.sin(ang))])[idx]
        pc = np.concatenate([[0.0], np.cumsum(np.cos(ang))])[idx]
        dev = _circular_dev(hc, ps, pc)
        dev[idx < MIN_HISTORY] = np.nan
        cols["hour_dev"][rows] = dev

        # Speed needed to get here from the previous in-store payment.
        instore = np.flatnonzero(~online[rows])
        if len(instore) > 1:
            r = rows[instore]
            km = haversine_km(lat[r[1:]], lon[r[1:]], lat[r[:-1]], lon[r[:-1]])
            hrs = np.maximum(np.diff(t[r]), 60) / 3600
            cols["speed_kmh"][r[1:]] = np.minimum(km / hrs, 5000)

    for k, v in cols.items():
        out[k] = v

    # Novelty: first time this customer pays this shop / uses this device;
    # and how many other customers were already seen on this device.
    seen_merchant, seen_device, device_owners = {}, {}, {}
    new_merchant = np.zeros(n, int)
    new_device = np.zeros(n, int)
    device_customers = np.zeros(n, int)
    merchants = tx["merchant_id"].to_numpy()
    devices = tx["device_id"].to_numpy(object)
    for i in range(n):
        c = cust[i]
        m = seen_merchant.setdefault(c, set())
        if merchants[i] not in m:
            new_merchant[i] = 1
            m.add(merchants[i])
        if online[i]:
            d = devices[i]
            mine = seen_device.setdefault(c, set())
            if d not in mine:
                new_device[i] = 1
                mine.add(d)
            owners = device_owners.setdefault(d, set())
            device_customers[i] = len(owners - {c})
            owners.add(c)
    out["new_merchant"] = new_merchant
    out["new_device"] = new_device
    out["device_customers"] = device_customers
    return out[FEATURES]

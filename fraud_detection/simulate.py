"""Simulate six months of card transactions with realistic fraud.

Each customer has a home city, a spending level, usual hours, favourite shops
and one or two devices for online shopping. On top of normal spending, some
customers travel, buy something expensive or get a new phone, so not every
unusual transaction is fraud.

Fraud is injected as whole cases, the way it happens in practice:
- card testing: a burst of tiny online payments on a new device, then big ones
- stolen card: in-store purchases in another city (or nearby), several in a few hours
- account takeover: large online purchases from a new device, often at night
- subtle: one or two ordinary-looking online purchases, half from the customer's own device

Everything is driven by one random seed, so the data is the same on every run.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

START = datetime(2026, 1, 1)
DAYS = 181                       # 1 Jan to 30 Jun 2026

CITIES = {  # name: (lat, lon)
    "Porto": (41.15, -8.61), "Lisbon": (38.72, -9.14), "Braga": (41.55, -8.42),
    "Coimbra": (40.21, -8.43), "Aveiro": (40.64, -8.65), "Faro": (37.02, -7.93),
    "Madrid": (40.42, -3.70), "Barcelona": (41.39, 2.17), "Paris": (48.86, 2.35),
    "London": (51.51, -0.13), "Berlin": (52.52, 13.40), "Amsterdam": (52.37, 4.90),
    "Rome": (41.90, 12.50), "Bucharest": (44.43, 26.10), "New York": (40.71, -74.01),
}
CITY_NAMES = list(CITIES)
HOME_CITIES = ["Porto", "Lisbon", "Braga", "Coimbra", "Aveiro", "Faro"]
HOME_WEIGHTS = [0.40, 0.30, 0.10, 0.08, 0.07, 0.05]

# category: (amount scale vs the customer's usual, in-store, online, base preference)
CATEGORIES = {
    "grocery":       (1.0, True,  False, 0.26),
    "restaurant":    (1.2, True,  False, 0.16),
    "fuel":          (2.0, True,  False, 0.08),
    "transport":     (0.5, True,  True,  0.10),
    "pharmacy":      (0.8, True,  False, 0.05),
    "clothing":      (2.5, True,  True,  0.07),
    "electronics":   (6.0, True,  True,  0.03),
    "travel":        (10., False, True,  0.01),
    "hotel":         (6.0, True,  True,  0.01),
    "marketplace":   (2.0, False, True,  0.10),
    "digital_goods": (0.4, False, True,  0.08),
    "gift_cards":    (3.0, True,  True,  0.01),
    "jewelry":       (8.0, True,  False, 0.005),
    "entertainment": (1.0, True,  True,  0.035),
}
CAT_NAMES = list(CATEGORIES)
FRAUD_TYPES = ["card_testing", "stolen_card", "account_takeover", "subtle"]
COLUMNS = ["time", "customer_id", "amount", "category", "merchant_id", "city", "online",
           "device_id", "is_fraud", "fraud_type", "case_id"]


@dataclass
class Settings:
    customers: int = 3000
    seed: int = 7
    fraud_cases: int = 330
    fraud_type_mix: tuple = (0.20, 0.30, 0.25, 0.25)
    travel_share: float = 0.25
    new_phone_share: float = 0.15
    gamer_share: float = 0.06          # bursts of small in-app payments
    shared_device_share: float = 0.10  # family members sharing a tablet or laptop
    fresh_device_rate: float = 0.02    # online payments from a device used once (work laptop, friend's phone)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _merchants(rng):
    """Shops per city and category, plus online shops. Returns (table, lookup)."""
    rows, lookup = [], {}
    for city in CITY_NAMES:
        for cat, (_, in_store, _, _) in CATEGORIES.items():
            if in_store:
                n = 6 if city in HOME_CITIES else 3
                for _ in range(n):
                    lookup.setdefault((city, cat, False), []).append(len(rows))
                    rows.append((cat, city, False))
    for cat, (_, _, online, _) in CATEGORIES.items():
        if online:
            for _ in range(25):
                lookup.setdefault((None, cat, True), []).append(len(rows))
                rows.append((cat, None, True))
    table = pd.DataFrame(rows, columns=["category", "city", "online"])
    return table, {k: np.array(v) for k, v in lookup.items()}


class _Builder:
    def __init__(self, settings):
        self.s = settings
        self.rng = np.random.default_rng(settings.seed)
        self.merchants, self.lookup = _merchants(self.rng)
        self.cols = {k: [] for k in COLUMNS}
        self.next_device = 0

    def new_device(self):
        self.next_device += 1
        return f"D{self.next_device:06d}"

    def add(self, t, cust, amount, cat, merchant, city, online, device, fraud_type=None, case=-1):
        self.add_many(dict(time=[t], customer_id=[cust], amount=[round(float(amount), 2)], category=[cat],
                           merchant_id=[int(merchant)], city=[city], online=[online],
                           device_id=[device if online else None], is_fraud=[fraud_type is not None],
                           fraud_type=[fraud_type], case_id=[case]))

    def add_many(self, cols):
        n = len(cols["time"])
        for k in COLUMNS:
            v = cols[k]
            self.cols[k].append(np.asarray(v, dtype=object) if k in ("city", "device_id", "fraud_type", "category")
                                else np.broadcast_to(np.asarray(v), (n,)))

    def frame(self):
        return pd.DataFrame({k: np.concatenate(v) for k, v in self.cols.items()})

    def pick_merchant(self, cat, city, online):
        pool = self.lookup[(None, cat, True)] if online else self.lookup[(city, cat, False)]
        return self.rng.choice(pool)


def _hour(rng, mean, sd=2.8):
    return (rng.normal(mean, sd) % 24)


def simulate(settings=None):
    """Return a DataFrame of transactions sorted by time (seconds since 1 Jan 2026)."""
    s = settings or Settings()
    b = _Builder(s)
    rng = b.rng
    cats = np.array(CAT_NAMES)
    base_pref = np.array([v[3] for v in CATEGORIES.values()])
    in_store = np.array([v[1] for v in CATEGORIES.values()])
    can_online = np.array([v[2] for v in CATEGORIES.values()])
    customers = []

    for c in range(s.customers):
        home = rng.choice(HOME_CITIES, p=HOME_WEIGHTS)
        level = rng.lognormal(np.log(22), 0.45)              # usual spend, euros
        rate = rng.gamma(2.0, 0.3)                           # transactions per day
        night_owl = rng.random() < 0.08
        mean_hour = rng.uniform(20, 23.5) if night_owl else np.clip(rng.normal(14.5, 2.2), 9, 20)
        online_share = rng.beta(2, 5)
        pref = rng.dirichlet(base_pref * 40)
        devices = [b.new_device() for _ in range(1 + (rng.random() < 0.3))]
        if customers and rng.random() < s.shared_device_share:
            devices.append(customers[rng.integers(len(customers))]["devices"][0])
        fav_store = {cat: rng.choice(b.lookup[(home, cat, False)], size=2)
                     for cat in cats[in_store]}
        fav_online = {cat: rng.choice(b.lookup[(None, cat, True)], size=2) for cat in cats[can_online]}

        trips = []
        if rng.random() < s.travel_share:
            for _ in range(rng.integers(1, 3)):
                start = rng.uniform(0, DAYS - 8)
                dest = rng.choice([x for x in CITY_NAMES if x != home])
                trips.append((start, start + rng.uniform(2, 7), dest))
        phone_day = rng.uniform(10, DAYS) if rng.random() < s.new_phone_share else None
        new_phone = b.new_device() if phone_day is not None else None
        customers.append(dict(home=home, level=level, mean_hour=mean_hour, devices=devices))

        # Book the trips online a week or two ahead: big, legitimate purchases.
        for start, _, _ in trips:
            t = max(0.0, start - rng.uniform(5, 20)) * 86400 + _hour(rng, mean_hour) * 3600
            for cat in ("travel", "hotel"):
                b.add(t, c, level * CATEGORIES[cat][0] * rng.lognormal(0, 0.4), cat,
                      b.pick_merchant(cat, None, True), None, True, devices[0])
                t += rng.uniform(60, 1800)

        if rng.random() < s.gamer_share:
            _gaming(b, c, devices[0], mean_hour)

        n = rng.poisson(rate * DAYS)
        if n:
            _spend(b, c, n, home, level, mean_hour, online_share, pref, devices, fav_store, fav_online,
                   trips, phone_day, new_phone)

    _inject_fraud(b, customers)
    df = b.frame()
    df = df[(df["time"] >= 0) & (df["time"] < DAYS * 86400)]
    df = df.sort_values(["time", "customer_id"], kind="mergesort").reset_index(drop=True)
    df["time"] = df["time"].round().astype(int)
    df.insert(0, "transaction_id", np.arange(len(df)))
    df = df.rename(columns={"category": "merchant_category"})
    df["merchant_category"] = df["merchant_category"].astype(str)
    customers = pd.DataFrame(customers).rename_axis("customer_id").reset_index()
    return df, customers, b.merchants


def _spend(b, c, n, home, level, mean_hour, online_share, pref, devices, fav_store, fav_online,
           trips, phone_day, new_phone):
    """A customer's normal spending: n transactions, generated in one go."""
    rng = b.rng
    cats = np.array(CAT_NAMES)
    in_store = np.array([v[1] for v in CATEGORIES.values()])
    can_online = np.array([v[2] for v in CATEGORIES.values()])
    scale = np.array([v[0] for v in CATEGORIES.values()])
    days = np.sort(rng.uniform(0, DAYS, n).astype(int))
    t = days * 86400 + (rng.normal(mean_hour, 2.8, n) % 24) * 3600
    city = np.full(n, home, dtype=object)
    on_trip = np.zeros(n, bool)
    for start, stop, dest in trips:
        m = (days + 0.5 >= start) & (days + 0.5 <= stop)
        city[m], on_trip[m] = dest, True
    online = rng.random(n) < online_share
    cat_idx = np.empty(n, int)
    for flag, mask in ((True, can_online), (False, in_store)):
        sel = online == flag
        p = pref * mask
        cat_idx[sel] = rng.choice(len(cats), size=sel.sum(), p=p / p.sum())
    amount = level * scale[cat_idx] * rng.lognormal(0, 0.55, n)
    big = np.isin(cats[cat_idx], ["electronics", "travel", "jewelry", "clothing"]) & (rng.random(n) < 0.06)
    amount[big] *= rng.uniform(3, 10, big.sum())             # a planned big purchase
    amount = np.maximum(amount, 0.5).round(2)

    merchant = np.empty(n, int)
    fav = (rng.random(n) < np.where(online, 0.7, 0.75)) & ~(on_trip & ~online)
    pick = rng.integers(0, 2, n)
    for ci in np.unique(cat_idx):
        cat = cats[ci]
        for flag, favs in ((True, fav_online), (False, fav_store)):
            m = (cat_idx == ci) & (online == flag)
            if not m.any():
                continue
            mf, mr = m & fav, m & ~fav
            merchant[mf] = np.asarray(favs[cat])[pick[mf]]
            if flag:
                pool = b.lookup[(None, cat, True)]
                merchant[mr] = pool[rng.integers(0, len(pool), mr.sum())]
            else:
                for town in np.unique(city[mr]):
                    mt = mr & (city == town)
                    pool = b.lookup[(town, cat, False)]
                    merchant[mt] = pool[rng.integers(0, len(pool), mt.sum())]
    device = np.where(online, np.array(devices, dtype=object)[rng.integers(0, len(devices), n)], None)
    if phone_day is not None:
        device[online & (days >= phone_day)] = new_phone
    fresh = online & (rng.random(n) < b.s.fresh_device_rate)
    device[fresh] = [b.new_device() for _ in range(fresh.sum())]
    b.add_many({
        "time": t, "customer_id": np.full(n, c), "amount": amount, "category": cats[cat_idx],
        "merchant_id": merchant, "city": np.where(online, None, city), "online": online, "device_id": device,
        "is_fraud": np.zeros(n, bool), "fraud_type": np.full(n, None, dtype=object),
        "case_id": np.full(n, -1)})


def _gaming(b, c, device, mean_hour):
    """In-app purchases: several small payments within minutes, a few times a month."""
    rng = b.rng
    for _ in range(rng.poisson(DAYS / 12)):
        t = rng.uniform(0, DAYS) // 1 * 86400 + (rng.normal(mean_hour + 2, 2) % 24) * 3600
        merchant = b.pick_merchant("digital_goods", None, True)
        for _ in range(rng.integers(2, 7)):
            b.add(t, c, rng.choice([0.99, 1.99, 2.99, 4.99]), "digital_goods", merchant, None, True, device)
            t += rng.uniform(30, 600)


def _inject_fraud(b, customers):
    rng, s = b.rng, b.s
    fraud_devices = [b.new_device() for _ in range(40)]   # devices fraudsters reuse
    victims = rng.choice(len(customers), size=s.fraud_cases, replace=True)
    kinds = rng.choice(FRAUD_TYPES, size=s.fraud_cases, p=s.fraud_type_mix)
    for case, (c, kind) in enumerate(zip(victims.tolist(), kinds.tolist())):
        cust = customers[c]
        t = rng.uniform(1, DAYS - 1) * 86400
        level = cust["level"]
        if kind == "card_testing":
            device = rng.choice(fraud_devices) if rng.random() < 0.35 else b.new_device()
            t = (t // 86400) * 86400 + _hour(rng, 2.5, 4) * 3600
            for _ in range(rng.integers(3, 9)):
                cat = rng.choice(["digital_goods", "marketplace"])
                b.add(t, c, rng.uniform(0.5, 3.0), cat, b.pick_merchant(cat, None, True), None, True,
                      device, kind, case)
                t += rng.uniform(40, 400)
            t += rng.uniform(600, 7200)
            for _ in range(rng.integers(1, 4)):
                cat = rng.choice(["electronics", "gift_cards"])
                b.add(t, c, rng.uniform(150, 1200), cat, b.pick_merchant(cat, None, True), None, True,
                      device, kind, case)
                t += rng.uniform(120, 3600)
        elif kind == "stolen_card":
            city = cust["home"] if rng.random() < 0.3 else rng.choice([x for x in CITY_NAMES if x != cust["home"]])
            t = (t // 86400) * 86400 + rng.uniform(10, 21) * 3600
            for _ in range(rng.integers(2, 7)):
                cat = rng.choice(["electronics", "jewelry", "clothing", "gift_cards", "fuel"],
                                 p=[0.3, 0.15, 0.25, 0.15, 0.15])
                amount = level * CATEGORIES[cat][0] * rng.uniform(1.2, 4)
                b.add(t, c, amount, cat, b.pick_merchant(cat, city, False), city, False, None, kind, case)
                t += rng.uniform(600, 4 * 3600)
        elif kind == "account_takeover":
            device = rng.choice(fraud_devices) if rng.random() < 0.3 else b.new_device()
            night = rng.random() < 0.4
            t = (t // 86400) * 86400 + (rng.uniform(0, 5) if night else _hour(rng, 15, 3)) * 3600
            for _ in range(rng.integers(1, 5)):
                cat = rng.choice(["electronics", "travel", "gift_cards", "marketplace"])
                # careful fraudsters stay close to what the customer normally spends
                amount = (level * CATEGORIES[cat][0] * rng.uniform(1.5, 5) if rng.random() < 0.5
                          else rng.uniform(100, 1500))
                b.add(t, c, amount, cat, b.pick_merchant(cat, None, True), None, True,
                      device, kind, case)
                t += rng.uniform(300, 2 * 3600)
        else:  # subtle: half of it from the customer's own device (malware, or someone at home)
            device = rng.choice(cust["devices"]) if rng.random() < 0.5 else b.new_device()
            t = (t // 86400) * 86400 + _hour(rng, cust["mean_hour"]) * 3600
            for _ in range(rng.integers(1, 3)):
                cat = rng.choice(["marketplace", "clothing", "entertainment"])
                amount = level * CATEGORIES[cat][0] * rng.uniform(1.0, 2.5)
                b.add(t, c, amount, cat, b.pick_merchant(cat, None, True), None, True, device, kind, case)
                t += rng.uniform(1800, 6 * 3600)


def to_datetime(seconds):
    return pd.Timestamp(START) + pd.to_timedelta(seconds, unit="s")

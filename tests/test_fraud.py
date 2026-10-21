"""Tests. Run with: python -m unittest"""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from fraud_detection import evaluate as ev
from fraud_detection import explain, models, service
from fraud_detection.features import FEATURES, build_features
from fraud_detection.simulate import Settings, simulate

SMALL = Settings(customers=600, fraud_cases=80, seed=3)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tx, cls.customers, cls.merchants = simulate(SMALL)

    def test_same_seed_same_data(self):
        again, _, _ = simulate(SMALL)
        pd.testing.assert_frame_equal(self.tx, again)

    def test_sorted_and_within_six_months(self):
        self.assertTrue(self.tx["time"].is_monotonic_increasing)
        self.assertGreaterEqual(self.tx["time"].min(), 0)
        self.assertLess(self.tx["time"].max(), 181 * 86400)

    def test_fraud_is_rare_and_every_fraud_has_a_case(self):
        rate = self.tx["is_fraud"].mean()
        self.assertTrue(0.001 < rate < 0.01, rate)
        fraud = self.tx[self.tx["is_fraud"]]
        self.assertTrue((fraud["case_id"] >= 0).all())
        self.assertTrue((self.tx.loc[~self.tx["is_fraud"], "case_id"] == -1).all())
        self.assertEqual(set(fraud["fraud_type"]), {"card_testing", "stolen_card", "account_takeover", "subtle"})

    def test_online_payments_have_a_device_and_no_city(self):
        online = self.tx[self.tx["online"]]
        self.assertTrue(online["device_id"].notna().all())
        self.assertTrue(online["city"].isna().all())
        self.assertTrue(self.tx.loc[~self.tx["online"], "city"].notna().all())


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tx, cls.customers, _ = simulate(SMALL)
        cls.X = build_features(cls.tx, cls.customers)

    def test_no_future_leakage(self):
        """Changing everything after a moment must not change any earlier features."""
        cut = self.tx["time"].quantile(0.5)
        changed = self.tx.copy()
        later = changed["time"] > cut
        changed.loc[later, "amount"] = 9999.0
        changed.loc[later, "merchant_id"] = 0
        changed.loc[later & changed["online"], "device_id"] = "D-ATTACK"
        X2 = build_features(changed, self.customers)
        early = ~later.to_numpy()
        pd.testing.assert_frame_equal(self.X[early], X2[early])

    def test_columns_and_first_payment(self):
        self.assertEqual(list(self.X.columns), FEATURES)
        first = self.tx.groupby("customer_id").head(1).index
        self.assertTrue((self.X.loc[first, "n_prior"] == 0).all())
        self.assertTrue((self.X.loc[first, "new_merchant"] == 1).all())
        self.assertTrue(self.X.loc[first, "amount_ratio"].isna().all())

    def test_velocity_counts(self):
        tx = pd.DataFrame({"time": [0, 600, 1200, 1800, 90000], "customer_id": 0, "amount": [1, 2, 3, 100, 5.0],
                           "merchant_category": "digital_goods", "merchant_id": [1, 1, 2, 3, 3],
                           "city": None, "online": True, "device_id": "A"})
        X = build_features(tx, pd.DataFrame({"customer_id": [0], "home": ["Porto"]}))
        self.assertEqual(X["n_1h"].tolist(), [0, 1, 2, 3, 0])
        self.assertEqual(X["n_small_1h"].tolist(), [0, 1, 2, 3, 0])
        self.assertEqual(X["new_merchant"].tolist(), [1, 0, 1, 1, 0])
        self.assertEqual(X["new_device"].tolist(), [1, 0, 0, 0, 0])

    def test_impossible_travel(self):
        tx = pd.DataFrame({"time": [0, 3600], "customer_id": 0, "amount": [20.0, 30.0], "merchant_category": "grocery",
                           "merchant_id": [1, 2], "city": ["Porto", "New York"], "online": False, "device_id": None})
        X = build_features(tx, pd.DataFrame({"customer_id": [0], "home": ["Porto"]}))
        self.assertGreater(X["speed_kmh"].iloc[1], 5000 - 1)
        self.assertGreater(X["dist_home_km"].iloc[1], 5000)


class MoneyTests(unittest.TestCase):
    def test_blocking_stops_the_rest_of_the_case(self):
        tx = pd.DataFrame({"time": [1, 2, 3, 4], "amount": [10.0, 20.0, 30.0, 5.0],
                           "is_fraud": [True, True, True, False], "case_id": [7, 7, 7, -1]})
        m = ev.money(tx, np.array([False, True, False, False]))
        self.assertEqual(m["stopped_eur"], 50.0)       # the flagged payment and everything after it
        self.assertEqual(m["fraud_eur"], 60.0)
        self.assertEqual(m["cases_caught"], 1)

    def test_outcome_counts_review_cost(self):
        tx = pd.DataFrame({"time": [0, 86400], "amount": [100.0, 50.0], "is_fraud": [True, False], "case_id": [0, -1]})
        o = ev.outcome(tx, np.array([0.9, 0.8]), 0.5)
        self.assertEqual(o["alerts"], 2)
        self.assertEqual(o["net_saving_eur"], 100.0 - 2 * ev.REVIEW_COST)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tx, cls.customers, _ = simulate(Settings(customers=1200, fraud_cases=160, seed=5))
        cls.X = build_features(cls.tx, cls.customers)
        cls.y = cls.tx["is_fraud"].to_numpy()
        cls.split = models.time_split(cls.tx)

    def test_split_is_by_time_without_overlap(self):
        tr, va, te = self.split
        self.assertFalse((tr & va).any() or (va & te).any() or (tr & te).any())
        self.assertEqual(int(tr.sum() + va.sum() + te.sum()), len(self.tx))
        self.assertLess(self.tx["time"][tr].max(), self.tx["time"][va].min())
        self.assertLess(self.tx["time"][va].max(), self.tx["time"][te].min())

    def test_models_rank_in_the_expected_order(self):
        tr, _, te = self.split
        pr = {}
        for m in models.all_models():
            m.fit(self.X[tr], self.y[tr])
            pr[m.name] = ev.ranking_scores(self.y[te], m.predict_proba(self.X[te]))["pr_auc"]
        self.assertGreater(pr["gradient_boosting"], pr["rules"])
        self.assertGreater(pr["logistic"], pr["rules"])
        self.assertGreater(pr["gradient_boosting"], 0.5)

    def test_best_threshold_beats_extremes(self):
        tr, va, _ = self.split
        m = models.BoostingModel(max_iter=150).fit(self.X[tr], self.y[tr])
        s = m.predict_proba(self.X[va])
        t = ev.best_threshold(self.tx[va], s)
        best = ev.outcome(self.tx[va], s, t)["net_saving_eur"]
        self.assertGreaterEqual(best, ev.outcome(self.tx[va], s, np.quantile(s, 0.95))["net_saving_eur"])
        self.assertGreaterEqual(best, ev.outcome(self.tx[va], s, s.max() + 1)["net_saving_eur"])


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = service.build(Settings(customers=800, fraud_cases=110, seed=11))

    def test_alerts_have_reasons_from_unusual_signals(self):
        a = service.alerts(self.state)
        self.assertGreater(len(a), 0)
        self.assertTrue((a["risk"] >= self.state["threshold"]).all())
        self.assertTrue(a["risk"].is_monotonic_decreasing)
        self.assertTrue(all(isinstance(r, list) for r in a["reasons"]))

    def test_card_testing_scores_high_and_says_why(self):
        cid = int(self.state["customers"]["customer_id"].iloc[3])
        normal, _, _ = service.score_payment(self.state, cid, **service.PRESETS["Weekly groceries"])
        risky, why, _ = service.score_payment(self.state, cid, **service.PRESETS["Card testing"])
        self.assertLess(normal, self.state["threshold"])
        self.assertGreater(risky, self.state["threshold"])
        self.assertTrue(any("in the last hour" in text for text, _ in why))

    def test_reasons_only_use_unusual_signals(self):
        row = pd.DataFrame([{**{f: 0 for f in FEATURES}, "amount": 20.0, "amount_ratio": 1.0, "hour": 14.0,
                             "hour_dev": 1.0, "n_prior": 80, "category": "grocery"}])
        row["category"] = pd.Categorical(row["category"], categories=self.state["X"]["category"].cat.categories)
        for g in explain.GROUPS:
            self.assertFalse(explain._unusual(g, row.iloc[0]), g)


class BenchmarkTests(unittest.TestCase):
    def test_runs_on_a_small_file_with_the_real_columns(self):
        import benchmark_ulb
        rng = np.random.default_rng(0)
        n = 4000
        df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
        df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)).round())
        df["Amount"] = rng.lognormal(3, 1, n).round(2)
        df["Class"] = (rng.random(n) < 0.02).astype(int)
        df.loc[df["Class"] == 1, "V14"] -= 4          # make fraud learnable
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "creditcard.csv")
            df.to_csv(path, index=False)
            out = os.path.join(d, "bench.json")
            benchmark_ulb.main(path, out)
            with open(out) as f:
                report = json.load(f)
        self.assertEqual(report["data"]["rows"], n)
        self.assertGreater(report["models"]["gradient_boosting"]["pr_auc"], 0.5)


if __name__ == "__main__":
    unittest.main()

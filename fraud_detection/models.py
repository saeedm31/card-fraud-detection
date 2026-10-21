"""Three ways to score transactions, from simplest to best.

- Rules: the kind of hand-written checks many banks start with.
- Logistic regression: a simple, explainable model.
- Gradient boosting: learns interactions (e.g. a big amount is fine in a shop
  you know, suspicious on a new device at 3 am).
Each returns a risk score where higher means more likely fraud.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import FEATURES

RANDOM_STATE = 42
NUMERIC = [f for f in FEATURES if f != "category"]
SKEWED = ["amount", "amount_ratio", "dist_home_km", "speed_kmh", "mins_since_prev", "amt_24h", "n_prior"]


class RulesModel:
    """Classic rules. The score is the number of rules a transaction breaks."""
    name = "rules"
    params = {"rules": 5}
    RULES = {
        "amount over 5x usual": lambda X: X["amount_ratio"] > 5,
        "new device and over 200 euros": lambda X: (X["new_device"] == 1) & (X["amount"] > 200),
        "4+ payments in an hour": lambda X: X["n_1h"] >= 4,
        "over 500 km from home and 2x usual": lambda X: (X["dist_home_km"] > 500) & (X["amount_ratio"] > 2),
        "impossible travel (over 900 km/h)": lambda X: X["speed_kmh"] > 900,
    }

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        hits = sum(rule(X).fillna(False).astype(int) for rule in self.RULES.values())
        return hits.to_numpy() / len(self.RULES)


class LogisticModel:
    name = "logistic"

    def __init__(self, C=0.5):
        self.params = {"C": C}
        log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        pre = ColumnTransformer([
            ("skewed", make_pipeline(SimpleImputer(strategy="median"), log, StandardScaler()), SKEWED),
            ("other", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
             [f for f in NUMERIC if f not in SKEWED]),
            ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ])
        self.pipe = make_pipeline(pre, LogisticRegression(C=C, max_iter=2000))

    def fit(self, X, y):
        self.pipe.fit(X[FEATURES], y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X[FEATURES])[:, 1]


class BoostingModel:
    name = "gradient_boosting"

    def __init__(self, max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0):
        self.params = {"max_iter": max_iter, "learning_rate": learning_rate,
                       "max_leaf_nodes": max_leaf_nodes, "l2_regularization": l2_regularization}
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization, categorical_features="from_dtype",
            early_stopping=False, random_state=RANDOM_STATE)

    def fit(self, X, y):
        self.clf.fit(X[FEATURES], y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X[FEATURES])[:, 1]


def all_models():
    return [RulesModel(), LogisticModel(), BoostingModel()]


def time_split(tx, train_days=120, valid_days=31):
    """Train on Jan-Apr, pick the alert threshold on May, report on June."""
    day = tx["time"] // 86400
    return (day < train_days).to_numpy(), \
        ((day >= train_days) & (day < train_days + valid_days)).to_numpy(), \
        (day >= train_days + valid_days).to_numpy()

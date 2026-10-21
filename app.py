"""Card fraud detection: a fraud analyst's view.

    streamlit run app.py
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from fraud_detection import evaluate as ev
from fraud_detection import service
from fraud_detection.features import FEATURES
from fraud_detection.simulate import CAT_NAMES, START

st.set_page_config(page_title="Card fraud detection", page_icon="💳", layout="wide")
REPORTS = Path(__file__).parent / "reports"
MODEL_NAMES = {"rules": "Rules", "logistic": "Logistic regression", "gradient_boosting": "Gradient boosting"}


@st.cache_resource(show_spinner="Simulating six months of card payments and training the model (about 30 s, first visit only)...")
def load_state():
    return service.build()


@st.cache_data
def load_json(name):
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def alert_table(_state):
    return service.alerts(_state)


def when(seconds):
    return (pd.Timestamp(START) + pd.to_timedelta(seconds, unit="s")).strftime("%d %b %H:%M")


def where(row):
    return "online" if row["online"] else row["city"]


def euro(x):
    return f"€{x:,.0f}"


state = load_state()
test = state["test"]
thr = state["threshold"]
june = ev.outcome(test, test["risk"].to_numpy(), thr)
metrics = load_json("metrics.json")
bench = load_json("benchmark_ulb.json")

st.title("💳 Card fraud detection")
st.caption(f"A fraud analyst's view of **June 2026**: {len(test):,} simulated card payments, scored by a model "
           f"trained on January to April. The alert threshold was chosen on May to save the most money.")
c = st.columns(5)
c[0].metric("Alerts per day", f"{june['alerts_per_day']:.1f}")
c[1].metric("Alerts that are fraud", f"{june['precision']:.0%}")
c[2].metric("Fraud cases caught", f"{june['cases_caught_pct']:.0f}%")
c[3].metric("Fraud money stopped", f"{june['stopped_pct']:.0f}%", help=f"{euro(june['stopped_eur'])} of {euro(june['fraud_eur'])}")
c[4].metric("Net saving in June", euro(june["net_saving_eur"]), help=f"Fraud stopped minus {euro(ev.REVIEW_COST)} per alert reviewed")

tabs = st.tabs(["🚨 Alert queue", "🔍 Check a payment", "💶 Threshold & money", "📊 Models", "🌍 Real data", "ℹ️ About"])

# --- Alert queue -------------------------------------------------------------------------------
with tabs[0]:
    alerts = alert_table(state)
    st.markdown(f"**{len(alerts)} alerts in June**, riskiest first. When an alert fires the payment is declined "
                f"and the card is blocked. *Outcome* is what turned out to be true later; analysts don't see it.")
    show = st.radio("Show", ["All alerts", "Fraud", "False alarms"], horizontal=True, label_visibility="collapsed")
    view = alerts if show == "All alerts" else alerts[alerts["is_fraud"] == (show == "Fraud")]
    table = pd.DataFrame({
        "When": view["time"].map(when),
        "Customer": view["customer_id"],
        "Amount": view["amount"].map(lambda a: f"€{a:,.2f}"),
        "Shop type": view["merchant_category"].str.replace("_", " "),
        "Where": [where(r) for _, r in view.iterrows()],
        "Risk": (100 * view["risk"]).round(1),
        "Main reason": [r[0][0] if r else "combination of small signals" for r in view["reasons"]],
        "Outcome": np.where(view["is_fraud"], "fraud", "false alarm"),
    })
    st.dataframe(table, hide_index=True, width="stretch", height=320,
                 column_config={"Risk": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100,
                                                                        format="%.1f")})
    if len(view):
        pick = st.selectbox("Open an alert", range(len(view)), format_func=lambda i: (
            f"{table.iloc[i]['When']} · customer {table.iloc[i]['Customer']} · {table.iloc[i]['Amount']} "
            f"{table.iloc[i]['Shop type']} · {table.iloc[i]['Outcome']}"))
        row = view.iloc[pick]
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown(f"#### Risk {100 * row['risk']:.1f}%")
            st.markdown("**Why it was flagged** (how much the risk falls if this signal were normal):")
            if row["reasons"]:
                for text, points in row["reasons"]:
                    st.progress(min(points, 100) / 100, text=f"{text}  (−{points} pts)")
            else:
                st.write("No single signal stands out: several small ones add up.")
            if row["is_fraud"]:
                case = state["tx"][state["tx"]["case_id"] == row["case_id"]]
                hit = case["transaction_id"].isin(alerts["transaction_id"])
                first = case["time"][hit.to_numpy()].min()
                stopped = case.loc[case["time"] >= first, "amount"].sum()
                st.success(f"Real fraud ({str(row['fraud_type']).replace('_', ' ')}): {len(case)} payments, "
                           f"{euro(case['amount'].sum())} in total. Blocking the card at the first alert "
                           f"stopped {euro(stopped)}.")
            else:
                st.info("False alarm: the customer's own payment. An analyst call or an app notification clears it.")
        with right:
            st.markdown("**The customer's payments before this one**")
            h = service.history(state, row["customer_id"], before=row["time"], n=8)
            st.dataframe(pd.DataFrame({
                "When": h["time"].map(when), "Amount": h["amount"].map(lambda a: f"€{a:,.2f}"),
                "Shop type": h["merchant_category"].str.replace("_", " "),
                "Where": [where(r) for _, r in h.iterrows()],
            }), hide_index=True, width="stretch")

# --- Check a payment ----------------------------------------------------------------------------
with tabs[1]:
    st.markdown("Make up a payment for a real (simulated) customer and see what the model says. "
                "Start from an example, then change anything.")
    customers = state["customers"]
    usual = state["tx"].groupby("customer_id")["amount"].median()
    left, right = st.columns([1, 1])
    with left:
        preset = st.radio("Example", list(service.PRESETS), horizontal=True)
        p = service.PRESETS[preset]
        options = customers["customer_id"].head(60).tolist()
        cid = st.selectbox("Customer", options, index=5, format_func=lambda i: (
            f"Customer {i} · lives in {customers.loc[i, 'home']} · usually spends €{usual.get(i, 0):.0f}"))
        a, b = st.columns(2)
        amount = a.number_input("Amount (€)", 0.5, 20000.0, float(p["amount"]), step=10.0, key=f"amt{preset}")
        category = b.selectbox("Shop type", CAT_NAMES, CAT_NAMES.index(p["category"]), key=f"cat{preset}",
                               format_func=lambda x: x.replace("_", " "))
        online = a.toggle("Online payment", p["online"], key=f"onl{preset}")
        hour = b.slider("Time of day", 0.0, 23.75, float(p["hour"]), 0.25, key=f"hr{preset}",
                        format="%.2f h")
        if online:
            device = a.radio("Device", ["own", "new", "shared"], ["own", "new", "shared"].index(p["device"]),
                             key=f"dev{preset}", format_func={"own": "their usual phone",
                                                              "new": "never seen before",
                                                              "shared": "seen on 3 other cards"}.get)
            city = "home"
        else:
            device = "own"
            cities = ["home"] + [x for x in service.city_names()]
            city = a.selectbox("City", cities, cities.index(p["city"]) if p["city"] in cities else 0,
                               key=f"city{preset}", format_func=lambda x: "their home city" if x == "home" else x)
        shop = b.radio("Shop", ["usual", "new"], ["usual", "new"].index(p["shop"]), key=f"shop{preset}",
                       format_func={"usual": "one they use", "new": "first time"}.get)
        burst = b.slider("Tiny payments in the hour before", 0, 10, int(p["burst"]), key=f"burst{preset}")
    with right:
        risk, why, x = service.score_payment(state, cid, amount, category, online, city, hour, device, shop, burst)
        flagged = risk >= thr
        st.markdown(f"### Risk {100 * risk:.1f}%")
        st.progress(min(risk, 1.0))
        if flagged:
            st.error(f"🚨 Alert: decline the payment and block the card (threshold {100 * thr:.1f}%).")
        else:
            st.success(f"✅ Approved (below the {100 * thr:.1f}% alert threshold).")
        if why:
            st.markdown("**Why**")
            for text, points in why:
                st.progress(min(points, 100) / 100, text=f"{text}  (−{points} pts)")
        with st.expander("Features the model saw"):
            st.dataframe(x.to_frame("value").astype(str), width="stretch")

# --- Threshold & money ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("Every alert costs analyst time; every missed fraud costs money. Choose how many alerts the team "
                f"can handle per day and see what June would have looked like. Review cost: {euro(ev.REVIEW_COST)} per alert.")
    days = (test["time"].max() - test["time"].min()) / 86400
    per_day = st.slider("Alerts per day", 1, 100, int(round(june["alerts_per_day"])))
    scores = test["risk"].to_numpy()
    o = ev.outcome(test, scores, ev.budget_threshold(scores, per_day * days / len(test)))
    c = st.columns(4)
    c[0].metric("Fraud cases caught", f"{o['cases_caught_pct']:.0f}%")
    c[1].metric("Fraud money stopped", euro(o["stopped_eur"]), f"{o['stopped_pct']:.0f}% of {euro(o['fraud_eur'])}",
                delta_color="off")
    c[2].metric("Review cost", euro(o["review_cost_eur"]))
    c[3].metric("Net saving", euro(o["net_saving_eur"]))
    curve = pd.DataFrame(metrics["models"]["gradient_boosting"]["threshold_curve"]) if metrics else pd.DataFrame()
    if len(curve):
        base = alt.Chart(curve).encode(x=alt.X("alerts_per_day:Q", scale=alt.Scale(type="log"),
                                               title="Alerts per day (log scale)"))
        line = base.mark_line(strokeWidth=2.5).encode(
            y=alt.Y("net_saving_eur:Q", title="Net saving in June (€)"),
            tooltip=[alt.Tooltip("alerts_per_day:Q", format=".1f"), alt.Tooltip("net_saving_eur:Q", format=",.0f"),
                     alt.Tooltip("precision:Q", format=".0%"), alt.Tooltip("stopped_pct:Q", format=".0f")])
        rule = alt.Chart(pd.DataFrame({"x": [per_day]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(line + rule, width="stretch")
        st.caption("Too few alerts and fraud slips through; too many and the review cost eats the savings. "
                   "The best point is around 10 a day.")

# --- Models --------------------------------------------------------------------------------------
with tabs[3]:
    if metrics:
        st.markdown(f"All three were trained on Jan-Apr and tested on June ({metrics['split']['test_payments']:,} "
                    f"payments, {metrics['split']['test_frauds']} of them fraud in {metrics['split']['test_cases']} cases).")
        rows = []
        for key, m in metrics["models"].items():
            at = m["at_threshold"]
            rows.append({"Model": MODEL_NAMES[key], "PR-AUC": round(m["pr_auc"], 3),
                         "Alerts/day": round(at["alerts_per_day"], 1), "Precision": f"{at['precision']:.0%}",
                         "Fraud caught": f"{at['recall']:.0%}", "Money stopped": f"{at['stopped_pct']:.0f}%",
                         "Net saving": euro(at["net_saving_eur"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pr = pd.concat([pd.DataFrame(m["pr_curve"]).assign(model=MODEL_NAMES[k])
                        for k, m in metrics["models"].items()])
        st.altair_chart(alt.Chart(pr).mark_line(strokeWidth=2.5).encode(
            x=alt.X("recall:Q", title="Share of fraud caught"), y=alt.Y("precision:Q", title="Share of alerts that are fraud"),
            color=alt.Color("model:N", title=None)).properties(height=320), width="stretch")
        by_type = pd.DataFrame([{"Model": MODEL_NAMES[k], "Fraud type": t.replace("_", " "), "Caught": v}
                                for k, m in metrics["models"].items() for t, v in m["recall_by_type"].items()])
        st.markdown("**Which fraud gets caught**")
        st.altair_chart(alt.Chart(by_type).mark_bar().encode(
            x=alt.X("Caught:Q", axis=alt.Axis(format="%")), y=alt.Y("Model:N", title=None),
            color=alt.Color("Model:N", legend=None), row=alt.Row("Fraud type:N", title=None)),
            width="content")

# --- Real data -----------------------------------------------------------------------------------
with tabs[4]:
    st.markdown("The same models and the same money-based evaluation on **real card data**: the "
                "[ULB credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), "
                "284,807 payments by European cardholders over two days in September 2013, 492 of them fraud. "
                "Its features are anonymised (PCA), so there are no readable reasons here.")
    if bench:
        d = bench["data"]
        st.caption(f"Split by time: {d['split']}. Test window: {d['test_rows']:,} payments, {d['test_frauds']} frauds.")
        rows = []
        for key, m in bench["models"].items():
            at = m["at_threshold"]
            rows.append({"Model": MODEL_NAMES[key], "PR-AUC": round(m["pr_auc"], 3), "ROC-AUC": round(m["roc_auc"], 3),
                         "Alerts": at["alerts"], "Precision": f"{at['precision']:.0%}",
                         "Fraud caught": f"{at['recall']:.0%}", "Money stopped": f"{at['stopped_pct']:.0f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pr = pd.concat([pd.DataFrame(m["pr_curve"]).assign(model=MODEL_NAMES[k]) for k, m in bench["models"].items()])
        st.altair_chart(alt.Chart(pr).mark_line(strokeWidth=2.5).encode(
            x=alt.X("recall:Q", title="Share of fraud caught"), y=alt.Y("precision:Q", title="Share of alerts that are fraud"),
            color=alt.Color("model:N", title=None)).properties(height=320), width="stretch")
        st.caption("Gradient boosting beats logistic regression by a similar margin on real and simulated data, "
                   "which suggests the simulation is not unrealistically easy for the simple model.")

# --- About ---------------------------------------------------------------------------------------
with tabs[5]:
    if metrics:
        d = metrics["data"]
        st.markdown(f"""
**The data is simulated**, because real card data with customer histories can't be shared.
{d['customers']:,} customers in Portugal, {d['payments']:,} payments from January to June 2026,
{d['fraud_payments']:,} of them fraud ({d['fraud_rate_pct']}%) in {d['fraud_cases']} cases.

Customers have a home city, a usual spending level and time of day, favourite shops and their own devices.
Normal life includes trips abroad, big planned purchases, new phones, shared family tablets and bursts of
small in-app payments, so unusual does not always mean fraud.

**Fraud types**: card testing (tiny online payments, then big ones), stolen card (in-store, often in another city),
account takeover (big online purchases from a new device, often at night) and subtle fraud (ordinary-looking
online purchases, half of them from the customer's own device).
""")
        st.markdown("**Features** (each uses only what was known at the time of the payment)")
        st.write(", ".join(f"`{f}`" for f in FEATURES))
    st.caption("Built by [Saeed Hojjatpanah](https://github.com/saeedm31) · "
               "[Source code](https://github.com/saeedm31/card-fraud-detection)")

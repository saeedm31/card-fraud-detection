# Model card: card fraud risk

| | |
|---|---|
| **Task** | Rank card payments by the chance they are fraud, so the alerts analysts review are the right ones |
| **Model** | scikit-learn `HistGradientBoostingClassifier` (400 trees), compared with logistic regression and 5 classic rules |
| **Output** | Risk score per payment, an alert above the chosen threshold, and plain-language reasons |
| **Data** | Simulated card payments, plus a benchmark on the real ULB dataset (see [data card](data_card.md)) |
| **Owner** | Saeed Hojjatpanah |

## Features (17, known at payment time only)

Amount and amount vs the customer's usual · hour and how unusual it is for the customer · online or in store ·
shop category · distance from home and travel speed since the last payment (impossible travel) · new shop · new
device · how many other customers use the device · payment history length · minutes since the previous payment ·
payments in the last hour and day, money in the last day, small payments in the last hour (card testing).

## The rules baseline

Amount over 5x usual · new device and over EUR 200 · 4+ payments in an hour · over 500 km from home and 2x usual ·
impossible travel (over 900 km/h). The score is the number of rules broken.

## Intended use
Choosing which payments a fraud analyst reviews, and why. An alert blocks the card until a person or the customer
confirms; the model does not close accounts or decide disputes.

## Not intended for
Credit decisions, judging customers, or blocking without a review path. Not validated on a real bank's live data.

## Evaluation
Time split: train January to April, choose the threshold on May (maximum net saving), report June. Metrics:
PR-AUC (fraud is under 1 in 200 payments, so accuracy is useless), alerts per day, share of alerts that are fraud,
fraud caught, fraud money stopped and net saving after EUR 5 per alert. Also recall per fraud type. The same
models are run on the real ULB data with a time split. Results are in the [README](../README.md#-results).

## Reasons
Each signal group (amount, time of day, location, new shop, new device, shared device, burst of payments) is reset to a normal value
and the payment is scored again. Groups that drop the risk by at least 5 points are shown as reasons. This explains
the model's behaviour, not the true cause, and signals that work together share the credit.

## Known limits
- Labels arrive instantly here; real chargebacks come days or weeks later, which delays retraining.
- Subtle fraud (ordinary-looking online purchases) is hard on purpose and caught much less often.
- Fraudsters adapt, so performance must be watched every week.

## Monitoring (production plan)
Weekly: alerts per day, share of alerts confirmed as fraud, fraud money stopped, net saving, and feature drift
(amounts, devices, countries). Retrain monthly on labelled chargebacks; re-pick the threshold when review costs change.

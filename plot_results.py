"""Draw the README charts from the saved results (no plotting library needed).

Usage:
    python train.py                       # writes reports/metrics.json
    python benchmark_ulb.py creditcard.csv  # writes reports/benchmark_ulb.json
    python plot_results.py                # writes docs/models.svg and docs/money.svg
"""

import json
import os

W = 900
NAMES = {"gradient_boosting": "Gradient boosting", "logistic": "Logistic regression", "rules": "Rules"}
ORDER = ["gradient_boosting", "logistic", "rules"]
STYLE = """
  .bg { fill: #fcfcfb; } .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { fill: #52514e; font: 12px system-ui, -apple-system, sans-serif; }
  .title { fill: #0b0b0b; font: 600 15px system-ui, -apple-system, sans-serif; }
  .sub { fill: #0b0b0b; font: 600 13px system-ui, -apple-system, sans-serif; }
  .s0 { stroke: #2a78d6; } .s1 { stroke: #eb6834; } .s2 { stroke: #1baf7a; }
  .f0 { fill: #2a78d6; } .f1 { fill: #eb6834; } .f2 { fill: #1baf7a; }
  .line { fill: none; stroke-width: 2.5; } .mark { stroke: #52514e; stroke-width: 1.5; stroke-dasharray: 4 4; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; } .grid { stroke: #34332f; } .axis { fill: #c3c2b7; } .title, .sub { fill: #f0efec; }
    .s0 { stroke: #3987e5; } .s1 { stroke: #d95926; } .s2 { stroke: #199e70; }
    .f0 { fill: #3987e5; } .f1 { fill: #d95926; } .f2 { fill: #199e70; } .mark { stroke: #c3c2b7; }
  }
"""


def svg(height, label, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
            f'role="img" aria-label="{label}">\n<style>{STYLE}</style>\n'
            f'<rect class="bg" width="{W}" height="{height}" rx="8"/>\n{body}\n</svg>\n')


def pr_panel(models, x0, x1, top, bottom, title):
    short = {"gradient_boosting": "boosting", "logistic": "logistic", "rules": "rules"}
    scores = " · ".join(f"{short[k]} {models[k]['pr_auc']:.2f}" for k in ORDER if k in models)

    def px(v):
        return x0 + v * (x1 - x0)

    def py(v):
        return bottom - v * (bottom - top)

    out = [f'<text class="sub" x="{x0}" y="{top - 30}">{title}</text>',
           f'<text class="axis" x="{x0}" y="{top - 12}">PR-AUC: {scores}</text>']
    for v in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{py(v):.0f}" y2="{py(v):.0f}"/>'
                   f'<text class="axis" x="{x0 - 6}" y="{py(v) + 4:.0f}" text-anchor="end">{v:.0%}</text>'
                   f'<text class="axis" x="{px(v):.0f}" y="{bottom + 18}" text-anchor="middle">{v:.0%}</text>')
    for i, key in enumerate(k for k in ORDER if k in models):
        c = models[key]["pr_curve"]
        pts = " ".join(f"{px(r):.0f},{py(p):.0f}" for r, p in zip(c["recall"], c["precision"]))
        out.append(f'<polyline class="line s{i}" points="{pts}"/>')
    out.append(f'<text class="axis" x="{(x0 + x1) / 2:.0f}" y="{bottom + 36}" text-anchor="middle">Share of fraud caught</text>')
    return out


def legend(models, x, y):
    out, dx = [], 0
    for i, key in enumerate(k for k in ORDER if k in models):
        out.append(f'<rect class="f{i}" x="{x + dx}" y="{y - 9}" width="14" height="10" rx="2"/>'
                   f'<text class="axis" x="{x + dx + 20}" y="{y}">{NAMES[key]}</text>')
        dx += 190
    return out


def models_chart(sim, real):
    H = 420
    body = ['<text class="title" x="58" y="26">How well each model separates fraud from normal payments</text>']
    body += legend(sim["models"], 58, 52)
    body += pr_panel(sim["models"], 58, 420, 116, 350,
                     f"Simulated data, June 2026 ({sim['split']['test_frauds']} frauds)")
    if real:
        body += pr_panel(real["models"], 510, 872, 116, 350,
                         f"Real data: ULB credit cards ({real['data']['test_frauds']} frauds)")
    body.append(f'<text class="axis" x="22" y="227" text-anchor="middle" transform="rotate(-90 22 227)">'
                f'Share of alerts that are fraud</text>')
    return svg(H, "Precision-recall curves for each model on simulated and real card data", "\n".join(body))


def money_chart(sim):
    import math
    H = 380
    gb = sim["models"]["gradient_boosting"]
    curve = gb["threshold_curve"]
    chosen = gb["at_threshold"]
    x0, x1, top, bottom = 80, 870, 80, 320
    lo, hi = math.log10(curve[0]["alerts_per_day"]), math.log10(curve[-1]["alerts_per_day"])
    ymax = max(p["net_saving_eur"] for p in curve)
    ymax = math.ceil(ymax / 10000) * 10000

    def px(v):
        return x0 + (math.log10(v) - lo) * (x1 - x0) / (hi - lo)

    def py(v):
        return bottom - v * (bottom - top) / ymax

    body = ['<text class="title" x="80" y="26">Money saved in June vs how many alerts the team reviews</text>',
            f'<text class="axis" x="80" y="48">Gradient boosting. Each alert costs €{sim["costs"]["review_eur_per_alert"]:.0f} '
            f'of analyst time; each alert blocks the card and stops the rest of that fraud.</text>']
    for v in range(0, int(ymax) + 1, 10000):
        body.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{py(v):.0f}" y2="{py(v):.0f}"/>'
                    f'<text class="axis" x="{x0 - 8}" y="{py(v) + 4:.0f}" text-anchor="end">€{v // 1000}k</text>')
    for v in (1, 2, 5, 10, 20, 50, 100, 200):
        if curve[0]["alerts_per_day"] <= v <= curve[-1]["alerts_per_day"] * 1.001:
            body.append(f'<text class="axis" x="{px(v):.0f}" y="{bottom + 18}" text-anchor="middle">{v}</text>')
    pts = " ".join(f"{px(p['alerts_per_day']):.0f},{py(p['net_saving_eur']):.0f}" for p in curve)
    body.append(f'<polyline class="line s0" points="{pts}"/>')
    cx, cy = px(chosen["alerts_per_day"]), py(chosen["net_saving_eur"])
    body.append(f'<line class="mark" x1="{cx:.0f}" x2="{cx:.0f}" y1="{top}" y2="{bottom}"/>'
                f'<circle class="f0" cx="{cx:.0f}" cy="{cy:.0f}" r="5"/>'
                f'<text class="axis" x="{cx + 10:.0f}" y="{py(ymax * 0.22):.0f}">Chosen on May data: {chosen["alerts_per_day"]:.0f} alerts/day, '
                f'{chosen["precision"]:.0%} of them fraud,</text>'
                f'<text class="axis" x="{cx + 10:.0f}" y="{py(ymax * 0.22) + 16:.0f}">{chosen["stopped_pct"]:.0f}% of fraud money stopped, '
                f'net €{chosen["net_saving_eur"]:,.0f}</text>')
    body.append(f'<text class="axis" x="{(x0 + x1) / 2:.0f}" y="{bottom + 40}" text-anchor="middle">'
                f'Alerts per day (log scale)</text>')
    body.append(f'<text class="axis" x="24" y="{(top + bottom) / 2:.0f}" text-anchor="middle" '
                f'transform="rotate(-90 24 {(top + bottom) / 2:.0f})">Net saving (fraud stopped − review cost)</text>')
    return svg(H, "Net money saved in June against the number of alerts per day", "\n".join(body))


def main():
    with open("reports/metrics.json") as f:
        sim = json.load(f)
    real = None
    if os.path.exists("reports/benchmark_ulb.json"):
        with open("reports/benchmark_ulb.json") as f:
            real = json.load(f)
    os.makedirs("docs", exist_ok=True)
    for name, text in (("docs/models.svg", models_chart(sim, real)), ("docs/money.svg", money_chart(sim))):
        with open(name, "w") as f:
            f.write(text)
        print("Wrote", name)


if __name__ == "__main__":
    main()

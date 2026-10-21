"""Compare a deterministic plan with a stochastic (resilient) plan and a fairness-constrained plan.

Usage:  python src/run.py            (from the project root)
Outputs go to results/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import HOSPITALS, WAREHOUSES, build_instance, expected_value_scenario, sample_scenarios
from model import solve_network

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

N_TRAIN, N_TEST = 60, 200
MIN_SERVICE = 0.95


def evaluate(inst, plan, scenarios):
    """Fix a plan's first-stage decisions and measure how it performs on the given scenarios."""
    ev = solve_network(inst, scenarios, fixed_open=plan.open, fixed_stock=plan.stock)
    demand = np.array([sc.demand for sc in scenarios]).T          # (H, K)
    service = 1 - ev.shortage / np.maximum(demand, 1)              # (H, K)
    return {
        "expected_total_cost": ev.objective,
        "fixed_cost": ev.fixed_cost,
        "holding_cost": ev.holding_cost,
        "expected_transport_cost": ev.transport_cost,
        "expected_shortage_cost": ev.shortage_cost,
        "avg_service_level": service.mean(),
        "worst_hospital_avg_service": service.mean(axis=1).min(),
        "worst_case_service": service.min(),
        "share_scenarios_with_shortage": float((ev.shortage.sum(axis=0) > 0.5).mean()),
    }


def worst_case_cost(inst, plan, scenarios):
    """Highest weekly cost of the plan over all test scenarios."""
    costs = [solve_network(inst, [type(sc)(1.0, sc.demand, sc.availability)],
                           fixed_open=plan.open, fixed_stock=plan.stock).objective for sc in scenarios]
    return max(costs)


def describe(inst, plan):
    return ", ".join(f"{inst.warehouses[w]} ({plan.stock[w]:.0f})"
                     for w in range(inst.W) if plan.open[w] > 0.5)


def main():
    inst = build_instance()
    train = sample_scenarios(inst, N_TRAIN, seed=1)
    test = sample_scenarios(inst, N_TEST, seed=2)

    print("Solving deterministic (expected-value) model ...")
    det = solve_network(inst, expected_value_scenario(inst))
    print("Solving two-stage stochastic model ...")
    sto = solve_network(inst, train)
    print(f"Solving stochastic model with fairness constraint (min service {MIN_SERVICE:.0%}) ...")
    fair = solve_network(inst, train, min_service=MIN_SERVICE)
    print("Solving min-max robust model ...")
    rob = solve_network(inst, train, robust=True)

    # Classic stochastic-programming metrics on the training scenarios
    rp = sto.objective
    eev = solve_network(inst, train, fixed_open=det.open, fixed_stock=det.stock).objective
    ws = sum(sc.prob * solve_network(inst, [type(sc)(1.0, sc.demand, sc.availability)]).objective
             for sc in train)
    vss, evpi = eev - rp, rp - ws

    plans = {"Deterministic": det, "Stochastic": sto, f"Stochastic + fairness ({MIN_SERVICE:.0%})": fair,
             "Robust (min-max)": rob}
    rows = []
    for name, plan in plans.items():
        r = {"plan": name, "open_warehouses (stock)": describe(inst, plan)}
        r.update(evaluate(inst, plan, test))
        r["worst_scenario_total_cost"] = worst_case_cost(inst, plan, test)
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary_out_of_sample.csv", index=False)

    metrics = pd.DataFrame([
        {"metric": "RP  (stochastic solution, expected cost)", "value_eur": rp},
        {"metric": "EEV (deterministic plan under uncertainty)", "value_eur": eev},
        {"metric": "WS  (wait-and-see, perfect information)", "value_eur": ws},
        {"metric": "VSS = EEV - RP (value of stochastic solution)", "value_eur": vss},
        {"metric": "EVPI = RP - WS (value of perfect information)", "value_eur": evpi},
    ])
    metrics.to_csv(OUT / "stochastic_metrics.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 80)
    print("\n=== Stochastic programming metrics (training scenarios, EUR/week) ===")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print(f"\n=== Out-of-sample evaluation on {N_TEST} unseen scenarios ===")
    print(summary.T.to_string(header=False))

    plot_network(inst, sto, "Stochastic plan: opened warehouses and expected flows", OUT / "network_stochastic.png")
    plot_network(inst, det, "Deterministic plan: opened warehouses and flows", OUT / "network_deterministic.png")
    plot_comparison(summary)
    print(f"\nResults written to {OUT}")


def plot_network(inst, plan, title, path):
    fig, ax = plt.subplots(figsize=(7, 8))
    flows = plan.flows.mean(axis=2)
    wl = [(w[2], w[1]) for w in WAREHOUSES]
    hl = [(h[2], h[1]) for h in HOSPITALS]
    vmax = flows.max() if flows.max() > 0 else 1
    for w in range(inst.W):
        for h in range(inst.H):
            if flows[w, h] > 1:
                ax.plot([wl[w][0], hl[h][0]], [wl[w][1], hl[h][1]], color="#4a7ab5",
                        lw=0.5 + 3 * flows[w, h] / vmax, alpha=0.6, zorder=1)
    for h, (lon, lat) in enumerate(hl):
        ax.scatter(lon, lat, s=inst.base_demand[h] / 2, color="#d9534f", zorder=2)
        ax.annotate(inst.hospitals[h], (lon, lat), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for w, (lon, lat) in enumerate(wl):
        is_open = plan.open[w] > 0.5
        ax.scatter(lon, lat, marker="s", s=140, zorder=3,
                   color="#2e7d32" if is_open else "white", edgecolor="#2e7d32", lw=1.5)
        label = f"{inst.warehouses[w]}\n{plan.stock[w]:.0f} packs" if is_open else inst.warehouses[w]
        ax.annotate(label, (lon, lat), fontsize=7, xytext=(5, -12), textcoords="offset points",
                    color="#2e7d32", weight="bold" if is_open else "normal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(summary):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    names = ["Deterministic", "Stochastic", "Stoch. + fairness", "Robust"]
    parts = ["fixed_cost", "holding_cost", "expected_transport_cost", "expected_shortage_cost"]
    labels = ["Fixed", "Holding", "Transport", "Shortage penalty"]
    bottom = np.zeros(len(summary))
    for p, lab in zip(parts, labels):
        a1.bar(names, summary[p], bottom=bottom, label=lab)
        bottom += summary[p].values
    a1.set_ylabel("EUR per week (out-of-sample)")
    a1.set_title("Expected cost breakdown")
    a1.legend(fontsize=8)
    x = np.arange(len(summary))
    a2.bar(x - 0.2, summary["avg_service_level"] * 100, 0.4, label="Average service level")
    a2.bar(x + 0.2, summary["worst_hospital_avg_service"] * 100, 0.4, label="Worst-served hospital (average)")
    a2.set_xticks(x, names)
    a2.set_ylabel("% of demand served")
    a2.set_ylim(75, 101)
    a2.set_title("Service level")
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "plan_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

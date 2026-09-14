"""Aggregate results.csv, make the plots, and compute the headline statistics."""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ORDER = ["occlusion", "accidental_success", "flailing", "truncation",
         "inconsistent_strategy"]
LABEL = {"occlusion": "Occlusion (stale obs)",
         "accidental_success": "Accidental success",
         "flailing": "Corrective flailing",
         "truncation": "Truncated episodes",
         "inconsistent_strategy": "Inconsistent strategy"}
COLOR = dict(zip(ORDER, ["#D55E00", "#CC79A7", "#009E73", "#E69F00", "#0072B2"]))
MARKER = dict(zip(ORDER, ["o", "s", "^", "D", "P"]))


RESULTS = "results/results.csv"
TREE_RESULTS = "results/results_tree.csv"
CONTROL_RESULTS = "results/control_results.csv"
T95_10 = 2.262157  # two-sided 95% Student-t critical value, df=9


def load(path=RESULTS):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("rho", "open_loop_mse", "closed_loop_success",
                      "final_dist", "wrong_side_rate"):
                r[k] = float(r[k])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return [r for r in rows if r["corruption"] != "clean"]


def load_controls(path=CONTROL_RESULTS):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key in ("rho", "open_loop_mse", "closed_loop_success",
                        "final_dist"):
                row[key] = float(row[key])
            row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def summarize(values):
    """Return mean, sample SD, and 95% t-interval half-width."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    sd = float(values.std(ddof=1))
    critical = T95_10 if len(values) == 10 else 1.96
    return mean, sd, critical * sd / np.sqrt(len(values))


def agg(rows):
    """(corruption, rho) -> mean/sample-SD/95%-CI half-width."""
    d = defaultdict(list)
    for r in rows:
        d[(r["corruption"], r["rho"])].append(r)
    out = {}
    for k, v in d.items():
        out[k] = {m: summarize([x[m] for x in v])
                  for m in ("open_loop_mse", "closed_loop_success",
                            "wrong_side_rate")}
    return out


def agg_controls(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["rho"])].append(row)
    return {
        key: {
            metric: summarize([row[metric] for row in group])
            for metric in ("open_loop_mse", "closed_loop_success")
        }
        for key, group in grouped.items()
    }


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values):
    """Average ranks for ties, equivalent to scipy.stats.rankdata."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def paired_change(rows, corruption, rho=1.0):
    clean = {row["seed"]: row["closed_loop_success"] for row in rows
             if row["corruption"] == corruption and row["rho"] == 0}
    treated = {row["seed"]: row["closed_loop_success"] for row in rows
               if row["corruption"] == corruption and row["rho"] == rho}
    seeds = sorted(set(clean) & set(treated))
    return summarize([treated[seed] - clean[seed] for seed in seeds])


def main():
    rows = load()
    A = agg(rows)
    rhos = sorted({r["rho"] for r in rows})

    # ---- Figure 1: dose-response ------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for c in ORDER:
        m = [A[(c, r)]["closed_loop_success"][0] for r in rhos]
        s = [A[(c, r)]["closed_loop_success"][2] for r in rhos]
        ax[0].errorbar(rhos, m, yerr=s, marker=MARKER[c], capsize=3, lw=2,
                       color=COLOR[c], label=LABEL[c])
        m = [A[(c, r)]["open_loop_mse"][0] for r in rhos]
        s = [A[(c, r)]["open_loop_mse"][2] for r in rhos]
        ax[1].errorbar(rhos, m, yerr=s, marker=MARKER[c], capsize=3, lw=2,
                       color=COLOR[c], label=LABEL[c])
    ax[0].set_xlabel("Contamination rate, ρ")
    ax[0].set_ylabel("Closed-loop success rate")
    ax[0].set_title("Closed-loop task performance")
    ax[0].set_ylim(-0.03, 1.05)
    ax[1].set_xlabel("Contamination rate, ρ")
    ax[1].set_ylabel("Open-loop action MSE")
    ax[1].set_title("Open-loop prediction error")
    for a in ax:
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig("results/fig1_dose_response.png", dpi=160)

    # ---- Figure 2: does open-loop MSE predict closed-loop success? --------
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for c in ORDER:
        x = [r["open_loop_mse"] for r in rows
             if r["corruption"] == c and r["rho"] > 0]
        y = [r["closed_loop_success"] for r in rows
             if r["corruption"] == c and r["rho"] > 0]
        ax.scatter(x, y, s=42, marker=MARKER[c], color=COLOR[c], alpha=0.85,
                   label=LABEL[c], edgecolor="white", lw=0.7)
    clean = [r for r in rows if r["corruption"] == ORDER[0] and r["rho"] == 0]
    ax.scatter([r["open_loop_mse"] for r in clean],
               [r["closed_loop_success"] for r in clean], s=48, marker="X",
               color="#555555", label="Clean baseline", edgecolor="white", lw=0.7)
    ax.set_xlabel("Open-loop action MSE")
    ax.set_ylabel("Closed-loop success rate")
    ax.set_title("Open-loop error does not uniquely determine task success")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/fig2_openloop_vs_closedloop.png", dpi=160)

    # ---- Figure 3: policy-family and transition-matching controls ---------
    if os.path.exists(TREE_RESULTS) and os.path.exists(CONTROL_RESULTS):
        tree_rows = load(TREE_RESULTS)
        tree_agg = agg(tree_rows)
        control_agg = agg_controls(load_controls())
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for policy_name, policy_label, color in (
            ("mlp", "MLP", "#0072B2"),
            ("trees", "Extra Trees", "#D55E00"),
        ):
            episode_agg = A if policy_name == "mlp" else tree_agg
            means = [episode_agg[("inconsistent_strategy", rho)]
                     ["closed_loop_success"][0] for rho in rhos]
            errors = [episode_agg[("inconsistent_strategy", rho)]
                      ["closed_loop_success"][2] for rho in rhos]
            ax.errorbar(rhos, means, yerr=errors, color=color, marker="o",
                        capsize=3, lw=2, label=f"{policy_label}, episode-matched")
            means = [control_agg[(policy_name, rho)]["closed_loop_success"][0]
                     for rho in rhos]
            errors = [control_agg[(policy_name, rho)]["closed_loop_success"][2]
                      for rho in rhos]
            ax.errorbar(rhos, means, yerr=errors, color=color, marker="s",
                        capsize=3, lw=2, ls="--",
                        label=f"{policy_label}, transition-matched")
        ax.set_xlabel("Alternate-strategy fraction, ρ")
        ax.set_ylabel("Closed-loop success rate")
        ax.set_title("Strategy control across policy families and matching units")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig("results/fig3_strategy_controls.png", dpi=160)

    # ---- statistics -------------------------------------------------------
    lines = []
    lines.append("## Per-mode correlation between open-loop MSE and closed-loop success\n")
    lines.append("| corruption mode | Pearson r | Spearman rho | success @ ρ=0.5 | success @ ρ=1.0 |")
    lines.append("|---|---|---|---|---|")
    for c in ORDER:
        x = [r["open_loop_mse"] for r in rows if r["corruption"] == c]
        y = [r["closed_loop_success"] for r in rows if r["corruption"] == c]
        lines.append(f"| {LABEL[c]} | {pearson(x, y):+.2f} | {spearman(x, y):+.2f} | "
                     f"{A[(c, 0.5)]['closed_loop_success'][0]:.3f} | "
                     f"{A[(c, 1.0)]['closed_loop_success'][0]:.3f} |")
    unique_rows = ([r for r in rows if r["rho"] > 0] +
                   [r for r in rows
                    if r["corruption"] == ORDER[0] and r["rho"] == 0])
    allx = [r["open_loop_mse"] for r in unique_rows]
    ally = [r["closed_loop_success"] for r in unique_rows]
    lines.append(f"\nPooled across all modes: Pearson r = {pearson(allx, ally):+.2f}, "
                 f"Spearman rho = {spearman(allx, ally):+.2f}\n")

    # Descriptive similar-MSE band.
    lo, hi = 0.20, 0.30
    band = [r for r in unique_rows if lo <= r["open_loop_mse"] <= hi]
    if len(band) >= 2:
        ys = [r["closed_loop_success"] for r in band]
        lines.append(f"### Matched open-loop score, unmatched reality\n")
        lines.append(f"Among the {len(band)} runs whose open-loop MSE falls in "
                     f"[{lo}, {hi}] — a descriptive similar-score band — "
                     f"closed-loop success ranges from **{min(ys):.3f} to {max(ys):.3f}** "
                     f"(spread of {max(ys)-min(ys):.3f}).\n")
        for r in sorted(band, key=lambda r: r["closed_loop_success"]):
            lines.append(f"- {LABEL[r['corruption']]}, ρ={r['rho']}: "
                         f"MSE {r['open_loop_mse']:.3f} → success {r['closed_loop_success']:.3f}")

    lines.append("\n## Full aggregated table (mean and 95% t interval over 10 seeds)\n")
    lines.append("| corruption | ρ | open-loop MSE | closed-loop success | wrong-side push rate |")
    lines.append("|---|---|---|---|---|")
    for c in ORDER:
        for r in rhos:
            a = A[(c, r)]
            lines.append(f"| {LABEL[c]} | {r} | {a['open_loop_mse'][0]:.4f} "
                         f"| {a['closed_loop_success'][0]:.3f} ± {a['closed_loop_success'][2]:.3f} "
                         f"| {a['wrong_side_rate'][0]:.3f} |")

    lines.append("\n## Paired full-contamination effects\n")
    lines.append("| corruption | success at ρ=1 (95% CI) | paired change from clean (95% CI) |")
    lines.append("|---|---|---|")
    for c in ORDER:
        mean, _, ci = A[(c, 1.0)]["closed_loop_success"]
        change, _, change_ci = paired_change(rows, c)
        lines.append(
            f"| {LABEL[c]} | {mean:.3f} ± {ci:.3f} | "
            f"{change:+.3f} ± {change_ci:.3f} |"
        )

    if os.path.exists(TREE_RESULTS) and os.path.exists(CONTROL_RESULTS):
        tree_rows = load(TREE_RESULTS)
        tree_agg = agg(tree_rows)
        lines.append("\n## Policy-family robustness at full contamination\n")
        lines.append("| corruption | MLP success | Extra Trees success |")
        lines.append("|---|---|---|")
        for c in ORDER:
            mlp = A[(c, 1.0)]["closed_loop_success"]
            trees = tree_agg[(c, 1.0)]["closed_loop_success"]
            lines.append(
                f"| {LABEL[c]} | {mlp[0]:.3f} ± {mlp[2]:.3f} | "
                f"{trees[0]:.3f} ± {trees[2]:.3f} |"
            )

        controls = agg_controls(load_controls())
        lines.append("\n## Transition-matched inconsistent-strategy control\n")
        lines.append("| ρ | MLP success | Extra Trees success |")
        lines.append("|---|---|---|")
        for rho in rhos:
            mlp = controls[("mlp", rho)]["closed_loop_success"]
            trees = controls[("trees", rho)]["closed_loop_success"]
            lines.append(
                f"| {rho} | {mlp[0]:.3f} ± {mlp[2]:.3f} | "
                f"{trees[0]:.3f} ± {trees[2]:.3f} |"
            )

    lines.append(clonability_control())
    open("results/findings.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:24]))
    print("\nwrote results/findings.md and analysis figures")




def clonability_control(path=RESULTS):
    """Inconsistent-strategy needs a control the other modes don't.

    At rho=1.0 the dataset is 100% alternate-strategy and therefore fully
    self-consistent -- there is no mixing left to blame. Whatever success rate
    a policy reaches there is the CLONABILITY FLOOR of that strategy, not
    evidence of inconsistency harm. Intermediate rho must be read against the
    straight line joining the rho=0 and rho=1.0 endpoints: only degradation
    BELOW that line is attributable to mixing two strategies rather than to
    one of them simply being harder to clone.
    """
    rows = load(path)
    A = agg(rows)
    rhos = sorted({r["rho"] for r in rows})
    out = ["\n## Clonability control for inconsistent strategy\n",
           "| ρ | observed success | interpolated (clonability-only) | residual vs. interpolation |",
           "|---|---|---|---|"]
    lo = A[("inconsistent_strategy", 0.0)]["closed_loop_success"][0]
    hi = A[("inconsistent_strategy", 1.0)]["closed_loop_success"][0]
    for r in rhos:
        obs = A[("inconsistent_strategy", r)]["closed_loop_success"][0]
        interp = lo + (hi - lo) * r
        out.append(f"| {r} | {obs:.3f} | {interp:.3f} | {obs - interp:+.3f} |")
    out.append(f"\nBoth experts solve the task on 100% of episodes, so the "
               f"rho=1.0 result ({hi:.3f}) is purely a statement about how hard "
               f"the alternate strategy is to clone, not about inconsistency.\n")
    return "\n".join(out)


if __name__ == "__main__":
    main()

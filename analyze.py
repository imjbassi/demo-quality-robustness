"""Aggregate results.csv, make the plots, and compute the headline statistics."""

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["occlusion", "accidental_success", "flailing", "truncation",
         "inconsistent_strategy"]
LABEL = {"occlusion": "Occlusion (stale obs)",
         "accidental_success": "Accidental success",
         "flailing": "Corrective flailing",
         "truncation": "Truncated episodes",
         "inconsistent_strategy": "Inconsistent strategy"}
COLOR = dict(zip(ORDER, ["#c0392b", "#8e44ad", "#16a085", "#d68910", "#2471a3"]))


RESULTS = "results/results.csv"


def load(path=RESULTS):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("rho", "open_loop_mse", "closed_loop_success",
                      "final_dist", "wrong_side_rate"):
                r[k] = float(r[k])
            rows.append(r)
    return [r for r in rows if r["corruption"] != "clean"]


def agg(rows):
    """(corruption, rho) -> mean/std over seeds."""
    d = defaultdict(list)
    for r in rows:
        d[(r["corruption"], r["rho"])].append(r)
    out = {}
    for k, v in d.items():
        out[k] = {m: (float(np.mean([x[m] for x in v])),
                      float(np.std([x[m] for x in v])))
                  for m in ("open_loop_mse", "closed_loop_success",
                            "wrong_side_rate")}
    return out


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    r = lambda v: np.argsort(np.argsort(v))
    return pearson(r(x), r(y))


def main():
    rows = load()
    A = agg(rows)
    rhos = sorted({r["rho"] for r in rows})

    # ---- Figure 1: dose-response ------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for c in ORDER:
        m = [A[(c, r)]["closed_loop_success"][0] for r in rhos]
        s = [A[(c, r)]["closed_loop_success"][1] for r in rhos]
        ax[0].errorbar(rhos, m, yerr=s, marker="o", capsize=3, lw=2,
                       color=COLOR[c], label=LABEL[c])
        m = [A[(c, r)]["open_loop_mse"][0] for r in rhos]
        s = [A[(c, r)]["open_loop_mse"][1] for r in rhos]
        ax[1].errorbar(rhos, m, yerr=s, marker="o", capsize=3, lw=2,
                       color=COLOR[c], label=LABEL[c])
    ax[0].set_xlabel("contamination rate ρ (fraction of dataset)")
    ax[0].set_ylabel("closed-loop success rate")
    ax[0].set_title("Closed-loop: what actually happens")
    ax[0].set_ylim(-0.03, 1.05)
    ax[1].set_xlabel("contamination rate ρ (fraction of dataset)")
    ax[1].set_ylabel("open-loop action MSE (vs expert, held-out clean states)")
    ax[1].set_title("Open-loop: what the cheap metric reports")
    for a in ax:
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Not all bad demonstrations are equally bad", fontsize=13)
    fig.tight_layout()
    fig.savefig("results/fig1_dose_response.png", dpi=160)

    # ---- Figure 2: does open-loop MSE predict closed-loop success? --------
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for c in ORDER:
        x = [r["open_loop_mse"] for r in rows if r["corruption"] == c]
        y = [r["closed_loop_success"] for r in rows if r["corruption"] == c]
        ax.scatter(x, y, s=42, color=COLOR[c], alpha=0.85, label=LABEL[c],
                   edgecolor="white", lw=0.7)
    ax.set_xlabel("open-loop action MSE  (cheap, offline)")
    ax.set_ylabel("closed-loop success rate  (expensive, on-robot)")
    ax.set_title("Same open-loop score, wildly different real performance")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/fig2_openloop_vs_closedloop.png", dpi=160)

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
    allx = [r["open_loop_mse"] for r in rows]
    ally = [r["closed_loop_success"] for r in rows]
    lines.append(f"\nPooled across all modes: Pearson r = {pearson(allx, ally):+.2f}, "
                 f"Spearman rho = {spearman(allx, ally):+.2f}\n")

    # matched-MSE band: the money statistic
    lo, hi = 0.20, 0.30
    band = [r for r in rows if lo <= r["open_loop_mse"] <= hi]
    if len(band) >= 2:
        ys = [r["closed_loop_success"] for r in band]
        lines.append(f"### Matched open-loop score, unmatched reality\n")
        lines.append(f"Among the {len(band)} runs whose open-loop MSE falls in "
                     f"[{lo}, {hi}] — i.e. indistinguishable on the offline metric — "
                     f"closed-loop success ranges from **{min(ys):.3f} to {max(ys):.3f}** "
                     f"(spread of {max(ys)-min(ys):.3f}).\n")
        for r in sorted(band, key=lambda r: r["closed_loop_success"]):
            lines.append(f"- {LABEL[r['corruption']]}, ρ={r['rho']}: "
                         f"MSE {r['open_loop_mse']:.3f} → success {r['closed_loop_success']:.3f}")

    lines.append("\n## Full aggregated table (mean over 3 seeds)\n")
    lines.append("| corruption | ρ | open-loop MSE | closed-loop success | wrong-side push rate |")
    lines.append("|---|---|---|---|---|")
    for c in ORDER:
        for r in rhos:
            a = A[(c, r)]
            lines.append(f"| {LABEL[c]} | {r} | {a['open_loop_mse'][0]:.4f} "
                         f"| {a['closed_loop_success'][0]:.3f} ± {a['closed_loop_success'][1]:.3f} "
                         f"| {a['wrong_side_rate'][0]:.3f} |")

    lines.append(clonability_control())
    open("results/findings.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:24]))
    print("\nwrote results/findings.md and both figures")




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
           "| ρ | observed success | interpolated (clonability-only) | excess harm from mixing |",
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

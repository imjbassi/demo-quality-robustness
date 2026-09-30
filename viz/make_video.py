"""Render real closed-loop rollouts of the checkpointed video policies into
the results video (1080p MP4) and the README GIF.

Everything shown is measured, nothing is staged:

- The five panels of each shot share identical initial states because each
  policy's environment is constructed with the same fixed seed
  (``common.INIT_SEED``); the seed was fixed a priori, not chosen by
  scanning rollouts for dramatic failures.
- Panel labels are the policy's own closed-loop success rate over 200
  evaluation episodes, read from ``viz/checkpoints/manifest.json``, which
  ``train_video_policies.py`` only writes after the retrained policies
  reproduce the committed seed-0 results.
- The wrong-side highlight is the repo's own predicate from
  ``policy.eval_wrong_side_rate`` -- block farther from the goal than it
  started (by > 0.02) and the episode not succeeded -- evaluated on the
  current state of the rollout; the end-of-episode tag is exactly the
  condition that counts toward the reported wrong-side rate.
- The closing card plots ``results/results.csv`` (10-seed means), the same
  data behind the paper's Figure 1.

Usage:  python viz/make_video.py            (checkpoints must exist already)
Output: viz/media/demo_quality_video.mp4, viz/media/main_shot.gif
"""

import os
import subprocess

import matplotlib.pyplot as plt

from common import (BLUE, FPS, GRID, INIT_SEED, INK, MEDIA_DIR, MUTED,
                    ORANGE, SECONDARY, SURFACE, WRONG, Panel, SegmentWriter,
                    load_manifest, load_policy, make_gif, paper_means,
                    record_rollouts, render_grid_shot)

N_STATES = 5            # initial states played in sequence
CARD_SECONDS = 6
RHOS = [0.0, 0.5, 0.75, 0.9, 1.0]


def build_info_axis(ax, title_lines, mode_key, means, series_color):
    """The sixth grid cell: shot title, provenance, and a mini dose-response
    curve with the five panel rhos marked."""
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    y = 0.97
    for i, line in enumerate(title_lines[:3]):
        ax.text(0.03, y, line, transform=ax.transAxes, va="top",
                color=INK if i == 0 else SECONDARY,
                fontsize=16 if i == 0 else 12.5,
                fontweight="bold" if i == 0 else "normal", wrap=True)
        y -= 0.085 if i == 0 else 0.062
    ax.text(0.03, 0.285,
            f"every panel: the same {N_STATES} initial states\n"
            f"(env seed {INIT_SEED}, fixed a priori) — only the\n"
            "training data differs\n"
            "labels: measured over 200 eval episodes",
            transform=ax.transAxes, va="top", color=MUTED, fontsize=11)

    # mini curve: 10-seed mean success vs rho for this mode
    inset = ax.inset_axes([0.10, 0.42, 0.84, 0.30])
    inset.set_facecolor(SURFACE)
    rhos_all = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    ys = [means[(mode_key if r > 0 else "clean", r)][0] for r in rhos_all]
    inset.plot(rhos_all, ys, color=series_color, lw=2, marker="o", ms=5,
               mec="white", mew=0.8)
    for r in RHOS:
        m = means[(mode_key if r > 0 else "clean", r)][0]
        inset.plot([r], [m], "o", ms=8, color=series_color, mec="white")
    inset.set_ylim(-0.06, 1.06)
    inset.set_xlim(-0.05, 1.05)
    inset.tick_params(colors=MUTED, labelsize=9, length=2)
    for s in inset.spines.values():
        s.set_color(GRID)
    inset.set_xlabel("contamination ρ", color=MUTED, fontsize=10,
                     labelpad=1)
    inset.set_ylabel("success\n(10-seed mean)", color=MUTED, fontsize=9)
    inset.grid(color=GRID, lw=0.6, alpha=0.7)
    return ax


def make_shot_figure(mode_key, entries_by, means, series_color, title_lines):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 3, left=0.025, right=0.985, top=0.93,
                          bottom=0.045, wspace=0.14, hspace=0.30)
    cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    panels = []
    for (r, c), rho in zip(cells, RHOS):
        key = "clean" if rho == 0 else mode_key
        e = entries_by[(key, rho)]
        ax = fig.add_subplot(gs[r, c])
        label = "clean data" if rho == 0 else f"ρ = {rho}"
        panels.append(Panel(ax, label,
                            f"success rate {e['closed_loop_success']:.2f}"))
    info_ax = fig.add_subplot(gs[1, 2])
    build_info_axis(info_ax, title_lines, mode_key, means, series_color)
    counter = info_ax.text(0.03, 0.015, "", transform=info_ax.transAxes,
                           color=MUTED, fontsize=11, va="bottom")
    return fig, panels, counter


def make_closing_card_figure(means):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.10, 0.13, 0.80, 0.66])
    ax.set_facecolor(SURFACE)
    rhos_all = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

    for mode, color in [("flailing", ORANGE), ("accidental_success", BLUE)]:
        ys = [means[(mode if r > 0 else "clean", r)][0] for r in rhos_all]
        ax.plot(rhos_all, ys, color=color, lw=2.5, marker="o", ms=9,
                mec="white", mew=1.2, zorder=5)
    fl = [means[("flailing" if r > 0 else "clean", r)][0] for r in rhos_all]
    acc = [means[("accidental_success" if r > 0 else "clean", r)][0]
           for r in rhos_all]
    ax.annotate("corrective flailing", (0.5, fl[2]), xytext=(0, 14),
                textcoords="offset points", color=ORANGE, fontsize=16,
                fontweight="bold", ha="center")
    ax.annotate("accidental success", (0.9, acc[4]), xytext=(-90, 30),
                textcoords="offset points", color=BLUE, fontsize=16,
                fontweight="bold", ha="center")
    ax.annotate(
        f"the cliff: {acc[-2]:.3f} at ρ = 0.9 → "
        f"{acc[-1]:.3f} at ρ = 1.0",
        xy=(0.997, acc[-1] + 0.02), xytext=(0.42, 0.34),
        textcoords="axes fraction", color=WRONG, fontsize=17,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=WRONG, lw=2))
    for r, y in zip(rhos_all, acc):
        ax.annotate(f"{y:.2f}", (r, y), xytext=(0, -24),
                    textcoords="offset points", ha="center",
                    color=SECONDARY, fontsize=12)

    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.08, 1.10)
    ax.set_xticks(rhos_all)
    ax.set_xlabel("contamination rate ρ", color=SECONDARY, fontsize=15)
    ax.set_ylabel("closed-loop success (mean of 10 seeds)",
                  color=SECONDARY, fontsize=15)
    ax.tick_params(colors=MUTED, labelsize=12)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, lw=0.7, alpha=0.7)
    fig.text(0.10, 0.905, "Not all bad demonstrations are equally bad",
             color=INK, fontsize=25, fontweight="bold")
    fig.text(0.10, 0.862,
             "Accidental-success contamination is a cliff, not a slope; "
             "corrective flailing is nearly free.",
             color=SECONDARY, fontsize=15)
    fig.text(0.10, 0.055,
             "Data: results/results.csv — MLP behaviour cloning, "
             "10 seeds × 200 evaluation episodes per point.",
             color=MUTED, fontsize=12)
    return fig


def main():
    os.makedirs(MEDIA_DIR, exist_ok=True)
    manifest = load_manifest()
    entries = {(e["corruption"], e["rho"]): e for e in manifest["policies"]
               if e.get("train_seed", 0) == 0}
    means = paper_means()

    print("recording rollouts (init seed", INIT_SEED, ")")
    rollouts = {}
    for (key, rho), e in entries.items():
        pol = load_policy(e)
        rollouts[(key, rho)] = record_rollouts(pol, N_STATES, INIT_SEED)
        r = rollouts[(key, rho)]
        print(f"  {key:22s} rho={rho:<4g} "
              f"succ {int(r['succ'][:, -1].sum())}/{N_STATES} "
              f"steps {r['steps'].tolist()}")

    def rolls_for(mode):
        return [rollouts[("clean" if r == 0 else mode, r)] for r in RHOS]

    sw = SegmentWriter(MEDIA_DIR)

    # main shot: accidental success
    fig, panels, counter = make_shot_figure(
        "accidental_success", entries, means, BLUE,
        ["Accidental success",
         "demos that reached the goal by luck,",
         "not by correct behaviour"])
    main_seg = sw.render(
        "main", fig,
        lambda w: render_grid_shot(w, panels,
                                   rolls_for("accidental_success"),
                                   N_STATES, counter))
    main_only = os.path.join(MEDIA_DIR, "_main_only.mp4")
    subprocess.run(["cp", main_seg, main_only], check=True)

    # contrast shot: corrective flailing
    fig, panels, counter = make_shot_figure(
        "flailing", entries, means, ORANGE,
        ["Corrective flailing",
         "demos that wobble hard at the start,",
         "then recover and succeed"])
    sw.render("contrast", fig,
              lambda w: render_grid_shot(w, panels, rolls_for("flailing"),
                                         N_STATES, counter))

    # closing card
    sw.render_static("card", make_closing_card_figure(means),
                     CARD_SECONDS * FPS)

    out_mp4 = os.path.join(MEDIA_DIR, "demo_quality_video.mp4")
    sw.concat(out_mp4)
    print("wrote", out_mp4)

    out_gif = os.path.join(MEDIA_DIR, "main_shot.gif")
    make_gif(main_only, out_gif)
    os.remove(main_only)
    print("wrote", out_gif)


if __name__ == "__main__":
    main()

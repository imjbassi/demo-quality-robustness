"""Render real closed-loop rollouts of the checkpointed video policies into
the results video (1080p MP4) and the README GIF.

Everything shown is measured, nothing is staged:

- The five panels of each shot share identical initial states because each
  policy's environment is constructed with the same fixed seed
  (``INIT_SEED``); the seed was fixed a priori, not chosen by scanning
  rollouts for dramatic failures.
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

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from env2d import CONTACT_R, GOAL_R, MAX_STEPS, WS_HI, WS_LO
from env2d import Push2DVec
from policy import BCPolicy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "viz", "checkpoints")
MEDIA_DIR = os.path.join(REPO, "viz", "media")
RESULTS_CSV = os.path.join(REPO, "results", "results.csv")

# Fixed a priori. All panels of a shot reset their environments with this
# seed, so every policy faces the exact same initial states.
INIT_SEED = 42
N_STATES = 5            # initial states played in sequence (main shot)
N_STATES_CONTRAST = 5   # contrast shot reuses the same states
FPS = 30
HOLD_FRAMES = 24        # freeze on each episode's final state
CARD_SECONDS = 6
TRAIL = 55              # fading-trail window, in env steps

RHOS = [0.0, 0.5, 0.75, 0.9, 1.0]

# Palette (validated reference set: light surface, series blue/orange,
# aqua goal accent, reserved critical red for wrong-side).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"        # gripper + accidental-success series
ORANGE = "#eb6834"      # flailing series
GOAL_ACCENT = "#1baf7a"
BLOCK_FILL = "#b8b6ae"
BLOCK_EDGE = "#52514e"
WRONG = "#d03b3b"       # critical: wrong-side push
GOOD = "#0ca30c"

WS_PAD = 0.03


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_manifest():
    path = os.path.join(CKPT_DIR, "manifest.json")
    if not os.path.exists(path):
        sys.exit("viz/checkpoints/manifest.json not found. Run "
                 "`python viz/train_video_policies.py` first; it only writes "
                 "the manifest after the reproduction check passes.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_policy(entry):
    pol = BCPolicy()
    pol.load_state_dict(torch.load(os.path.join(CKPT_DIR, entry["checkpoint"]),
                                   weights_only=True))
    return pol.eval()


@torch.no_grad()
def record_rollouts(pol, n_states, seed):
    """Run ``n_states`` real episodes and log every state. The env freezes
    finished episodes, so trajectories are constant after termination."""
    env = Push2DVec(n_states, seed=seed)
    obs = env.reset()
    grip = [env.grip.copy()]
    block = [env.block.copy()]
    succ_h = [env.success.copy()]
    done_h = [env.done.copy()]
    for _ in range(MAX_STEPS):
        act = pol(torch.tensor(obs, dtype=torch.float32)).numpy()
        obs, _, done = env.step(np.clip(act, -1, 1))
        grip.append(env.grip.copy())
        block.append(env.block.copy())
        succ_h.append(env.success.copy())
        done_h.append(env.done.copy())
        if done.all():
            break
    grip = np.stack(grip, 1)          # (B, T+1, 2)
    block = np.stack(block, 1)
    succ_h = np.stack(succ_h, 1)      # (B, T+1)
    done_h = np.stack(done_h, 1)
    goal = env.goal.copy()
    d = np.linalg.norm(block - goal[:, None, :], axis=-1)   # (B, T+1)
    # Repo wrong-side predicate (policy.eval_wrong_side_rate), evaluated on
    # the rollout's current state at every step.
    wrong = (d > d[:, :1] + 0.02) & ~succ_h
    # first step index at which the episode is frozen (done), else last
    T = grip.shape[1] - 1
    steps = np.where(done_h.any(1), done_h.argmax(1), T)
    return {"grip": grip, "block": block, "goal": goal, "succ": succ_h,
            "wrong": wrong, "steps": steps, "T": T}


def paper_means():
    """10-seed mean success per (corruption, rho) from the committed CSV."""
    acc = defaultdict(list)
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["policy"] != "mlp":
                continue
            acc[(r["corruption"], float(r["rho"]))].append(
                float(r["closed_loop_success"]))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(WS_LO - WS_PAD, WS_HI + WS_PAD)
    ax.set_ylim(WS_LO - WS_PAD, WS_HI + WS_PAD)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(1.2)


def fading_segments(path, t, color, lw):
    """LineCollection segments + RGBA colors for the last TRAIL steps."""
    lo = max(0, t - TRAIL)
    pts = path[lo:t + 1]
    if len(pts) < 2:
        return np.zeros((0, 2, 2)), np.zeros((0, 4))
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    alpha = np.linspace(0.06, 0.85, n)
    rgba = np.tile(matplotlib.colors.to_rgba(color), (n, 1))
    rgba[:, 3] = alpha
    return segs, rgba


class Panel:
    """One policy's panel: goal, block, gripper, fading trails, labels."""

    def __init__(self, ax, rho, success_rate, series_color):
        self.ax = ax
        self.rho = rho
        style_axes(ax)
        self.goal_ring = Circle((0, 0), GOAL_R, facecolor=GOAL_ACCENT,
                                edgecolor=GOAL_ACCENT, alpha=0.18, lw=0)
        self.goal_edge = Circle((0, 0), GOAL_R, facecolor="none",
                                edgecolor=GOAL_ACCENT, lw=2.0)
        ax.add_patch(self.goal_ring)
        ax.add_patch(self.goal_edge)
        self.block = Circle((0, 0), CONTACT_R, facecolor=BLOCK_FILL,
                            edgecolor=BLOCK_EDGE, lw=1.6, zorder=4)
        ax.add_patch(self.block)
        (self.grip_dot,) = ax.plot([], [], "o", color=BLUE, ms=9, zorder=6,
                                   mec="white", mew=1.0)
        self.grip_trail = LineCollection([], linewidths=2.0, zorder=3,
                                         capstyle="round")
        self.block_trail = LineCollection([], linewidths=2.4, zorder=2,
                                          capstyle="round")
        ax.add_collection(self.grip_trail)
        ax.add_collection(self.block_trail)
        label = "clean data" if rho == 0 else f"ρ = {rho}"
        ax.set_title(label, color=INK, fontsize=17, pad=26,
                     fontfamily="sans-serif", fontweight="bold")
        self.sub = ax.text(0.5, 1.015, f"success rate {success_rate:.2f}",
                           transform=ax.transAxes, ha="center", va="bottom",
                           color=SECONDARY, fontsize=12.5)
        self.badge = ax.text(0.5, 0.045, "", transform=ax.transAxes,
                             ha="center", va="bottom", fontsize=13,
                             fontweight="bold")
        self.wrong_tag = ax.text(0.5, 0.955, "", transform=ax.transAxes,
                                 ha="center", va="top", color=WRONG,
                                 fontsize=12.5, fontweight="bold")

    def set_episode(self, roll, ep):
        self.roll = roll
        self.ep = ep
        g = roll["goal"][ep]
        self.goal_ring.center = g
        self.goal_edge.center = g
        self.badge.set_text("")
        self.wrong_tag.set_text("")

    def draw_step(self, t):
        roll, ep = self.roll, self.ep
        tt = min(t, roll["T"])
        gp = roll["grip"][ep, :tt + 1]
        bp = roll["block"][ep, :tt + 1]
        self.grip_dot.set_data([gp[-1, 0]], [gp[-1, 1]])
        self.block.center = (bp[-1, 0], bp[-1, 1])
        segs, cols = fading_segments(gp, tt, BLUE, 2.0)
        self.grip_trail.set_segments(segs)
        self.grip_trail.set_color(cols)
        segs, cols = fading_segments(bp, tt, BLOCK_EDGE, 2.4)
        self.block_trail.set_segments(segs)
        self.block_trail.set_color(cols)

        wrong_now = bool(roll["wrong"][ep, tt])
        self.block.set_edgecolor(WRONG if wrong_now else BLOCK_EDGE)
        self.block.set_linewidth(2.6 if wrong_now else 1.6)
        self.wrong_tag.set_text("wrong-side push" if wrong_now else "")

        ended = tt >= roll["steps"][ep]
        if ended:
            if roll["succ"][ep, tt]:
                self.badge.set_text("✓ success")
                self.badge.set_color(GOOD)
            else:
                # exactly the episode-level condition of eval_wrong_side_rate
                tag = "✗ failed (wrong side)" if wrong_now else "✗ failed"
                self.badge.set_text(tag)
                self.badge.set_color(WRONG)


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


# ---------------------------------------------------------------------------
# shots
# ---------------------------------------------------------------------------

def render_grid_shot(writer, fig, panels, info_builder, rollouts, n_states,
                     counter_text):
    """Play n_states episodes in sequence, all panels in lockstep."""
    for ep in range(n_states):
        for p, roll in zip(panels, rollouts):
            p.set_episode(roll, ep)
        seg = int(max(roll["steps"][ep] for roll in rollouts))
        for t in range(seg + 1):
            for p in panels:
                p.draw_step(t)
            counter_text.set_text(
                f"initial state {ep + 1}/{n_states}   ·   t = {t:3d}")
            writer.grab_frame()
        for _ in range(HOLD_FRAMES):
            writer.grab_frame()


def make_shot_figure(mode_key, entries_by, manifest, means, series_color,
                     title_lines):
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
        panels.append(Panel(ax, rho, e["closed_loop_success"], series_color))
    info_ax = fig.add_subplot(gs[1, 2])
    build_info_axis(info_ax, title_lines, mode_key, means, series_color)
    counter = info_ax.text(0.03, 0.015, "", transform=info_ax.transAxes,
                           color=MUTED, fontsize=11, va="bottom")
    return fig, panels, counter


class SegmentWriter:
    """FFMpegWriter that can be re-pointed at successive figures by writing
    per-shot segment files, concatenated afterwards."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.segments = []

    def render(self, name, fig, fn):
        path = os.path.join(self.out_dir, f"_seg_{name}.mp4")
        w = FFMpegWriter(fps=FPS, codec="h264",
                         extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
        with w.saving(fig, path, dpi=100):
            fn(w)
        self.segments.append(path)
        plt.close(fig)

    def render_static(self, name, fig, n_frames):
        self.render(name, fig, lambda w: [w.grab_frame()
                                          for _ in range(n_frames)])

    def concat(self, out_path):
        lst = os.path.join(self.out_dir, "_segments.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for s in self.segments:
                f.write(f"file '{os.path.abspath(s)}'\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", lst, "-c", "copy", out_path],
                       check=True)
        for s in self.segments + [lst]:
            os.remove(s)


def make_gif(src_mp4, out_gif, width=640, fps=15):
    palette = out_gif + ".palette.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_mp4,
                    "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,"
                    "palettegen=stats_mode=diff", palette], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_mp4,
                    "-i", palette, "-lavfi",
                    f"fps={fps},scale={width}:-1:flags=lanczos[x];"
                    "[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                    out_gif], check=True)
    os.remove(palette)


# ---------------------------------------------------------------------------

def main():
    os.makedirs(MEDIA_DIR, exist_ok=True)
    manifest = load_manifest()
    entries = {(e["corruption"], e["rho"]): e for e in manifest["policies"]}
    means = paper_means()

    print("recording rollouts (init seed", INIT_SEED, ")")
    rollouts = {}
    for (key, rho), e in entries.items():
        pol = load_policy(e)
        rollouts[(key, rho)] = record_rollouts(pol, N_STATES, INIT_SEED)
        r = rollouts[(key, rho)]
        n_wrong = int((r["wrong"][np.arange(N_STATES), r["steps"]]
                       & ~r["succ"][np.arange(N_STATES), r["steps"]]).sum())
        print(f"  {key:22s} rho={rho:<4g} "
              f"succ {int(r['succ'][:, -1].sum())}/{N_STATES} "
              f"wrong-side {n_wrong}/{N_STATES} "
              f"steps {list(r['steps'])}")

    def rolls_for(mode):
        return [rollouts[("clean" if r == 0 else mode, r)] for r in RHOS]

    sw = SegmentWriter(MEDIA_DIR)

    # main shot: accidental success
    fig, panels, counter = make_shot_figure(
        "accidental_success", entries, manifest, means, BLUE,
        ["Accidental success",
         "demos that reached the goal by luck,",
         "not by correct behaviour"])
    sw.render("main", fig,
              lambda w: render_grid_shot(w, fig, panels, None,
                                         rolls_for("accidental_success"),
                                         N_STATES, counter))
    main_only = os.path.join(MEDIA_DIR, "_main_only.mp4")
    subprocess.run(["cp", sw.segments[0], main_only], check=True)

    # contrast shot: corrective flailing
    fig, panels, counter = make_shot_figure(
        "flailing", entries, manifest, means, ORANGE,
        ["Corrective flailing",
         "demos that wobble hard at the start,",
         "then recover and succeed"])
    sw.render("contrast", fig,
              lambda w: render_grid_shot(w, fig, panels, None,
                                         rolls_for("flailing"),
                                         N_STATES_CONTRAST, counter))

    # closing card
    card_fig = make_closing_card_figure(means)
    sw.render_static("card", card_fig, CARD_SECONDS * FPS)

    out_mp4 = os.path.join(MEDIA_DIR, "demo_quality_video.mp4")
    sw.concat(out_mp4)
    print("wrote", out_mp4)

    out_gif = os.path.join(MEDIA_DIR, "main_shot.gif")
    make_gif(main_only, out_gif)
    os.remove(main_only)
    print("wrote", out_gif)


def make_closing_card_figure(means):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.10, 0.13, 0.80, 0.66])
    ax.set_facecolor(SURFACE)
    rhos_all = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

    series = [("flailing", ORANGE, "corrective flailing", (0, 12)),
              ("accidental_success", BLUE, "accidental success", (0, 14))]
    for mode, color, label, _ in series:
        ys = [means[(mode if r > 0 else "clean", r)][0] for r in rhos_all]
        ax.plot(rhos_all, ys, color=color, lw=2.5, marker="o", ms=9,
                mec="white", mew=1.2, zorder=5)
    # direct labels
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


if __name__ == "__main__":
    main()

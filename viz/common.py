"""Shared helpers for the viz/ videos: palette, panel drawing, rollout
recording, segment writing, and GIF encoding.

Used by make_video.py (dose-response video) and make_video2.py (compounding
error video). Nothing here touches the research code; the environment and
policies are imported unchanged.
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

from env2d import CONTACT_R, GOAL_R, MAX_STEPS, WS_HI, WS_LO, Push2DVec
from policy import BCPolicy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "viz", "checkpoints")
MEDIA_DIR = os.path.join(REPO, "viz", "media")
RESULTS_CSV = os.path.join(REPO, "results", "results.csv")

# Fixed a priori. Every panel of a shot resets its environment with this
# seed, so all policies face the exact same initial states.
INIT_SEED = 42
FPS = 30
HOLD_FRAMES = 24        # freeze on each episode's final state
TRAIL = 55              # fading-trail window, in env steps
WS_PAD = 0.03

# Palette (validated reference set: light surface, series blue/orange,
# aqua goal accent, reserved critical red for wrong-side).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"        # gripper + primary series
ORANGE = "#eb6834"      # second series
GOAL_ACCENT = "#1baf7a"
BLOCK_FILL = "#b8b6ae"
BLOCK_EDGE = "#52514e"
WRONG = "#d03b3b"       # critical: wrong-side push
GOOD = "#0ca30c"


# ---------------------------------------------------------------------------
# data loading
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


def load_fits():
    """All unique MLP fits from the committed CSV, deduplicating the clean
    rows that run_experiment.py mirrors onto every corruption mode.
    Returns a list of dicts with corruption, rho, seed, mse, success."""
    seen, fits = set(), []
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["policy"] != "mlp":
                continue
            rho = float(r["rho"])
            key = (r["corruption"] if rho > 0 else "clean", rho,
                   int(r["seed"]))
            if key in seen:
                continue
            seen.add(key)
            fits.append({"corruption": key[0], "rho": rho, "seed": key[2],
                         "mse": float(r["open_loop_mse"]),
                         "success": float(r["closed_loop_success"])})
    return fits


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
# rollout recording
# ---------------------------------------------------------------------------

@torch.no_grad()
def record_rollouts(pol, n_states, seed):
    """Run ``n_states`` real episodes and log every state. The env freezes
    finished episodes, so trajectories are constant after termination."""
    env = Push2DVec(n_states, seed=seed)
    obs = env.reset()
    grip = [env.grip.copy()]
    block = [env.block.copy()]
    obs_h = [obs.copy()]
    succ_h = [env.success.copy()]
    done_h = [env.done.copy()]
    for _ in range(MAX_STEPS):
        act = pol(torch.tensor(obs, dtype=torch.float32)).numpy() \
            if isinstance(pol, torch.nn.Module) else pol(obs)
        obs, _, done = env.step(np.clip(act, -1, 1))
        grip.append(env.grip.copy())
        block.append(env.block.copy())
        obs_h.append(obs.copy())
        succ_h.append(env.success.copy())
        done_h.append(env.done.copy())
        if done.all():
            break
    grip = np.stack(grip, 1)          # (B, T+1, 2)
    block = np.stack(block, 1)
    obs_h = np.stack(obs_h, 1)        # (B, T+1, obs_dim)
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
    return {"grip": grip, "block": block, "obs": obs_h, "goal": goal,
            "succ": succ_h, "wrong": wrong, "steps": steps, "T": T}


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


def fading_segments(path, t, color, trail=TRAIL, max_alpha=0.85):
    """LineCollection segments + RGBA colors for the last ``trail`` steps."""
    lo = max(0, t - trail)
    pts = path[lo:t + 1]
    if len(pts) < 2:
        return np.zeros((0, 2, 2)), np.zeros((0, 4))
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    alpha = np.linspace(0.06, max_alpha, n)
    rgba = np.tile(matplotlib.colors.to_rgba(color), (n, 1))
    rgba[:, 3] = alpha
    return segs, rgba


class Panel:
    """One policy's panel: goal, block, gripper, fading trails, labels."""

    def __init__(self, ax, title, subtitle):
        self.ax = ax
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
        ax.set_title(title, color=INK, fontsize=17, pad=26,
                     fontfamily="sans-serif", fontweight="bold")
        self.sub = ax.text(0.5, 1.015, subtitle,
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
        segs, cols = fading_segments(gp, tt, BLUE)
        self.grip_trail.set_segments(segs)
        self.grip_trail.set_color(cols)
        segs, cols = fading_segments(bp, tt, BLOCK_EDGE)
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


def render_grid_shot(writer, panels, rollouts, n_states, counter_text,
                     counter_fmt="initial state {ep}/{n}   ·   t = {t:3d}"):
    """Play n_states episodes in sequence, all panels in lockstep."""
    for ep in range(n_states):
        for p, roll in zip(panels, rollouts):
            p.set_episode(roll, ep)
        seg = int(max(roll["steps"][ep] for roll in rollouts))
        for t in range(seg + 1):
            for p in panels:
                p.draw_step(t)
            counter_text.set_text(
                counter_fmt.format(ep=ep + 1, n=n_states, t=t))
            writer.grab_frame()
        for _ in range(HOLD_FRAMES):
            writer.grab_frame()


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

class SegmentWriter:
    """FFMpegWriter that renders per-shot segment files and concatenates
    them afterwards."""

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
        return path

    def render_static(self, name, fig, n_frames):
        return self.render(name, fig,
                           lambda w: [w.grab_frame()
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

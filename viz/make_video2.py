"""Render the compounding-error video: why open-loop evaluation can look
fine while closed-loop behaviour fails.

Everything shown is measured, nothing is staged:

- The ghost is the repo's own open-loop definition
  (``policy.eval_open_loop``: the policy queried on expert states) made
  spatial: at each step t the policy is queried on the EXPERT's recorded
  observation s_t, one environment step is applied from the expert's exact
  state, and the one-step prediction is drawn; the state then resets to the
  expert's s_{t+1}. No new metric is invented.
- The policy pair in shot 2 is picked by a stated rule, computed from the
  committed ``results/results.csv``, not by eye: among the 49 runs whose
  open-loop MSE lies in [0.20, 0.30], excluding inconsistent-strategy runs
  (whose low success the paper attributes to the alternate strategy's lower
  clonability), take the pair with the largest closed-loop success gap
  subject to |MSE difference| <= 0.01.
- Initial states come from the same fixed a-priori seed as the first video
  (``common.INIT_SEED``). Shot 1 shows the first initial state (lowest
  index) that the low policy fails; if it failed none, the state with the
  largest final block-goal distance would be used.
- All labels are the retrained policies' own measured numbers from
  ``viz/checkpoints/manifest.json`` (written only after the reproduction
  gate passes) or the committed CSV.

Usage:  python viz/make_video2.py           (checkpoints must exist already)
Output: viz/media/compounding_error_video.mp4, viz/media/matched_mse.gif
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from env2d import Push2DVec, expert_action
from common import (BLOCK_EDGE, BLUE, FPS, GRID, GOOD, HOLD_FRAMES,
                    INIT_SEED, INK, MEDIA_DIR, MUTED, ORANGE, SECONDARY,
                    SURFACE, WRONG, Panel, SegmentWriter, load_fits,
                    load_manifest, load_policy, make_gif, record_rollouts)

N_STATES = 5
CARD_SECONDS = 7
SHOT1_FRAMES_PER_STEP = 3

MSE_BAND = (0.20, 0.30)
MAX_MSE_GAP = 0.01

EXPERT_GRAY = "#a5a29a"
GHOST_ALPHA = 0.30

MODE_LABELS = {
    "clean": "clean", "occlusion": "occlusion",
    "accidental_success": "accidental success",
    "flailing": "corrective flailing", "truncation": "truncation",
    "inconsistent_strategy": "inconsistent strategy",
}
# Categorical palette slots in fixed order + distinct markers as the
# secondary (CVD-safe) encoding for the 6-way scatter.
MODE_STYLE = {
    "accidental_success": ("#2a78d6", "o"),
    "flailing": ("#eb6834", "s"),
    "occlusion": ("#1baf7a", "^"),
    "truncation": ("#eda100", "D"),
    "inconsistent_strategy": ("#e87ba4", "v"),
    "clean": ("#898781", "P"),
}


# ---------------------------------------------------------------------------
# pair selection (the stated rule, executed)
# ---------------------------------------------------------------------------

def select_pair(fits):
    band = [f for f in fits if MSE_BAND[0] <= f["mse"] <= MSE_BAND[1]]
    eligible = [f for f in band if f["corruption"] != "inconsistent_strategy"]
    best = None
    for lo in eligible:
        for hi in eligible:
            gap = hi["success"] - lo["success"]
            if gap > 0 and abs(hi["mse"] - lo["mse"]) <= MAX_MSE_GAP:
                if best is None or gap > best[0]:
                    best = (gap, lo, hi)
    if best is None:
        raise SystemExit("no eligible pair in the MSE band")
    return best[1], best[2], band


# ---------------------------------------------------------------------------
# expert + ghost recording
# ---------------------------------------------------------------------------

@torch.no_grad()
def record_ghost(pol, expert_roll, seed):
    """One-step open-loop predictions from the expert's recorded states:
    exactly the states and policy queries behind eval_open_loop, with one
    environment step applied to make the prediction spatial."""
    grip_e, block_e = expert_roll["grip"], expert_roll["block"]
    obs_e, goal = expert_roll["obs"], expert_roll["goal"]
    B, T1 = grip_e.shape[0], grip_e.shape[1]
    env = Push2DVec(B, seed=seed)
    env.reset()
    g_pred = np.zeros((B, T1 - 1, 2))
    b_pred = np.zeros((B, T1 - 1, 2))
    for t in range(T1 - 1):
        env.grip = grip_e[:, t].copy()
        env.block = block_e[:, t].copy()
        env.goal = goal.copy()
        env.t = 0
        env.done = np.zeros(B, dtype=bool)
        env.success = np.zeros(B, dtype=bool)
        act = pol(torch.tensor(obs_e[:, t], dtype=torch.float32)).numpy()
        env.step(np.clip(act, -1, 1))
        g_pred[:, t] = env.grip
        b_pred[:, t] = env.block
    return {"grip": g_pred, "block": b_pred}


def state_dist(g1, b1, g2, b2):
    """Euclidean distance in the joint (gripper, block) state space."""
    return np.sqrt(((g1 - g2) ** 2).sum(-1) + ((b1 - b2) ** 2).sum(-1))


def divergence_series(roll, expert_roll, ghost, ep):
    """Per-step closed-loop drift from the expert, and the ghost's one-step
    error, for one episode."""
    Tc, Te = roll["T"], expert_roll["T"]
    te_idx = np.minimum(np.arange(Tc + 1), Te)   # expert frozen after done
    drift = state_dist(roll["grip"][ep], roll["block"][ep],
                       expert_roll["grip"][ep, te_idx],
                       expert_roll["block"][ep, te_idx])
    n_ghost = int(min(expert_roll["steps"][ep], Te))
    ghost_err = state_dist(ghost["grip"][ep, :n_ghost],
                           ghost["block"][ep, :n_ghost],
                           expert_roll["grip"][ep, 1:n_ghost + 1],
                           expert_roll["block"][ep, 1:n_ghost + 1])
    return drift, ghost_err


# ---------------------------------------------------------------------------
# the core visual: expert + ghost + closed-loop, divergence below
# ---------------------------------------------------------------------------

class CompoundPanel(Panel):
    """Panel with the expert trajectory, the open-loop ghost, and a live
    divergence plot underneath."""

    def __init__(self, ax, div_ax, title, subtitle):
        super().__init__(ax, title, subtitle)
        (self.exp_grip_line,) = ax.plot([], [], color=EXPERT_GRAY, lw=2.0,
                                        alpha=0.9, zorder=1)
        (self.exp_block_line,) = ax.plot([], [], color=EXPERT_GRAY, lw=2.4,
                                         alpha=0.5, zorder=1)
        (self.exp_dot,) = ax.plot([], [], "o", color=EXPERT_GRAY, ms=7,
                                  zorder=2, mec="white", mew=0.8)
        (self.ghost_dots,) = ax.plot([], [], "o", color=BLUE, ms=5,
                                     alpha=GHOST_ALPHA, zorder=2,
                                     markeredgewidth=0, ls="none")
        # the legend occupies the top of the panel here; anchor the
        # wrong-side tag bottom-right instead
        self.wrong_tag.set_position((0.97, 0.115))
        self.wrong_tag.set_ha("right")
        self.wrong_tag.set_va("bottom")
        self.div_ax = div_ax
        div_ax.set_facecolor(SURFACE)
        for s in ["top", "right"]:
            div_ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            div_ax.spines[s].set_color(GRID)
        div_ax.tick_params(colors=MUTED, labelsize=9, length=2)
        div_ax.grid(color=GRID, lw=0.6, alpha=0.7)
        (self.drift_line,) = div_ax.plot([], [], color=BLUE, lw=2.2)
        (self.ghost_line,) = div_ax.plot([], [], color=BLUE, lw=2.0,
                                         alpha=GHOST_ALPHA)
        self.drift_label = div_ax.text(0, 0, "", color=BLUE, fontsize=10.5,
                                       fontweight="bold", va="center")
        self.ghost_label = div_ax.text(0, 0, "", color=BLUE, fontsize=10.5,
                                       alpha=0.55, va="center")
        div_ax.set_xlabel("timestep", color=MUTED, fontsize=10, labelpad=1)
        div_ax.set_ylabel("distance from expert", color=MUTED, fontsize=10)

    def set_episode_data(self, roll, expert_roll, ghost, ep, seg):
        self.set_episode(roll, ep)
        self.expert_roll = expert_roll
        self.ghost = ghost
        self.drift, self.ghost_err = divergence_series(
            roll, expert_roll, ghost, ep)
        self.seg = seg
        self.div_ax.set_xlim(0, seg)
        top = max(0.30, float(self.drift.max()) * 1.15)
        self.div_ax.set_ylim(0, top)
        self.exp_steps = int(expert_roll["steps"][ep])

    def draw_step(self, t):
        super().draw_step(t)
        ep = self.ep
        te = min(t, self.exp_steps)
        eg = self.expert_roll["grip"][ep, :te + 1]
        eb = self.expert_roll["block"][ep, :te + 1]
        self.exp_grip_line.set_data(eg[:, 0], eg[:, 1])
        self.exp_block_line.set_data(eb[:, 0], eb[:, 1])
        self.exp_dot.set_data([eg[-1, 0]], [eg[-1, 1]])
        ng = min(te, len(self.ghost["grip"][ep]))
        self.ghost_dots.set_data(self.ghost["grip"][ep, :ng, 0],
                                 self.ghost["grip"][ep, :ng, 1])

        tt = min(t, self.roll["T"])
        xs = np.arange(tt + 1)
        self.drift_line.set_data(xs, self.drift[:tt + 1])
        ngh = min(tt, len(self.ghost_err))
        self.ghost_line.set_data(np.arange(1, ngh + 1), self.ghost_err[:ngh])
        if tt > 2:
            self.drift_label.set_position((tt, self.drift[tt]))
            self.drift_label.set_text("  closed-loop drift")
            if ngh > 0:
                self.ghost_label.set_position((ngh, self.ghost_err[ngh - 1]))
                self.ghost_label.set_text("  one-step (open-loop) error")


def tri_legend(ax):
    handles = [
        Line2D([], [], color=EXPERT_GRAY, lw=2.2, label="expert (scripted)"),
        Line2D([], [], color=BLUE, marker="o", ls="none", ms=6,
               alpha=GHOST_ALPHA, label="open-loop ghost (one step from expert states)"),
        Line2D([], [], color=BLUE, lw=2.2, label="closed-loop policy"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10,
              labelcolor=SECONDARY, handlelength=1.6,
              borderaxespad=0.2)


def render_compound_shot(writer, panels, episodes, counter, frames_per_step,
                         n_states_label):
    """episodes: list of (ep_index, seg_length). All panels lockstep."""
    for k, (ep, seg) in enumerate(episodes, 1):
        for p in panels:
            p.prepare(ep, seg)
        # shared divergence scale so side-by-side panels compare honestly
        top = max(p.panel.div_ax.get_ylim()[1] for p in panels)
        for p in panels:
            p.panel.div_ax.set_ylim(0, top)
        for t in range(seg + 1):
            for p in panels:
                p.draw_step(t)
            counter.set_text(f"initial state {k}/{n_states_label}   "
                             f"·   t = {t:3d}")
            for _ in range(frames_per_step):
                writer.grab_frame()
        for _ in range(HOLD_FRAMES):
            writer.grab_frame()


# ---------------------------------------------------------------------------
# shots
# ---------------------------------------------------------------------------

def policy_label(e):
    return (f"{MODE_LABELS[e['corruption']]}  ρ = {e['rho']}  "
            f"seed {e['train_seed']}")


def policy_subtitle(e):
    return (f"open-loop MSE {e['open_loop_mse']:.3f}   ·   "
            f"closed-loop success {e['closed_loop_success']:.2f}")


class BoundPanel:
    """Binds a CompoundPanel to one policy's rollout data so multi-policy
    shots can prepare episodes uniformly."""

    def __init__(self, panel, roll, expert_roll, ghost):
        self.panel = panel
        self.roll = roll
        self.expert_roll = expert_roll
        self.ghost = ghost

    def prepare(self, ep, seg):
        self.panel.set_episode_data(self.roll, self.expert_roll, self.ghost,
                                    ep, seg)

    def draw_step(self, t):
        self.panel.draw_step(t)


def make_shot1_figure(entry, roll, expert_roll, ghost):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.055, 0.34, 0.50, 0.56])
    dax = fig.add_axes([0.075, 0.075, 0.46, 0.19])
    panel = CompoundPanel(ax, dax, policy_label(entry),
                          policy_subtitle(entry))
    tri_legend(ax)

    fig.text(0.62, 0.86, "Compounding error", color=INK, fontsize=24,
             fontweight="bold")
    fig.text(0.62, 0.815,
             "The ghost asks: from each state the EXPERT visited,\n"
             "what would the policy do for one step?\n"
             "That is open-loop evaluation, and it looks fine.",
             color=SECONDARY, fontsize=14, va="top")
    fig.text(0.62, 0.665,
             "The closed-loop run starts from the same state and\n"
             "must live with its own small mistakes. They move it\n"
             "off the expert's states, where the policy was never\n"
             "trained — so the next mistake is bigger.",
             color=SECONDARY, fontsize=14, va="top")
    fig.text(0.62, 0.48,
             f"this policy: open-loop MSE {entry['open_loop_mse']:.3f} "
             f"(committed {entry['committed_mse']:.3f})\n"
             f"closed-loop success {entry['closed_loop_success']:.2f} "
             f"(committed {entry['committed_success']:.2f})\n"
             f"initial state: first one it fails under env seed {INIT_SEED}",
             color=MUTED, fontsize=12, va="top")
    counter = fig.text(0.62, 0.10, "", color=MUTED, fontsize=11)
    return fig, panel, counter


def make_shot2_figure(entries, rolls, expert_roll, ghosts):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.5, 0.955, "Same open-loop score, different robot",
             color=INK, fontsize=22, fontweight="bold", ha="center")
    fig.text(0.5, 0.915,
             "two policies, nearly identical open-loop MSE — "
             "same initial states, same moment",
             color=SECONDARY, fontsize=14, ha="center")
    bound = []
    for i, (e, roll, ghost) in enumerate(zip(entries, rolls, ghosts)):
        ax = fig.add_axes([0.075 + i * 0.47, 0.345, 0.38, 0.50])
        dax = fig.add_axes([0.085 + i * 0.47, 0.065, 0.36, 0.185])
        panel = CompoundPanel(ax, dax, policy_label(e), policy_subtitle(e))
        if i == 0:
            tri_legend(ax)
        bound.append(BoundPanel(panel, roll, expert_roll, ghost))
    counter = fig.text(0.5, 0.285, "", color=MUTED, fontsize=11,
                       ha="center")
    return fig, bound, counter


def make_scatter_card(fits, band, lo, hi):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.09, 0.12, 0.60, 0.70])
    ax.set_facecolor(SURFACE)

    ax.axvspan(*MSE_BAND, color=GRID, alpha=0.45, zorder=0)
    for mode, (color, marker) in MODE_STYLE.items():
        pts = [f for f in fits if f["corruption"] == mode]
        ax.plot([f["mse"] for f in pts], [f["success"] for f in pts],
                marker, color=color, ms=7, alpha=0.75, mec="white",
                mew=0.5, ls="none", label=MODE_LABELS[mode])
    for f, name in [(lo, "shown: success "
                     f"{lo['success']:.2f}"),
                    (hi, f"shown: success {hi['success']:.2f}")]:
        ax.plot([f["mse"]], [f["success"]], "o", ms=15, mfc="none",
                mec=INK, mew=2.0, zorder=6)
        ax.annotate(name, (f["mse"], f["success"]),
                    xytext=(14, -4 if f is lo else 10),
                    textcoords="offset points", color=INK, fontsize=12,
                    fontweight="bold")

    mse = np.array([f["mse"] for f in fits])
    succ = np.array([f["success"] for f in fits])
    r = float(np.corrcoef(mse, succ)[0, 1])
    band_lo = min(f["success"] for f in band)
    band_hi = max(f["success"] for f in band)
    noninc = [f for f in band if f["corruption"] != "inconsistent_strategy"]
    nl, nh = (min(f["success"] for f in noninc),
              max(f["success"] for f in noninc))

    ax.set_xlabel("open-loop MSE (held-out clean expert states)",
                  color=SECONDARY, fontsize=14)
    ax.set_ylabel("closed-loop success", color=SECONDARY, fontsize=14)
    ax.set_ylim(-0.05, 1.07)
    ax.tick_params(colors=MUTED, labelsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, lw=0.7, alpha=0.7)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=11,
                    labelcolor=SECONDARY)

    fig.text(0.09, 0.90, "One score, many robots", color=INK, fontsize=25,
             fontweight="bold")
    fig.text(0.09, 0.862,
             f"All {len(fits)} MLP fits. Pooled correlation r = {r:.2f} "
             "looks reassuring; within single failure modes it ranges "
             "from −0.13 to −0.95.",
             color=SECONDARY, fontsize=14)
    fig.text(0.72, 0.60,
             f"the matched band\nMSE {MSE_BAND[0]:.2f}–{MSE_BAND[1]:.2f}: "
             f"{len(band)} runs,\nsuccess {band_lo:.2f} to {band_hi:.2f}",
             color=INK, fontsize=15, fontweight="bold", va="top")
    fig.text(0.72, 0.47,
             "caveat: the band's lowest runs are\n"
             "inconsistent-strategy fits, whose low\n"
             "success reflects the alternate strategy\n"
             "being harder to clone, not open-loop\n"
             "blindness. Excluding them, the band\n"
             f"still spans {nl:.2f} to {nh:.2f}.",
             color=SECONDARY, fontsize=12.5, va="top")
    fig.text(0.09, 0.045,
             "Data: results/results.csv — 260 unique MLP fits; ringed "
             "points are the two policies shown in this video.",
             color=MUTED, fontsize=12)
    return fig


# ---------------------------------------------------------------------------

def main():
    os.makedirs(MEDIA_DIR, exist_ok=True)
    manifest = load_manifest()
    fits = load_fits()
    lo, hi, band = select_pair(fits)
    print("pair rule -> low:", lo, "\n            high:", hi)

    def manifest_entry(f):
        for e in manifest["policies"]:
            if (e["corruption"] == f["corruption"] and e["rho"] == f["rho"]
                    and e["train_seed"] == f["seed"]):
                return e
        raise SystemExit(
            f"no checkpoint for {f}; add it to train_video_policies.py "
            "CONFIGS and rerun the training step")

    e_lo, e_hi = manifest_entry(lo), manifest_entry(hi)
    print("\nreproduction check (from gated manifest):")
    for e in (e_lo, e_hi):
        print(f"  {e['corruption']} rho={e['rho']} seed={e['train_seed']}: "
              f"success {e['closed_loop_success']:.3f} vs committed "
              f"{e['committed_success']:.3f}; MSE {e['open_loop_mse']:.3f} "
              f"vs committed {e['committed_mse']:.3f}")

    pol_lo, pol_hi = load_policy(e_lo), load_policy(e_hi)

    print("\nrecording rollouts (init seed", INIT_SEED, ")")
    expert_roll = record_rollouts(lambda o: expert_action(o), N_STATES,
                                  INIT_SEED)
    roll_lo = record_rollouts(pol_lo, N_STATES, INIT_SEED)
    roll_hi = record_rollouts(pol_hi, N_STATES, INIT_SEED)
    ghost_lo = record_ghost(pol_lo, expert_roll, INIT_SEED)
    ghost_hi = record_ghost(pol_hi, expert_roll, INIT_SEED)
    for name, r in [("expert", expert_roll), ("low", roll_lo),
                    ("high", roll_hi)]:
        print(f"  {name:6s} succ {int(r['succ'][:, -1].sum())}/{N_STATES} "
              f"steps {r['steps'].tolist()}")

    # shot 1 episode: first initial state the low policy fails
    final_succ = roll_lo["succ"][:, -1]
    if (~final_succ).any():
        ep1 = int(np.argmax(~final_succ))
        rule_note = "first failed state"
    else:
        dists = state_dist(roll_lo["grip"][:, -1], roll_lo["block"][:, -1],
                           roll_lo["grip"][:, -1], expert_roll["goal"])
        ep1 = int(np.argmax(dists))
        rule_note = "largest final distance (no failures)"
    print(f"  shot 1 episode index: {ep1} ({rule_note})")

    sw = SegmentWriter(MEDIA_DIR)

    # shot 1: compounding error in the low policy
    fig, panel, counter = make_shot1_figure(e_lo, roll_lo, expert_roll,
                                            ghost_lo)
    b = BoundPanel(panel, roll_lo, expert_roll, ghost_lo)
    seg1 = int(roll_lo["steps"][ep1])
    sw.render("shot1", fig,
              lambda w: render_compound_shot(w, [b], [(ep1, seg1)], counter,
                                             SHOT1_FRAMES_PER_STEP, 1))

    # shot 2: the matched-MSE pair on the same states
    fig, bound, counter = make_shot2_figure(
        [e_lo, e_hi], [roll_lo, roll_hi], expert_roll,
        [ghost_lo, ghost_hi])
    episodes = [(ep, int(max(roll_lo["steps"][ep], roll_hi["steps"][ep])))
                for ep in range(N_STATES)]
    shot2_seg = sw.render(
        "shot2", fig,
        lambda w: render_compound_shot(w, bound, episodes, counter, 1,
                                       N_STATES))
    shot2_only = os.path.join(MEDIA_DIR, "_shot2_only.mp4")
    subprocess.run(["cp", shot2_seg, shot2_only], check=True)

    # closing card: the scatter
    sw.render_static("card", make_scatter_card(fits, band, lo, hi),
                     CARD_SECONDS * FPS)

    out_mp4 = os.path.join(MEDIA_DIR, "compounding_error_video.mp4")
    sw.concat(out_mp4)
    print("wrote", out_mp4)

    out_gif = os.path.join(MEDIA_DIR, "matched_mse.gif")
    make_gif(shot2_only, out_gif)
    os.remove(shot2_only)
    print("wrote", out_gif)


if __name__ == "__main__":
    main()

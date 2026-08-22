"""Render a single Push2D expert episode to results/demo.gif.

Standalone visualization script, not part of the experiment pipeline.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from env2d import CONTACT_R, GOAL_R, WS_LO, WS_HI, Push2DVec, expert_action

SEED = 35
OUT = "results/demo.gif"

env = Push2DVec(1, seed=SEED)
obs = env.reset()

grip_hist = [env.grip[0].copy()]
block_hist = [env.block[0].copy()]
goal = env.goal[0].copy()

for _ in range(200):
    act = expert_action(obs)
    obs, succ, done = env.step(act)
    grip_hist.append(env.grip[0].copy())
    block_hist.append(env.block[0].copy())
    if done[0]:
        break

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.set_xlim(WS_LO - 0.03, WS_HI + 0.03)
ax.set_ylim(WS_LO - 0.03, WS_HI + 0.03)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_color("#30363d")

goal_ring = Circle(goal, GOAL_R, facecolor="none", edgecolor="#3fb950", linewidth=2, linestyle="--")
ax.add_patch(goal_ring)
ax.plot(*goal, marker="*", color="#3fb950", markersize=14, zorder=5)

block_patch = Circle(block_hist[0], CONTACT_R, facecolor="#f0883e", edgecolor="#f0883e", zorder=4)
ax.add_patch(block_patch)
grip_dot, = ax.plot([], [], marker="o", color="#58a6ff", markersize=10, zorder=6)
grip_trail, = ax.plot([], [], color="#58a6ff", linewidth=1, alpha=0.5)
block_trail, = ax.plot([], [], color="#f0883e", linewidth=1, alpha=0.35)
status = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="#c9d1d9",
                  fontsize=11, va="top", family="monospace")

title = "Push2D: expert orbit-then-push"
ax.set_title(title, color="#c9d1d9", fontsize=12, family="monospace")

N = len(grip_hist)
HOLD = 12  # extra frames to hold on the final (success) state


def update(i):
    i = min(i, N - 1)
    gx, gy = zip(*grip_hist[: i + 1])
    bx, by = zip(*block_hist[: i + 1])
    grip_dot.set_data([gx[-1]], [gy[-1]])
    grip_trail.set_data(gx, gy)
    block_trail.set_data(bx, by)
    block_patch.center = (bx[-1], by[-1])
    ok = succ[0] if i == N - 1 else False
    status.set_text(f"t={i:3d}  {'SUCCESS' if ok else ''}")
    return grip_dot, grip_trail, block_trail, block_patch, status


anim = FuncAnimation(fig, update, frames=N + HOLD, interval=45, blit=False)
anim.save(OUT, writer=PillowWriter(fps=22))
print(f"wrote {OUT}, {N} steps, success={bool(succ[0])}")

"""
Push2D: a minimal, fully vectorized 2D block-pushing environment.

Why this task and not a pure reach task: pushing has a *staging* requirement.
The gripper must first get behind the block (on the far side from the goal)
and only then push. That makes the task multi-stage and makes compounding
error meaningful: a policy that drifts during the approach ends up on the
wrong side of the block and shoves it *away* from the goal, which is
unrecoverable within the episode. That is precisely the kind of failure that
open-loop action-prediction error is bad at surfacing.

Everything is batched over `B` parallel episodes so that generating 1000
demos or running 200 eval rollouts is a handful of numpy ops.
"""

import numpy as np

WS_LO, WS_HI = 0.05, 0.95      # workspace bounds
STEP = 0.04                    # max gripper displacement per timestep
CONTACT_R = 0.06               # gripper/block contact radius
GOAL_R = 0.05                  # success threshold (block centre to goal)
MAX_STEPS = 110

OBS_DIM = 10
ACT_DIM = 2


def _norm(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


class Push2DVec:
    """Batched Push2D. All state arrays are (B, 2)."""

    def __init__(self, batch, seed=0):
        self.B = batch
        self.rng = np.random.default_rng(seed)

    def reset(self):
        B = self.B
        self.block = self.rng.uniform(0.20, 0.80, size=(B, 2))
        # goal at a workable distance from the block, kept inside the workspace
        ang = self.rng.uniform(0, 2 * np.pi, size=B)
        dist = self.rng.uniform(0.25, 0.45, size=B)
        goal = self.block + np.stack([np.cos(ang), np.sin(ang)], axis=-1) * dist[:, None]
        self.goal = np.clip(goal, WS_LO + 0.05, WS_HI - 0.05)
        # gripper anywhere not already touching the block
        g = self.rng.uniform(WS_LO, WS_HI, size=(B, 2))
        bad = np.linalg.norm(g - self.block, axis=-1) < CONTACT_R + 0.06
        while bad.any():
            g[bad] = self.rng.uniform(WS_LO, WS_HI, size=(bad.sum(), 2))
            bad = np.linalg.norm(g - self.block, axis=-1) < CONTACT_R + 0.06
        self.grip = g
        self.t = 0
        self.done = np.zeros(B, dtype=bool)
        self.success = np.zeros(B, dtype=bool)
        return self.obs()

    def obs(self):
        return np.concatenate(
            [self.grip, self.block, self.goal,
             self.block - self.grip, self.goal - self.block], axis=-1
        ).astype(np.float32)

    def step(self, action):
        """action: (B, 2) in [-1, 1]. Frozen episodes (done) stop updating."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        live = ~self.done

        g_new = np.clip(self.grip + a * STEP, WS_LO, WS_HI)
        delta = self.block - g_new
        d = np.linalg.norm(delta, axis=-1, keepdims=True)
        contact = (d < CONTACT_R).squeeze(-1) & live
        # rigid push: block is displaced to sit exactly at contact radius
        dirn = np.where(d > 1e-6, delta / np.maximum(d, 1e-9), _norm(a))
        b_new = np.where(contact[:, None], g_new + dirn * CONTACT_R, self.block)
        b_new = np.clip(b_new, WS_LO, WS_HI)

        self.grip = np.where(live[:, None], g_new, self.grip)
        self.block = np.where(live[:, None], b_new, self.block)

        self.t += 1
        reached = np.linalg.norm(self.block - self.goal, axis=-1) < GOAL_R
        self.success |= reached & live
        self.done |= reached
        if self.t >= MAX_STEPS:
            self.done[:] = True
        return self.obs(), self.success.copy(), self.done.copy()

    def dist_to_goal(self):
        return np.linalg.norm(self.block - self.goal, axis=-1)


# ----------------------------------------------------------------------------
# Scripted experts
# ----------------------------------------------------------------------------

STAGE_R = CONTACT_R * 1.25     # where the gripper stages before pushing
SAFE_R = CONTACT_R + 0.04      # orbit radius while circling the block
ALIGN_TOL = 0.20               # radians


def _rot90(v, sign):
    """Rotate (B,2) by +/-90 degrees; sign is (B,) of +1/-1."""
    out = np.stack([-v[:, 1], v[:, 0]], axis=-1)
    return out * sign[:, None]


def expert_action(obs, orbit_mode="short", safe_r=SAFE_R):
    """Primary scripted expert: orbit around the block to the staging point
    behind it, then push straight toward the goal.

    orbit_mode="short" takes the shorter way around the block.
    orbit_mode="ccw"  always circles counter-clockwise -- a *different but
    equally valid* strategy, used as the alternate expert for the
    inconsistent-strategy corruption.
    """
    grip, block, goal = obs[:, 0:2], obs[:, 2:4], obs[:, 4:6]
    u = _norm(block - goal)                      # unit vector pointing "behind" the block
    stage = block + u * STAGE_R
    r = grip - block
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    rhat = r / np.maximum(rn, 1e-9)

    cos = np.clip((rhat * u).sum(-1), -1, 1)
    angle_err = np.arccos(cos)
    cross = rhat[:, 0] * u[:, 1] - rhat[:, 1] * u[:, 0]
    if orbit_mode == "short":
        sign = np.where(cross >= 0, 1.0, -1.0)
    else:
        sign = np.ones_like(cross)

    # far away and misaligned -> head for a point outside the orbit ring
    outer = block + u * (safe_r + 0.10)
    far = (rn.squeeze(-1) > safe_r * 1.8)

    tang = _rot90(rhat, sign)
    radial = rhat * np.clip((safe_r - rn) * 8.0, -1, 1)
    orbit = _norm(tang + radial)

    to_outer = _norm(outer - grip)
    to_stage = _norm(stage - grip)
    push = _norm(goal - block)

    aligned = angle_err < ALIGN_TOL
    staged = rn.squeeze(-1) <= STAGE_R + 0.02

    action = np.where(far[:, None], to_outer, orbit)
    action = np.where((aligned & ~staged)[:, None], to_stage, action)
    action = np.where((aligned & staged)[:, None], push, action)
    return np.clip(action, -1, 1)


def alt_expert_action(obs):
    """Valid alternative strategy: always circle the same rotational direction
    and stage from a wider radius. Reaches the goal just as reliably, but
    traverses a visibly different region of state space."""
    return expert_action(obs, orbit_mode="ccw", safe_r=CONTACT_R + 0.03)


def naive_expert_action(obs):
    """Deliberately WRONG controller: charge straight at the block and keep
    shoving. It never stages and never reasons about the goal, so the block
    travels along whatever ray the gripper happened to start on. It succeeds
    only when that ray happens to pass through the goal -- which is exactly
    what an 'accidental success' demonstration is: right outcome, wrong
    behaviour, and it gets logged as a success either way."""
    grip, block, goal = obs[:, 0:2], obs[:, 2:4], obs[:, 4:6]
    del goal  # deliberately unused: this controller never reasons about the goal
    return np.clip(_norm(block - grip), -1, 1)

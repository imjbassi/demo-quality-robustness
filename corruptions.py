"""
Demonstration generation and the five corruption modes.

Two deliberate departures from a naive implementation, both of which matter:

1. Action-level corruptions are RE-SIMULATED, not pasted onto a clean
   trajectory. If you overwrite actions in a logged episode without rerunning
   the dynamics, the observations no longer follow from the actions, and you
   have built a physically impossible trajectory. That is a different (and
   much weaker) experiment: it tests label noise, not bad demonstrations.
   Every corruption here that touches actions replays the episode.

2. The sweep axis is CONTAMINATION RATE, not per-episode severity. Every
   config trains on the same number of episodes; rho of them are bad and
   (1 - rho) are clean. Per-episode severity is held fixed at the constants
   below. This makes all five modes comparable on one axis and answers the
   question a data pipeline actually asks: "what fraction of my dataset can
   be this kind of bad before the policy breaks?"

Corrupted episodes are only admitted to the dataset if they would have passed
a success-based QC filter, exactly as they would in a real pipeline. The
exception is truncation, which is what a QC filter failing to catch a
prematurely-ended episode looks like.
"""

import numpy as np

from env2d import (MAX_STEPS, Push2DVec, alt_expert_action, expert_action,
                   naive_expert_action)

# fixed per-episode severity
OCCLUSION_FRAC = 0.35      # fraction of timesteps with a blocked camera
FLAIL_STEPS = 12           # prefix timesteps of corrective flailing
FLAIL_SCALE = 0.9          # jitter magnitude added to expert action
TRUNC_KEEP = 0.70          # fraction of the episode retained


def rollout(policy_fn, n, seed, jitter_steps=0, jitter_scale=0.0, rng=None):
    """Roll out `n` episodes in parallel and return per-episode arrays."""
    env = Push2DVec(n, seed=seed)
    obs = env.reset()
    rng = rng or np.random.default_rng(seed + 7919)
    obs_log, act_log, live_log = [], [], []
    for t in range(MAX_STEPS):
        act = policy_fn(obs)
        if t < jitter_steps:
            act = np.clip(act + rng.uniform(-jitter_scale, jitter_scale, act.shape), -1, 1)
        obs_log.append(obs.copy())
        act_log.append(act.copy())
        live_log.append(~env.done.copy())
        obs, succ, done = env.step(act)
        if done.all():
            break
    O = np.stack(obs_log, 1)      # (n, T, obs_dim)
    A = np.stack(act_log, 1)
    L = np.stack(live_log, 1)     # (n, T) True while the episode was running
    eps = []
    for i in range(n):
        k = L[i].sum()
        eps.append({"obs": O[i, :k].astype(np.float32),
                    "act": A[i, :k].astype(np.float32),
                    "success": bool(succ[i])})
    return eps


def _collect_successes(policy_fn, n_wanted, seed, **kw):
    """Rejection-sample until we have n_wanted *successful* episodes."""
    out, s = [], seed
    while len(out) < n_wanted:
        batch = rollout(policy_fn, max(256, n_wanted), seed=s, **kw)
        out.extend([e for e in batch if e["success"]])
        s += 1
        if s - seed > 400:
            raise RuntimeError("controller succeeds too rarely to sample")
    return out[:n_wanted]


# --------------------------------------------------------------------------
# corruption modes: each returns n corrupted episodes
# --------------------------------------------------------------------------

def make_clean(n, seed):
    return _collect_successes(expert_action, n, seed)


def make_occlusion(n, seed):
    """Blocked camera. The demonstrator can still see, so the ACTIONS stay
    expert-quality; the logged OBSERVATIONS go stale for a contiguous window
    (last visible frame is held). The result is a set of transitions whose
    inputs are wrong but whose labels look perfect."""
    eps = make_clean(n, seed)
    rng = np.random.default_rng(seed + 101)
    out = []
    for e in eps:
        T = len(e["obs"])
        w = max(1, int(T * OCCLUSION_FRAC))
        if T - w <= 1:
            out.append(e)
            continue
        s = rng.integers(0, T - w)
        obs = e["obs"].copy()
        obs[s:s + w] = obs[max(s - 1, 0)]          # hold last visible frame
        out.append({"obs": obs, "act": e["act"], "success": True})
    return out


def make_accidental_success(n, seed):
    """A controller that charges blindly at the block, keeping only the runs
    where the block happened to land on the goal. Right outcome, no correct
    behaviour anywhere in the episode."""
    return _collect_successes(naive_expert_action, n, seed + 202)


def make_flailing(n, seed):
    """Corrective flailing: heavy jitter on the opening timesteps, then the
    demonstrator settles into the correct motion and still succeeds.
    Re-simulated, so the wobble genuinely happened."""
    return _collect_successes(expert_action, n, seed + 303,
                              jitter_steps=FLAIL_STEPS, jitter_scale=FLAIL_SCALE)


def make_truncation(n, seed):
    """Episode cut off before the block reaches the goal, but logged anyway.
    This is the one mode that a success-based QC filter is supposed to catch;
    including it measures what happens when the filter misses."""
    eps = make_clean(n, seed + 404)
    out = []
    for e in eps:
        k = max(2, int(len(e["obs"]) * TRUNC_KEEP))
        out.append({"obs": e["obs"][:k], "act": e["act"][:k], "success": False})
    return out


def make_inconsistent_strategy(n, seed):
    """A second, equally valid strategy: always circle the block the same
    rotational direction from a wider radius. Every episode is a genuine
    success -- the only problem is that it disagrees with the primary
    strategy about what to do in the same state."""
    return _collect_successes(alt_expert_action, n, seed + 505)


CORRUPTIONS = {
    "occlusion": make_occlusion,
    "accidental_success": make_accidental_success,
    "flailing": make_flailing,
    "truncation": make_truncation,
    "inconsistent_strategy": make_inconsistent_strategy,
}


def build_dataset(corruption, rho, n_episodes, seed):
    """(1 - rho) clean episodes + rho corrupted ones. Size held constant."""
    n_bad = int(round(n_episodes * rho))
    n_good = n_episodes - n_bad
    eps = make_clean(n_good, seed)
    if n_bad > 0:
        eps = eps + CORRUPTIONS[corruption](n_bad, seed + 1000)
    return eps


def flatten(eps):
    """Episode list -> (N, obs_dim), (N, act_dim) transition arrays."""
    O = np.concatenate([e["obs"] for e in eps], 0)
    A = np.concatenate([e["act"] for e in eps], 0)
    return O, A

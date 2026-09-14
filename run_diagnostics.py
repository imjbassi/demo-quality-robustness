"""Generate label-fidelity and alternate-strategy clonability diagnostics."""

import csv
from pathlib import Path

import numpy as np
import torch

import corruptions as C
import policy as P
from env2d import (ALIGN_TOL, CONTACT_R, MAX_STEPS, STAGE_R, Push2DVec,
                   alt_expert_action, expert_action)


N_EPISODES = 400
SEEDS = list(range(10))
HELDOUT_SEED = 99999
EVAL_SEED_BASE = 12345
RESULTS_DIR = Path("results")


class AlternateOracle:
    """Torch-compatible adapter for the state-only alternate controller."""

    def __call__(self, obs):
        action = alt_expert_action(obs.detach().cpu().numpy())
        return torch.tensor(action, dtype=torch.float32)


def alternate_phase(obs):
    """Return the active branch of the state-only alternate controller."""
    grip, block, goal = obs[:, 0:2], obs[:, 2:4], obs[:, 4:6]
    behind = block - goal
    behind /= np.maximum(np.linalg.norm(behind, axis=1, keepdims=True), 1e-9)
    radius = grip - block
    radius_norm = np.linalg.norm(radius, axis=1)
    radius_unit = radius / np.maximum(radius_norm[:, None], 1e-9)
    angle_error = np.arccos(np.clip(np.sum(radius_unit * behind, axis=1), -1, 1))
    aligned = angle_error < ALIGN_TOL
    staged = radius_norm <= STAGE_R + 0.02
    far = radius_norm > (CONTACT_R + 0.03) * 1.8
    phase = np.full(len(obs), "orbit", dtype=object)
    phase[far] = "approach_outer"
    phase[aligned & ~staged] = "approach_stage"
    phase[aligned & staged] = "push"
    return phase


def mse_by_phase(predicted, target, obs):
    per_state = np.mean((predicted - target) ** 2, axis=1)
    rows = []
    phases = alternate_phase(obs)
    for phase in ("approach_outer", "orbit", "approach_stage", "push"):
        selected = phases == phase
        if selected.any():
            rows.append((phase, int(selected.sum()), float(per_state[selected].mean())))
    return rows


def on_policy_error(cloned, seed):
    """Measure clone-oracle disagreement on states induced by the clone."""
    env = Push2DVec(200, seed=seed)
    obs = env.reset()
    time_rows = []
    phase_sse = {}
    totals = {"all": [0.0, 0], "early": [0.0, 0], "late": [0.0, 0]}
    for step in range(MAX_STEPS):
        live = ~env.done.copy()
        predicted = cloned(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
        target = alt_expert_action(obs)
        per_state = np.mean((predicted - target) ** 2, axis=1)
        live_error = per_state[live]
        time_rows.append(
            {"step": step, "n_live": int(live.sum()),
             "mse": float(live_error.mean())}
        )
        for key in ("all", "early" if step < 20 else "late" if step >= 90 else None):
            if key is not None:
                totals[key][0] += float(live_error.sum())
                totals[key][1] += len(live_error)
        phases = alternate_phase(obs)
        for phase in ("approach_outer", "orbit", "approach_stage", "push"):
            selected = live & (phases == phase)
            if selected.any():
                sse, count = phase_sse.setdefault(phase, [0.0, 0])
                phase_sse[phase] = [sse + float(per_state[selected].sum()),
                                    count + int(selected.sum())]
        obs, _, done = env.step(np.clip(predicted, -1, 1))
        if done.all():
            break
    phase_rows = [
        {"phase": phase, "n_states": count, "mse": sse / count}
        for phase, (sse, count) in phase_sse.items()
    ]
    summary = {key: sse / count for key, (sse, count) in totals.items()}
    return summary, time_rows, phase_rows


def label_fidelity_rows():
    """MSE between logged actions and the primary expert on logged states."""
    rows = []
    for corruption in ["clean", *C.CORRUPTIONS]:
        for seed in SEEDS:
            data_seed = seed * 7717
            if corruption == "clean":
                episodes = C.make_clean(N_EPISODES, data_seed)
            else:
                episodes = C.build_dataset(
                    corruption, rho=1.0, n_episodes=N_EPISODES, seed=data_seed
                )
            obs, logged_action = C.flatten(episodes)
            primary_action = expert_action(obs)
            mse = float(np.mean((logged_action - primary_action) ** 2))
            rows.append(
                {
                    "corruption": corruption,
                    "seed": seed,
                    "n_transitions": len(obs),
                    "label_fidelity_mse": round(mse, 8),
                }
            )
    return rows


def alternate_clonability_rows():
    """Compare a cloned alternate expert with its state-only oracle."""
    heldout = C.make_inconsistent_strategy(300, HELDOUT_SEED)
    heldout_obs, heldout_action = C.flatten(heldout)
    oracle = AlternateOracle()
    rows, time_rows, phase_rows = [], [], []
    for seed in SEEDS:
        episodes = C.build_dataset(
            "inconsistent_strategy", rho=1.0, n_episodes=N_EPISODES,
            seed=seed * 7717
        )
        obs, action = C.flatten(episodes)
        cloned = P.train_bc(obs, action, seed=seed, epochs=50)
        eval_seed = EVAL_SEED_BASE + seed
        clone_success, _ = P.eval_closed_loop(cloned, 200, eval_seed)
        oracle_success, _ = P.eval_closed_loop(oracle, 200, eval_seed)
        on_policy, seed_time_rows, seed_phase_rows = on_policy_error(
            cloned, eval_seed
        )
        predicted_heldout = cloned(
            torch.tensor(heldout_obs, dtype=torch.float32)
        ).detach().numpy()
        heldout_phase_rows = mse_by_phase(
            predicted_heldout, heldout_action, heldout_obs
        )
        rows.append(
            {
                "seed": seed,
                "eval_seed": eval_seed,
                "n_transitions": len(obs),
                "train_mse_alt": round(P.eval_open_loop(cloned, obs, action), 8),
                "heldout_mse_alt": round(
                    P.eval_open_loop(cloned, heldout_obs, heldout_action), 8
                ),
                "on_policy_mse_alt": round(on_policy["all"], 8),
                "on_policy_mse_steps_0_19": round(on_policy["early"], 8),
                "on_policy_mse_steps_90_109": round(on_policy["late"], 8),
                "clone_success": round(clone_success, 4),
                "oracle_success": round(oracle_success, 4),
            }
        )
        for row in seed_time_rows:
            time_rows.append({"seed": seed, **row})
        for distribution, values in (
            ("on_policy", [(r["phase"], r["n_states"], r["mse"])
                           for r in seed_phase_rows]),
            ("heldout_expert", heldout_phase_rows),
        ):
            for phase, n_states, mse in values:
                phase_rows.append(
                    {"seed": seed, "distribution": distribution,
                     "phase": phase, "n_states": n_states,
                     "mse": round(mse, 8)}
                )
    return rows, time_rows, phase_rows


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    write_rows(RESULTS_DIR / "label_fidelity.csv", label_fidelity_rows())
    alternate, by_step, by_phase = alternate_clonability_rows()
    write_rows(RESULTS_DIR / "alternate_clonability.csv", alternate)
    write_rows(RESULTS_DIR / "on_policy_error_by_step.csv", by_step)
    write_rows(RESULTS_DIR / "on_policy_error_by_phase.csv", by_phase)


if __name__ == "__main__":
    main()

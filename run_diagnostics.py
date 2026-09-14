"""Generate label-fidelity and alternate-strategy clonability diagnostics."""

import csv
from pathlib import Path

import numpy as np
import torch

import corruptions as C
import policy as P
from env2d import alt_expert_action, expert_action


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
    rows = []
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
        rows.append(
            {
                "seed": seed,
                "eval_seed": eval_seed,
                "n_transitions": len(obs),
                "train_mse_alt": round(P.eval_open_loop(cloned, obs, action), 8),
                "heldout_mse_alt": round(
                    P.eval_open_loop(cloned, heldout_obs, heldout_action), 8
                ),
                "clone_success": round(clone_success, 4),
                "oracle_success": round(oracle_success, 4),
            }
        )
    return rows


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    write_rows(RESULTS_DIR / "label_fidelity.csv", label_fidelity_rows())
    write_rows(
        RESULTS_DIR / "alternate_clonability.csv", alternate_clonability_rows()
    )


if __name__ == "__main__":
    main()

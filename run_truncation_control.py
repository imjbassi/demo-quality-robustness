"""Control full truncation for the number of training transitions."""

import csv
from pathlib import Path

import corruptions as C
import policy as P


N_CLEAN_EPISODES = 400
POOL_EPISODES = 800
SEEDS = list(range(10))
HELDOUT_SEED = 99999
EVAL_SEED_BASE = 12345
OUT = Path("results/truncation_transition_control.csv")


def closest_prefix(episodes, target_transitions):
    """Choose a whole-episode prefix closest to the requested sample count."""
    cumulative = 0
    best_n = 0
    best_gap = target_transitions
    for index, episode in enumerate(episodes, start=1):
        cumulative += len(episode["obs"])
        gap = abs(cumulative - target_transitions)
        if gap < best_gap:
            best_n, best_gap = index, gap
        if cumulative > target_transitions and gap > best_gap:
            break
    return episodes[:best_n]


def main():
    holdout = C.make_clean(300, seed=HELDOUT_SEED)
    heldout_obs, heldout_action = C.flatten(holdout)
    rows = []
    for seed in SEEDS:
        data_seed = seed * 7717
        clean = C.make_clean(N_CLEAN_EPISODES, data_seed)
        target_transitions = len(C.flatten(clean)[0])

        # This seed matches the full-truncation branch of build_dataset.
        pool = C.make_truncation(POOL_EPISODES, data_seed + 1000)
        truncated = closest_prefix(pool, target_transitions)
        obs, action = C.flatten(truncated)

        for policy_name, trainer in (
            ("mlp", P.train_bc),
            ("trees", P.train_tree_bc),
        ):
            cloned = trainer(obs, action, seed=seed, epochs=50)
            eval_seed = EVAL_SEED_BASE + seed
            success, final_dist = P.eval_closed_loop(cloned, 200, eval_seed)
            rows.append(
                {
                    "policy": policy_name,
                    "seed": seed,
                    "eval_seed": eval_seed,
                    "n_episodes": len(truncated),
                    "target_transitions": target_transitions,
                    "n_transitions": len(obs),
                    "transition_gap": len(obs) - target_transitions,
                    "open_loop_mse": round(
                        P.eval_open_loop(cloned, heldout_obs, heldout_action), 8
                    ),
                    "closed_loop_success": round(success, 4),
                    "final_dist": round(final_dist, 6),
                }
            )
            print(
                f"seed={seed} policy={policy_name} episodes={len(truncated)} "
                f"transitions={len(obs)}/{target_transitions} "
                f"success={success:.3f}",
                flush=True,
            )

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

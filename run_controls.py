"""Run transition-matched policy controls.

The main sweep defines contamination by episode. Alternate-strategy episodes
are longer, so this control instead fixes both the total number of transitions
and the fraction contributed by each strategy. It trains both the original
MLP and an extremely-randomized-tree regression policy.
"""

import csv
import argparse
import time

import numpy as np

import corruptions as C
import policy as P
from run_experiment import EPOCHS, EVAL_SEED_BASE, HELDOUT_SEED, N_EVAL, RHOS, SEEDS

N_TRANSITIONS = 13_000
POOL_EPISODES = 500
OUT = "results/control_results.csv"


def build_transition_matched(rho, clean, alternate, seed):
    """Return exactly N_TRANSITIONS with rho defined at transition level."""
    clean_obs, clean_act = clean
    alt_obs, alt_act = alternate
    n_alt = int(round(N_TRANSITIONS * rho))
    n_clean = N_TRANSITIONS - n_alt
    if len(clean_obs) < n_clean or len(alt_obs) < n_alt:
        raise ValueError("demonstration pool is too small for requested mixture")
    rng_clean = np.random.default_rng(seed + 11)
    rng_alt = np.random.default_rng(seed + 12)
    clean_idx = rng_clean.choice(len(clean_obs), size=n_clean, replace=False)
    alt_idx = rng_alt.choice(len(alt_obs), size=n_alt, replace=False)
    obs = np.concatenate([clean_obs[clean_idx], alt_obs[alt_idx]])
    act = np.concatenate([clean_act[clean_idx], alt_act[alt_idx]])
    perm = np.random.default_rng(seed + 13).permutation(N_TRANSITIONS)
    return obs[perm], act[perm]


def main(seeds=SEEDS, out=OUT):
    holdout = C.make_clean(300, seed=HELDOUT_SEED)
    holdout_obs, holdout_act = C.flatten(holdout)
    rows = []
    total = len(seeds) * len(RHOS) * 2
    job = 0
    start = time.time()

    for seed in seeds:
        data_seed = seed * 7717
        clean = C.flatten(C.make_clean(POOL_EPISODES, data_seed))
        alternate = C.flatten(C.make_inconsistent_strategy(POOL_EPISODES,
                                                            data_seed + 1000))
        for rho in RHOS:
            obs, act = build_transition_matched(
                rho, clean, alternate, seed=seed * 100_003 + int(rho * 10_000)
            )
            for policy_name, trainer in (
                ("mlp", P.train_bc),
                ("trees", P.train_tree_bc),
            ):
                job += 1
                model = trainer(obs, act, seed=seed, epochs=EPOCHS)
                mse = P.eval_open_loop(model, holdout_obs, holdout_act)
                eval_seed = EVAL_SEED_BASE + seed
                success, distance = P.eval_closed_loop(
                    model, n_episodes=N_EVAL, seed=eval_seed
                )
                rows.append({
                    "policy": policy_name,
                    "rho": rho,
                    "seed": seed,
                    "eval_seed": eval_seed,
                    "n_transitions": len(obs),
                    "open_loop_mse": round(mse, 6),
                    "closed_loop_success": round(success, 4),
                    "final_dist": round(distance, 4),
                })
                print(
                    f"[{job:3d}/{total}] {policy_name:3s} rho={rho:.2f} "
                    f"seed={seed} mse={mse:.4f} success={success:.3f} "
                    f"({time.time() - start:.0f}s)",
                    flush=True,
                )

    with open(out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, SEEDS)),
        help="comma-separated training seeds (supports parallel shards)",
    )
    args = parser.parse_args()
    main(seeds=[int(value) for value in args.seeds.split(",")], out=args.out)

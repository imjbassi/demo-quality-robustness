"""Run the full grid: 5 corruption modes x 5 contamination rates x 10 seeds.

Everything except the training data is held fixed -- same architecture, same
hyperparameters, same number of episodes, same evaluation initial conditions.
"""

import csv
import argparse
import sys
import time

import corruptions as C
import policy as P

N_EPISODES = 400          # dataset size, held constant across every config
EPOCHS = 50
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
SEEDS = list(range(10))
HELDOUT_SEED = 99999
EVAL_SEED_BASE = 12345
N_EVAL = 200
OUT = "results/results.csv"


def main(policy_name="mlp", out=OUT, seeds=SEEDS):
    trainers = {"mlp": P.train_bc, "trees": P.train_tree_bc}
    trainer = trainers[policy_name]
    holdout = C.make_clean(300, seed=HELDOUT_SEED)
    Oh, Ah = C.flatten(holdout)

    rows = []
    jobs = []
    for seed in seeds:
        jobs.append(("clean", 0.0, seed))
        for name in C.CORRUPTIONS:
            for rho in RHOS[1:]:
                jobs.append((name, rho, seed))

    t0 = time.time()
    for i, (name, rho, seed) in enumerate(jobs, 1):
        eps = C.build_dataset(name, rho, N_EPISODES, seed=seed * 7717)
        O, A = C.flatten(eps)
        pol = trainer(O, A, seed=seed, epochs=EPOCHS)
        mse = P.eval_open_loop(pol, Oh, Ah)
        # Vary evaluation states across training seeds while preserving paired
        # comparisons: every configuration for a given training seed sees the
        # same evaluation episodes.
        eval_seed = EVAL_SEED_BASE + seed
        succ, dist = P.eval_closed_loop(pol, n_episodes=N_EVAL, seed=eval_seed)
        wrong = P.eval_wrong_side_rate(pol, n_episodes=N_EVAL, seed=eval_seed)
        row = {"policy": policy_name,
               "corruption": name, "rho": rho, "seed": seed,
               "eval_seed": eval_seed,
               "n_transitions": len(O), "open_loop_mse": round(mse, 6),
               "closed_loop_success": round(succ, 4),
               "final_dist": round(dist, 4), "wrong_side_rate": round(wrong, 4)}
        rows.append(row)
        print(f"[{i:3d}/{len(jobs)}] {name:22s} rho={rho:.1f} s={seed} "
              f"mse={mse:.4f} succ={succ:.3f} wrong={wrong:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # also mirror rho=0 clean rows onto every corruption type for plotting
    for seed in seeds:
        base = [r for r in rows if r["corruption"] == "clean" and r["seed"] == seed][0]
        for name in C.CORRUPTIONS:
            rows.append({**base, "corruption": name})

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("mlp", "trees"), default="mlp")
    parser.add_argument("--out", default=OUT)
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, SEEDS)),
        help="comma-separated training seeds (supports parallel shards)",
    )
    args = parser.parse_args()
    selected_seeds = [int(value) for value in args.seeds.split(",")]
    sys.exit(main(policy_name=args.policy, out=args.out, seeds=selected_seeds))

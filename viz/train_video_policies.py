"""Train and checkpoint the policies used in the video, then verify that they
reproduce the committed results before anything is rendered.

This script is deliberately a thin wrapper around the research code: datasets
come from ``corruptions.build_dataset`` and training from ``policy.train_bc``
with the same constants as ``run_experiment.py`` (400 episodes, 50 epochs,
training seed 0, evaluation seed 12345). Nothing in the research code is
modified or re-implemented here.

For every policy it re-runs the repo's own closed-loop evaluation
(``policy.eval_closed_loop`` / ``policy.eval_wrong_side_rate``) and compares
the numbers against the matching seed-0 row committed in
``results/results.csv``. If any retrained policy is off by more than
``TOLERANCE`` in success rate, the script exits non-zero and no manifest is
written, so the renderer refuses to run on unverified checkpoints.

Usage:  python viz/train_video_policies.py
Output: viz/checkpoints/*.pt and viz/checkpoints/manifest.json
"""

import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import corruptions as C
import policy as P

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "viz", "checkpoints")
RESULTS_CSV = os.path.join(REPO, "results", "results.csv")

# Same constants as run_experiment.py, training seed 0 (the first sweep seed,
# fixed here so the choice of which policy appears in the video is not ours).
N_EPISODES = 400
EPOCHS = 50
TRAIN_SEED = 0
EVAL_SEED = 12345          # EVAL_SEED_BASE + TRAIN_SEED
N_EVAL = 200
TOLERANCE = 0.05           # max |retrained - committed| closed-loop success

VIDEO_RHOS = [0.5, 0.75, 0.9, 1.0]
CONFIGS = [("clean", 0.0)]
CONFIGS += [("accidental_success", r) for r in VIDEO_RHOS]
CONFIGS += [("flailing", r) for r in VIDEO_RHOS]


def ckpt_name(corruption, rho):
    return f"{corruption}_rho{rho:g}.pt"


def committed_rows():
    """Seed-0 MLP rows of results.csv, keyed by (corruption, rho)."""
    out = {}
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["policy"] == "mlp" and int(r["seed"]) == TRAIN_SEED:
                out[(r["corruption"], float(r["rho"]))] = r
    return out


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    committed = committed_rows()
    entries, failures = [], []
    t0 = time.time()

    print(f"training {len(CONFIGS)} policies "
          f"(seed={TRAIN_SEED}, epochs={EPOCHS}, eval_seed={EVAL_SEED})")
    for i, (name, rho) in enumerate(CONFIGS, 1):
        eps = C.build_dataset(name, rho, N_EPISODES, seed=TRAIN_SEED * 7717)
        O, A = C.flatten(eps)
        pol = P.train_bc(O, A, seed=TRAIN_SEED, epochs=EPOCHS)
        succ, dist = P.eval_closed_loop(pol, n_episodes=N_EVAL, seed=EVAL_SEED)
        wrong = P.eval_wrong_side_rate(pol, n_episodes=N_EVAL, seed=EVAL_SEED)

        ref = committed[(name, rho)]
        ref_succ = float(ref["closed_loop_success"])
        ref_wrong = float(ref["wrong_side_rate"])
        diff = abs(succ - ref_succ)
        ok = diff <= TOLERANCE
        if not ok:
            failures.append((name, rho, succ, ref_succ))

        path = os.path.join(CKPT_DIR, ckpt_name(name, rho))
        torch.save(pol.state_dict(), path)
        entries.append({
            "corruption": name, "rho": rho, "checkpoint": ckpt_name(name, rho),
            "train_seed": TRAIN_SEED, "eval_seed": EVAL_SEED,
            "n_episodes": N_EPISODES, "epochs": EPOCHS,
            "n_transitions": int(len(O)),
            "closed_loop_success": round(succ, 4),
            "wrong_side_rate": round(wrong, 4),
            "final_dist": round(dist, 4),
            "committed_success": ref_succ,
            "committed_wrong_side_rate": ref_wrong,
            "abs_diff": round(diff, 4),
        })
        print(f"[{i}/{len(CONFIGS)}] {name:22s} rho={rho:<4g} "
              f"succ={succ:.3f} (committed {ref_succ:.3f}, "
              f"diff {diff:.3f} {'OK' if ok else 'MISMATCH'}) "
              f"wrong={wrong:.3f} (committed {ref_wrong:.3f}) "
              f"({time.time() - t0:.0f}s)", flush=True)

    if failures:
        print("\nREPRODUCTION FAILED -- refusing to write manifest.")
        for name, rho, succ, ref in failures:
            print(f"  {name} rho={rho}: retrained {succ:.3f} "
                  f"vs committed {ref:.3f} (tolerance {TOLERANCE})")
        return 1

    manifest = {
        "train_seed": TRAIN_SEED, "eval_seed": EVAL_SEED,
        "n_eval_episodes": N_EVAL, "tolerance": TOLERANCE,
        "policies": entries,
    }
    with open(os.path.join(CKPT_DIR, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nall {len(entries)} policies reproduce committed seed-0 results "
          f"within {TOLERANCE}; wrote viz/checkpoints/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

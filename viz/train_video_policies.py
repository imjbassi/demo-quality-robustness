"""Train and checkpoint the policies used in the video, then verify that they
reproduce the committed results before anything is rendered.

This script is deliberately a thin wrapper around the research code: datasets
come from ``corruptions.build_dataset`` and training from ``policy.train_bc``
with the same constants as ``run_experiment.py`` (400 episodes, 50 epochs,
training seed 0, evaluation seed 12345). Nothing in the research code is
modified or re-implemented here.

For every policy it re-runs the repo's own closed-loop evaluation
(``policy.eval_closed_loop`` / ``policy.eval_wrong_side_rate``) plus the
repo's open-loop MSE (``policy.eval_open_loop`` on the same held-out clean
set) and compares the numbers against the matching per-seed row committed
in ``results/results.csv``. If any retrained policy is off by more than
``TOLERANCE`` in success rate or ``MSE_TOLERANCE`` in open-loop MSE, the
script exits non-zero and no manifest is written, so the renderers refuse
to run on unverified checkpoints.

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

# Same constants as run_experiment.py. Training seed 0 (the first sweep
# seed) for the dose-response video; the compounding-error video's policy
# pair additionally needs seed 8 (picked by the stated band rule in
# make_video2.py, not by eye).
N_EPISODES = 400
EPOCHS = 50
EVAL_SEED_BASE = 12345     # eval seed = EVAL_SEED_BASE + train seed
N_EVAL = 200
TOLERANCE = 0.05           # max |retrained - committed| closed-loop success
MSE_TOLERANCE = 0.02       # max |retrained - committed| open-loop MSE
HELDOUT_SEED = 99999       # same held-out clean set as run_experiment.py

VIDEO_RHOS = [0.5, 0.75, 0.9, 1.0]
CONFIGS = [("clean", 0.0, 0)]
CONFIGS += [("accidental_success", r, 0) for r in VIDEO_RHOS]
CONFIGS += [("flailing", r, 0) for r in VIDEO_RHOS]
CONFIGS += [("accidental_success", 0.9, 8)]   # low side of the MSE-band pair


def ckpt_name(corruption, rho, seed):
    base = f"{corruption}_rho{rho:g}"
    return f"{base}.pt" if seed == 0 else f"{base}_seed{seed}.pt"


def committed_rows():
    """MLP rows of results.csv, keyed by (corruption, rho, seed)."""
    out = {}
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["policy"] == "mlp":
                out[(r["corruption"], float(r["rho"]), int(r["seed"]))] = r
    return out


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    committed = committed_rows()
    holdout = C.make_clean(300, seed=HELDOUT_SEED)
    Oh, Ah = C.flatten(holdout)
    entries, failures = [], []
    t0 = time.time()

    print(f"training {len(CONFIGS)} policies (epochs={EPOCHS})")
    for i, (name, rho, seed) in enumerate(CONFIGS, 1):
        eval_seed = EVAL_SEED_BASE + seed
        eps = C.build_dataset(name, rho, N_EPISODES, seed=seed * 7717)
        O, A = C.flatten(eps)
        pol = P.train_bc(O, A, seed=seed, epochs=EPOCHS)
        mse = P.eval_open_loop(pol, Oh, Ah)
        succ, dist = P.eval_closed_loop(pol, n_episodes=N_EVAL,
                                        seed=eval_seed)
        wrong = P.eval_wrong_side_rate(pol, n_episodes=N_EVAL,
                                       seed=eval_seed)

        ref = committed[(name if rho > 0 else "clean", rho, seed)]
        ref_succ = float(ref["closed_loop_success"])
        ref_wrong = float(ref["wrong_side_rate"])
        ref_mse = float(ref["open_loop_mse"])
        diff = abs(succ - ref_succ)
        mse_diff = abs(mse - ref_mse)
        ok = diff <= TOLERANCE and mse_diff <= MSE_TOLERANCE
        if not ok:
            failures.append((name, rho, seed, succ, ref_succ, mse, ref_mse))

        path = os.path.join(CKPT_DIR, ckpt_name(name, rho, seed))
        torch.save(pol.state_dict(), path)
        entries.append({
            "corruption": name, "rho": rho,
            "checkpoint": ckpt_name(name, rho, seed),
            "train_seed": seed, "eval_seed": eval_seed,
            "n_episodes": N_EPISODES, "epochs": EPOCHS,
            "n_transitions": int(len(O)),
            "open_loop_mse": round(mse, 6),
            "closed_loop_success": round(succ, 4),
            "wrong_side_rate": round(wrong, 4),
            "final_dist": round(dist, 4),
            "committed_success": ref_succ,
            "committed_mse": ref_mse,
            "committed_wrong_side_rate": ref_wrong,
            "abs_diff": round(diff, 4),
            "abs_mse_diff": round(mse_diff, 4),
        })
        print(f"[{i}/{len(CONFIGS)}] {name:22s} rho={rho:<4g} s={seed} "
              f"succ={succ:.3f} (committed {ref_succ:.3f}, "
              f"diff {diff:.3f}) "
              f"mse={mse:.3f} (committed {ref_mse:.3f}, "
              f"diff {mse_diff:.3f}) "
              f"{'OK' if ok else 'MISMATCH'} "
              f"({time.time() - t0:.0f}s)", flush=True)

    if failures:
        print("\nREPRODUCTION FAILED -- refusing to write manifest.")
        for name, rho, seed, succ, ref, mse, ref_mse in failures:
            print(f"  {name} rho={rho} seed={seed}: retrained "
                  f"succ {succ:.3f} vs {ref:.3f} (tol {TOLERANCE}), "
                  f"mse {mse:.3f} vs {ref_mse:.3f} (tol {MSE_TOLERANCE})")
        return 1

    manifest = {
        "eval_seed_base": EVAL_SEED_BASE,
        "n_eval_episodes": N_EVAL, "tolerance": TOLERANCE,
        "mse_tolerance": MSE_TOLERANCE,
        "policies": entries,
    }
    with open(os.path.join(CKPT_DIR, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nall {len(entries)} policies reproduce their committed rows "
          f"(success within {TOLERANCE}, MSE within {MSE_TOLERANCE}); "
          f"wrote viz/checkpoints/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

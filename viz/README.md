# viz/ — results videos

A self-contained rendering pipeline that turns the paper's headline results
into short videos of **real closed-loop rollouts**. Nothing here touches the
research code; `env2d.py`, `corruptions.py`, and `policy.py` are imported
unchanged. Shared drawing/recording helpers live in `common.py`.

## Video 1 — dose-response (`make_video.py`)

1. **Main shot (~25 s).** A 2×3 grid: five panels are behaviour-cloned
   policies trained on accidental-success contamination at
   ρ ∈ {0, 0.5, 0.75, 0.9, 1.0} (ρ = 0 is the clean baseline); the sixth
   cell holds the legend and a mini dose-response curve. Every panel plays
   the **same five initial states** (environment seed 42, fixed a priori —
   not chosen by scanning for dramatic failures), so any difference between
   panels comes only from the training data. Wrong-side pushes are
   highlighted in red using the repo's own predicate from
   `policy.eval_wrong_side_rate` (block ends farther than 0.02 from where it
   started, episode failed), evaluated on the rollout's current state.
2. **Contrast shot (~15 s).** The same layout for corrective flailing across
   the same ρ values: it stays flat while accidental success collapses.
3. **Closing card (6 s).** The dose-response curves from
   `results/results.csv` (10-seed means), with the cliff called out.

Panel labels are each policy's measured success rate over 200 evaluation
episodes, not numbers typed in by hand.

## Video 2 — compounding error (`make_video2.py`)

Why open-loop evaluation can look fine while closed-loop behaviour fails.

1. **Shot 1 (~14 s).** One panel with three trajectories from the same
   initial state: the scripted expert (gray), the **open-loop ghost**
   (translucent), and the closed-loop policy (solid). The ghost is the
   repo's own open-loop definition (`policy.eval_open_loop`: the policy
   queried on expert states) made spatial — at each step the policy acts
   once from the expert's exact recorded state, the one-step prediction is
   drawn, and the state resets to the expert's next state. Below, a live
   plot of the closed-loop state's distance from the expert next to the
   ghost's one-step error on the same axes. The shown state is the first
   one (lowest index, env seed 42) that the policy fails.
2. **Shot 2 (~20 s).** Two such panels side by side: the pair with the
   **largest closed-loop success gap** among the 49 runs whose open-loop
   MSE lies in [0.20, 0.30], subject to |ΔMSE| ≤ 0.01 and excluding
   inconsistent-strategy runs (whose low success the paper attributes to
   the alternate strategy's lower clonability). The rule is executed in
   code from the committed `results/results.csv`, not chosen by eye; it
   selects accidental success ρ=0.9, seeds 8 and 0: MSE 0.292 vs 0.300,
   success 0.530 vs 0.855. Same initial states in both panels, in lockstep.
3. **Closing card (7 s).** Open-loop MSE vs closed-loop success for all 260
   unique MLP fits, colored by failure mode (with distinct marker shapes),
   the matched band highlighted, the two shown policies ringed, and the
   paper's caveat about inconsistent-strategy runs on screen: excluding
   them, the band still spans 0.530–0.990.

## Reproduction gate

`train_video_policies.py` retrains the ten video policies with the exact
constants of `run_experiment.py` (400 episodes, 50 epochs, training seeds 0
and 8, evaluation seed 12345 + seed), re-runs the repo's closed-loop
evaluation **and** its open-loop MSE (`policy.eval_open_loop` on the same
held-out clean set, seed 99999), and compares each number to the matching
per-seed row committed in `results/results.csv`. The checkpoint manifest is
only written if every policy lands within 0.05 in success and 0.02 in MSE
of its committed values; both renderers refuse to run without the manifest.
Last verified run: maximum deviation 0.035 in success, 0.001 in MSE.

## Regenerating

```bash
pip install -r requirements.txt   # plus ffmpeg on PATH
make -C viz all                   # train checkpoints (~1 min CPU) + render
make -C viz video                 # one command: re-render both videos
```

Outputs: `viz/media/demo_quality_video.mp4` and
`viz/media/compounding_error_video.mp4` (1080p, 30 fps), plus the README
embeds `viz/media/main_shot.gif` and `viz/media/matched_mse.gif`.

## Note: a 3D arm version is out of scope

A MuJoCo (or similar) 7-DOF arm version of this study would be a **new
experiment**, not a port: it needs a contact-rich pushing task, a scripted
or teleoperated expert with a staging structure, re-simulated (not pasted)
corruptions, and its own sweep and seeds — roughly the full pipeline again
plus GPU training time. If built, its results must be reported on their own
terms and **must not reuse or be presented alongside the Push2D numbers** as
if they were comparable; the paper's thresholds are specific to this
environment and policy class.

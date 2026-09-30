# viz/ — results video

A self-contained rendering pipeline that turns the paper's headline result
into a short video of **real closed-loop rollouts**. Nothing here touches the
research code; `env2d.py`, `corruptions.py`, and `policy.py` are imported
unchanged.

## What the video shows

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

## Reproduction gate

`train_video_policies.py` retrains the nine video policies with the exact
constants of `run_experiment.py` (400 episodes, 50 epochs, training seed 0,
evaluation seed 12345), re-runs the repo's closed-loop evaluation, and
compares each success rate to the matching seed-0 row committed in
`results/results.csv`. The checkpoint manifest is only written if every
policy lands within 0.05 of its committed value; `make_video.py` refuses to
run without the manifest. Last verified run: maximum deviation **0.015**.

## Regenerating

```bash
pip install -r requirements.txt   # plus ffmpeg on PATH
make -C viz all                   # train checkpoints (~1 min CPU) + render
make -C viz video                 # one command: re-render from checkpoints
```

Outputs: `viz/media/demo_quality_video.mp4` (1080p, 30 fps) and
`viz/media/main_shot.gif` (README embed).

## Note: a 3D arm version is out of scope

A MuJoCo (or similar) 7-DOF arm version of this study would be a **new
experiment**, not a port: it needs a contact-rich pushing task, a scripted
or teleoperated expert with a staging structure, re-simulated (not pasted)
corruptions, and its own sweep and seeds — roughly the full pipeline again
plus GPU training time. If built, its results must be reported on their own
terms and **must not reuse or be presented alongside the Push2D numbers** as
if they were comparable; the paper's thresholds are specific to this
environment and policy class.

## Per-mode correlation between open-loop MSE and closed-loop success

| corruption mode | Pearson r | Spearman rho | success @ ρ=0.5 | success @ ρ=1.0 |
|---|---|---|---|---|
| Occlusion (stale obs) | -0.58 | -0.65 | 0.910 | 0.867 |
| Accidental success | -0.94 | -0.89 | 0.965 | 0.012 |
| Corrective flailing | -0.35 | -0.26 | 0.972 | 0.963 |
| Truncated episodes | -0.15 | +0.05 | 0.930 | 0.987 |
| Inconsistent strategy | -0.90 | -0.92 | 0.933 | 0.303 |

Pooled across all modes: Pearson r = -0.91, Spearman rho = -0.72

### Matched open-loop score, unmatched reality

Among the 15 runs whose open-loop MSE falls in [0.2, 0.3] — i.e. indistinguishable on the offline metric — closed-loop success ranges from **0.315 to 0.980** (spread of 0.665).

- Inconsistent strategy, ρ=0.9: MSE 0.294 → success 0.315
- Inconsistent strategy, ρ=0.9: MSE 0.299 → success 0.355
- Inconsistent strategy, ρ=0.9: MSE 0.283 → success 0.405
- Inconsistent strategy, ρ=0.75: MSE 0.247 → success 0.645
- Occlusion (stale obs), ρ=1.0: MSE 0.206 → success 0.735
- Inconsistent strategy, ρ=0.75: MSE 0.262 → success 0.735
- Accidental success, ρ=0.9: MSE 0.285 → success 0.770
- Accidental success, ρ=0.75: MSE 0.234 → success 0.785
- Inconsistent strategy, ρ=0.75: MSE 0.246 → success 0.805
- Inconsistent strategy, ρ=0.5: MSE 0.206 → success 0.905
- Inconsistent strategy, ρ=0.5: MSE 0.207 → success 0.915
- Accidental success, ρ=0.75: MSE 0.224 → success 0.935
- Accidental success, ρ=0.75: MSE 0.217 → success 0.950
- Occlusion (stale obs), ρ=1.0: MSE 0.213 → success 0.955
- Inconsistent strategy, ρ=0.5: MSE 0.204 → success 0.980

## Full aggregated table (mean over 3 seeds)

| corruption | ρ | open-loop MSE | closed-loop success | wrong-side push rate |
|---|---|---|---|---|
| Occlusion (stale obs) | 0.0 | 0.1231 | 0.972 ± 0.010 | 0.005 |
| Occlusion (stale obs) | 0.25 | 0.1337 | 0.925 ± 0.015 | 0.010 |
| Occlusion (stale obs) | 0.5 | 0.1579 | 0.910 ± 0.051 | 0.022 |
| Occlusion (stale obs) | 0.75 | 0.1734 | 0.912 ± 0.010 | 0.033 |
| Occlusion (stale obs) | 0.9 | 0.1802 | 0.897 ± 0.008 | 0.037 |
| Occlusion (stale obs) | 1.0 | 0.2049 | 0.867 ± 0.095 | 0.038 |
| Accidental success | 0.0 | 0.1231 | 0.972 ± 0.010 | 0.005 |
| Accidental success | 0.25 | 0.1384 | 0.958 ± 0.006 | 0.002 |
| Accidental success | 0.5 | 0.1689 | 0.965 ± 0.018 | 0.008 |
| Accidental success | 0.75 | 0.2249 | 0.890 ± 0.074 | 0.062 |
| Accidental success | 0.9 | 0.2996 | 0.747 ± 0.095 | 0.178 |
| Accidental success | 1.0 | 0.4536 | 0.012 ± 0.006 | 0.468 |
| Corrective flailing | 0.0 | 0.1231 | 0.972 ± 0.010 | 0.005 |
| Corrective flailing | 0.25 | 0.1261 | 0.983 ± 0.020 | 0.003 |
| Corrective flailing | 0.5 | 0.1283 | 0.972 ± 0.013 | 0.007 |
| Corrective flailing | 0.75 | 0.1440 | 0.960 ± 0.023 | 0.003 |
| Corrective flailing | 0.9 | 0.1313 | 0.963 ± 0.012 | 0.008 |
| Corrective flailing | 1.0 | 0.1351 | 0.963 ± 0.027 | 0.010 |
| Truncated episodes | 0.0 | 0.1231 | 0.972 ± 0.010 | 0.005 |
| Truncated episodes | 0.25 | 0.1249 | 0.983 ± 0.009 | 0.002 |
| Truncated episodes | 0.5 | 0.1356 | 0.930 ± 0.028 | 0.008 |
| Truncated episodes | 0.75 | 0.1437 | 0.958 ± 0.014 | 0.010 |
| Truncated episodes | 0.9 | 0.1429 | 0.960 ± 0.008 | 0.012 |
| Truncated episodes | 1.0 | 0.1442 | 0.987 ± 0.012 | 0.000 |
| Inconsistent strategy | 0.0 | 0.1231 | 0.972 ± 0.010 | 0.005 |
| Inconsistent strategy | 0.25 | 0.1659 | 0.972 ± 0.022 | 0.000 |
| Inconsistent strategy | 0.5 | 0.2057 | 0.933 ± 0.033 | 0.000 |
| Inconsistent strategy | 0.75 | 0.2519 | 0.728 ± 0.065 | 0.000 |
| Inconsistent strategy | 0.9 | 0.2917 | 0.358 ± 0.037 | 0.000 |
| Inconsistent strategy | 1.0 | 0.3214 | 0.303 ± 0.174 | 0.000 |

## Clonability control for inconsistent strategy

| ρ | observed success | interpolated (clonability-only) | excess harm from mixing |
|---|---|---|---|
| 0.0 | 0.972 | 0.972 | +0.000 |
| 0.25 | 0.972 | 0.805 | +0.167 |
| 0.5 | 0.933 | 0.637 | +0.296 |
| 0.75 | 0.728 | 0.470 | +0.258 |
| 0.9 | 0.358 | 0.370 | -0.012 |
| 1.0 | 0.303 | 0.303 | +0.000 |

Both experts solve the task on 100% of episodes, so the rho=1.0 result (0.303) is purely a statement about how hard the alternate strategy is to clone, not about inconsistency.


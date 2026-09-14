## Per-mode correlation between open-loop MSE and closed-loop success

| corruption mode | Pearson r | Spearman rho | success @ ρ=0.5 | success @ ρ=1.0 |
|---|---|---|---|---|
| Occlusion (stale obs) | -0.64 | -0.67 | 0.905 | 0.866 |
| Accidental success | -0.95 | -0.73 | 0.939 | 0.009 |
| Corrective flailing | -0.26 | +0.03 | 0.963 | 0.965 |
| Truncated episodes | -0.13 | -0.06 | 0.940 | 0.962 |
| Inconsistent strategy | -0.89 | -0.85 | 0.934 | 0.229 |

Pooled across all modes: Pearson r = -0.92, Spearman rho = -0.66

### Matched open-loop score, unmatched reality

Among the 49 runs whose open-loop MSE falls in [0.2, 0.3] — a descriptive similar-score band — closed-loop success ranges from **0.130 to 0.995** (spread of 0.865).

- Inconsistent strategy, ρ=0.9: MSE 0.285 → success 0.130
- Inconsistent strategy, ρ=0.9: MSE 0.299 → success 0.255
- Inconsistent strategy, ρ=0.9: MSE 0.288 → success 0.280
- Inconsistent strategy, ρ=0.75: MSE 0.264 → success 0.380
- Inconsistent strategy, ρ=0.9: MSE 0.293 → success 0.385
- Inconsistent strategy, ρ=0.9: MSE 0.287 → success 0.405
- Inconsistent strategy, ρ=0.9: MSE 0.287 → success 0.425
- Inconsistent strategy, ρ=0.9: MSE 0.281 → success 0.430
- Inconsistent strategy, ρ=0.9: MSE 0.299 → success 0.470
- Inconsistent strategy, ρ=0.75: MSE 0.262 → success 0.505
- Accidental success, ρ=0.9: MSE 0.292 → success 0.530
- Inconsistent strategy, ρ=0.75: MSE 0.256 → success 0.540
- Accidental success, ρ=0.9: MSE 0.297 → success 0.575
- Inconsistent strategy, ρ=0.9: MSE 0.279 → success 0.580
- Accidental success, ρ=0.9: MSE 0.286 → success 0.600
- Inconsistent strategy, ρ=0.75: MSE 0.246 → success 0.630
- Inconsistent strategy, ρ=0.75: MSE 0.246 → success 0.665
- Accidental success, ρ=0.9: MSE 0.298 → success 0.670
- Accidental success, ρ=0.9: MSE 0.294 → success 0.675
- Inconsistent strategy, ρ=0.75: MSE 0.254 → success 0.690
- Inconsistent strategy, ρ=0.75: MSE 0.263 → success 0.695
- Inconsistent strategy, ρ=0.75: MSE 0.248 → success 0.720
- Occlusion (stale obs), ρ=1.0: MSE 0.207 → success 0.765
- Accidental success, ρ=0.9: MSE 0.284 → success 0.765
- Accidental success, ρ=0.75: MSE 0.233 → success 0.770
- Accidental success, ρ=0.75: MSE 0.236 → success 0.780
- Accidental success, ρ=0.9: MSE 0.289 → success 0.790
- Inconsistent strategy, ρ=0.75: MSE 0.245 → success 0.800
- Inconsistent strategy, ρ=0.5: MSE 0.220 → success 0.825
- Inconsistent strategy, ρ=0.75: MSE 0.248 → success 0.830
- Accidental success, ρ=0.9: MSE 0.300 → success 0.855
- Inconsistent strategy, ρ=0.5: MSE 0.211 → success 0.875
- Inconsistent strategy, ρ=0.5: MSE 0.206 → success 0.880
- Inconsistent strategy, ρ=0.5: MSE 0.209 → success 0.895
- Accidental success, ρ=0.75: MSE 0.228 → success 0.895
- Inconsistent strategy, ρ=0.5: MSE 0.208 → success 0.925
- Accidental success, ρ=0.75: MSE 0.223 → success 0.940
- Accidental success, ρ=0.75: MSE 0.217 → success 0.950
- Occlusion (stale obs), ρ=1.0: MSE 0.212 → success 0.950
- Accidental success, ρ=0.75: MSE 0.215 → success 0.950
- Accidental success, ρ=0.75: MSE 0.221 → success 0.955
- Accidental success, ρ=0.75: MSE 0.220 → success 0.965
- Inconsistent strategy, ρ=0.5: MSE 0.205 → success 0.975
- Accidental success, ρ=0.75: MSE 0.209 → success 0.975
- Inconsistent strategy, ρ=0.5: MSE 0.205 → success 0.985
- Inconsistent strategy, ρ=0.5: MSE 0.206 → success 0.990
- Accidental success, ρ=0.75: MSE 0.209 → success 0.990
- Inconsistent strategy, ρ=0.5: MSE 0.203 → success 0.995
- Inconsistent strategy, ρ=0.5: MSE 0.202 → success 0.995

## Full aggregated table (mean and 95% t interval over 10 seeds)

| corruption | ρ | open-loop MSE | closed-loop success | wrong-side push rate |
|---|---|---|---|---|
| Occlusion (stale obs) | 0.0 | 0.1218 | 0.949 ± 0.015 | 0.005 |
| Occlusion (stale obs) | 0.25 | 0.1376 | 0.932 ± 0.023 | 0.008 |
| Occlusion (stale obs) | 0.5 | 0.1532 | 0.905 ± 0.032 | 0.021 |
| Occlusion (stale obs) | 0.75 | 0.1684 | 0.900 ± 0.031 | 0.017 |
| Occlusion (stale obs) | 0.9 | 0.1816 | 0.883 ± 0.015 | 0.040 |
| Occlusion (stale obs) | 1.0 | 0.1935 | 0.866 ± 0.038 | 0.041 |
| Accidental success | 0.0 | 0.1218 | 0.949 ± 0.015 | 0.005 |
| Accidental success | 0.25 | 0.1402 | 0.933 ± 0.026 | 0.011 |
| Accidental success | 0.5 | 0.1637 | 0.939 ± 0.034 | 0.011 |
| Accidental success | 0.75 | 0.2210 | 0.917 ± 0.056 | 0.033 |
| Accidental success | 0.9 | 0.2957 | 0.666 ± 0.078 | 0.223 |
| Accidental success | 1.0 | 0.4590 | 0.009 ± 0.008 | 0.415 |
| Corrective flailing | 0.0 | 0.1218 | 0.949 ± 0.015 | 0.005 |
| Corrective flailing | 0.25 | 0.1271 | 0.952 ± 0.038 | 0.006 |
| Corrective flailing | 0.5 | 0.1299 | 0.963 ± 0.017 | 0.005 |
| Corrective flailing | 0.75 | 0.1346 | 0.964 ± 0.016 | 0.006 |
| Corrective flailing | 0.9 | 0.1305 | 0.957 ± 0.021 | 0.008 |
| Corrective flailing | 1.0 | 0.1317 | 0.965 ± 0.015 | 0.004 |
| Truncated episodes | 0.0 | 0.1218 | 0.949 ± 0.015 | 0.005 |
| Truncated episodes | 0.25 | 0.1267 | 0.970 ± 0.013 | 0.003 |
| Truncated episodes | 0.5 | 0.1351 | 0.940 ± 0.036 | 0.003 |
| Truncated episodes | 0.75 | 0.1404 | 0.966 ± 0.013 | 0.005 |
| Truncated episodes | 0.9 | 0.1452 | 0.955 ± 0.017 | 0.012 |
| Truncated episodes | 1.0 | 0.1471 | 0.962 ± 0.017 | 0.005 |
| Inconsistent strategy | 0.0 | 0.1218 | 0.949 ± 0.015 | 0.005 |
| Inconsistent strategy | 0.25 | 0.1661 | 0.980 ± 0.009 | 0.000 |
| Inconsistent strategy | 0.5 | 0.2075 | 0.934 ± 0.044 | 0.000 |
| Inconsistent strategy | 0.75 | 0.2531 | 0.645 ± 0.098 | 0.000 |
| Inconsistent strategy | 0.9 | 0.2901 | 0.383 ± 0.092 | 0.000 |
| Inconsistent strategy | 1.0 | 0.3233 | 0.229 ± 0.102 | 0.000 |

## Paired full-contamination effects

| corruption | success at ρ=1 (95% CI) | paired change from clean (95% CI) |
|---|---|---|
| Occlusion (stale obs) | 0.866 ± 0.038 | -0.083 ± 0.045 |
| Accidental success | 0.009 ± 0.008 | -0.940 ± 0.020 |
| Corrective flailing | 0.965 ± 0.015 | +0.016 ± 0.022 |
| Truncated episodes | 0.962 ± 0.017 | +0.013 ± 0.017 |
| Inconsistent strategy | 0.229 ± 0.102 | -0.719 ± 0.099 |

## Policy-family robustness at full contamination

| corruption | MLP success | Extra Trees success |
|---|---|---|
| Occlusion (stale obs) | 0.866 ± 0.038 | 0.990 ± 0.006 |
| Accidental success | 0.009 ± 0.008 | 0.002 ± 0.003 |
| Corrective flailing | 0.965 ± 0.015 | 0.993 ± 0.007 |
| Truncated episodes | 0.962 ± 0.017 | 0.614 ± 0.033 |
| Inconsistent strategy | 0.229 ± 0.102 | 0.142 ± 0.050 |

## Transition-matched inconsistent-strategy control

| ρ | MLP success | Extra Trees success |
|---|---|---|
| 0.0 | 0.952 ± 0.022 | 0.996 ± 0.003 |
| 0.25 | 0.962 ± 0.018 | 0.984 ± 0.006 |
| 0.5 | 0.966 ± 0.015 | 0.888 ± 0.024 |
| 0.75 | 0.741 ± 0.096 | 0.797 ± 0.026 |
| 0.9 | 0.386 ± 0.099 | 0.563 ± 0.051 |
| 1.0 | 0.141 ± 0.064 | 0.226 ± 0.034 |

## Clonability control for inconsistent strategy

| ρ | observed success | interpolated (clonability-only) | residual vs. interpolation |
|---|---|---|---|
| 0.0 | 0.949 | 0.949 | +0.000 |
| 0.25 | 0.980 | 0.769 | +0.210 |
| 0.5 | 0.934 | 0.589 | +0.345 |
| 0.75 | 0.645 | 0.409 | +0.236 |
| 0.9 | 0.383 | 0.301 | +0.082 |
| 1.0 | 0.229 | 0.230 | -0.000 |

Both experts solve the task on 100% of episodes, so the rho=1.0 result (0.229) is purely a statement about how hard the alternate strategy is to clone, not about inconsistency.


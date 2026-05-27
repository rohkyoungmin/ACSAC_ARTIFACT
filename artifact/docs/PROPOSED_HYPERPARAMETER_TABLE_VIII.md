# Proposed Hyperparameter Table VIII

This table reports every parameter that appears in Algorithm 1, using the final Proposed configuration. Tuned values come from `results/proposed_performance/00_validation_tuned_params.csv`; fixed values come from `common.py::Config` and the cluster-priority implementation.

| Symbol | Code name | Proposed value | Tuned? | Location | Role |
|---|---:|---:|---:|---|---|
| $D$ | `hash_dim` | $2^{18}$ = 262,144 | No | §IV-B | Signed feature-hashing dimension. |
| $d$ | `svd_components` | 64 | No | §IV-C | Semantic embedding dimension. |
| $K_0$ | `n_clusters_init` | 20 | No | §IV-D | Initial MiniBatchKMeans cluster count. |
| $\delta_{\mathrm{new}}$ | `new_cluster_dist` | **0.25** | Yes | §IV-D | New-cluster distance threshold. |
| $T_{\mathrm{ttl}}$ | `cluster_ttl` | 3 months | No | §IV-D | Cluster pruning TTL. |
| $\epsilon_{\mathrm{ks}}$ | `ks_min_effect` | **0.10** | Yes | §IV-E | Minimum KS effect size. |
| $\alpha_{\mathrm{ks}}$ | `ks_alpha` | 0.05 | No | §IV-E | KS significance level. |
| $\alpha_{\mathrm{EMA}}$ | `perf_ema_alpha` | 0.30 | No | §IV-E | EMA smoothing for confidence-score shift. |
| $\beta$ | `burst_threshold` | 3.0 | No | §IV-E | Novelty burst threshold. |
| $(\omega_{\mathrm{ks}},\omega_{\mathrm{conf}},\omega_{\mathrm{nov}})$ | `drift_weights` | (0.40, 0.30, 0.30) | No | §IV-E | Combined drift-score weights. |
| $\theta_D$ | `combined_threshold` | **0.35** | Yes | §IV-E | Combined drift threshold. |
| $\theta_{\mathrm{emg}}$ | `emergency_threshold` | 0.50 | No | §IV-E | Immediate-response threshold. |
| $\rho_t$ | review ratio schedule | 0.30 normally; 0.45 under drift | No | §IV-F | Monthly review budget schedule; observed overall Proposed label ratio is 0.3327. |
| $\lambda_{\mathrm{new}}$ | new-cluster priority | 2.0 | No | §IV-F | Selection priority for newly created clusters. |
| $\lambda_{\mathrm{drift}}$ | drift-period priority | 1.5 | No | §IV-F | Selection priority for existing clusters during drift. |
| $\lambda_{\mathrm{stable}}$ | stable-cluster priority | 0.5 | No | §IV-F | Selection priority outside drift periods. |
| $\lambda_D$ | `drift_lambda` | **0.50** | Yes | §IV-G | Drift-amplified update-loss strength. |
| TopK | `xai_top_k` | 10 | No | §IV-H | Number of evidence features shown per cluster. |

Tuned grid:

| Parameter | Candidate values | Selected |
|---|---:|---:|
| $\epsilon_{\mathrm{ks}}$ | {0.05, 0.10, 0.15} | 0.10 |
| $\theta_D$ | {0.15, 0.25, 0.35} | 0.35 |
| $\lambda_D$ | {0.50, 1.00, 2.00} | 0.50 |
| $\delta_{\mathrm{new}}$ | {0.25, 0.35, 0.45} | 0.25 |

Notes:

- `results/tuned_params.json` contains older exploratory parameters and should not be used for the Proposed submission table.
- The authoritative Proposed tuning record is `results/proposed_performance/00_validation_tuned_params.csv`.

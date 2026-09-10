# Full-suite trajectory-aligned uncertainty run

The 33-scenario GPU MPPI suite completed on 2026-09-09 using ORCA prediction,
4096 samples, a 60-step/3.0-second horizon, and the configured elliptical
uncertainty growth of 0.10 m/s longitudinally and 0.05 m/s laterally.

## Navigation result

- Success: 32/33 (97.0%), compared with 30/33 in the previous GPU ORCA run.
- Remaining failure: `t_eth_dense_against`, pedestrian collision at 6.55 s.
- Recovered previous failures: `t_dhotel_bottopmid` and `t_univ_trapBL`.
- All-episode means: 18.11 s navigation time, 17.27 m path length, and
  366.22 J robot motion energy.
- Navigation took 25 minutes 26 seconds; post-run diagnostics took
  2 minutes 10 seconds.

This is one seeded suite, so the two-success improvement is evidence for the
ellipse but not yet a variance estimate.

## Prediction diagnostics

Diagnostics reconstruct the ORCA predictions from the same prerecorded
pedestrian trajectories, use observations every 0.5 s, and stop each scene at
the navigation run's actual termination time. Ground truth is used only after
navigation for evaluation and is never exposed to MPPI.

At 0.5, 1.0, 1.5, 2.0, 2.5, and 3.0 seconds:

- FDE-like mean error grows from 0.071 m to 0.582 m.
- ADE-like cumulative mean error grows from 0.037 m to 0.264 m.
- Coverage by only the added 0.30/0.15 m uncertainty ellipse at 3.0 s is
  27.6%, or 29.0% when adjacent prediction times are included.
- The actual MPPI soft-cost support also contains the pedestrian radius
  (0.20 m) and dynamic clearance (0.55 m). Its 3.0-second axes are therefore
  1.05/0.90 m and cover 83.0%, or 83.3% with temporal smearing.
- Full cost-support-plus-smearing coverage is 99.9%, 99.2%, 96.8%, 93.6%,
  90.3%, and 83.3% at the six selected lead times.

The distinction between added-uncertainty coverage and complete cost-support
coverage is important: the small ellipse alone is not intended to contain all
future positions, while the collision-clearance region supplies a substantial
base radius.

## Artifacts

- `mppi_summary.json`, `.csv`, and `.txt`: navigation outcomes and metrics.
- `policy.config`: exact run configuration.
- `diagnostics/selected_lead_times.csv`: requested 0.5-second summary.
- `diagnostics/aggregate_by_lead_time.csv`: all 0.05-second prediction steps.
- `diagnostics/scene_timeline.csv`: per-scene, per-observation diagnostics.
- `diagnostics/error_by_prediction_horizon.png`: FDE/ADE and ellipse axes.
- `diagnostics/coverage_by_prediction_horizon.png`: uncertainty and full
  cost-support coverage.
- `diagnostics/scene_timeline_error.png` and
  `scene_timeline_coverage.png`: metrics as each scene develops.

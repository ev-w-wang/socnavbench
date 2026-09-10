# Five-scenario ORCA uncertainty calibration

This directory contains an offline calibration run of
`evaluate_orca_uncertainty.py` over:

- `t_ETH1`
- `t_eth_dense_against`
- `t_zara1_dense_cross`
- `t_univ_trapBL`
- `t_dhotel_bottopmid`

The evaluator samples each scene every 0.5 s, estimates each active
pedestrian's velocity from the preceding 0.05 s, predicts 3.0 s with ORCA,
and compares each predicted center with the prerecorded center at the same
future time.

## Coverage definitions

- **Nominal coverage:** ground truth is inside the trajectory-aligned ellipse
  at the same ORCA prediction index. Its longitudinal semi-axis grows at
  0.10 m/s and its lateral semi-axis at 0.05 m/s.
- **Smeared coverage:** ground truth is inside the corresponding ellipse of
  any ORCA prediction in the configured +/-3-index temporal window.

These are geometric coverage measurements. The temporal decay and future
discount used by the MPPI cost affect cost magnitude, not whether a sample is
counted as covered.

## Aggregate results

- 0.5 s lead: 0.05/0.025 m longitudinal/lateral axes, 0.086 m mean
  error, 46.4% nominal coverage, and 61.9% smeared coverage.
- 1.0 s lead: 0.10/0.05 m axes, 0.187 m mean error, 37.3% nominal
  coverage, and 47.0% smeared coverage.
- 2.0 s lead: 0.20/0.10 m axes, 0.408 m mean error, 28.9% nominal
  coverage, and 34.5% smeared coverage.
- 3.0 s lead: 0.30/0.15 m axes, 0.646 m mean error, 23.5% nominal
  coverage, and 24.8% smeared coverage.

The current ellipse axes are smaller than the aggregate mean ORCA error at
every listed horizon. Temporal smearing helps substantially through the early
and middle horizon, but helps less at 3.0 s because the prediction array ends
there and the smear window becomes one-sided.

Performance is strongly scene-dependent. `t_univ_trapBL` has 70.0% smeared
coverage across all evaluated leads, while the two ETH runs have 16.1% and
21.8%. A single globally scaled ellipse therefore cannot represent every
scene equally well.

## Artifacts

- `calibration_summary.json`: compact run metadata, aggregate lead-time
  metrics, episode totals, and selected 0.5/1/2/3 s metrics.
- `aggregate_by_lead_time.csv`: aggregate metrics at every 0.05 s lead.
- `scene_timeline.csv`: per-scene metrics over observation time.
- `error_by_prediction_horizon.png`: aggregate error versus ellipse axes.
- `coverage_by_prediction_horizon.png`: nominal and smeared coverage.
- `scene_timeline_error.png`: error as each of the five scenes develops.
- `scene_timeline_coverage.png`: smeared coverage as each scene develops.

This is predictor calibration, not a closed-loop navigation test. It uses
clean finite-difference velocities from recorded positions and does not model
perception noise, robot feedback, or collision-induced pedestrian pauses.

The two-dataset DoubleHotel episode generates 70 sources but retains 48
name-keyed sources, matching the simulator's current prerecorded-agent
dictionary behavior.

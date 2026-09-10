# Five-scenario uncertainty navigation comparison

Run completed on 2026-09-09 with seed 991, GPU MPPI, 4096 samples, a
60-step/3.0-second horizon, ORCA prediction, and the same temporal-smearing
configuration for every profile.

The five scenarios were `t_ETH1`, `t_eth_dense_against`,
`t_zara1_dense_cross`, `t_univ_trapBL`, and `t_dhotel_bottopmid`.

## Profiles

- `zero`: Gaussian dynamic risk with uncertainty growth set to zero.
- `ellipse`: 0.10 m/s longitudinal and 0.05 m/s lateral uncertainty growth.
- `circle`: 0.10 m/s growth in both directions.

All profiles used a Gaussian sigma level of 2.0. Therefore this comparison
isolates uncertainty growth and shape; it does not compare against the older
pre-Gaussian exponential dynamic cost.

## Main result

- Zero uncertainty succeeded in 3/5 scenarios. It collided with a pedestrian
  in `t_eth_dense_against` and `t_dhotel_bottopmid`.
- Elliptical uncertainty succeeded in 4/5. It resolved
  `t_dhotel_bottopmid` but retained the `t_eth_dense_against` collision.
- Circular uncertainty also succeeded in 4/5, with the same remaining
  `t_eth_dense_against` failure.

The additional uncertainty therefore provided a real safety benefit in this
small set, but increasing lateral support from the ellipse to the circle did
not add another success.

## Efficiency

Across the three scenarios that all profiles completed (`t_ETH1`,
`t_zara1_dense_cross`, and `t_univ_trapBL`):

- Zero uncertainty averaged 16.67 m path length, 18.32 s navigation time, and
  347.94 J.
- Elliptical uncertainty averaged 17.71 m, 20.35 s, and 359.44 J.
- Circular uncertainty averaged 17.58 m, 20.33 s, and 354.16 J.

Most of the increase came from `t_univ_trapBL`, where navigation increased
from 23.15 s with zero uncertainty to 29.25/29.15 s with the
ellipse/circle. `t_ETH1` was identical across profiles.

In the recovered `t_dhotel_bottopmid` scenario, the ellipse succeeded in
25.30 s with a 23.61 m path and 0.71 m minimum recorded pedestrian distance.
The circle succeeded in 22.35 s with a 22.65 m path and 1.16 m minimum
distance.

## Remaining failure

`t_eth_dense_against` remained a pedestrian collision:

- Zero: collision at 6.55 s, minimum recorded distance 0.254 m.
- Ellipse: collision at 6.55 s, minimum distance 0.312 m.
- Circle: collision at 4.90 s, minimum distance 0.286 m.

The circle's earlier failure is evidence that simply enlarging the cost
region is not monotonically safer for sampled local control.

## Stopping metric caveat

The summarizer records the fraction of logged robot-speed samples below
0.05 m/s. It was approximately 51% in every profile. The near-identical
values and alternating simulator trajectory samples indicate that this is
dominated by the logging/update representation rather than genuine freezing.
It is retained in the CSV for transparency but should not be used as a
behavioral conclusion without command-level velocity logging.

## Artifacts

- `comparison.json` and `comparison.csv`: aggregate all-episode and
  successful-episode statistics.
- `episode_comparison.csv`: individual episode outcomes and metrics.
- `<profile>/mppi_summary.{json,csv,txt}`: source summaries.
- `<profile>/policy.config`: exact profile configuration.
- `run_uncertainty_nav5.sh`: reproducible runner in the parent directory.
- `summarize_uncertainty_nav5.py`: comparison aggregator in the parent
  directory.

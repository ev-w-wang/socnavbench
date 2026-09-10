# MPPI controller integration

## Overview

The MPPI controller is integrated through SocNavBench's existing joystick
process. The simulator continues to own robot dynamics, collision detection,
episode termination, and metric generation. The MPPI process receives each
`SimState`, predicts candidate robot trajectories, and returns one `(v, w)`
velocity command per 0.05-second simulation step.

Data flow:

1. `tests/test_episodes.py` starts the simulator and publishes `SimState`
   messages over the existing Unix sockets.
2. `JoystickMPPI` converts the robot and pedestrian `AgentState` objects into
   the compact state consumed by `MPPIController`.
3. Pedestrian velocities are estimated by finite differences between
   consecutive sensed states.
4. Python-RVO2/ORCA predicts pedestrian positions over the MPPI horizon.
5. MPPI rolls out SocNavBench-compatible Euler Dubins robot dynamics and scores
   goal progress, heading, speed, controls, static-map clearance, and predicted
   pedestrian clearance.
6. The first command from the optimized sequence is sent to `RobotAgent`; the
   remaining sequence is retained as the next warm start.

Python-RVO2 predicts pedestrian trajectories. It does not replace the robot's
Dubins motion model.

## Trajectory prediction backends

Pedestrian prediction is isolated in `trajectory_predictors.py`. Every backend
implements the same `TrajectoryPredictor.predict(state, dynamic_obstacles)`
interface and returns an array shaped
`(prediction_horizon, pedestrians, xy)`. Select a backend with
`dynamic_prediction_method` in `policy.config`:

- `orca`: runs Python-RVO2 using measured pedestrian velocities as preferred
  velocities.
- `constant_velocity`: linearly extrapolates each measured position and
  velocity independently.
- `ground_truth`: an oracle for analysis. The joystick reconstructs the
  episode's prerecorded pedestrian trajectories from the SocNavBench dataset
  and samples their interpolation functions at future simulation times.

Ground truth intentionally uses information unavailable to a deployed robot.
It should be used to estimate the performance ceiling caused by prediction
error, not as a benchmark-valid policy result.

Future learned predictors can subclass `TrajectoryPredictor` and be added with
`register_trajectory_predictor(name, factory)` without changing either MPPI
controller implementation.

## Implementation changes

- `mppi.py`
  - Aligns CPU and GPU dynamics with SocNavBench's Euler Dubins update.
  - Adds `OccupancyGridDistanceField`, built once per episode from
    `map_traversible` using SciPy's Euclidean distance transform.
  - Includes robot radius in static clearance and robot-plus-human radii in
    dynamic clearance.
  - Fixes Python-RVO2 argument types and retains ORCA prediction.
  - Adds a one-command receding-horizon API.
  - Makes Torch optional so the NumPy controller works in the original Python
    3.6 environment.
- `trajectory_predictors.py`
  - Defines the common predictor interface and registry.
  - Implements ORCA, constant-velocity, and ground-truth replay prediction.
- `joystick/joystick_py/joystick_mppi.py`
  - Loads and type-checks `policy.config`.
  - Adapts SocNavBench states, goals, pedestrian velocities, radii, and maps.
  - Selects the GPU implementation when requested and available, otherwise
    falls back to NumPy.
  - Implements the standard joystick sense-plan-act loop.
- `joystick/joystick_client.py`
  - Adds `--algo mppi`.
- `policy.config`
  - Adds the reproducible MPPI profile described below.
- `params/user_params.ini`
  - Enables velocity commands, disables movie generation for evaluation, and
    terminates episodes on pedestrian collision.
- `params/episode_params_val.ini`
  - Selects the documented 10-scenario evaluation subset.
- `tests/unit_tests/test_mppi.py`
  - Covers configuration, state adaptation, pedestrian velocity estimation,
    map clearance, action limits, dynamics consistency, ORCA output, and the
    optional CUDA path.
- `tests/summarize_mppi_results.py`
  - Produces per-episode CSV and JSON summaries from SocNavBench score pickles.

## Hyperparameter tuning process

Tune in stages so controller quality is not confused with prediction error:

1. Use `ground_truth` pedestrian trajectories on a small set containing open,
   dense-crossing, static-obstacle, and trap scenarios.
2. Change one parameter family at a time: sampling budget, temporal horizon,
   control noise/temperature, then safety costs.
3. Promote a profile only after it improves success without materially
   degrading path ratio, navigation time, energy, or closest-pedestrian
   distance.
4. Run the complete 33-episode ground-truth suite to establish the controller
   ceiling.
5. Run the identical profile with ORCA prediction. The difference isolates
   prediction quality from MPPI formulation and tuning.
6. Inspect failures by cause before tuning further. Pedestrian, static-map,
   timeout, and local-kinematic failures require different changes.

The initial CPU progression was:

| Profile | Horizon | Samples | Iterations | Small-set result |
| --- | --- | --- | --- | --- |
| Initial | 15 | 64 | 2 | 7/10 |
| Longer horizon | 30 | 128 | 3 | 9/10 |
| More samples | 35 | 192 | 2 | 9/10 |
| Tuned CPU | 40 | 256 | 2 | 10/10 |

This established `temp=10`, `std_v=0.45`, `std_w=0.9`,
`action_noise_beta=0.85`, and dynamic clearance/weight `0.55 m / 750`.
The GPU sweep then increased samples to 4096 nearly for free and extended the
horizon to 60 steps so the 3-second rollout could represent the
`t_univ_trapTL` U-turn.

## Current reproducible profile

The validated full-suite snapshots remain under `tests/mppi_tuning/gpu4096_*`.
The active `policy.config` now adds the experimental prediction-risk envelope
described below.

- CUDA GPU backend (`use_gpu=true`)
- seed 991, timestep 0.05 s
- horizon 60 steps (3.0 s), 4096 samples, 2 iterations
- velocity limits 0–1.2 m/s and ±1.1 rad/s
- temperature 10; control noise standard deviations 0.45 and 0.9
- goal-directed warm start using the full angular limit
- dynamic clearance/weight 0.55 m / 750
- static clearance/weight 0.2 m / 200
- trajectory-aligned uncertainty growth: 0.10 m/s longitudinally and
  0.05 m/s laterally
- Gaussian spatial risk with the support boundary at two standard deviations
- future-risk discount 0.10 per second
- temporal smearing ±3 steps (±0.15 s), with 0.10 s decay constant
- ORCA pedestrian prediction for the deployable profile

Detailed rationale is retained in
`tests/mppi_tuning/gpu4096/NOTES.txt`.

### Prediction-risk envelope

The original loss treated each predicted pedestrian position as exact. The
experimental loss preserves the hard `1e4` penalty for physical overlap with
the nominal prediction, then forms a soft trajectory-aligned risk envelope:

- The local direction of travel is estimated with a centered difference of
  the predicted trajectory, with observed velocity and the x-axis as
  fallbacks for stationary predictions.
- Longitudinal and lateral semi-axes grow independently with lead time. The
  active profile uses `growth * 1.0` longitudinally and `growth * 0.5`
  laterally.
- The pedestrian/robot collision radius and dynamic clearance inflate both
  axes equally.
- Spatial cost is Gaussian in normalized elliptical distance. The configured
  sigma level of 2.0 makes cost at the support boundary
  `exp(-0.5 * 2^2)`, or about 13.5% of center cost.
- Future soft costs receive
  `exp(-dynamic_uncertainty_discount * prediction_time)`. This discount is
  deliberately mild and never weakens nominal physical-overlap penalties.
- At robot timestep `t`, predictions in
  `[t - dynamic_time_smear_steps, t + dynamic_time_smear_steps]` contribute
  with weight `exp(-abs(offset) * dt / dynamic_time_smear_tau)`.
- Costs across nearby prediction times use a maximum rather than a sum. This
  avoids counting one pedestrian repeatedly while retaining the highest-risk
  nearby timing.

All parameters can be set to zero (`dynamic_time_smear_steps=0`,
`dynamic_uncertainty_growth=0`, and
`dynamic_uncertainty_discount=0`) to recover point-prediction behavior.
CPU and GPU implementations share the same semantics and are covered by a
numerical parity test.

## Current full-suite results

| Predictor | Success | Mean path | Mean navigation | Mean energy | Mean planner wait |
| --- | --- | --- | --- | --- | --- |
| Ground truth | 32/33 | 17.20 m | 17.97 s | 366 J | 28 s |
| ORCA | 30/33 | 16.33 m | 17.46 s | 348 J | 22 s |

Compact per-episode JSON/CSV summaries and exact policy snapshots are retained
under `tests/mppi_tuning/gpu4096_gt/` and
`tests/mppi_tuning/gpu4096_orca/`. Raw episode logs and superseded tuning
artifacts are intentionally not versioned.

Ground truth resolved the former `t_univ_trapTL` obstacle collision but
collided with a pedestrian in `t_dhotel_bottopmid`, leaving success at 32/33.
ORCA improved from the earlier 29/33 CPU result to 30/33. Similar motion
metrics and the large wall-time reduction validate the GPU implementation as
the development baseline.

The benchmark uses `keep_episode_running=false`; collisions terminate episodes.

## Environment and commands

Relevant dependencies:

- Python 3.6.9
- NumPy 1.19.2 and SciPy 1.5.4
- pyrvo2 built from `../Python-RVO2`
- PyTorch 1.10.2+cu113 on an RTX 3080 (`sm_86`)

From `SocNavBench/`:

```bash
conda activate socnavbench
PYTHONPATH='.:joystick' python tests/unit_tests/test_mppi.py
bash tests/mppi_tuning/run_gpu4096_full33.sh
```

The runner executes ground truth followed by ORCA and writes compact summaries.

## Known limitations

- Ground truth is an oracle ceiling, not a deployable policy.
- ORCA uses measured pedestrian velocity as preferred velocity and does not
  model static obstacles or the robot inside its prediction.
- The uncertainty growth and timing kernel are heuristic rather than calibrated
  from held-out trajectory-prediction errors.
- Occupancy-grid clearance uses nearest-cell lookup.
- CPU and GPU sampling use different random-number generators and are not
  expected to produce bit-identical trajectories.
- A single 33-episode run is a baseline, not a statistical estimate across
  random seeds.

"""Offline calibration of ORCA forecasts against prerecorded trajectories.

The evaluator reconstructs each episode's prerecorded pedestrians, observes
their current positions and finite-difference velocities, then compares an
ORCA rollout with the recorded future. It reports center-position error and
coverage by the trajectory-aligned MPPI uncertainty ellipse, both at the
nominal prediction index and after the configured temporal-smearing window is
considered.
"""

import argparse
import configparser
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from agents.humans.recorded_human import PrerecordedHuman
from params.central_params import create_socnav_params, create_test_params
from trajectory_predictors import ORCAPredictor


DEFAULT_SCENARIOS = [
    "t_ETH1",
    "t_eth_dense_against",
    "t_zara1_dense_cross",
    "t_univ_trapBL",
    "t_dhotel_bottopmid",
]
TIMELINE_LEADS_S = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


class _HumanState:
    pass


class _PredictionState:
    pass


def _position(source, query_time):
    return np.array(
        [source.xinterp(query_time), source.yinterp(query_time)],
        dtype=float,
    )


def _load_sources(params, episode):
    """Load the same prerecorded-human set used by the simulator.

    The dictionary update intentionally mirrors Simulator's name-keyed
    backstage pool when an episode contains more than one dataset.
    """
    sources = {}
    generated_count = 0
    for index, dataset in enumerate(episode.pedestrian_datasets):
        humans = PrerecordedHuman.generate_humans(
            params,
            max_time=episode.max_time,
            start_t=episode.datasets_start_t[index],
            ped_range=episode.ped_ranges[index],
            dataset=dataset,
        )
        generated_count += len(humans)
        for human in humans:
            sources[human.get_name()] = human
    return list(sources.values()), generated_count


def _make_prediction_state(sources, observation_time, dt, radius):
    state = _PredictionState()
    state.human_states = []
    active_sources = []
    for source in sources:
        start_time = float(source.get_start_time())
        end_time = float(source.get_end_time())
        if observation_time < start_time or observation_time >= end_time:
            continue

        position = _position(source, observation_time)
        human_state = _HumanState()
        human_state.position = position
        if observation_time <= start_time + 1e-9:
            human_state.velocity = np.zeros(2, dtype=float)
        else:
            previous_time = max(start_time, observation_time - dt)
            human_state.velocity = (
                position - _position(source, previous_time)
            ) / (observation_time - previous_time)
        human_state.radius = radius
        state.human_states.append(human_state)
        active_sources.append(source)
    return state, active_sources


def _trajectory_unit_directions(predictions, current_velocities):
    indices = np.arange(predictions.shape[0])
    previous_indices = np.maximum(indices - 1, 0)
    next_indices = np.minimum(indices + 1, predictions.shape[0] - 1)
    directions = (
        predictions[next_indices] - predictions[previous_indices]
    )
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    velocity_directions = np.broadcast_to(
        current_velocities[np.newaxis, :, :],
        directions.shape,
    )
    directions = np.where(
        norms > np.finfo(float).eps,
        directions,
        velocity_directions,
    )
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    default_directions = np.zeros_like(directions)
    default_directions[:, :, 0] = 1.0
    return np.where(
        norms > np.finfo(float).eps,
        directions / np.maximum(norms, np.finfo(float).eps),
        default_directions,
    )


def _ellipse_contains(
    prediction,
    unit_direction,
    ground_truth,
    longitudinal_axis,
    lateral_axis,
):
    difference = ground_truth - prediction
    longitudinal_distance = np.dot(difference, unit_direction)
    lateral_distance = (
        -difference[0] * unit_direction[1]
        + difference[1] * unit_direction[0]
    )
    normalized_distance_squared = (
        (longitudinal_distance / longitudinal_axis) ** 2
        + (lateral_distance / lateral_axis) ** 2
    )
    return bool(normalized_distance_squared <= 1.0)


def _smear_coverage(
    predictions,
    unit_directions,
    pedestrian_index,
    ground_truth,
    nominal_index,
    longitudinal_axes,
    lateral_axes,
    smear_steps,
):
    start = max(0, nominal_index - smear_steps)
    stop = min(predictions.shape[0], nominal_index + smear_steps + 1)
    return any(
        _ellipse_contains(
            predictions[index, pedestrian_index],
            unit_directions[index, pedestrian_index],
            ground_truth,
            longitudinal_axes[index],
            lateral_axes[index],
        )
        for index in range(start, stop)
    )


def _summarize_samples(
    errors,
    uncertainty_covered,
    uncertainty_smeared_covered,
    base_covered=None,
    cost_support_covered=None,
    cost_support_smeared_covered=None,
):
    errors = np.asarray(errors, dtype=float)
    result = {
        "count": int(errors.size),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "p95_error_m": float(np.percentile(errors, 95)),
        "uncertainty_ellipse_coverage": float(
            np.mean(uncertainty_covered)
        ),
        "uncertainty_smear_coverage": float(
            np.mean(uncertainty_smeared_covered)
        ),
    }
    if base_covered is not None:
        result.update(
            {
                "base_support_coverage": float(np.mean(base_covered)),
                "cost_support_coverage": float(
                    np.mean(cost_support_covered)
                ),
                "cost_support_smear_coverage": float(
                    np.mean(cost_support_smeared_covered)
                ),
            }
        )
    return result


def evaluate_episode(
    scenario,
    predictor,
    params,
    config,
    observation_interval,
    navigation_end_time=None,
):
    episode = create_test_params(scenario)
    sources, generated_count = _load_sources(params, episode)
    unique_count = len(sources)
    maximum_source_time = max(
        [float(source.get_end_time()) for source in sources] or [0.0]
    )
    stop_time = min(float(episode.max_time), maximum_source_time)
    if navigation_end_time is not None:
        stop_time = min(stop_time, float(navigation_end_time))

    horizon = predictor.horizon
    dt = predictor.dt
    longitudinal_axes = (
        config["uncertainty_base"]
        + config["uncertainty_growth"]
        * config["longitudinal_scale"]
        * (np.arange(horizon, dtype=float) + 1.0)
        * dt
    )
    lateral_axes = (
        config["uncertainty_base"]
        + config["uncertainty_growth"]
        * config["lateral_scale"]
        * (np.arange(horizon, dtype=float) + 1.0)
        * dt
    )
    base_support = (
        config["pedestrian_radius"] + config["dynamic_obs_clearance"]
    )
    cost_longitudinal_axes = base_support + longitudinal_axes
    cost_lateral_axes = base_support + lateral_axes
    by_lead = [
        {
            "errors": [],
            "uncertainty": [],
            "uncertainty_smeared": [],
            "base": [],
            "cost_support": [],
            "cost_support_smeared": [],
        }
        for _ in range(horizon)
    ]
    timeline = []
    observed_frames = 0
    active_pedestrian_observations = 0
    times = np.arange(0.0, stop_time + 1e-9, observation_interval)

    for observation_time in times:
        state, active_sources = _make_prediction_state(
            sources,
            float(observation_time),
            dt,
            config["pedestrian_radius"],
        )
        if not active_sources:
            continue

        dynamic_obstacles = np.zeros((len(active_sources), 5), dtype=float)
        predictions = predictor.predict(state, dynamic_obstacles)
        unit_directions = _trajectory_unit_directions(
            predictions,
            np.array(
                [human.velocity for human in state.human_states],
                dtype=float,
            ),
        )
        observed_frames += 1
        active_pedestrian_observations += len(active_sources)
        frame_metrics = {}

        for lead_index in range(horizon):
            future_time = float(observation_time) + (lead_index + 1) * dt
            frame_errors = []
            frame_uncertainty = []
            frame_uncertainty_smeared = []
            frame_base = []
            frame_cost_support = []
            frame_cost_support_smeared = []
            for pedestrian_index, source in enumerate(active_sources):
                if (
                    future_time > float(source.get_end_time()) + 1e-9
                    or future_time > float(episode.max_time) + 1e-9
                ):
                    continue
                ground_truth = _position(source, future_time)
                error = float(
                    np.linalg.norm(
                        predictions[lead_index, pedestrian_index] - ground_truth
                    )
                )
                uncertainty_covered = _ellipse_contains(
                    predictions[lead_index, pedestrian_index],
                    unit_directions[lead_index, pedestrian_index],
                    ground_truth,
                    longitudinal_axes[lead_index],
                    lateral_axes[lead_index],
                )
                uncertainty_smeared_covered = _smear_coverage(
                    predictions,
                    unit_directions,
                    pedestrian_index,
                    ground_truth,
                    lead_index,
                    longitudinal_axes,
                    lateral_axes,
                    config["smear_steps"],
                )
                base_covered = _ellipse_contains(
                    predictions[lead_index, pedestrian_index],
                    unit_directions[lead_index, pedestrian_index],
                    ground_truth,
                    base_support,
                    base_support,
                )
                cost_support_covered = _ellipse_contains(
                    predictions[lead_index, pedestrian_index],
                    unit_directions[lead_index, pedestrian_index],
                    ground_truth,
                    cost_longitudinal_axes[lead_index],
                    cost_lateral_axes[lead_index],
                )
                cost_support_smeared_covered = _smear_coverage(
                    predictions,
                    unit_directions,
                    pedestrian_index,
                    ground_truth,
                    lead_index,
                    cost_longitudinal_axes,
                    cost_lateral_axes,
                    config["smear_steps"],
                )
                by_lead[lead_index]["errors"].append(error)
                by_lead[lead_index]["uncertainty"].append(
                    uncertainty_covered
                )
                by_lead[lead_index]["uncertainty_smeared"].append(
                    uncertainty_smeared_covered
                )
                by_lead[lead_index]["base"].append(base_covered)
                by_lead[lead_index]["cost_support"].append(
                    cost_support_covered
                )
                by_lead[lead_index]["cost_support_smeared"].append(
                    cost_support_smeared_covered
                )
                frame_errors.append(error)
                frame_uncertainty.append(uncertainty_covered)
                frame_uncertainty_smeared.append(
                    uncertainty_smeared_covered
                )
                frame_base.append(base_covered)
                frame_cost_support.append(cost_support_covered)
                frame_cost_support_smeared.append(
                    cost_support_smeared_covered
                )

            lead_s = round((lead_index + 1) * dt, 6)
            if frame_errors and any(
                abs(lead_s - selected) < dt / 2.0
                for selected in TIMELINE_LEADS_S
            ):
                frame_metrics[str(lead_s)] = _summarize_samples(
                    frame_errors,
                    frame_uncertainty,
                    frame_uncertainty_smeared,
                    frame_base,
                    frame_cost_support,
                    frame_cost_support_smeared,
                )

        if frame_metrics:
            timeline.append(
                {
                    "observation_time_s": round(float(observation_time), 6),
                    "active_pedestrians": len(active_sources),
                    "leads": frame_metrics,
                }
            )

    lead_metrics = []
    for lead_index, samples in enumerate(by_lead):
        if not samples["errors"]:
            continue
        metrics = _summarize_samples(
            samples["errors"],
            samples["uncertainty"],
            samples["uncertainty_smeared"],
            samples["base"],
            samples["cost_support"],
            samples["cost_support_smeared"],
        )
        metrics.update(
            {
                "lead_time_s": round((lead_index + 1) * dt, 6),
                "longitudinal_axis_m": float(
                    longitudinal_axes[lead_index]
                ),
                "lateral_axis_m": float(lateral_axes[lead_index]),
                "cost_longitudinal_axis_m": float(
                    cost_longitudinal_axes[lead_index]
                ),
                "cost_lateral_axis_m": float(
                    cost_lateral_axes[lead_index]
                ),
            }
        )
        lead_metrics.append(metrics)

    all_errors = [
        error
        for samples in by_lead
        for error in samples["errors"]
    ]
    all_uncertainty = [
        covered
        for samples in by_lead
        for covered in samples["uncertainty"]
    ]
    all_uncertainty_smeared = [
        covered
        for samples in by_lead
        for covered in samples["uncertainty_smeared"]
    ]
    all_base = [
        covered for samples in by_lead for covered in samples["base"]
    ]
    all_cost_support = [
        covered
        for samples in by_lead
        for covered in samples["cost_support"]
    ]
    all_cost_support_smeared = [
        covered
        for samples in by_lead
        for covered in samples["cost_support_smeared"]
    ]
    overall = (
        _summarize_samples(
            all_errors,
            all_uncertainty,
            all_uncertainty_smeared,
            all_base,
            all_cost_support,
            all_cost_support_smeared,
        )
        if all_errors
        else {"count": 0}
    )
    overall.update(
        {
            "scenario": scenario,
            "generated_sources": generated_count,
            "unique_simulator_sources": unique_count,
            "observed_frames": observed_frames,
            "active_pedestrian_observations": active_pedestrian_observations,
        }
    )
    return {
        "scenario": scenario,
        "overall": overall,
        "by_lead_time": lead_metrics,
        "timeline": timeline,
    }


def _aggregate(results, horizon, dt, longitudinal_axes, lateral_axes):
    aggregated = []
    cumulative_error_sum = 0.0
    cumulative_count = 0
    for lead_index in range(horizon):
        lead_rows = [
            result["by_lead_time"][lead_index]
            for result in results
            if lead_index < len(result["by_lead_time"])
        ]
        if not lead_rows:
            continue
        total = sum(row["count"] for row in lead_rows)
        fixed_lead_mean = sum(
            row["mean_error_m"] * row["count"] for row in lead_rows
        ) / total
        cumulative_error_sum += fixed_lead_mean * total
        cumulative_count += total
        aggregated.append(
            {
                "lead_time_s": round((lead_index + 1) * dt, 6),
                "longitudinal_axis_m": float(
                    longitudinal_axes[lead_index]
                ),
                "lateral_axis_m": float(lateral_axes[lead_index]),
                "cost_longitudinal_axis_m": float(
                    lead_rows[0]["cost_longitudinal_axis_m"]
                ),
                "cost_lateral_axis_m": float(
                    lead_rows[0]["cost_lateral_axis_m"]
                ),
                "count": total,
                "mean_error_m": fixed_lead_mean,
                "fde_like_mean_error_m": fixed_lead_mean,
                "ade_like_mean_error_m": (
                    cumulative_error_sum / cumulative_count
                ),
                **{
                    key: sum(
                        row[key] * row["count"] for row in lead_rows
                    ) / total
                    for key in (
                        "uncertainty_ellipse_coverage",
                        "uncertainty_smear_coverage",
                        "base_support_coverage",
                        "cost_support_coverage",
                        "cost_support_smear_coverage",
                    )
                },
            }
        )
    return aggregated


def _write_csv(path, fieldnames, rows):
    with open(path, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compact_result(result, dt):
    selected_leads = []
    for row in result["by_lead_time"]:
        if any(
            abs(row["lead_time_s"] - lead_s) < dt / 2.0
            for lead_s in TIMELINE_LEADS_S
        ):
            selected_leads.append(row)
    return {
        "scenario": result["scenario"],
        "overall": result["overall"],
        "selected_lead_times": selected_leads,
    }


def _plot_aggregate(output_dir, aggregate):
    lead = np.array([row["lead_time_s"] for row in aggregate])
    longitudinal_axis = np.array(
        [row["longitudinal_axis_m"] for row in aggregate]
    )
    lateral_axis = np.array(
        [row["lateral_axis_m"] for row in aggregate]
    )
    mean_error = np.array([row["mean_error_m"] for row in aggregate])
    ade_like_error = np.array(
        [row["ade_like_mean_error_m"] for row in aggregate]
    )
    uncertainty = np.array(
        [row["uncertainty_ellipse_coverage"] for row in aggregate]
    )
    uncertainty_smeared = np.array(
        [row["uncertainty_smear_coverage"] for row in aggregate]
    )
    cost_support = np.array(
        [row["cost_support_coverage"] for row in aggregate]
    )
    cost_support_smeared = np.array(
        [row["cost_support_smear_coverage"] for row in aggregate]
    )

    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(lead, mean_error, label="FDE-like error at lead", linewidth=2)
    axis.plot(
        lead,
        ade_like_error,
        label="ADE-like error through lead",
        linewidth=2,
    )
    axis.plot(
        lead,
        longitudinal_axis,
        label="Longitudinal uncertainty axis",
        linewidth=2,
    )
    axis.plot(
        lead,
        lateral_axis,
        label="Lateral uncertainty axis",
        linewidth=2,
    )
    axis.set(
        xlabel="Prediction lead time (s)",
        ylabel="Center-position distance (m)",
        title="ORCA error and configured uncertainty ellipse axes",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "error_by_prediction_horizon.png"),
        dpi=180,
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(
        lead,
        100.0 * uncertainty,
        label="Added uncertainty only",
        linewidth=2,
    )
    axis.plot(
        lead,
        100.0 * uncertainty_smeared,
        label="Added uncertainty + smearing",
        linewidth=2,
    )
    axis.plot(
        lead,
        100.0 * cost_support,
        label="Full MPPI cost support",
        linewidth=2,
    )
    axis.plot(
        lead,
        100.0 * cost_support_smeared,
        label="Full cost support + smearing",
        linewidth=2,
    )
    axis.set(
        xlabel="Prediction lead time (s)",
        ylabel="Ground-truth coverage (%)",
        title="Ground truth inside ORCA uncertainty envelope",
        ylim=(0.0, 100.0),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "coverage_by_prediction_horizon.png"),
        dpi=180,
    )
    plt.close(fig)


def _timeline_series(result, lead_s, metric):
    times = []
    values = []
    lead_key = str(float(lead_s))
    for row in result["timeline"]:
        lead_metrics = row["leads"].get(lead_key)
        if lead_metrics is None:
            continue
        times.append(row["observation_time_s"])
        value = lead_metrics[metric]
        if "coverage" in metric:
            value *= 100.0
        values.append(value)
    return times, values


def _plot_timelines(output_dir, results):
    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(10, 2.5 * len(results)),
        sharex=False,
    )
    for axis, result in zip(np.atleast_1d(axes), results):
        for lead_s in TIMELINE_LEADS_S:
            times, values = _timeline_series(result, lead_s, "mean_error_m")
            if values:
                axis.plot(times, values, label="%.1f s lead" % lead_s)
        axis.set_title(result["scenario"])
        axis.set_xlabel("Scene observation time (s)")
        axis.set_ylabel("Mean error (m)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, fontsize=8)
    fig.suptitle("ORCA prediction error as each scene develops")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scene_timeline_error.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(10, 2.5 * len(results)),
        sharex=False,
    )
    for axis, result in zip(np.atleast_1d(axes), results):
        for lead_s in TIMELINE_LEADS_S:
            times, values = _timeline_series(
                result,
                lead_s,
                "cost_support_smear_coverage",
            )
            if values:
                axis.plot(times, values, label="%.1f s lead" % lead_s)
        axis.set_title(result["scenario"])
        axis.set_xlabel("Scene observation time (s)")
        axis.set_ylabel("Smeared coverage (%)")
        axis.set_ylim(0.0, 100.0)
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, fontsize=8)
    fig.suptitle("Uncertainty-envelope coverage as each scene develops")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "scene_timeline_coverage.png"),
        dpi=180,
    )
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--navigation-summary",
        help=(
            "Optional mppi_summary.json. Its episode list selects scenarios "
            "and its navigation times cap each diagnostic timeline."
        ),
    )
    parser.add_argument("--observation-interval", type=float, default=0.5)
    parser.add_argument(
        "--policy-config",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "policy.config",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(__file__),
            "orca_calibration5",
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    navigation_end_times = {}
    if args.navigation_summary:
        with open(args.navigation_summary) as summary_file:
            navigation_summary = json.load(summary_file)
        navigation_rows = navigation_summary["episodes"]
        navigation_end_times = {
            row["episode"]: row.get("navigation_time")
            for row in navigation_rows
        }
        if args.scenarios is None:
            args.scenarios = [row["episode"] for row in navigation_rows]
    if args.scenarios is None:
        args.scenarios = DEFAULT_SCENARIOS
    parser = configparser.ConfigParser()
    parser.read(args.policy_config)
    policy = parser["mppi"]
    config = {
        "uncertainty_base": policy.getfloat(
            "dynamic_uncertainty_base",
            fallback=0.0,
        ),
        "uncertainty_growth": policy.getfloat("dynamic_uncertainty_growth"),
        "longitudinal_scale": policy.getfloat(
            "dynamic_uncertainty_longitudinal_scale"
        ),
        "lateral_scale": policy.getfloat(
            "dynamic_uncertainty_lateral_scale"
        ),
        "sigma_level": policy.getfloat(
            "dynamic_uncertainty_sigma_level"
        ),
        "smear_steps": policy.getint("dynamic_time_smear_steps"),
        "smear_tau": policy.getfloat("dynamic_time_smear_tau"),
        "uncertainty_discount": policy.getfloat(
            "dynamic_uncertainty_discount"
        ),
        "dynamic_obs_clearance": policy.getfloat("dynamic_obs_clearance"),
        "pedestrian_radius": 0.2,
    }
    horizon = policy.getint("horizon")
    dt = policy.getfloat("dt")
    predictor = ORCAPredictor(
        horizon=horizon,
        dt=dt,
        neighbor_dist=policy.getfloat("orca_neighbor_dist"),
        max_neighbors=policy.getint("orca_max_neighbors"),
        time_horizon=policy.getfloat("orca_time_horizon"),
        time_horizon_obst=policy.getfloat("orca_time_horizon_obst"),
        max_speed=policy.getfloat("v_max"),
    )
    params = create_socnav_params()
    results = [
        evaluate_episode(
            scenario,
            predictor,
            params,
            config,
            args.observation_interval,
            navigation_end_time=navigation_end_times.get(scenario),
        )
        for scenario in args.scenarios
    ]

    longitudinal_axes = (
        config["uncertainty_base"]
        + config["uncertainty_growth"]
        * config["longitudinal_scale"]
        * (np.arange(horizon, dtype=float) + 1.0)
        * dt
    )
    lateral_axes = (
        config["uncertainty_base"]
        + config["uncertainty_growth"]
        * config["lateral_scale"]
        * (np.arange(horizon, dtype=float) + 1.0)
        * dt
    )
    aggregate = _aggregate(
        results,
        horizon,
        dt,
        longitudinal_axes,
        lateral_axes,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "method": (
            "post-run ORCA diagnostics reconstructed from the same "
            "prerecorded pedestrian trajectories"
        ),
        "scenarios": args.scenarios,
        "observation_interval_s": args.observation_interval,
        "prediction_dt_s": dt,
        "prediction_horizon_steps": horizon,
        "prediction_horizon_s": horizon * dt,
        "timeline_capped_by_navigation_run": bool(args.navigation_summary),
        "uncertainty": config,
        "aggregate_by_lead_time": aggregate,
        "episodes": [
            _compact_result(result, dt)
            for result in results
        ],
    }
    with open(os.path.join(args.output_dir, "calibration_summary.json"), "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")

    _write_csv(
        os.path.join(args.output_dir, "aggregate_by_lead_time.csv"),
        list(aggregate[0].keys()),
        aggregate,
    )
    selected_aggregate = [
        row
        for row in aggregate
        if any(
            abs(row["lead_time_s"] - lead_s) < dt / 2.0
            for lead_s in TIMELINE_LEADS_S
        )
    ]
    _write_csv(
        os.path.join(args.output_dir, "selected_lead_times.csv"),
        list(selected_aggregate[0].keys()),
        selected_aggregate,
    )
    timeline_rows = []
    for result in results:
        for row in result["timeline"]:
            for lead_s, metrics in row["leads"].items():
                timeline_rows.append(
                    {
                        "scenario": result["scenario"],
                        "observation_time_s": row["observation_time_s"],
                        "active_pedestrians": row["active_pedestrians"],
                        "lead_time_s": lead_s,
                        "count": metrics["count"],
                        "mean_error_m": metrics["mean_error_m"],
                        "p90_error_m": metrics["p90_error_m"],
                        "uncertainty_ellipse_coverage": metrics[
                            "uncertainty_ellipse_coverage"
                        ],
                        "uncertainty_smear_coverage": metrics[
                            "uncertainty_smear_coverage"
                        ],
                        "base_support_coverage": metrics[
                            "base_support_coverage"
                        ],
                        "cost_support_coverage": metrics[
                            "cost_support_coverage"
                        ],
                        "cost_support_smear_coverage": metrics[
                            "cost_support_smear_coverage"
                        ],
                    }
                )
    _write_csv(
        os.path.join(args.output_dir, "scene_timeline.csv"),
        list(timeline_rows[0].keys()),
        timeline_rows,
    )
    _plot_aggregate(args.output_dir, aggregate)
    _plot_timelines(args.output_dir, results)

    print("Wrote ORCA calibration report to %s" % args.output_dir)
    for result in results:
        overall = result["overall"]
        print(
            "%s: n=%d mean=%.3f m uncertainty=%.1f%% "
            "full-support-smeared=%.1f%%"
            % (
                result["scenario"],
                overall["count"],
                overall["mean_error_m"],
                100.0 * overall["uncertainty_ellipse_coverage"],
                100.0 * overall["cost_support_smear_coverage"],
            )
        )


if __name__ == "__main__":
    main()

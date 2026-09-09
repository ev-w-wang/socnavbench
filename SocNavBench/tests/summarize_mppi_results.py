import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np


def _scalar(value, reduction="mean"):
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=float)
        if array.size == 0:
            return None
        if reduction == "min":
            return float(np.nanmin(array))
        return float(np.nanmean(array))
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def load_results(root):
    rows = []
    pattern = os.path.join(root, "*", "episode_score_*.pkl")
    for filename in sorted(glob.glob(pattern)):
        with open(filename, "rb") as handle:
            metrics = pickle.load(handle)
        row = {
            "episode": os.path.basename(os.path.dirname(filename)),
            "success": bool(metrics.get("success", False)),
            "termination_cause": metrics.get("termination_cause"),
            "map": metrics.get("map"),
            "navigation_time": _scalar(metrics.get("total_sim_time_taken")),
            "path_length": _scalar(metrics.get("path_length")),
            "path_length_ratio": _scalar(metrics.get("path_length_ratio")),
            "goal_traversal_ratio": _scalar(metrics.get("goal_traversal_ratio")),
            "closest_pedestrian_distance": _scalar(
                metrics.get("closest_pedestrian_distance"), "min"
            ),
            "mean_robot_speed": _scalar(metrics.get("robot_speed")),
            "robot_motion_energy": _scalar(
                metrics.get("robot_motion_energy")
            ),
            "wall_wait_time": _scalar(metrics.get("wall_wait_time")),
        }
        rows.append(row)
    return rows


def summarize(rows):
    numeric_fields = [
        "navigation_time",
        "path_length",
        "path_length_ratio",
        "goal_traversal_ratio",
        "closest_pedestrian_distance",
        "mean_robot_speed",
        "robot_motion_energy",
        "wall_wait_time",
    ]
    summary = {
        "episodes": len(rows),
        "successes": sum(1 for row in rows if row["success"]),
        "success_rate": (
            float(sum(1 for row in rows if row["success"])) / len(rows)
            if rows
            else 0.0
        ),
        "termination_counts": {},
        "metrics": {},
    }
    for row in rows:
        cause = row["termination_cause"]
        summary["termination_counts"][cause] = (
            summary["termination_counts"].get(cause, 0) + 1
        )
    for field in numeric_fields:
        values = [
            row[field] for row in rows if row.get(field) is not None
        ]
        if values:
            summary["metrics"][field] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="tests/socnav/test_MPPI",
        help="MPPI episode output directory",
    )
    parser.add_argument(
        "--output",
        default="tests/socnav/test_MPPI/mppi_summary",
        help="Output path without extension",
    )
    args = parser.parse_args()
    rows = load_results(args.root)
    if not rows:
        raise RuntimeError("No MPPI episode scores found under {}".format(args.root))
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output + ".csv", "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.output + ".json", "w") as handle:
        json.dump(
            {"episodes": rows, "summary": summarize(rows)},
            handle,
            indent=2,
            sort_keys=True,
        )
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Aggregate the three uncertainty navigation profiles."""

import csv
import json
import os

import numpy as np


ROOT = os.path.join(os.path.dirname(__file__), "uncertainty_nav5")
PROFILES = ["zero", "ellipse", "circle"]
METRICS = [
    "navigation_time",
    "path_length",
    "robot_motion_energy",
    "closest_pedestrian_distance",
    "mean_robot_speed",
    "stopped_fraction",
    "stopped_time",
    "wall_wait_time",
]


def _mean(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def _load_profile(profile):
    path = os.path.join(ROOT, profile, "mppi_summary.json")
    with open(path) as source:
        return json.load(source)


def main():
    reports = {profile: _load_profile(profile) for profile in PROFILES}
    profile_rows = []
    episode_rows = []

    for profile in PROFILES:
        episodes = reports[profile]["episodes"]
        successes = [row for row in episodes if row["success"]]
        profile_row = {
            "profile": profile,
            "episodes": len(episodes),
            "successes": len(successes),
            "pedestrian_collisions": sum(
                row["termination_cause"] == "Pedestrian Collision"
                for row in episodes
            ),
            "minimum_pedestrian_distance": min(
                row["closest_pedestrian_distance"]
                for row in episodes
                if row.get("closest_pedestrian_distance") is not None
            ),
        }
        for metric in METRICS:
            profile_row["mean_{}".format(metric)] = _mean(episodes, metric)
            profile_row["successful_mean_{}".format(metric)] = _mean(
                successes,
                metric,
            )
        profile_rows.append(profile_row)

        for episode in episodes:
            row = {
                "profile": profile,
                "episode": episode["episode"],
                "success": episode["success"],
                "termination_cause": episode["termination_cause"],
            }
            for metric in METRICS:
                row[metric] = episode.get(metric)
            episode_rows.append(row)

    with open(os.path.join(ROOT, "comparison.json"), "w") as output:
        json.dump(
            {
                "profiles": profile_rows,
                "episodes": episode_rows,
                "stopped_speed_threshold_mps": 0.05,
            },
            output,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")

    with open(os.path.join(ROOT, "comparison.csv"), "w") as output:
        writer = csv.DictWriter(output, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    with open(os.path.join(ROOT, "episode_comparison.csv"), "w") as output:
        writer = csv.DictWriter(output, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    for row in profile_rows:
        print(
            "{profile}: {successes}/{episodes} success, "
            "{pedestrian_collisions} pedestrian collisions, "
            "successful path={successful_mean_path_length:.2f} m, "
            "nav={successful_mean_navigation_time:.2f} s, "
            "stopped={successful_mean_stopped_fraction:.1%}".format(**row)
        )


if __name__ == "__main__":
    main()

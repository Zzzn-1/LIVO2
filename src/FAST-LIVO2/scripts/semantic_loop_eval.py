#!/usr/bin/env python3
import argparse
import csv
import math
import os
from statistics import mean


def read_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalize_event_row(row)
            rows.append(row)
    return rows


def normalize_event_row(row):
    extra = row.pop(None, None)
    if extra:
        new_fields = [
            "query_x", "query_y", "query_z",
            "match_x", "match_y", "match_z",
            "rel_x", "rel_y", "rel_z", "rel_dist",
        ]
        for key, value in zip(new_fields, extra):
            row.setdefault(key, value)

    if "rel_dist" not in row or row.get("rel_dist", "") == "":
        if all(k in row for k in ("rel_x", "rel_y", "rel_z")):
            dx = to_float(row, "rel_x")
            dy = to_float(row, "rel_y")
            dz = to_float(row, "rel_z")
            row["rel_dist"] = str(math.sqrt(dx * dx + dy * dy + dz * dz))
        elif all(k in row for k in ("query_x", "query_y", "query_z", "match_x", "match_y", "match_z")):
            dx = to_float(row, "query_x") - to_float(row, "match_x")
            dy = to_float(row, "query_y") - to_float(row, "match_y")
            dz = to_float(row, "query_z") - to_float(row, "match_z")
            row["rel_dist"] = str(math.sqrt(dx * dx + dy * dy + dz * dz))
        elif "euclid" in row:
            row["rel_dist"] = row["euclid"]


def to_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def summarize(events):
    if not events:
        return None
    summary = {
        "count": len(events),
        "score_mean": mean(to_float(r, "semantic_score") for r in events),
        "geo_ratio_mean": mean(to_float(r, "geo_ratio") for r in events),
        "icp_fitness_mean": mean(to_float(r, "icp_fitness") for r in events),
        "dt_mean": mean(to_float(r, "dt") for r in events),
        "path_mean": mean(to_float(r, "path") for r in events),
        "euclid_mean": mean(to_float(r, "euclid") for r in events),
        "rel_dist_mean": mean(to_float(r, "rel_dist") for r in events),
    }
    return summary


def print_summary(summary):
    print("[SemanticLoopEval] event_count =", summary["count"])
    print("[SemanticLoopEval] mean_score = {:.4f}".format(summary["score_mean"]))
    print("[SemanticLoopEval] mean_geo_ratio = {:.4f}".format(summary["geo_ratio_mean"]))
    print("[SemanticLoopEval] mean_icp_fitness = {:.4f}".format(summary["icp_fitness_mean"]))
    print("[SemanticLoopEval] mean_dt_s = {:.3f}".format(summary["dt_mean"]))
    print("[SemanticLoopEval] mean_path_m = {:.3f}".format(summary["path_mean"]))
    print("[SemanticLoopEval] mean_euclid_m = {:.3f}".format(summary["euclid_mean"]))
    print("[SemanticLoopEval] mean_rel_dist_m = {:.3f}".format(summary["rel_dist_mean"]))


def latest_run_start_time(keyframes):
    if not keyframes:
        return None
    latest_start_idx = 0
    prev_id = None
    for i, row in enumerate(keyframes):
        kid = int(row.get("id", -1))
        if prev_id is not None and kid <= prev_id:
            latest_start_idx = i
        prev_id = kid
    return to_float(keyframes[latest_start_idx], "time")


def filter_latest_run(keyframes, events):
    start_time = latest_run_start_time(keyframes)
    if start_time is None:
        return keyframes, events
    filtered_keyframes = [r for r in keyframes if to_float(r, "time") >= start_time]
    filtered_events = [r for r in events if to_float(r, "query_time") >= start_time]
    return filtered_keyframes, filtered_events


def try_plot(keyframes, events, out_png):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[SemanticLoopEval] matplotlib not available, skip plotting.")
        return

    if not keyframes:
        print("[SemanticLoopEval] no keyframes, skip plotting.")
        return

    id_to_pose = {}
    xs = []
    ys = []
    for r in keyframes:
        kid = int(r["id"])
        x = to_float(r, "x")
        y = to_float(r, "y")
        id_to_pose[kid] = (x, y)
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(10, 7))
    plt.plot(xs, ys, "-", linewidth=1.0, label="semantic keyframe path")
    plt.scatter(xs, ys, s=8)

    for e in events:
        qid = int(e["query_id"])
        mid = int(e["match_id"])
        if qid not in id_to_pose or mid not in id_to_pose:
            continue
        qx, qy = id_to_pose[qid]
        mx, my = id_to_pose[mid]
        plt.plot([qx, mx], [qy, my], "r-", linewidth=0.8, alpha=0.7)

    plt.title("Semantic Loop Events")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print("[SemanticLoopEval] saved plot:", out_png)


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic loop events.")
    parser.add_argument("--log-dir", default=os.path.join("src", "FAST-LIVO2", "Log"))
    parser.add_argument("--plot", action="store_true", help="save 2D loop-edge plot")
    parser.add_argument("--plot-out", default="semantic_loop_eval.png")
    parser.add_argument("--all-runs", action="store_true", help="evaluate every appended run in the csv files")
    args = parser.parse_args()

    events_csv = os.path.join(args.log_dir, "semantic_loop_events.csv")
    keyframes_csv = os.path.join(args.log_dir, "semantic_keyframes.csv")

    if not os.path.exists(events_csv):
        print("[SemanticLoopEval] missing:", events_csv)
        return
    if not os.path.exists(keyframes_csv):
        print("[SemanticLoopEval] missing:", keyframes_csv)
        return

    events = read_csv(events_csv)
    keyframes = read_csv(keyframes_csv)
    if not args.all_runs:
        keyframes, events = filter_latest_run(keyframes, events)

    summary = summarize(events)
    if summary is None:
        print("[SemanticLoopEval] no events")
        return

    print_summary(summary)
    if args.plot:
        try_plot(keyframes, events, args.plot_out)


if __name__ == "__main__":
    main()

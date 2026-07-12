#!/usr/bin/env python3
import argparse
import csv
import os
import time

import rospy
from fast_livo.msg import Keyframe, LoopConstraint


def to_float(row, key, default=0.0):
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def to_int(row, key, default=-1):
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return int(v)
    except Exception:
        return default


def read_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def latest_run_start_time(keyframes):
    if not keyframes:
        return None
    latest_start_idx = 0
    prev_id = None
    for i, row in enumerate(keyframes):
        kid = to_int(row, "id", -1)
        if prev_id is not None and kid <= prev_id:
            latest_start_idx = i
        prev_id = kid
    return to_float(keyframes[latest_start_idx], "time", -1.0)


def filter_latest_run(keyframes, loop_edges):
    start_t = latest_run_start_time(keyframes)
    if start_t is None or start_t < 0.0:
        return keyframes, loop_edges
    kf = [r for r in keyframes if to_float(r, "time", -1.0) >= start_t]
    lp = [r for r in loop_edges if to_float(r, "to_time", -1.0) >= start_t]
    return kf, lp


def build_keyframes(rows):
    keyframes = []
    for r in rows:
        kf = {
            "id": to_int(r, "id", -1),
            "time": to_float(r, "time", -1.0),
            "x": to_float(r, "x", 0.0),
            "y": to_float(r, "y", 0.0),
            "z": to_float(r, "z", 0.0),
            "qx": to_float(r, "qx", 0.0),
            "qy": to_float(r, "qy", 0.0),
            "qz": to_float(r, "qz", 0.0),
            "qw": to_float(r, "qw", 1.0),
            "travel": to_float(r, "travel", 0.0),
            "source": "online_livmapper_semantic_keyframe",
        }
        if kf["id"] >= 0 and kf["time"] >= 0.0:
            keyframes.append(kf)
    keyframes.sort(key=lambda x: (x["time"], x["id"]))
    return keyframes


def build_loop_constraints(rows):
    loops = []
    for r in rows:
        if r.get("edge_type", "") != "semantic_loop":
            continue
        edge = {
            "edge_type": "semantic_loop",
            "from_id": to_int(r, "from_id", -1),
            "to_id": to_int(r, "to_id", -1),
            "from_time": to_float(r, "from_time", -1.0),
            "to_time": to_float(r, "to_time", -1.0),
            "rel_x": to_float(r, "rel_x", 0.0),
            "rel_y": to_float(r, "rel_y", 0.0),
            "rel_z": to_float(r, "rel_z", 0.0),
            "rel_qx": to_float(r, "rel_qx", 0.0),
            "rel_qy": to_float(r, "rel_qy", 0.0),
            "rel_qz": to_float(r, "rel_qz", 0.0),
            "rel_qw": to_float(r, "rel_qw", 1.0),
            "info_x": to_float(r, "info_x", 0.0),
            "info_y": to_float(r, "info_y", 0.0),
            "info_z": to_float(r, "info_z", 0.0),
            "info_roll": to_float(r, "info_roll", 0.0),
            "info_pitch": to_float(r, "info_pitch", 0.0),
            "info_yaw": to_float(r, "info_yaw", 0.0),
            "score": to_float(r, "score", 0.0),
            "geo_ratio": to_float(r, "geo_ratio", 0.0),
            "geo_inliers": to_int(r, "geo_inliers", 0),
            "icp_fitness": to_float(r, "icp_fitness", -1.0),
            "source": "pose_graph_edges.csv",
        }
        if edge["from_id"] >= 0 and edge["to_id"] >= 0 and edge["to_time"] >= 0.0:
            loops.append(edge)
    loops.sort(key=lambda x: (x["to_time"], x["to_id"]))
    return loops


def merge_timeline(keyframes, loops):
    timeline = []
    for kf in keyframes:
        timeline.append((kf["time"], "keyframe", kf))
    for edge in loops:
        timeline.append((edge["to_time"], "loop_constraint", edge))
    timeline.sort(key=lambda x: (x[0], 0 if x[1] == "keyframe" else 1))
    return timeline


def to_ros_time(sec):
    if sec is None or sec < 0.0:
        return rospy.Time(0)
    return rospy.Time.from_sec(sec)


def to_keyframe_msg(data):
    msg = Keyframe()
    msg.header.stamp = to_ros_time(data["time"])
    msg.header.frame_id = "map"
    msg.id = max(int(data["id"]), 0)
    msg.time = float(data["time"])
    msg.position.x = float(data["x"])
    msg.position.y = float(data["y"])
    msg.position.z = float(data["z"])
    msg.orientation.x = float(data["qx"])
    msg.orientation.y = float(data["qy"])
    msg.orientation.z = float(data["qz"])
    msg.orientation.w = float(data["qw"])
    msg.travel = float(data["travel"])
    msg.source = data.get("source", "")
    return msg


def to_loop_msg(data):
    msg = LoopConstraint()
    msg.header.stamp = to_ros_time(data["to_time"])
    msg.header.frame_id = "map"
    msg.from_id = max(int(data["from_id"]), 0)
    msg.to_id = max(int(data["to_id"]), 0)
    msg.from_time = float(data["from_time"])
    msg.to_time = float(data["to_time"])
    msg.rel_translation.x = float(data["rel_x"])
    msg.rel_translation.y = float(data["rel_y"])
    msg.rel_translation.z = float(data["rel_z"])
    msg.rel_rotation.x = float(data["rel_qx"])
    msg.rel_rotation.y = float(data["rel_qy"])
    msg.rel_rotation.z = float(data["rel_qz"])
    msg.rel_rotation.w = float(data["rel_qw"])
    msg.info_x = float(data["info_x"])
    msg.info_y = float(data["info_y"])
    msg.info_z = float(data["info_z"])
    msg.info_roll = float(data["info_roll"])
    msg.info_pitch = float(data["info_pitch"])
    msg.info_yaw = float(data["info_yaw"])
    msg.score = float(data["score"])
    msg.geo_ratio = float(data["geo_ratio"])
    msg.geo_inliers = max(int(data["geo_inliers"]), 0)
    msg.icp_fitness = float(data["icp_fitness"])
    msg.edge_type = data.get("edge_type", "semantic_loop")
    msg.source = data.get("source", "")
    return msg


def run_replay(args):
    nodes_csv = os.path.join(args.log_dir, args.nodes)
    edges_csv = os.path.join(args.log_dir, args.edges)
    if not os.path.exists(nodes_csv):
        rospy.logerr("[SemanticPGReplay] missing: %s", nodes_csv)
        return
    if not os.path.exists(edges_csv):
        rospy.logerr("[SemanticPGReplay] missing: %s", edges_csv)
        return

    keyframes_rows = read_csv(nodes_csv)
    edges_rows = read_csv(edges_csv)
    if not args.all_runs:
        keyframes_rows, edges_rows = filter_latest_run(keyframes_rows, edges_rows)

    keyframes = build_keyframes(keyframes_rows)
    loops = build_loop_constraints(edges_rows)
    timeline = merge_timeline(keyframes, loops)
    if not timeline:
        rospy.logwarn("[SemanticPGReplay] empty timeline, nothing to publish")
        return

    pub_kf = rospy.Publisher(args.keyframe_topic, Keyframe, queue_size=200)
    pub_loop = rospy.Publisher(args.loop_topic, LoopConstraint, queue_size=200)
    rospy.sleep(max(args.start_delay, 0.0))

    t0_data = timeline[0][0]
    t0_wall = time.time()
    pub_count_kf = 0
    pub_count_loop = 0
    rate_scale = max(args.replay_rate, 1e-6)

    rospy.loginfo("[SemanticPGReplay] start replay: %d keyframes, %d loop constraints, rate=%.3fx",
                  len(keyframes), len(loops), rate_scale)

    for t_data, kind, payload in timeline:
        if rospy.is_shutdown():
            break
        dt_data = t_data - t0_data
        target_wall = t0_wall + dt_data / rate_scale
        while not rospy.is_shutdown():
            now = time.time()
            remain = target_wall - now
            if remain <= 0.0:
                break
            time.sleep(min(0.002, remain))

        if kind == "keyframe":
            pub_kf.publish(to_keyframe_msg(payload))
            pub_count_kf += 1
        else:
            pub_loop.publish(to_loop_msg(payload))
            pub_count_loop += 1

    rospy.loginfo("[SemanticPGReplay] done: published keyframes=%d, loop_constraints=%d",
                  pub_count_kf, pub_count_loop)

    if args.keep_alive:
        rospy.loginfo("[SemanticPGReplay] keep_alive enabled, waiting for Ctrl-C ...")
        rospy.spin()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay pose_graph_nodes/edges CSV as ROS topics for online backend validation."
    )
    parser.add_argument("--log-dir", default=os.path.join("src", "FAST-LIVO2", "Log"))
    parser.add_argument("--nodes", default="pose_graph_nodes.csv")
    parser.add_argument("--edges", default="pose_graph_edges.csv")
    parser.add_argument("--keyframe-topic", default="/semantic_pg/keyframe")
    parser.add_argument("--loop-topic", default="/semantic_pg/loop_constraint")
    parser.add_argument("--replay-rate", type=float, default=1.0, help="time scale, 1.0=real-time, 2.0=2x faster")
    parser.add_argument("--start-delay", type=float, default=0.5)
    parser.add_argument("--all-runs", action="store_true", help="replay all appended runs")
    parser.add_argument("--keep-alive", action="store_true", help="keep node alive after replay")
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("semantic_pg_replay", anonymous=False)
    run_replay(args)


if __name__ == "__main__":
    main()

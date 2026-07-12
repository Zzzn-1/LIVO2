#!/usr/bin/env python3
import argparse
import csv
import math
import os


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


def normalize_quat(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return qx / n, qy / n, qz / n, qw / n


def quat_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vec_by_quat(v, q):
    vx, vy, vz = v
    qv = (vx, vy, vz, 0.0)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]


def relative_pose(from_pose, to_pose):
    fx, fy, fz, fqx, fqy, fqz, fqw = from_pose
    tx, ty, tz, tqx, tqy, tqz, tqw = to_pose
    q_from = normalize_quat(fqx, fqy, fqz, fqw)
    q_to = normalize_quat(tqx, tqy, tqz, tqw)
    q_rel = quat_mul(quat_conj(q_from), q_to)
    dp = (tx - fx, ty - fy, tz - fz)
    t_rel = rotate_vec_by_quat(dp, quat_conj(q_from))
    return (
        t_rel[0], t_rel[1], t_rel[2],
        q_rel[0], q_rel[1], q_rel[2], q_rel[3],
    )


def quat_angle_deg(q):
    qx, qy, qz, qw = normalize_quat(q[0], q[1], q[2], q[3])
    qw_abs = max(-1.0, min(1.0, abs(qw)))
    return 2.0 * math.acos(qw_abs) * 180.0 / math.pi


def edge_rel_dist(edge):
    x = float(edge.get("rel_x", 0.0))
    y = float(edge.get("rel_y", 0.0))
    z = float(edge.get("rel_z", 0.0))
    return math.sqrt(x * x + y * y + z * z)


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


def filter_latest_run(keyframes, loops):
    start_t = latest_run_start_time(keyframes)
    if start_t is None or start_t < 0.0:
        return keyframes, loops
    kf = [r for r in keyframes if to_float(r, "time", -1.0) >= start_t]
    lp = [r for r in loops if to_float(r, "to_time", -1.0) >= start_t]
    return kf, lp


def build_nodes(keyframes):
    nodes = []
    for r in keyframes:
        qx = to_float(r, "qx", 0.0)
        qy = to_float(r, "qy", 0.0)
        qz = to_float(r, "qz", 0.0)
        qw = to_float(r, "qw", 1.0)
        qx, qy, qz, qw = normalize_quat(qx, qy, qz, qw)
        nodes.append({
            "id": to_int(r, "id", -1),
            "time": to_float(r, "time", -1.0),
            "x": to_float(r, "x", 0.0),
            "y": to_float(r, "y", 0.0),
            "z": to_float(r, "z", 0.0),
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "travel": to_float(r, "travel", 0.0),
        })
    nodes = [n for n in nodes if n["id"] >= 0 and n["time"] >= 0.0]
    nodes.sort(key=lambda n: n["id"])
    return nodes


def build_semantic_odom_edges(nodes):
    edges = []
    for i in range(1, len(nodes)):
        a = nodes[i - 1]
        b = nodes[i]
        rel = relative_pose(
            (a["x"], a["y"], a["z"], a["qx"], a["qy"], a["qz"], a["qw"]),
            (b["x"], b["y"], b["z"], b["qx"], b["qy"], b["qz"], b["qw"]),
        )
        edges.append({
            "edge_type": "semantic_odom",
            "from_id": a["id"],
            "to_id": b["id"],
            "from_time": a["time"],
            "to_time": b["time"],
            "rel_x": rel[0],
            "rel_y": rel[1],
            "rel_z": rel[2],
            "rel_qx": rel[3],
            "rel_qy": rel[4],
            "rel_qz": rel[5],
            "rel_qw": rel[6],
            "info_x": 100.0,
            "info_y": 100.0,
            "info_z": 100.0,
            "info_roll": 100.0,
            "info_pitch": 100.0,
            "info_yaw": 100.0,
            "score": 1.0,
            "geo_ratio": 1.0,
            "geo_inliers": 0,
            "icp_fitness": 0.0,
        })
    return edges


def build_loop_edges(loops):
    edges = []
    for r in loops:
        if r.get("edge_type", "") != "semantic_loop":
            continue
        edges.append({
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
        })
    edges = [e for e in edges if e["from_id"] >= 0 and e["to_id"] >= 0]
    return edges


def write_nodes_csv(path, nodes):
    fieldnames = ["id", "time", "x", "y", "z", "qx", "qy", "qz", "qw", "travel"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for n in nodes:
            w.writerow(n)


def write_edges_csv(path, edges):
    fieldnames = [
        "edge_type", "from_id", "to_id", "from_time", "to_time",
        "rel_x", "rel_y", "rel_z", "rel_qx", "rel_qy", "rel_qz", "rel_qw",
        "info_x", "info_y", "info_z", "info_roll", "info_pitch", "info_yaw",
        "score", "geo_ratio", "geo_inliers", "icp_fitness",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in edges:
            w.writerow(e)


def summarize_edges(edges):
    odom = [e for e in edges if e["edge_type"] == "semantic_odom"]
    loops = [e for e in edges if e["edge_type"] == "semantic_loop"]
    return odom, loops


def check_pose_graph(nodes, edges):
    id_set = set(n["id"] for n in nodes)
    odom_edges, loop_edges = summarize_edges(edges)
    missing_node_edges = [
        e for e in edges if e["from_id"] not in id_set or e["to_id"] not in id_set
    ]
    non_contiguous_odom = [
        e for e in odom_edges if e["to_id"] != e["from_id"] + 1
    ]

    loop_dists = [edge_rel_dist(e) for e in loop_edges]
    loop_rots = [
        quat_angle_deg((e["rel_qx"], e["rel_qy"], e["rel_qz"], e["rel_qw"]))
        for e in loop_edges
    ]
    loop_id_gaps = [abs(e["to_id"] - e["from_id"]) for e in loop_edges]

    print("[PoseGraphCheck] nodes =", len(nodes))
    print("[PoseGraphCheck] edges =", len(edges))
    print("[PoseGraphCheck] odom_edges =", len(odom_edges))
    print("[PoseGraphCheck] loop_edges =", len(loop_edges))
    print("[PoseGraphCheck] missing_node_edges =", len(missing_node_edges))
    print("[PoseGraphCheck] non_contiguous_odom_edges =", len(non_contiguous_odom))
    if nodes:
        print("[PoseGraphCheck] node_id_range = {}..{}".format(nodes[0]["id"], nodes[-1]["id"]))
    if loop_edges:
        print("[PoseGraphCheck] loop_id_gap_min/max = {:.0f}/{:.0f}".format(min(loop_id_gaps), max(loop_id_gaps)))
        print("[PoseGraphCheck] loop_rel_dist_min/max_m = {:.3f}/{:.3f}".format(min(loop_dists), max(loop_dists)))
        print("[PoseGraphCheck] loop_rel_rot_min/max_deg = {:.3f}/{:.3f}".format(min(loop_rots), max(loop_rots)))
    if missing_node_edges:
        for e in missing_node_edges[:5]:
            print("[PoseGraphCheck] missing-node edge:", e["edge_type"], e["from_id"], "->", e["to_id"])
    if non_contiguous_odom:
        for e in non_contiguous_odom[:5]:
            print("[PoseGraphCheck] non-contiguous odom:", e["from_id"], "->", e["to_id"])


def plot_pose_graph(nodes, edges, out_png):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[PoseGraphPlot] matplotlib not available, skip plotting.")
        return

    if not nodes:
        print("[PoseGraphPlot] no nodes, skip plotting.")
        return

    id_to_node = {n["id"]: n for n in nodes}
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    odom_edges, loop_edges = summarize_edges(edges)

    plt.figure(figsize=(10, 7))
    plt.plot(xs, ys, "-", linewidth=1.0, color="#2d6cdf", label="odom/keyframe path")
    plt.scatter(xs, ys, s=12, color="#2d6cdf")

    for e in loop_edges:
        a = id_to_node.get(e["from_id"])
        b = id_to_node.get(e["to_id"])
        if not a or not b:
            continue
        plt.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#d62728", linewidth=1.0, alpha=0.8)

    for n in nodes[::max(1, len(nodes) // 20)]:
        plt.text(n["x"], n["y"], str(n["id"]), fontsize=7, alpha=0.8)

    plt.title("Offline Pose Graph")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print("[PoseGraphPlot] saved plot:", out_png)


def main():
    parser = argparse.ArgumentParser(description="Build offline pose-graph nodes/edges from semantic loop logs.")
    parser.add_argument("--log-dir", default=os.path.join("src", "FAST-LIVO2", "Log"))
    parser.add_argument("--all-runs", action="store_true", help="use all appended runs in CSV files")
    parser.add_argument("--nodes-out", default="pose_graph_nodes.csv")
    parser.add_argument("--edges-out", default="pose_graph_edges.csv")
    parser.add_argument("--check", action="store_true", help="print pose-graph consistency diagnostics")
    parser.add_argument("--plot", action="store_true", help="save a 2D pose-graph plot")
    parser.add_argument("--plot-out", default="pose_graph.png")
    args = parser.parse_args()

    keyframes_csv = os.path.join(args.log_dir, "semantic_keyframes.csv")
    loops_csv = os.path.join(args.log_dir, "semantic_loop_constraints.csv")
    if not os.path.exists(keyframes_csv):
        print("[PoseGraphBuild] missing:", keyframes_csv)
        return
    if not os.path.exists(loops_csv):
        print("[PoseGraphBuild] missing:", loops_csv)
        return

    keyframes = read_csv(keyframes_csv)
    loops = read_csv(loops_csv)
    if not args.all_runs:
        keyframes, loops = filter_latest_run(keyframes, loops)

    nodes = build_nodes(keyframes)
    odom_edges = build_semantic_odom_edges(nodes)
    loop_edges = build_loop_edges(loops)
    edges = odom_edges + loop_edges

    nodes_out = os.path.join(args.log_dir, args.nodes_out)
    edges_out = os.path.join(args.log_dir, args.edges_out)
    write_nodes_csv(nodes_out, nodes)
    write_edges_csv(edges_out, edges)

    print("[PoseGraphBuild] nodes:", len(nodes), "->", nodes_out)
    print("[PoseGraphBuild] odom_edges:", len(odom_edges))
    print("[PoseGraphBuild] loop_edges:", len(loop_edges))
    print("[PoseGraphBuild] total_edges:", len(edges), "->", edges_out)
    if args.check:
        check_pose_graph(nodes, edges)
    if args.plot:
        plot_out = os.path.join(args.log_dir, args.plot_out)
        plot_pose_graph(nodes, edges, plot_out)


if __name__ == "__main__":
    main()

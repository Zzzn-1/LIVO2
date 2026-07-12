#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys

import numpy as np


def import_gtsam():
    try:
        import gtsam
        return gtsam
    except Exception as exc:
        print("[GtsamPoseGraph] failed to import gtsam:", exc)
        print("[GtsamPoseGraph] install python gtsam first, then rerun this script.")
        return None


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
    qv = (v[0], v[1], v[2], 0.0)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]


def relative_pose_row(from_node, to_node):
    q_from = normalize_quat(from_node["qx"], from_node["qy"], from_node["qz"], from_node["qw"])
    q_to = normalize_quat(to_node["qx"], to_node["qy"], to_node["qz"], to_node["qw"])
    q_rel = quat_mul(quat_conj(q_from), q_to)
    dp = (
        to_node["x"] - from_node["x"],
        to_node["y"] - from_node["y"],
        to_node["z"] - from_node["z"],
    )
    t_rel = rotate_vec_by_quat(dp, quat_conj(q_from))
    return {
        "x": t_rel[0],
        "y": t_rel[1],
        "z": t_rel[2],
        "qx": q_rel[0],
        "qy": q_rel[1],
        "qz": q_rel[2],
        "qw": q_rel[3],
    }


def quat_angle_deg(q):
    qx, qy, qz, qw = normalize_quat(q[0], q[1], q[2], q[3])
    qw_abs = max(-1.0, min(1.0, abs(qw)))
    return 2.0 * math.acos(qw_abs) * 180.0 / math.pi


def quat_error_deg(a, b):
    qa = normalize_quat(a[0], a[1], a[2], a[3])
    qb = normalize_quat(b[0], b[1], b[2], b[3])
    return quat_angle_deg(quat_mul(quat_conj(qa), qb))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def quat_to_pose3(gtsam, x, y, z, qx, qy, qz, qw):
    qx, qy, qz, qw = normalize_quat(qx, qy, qz, qw)
    rot = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
    return gtsam.Pose3(rot, gtsam.Point3(x, y, z))


def pose3_to_row(node, pose):
    t = pose.translation()
    q = pose.rotation().toQuaternion()
    return {
        "id": node["id"],
        "time": node["time"],
        "x": float(t[0]),
        "y": float(t[1]),
        "z": float(t[2]),
        "qx": float(q.x()),
        "qy": float(q.y()),
        "qz": float(q.z()),
        "qw": float(q.w()),
        "travel": node.get("travel", 0.0),
    }


def build_nodes(rows):
    nodes = []
    for r in rows:
        qx, qy, qz, qw = normalize_quat(
            to_float(r, "qx", 0.0),
            to_float(r, "qy", 0.0),
            to_float(r, "qz", 0.0),
            to_float(r, "qw", 1.0),
        )
        node = {
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
        }
        if node["id"] >= 0:
            nodes.append(node)
    nodes.sort(key=lambda n: n["id"])
    return nodes


def build_edges(rows):
    edges = []
    for r in rows:
        edge_type = r.get("edge_type", "")
        if edge_type not in ("semantic_odom", "semantic_loop"):
            continue
        edge = {
            "edge_type": edge_type,
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
            "info_x": to_float(r, "info_x", 1.0),
            "info_y": to_float(r, "info_y", 1.0),
            "info_z": to_float(r, "info_z", 1.0),
            "info_roll": to_float(r, "info_roll", 1.0),
            "info_pitch": to_float(r, "info_pitch", 1.0),
            "info_yaw": to_float(r, "info_yaw", 1.0),
        }
        if edge["from_id"] >= 0 and edge["to_id"] >= 0:
            edges.append(edge)
    return edges


def build_path_points(rows):
    points = []
    for r in rows:
        x = to_float(r, "x", None)
        y = to_float(r, "y", None)
        z = to_float(r, "z", None)
        if x is None or y is None or z is None:
            continue
        points.append((x, y, z))
    return points


def make_noise_model(gtsam, edge, odom_weight_scale, loop_weight_scale, robust_delta):
    scale = loop_weight_scale if edge["edge_type"] == "semantic_loop" else odom_weight_scale
    scale = max(scale, 1e-9)

    # GTSAM Pose3 tangent order is rotation first, then translation.
    info = np.array([
        max(edge["info_roll"] * scale, 1e-9),
        max(edge["info_pitch"] * scale, 1e-9),
        max(edge["info_yaw"] * scale, 1e-9),
        max(edge["info_x"] * scale, 1e-9),
        max(edge["info_y"] * scale, 1e-9),
        max(edge["info_z"] * scale, 1e-9),
    ], dtype=float)
    sigmas = 1.0 / np.sqrt(info)
    base = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    if edge["edge_type"] == "semantic_loop" and robust_delta > 0.0:
        huber = gtsam.noiseModel.mEstimator.Huber.Create(robust_delta)
        return gtsam.noiseModel.Robust.Create(huber, base)
    return base


def symbol(gtsam, node_id):
    return gtsam.symbol("x", int(node_id))


def optimize_graph(gtsam, nodes, edges, args):
    id_to_node = {n["id"]: n for n in nodes}
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    for node in nodes:
        pose = quat_to_pose3(
            gtsam,
            node["x"], node["y"], node["z"],
            node["qx"], node["qy"], node["qz"], node["qw"],
        )
        initial.insert(symbol(gtsam, node["id"]), pose)

    if not nodes:
        raise RuntimeError("no nodes to optimize")

    first = nodes[0]
    prior_pose = quat_to_pose3(
        gtsam,
        first["x"], first["y"], first["z"],
        first["qx"], first["qy"], first["qz"], first["qw"],
    )
    prior_info = np.array([
        args.prior_rot_info,
        args.prior_rot_info,
        args.prior_rot_info,
        args.prior_pos_info,
        args.prior_pos_info,
        args.prior_pos_info,
    ], dtype=float)
    prior_sigmas = 1.0 / np.sqrt(np.maximum(prior_info, 1e-9))
    graph.add(gtsam.PriorFactorPose3(
        symbol(gtsam, first["id"]),
        prior_pose,
        gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas),
    ))

    used_edges = 0
    skipped_edges = 0
    for edge in edges:
        if edge["from_id"] not in id_to_node or edge["to_id"] not in id_to_node:
            skipped_edges += 1
            continue
        rel_pose = quat_to_pose3(
            gtsam,
            edge["rel_x"], edge["rel_y"], edge["rel_z"],
            edge["rel_qx"], edge["rel_qy"], edge["rel_qz"], edge["rel_qw"],
        )
        graph.add(gtsam.BetweenFactorPose3(
            symbol(gtsam, edge["from_id"]),
            symbol(gtsam, edge["to_id"]),
            rel_pose,
            make_noise_model(gtsam, edge, args.odom_weight_scale, args.loop_weight_scale, args.robust_delta),
        ))
        used_edges += 1

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(args.max_iterations)
    if args.verbosity:
        params.setVerbosity(args.verbosity)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    initial_error = graph.error(initial)
    result = optimizer.optimize()
    final_error = graph.error(result)

    optimized = []
    for node in nodes:
        pose = result.atPose3(symbol(gtsam, node["id"]))
        optimized.append(pose3_to_row(node, pose))

    return optimized, {
        "factors": graph.size(),
        "used_edges": used_edges,
        "skipped_edges": skipped_edges,
        "initial_error": initial_error,
        "final_error": final_error,
    }


def write_nodes_csv(path, nodes):
    fieldnames = ["id", "time", "x", "y", "z", "qx", "qy", "qz", "qw", "travel"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for node in nodes:
            w.writerow(node)


def node_change_summary(before, after):
    after_by_id = {n["id"]: n for n in after}
    pos_changes = []
    rot_changes = []
    for b in before:
        a = after_by_id.get(b["id"])
        if not a:
            continue
        dx = a["x"] - b["x"]
        dy = a["y"] - b["y"]
        dz = a["z"] - b["z"]
        pos_changes.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        rot_changes.append(quat_error_deg(
            (b["qx"], b["qy"], b["qz"], b["qw"]),
            (a["qx"], a["qy"], a["qz"], a["qw"]),
        ))
    return {
        "mean_pos": mean(pos_changes),
        "max_pos": max(pos_changes) if pos_changes else 0.0,
        "mean_rot": mean(rot_changes),
        "max_rot": max(rot_changes) if rot_changes else 0.0,
    }


def edge_residual(edge, nodes_by_id):
    a = nodes_by_id.get(edge["from_id"])
    b = nodes_by_id.get(edge["to_id"])
    if not a or not b:
        return None
    pred = relative_pose_row(a, b)
    tx = pred["x"] - edge["rel_x"]
    ty = pred["y"] - edge["rel_y"]
    tz = pred["z"] - edge["rel_z"]
    trans = math.sqrt(tx * tx + ty * ty + tz * tz)
    rot = quat_error_deg(
        (pred["qx"], pred["qy"], pred["qz"], pred["qw"]),
        (edge["rel_qx"], edge["rel_qy"], edge["rel_qz"], edge["rel_qw"]),
    )
    return trans, rot


def residual_summary(edges, nodes):
    nodes_by_id = {n["id"]: n for n in nodes}
    by_type = {
        "semantic_odom": {"trans": [], "rot": []},
        "semantic_loop": {"trans": [], "rot": [], "details": []},
    }
    for edge in edges:
        residual = edge_residual(edge, nodes_by_id)
        if residual is None:
            continue
        trans, rot = residual
        bucket = by_type.get(edge["edge_type"])
        if bucket is None:
            continue
        bucket["trans"].append(trans)
        bucket["rot"].append(rot)
        if edge["edge_type"] == "semantic_loop":
            bucket["details"].append((edge["from_id"], edge["to_id"], trans, rot))
    return by_type


def print_report(before, after, edges):
    changes = node_change_summary(before, after)
    before_res = residual_summary(edges, before)
    after_res = residual_summary(edges, after)

    print("[GtsamReport] mean_position_change_m:", "{:.6f}".format(changes["mean_pos"]))
    print("[GtsamReport] max_position_change_m:", "{:.6f}".format(changes["max_pos"]))
    print("[GtsamReport] mean_rotation_change_deg:", "{:.6f}".format(changes["mean_rot"]))
    print("[GtsamReport] max_rotation_change_deg:", "{:.6f}".format(changes["max_rot"]))

    for edge_type in ("semantic_odom", "semantic_loop"):
        b = before_res[edge_type]
        a = after_res[edge_type]
        print("[GtsamReport] {}_trans_residual_mean_before_after_m: {:.6f} -> {:.6f}".format(
            edge_type, mean(b["trans"]), mean(a["trans"])))
        print("[GtsamReport] {}_rot_residual_mean_before_after_deg: {:.6f} -> {:.6f}".format(
            edge_type, mean(b["rot"]), mean(a["rot"])))

    before_details = before_res["semantic_loop"]["details"]
    after_details = after_res["semantic_loop"]["details"]
    if before_details:
        print("[GtsamReport] semantic_loop_edges:")
        after_map = {(d[0], d[1]): d for d in after_details}
        for from_id, to_id, b_trans, b_rot in before_details:
            _, _, a_trans, a_rot = after_map.get((from_id, to_id), (from_id, to_id, 0.0, 0.0))
            print("[GtsamReport]   {}->{} trans_m {:.6f}->{:.6f}, rot_deg {:.6f}->{:.6f}".format(
                from_id, to_id, b_trans, a_trans, b_rot, a_rot))


def nearest_distance_to_path(point, path_points):
    px, py, pz = point
    best = None
    for x, y, z in path_points:
        dx = px - x
        dy = py - y
        dz = pz - z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if best is None or d < best:
            best = d
    return best if best is not None else 0.0


def print_corrected_path_report(optimized_nodes, corrected_path_csv, node_source="current optimization"):
    if not corrected_path_csv:
        return
    if not os.path.exists(corrected_path_csv):
        print("[GtsamReport] corrected_path_nearest_distance: skipped missing {}".format(
            corrected_path_csv))
        return

    path_points = build_path_points(read_csv(corrected_path_csv))
    if not optimized_nodes or not path_points:
        print("[GtsamReport] corrected_path_nearest_distance: skipped empty data")
        return

    print("[GtsamReport] corrected_path_reference_nodes:", node_source)
    distances = []
    details = []
    for node in optimized_nodes:
        d = nearest_distance_to_path((node["x"], node["y"], node["z"]), path_points)
        distances.append(d)
        details.append((d, node["id"]))

    details.sort(reverse=True)
    print("[GtsamReport] corrected_path_to_optimized_keyframes_mean_m: {:.6f}".format(
        mean(distances)))
    print("[GtsamReport] corrected_path_to_optimized_keyframes_max_m: {:.6f}".format(
        max(distances) if distances else 0.0))
    top = details[:5]
    if top:
        print("[GtsamReport] corrected_path_to_optimized_keyframes_top:")
        for d, node_id in top:
            print("[GtsamReport]   id={} nearest_m {:.6f}".format(node_id, d))


def plot_before_after(before, after, edges, out_png, plot_loops=False):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[GtsamPoseGraph] matplotlib not available, skip plotting.")
        return

    if not before or not after:
        print("[GtsamPoseGraph] no nodes to plot.")
        return

    bx = [n["x"] for n in before]
    by = [n["y"] for n in before]
    ax = [n["x"] for n in after]
    ay = [n["y"] for n in after]

    plt.figure(figsize=(10, 7))
    plt.plot(bx, by, "--", linewidth=1.0, color="#2d6cdf", alpha=0.75, label="before")
    plt.scatter(bx, by, s=10, color="#2d6cdf", alpha=0.75)
    plt.plot(ax, ay, "-", linewidth=1.4, color="#d62728", alpha=0.8, label="after")
    plt.scatter(ax, ay, s=10, color="#d62728", alpha=0.8)
    for b, a in zip(before, after):
        plt.plot([b["x"], a["x"]], [b["y"], a["y"]], color="#777777", linewidth=0.5, alpha=0.35)

    if plot_loops:
        after_by_id = {n["id"]: n for n in after}
        loop_label_used = False
        for edge in edges:
            if edge["edge_type"] != "semantic_loop":
                continue
            a = after_by_id.get(edge["from_id"])
            b = after_by_id.get(edge["to_id"])
            if not a or not b:
                continue
            label = "semantic loop" if not loop_label_used else None
            plt.plot([a["x"], b["x"]], [a["y"], b["y"]],
                     color="#2ca02c", linewidth=1.1, alpha=0.9, label=label)
            mx = 0.5 * (a["x"] + b["x"])
            my = 0.5 * (a["y"] + b["y"])
            plt.text(mx, my, "{}->{}".format(edge["from_id"], edge["to_id"]),
                     fontsize=7, color="#1f7a1f", alpha=0.9)
            loop_label_used = True

    for node in after[::max(1, len(after) // 20)]:
        plt.text(node["x"], node["y"], str(node["id"]), fontsize=7, alpha=0.8)

    plt.title("GTSAM Pose Graph: Before / After")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print("[GtsamPoseGraph] saved plot:", out_png)


def main():
    parser = argparse.ArgumentParser(description="Offline GTSAM batch optimizer for pose_graph_nodes/edges CSV.")
    parser.add_argument("--log-dir", default=os.path.join("src", "FAST-LIVO2", "Log"))
    parser.add_argument("--online", action="store_true",
                        help="use online-exported graph csv: pose_graph_nodes_online.csv / pose_graph_edges_online.csv")
    parser.add_argument("--nodes", default="pose_graph_nodes.csv")
    parser.add_argument("--edges", default="pose_graph_edges.csv")
    parser.add_argument("--out-nodes", default="pose_graph_optimized_nodes.csv")
    parser.add_argument("--plot-out", default="pose_graph_before_after.png")
    parser.add_argument("--corrected-path-csv", default="",
                        help="optional corrected path CSV for nearest-distance report")
    parser.add_argument("--corrected-reference-nodes", default="",
                        help="optional optimized-node CSV used only for corrected-path nearest-distance report")
    parser.add_argument("--odom-weight-scale", type=float, default=0.5)
    parser.add_argument("--loop-weight-scale", type=float, default=0.5)
    parser.add_argument("--robust-delta", type=float, default=1.0)
    parser.add_argument("--prior-pos-info", type=float, default=1e6)
    parser.add_argument("--prior-rot-info", type=float, default=1e6)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--verbosity", default="", help="GTSAM LM verbosity string, empty by default")
    parser.add_argument("--report", action="store_true", help="print before/after residual and node-change report")
    parser.add_argument("--plot-loops", action="store_true", help="draw semantic loop edges on the before/after plot")
    args = parser.parse_args()

    if args.online:
        if args.nodes == "pose_graph_nodes.csv":
            args.nodes = "pose_graph_nodes_online.csv"
        if args.edges == "pose_graph_edges.csv":
            args.edges = "pose_graph_edges_online.csv"
        if args.out_nodes == "pose_graph_optimized_nodes.csv":
            args.out_nodes = "pose_graph_optimized_nodes_online.csv"
        if args.plot_out == "pose_graph_before_after.png":
            args.plot_out = "pose_graph_before_after_online.png"
        if not args.corrected_path_csv:
            args.corrected_path_csv = "corrected_path_online.csv"
        if not args.corrected_reference_nodes:
            args.corrected_reference_nodes = "pose_graph_optimized_nodes_online_backend.csv"

    gtsam = import_gtsam()
    if gtsam is None:
        sys.exit(1)

    nodes_path = os.path.join(args.log_dir, args.nodes)
    edges_path = os.path.join(args.log_dir, args.edges)
    if not os.path.exists(nodes_path):
        print("[GtsamPoseGraph] missing:", nodes_path)
        sys.exit(1)
    if not os.path.exists(edges_path):
        print("[GtsamPoseGraph] missing:", edges_path)
        sys.exit(1)

    nodes = build_nodes(read_csv(nodes_path))
    edges = build_edges(read_csv(edges_path))
    optimized, summary = optimize_graph(gtsam, nodes, edges, args)

    out_nodes = os.path.join(args.log_dir, args.out_nodes)
    plot_out = os.path.join(args.log_dir, args.plot_out)
    write_nodes_csv(out_nodes, optimized)
    plot_before_after(nodes, optimized, edges, plot_out, args.plot_loops)

    print("[GtsamPoseGraph] nodes:", len(nodes))
    print("[GtsamPoseGraph] edges:", len(edges))
    print("[GtsamPoseGraph] factors:", summary["factors"])
    print("[GtsamPoseGraph] used_edges:", summary["used_edges"])
    print("[GtsamPoseGraph] skipped_edges:", summary["skipped_edges"])
    print("[GtsamPoseGraph] initial_error:", "{:.6f}".format(summary["initial_error"]))
    print("[GtsamPoseGraph] final_error:", "{:.6f}".format(summary["final_error"]))
    print("[GtsamPoseGraph] optimized_nodes:", out_nodes)
    if args.report:
        print_report(nodes, optimized, edges)
        corrected_path_csv = args.corrected_path_csv
        if corrected_path_csv and not os.path.isabs(corrected_path_csv):
            corrected_path_csv = os.path.join(args.log_dir, corrected_path_csv)
        corrected_reference_nodes = args.corrected_reference_nodes
        corrected_reference_source = "current optimization"
        corrected_reference = optimized
        if corrected_reference_nodes:
            if not os.path.isabs(corrected_reference_nodes):
                corrected_reference_nodes = os.path.join(args.log_dir, corrected_reference_nodes)
            if os.path.exists(corrected_reference_nodes):
                corrected_reference = build_nodes(read_csv(corrected_reference_nodes))
                corrected_reference_source = corrected_reference_nodes
            else:
                corrected_reference_source = "current optimization (missing {})".format(
                    corrected_reference_nodes)
        print_corrected_path_report(corrected_reference, corrected_path_csv, corrected_reference_source)


if __name__ == "__main__":
    main()

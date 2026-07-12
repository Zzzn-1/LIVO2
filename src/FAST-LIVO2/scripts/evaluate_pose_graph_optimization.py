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
    qv = (v[0], v[1], v[2], 0.0)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]


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
            "rel_x": to_float(r, "rel_x", 0.0),
            "rel_y": to_float(r, "rel_y", 0.0),
            "rel_z": to_float(r, "rel_z", 0.0),
            "rel_qx": to_float(r, "rel_qx", 0.0),
            "rel_qy": to_float(r, "rel_qy", 0.0),
            "rel_qz": to_float(r, "rel_qz", 0.0),
            "rel_qw": to_float(r, "rel_qw", 1.0),
        }
        if edge["from_id"] >= 0 and edge["to_id"] >= 0:
            edges.append(edge)
    return edges


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


def node_changes(before_nodes, after_nodes):
    after_by_id = {n["id"]: n for n in after_nodes}
    changes = []
    for b in before_nodes:
        a = after_by_id.get(b["id"])
        if not a:
            continue
        dx = a["x"] - b["x"]
        dy = a["y"] - b["y"]
        dz = a["z"] - b["z"]
        pos = math.sqrt(dx * dx + dy * dy + dz * dz)
        rot = quat_error_deg(
            (b["qx"], b["qy"], b["qz"], b["qw"]),
            (a["qx"], a["qy"], a["qz"], a["qw"]),
        )
        changes.append({
            "id": b["id"],
            "pos_change": pos,
            "rot_change_deg": rot,
            "before_x": b["x"],
            "before_y": b["y"],
            "before_z": b["z"],
            "after_x": a["x"],
            "after_y": a["y"],
            "after_z": a["z"],
        })
    changes.sort(key=lambda r: (r["pos_change"], r["rot_change_deg"]), reverse=True)
    return changes


def residual_rows(edges, before_nodes, after_nodes):
    before_by_id = {n["id"]: n for n in before_nodes}
    after_by_id = {n["id"]: n for n in after_nodes}
    rows = []
    for e in edges:
        before = edge_residual(e, before_by_id)
        after = edge_residual(e, after_by_id)
        if before is None or after is None:
            continue
        rows.append({
            "edge_type": e["edge_type"],
            "from_id": e["from_id"],
            "to_id": e["to_id"],
            "trans_before": before[0],
            "trans_after": after[0],
            "rot_before_deg": before[1],
            "rot_after_deg": after[1],
            "trans_delta": after[0] - before[0],
            "rot_delta_deg": after[1] - before[1],
        })
    return rows


def print_top_node_changes(changes, top_k):
    print("[PoseGraphEval] top-{} node changes:".format(top_k))
    for r in changes[:top_k]:
        print("[PoseGraphEval]   id={} pos_change_m={:.6f}, rot_change_deg={:.6f}".format(
            r["id"], r["pos_change"], r["rot_change_deg"]))


def print_top_loop_residuals(loop_rows, top_k):
    print("[PoseGraphEval] top-{} loop residuals after optimization:".format(top_k))
    ranked = sorted(loop_rows, key=lambda r: (r["trans_after"], r["rot_after_deg"]), reverse=True)
    for r in ranked[:top_k]:
        print("[PoseGraphEval]   {}->{} trans_m {:.6f}->{:.6f}, rot_deg {:.6f}->{:.6f}".format(
            r["from_id"], r["to_id"],
            r["trans_before"], r["trans_after"],
            r["rot_before_deg"], r["rot_after_deg"]))


def print_loop_table(loop_rows):
    if not loop_rows:
        print("[PoseGraphEval] no semantic_loop edges")
        return
    print("[PoseGraphEval] loop residual table:")
    print("[PoseGraphEval]   edge,trans_before_m,trans_after_m,rot_before_deg,rot_after_deg")
    for r in loop_rows:
        print("[PoseGraphEval]   {}->{},{:.6f},{:.6f},{:.6f},{:.6f}".format(
            r["from_id"], r["to_id"], r["trans_before"], r["trans_after"],
            r["rot_before_deg"], r["rot_after_deg"]))


def health_warnings(changes, residuals, args):
    warnings = []
    for r in changes:
        if r["pos_change"] > args.max_position_change_warn:
            warnings.append("node {} position changed {:.3f}m > {:.3f}m".format(
                r["id"], r["pos_change"], args.max_position_change_warn))
        if r["rot_change_deg"] > args.max_rotation_change_warn:
            warnings.append("node {} rotation changed {:.3f}deg > {:.3f}deg".format(
                r["id"], r["rot_change_deg"], args.max_rotation_change_warn))

    loop_rows = [r for r in residuals if r["edge_type"] == "semantic_loop"]
    for r in loop_rows:
        if r["trans_after"] > args.loop_residual_after_warn:
            warnings.append("loop {}->{} residual after {:.3f}m > {:.3f}m".format(
                r["from_id"], r["to_id"], r["trans_after"], args.loop_residual_after_warn))

    odom_rows = [r for r in residuals if r["edge_type"] == "semantic_odom"]
    odom_before = mean([r["trans_before"] for r in odom_rows])
    odom_after = mean([r["trans_after"] for r in odom_rows])
    if odom_after > args.odom_residual_after_warn and odom_after > odom_before * args.odom_residual_growth_warn:
        warnings.append("odom translation residual grew {:.6f}m -> {:.6f}m".format(odom_before, odom_after))
    return warnings


def print_summary(before_nodes, after_nodes, residuals, changes):
    odom_rows = [r for r in residuals if r["edge_type"] == "semantic_odom"]
    loop_rows = [r for r in residuals if r["edge_type"] == "semantic_loop"]
    print("[PoseGraphEval] nodes =", len(before_nodes))
    print("[PoseGraphEval] optimized_nodes =", len(after_nodes))
    print("[PoseGraphEval] odom_edges =", len(odom_rows))
    print("[PoseGraphEval] loop_edges =", len(loop_rows))
    print("[PoseGraphEval] mean_position_change_m = {:.6f}".format(mean([r["pos_change"] for r in changes])))
    print("[PoseGraphEval] max_position_change_m = {:.6f}".format(max([r["pos_change"] for r in changes]) if changes else 0.0))
    print("[PoseGraphEval] mean_rotation_change_deg = {:.6f}".format(mean([r["rot_change_deg"] for r in changes])))
    print("[PoseGraphEval] max_rotation_change_deg = {:.6f}".format(max([r["rot_change_deg"] for r in changes]) if changes else 0.0))
    print("[PoseGraphEval] odom_trans_mean_before_after_m = {:.6f} -> {:.6f}".format(
        mean([r["trans_before"] for r in odom_rows]), mean([r["trans_after"] for r in odom_rows])))
    print("[PoseGraphEval] loop_trans_mean_before_after_m = {:.6f} -> {:.6f}".format(
        mean([r["trans_before"] for r in loop_rows]), mean([r["trans_after"] for r in loop_rows])))
    print("[PoseGraphEval] loop_rot_mean_before_after_deg = {:.6f} -> {:.6f}".format(
        mean([r["rot_before_deg"] for r in loop_rows]), mean([r["rot_after_deg"] for r in loop_rows])))


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimized pose graph CSV outputs.")
    parser.add_argument("--log-dir", default=os.path.join("src", "FAST-LIVO2", "Log"))
    parser.add_argument("--nodes", default="pose_graph_nodes.csv")
    parser.add_argument("--optimized-nodes", default="pose_graph_optimized_nodes.csv")
    parser.add_argument("--edges", default="pose_graph_edges.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-position-change-warn", type=float, default=0.5)
    parser.add_argument("--max-rotation-change-warn", type=float, default=5.0)
    parser.add_argument("--loop-residual-after-warn", type=float, default=0.5)
    parser.add_argument("--odom-residual-after-warn", type=float, default=0.05)
    parser.add_argument("--odom-residual-growth-warn", type=float, default=10.0)
    args = parser.parse_args()

    nodes_path = os.path.join(args.log_dir, args.nodes)
    opt_nodes_path = os.path.join(args.log_dir, args.optimized_nodes)
    edges_path = os.path.join(args.log_dir, args.edges)
    for path in (nodes_path, opt_nodes_path, edges_path):
        if not os.path.exists(path):
            print("[PoseGraphEval] missing:", path)
            return

    before_nodes = build_nodes(read_csv(nodes_path))
    after_nodes = build_nodes(read_csv(opt_nodes_path))
    edges = build_edges(read_csv(edges_path))

    changes = node_changes(before_nodes, after_nodes)
    residuals = residual_rows(edges, before_nodes, after_nodes)
    loop_rows = [r for r in residuals if r["edge_type"] == "semantic_loop"]

    print_summary(before_nodes, after_nodes, residuals, changes)
    print_top_node_changes(changes, args.top_k)
    print_top_loop_residuals(loop_rows, args.top_k)
    print_loop_table(loop_rows)

    warnings = health_warnings(changes, residuals, args)
    if warnings:
        print("[PoseGraphEval] health = WARN")
        for w in warnings[:20]:
            print("[PoseGraphEval]   warning:", w)
    else:
        print("[PoseGraphEval] health = OK")


if __name__ == "__main__":
    main()

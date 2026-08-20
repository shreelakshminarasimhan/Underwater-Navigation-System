"""
waypoint_path.py  —  CCZ survey waypoint node generator
================================================================
CONCEPT
-------
Replaces the lawnmower strip abstraction with discrete WAYPOINT NODES.

The CCZ is divided into a grid of survey nodes spaced STRIP_WIDTH_M apart
in Y and NODE_SPACING_M apart in X.  The submarine navigates node-to-node
in a boustrophedon (alternating left-right) order.  Once the sub reaches
a node it is marked VISITED and never revisited.

After a USBL fix the sub plots a course to the NEXT UNVISITED node from
its corrected true position.  This makes the "waypoint jump" visible and
meaningful: the sub never backtracks.

WHY THIS IS BETTER
------------------
- Each waypoint is a clear, numbered target in 3D space.
- After a GPS fix the sub immediately heads to the next unvisited node
  rather than replaying any path segment.
- Coverage is trivially trackable: visited / total nodes.
- The 3D viewer can show the node cloud and the sub trajectory through it.

OUTPUT
------
build_waypoints(ccz)  ->  dict with:
    nodes_x, nodes_y  [m]   all node E/N coordinates (flat list)
    nodes_z           [m]   all node depths (all SUB_DEPTH_M for now)
    n_nodes           int   total node count
    visited           set   node indices already reached

get_next_node(wp_dict)  ->  (x, y, z, idx)  next unvisited node
mark_visited(wp_dict, idx)  ->  None
"""

import math

# ── ADJUSTABLE PARAMETERS ─────────────────────────────────────────────────
STRIP_WIDTH_M   = 78_000    # m  N-S spacing between survey rows
NODE_SPACING_M  = 150_000   # m  E-W spacing of nodes within each row
BOUNDARY_MARGIN = 10_000    # m  clearance from CCZ edge
SUB_DEPTH_M     = 600.0     # m  operating depth (matches sub_model.py)
WAYPOINT_RADIUS = 8_000     # m  capture radius — larger for big CCZ grid
# ──────────────────────────────────────────────────────────────────────────

MISSION_DAYS    = 42.0      # hard stop after this many sim-days


def build_waypoints(ccz):
    """
    Build ordered waypoint node list for the CCZ survey.
    Populates ccz with 'waypoints' dict.
    Returns the waypoints dict.
    """
    x_min = ccz["x_lim"][0] + BOUNDARY_MARGIN
    x_max = ccz["x_lim"][1] - BOUNDARY_MARGIN
    y_min = ccz["y_lim"][0] + BOUNDARY_MARGIN
    y_max = ccz["y_lim"][1] - BOUNDARY_MARGIN

    # Build rows south→north, alternating E→W and W→E
    nodes_x, nodes_y, nodes_z = [], [], []
    row = 0
    y = y_min
    while y <= y_max + STRIP_WIDTH_M * 0.1:
        # Build E-W node positions for this row
        row_x = []
        x = x_min
        while x <= x_max + NODE_SPACING_M * 0.1:
            row_x.append(min(x, x_max))
            x += NODE_SPACING_M
        # Also always include the far end
        if row_x[-1] < x_max - NODE_SPACING_M * 0.05:
            row_x.append(x_max)

        # Reverse every other row (boustrophedon)
        if row % 2 == 1:
            row_x = row_x[::-1]

        for rx in row_x:
            nodes_x.append(rx)
            nodes_y.append(y)
            nodes_z.append(SUB_DEPTH_M)

        y += STRIP_WIDTH_M
        row += 1

    wp = {
        "nodes_x":  nodes_x,
        "nodes_y":  nodes_y,
        "nodes_z":  nodes_z,
        "n_nodes":  len(nodes_x),
        "visited":  set(),
        "current_target": 0,   # index of node sub is heading toward
    }

    ccz["waypoints"] = wp

    # Compute total straight-line distance for mission time estimate
    total_dist = 0.0
    for i in range(len(nodes_x) - 1):
        dx = nodes_x[i+1] - nodes_x[i]
        dy = nodes_y[i+1] - nodes_y[i]
        total_dist += math.hypot(dx, dy)

    from sub_model import CRUISE_SPEED_MS
    est_days = total_dist / (CRUISE_SPEED_MS * 86400)

    print(f"[waypoint_path] Nodes    : {wp['n_nodes']} waypoints")
    print(f"[waypoint_path] Rows     : {row}  x  ~{len(row_x)} nodes/row")
    print(f"[waypoint_path] Est dist : {total_dist/1e3:.0f} km")
    print(f"[waypoint_path] Est time : {est_days:.1f} days at {CRUISE_SPEED_MS} m/s")
    print(f"[waypoint_path] Mission  : hard stop at {MISSION_DAYS} sim-days")

    return wp


def get_next_unvisited(wp):
    """
    Return (x, y, z, idx) of the next unvisited node, in order.
    Returns None if all nodes visited.
    """
    for idx in range(wp["n_nodes"]):
        if idx not in wp["visited"]:
            return (wp["nodes_x"][idx],
                    wp["nodes_y"][idx],
                    wp["nodes_z"][idx],
                    idx)
    return None


def mark_visited(wp, idx):
    """Mark node idx as visited."""
    wp["visited"].add(idx)
    wp["current_target"] = idx + 1
    print(f"[waypoint_path] Node {idx} visited  —  "
          f"{len(wp['visited'])}/{wp['n_nodes']} complete")


def get_path_for_display(wp):
    """
    Return ordered node x/y lists for display (all nodes in survey order).
    """
    return wp["nodes_x"], wp["nodes_y"]
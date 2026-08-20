"""
sub_model.py  —  Submarine kinematic state and waypoint-node tracking
================================================================
PURPOSE
-------
Maintains the submarine's TRUE position and heading.
The sub follows DISCRETE WAYPOINT NODES at constant cruise speed.

KEY CHANGE: Waypoint-node navigation
- The sub navigates to node[current_target] in the waypoints dict.
- When within WAYPOINT_RADIUS of a node, marks it visited and advances.
- After a USBL fix, sub immediately heads to the next unvisited node
  from its corrected true position (no backtracking).

TWO CLASSES
-----------
SubState   — data container for all sub fields
SubModel   — step() method advances sub one dt_s of simulated time

FIELDS (SubState)
-----------------
    x_true      [m]    true East position
    y_true      [m]    true North position
    z_true      [m]    true depth (positive down from surface)
    heading_rad [rad]  heading (0=East, pi/2=North)
    speed_ms    [m/s]  cruise speed
    vx_true     [m/s]  true East velocity
    vy_true     [m/s]  true North velocity
    dist_total  [m]    odometer
    mission_complete   True when all nodes visited or time limit reached
"""

import math
from waypoint_path import mark_visited, get_next_unvisited, WAYPOINT_RADIUS

# ── ADJUSTABLE PARAMETERS ─────────────────────────────────────────────────
CRUISE_SPEED_MS = 3.0    # m/s  survey cruise speed (~5.8 knots)
DEPTH_M         = 600    # m    operating depth
# ──────────────────────────────────────────────────────────────────────────


class SubState:
    """Plain data container — all fields public, no methods."""
    def __init__(self, wp):
        # Start at first waypoint node
        self.x_true    = wp["nodes_x"][0]
        self.y_true    = wp["nodes_y"][0]
        self.z_true    = DEPTH_M          # depth in metres (positive down)
        self.speed_ms  = CRUISE_SPEED_MS
        self.dist_total = 0.0
        self.mission_complete = False

        # Mark node 0 as start (visited immediately)
        wp["visited"].add(0)
        wp["current_target"] = 1

        # Initial heading toward first real target node
        dx = wp["nodes_x"][1] - self.x_true
        dy = wp["nodes_y"][1] - self.y_true
        self.heading_rad = math.atan2(dy, dx)
        self.vx_true = CRUISE_SPEED_MS * math.cos(self.heading_rad)
        self.vy_true = CRUISE_SPEED_MS * math.sin(self.heading_rad)

        print(f"[sub_model] Initialised — speed {CRUISE_SPEED_MS} m/s,  depth {DEPTH_M} m")
        print(f"[sub_model] Start position: ({self.x_true/1e3:.0f}, {self.y_true/1e3:.0f}) km")


class SubModel:
    """Handles sub kinematics — call step() every simulation tick."""

    def __init__(self, wp):
        self.wp    = wp
        self.state = SubState(wp)

    def redirect_to_next_node(self):
        """
        After a USBL fix: plot course from current TRUE position to next
        unvisited node.  Does NOT change visited set — just updates heading.
        """
        nxt = get_next_unvisited(self.wp)
        if nxt is None:
            self.state.mission_complete = True
            return
        nx, ny, nz, nidx = nxt
        dx = nx - self.state.x_true
        dy = ny - self.state.y_true
        self.state.heading_rad = math.atan2(dy, dx)
        self.state.vx_true = self.state.speed_ms * math.cos(self.state.heading_rad)
        self.state.vy_true = self.state.speed_ms * math.sin(self.state.heading_rad)
        print(f"[sub_model] Redirect → Node {nidx} at "
              f"({nx/1e3:.0f}, {ny/1e3:.0f}) km")

    def step(self, dt_s):
        """Advance sub by dt_s simulated seconds."""
        s = self.state
        wp = self.wp

        if s.mission_complete:
            s.vx_true = 0.0
            s.vy_true = 0.0
            return

        # Get current target node
        nxt = get_next_unvisited(wp)
        if nxt is None:
            s.mission_complete = True
            print("[sub_model] All waypoint nodes visited — mission complete.")
            return

        nx, ny, nz, nidx = nxt

        # Vector to target node
        dx = nx - s.x_true
        dy = ny - s.y_true
        dist = math.hypot(dx, dy)

        # Arrived at node?
        if dist < WAYPOINT_RADIUS:
            mark_visited(wp, nidx)
            # Re-fetch next target
            nxt2 = get_next_unvisited(wp)
            if nxt2 is None:
                s.mission_complete = True
                s.vx_true = 0.0
                s.vy_true = 0.0
                print("[sub_model] Final node reached — mission complete.")
                return
            dx = nxt2[0] - s.x_true
            dy = nxt2[1] - s.y_true

        # Update heading and velocity
        s.heading_rad = math.atan2(dy, dx)
        s.vx_true     = s.speed_ms * math.cos(s.heading_rad)
        s.vy_true     = s.speed_ms * math.sin(s.heading_rad)

        # Advance true position
        s.x_true     += s.vx_true * dt_s
        s.y_true     += s.vy_true * dt_s
        s.dist_total += s.speed_ms * dt_s
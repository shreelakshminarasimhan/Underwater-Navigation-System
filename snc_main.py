"""
snc_main.py  —  SNC Simulator  Stage 1: Autonomous Submarine
=============================================================
RUN:   python snc_main.py

KEY CHANGES
-----------
1. WAYPOINT-NODE NAVIGATION
   The sub navigates to discrete numbered nodes in survey order.
   After each Buoy fix, it plots a straight course to the NEXT
   unvisited node — no lawnmower replay, no backtracking.

2. 42-DAY HARD STOP
   Simulation stops automatically at MISSION_DAYS_MAX sim-days.
   Sub surfaces and panel shows mission summary.

3. TRUE 3D VIEW (P1)
   Uses mpl_toolkits.mplot3d for a genuine 3D axes.
   Shows East (X), North (Y), Depth (Z) with full 3-axis rotation.
   Waterline plane at Z=0, no seafloor, max depth 1000 m.
   Sub marker and trail visible in all 3 dimensions.
   Buoy animates vertically on deployment.

LAYOUT
------
  ┌─────────────────────────┬─────────────────────────────────┐
  │  3D VIEW  (P1)          │  WAYPOINT MAP  (P2)             │
  │  East × North × Depth   │  Top-down CCZ + node progress   │
  ├───────────────┬──────────┴─────────────────────────────────┤
  │  SUB PARAMS   │  POSITION ERROR GRAPH  (P4)               │
  │  (P3)         │                                            │
  └───────────────┴────────────────────────────────────────────┘
"""

import time
import math
import numpy as np
import matplotlib
matplotlib.use("MacOSX")          # Mac — change to "Qt5Agg" on Windows/Linux
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, FancyArrow
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.art3d as art3d

from ccz_grid       import build_ccz_grid
from waypoint_path  import build_waypoints, get_next_unvisited, MISSION_DAYS
from sub_model      import SubModel
from drift_model    import DriftModel
from buoy_logic     import BuoyLogic

# ── Simulation parameters ──────────────────────────────────────────────────
SPEEDUP_INIT    = 15_000   # sim-s per real-s  (42 days → ~4 min demo)
SPEEDUP_MIN     =    500
SPEEDUP_MAX     = 80_000
FRAME_MS        = 50       # 20 fps
STEPS           = 10       # physics sub-steps per animation frame
MISSION_DAYS_MAX = MISSION_DAYS   # hard stop (from waypoint_path.py)

ERR_SAFE    = 100.0     # orange ref line [m]
ERR_DEPLOY  = 200.0     # red threshold — buoy deploys [m]
ERR_CRIT    = 1000.0    # grey mission-critical ref [m]

DRIFT_SCALE  = 100      # visual exaggeration of nav estimate offset in P2

# Flash overlay duration
FLASH_FRAMES = 10       # 10 × 50 ms = 0.5 s real time

# 3D view depth extents
MAX_DEPTH_M = 1000.0
SUB_DEPTH_M = 600.0     # must match sub_model.py DEPTH_M

# Mission-end surfacing animation
SURFACE_FRAMES = 60     # 3 real seconds

# ── Colours ────────────────────────────────────────────────────────────────
BG      = "#0b0b12"
PANEL   = "#111120"
TEXT    = "#d8dcea"
GRID    = "#18182a"
C_SUB   = "#3498db"
C_SUBAL = "#e74c3c"
C_TRUTH = "#2ecc71"
C_DRIFT = "#e74c3c"
C_PLAN  = "#1a6b3a"
C_BUOY  = "#f1c40f"
C_WATER = "#091525"
C_BOUND = "#1a6699"
C_NODE_UNVIS = "#1a3a5c"
C_NODE_VIS   = "#2ecc71"
C_NODE_NEXT  = "#f1c40f"

_spd = [SPEEDUP_INIT]
def _dt():
    return _spd[0] * (FRAME_MS / 1000.0) / STEPS


def _sub_polygon_2d(cx, cy, heading_rad, size_km):
    """Top-down torpedo polygon for P2 (East-North plane)."""
    L = size_km; W = L * 0.28
    pts = np.array([
        [ L*0.50,  0.00],
        [ L*0.20,  W*0.5],
        [-L*0.35,  W*0.5],
        [-L*0.50,  W*0.2],
        [-L*0.50, -W*0.2],
        [-L*0.35, -W*0.5],
        [ L*0.20, -W*0.5],
    ])
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    R = np.array([[c, -s], [s, c]])
    pts = (R @ pts.T).T
    pts[:, 0] += cx; pts[:, 1] += cy
    return pts


# ══════════════════════════════════════════════════════════════════════════
def run():
    print("\n" + "="*52)
    print("  SNC Simulator  —  Stage 1  Startup")
    print("="*52)

    ccz   = build_ccz_grid()
    wp    = build_waypoints(ccz)
    sub   = SubModel(wp)
    drift = DriftModel(sub.state)
    buoy  = BuoyLogic()

    t_sim   = 0.0
    t_start = time.time()

    # Map extent (km) for top-down panels
    PAD = 20.0
    XLO = ccz["x_lim"][0]/1e3 - PAD;  XHI = ccz["x_lim"][1]/1e3 + PAD
    YLO = ccz["y_lim"][0]/1e3 - PAD;  YHI = ccz["y_lim"][1]/1e3 + PAD
    MAP_W = XHI - XLO
    SUB_KM = MAP_W * 0.025   # torpedo size for P2

    print(f"  Nodes   : {wp['n_nodes']}")
    print(f"  Mission : {MISSION_DAYS_MAX} day hard stop")
    print(f"  Demo    : ~{MISSION_DAYS_MAX*86400/(SPEEDUP_INIT*60):.1f} min at {SPEEDUP_INIT}x")
    print()

    # History buffers
    h_tx, h_ty, h_tz = [sub.state.x_true], [sub.state.y_true], [SUB_DEPTH_M]
    h_dx, h_dy = [sub.state.x_true], [sub.state.y_true]

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 10), facecolor=BG)
    fig.suptitle(
        "SNC Simulator  |  Autonomous Navigation for CCZ Survey  ",
        color=TEXT, fontsize=11, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(
        3, 5, fig,
        height_ratios=[2.4, 1.25, 0.065],
        width_ratios=[2.2, 0.04, 0.9, 0.04, 2.4],
        hspace=0.38, wspace=0.0,
        left=0.03, right=0.99, top=0.96, bottom=0.04,
    )

    # P1: true 3D axes — replace flat axis with projection='3d'
    ax_p1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax_p2 = fig.add_subplot(gs[0, 2:5])   # P2: top-down path
    ax_p3 = fig.add_subplot(gs[1, 0:2])   # P3: parameter text (invisible)
    ax_p4 = fig.add_subplot(gs[1, 2:5])   # P4: error graph
    ax_sl = fig.add_subplot(gs[2, 2:5])   # speed slider

    def _style(ax, title=""):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e30"); sp.set_linewidth(0.6)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.9)
        if title:
            ax.set_title(title, color=TEXT, fontsize=10,
                         fontweight="bold", pad=5)

    ax_p3.set_visible(False)

    # ══════════════════════════════════════════════════════════════════════
    # P1 — TRUE 3D VIEW  (East × North × Depth)
    # ══════════════════════════════════════════════════════════════════════
    ax_p1.set_facecolor("#05111f")
    ax_p1.set_title("3D Sub Movement  (East × North × Depth)",
                    color=TEXT, fontsize=9, fontweight="bold", pad=6)

    # Axis labels
    ax_p1.set_xlabel("East [km]",  color=TEXT, fontsize=7, labelpad=2)
    ax_p1.set_ylabel("North [km]", color=TEXT, fontsize=7, labelpad=2)
    ax_p1.set_zlabel("Depth [m]",  color=TEXT, fontsize=7, labelpad=2)
    ax_p1.tick_params(colors=TEXT, labelsize=6)

    # Depth axis: 0 at top (surface), positive DOWN
    ax_p1.set_zlim(MAX_DEPTH_M, 0)   # inverted so surface is at top

    # Set initial 3D view angle — looking from above-south-west
    ax_p1.view_init(elev=25, azim=-60)

    # Set XY extents in km
    ax_p1.set_xlim(XLO, XHI)
    ax_p1.set_ylim(YLO, YHI)

    # Semi-transparent waterline plane at depth=0
    _wx = np.array([XLO, XHI, XHI, XLO])
    _wy = np.array([YLO, YLO, YHI, YHI])
    _wz = np.array([0, 0, 0, 0])
    verts = [list(zip(_wx, _wy, _wz))]
    water_plane = Poly3DCollection(verts, alpha=0.18,
                                   facecolor="#1a6699", edgecolor="#5dade2",
                                   linewidth=0.8, zorder=1)
    ax_p1.add_collection3d(water_plane)
    ax_p1.text2D(0.02, 0.97, "SURFACE  (0 m)", transform=ax_p1.transAxes,
                 color="#5dade2", fontsize=6.5, va="top")

    # Operating depth reference plane (faint)
    _opz = np.full(4, SUB_DEPTH_M)
    verts_op = [list(zip(_wx, _wy, _opz))]
    op_plane = Poly3DCollection(verts_op, alpha=0.07,
                                facecolor="#7788aa", edgecolor="#7788aa",
                                linewidth=0.5, zorder=1)
    ax_p1.add_collection3d(op_plane)

    # Style 3D panes dark
    ax_p1.xaxis.pane.fill = True;  ax_p1.xaxis.pane.set_facecolor("#05111f")
    ax_p1.yaxis.pane.fill = True;  ax_p1.yaxis.pane.set_facecolor("#05111f")
    ax_p1.zaxis.pane.fill = True;  ax_p1.zaxis.pane.set_facecolor("#080e1a")
    ax_p1.xaxis.pane.set_edgecolor("#1e2a3a")
    ax_p1.yaxis.pane.set_edgecolor("#1e2a3a")
    ax_p1.zaxis.pane.set_edgecolor("#1e2a3a")
    ax_p1.grid(True, color="#1e2a3a", lw=0.4, alpha=0.6)

    # 3D trail line (true path)
    ln_p1_trail, = ax_p1.plot([], [], [], "-",
                               color=C_TRUTH, lw=1.2, alpha=0.7, zorder=5)

    # 3D sub marker (scatter point — updates each frame)
    sub3d_scatter = ax_p1.scatter(
        [sub.state.x_true/1e3], [sub.state.y_true/1e3], [SUB_DEPTH_M],
        c=C_SUB, s=60, marker="o", zorder=10, depthshade=True)

    # Vertical line from surface to sub (depth indicator)
    ln_p1_depth, = ax_p1.plot([], [], [], "--",
                               color=C_SUB, lw=0.6, alpha=0.4, zorder=4)

    # 3D buoy marker
    buoy3d_scatter = ax_p1.scatter([], [], [], c=C_BUOY, s=45,
                                    marker="D", zorder=11, depthshade=True)
    # Buoy tether (deploy_x to surface)
    ln_p1_tether, = ax_p1.plot([], [], [], "-", color=C_BUOY,
                                lw=1.0, alpha=0.7, zorder=6)

    # P1 status annotation
    txt_p1 = ax_p1.text2D(0.5, 0.02, "Autonomous survey in progress",
                           transform=ax_p1.transAxes, color=TEXT,
                           fontsize=8, fontweight="bold", ha="center", va="bottom",
                           bbox=dict(fc=PANEL, ec="none", alpha=0.75, pad=3))

    # ── Mission-end surfacing state ────────────────────────────────────────
    _surfacing = {"active": False, "frame": 0, "done": False,
                  "current_depth": SUB_DEPTH_M}

    # ══════════════════════════════════════════════════════════════════════
    # P2 — WAYPOINT MAP (top-down)
    # ══════════════════════════════════════════════════════════════════════
    _style(ax_p2, "Waypoint Survey Map  —  Node Progress & Nav Estimate")
    ax_p2.set_facecolor(C_WATER)

    bx_km = [v/1e3 for v in ccz["boundary_x"]]
    by_km = [v/1e3 for v in ccz["boundary_y"]]
    ax_p2.fill(bx_km, by_km, color="#0c2035", alpha=1.0, zorder=1)
    ax_p2.plot(bx_km, by_km, color=C_BOUND, lw=1.8, zorder=3)

    # Draw all waypoint nodes (unvisited = dark, will update each frame)
    nx_km = [v/1e3 for v in wp["nodes_x"]]
    ny_km = [v/1e3 for v in wp["nodes_y"]]

    # Connections between consecutive nodes (faint planned path)
    ax_p2.plot(nx_km, ny_km, "-", color=C_PLAN, lw=0.8, alpha=0.20,
               zorder=4, label="Planned node path")

    # Node scatter (colour updates each frame)
    node_scatter = ax_p2.scatter(nx_km, ny_km, s=14, c=C_NODE_UNVIS,
                                  zorder=6, edgecolors="none")

    # Actual coverage trail — bright green
    ln_p2_truth, = ax_p2.plot([], [], "-", color=C_TRUTH,
                               lw=2.0, zorder=7, label="Actual path (truth)")

    # Nav estimate — scaled drift (red)
    ln_p2_drift, = ax_p2.plot([], [], "-", color=C_DRIFT,
                               lw=1.3, alpha=0.75, zorder=7,
                               label=f"Nav estimate (×{DRIFT_SCALE} visual)")

    # Buoy fix markers
    ln_p2_fixes, = ax_p2.plot([], [], "o", color=C_BUOY, ms=7,
                               mec="white", mew=0.8, zorder=9,
                               label="Buoy fix applied")

    # Next target node highlight
    next_node_marker, = ax_p2.plot([], [], "*", color=C_NODE_NEXT,
                                    ms=12, mec="white", mew=0.5,
                                    zorder=10, label="Next target node")

    # Sub polygon on P2 (top-down)
    sub2d_pts = _sub_polygon_2d(sub.state.x_true/1e3,
                                 sub.state.y_true/1e3,
                                 sub.state.heading_rad, SUB_KM)
    sub2d_patch = Polygon(sub2d_pts, closed=True,
                           facecolor=C_SUB, edgecolor="white",
                           linewidth=0.8, zorder=11)
    ax_p2.add_patch(sub2d_patch)

    _arr2d = [None]

    # Flash reroute overlay
    _flash_overlay = ax_p2.text(
        0.5, 0.5, "",
        transform=ax_p2.transAxes, color="white",
        fontsize=20, fontweight="bold", ha="center", va="center",
        zorder=20, alpha=0.0,
        bbox=dict(fc="#0a0a15", ec=C_BUOY, alpha=0.0, pad=18, lw=2))

    ax_p2.set_aspect("equal")
    ax_p2.set_xlim(XLO, XHI); ax_p2.set_ylim(YLO, YHI)
    ax_p2.set_xlabel("East [km]",  color=TEXT, fontsize=8)
    ax_p2.set_ylabel("North [km]", color=TEXT, fontsize=8)
    ax_p2.legend(fontsize=6.5, facecolor=PANEL, edgecolor="#1e1e30",
                 labelcolor=TEXT, loc="lower right", framealpha=0.8)
    ax_p2.text(0.01, 0.01,
               f"Nav estimate drift ×{DRIFT_SCALE} for visibility  |  "
               "True error shown in error graph",
               transform=ax_p2.transAxes, color="#7788aa",
               fontsize=5.5, va="bottom", style="italic")

    _plan_init = [False]
    _flash_ctr = [0]

    # ══════════════════════════════════════════════════════════════════════
    # P3 — Sub Parameter Panel (text on figure)
    # ══════════════════════════════════════════════════════════════════════
    p3 = ax_p3.get_position()
    def _T(fx, fy, s, fs=8.5, col=TEXT, wt="normal"):
        return fig.text(
            p3.x0 + fx*p3.width,
            p3.y0 + fy*p3.height,
            s, fontsize=fs, color=col, fontweight=wt,
            va="bottom", transform=fig.transFigure)

    _T(0.02, 0.96, "SUB PARAMETER PANEL", 9, TEXT, "bold")
    _T(0.02, 0.84, "Position (ENU)", 7, "#7788aa")
    t_x   = _T(0.02, 0.75, "X :  --- km")
    t_y   = _T(0.02, 0.66, "Y :  --- km")
    t_dep = _T(0.02, 0.57, "Depth :  600 m")
    _T(0.02, 0.46, "Navigation", 7, "#7788aa")
    t_spd = _T(0.02, 0.37, "Speed :  --- m/s")
    t_hdg = _T(0.02, 0.28, "Heading :  ---°")
    t_wp  = _T(0.02, 0.19, "Node :  -- / --")
    t_dst = _T(0.02, 0.10, "Dist :  --- km")

    _T(0.52, 0.96, "BUOY", 9, TEXT, "bold")
    t_bst  = _T(0.52, 0.84, "Status :  IDLE", 8.5, C_BUOY, "bold")
    t_bdep = _T(0.52, 0.75, "Depth :  --- m")
    t_bfx  = _T(0.52, 0.66, "Fixes :  0")
    _T(0.52, 0.55, "Error", 7, "#7788aa")
    t_err  = _T(0.52, 0.45, "Error :  --- m", 9, C_BUOY, "bold")
    t_elp  = _T(0.52, 0.36, "Since fix :  ---")
    t_cov  = _T(0.52, 0.27, "Nodes :  0 / --")

    _T(0.02, -0.02, "Sim :", 7, "#7788aa")
    t_tsim = _T(0.12, -0.02, "0.0 h  (Day 0.0)", 9, "#5dade2", "bold")
    _T(0.58, -0.02, "Real :", 7, "#7788aa")
    t_trl  = _T(0.67, -0.02, "0 s  |  ---x", 8.5, TEXT)

    # ══════════════════════════════════════════════════════════════════════
    # P4 — Position Error Graph
    # ══════════════════════════════════════════════════════════════════════
    _style(ax_p4, "Position Error Graph  (Config D: INS + DVL WT + ADCP)")

    ax_p4.axhline(ERR_SAFE,   color="#f39c12", lw=1.3, ls="--", alpha=0.85,
                  label=f"{ERR_SAFE:.0f} m  safe nav reference")
    ax_p4.axhline(ERR_DEPLOY, color="#e74c3c", lw=2.2, ls="-",
                  label=f"{ERR_DEPLOY:.0f} m  deploy threshold")
    ax_p4.axhline(ERR_CRIT,   color="#5d6d7e", lw=1.0, ls="--", alpha=0.5,
                  label=f"{ERR_CRIT:.0f} m  mission-critical ref")

    ln_p4_err, = ax_p4.plot([], [], "-", color=C_DRIFT,
                             lw=1.8, label="Position error [m]")
    ax_p4.set_xlim(0, 1); ax_p4.set_ylim(0, ERR_DEPLOY * 3.0)
    ax_p4.set_xlabel("Simulated time [hours]", color=TEXT, fontsize=8)
    ax_p4.set_ylabel("Error [m]", color=TEXT, fontsize=8)
    ax_p4.legend(fontsize=7, facecolor=PANEL, edgecolor="#1e1e30",
                 labelcolor=TEXT, loc="upper left", framealpha=0.85)
    _p4_vlines = []

    txt_p4_evt = ax_p4.text(
        0.99, 0.97, "", transform=ax_p4.transAxes,
        color=C_BUOY, fontsize=7.5, fontweight="bold",
        ha="right", va="top",
        bbox=dict(fc=PANEL, ec=C_BUOY, alpha=0.85, pad=3, lw=0.8))

    # Speed slider
    ax_sl.set_facecolor(PANEL)
    sld = Slider(ax_sl, "Speedup", SPEEDUP_MIN, SPEEDUP_MAX,
                 valinit=SPEEDUP_INIT, valstep=500,
                 color="#1a4a7a", track_color=GRID)
    sld.label.set_color(TEXT); sld.valtext.set_color(TEXT)
    sld.on_changed(lambda v: _spd.__setitem__(0, int(v)))

    # ══════════════════════════════════════════════════════════════════════
    # ANIMATION STATE
    # ══════════════════════════════════════════════════════════════════════
    _prev_fix  = [0]
    _done_flag = [False]

    # ══════════════════════════════════════════════════════════════════════
    # ANIMATION TICK
    # ══════════════════════════════════════════════════════════════════════
    def _tick(_f):
        nonlocal t_sim, h_tx, h_ty, h_tz, h_dx, h_dy

        s = sub.state

        # ── HARD STOP: 42 sim-days OR all nodes visited ───────────────────
        sim_days = t_sim / 86400.0
        if (s.mission_complete or sim_days >= MISSION_DAYS_MAX) and not _done_flag[0]:
            _done_flag[0] = True
            reason = ("All nodes visited" if s.mission_complete
                      else f"42-day mission limit reached")
            print(f"\n[snc_main] MISSION END — {reason}")
            print(f"  Sim time  : {t_sim/86400:.1f} days  ({t_sim/3600:.0f} h)")
            print(f"  Buoy fixes: {buoy.fix_count}")
            print(f"  Nodes done: {len(wp['visited'])}/{wp['n_nodes']}")

        if _done_flag[0]:
            # Surfacing animation
            if not _surfacing["done"]:
                if not _surfacing["active"]:
                    _surfacing["active"] = True
                    _surfacing["frame"]  = 0

                frac = _surfacing["frame"] / SURFACE_FRAMES
                cur_d = SUB_DEPTH_M * max(0.0, 1.0 - frac)
                _surfacing["current_depth"] = cur_d

                # Update 3D sub marker
                sub3d_scatter._offsets3d = (
                    np.array([s.x_true/1e3]),
                    np.array([s.y_true/1e3]),
                    np.array([cur_d])
                )

                txt_p1.set_text("▲  MISSION COMPLETE  —  SURFACING")
                txt_p1.set_color("#2ecc71")

                _surfacing["frame"] += 1
                if _surfacing["frame"] > SURFACE_FRAMES:
                    _surfacing["done"] = True
                    _surfacing["active"] = False
                    # Place at surface
                    sub3d_scatter._offsets3d = (
                        np.array([s.x_true/1e3]),
                        np.array([s.y_true/1e3]),
                        np.array([0.0])
                    )
                    txt_p1.set_text(f"✓  MISSION COMPLETE  —  {buoy.fix_count} Buoy fixes  |  "
                                    f"{len(wp['visited'])}/{wp['n_nodes']} nodes")
                    txt_p1.set_color("#2ecc71")

            t_tsim.set_text(f"{t_sim/3600:.1f} h  (Day {t_sim/86400:.1f})")
            t_trl.set_text(f"{time.time()-t_start:.0f} s  |  {_spd[0]}x")
            plt.draw()
            return

        # ── Physics sub-steps ─────────────────────────────────────────────
        fix_applied = False
        for _ in range(STEPS):
            sub.step(dt_s=_dt())
            drift.step(s, _dt(), t_sim)
            changed = buoy.step(drift, sub, _dt(), t_sim)
            if changed:
                fix_applied = True
            t_sim += _dt()

        # Append history
        h_tx.append(s.x_true); h_ty.append(s.y_true)
        h_tz.append(SUB_DEPTH_M)   # constant depth during survey
        ex = drift.x_est - s.x_true
        ey = drift.y_est - s.y_true
        h_dx.append(s.x_true + ex * DRIFT_SCALE)
        h_dy.append(s.y_true + ey * DRIFT_SCALE)

        # ── Fix flash overlay ──────────────────────────────────────────────
        if buoy.fix_count > _prev_fix[0]:
            _prev_fix[0] = buoy.fix_count
            _flash_ctr[0] = FLASH_FRAMES

        if _flash_ctr[0] > 0:
            nxt = get_next_unvisited(wp)
            nxt_txt = (f"→ Node {nxt[3]}" if nxt else "ALL NODES DONE")
            frac = _flash_ctr[0] / FLASH_FRAMES
            _flash_overlay.set_text(
                f"⟳  POSITION FIXED\nHEADING TO NEXT NODE\n"
                f"Fix #{buoy.fix_count}  —  {nxt_txt}")
            _flash_overlay.set_alpha(frac)
            _flash_overlay.get_bbox_patch().set_alpha(frac * 0.88)
            _flash_ctr[0] -= 1
        else:
            _flash_overlay.set_alpha(0.0)
            _flash_overlay.get_bbox_patch().set_alpha(0.0)

        # ── P1: 3D view update ─────────────────────────────────────────────
        # Trail (last 1500 points)
        tail = 1500
        tx_km = [v/1e3 for v in h_tx[-tail:]]
        ty_km = [v/1e3 for v in h_ty[-tail:]]
        tz    = list(h_tz[-tail:])
        ln_p1_trail.set_data_3d(tx_km, ty_km, tz)

        # Sub position dot
        flash_body = (drift.error_m >= ERR_DEPLOY) and (int(t_sim/2000) % 2 == 0)
        sub_col = C_SUBAL if flash_body else C_SUB
        sub3d_scatter._offsets3d = (
            np.array([s.x_true/1e3]),
            np.array([s.y_true/1e3]),
            np.array([SUB_DEPTH_M])
        )
        sub3d_scatter.set_color(sub_col)

        # Vertical depth indicator line (surface → sub)
        ln_p1_depth.set_data_3d(
            [s.x_true/1e3, s.x_true/1e3],
            [s.y_true/1e3, s.y_true/1e3],
            [0.0, SUB_DEPTH_M]
        )

        # Buoy animation
        buoy_active = buoy.status in ("ascending", "surface", "descending")
        if buoy_active:
            bd = buoy.depth_m
            buoy3d_scatter._offsets3d = (
                np.array([buoy.deploy_x/1e3]),
                np.array([buoy.deploy_y/1e3]),
                np.array([bd])
            )
            buoy3d_scatter.set_visible(True)
            ln_p1_tether.set_data_3d(
                [buoy.deploy_x/1e3, buoy.deploy_x/1e3],
                [buoy.deploy_y/1e3, buoy.deploy_y/1e3],
                [SUB_DEPTH_M, bd]
            )
        else:
            buoy3d_scatter.set_visible(False)
            ln_p1_tether.set_data_3d([], [], [])

        # P1 status text
        if buoy_active:
            if buoy.status == "ascending":
                pct = 100*(1 - buoy.depth_m/SUB_DEPTH_M)
                txt_p1.set_text(f"▲ BUOY ASCENDING  {pct:.0f}%  ({buoy.depth_m:.0f} m)")
                txt_p1.set_color(C_BUOY)
            elif buoy.status == "surface":
                txt_p1.set_text("● BUOY AT SURFACE — GPS fix acquired")
                txt_p1.set_color("#5dade2")
            else:
                pct = 100*(buoy.depth_m/SUB_DEPTH_M)
                txt_p1.set_text(f"▼ BUOY DESCENDING  {pct:.0f}%  ({buoy.depth_m:.0f} m)")
                txt_p1.set_color("#aaaaaa")
        elif drift.error_m >= ERR_DEPLOY:
            txt_p1.set_text(f"⚠  NAV ERROR {drift.error_m:.0f} m — DEPLOYING BUOY")
            txt_p1.set_color(C_SUBAL)
        elif buoy.fix_count > 0:
            txt_p1.set_text(f"✓ {buoy.fix_count} Buoy fix(es) — navigating waypoints")
            txt_p1.set_color(C_TRUTH)
        else:
            txt_p1.set_text("Autonomous waypoint survey in progress")
            txt_p1.set_color(TEXT)

        # Slow 3D rotation: 1° per 5 frames = 0.2 deg/s real
        current_azim = ax_p1.azim
        ax_p1.view_init(elev=25, azim=current_azim + 0.15)

        # ── P2: top-down path ─────────────────────────────────────────────
        ln_p2_truth.set_data([v/1e3 for v in h_tx],
                              [v/1e3 for v in h_ty])
        ln_p2_drift.set_data([v/1e3 for v in h_dx],
                              [v/1e3 for v in h_dy])
        if buoy.fix_x:
            ln_p2_fixes.set_data([v/1e3 for v in buoy.fix_x],
                                  [v/1e3 for v in buoy.fix_y])

        # Next target node star marker
        nxt = get_next_unvisited(wp)
        if nxt:
            next_node_marker.set_data([nxt[0]/1e3], [nxt[1]/1e3])
        else:
            next_node_marker.set_data([], [])

        # Update node colours: visited=green, next=yellow, unvisited=dark
        colours = []
        nxt_idx = nxt[3] if nxt else -1
        for i in range(wp["n_nodes"]):
            if i in wp["visited"]:
                colours.append(C_NODE_VIS)
            elif i == nxt_idx:
                colours.append(C_NODE_NEXT)
            else:
                colours.append(C_NODE_UNVIS)
        node_scatter.set_facecolor(colours)

        # Sub polygon on P2
        pts2d = _sub_polygon_2d(s.x_true/1e3, s.y_true/1e3,
                                 s.heading_rad, SUB_KM)
        sub2d_patch.set_xy(pts2d)
        sub2d_patch.set_facecolor(C_SUBAL if flash_body else C_SUB)

        # Heading arrow
        if _arr2d[0] is not None:
            try: _arr2d[0].remove()
            except: pass
        arr_len = SUB_KM * 1.4
        arrow = FancyArrow(
            s.x_true/1e3, s.y_true/1e3,
            math.cos(s.heading_rad) * arr_len,
            math.sin(s.heading_rad) * arr_len,
            width=arr_len*0.15, color="white",
            length_includes_head=True,
            head_width=arr_len*0.35, head_length=arr_len*0.4,
            zorder=12, alpha=0.85)
        ax_p2.add_patch(arrow)
        _arr2d[0] = arrow

        # ── P3: parameter text ────────────────────────────────────────────
        t_x.set_text(  f"X :  {s.x_true/1e3:.1f} km")
        t_y.set_text(  f"Y :  {s.y_true/1e3:.1f} km")
        t_dep.set_text(f"Depth :  {SUB_DEPTH_M:.0f} m")
        t_spd.set_text(f"Speed :  {s.speed_ms:.1f} m/s  ({s.speed_ms*1.944:.1f} kts)")
        t_hdg.set_text(f"Heading :  {math.degrees(s.heading_rad)%360:.0f}°")

        visited_count = len(wp["visited"])
        t_wp.set_text(f"Node :  {visited_count} / {wp['n_nodes']} visited")
        t_dst.set_text(f"Dist :  {s.dist_total/1e3:.0f} km")

        ec = ("#2ecc71" if drift.error_m < ERR_SAFE else
              "#f39c12" if drift.error_m < ERR_DEPLOY else "#e74c3c")
        t_err.set_text(f"Error :  {drift.error_m:.0f} m"); t_err.set_color(ec)
        t_elp.set_text(f"Since fix :  {drift._elapsed_s/3600:.1f} h")
        t_cov.set_text(f"Nodes :  {visited_count} / {wp['n_nodes']}")
        t_cov.set_color("#2ecc71" if visited_count > 0 else TEXT)

        sm = {
            "idle":        ("IDLE",           "#2ecc71"),
            "ascending":   ("ASCENDING ▲",    C_BUOY),
            "surface":     ("AT SURFACE ●",   "#5dade2"),
            "fix_applied": ("FIX APPLIED ✓",  "#2ecc71"),
            "descending":  ("DESCENDING ▼",   "#aaaaaa"),
        }
        bl, bc = sm.get(buoy.status, (buoy.status.upper(), TEXT))
        t_bst.set_text(f"Status :  {bl}"); t_bst.set_color(bc)
        t_bdep.set_text(f"Depth :  {buoy.depth_m:.0f} m")
        t_bfx.set_text( f"Fixes :  {buoy.fix_count}")
        t_tsim.set_text(f"{t_sim/3600:.1f} h  (Day {t_sim/86400:.1f})")
        t_trl.set_text( f"{time.time()-t_start:.0f} s  |  {_spd[0]}x")

        # ── P4: error graph ────────────────────────────────────────────────
        if drift.time_log:
            th = [v/3600 for v in drift.time_log]
            ln_p4_err.set_data(th, drift.error_log)
            ax_p4.set_xlim(0, max(1.0, th[-1]*1.05))
            ytop = max(ERR_DEPLOY*3.0, max(drift.error_log)*1.15)
            ax_p4.set_ylim(0, ytop)

            while len(_p4_vlines) < buoy.fix_count:
                n = len(_p4_vlines) + 1
                vl = ax_p4.axvline(th[-1], color=C_BUOY,
                                   lw=0.9, ls="--", alpha=0.65)
                ax_p4.text(th[-1] + 0.3, ytop * 0.80,
                           f"#{n}", color=C_BUOY, fontsize=6)
                _p4_vlines.append(vl)

        txt_p4_evt.set_text(buoy.event[:65])
        plt.draw()

    ani = animation.FuncAnimation(   # noqa: F841
        fig, _tick, interval=FRAME_MS, cache_frame_data=False)
    plt.show()
    print(f"\n[snc_main] Session ended — "
          f"{t_sim/3600:.1f} sim-h  ({t_sim/86400:.1f} days)  |  "
          f"{buoy.fix_count} Buoy fixes  |  "
          f"{len(wp['visited'])}/{wp['n_nodes']} nodes visited")


if __name__ == "__main__":
    run()
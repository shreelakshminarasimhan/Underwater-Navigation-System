"""
ccz_grid.py  —  Clarion-Clipperton Zone coordinate grid
================================================================
PURPOSE
-------
Converts real CCZ lat/lon bounds into a flat metre grid (ENU).
Every other module works in metres only — this is the only file
that touches geographic coordinates.

OUTPUT
------
Returns a plain Python dict (ccz) with fields:
    x_lim        [m]  (x_min, x_max)  East extent
    y_lim        [m]  (y_min, y_max)  North extent
    width_m      [m]  East-West span
    height_m     [m]  North-South span
    boundary_x   [m]  polygon x-coords (5-point closed loop)
    boundary_y   [m]  polygon y-coords
    origin_lon   [deg]
    origin_lat   [deg]
    waypoints    dict  populated by waypoint_path.build_waypoints()

CALLED BY:   snc_main.py  (once at startup)
USED BY:     waypoint_path.py, sub_model.py, buoy_logic.py
"""

import math

# ── ADJUSTABLE PARAMETERS ─────────────────────────────────────────────────
LON_LIM = (-130.0, -116.0)   # deg West  — eastern CCZ corridor
LAT_LIM = (  8.0,   13.0)   # deg North
# ──────────────────────────────────────────────────────────────────────────

R_EARTH = 6_371_000.0  # mean Earth radius [m]


def _geo2enu(lat, lon, lat0, lon0):
    """Flat-Earth tangent-plane projection.  Accurate to ~0.1% within 1000 km."""
    yn = R_EARTH * (lat  - lat0) * math.pi / 180.0
    xe = R_EARTH * math.cos(lat0 * math.pi / 180.0) * (lon - lon0) * math.pi / 180.0
    return xe, yn


def build_ccz_grid():
    """Build and return the CCZ coordinate grid dict."""
    lon0 = sum(LON_LIM) / 2.0
    lat0 = sum(LAT_LIM) / 2.0

    # Four corners → ENU metres
    x_sw, y_sw = _geo2enu(LAT_LIM[0], LON_LIM[0], lat0, lon0)
    x_ne, y_ne = _geo2enu(LAT_LIM[1], LON_LIM[1], lat0, lon0)
    x_nw, y_nw = _geo2enu(LAT_LIM[1], LON_LIM[0], lat0, lon0)
    x_se, y_se = _geo2enu(LAT_LIM[0], LON_LIM[1], lat0, lon0)

    ccz = {
        "boundary_x":  [x_sw, x_se, x_ne, x_nw, x_sw],
        "boundary_y":  [y_sw, y_se, y_ne, y_nw, y_sw],
        "x_lim":       (x_sw, x_ne),
        "y_lim":       (y_sw, y_ne),
        "width_m":     x_ne - x_sw,
        "height_m":    y_ne - y_sw,
        "origin_lon":  lon0,
        "origin_lat":  lat0,
        "waypoints":   None,   # filled by waypoint_path.build_waypoints()
    }

    print(f"[ccz_grid] Patch  : {ccz['width_m']/1e3:.0f} km (E-W)  x  {ccz['height_m']/1e3:.0f} km (N-S)")
    print(f"[ccz_grid] Origin : {abs(lat0):.2f}°N  {abs(lon0):.2f}°W")
    return ccz
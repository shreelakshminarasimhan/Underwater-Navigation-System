"""
drift_model.py  —  INS + DVL + ADCP navigation error model
================================================================
Uses Config (D) from INS_DVL_ADCP.m (matlab simulation models):
    INS + DVL water-track + ADCP current correction (direct subtraction)

ADCP PARAMETERS (from MATLAB)
    true_current_e = 0.08 m/s  (East component)
    true_current_n = 0.04 m/s  (North component)
    current_mag    = sqrt(0.08² + 0.04²) ≈ 0.0894 m/s
    ADCP sigma     = 0.015 m/s
    DVL vel_sigma_wt = 0.035 m/s  (water-track, from MATLAB)

    sigma_eff = sqrt(DVL_vel_sigma_wt² + ADCP_sigma²)
              = sqrt(0.035² + 0.015²)
              ≈ 0.0381 m/s

    CEP_rate_eff = sigma_eff × sqrt(3600) ≈ 2.28 m/hr

    NOTE: This is slightly higher than the pure water-track 2.10 m/hr
    because the ADCP measurement noise adds small residual uncertainty.
    However it is still better than DVL-biased (with current not corrected),
    which would be sigma_biased = 0.035 + 0.0894 ≈ 0.124 m/s → 7.45 m/hr.
    The ADCP correction removes the dominant current bias.

LATERAL DRIFT MODEL (visual design)
    The estimated position drifts LATERALLY relative to the sub's heading.
    This makes the drift trail visibly separate sideways from the truth
    trail in P2 — far more readable than along-track drift which would
    just look like the trail is slightly longer. This is to simplify the 2D 
    simulation model visually.


After a USBL fix:
    x_est, y_est snap exactly to the sub's current TRUE position.
    Elapsed time resets to 0. A new lateral drift side is drawn.

FIELDS
------
    x_est, y_est  [m]   estimated position
    error_m       [m]   current 2-D position error
    error_log     [m]   full history (for P4 error graph)
    time_log      [s]   simulated time at each sample
"""

import math
import random

# ── PARAMETERS (from INS_DVL_ADCP.m Config D) ─────────────────────────────
DVL_VEL_SIGMA_WT  = 0.035     # m/s  DVL water-track 1-sigma (MATLAB: DVL.vel_sigma_wt)
ADCP_SIGMA        = 0.015     # m/s  ADCP measurement noise (MATLAB: ADCP.sigma)
ADCP_CURRENT_E    = 0.08      # m/s  true ocean current east  (MATLAB: ADCP.true_current_e)
ADCP_CURRENT_N    = 0.04      # m/s  true ocean current north (MATLAB: ADCP.true_current_n)

# Effective sigma after ADCP current correction (MATLAB: ADCP.sigma_eff)
_SIGMA_EFF   = math.sqrt(DVL_VEL_SIGMA_WT**2 + ADCP_SIGMA**2)   # ≈ 0.0381 m/s
CEP_RATE_EFF = _SIGMA_EFF * math.sqrt(3600)                       # ≈ 2.28 m/hr

CEP_RATE_SPREAD = 0.20   # ±20% run-to-run variability (MATLAB: 0.20)

# Lateral drift fraction: what fraction of total error is cross-track
# 1.0 = purely lateral (most visible in P2), 0.5 = 50/50
LATERAL_FRACTION = 0.85
# ──────────────────────────────────────────────────────────────────────────


class DriftModel:

    def __init__(self, sub_state):
        self.x_est      = sub_state.x_true
        self.y_est      = sub_state.y_true
        self.error_m    = 0.0
        self.error_log  = []
        self.time_log   = []

        self._elapsed_s  = 0.0
        self._rwalk_x    = 0.0
        self._rwalk_y    = 0.0
        self._drift_side = random.choice([-1.0, 1.0])
        self._cep_rate   = self._new_cep_rate()

        current_mag = math.hypot(ADCP_CURRENT_E, ADCP_CURRENT_N)
        print(f"[drift_model] Config D: INS + DVL water-track + ADCP correction")
        print(f"[drift_model] DVL sigma_wt  = {DVL_VEL_SIGMA_WT*1000:.1f} mm/s")
        print(f"[drift_model] ADCP sigma    = {ADCP_SIGMA*1000:.1f} mm/s")
        print(f"[drift_model] Ocean current = {current_mag*100:.1f} cm/s "
              f"(E={ADCP_CURRENT_E:.3f}, N={ADCP_CURRENT_N:.3f} m/s)")
        print(f"[drift_model] sigma_eff     = {_SIGMA_EFF*1000:.2f} mm/s  →  "
              f"CEP ≈ {self._cep_rate:.2f} m/hr")

    def _new_cep_rate(self):
        return max(
            CEP_RATE_EFF * 0.5,
            CEP_RATE_EFF * (1.0 + CEP_RATE_SPREAD * random.gauss(0, 1))
        )

    def step(self, sub, dt_s, t_sim_s):
        """
        Advance navigation error by dt_s simulated seconds.

        Estimated position offset is split into:
          - Lateral (cross-track): perpendicular to sub heading → visually obvious
          - Along-track: parallel to sub heading → small, adds realism
        """
        self._elapsed_s += dt_s
        elapsed_hr = self._elapsed_s / 3600.0

        # Core: linear CEP-rate growth (Config D rate)
        base_error = self._cep_rate * elapsed_hr

        # Small bounded random walk for realism
        rw_sigma = 0.03 * max(base_error, 1.0) * math.sqrt(dt_s / 3600.0)
        self._rwalk_x += rw_sigma * random.gauss(0, 1)
        self._rwalk_y += rw_sigma * random.gauss(0, 1)
        rw_max = max(0.15 * base_error, 2.0)
        self._rwalk_x = max(-rw_max, min(rw_max, self._rwalk_x))
        self._rwalk_y = max(-rw_max, min(rw_max, self._rwalk_y))

        # Sub heading unit vectors
        hdg = sub.heading_rad
        ax  =  math.cos(hdg)                        # along-track East
        ay  =  math.sin(hdg)                        # along-track North
        lx  = -math.sin(hdg) * self._drift_side     # lateral East
        ly  =  math.cos(hdg) * self._drift_side     # lateral North

        lateral_err = base_error * LATERAL_FRACTION
        along_err   = base_error * (1.0 - LATERAL_FRACTION)

        ex = lateral_err * lx + along_err * ax + self._rwalk_x
        ey = lateral_err * ly + along_err * ay + self._rwalk_y

        self.x_est   = sub.x_true + ex
        self.y_est   = sub.y_true + ey
        self.error_m = math.hypot(ex, ey)

        self.error_log.append(self.error_m)
        self.time_log.append(t_sim_s)

    def reset(self, sub):
        """
        Apply USBL buoy position fix.

        Snaps x_est, y_est to the sub's CURRENT TRUE position —
        not to the deploy point, not to the origin.
        The sub kept moving during the buoy's 1200 s ascent, so we snap
        to wherever it actually is when the acoustic fix arrives.

        New drift side drawn so the next cycle diverges in a fresh direction.
        """
        print(f"[drift_model] USBL fix — snapping to true pos "
              f"({sub.x_true/1e3:.1f}, {sub.y_true/1e3:.1f}) km  "
              f"(error was {self.error_m:.0f} m)")

        self.x_est      = sub.x_true
        self.y_est      = sub.y_true
        self.error_m    = 0.0
        self._elapsed_s = 0.0
        self._rwalk_x   = 0.0
        self._rwalk_y   = 0.0
        self._drift_side = random.choice([-1.0, 1.0])
        self._cep_rate   = self._new_cep_rate()
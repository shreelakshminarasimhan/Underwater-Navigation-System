"""
buoy_logic.py  —  USBL buoy deployment state machine
================================================================
STATE MACHINE (five states)
---------------------------
IDLE        Watches drift.error_m. Deploy when >= ERROR_THRESHOLD_M.
ASCENDING   Buoy rises at ASCENT_SPEED_MS m/s. Error still grows.
SURFACE     GPS acquired. Waits ACOUSTIC_DELAY_S before fix lands.
FIX_APPLIED drift.reset() + sub.redirect_to_next_node() from corrected position.
DESCENDING  Buoy sinks back. Cooldown timer active.

HOW IT WORKS
--------------
After a position fix, buoy calls sub.redirect_to_next_node() instead of
build_reroute().  The sub immediately plots a straight-line course
to the next unvisited waypoint node.  No lawnmower replay.

THRESHOLD JUSTIFICATION
-----------------------
200 m = ~half sonar swath width for CCZ multibeam surveys.
        Operational trigger — error large enough to degrade survey quality.
100 m = safe nav reference line (shown on error graph, not a trigger).
1000 m = mission-critical abort threshold (shown on graph only).

Config D (INS + DVL WT + ADCP): CEP_rate ≈ 2.28 m/hr
→ error hits 200 m after ~200/2.28 ≈ 87.7 sim-hours ≈ 3.7 sim-days

ADJUSTABLE PARAMETERS
---------------------
ERROR_THRESHOLD_M  200 m operational deploy trigger
ASCENT_SPEED_MS    0.5 m/s  →  1200 s ascent from 600 m
SUB_DEPTH_M        600 m  (must match sub_model.py DEPTH_M)
ACOUSTIC_DELAY_S   2 s post-surface acoustic round-trip
COOLDOWN_S         3600 s min between deployments (1 sim-hour)
"""

# ── ADJUSTABLE PARAMETERS ─────────────────────────────────────────────────
ERROR_THRESHOLD_M = 200.0   # m   operational deploy threshold
ASCENT_SPEED_MS   = 0.5     # m/s buoy rise speed
SUB_DEPTH_M       = 600.0   # m   must match sub_model.py DEPTH_M
ACOUSTIC_DELAY_S  = 2.0     # s   acoustic round-trip + processing
COOLDOWN_S        = 3600.0  # s   min between deployments
# ──────────────────────────────────────────────────────────────────────────


class BuoyLogic:

    def __init__(self):
        self.status       = "idle"
        self.depth_m      = SUB_DEPTH_M
        self.deploy_x     = 0.0
        self.deploy_y     = 0.0
        self.fix_x        = []
        self.fix_y        = []
        self.fix_count    = 0
        self.cooldown_rem = 0.0
        self.event        = "Autonomous survey in progress"

        self._asc_timer     = 0.0
        self._ascent_time_s = SUB_DEPTH_M / ASCENT_SPEED_MS   # 1200 s
        self._total_delay_s = self._ascent_time_s + ACOUSTIC_DELAY_S

        print(f"[buoy_logic] Error threshold : {ERROR_THRESHOLD_M:.0f} m")
        print(f"[buoy_logic] Ascent time     : {self._ascent_time_s:.0f} s "
              f"({self._ascent_time_s/60:.1f} min)")
        print(f"[buoy_logic] Acoustic delay  : {ACOUSTIC_DELAY_S:.1f} s")
        print(f"[buoy_logic] Total fix delay : {self._total_delay_s:.0f} s")
        print(f"[buoy_logic] Cooldown        : {COOLDOWN_S:.0f} s")

    def step(self, drift, sub_model, dt_s, t_sim_s):
        """
        Advance state machine one dt_s step.

        Returns
        -------
        fix_applied : bool  True when a USBL fix was just delivered.
        """
        fix_applied = False

        if self.cooldown_rem > 0:
            self.cooldown_rem = max(0.0, self.cooldown_rem - dt_s)

        s = self.status

        # ── IDLE ──────────────────────────────────────────────────────────
        if s == "idle":
            if drift.error_m >= ERROR_THRESHOLD_M and self.cooldown_rem <= 0:
                self.status     = "ascending"
                self._asc_timer = 0.0
                self.depth_m    = SUB_DEPTH_M
                self.deploy_x   = sub_model.state.x_true
                self.deploy_y   = sub_model.state.y_true
                self.event = f"BUOY DEPLOYED  (error {drift.error_m:.0f} m)"
                print(f"[buoy_logic t={t_sim_s/3600:.1f}h] {self.event}")

        # ── ASCENDING ─────────────────────────────────────────────────────
        elif s == "ascending":
            self._asc_timer += dt_s
            self.depth_m = max(0.0, SUB_DEPTH_M - ASCENT_SPEED_MS * self._asc_timer)
            if self._asc_timer >= self._ascent_time_s:
                self.status  = "surface"
                self.depth_m = 0.0
                self.event   = "BUOY SURFACED — GPS acquired"
                print(f"[buoy_logic t={t_sim_s/3600:.1f}h] {self.event}")
            else:
                pct = 100 * self._asc_timer / self._ascent_time_s
                self.event = f"Buoy ascending {pct:.0f}%  ({self.depth_m:.0f} m depth)"

        # ── SURFACE ───────────────────────────────────────────────────────
        elif s == "surface":
            self._asc_timer += dt_s
            time_since_surface = self._asc_timer - self._ascent_time_s
            if time_since_surface >= ACOUSTIC_DELAY_S:
                self.status    = "fix_applied"
                self.fix_count += 1

                # Snap nav estimate to current TRUE position
                drift.reset(sub_model.state)

                # Record fix location for marker display
                self.fix_x.append(sub_model.state.x_true)
                self.fix_y.append(sub_model.state.y_true)

                # KEY CHANGE: redirect sub to next unvisited waypoint node
                # from its corrected true position
                sub_model.redirect_to_next_node()
                fix_applied = True

                self.event = (f"FIX #{self.fix_count} at "
                              f"({sub_model.state.x_true/1e3:.0f}, "
                              f"{sub_model.state.y_true/1e3:.0f}) km — HEADING TO NEXT NODE")
                print(f"[buoy_logic t={t_sim_s/3600:.1f}h] {self.event}")
            else:
                rem = ACOUSTIC_DELAY_S - time_since_surface
                self.event = f"Buoy at surface — TX in {rem:.1f} s"

        # ── FIX_APPLIED ───────────────────────────────────────────────────
        elif s == "fix_applied":
            self.status       = "descending"
            self._asc_timer   = 0.0
            self.cooldown_rem = COOLDOWN_S
            self.event = "Fix applied — buoy descending"

        # ── DESCENDING ────────────────────────────────────────────────────
        elif s == "descending":
            self._asc_timer += dt_s
            self.depth_m = min(SUB_DEPTH_M, ASCENT_SPEED_MS * self._asc_timer)
            if self.depth_m >= SUB_DEPTH_M:
                self.status  = "idle"
                self.depth_m = SUB_DEPTH_M
                self.event   = (f"Buoy recovered — IDLE  "
                                f"(cooldown {self.cooldown_rem/60:.0f} min)")
            else:
                pct = 100 * self._asc_timer / self._ascent_time_s
                self.event = f"Buoy descending {pct:.0f}%  ({self.depth_m:.0f} m)"

        return fix_applied
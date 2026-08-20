"""
snc_manual.py  —  SNC Simulator  |  Manual/Game Survey Mode
========================================================
NARRATIVE
---------
The Clarion-Clipperton Zone contains an estimated 21 billion tonnes of
polymetallic nodules worth trillions of dollars.  Mining contractors need
centimetre-accurate survey maps to know where to deploy equipment.

The Oriel is conducting that survey at 600 m depth — below GPS range.
Navigation relies on INS + DVL, which drifts over time.  Every survey
node you collect is tagged with your ESTIMATED position.  If that
estimate is wrong, the nodule map is wrong.  The contractor drills in
the wrong place, breaches a protected environmental zone, or wastes
millions on an empty patch of seabed.

THE TENSION
-----------
Deploying a USBL buoy resets your navigation — but each deployment
burns ship time and mission budget.  You have 3 buoys for the whole
survey.  Push on and collect more nodes with degrading accuracy, or
surface for a fix and lose time?

SCORING
-------
  Each node collected earns a DATA QUALITY score based on nav error:
    error < 5 km   →  HIGH quality   (green)   × 3.0
    error < 15 km  →  MEDIUM quality (orange)  × 1.5
    error >= 15 km →  LOW quality    (red)      × 0.5  ← nearly worthless
  Node capture uses ESTIMATED position only — if you're lost, you miss.
  Final score = sum of quality-weighted node values
              + time bonus + buoy conservation bonus

CONTROLS
--------
  Keyboard:
    A / ←    Turn port      D / →   Turn starboard
    SPACE    Deploy buoy    R       New game    Q/Esc  Quit
    I        Toggle instructions panel

  PS4 Controller (plug in before launch):
    Left thumbstick   Steer
    △ Triangle        Deploy USBL buoy
    L1 / L2           New survey run

  Phone (same WiFi — URL printed at startup):
    D-pad buttons   OR   tilt phone (gyro mode toggle on phone page)
    Tilt left/right = turn  (thrust always-on — no fwd/back needed)

NOTE: The Oriel always moves at cruise speed. You can only control heading.
This reflects real sub operation — stationkeeping is energy-costly and
avoided during survey missions.
"""

import math, time, random, socket, threading
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrow, Circle, FancyBboxPatch
from matplotlib.colors import to_rgba

try:
    from flask import Flask, request, jsonify
    FLASK_OK = True
except ImportError:
    FLASK_OK = False

try:
    import pygame
    pygame.init()
    pygame.joystick.init()
    PYGAME_OK = pygame.joystick.get_count() > 0
    if PYGAME_OK:
        _joystick = pygame.joystick.Joystick(0)
        _joystick.init()
        print(f"[controller] PS4/gamepad detected: {_joystick.get_name()}")
    else:
        _joystick = None
        print("[controller] No gamepad detected — keyboard only")
except ImportError:
    PYGAME_OK = False
    _joystick = None
    print("[controller] pygame not installed — no gamepad support")

# ══════════════════════════════════════════════════════════════════════════════
#  GAME / PHYSICS PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

GAME_DURATION_S   = 90

# Nodes
N_NODES           = 14
NODE_RADIUS_KM    = 8     # capture radius — uses ESTIMATED position

# Buoys
N_BUOYS           = 5
BUOY_ASCENT_S     = 2.0       # real-time animation

# Sub physics (units: km and real seconds)
# Sub always moves at constant cruise speed — player can only steer direction.
# This reflects real sub operation where stationkeeping is undesirable.
CRUISE_SPEED      = 50.0      # km/s constant forward speed (always on)
TURN_RATE_DEG     = 85.0      # deg/s when key held

# ── INS DRIFT MODEL ──────────────────────────────────────────────────────────
# Real Config-D INS+DVL+ADCP: CEP rate ≈ 2.28 m/hr
# Scaled up for game time (90 s game = ~3.7 sim-days compressed)
# At game speed the sub travels ~1400 km in 90 s → compress by factor ~500
# Effective drift rate chosen so error hits NODE_RADIUS_KM in ~25-30 s
# giving meaningful buoy tension with 3 buoys over 90 s.

INS_SIGMA_BASE    = 0.55      # km/s  base velocity uncertainty
INS_CORR_TIME     = 20.0      # s     shorter correlation → error builds quicker
INS_ACCEL_FACTOR  = 2.2       # steeper quadratic growth
# After ~8 s without fix:  error ≈ 0.5 km  (approaching MEDIUM threshold)
# After ~18 s: error ≈ 1.5 km  (LOW quality — nodes start glitching)
# After ~35 s: error ≈ 5+ km   (near impossible to capture nodes)

# Data quality thresholds
Q_HIGH_KM         = 2.5      # below this → HIGH quality
Q_MED_KM          = 4       # below this → MEDIUM quality
Q_WEIGHTS         = {         # score multiplier per quality level
    "HIGH":   3.0,
    "MEDIUM": 1.5,
    "LOW":    0.5,
}
Q_COLOURS         = {"HIGH": "#2ecc71", "MEDIUM": "#f39c12", "LOW": "#e74c3c"}

# Arena
CCZ_X = (-700.0, 700.0)
CCZ_Y = (-280.0, 280.0)

# Camera
VIEW_HW = 180.0    # half-width of map viewport (km)
VIEW_HH = 120.0

FLASK_PORT = 5000
FRAME_MS   = 50    # 20 fps

# ── COLOURS ──────────────────────────────────────────────────────────────────
BG     = "#0b0b12"
PANEL  = "#111120"
TEXT   = "#d8dcea"
GRID   = "#161628"
C_BDR  = "#1a6699"
C_SUB  = "#3498db"
C_WARN = "#e74c3c"
C_DRFT = "#e74c3c"
C_TRUE = "#2ecc71"
C_BUOY = "#f1c40f"
C_SCORE= "#f39c12"
C_ND_U = "#1a3a5c"


# ══════════════════════════════════════════════════════════════════════════════
#  INS DRIFT MODEL
# ══════════════════════════════════════════════════════════════════════════════
class INSDrift:
    """
    Realistic INS error model:
      - Velocity error is a correlated random walk (Gauss-Markov process)
      - Position error = time-integral of velocity error  → quadratic growth
      - Lateral bias: error drifts mostly sideways (cross-track)
        to match real DVL water-track behaviour and be visually clear
    """
    def __init__(self):
        self.reset(0.0, 0.0)

    def reset(self, x_true, y_true):
        self.x_est      = x_true
        self.y_est      = y_true
        self._vx_err    = 0.0          # velocity error components (km/s)
        self._vy_err    = 0.0
        self._elapsed   = 0.0          # seconds since last fix
        self._side      = random.choice([-1.0, 1.0])   # lateral bias side
        self.error_km   = 0.0
        # Slight random CEP-rate variation between runs (±20%)
        self._scale     = random.uniform(0.80, 1.20)

    def step(self, x_true, y_true, heading, speed, dt):
        """
        Advance drift model one frame.
        Only accumulates error while sub is moving (speed > 0.5 km/s).
        Returns (x_est, y_est, error_km).
        """
        if speed < 0.5:
            # Stationary: position error stays frozen, slight random micro-jitter
            self.x_est += random.gauss(0, 0.002)
            self.y_est += random.gauss(0, 0.002)
            self.error_km = math.hypot(self.x_est - x_true,
                                       self.y_est - y_true)
            return self.x_est, self.y_est, self.error_km

        self._elapsed += dt

        # ── Gauss-Markov velocity error update ───────────────────────────
        # Each component is an Ornstein-Uhlenbeck process
        alpha   = dt / INS_CORR_TIME          # mean-reversion strength
        sigma_w = INS_SIGMA_BASE * math.sqrt(2.0 * dt / INS_CORR_TIME) * self._scale

        self._vx_err += (-alpha * self._vx_err + sigma_w * random.gauss(0,1))
        self._vy_err += (-alpha * self._vy_err + sigma_w * random.gauss(0,1))

        # ── Add systematic lateral bias (cross-track) ─────────────────────
        # In real DVL, ocean currents not fully cancelled cause a
        # consistent lateral drift.  Grows with time × speed.
        t     = self._elapsed
        bias  = (INS_SIGMA_BASE * 0.6 * self._scale
                 * (t / 10.0) ** INS_ACCEL_FACTOR)   # quadratic growth
        lx    = -math.sin(heading) * self._side
        ly    =  math.cos(heading) * self._side

        vx_total = self._vx_err + bias * lx
        vy_total = self._vy_err + bias * ly

        # ── Integrate to position ─────────────────────────────────────────
        self.x_est += (speed * math.cos(heading) + vx_total) * dt
        self.y_est += (speed * math.sin(heading) + vy_total) * dt

        self.error_km = math.hypot(self.x_est - x_true,
                                   self.y_est - y_true)
        return self.x_est, self.y_est, self.error_km

    def quality(self):
        if self.error_km < Q_HIGH_KM:   return "HIGH"
        if self.error_km < Q_MED_KM:    return "MEDIUM"
        return "LOW"

    def glitch_offset(self):
        """
        Return (dx, dy) pixel jitter for node rendering based on nav error.
        HIGH: no jitter.  MEDIUM: gentle shimmer.  LOW: violent scramble.
        Simulates the effect of a corrupted position fix tagging survey data
        at the wrong location — the 'map' the contractor receives is wrong.
        """
        q = self.quality()
        if q == "HIGH":
            return 0.0, 0.0
        elif q == "MEDIUM":
            # Gentle oscillating shimmer — data is degrading
            amplitude = (self.error_km - Q_HIGH_KM) / (Q_MED_KM - Q_HIGH_KM)
            jitter = amplitude * 0.4   # up to 0.4 km shimmer
            return random.gauss(0, jitter), random.gauss(0, jitter)
        else:
            # LOW: violent scramble — data is effectively useless
            excess = min(self.error_km - Q_MED_KM, 3.0)   # cap at 3 km over
            jitter = 0.6 + excess * 0.5                     # 0.6-2.1 km scramble
            return random.gauss(0, jitter), random.gauss(0, jitter)


# ══════════════════════════════════════════════════════════════════════════════
#  GAME STATE
# ══════════════════════════════════════════════════════════════════════════════
class GameState:
    def __init__(self):
        self.leaderboard     = []
        self._reset_requested = False
        self.reset()

    def reset(self):
        self.x_true   = 0.0
        self.y_true   = 0.0
        self.heading  = math.pi / 2   # North
        self.speed    = CRUISE_SPEED   # The Oriel always moving — no stationkeeping

        self.drift    = INSDrift()
        self.x_est    = 0.0
        self.y_est    = 0.0
        self.error_km = 0.0

        self.trail_true = deque(maxlen=1500)
        self.trail_est  = deque(maxlen=1500)
        self.trail_true.append((0.0, 0.0))
        self.trail_est.append((0.0, 0.0))

        self.nodes    = []
        self.visited  = {}   # idx → {"quality": str, "x_est": float, "y_est": float}
        self._gen_nodes()

        self.buoys_left  = N_BUOYS
        self._buoy_phase = "idle"
        self._buoy_timer = 0.0
        self.buoy_x = self.buoy_y = 0.0
        self.fix_count   = 0
        self._reveal_rem = 0
        self.fix_markers = []

        # Survey quality log for the quality timeline graph
        self.quality_log  = []   # (t_real, error_km)
        self.q_warn_flash = 0    # frames of red flash when quality drops to LOW

        self.t_start   = time.time()
        self.time_left = float(GAME_DURATION_S)
        self.game_over = False
        self.final_score = 0
        self.game_started = False   # True once player clicks START

        self.inp = {
            "left":   False, "right": False,
            "deploy": False,
        }

    def _gen_nodes(self):
        margin = 60.0
        self.nodes = []
        tries = 0
        while len(self.nodes) < N_NODES and tries < 9999:
            tries += 1
            nx = random.uniform(CCZ_X[0]+margin, CCZ_X[1]-margin)
            ny = random.uniform(CCZ_Y[0]+margin, CCZ_Y[1]-margin)
            close = any(math.hypot(nx-ex, ny-ey) < NODE_RADIUS_KM*3.5
                        for ex,ey in self.nodes)
            if not close and math.hypot(nx, ny) > 100:
                self.nodes.append((nx, ny))

    def score(self):
        if not self.visited: return 0
        raw = sum(Q_WEIGHTS[v["quality"]] * 1000
                  for v in self.visited.values())
        t_elapsed  = GAME_DURATION_S - self.time_left
        spd_bonus  = max(1.0, 2.5 - t_elapsed / 50.0)
        buoy_bonus = 1.0 + (self.fix_count == 0) * 0.5 \
                         + max(0, N_BUOYS - self.fix_count) * 0.10
        return int(raw * spd_bonus * buoy_bonus)

    def quality_breakdown(self):
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for v in self.visited.values():
            counts[v["quality"]] += 1
        return counts

    def step(self, dt):
        if self.game_over: return
        if not self.game_started:
            # Reset start time each frame so the clock doesn't tick
            self.t_start = time.time()
            return
        self.time_left = max(0.0, GAME_DURATION_S
                             - (time.time() - self.t_start))
        if self.time_left <= 0:
            self._end(); return

        i = self.inp

        # ── Turning — only control available ────────────────────────────
        tr = math.radians(TURN_RATE_DEG) * dt
        if i["left"]:  self.heading += tr
        if i["right"]: self.heading -= tr
        self.heading %= (2 * math.pi)

        # ── Speed — always at cruise speed (The Oriel never station-keeps) ────
        self.speed = CRUISE_SPEED

        # ── True position ─────────────────────────────────────────────────
        self.x_true += self.speed * math.cos(self.heading) * dt
        self.y_true += self.speed * math.sin(self.heading) * dt
        self.x_true = max(CCZ_X[0], min(CCZ_X[1], self.x_true))
        self.y_true = max(CCZ_Y[0], min(CCZ_Y[1], self.y_true))

        # ── INS drift ────────────────────────────────────────────────────
        self.x_est, self.y_est, self.error_km = self.drift.step(
            self.x_true, self.y_true, self.heading, self.speed, dt)

        # Quality warning flash
        if self.drift.quality() == "LOW":
            self.q_warn_flash = max(self.q_warn_flash, 6)
        if self.q_warn_flash > 0:
            self.q_warn_flash -= 1

        self.trail_true.append((self.x_true, self.y_true))
        self.trail_est.append((self.x_est,  self.y_est))
        self.quality_log.append((time.time() - self.t_start, self.error_km))

        if self._reveal_rem > 0:
            self._reveal_rem -= 1

        # ── Node capture — uses ESTIMATED position ────────────────────────
        q_now = self.drift.quality()
        for idx, (nx, ny) in enumerate(self.nodes):
            if idx not in self.visited:
                # Use estimated position — if you're lost, you miss the node
                if math.hypot(self.x_est - nx, self.y_est - ny) < NODE_RADIUS_KM:
                    self.visited[idx] = {
                        "quality": q_now,
                        "x_est":   self.x_est,
                        "y_est":   self.y_est,
                        "error":   self.error_km,
                    }
                    print(f"[survey] Node {idx} collected  "
                          f"quality={q_now}  error={self.error_km:.1f}km")
                    if len(self.visited) == N_NODES:
                        self._end(); return

        # ── Deploy ────────────────────────────────────────────────────────
        if i["deploy"]:
            i["deploy"] = False
            if self.buoys_left > 0 and self._buoy_phase == "idle":
                self._buoy_phase = "ascending"
                self._buoy_timer = 0.0
                self.buoy_x = self.x_true
                self.buoy_y = self.y_true
                self.buoys_left -= 1
                print(f"[buoy] Deployed — {self.buoys_left} remaining")

        # ── Buoy state machine ────────────────────────────────────────────
        if self._buoy_phase == "ascending":
            self._buoy_timer += dt
            if self._buoy_timer >= BUOY_ASCENT_S:
                self._buoy_phase = "surface"
                self._buoy_timer = 0.0
                self.drift.reset(self.x_true, self.y_true)
                self.x_est = self.x_true
                self.y_est = self.y_true
                self.error_km = 0.0
                self.fix_count += 1
                self._reveal_rem = 100
                self.fix_markers.append((self.x_true, self.y_true))
                print(f"[buoy] Fix #{self.fix_count} applied")

        elif self._buoy_phase == "surface":
            self._buoy_timer += dt
            if self._buoy_timer >= 2.0:
                self._buoy_phase = "descending"
                self._buoy_timer = 0.0

        elif self._buoy_phase == "descending":
            self._buoy_timer += dt
            if self._buoy_timer >= BUOY_ASCENT_S:
                self._buoy_phase = "idle"

    def _end(self):
        self.game_over   = True
        self.final_score = self.score()
        qb = self.quality_breakdown()
        print(f"[game] GAME OVER  score={self.final_score}  "
              f"H={qb['HIGH']} M={qb['MEDIUM']} L={qb['LOW']}")

    @property
    def buoy_depth_frac(self):
        if self._buoy_phase == "ascending":
            return max(0.0, 1.0 - self._buoy_timer / BUOY_ASCENT_S)
        elif self._buoy_phase == "surface":   return 0.0
        elif self._buoy_phase == "descending":
            return min(1.0, self._buoy_timer / BUOY_ASCENT_S)
        return 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK PHONE CONTROLLER  (D-pad  +  gyro tilt mode)
# ══════════════════════════════════════════════════════════════════════════════
PHONE_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1,
  maximum-scale=1,user-scalable=no">
<title>SNC The Oriel Controller</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;
  -webkit-tap-highlight-color:transparent;touch-action:none}
body{background:#0b0b12;color:#d8dcea;font-family:monospace;
     display:flex;flex-direction:column;align-items:center;
     min-height:100vh;padding:12px;gap:10px;overflow:hidden}
h2{color:#3498db;letter-spacing:3px;font-size:1.05em;margin-top:4px}
#mode-bar{display:flex;gap:8px}
.mode-btn{flex:1;padding:8px;background:#111120;border-radius:10px;
          font-family:monospace;font-size:0.82em;cursor:pointer;
          border:2px solid #1a6699;color:#d8dcea}
.mode-btn.active{background:#1a6699;color:white}
#scorebox{background:#111120;border:1px solid #1a6699;border-radius:8px;
          padding:6px 12px;font-size:0.78em;color:#f39c12;
          text-align:center;width:100%;max-width:320px}
/* D-pad */
#dpad{display:grid;grid-template-columns:repeat(3,82px);
      grid-template-rows:repeat(3,82px);gap:7px}
.btn{background:#111120;border:2px solid #1a6699;border-radius:12px;
     color:#d8dcea;font-size:1.9em;display:flex;align-items:center;
     justify-content:center;cursor:pointer;user-select:none}
.btn:active{background:#1a6699}
.empty{background:transparent!important;border:none!important}
/* Wide buttons */
.wide{width:260px;height:62px;background:#111120;border-radius:12px;
      font-family:monospace;cursor:pointer;user-select:none}
#dep{border:2px solid #f1c40f;color:#f1c40f;font-size:0.95em}
#dep:active{background:#f1c40f;color:#0b0b12}
#rst{border:2px solid #e74c3c;color:#e74c3c;font-size:0.95em}
#rst:active{background:#e74c3c;color:white}
/* Gyro */
#gyro-panel{display:none;flex-direction:column;align-items:center;gap:8px;
            width:100%}
#horizon{width:200px;height:200px;border-radius:50%;
         border:3px solid #1a6699;background:#05111f;
         position:relative;overflow:hidden}
#horizon-line{position:absolute;width:200%;height:3px;
              background:#2ecc71;top:50%;left:-50%;
              transform-origin:center center}
#tilt-txt{font-size:0.75em;color:#888;text-align:center}
#sens-row{display:flex;align-items:center;gap:8px;font-size:0.78em}
#sens{width:120px}
#quality-badge{font-size:0.82em;padding:4px 10px;border-radius:6px;
               border:1px solid #555;text-align:center;min-width:180px}
</style>
</head>
<body>
<h2>⚓ SNC The Oriel CONTROLLER</h2>

<div id="mode-bar">
  <button class="mode-btn active" onclick="setMode('dpad')">🕹 D-Pad</button>
  <button class="mode-btn" onclick="setMode('gyro')">📱 Tilt (Gyro)</button>
</div>

<div id="scorebox">Score: 0 | Nodes: 0/14 | Time: 90s | Buoys: 3</div>
<div id="quality-badge" style="background:#111120;color:#2ecc71">
  NAV QUALITY: HIGH
</div>

<!-- D-PAD -->
<div id="dpad-panel">
  <div id="dpad">
    <div class="empty"></div>
    <div class="btn"
      ontouchstart="p('thrust')" ontouchend="r('thrust')"
      onmousedown="p('thrust')" onmouseup="r('thrust')">▲</div>
    <div class="empty"></div>
    <div class="btn"
      ontouchstart="p('left')" ontouchend="r('left')"
      onmousedown="p('left')" onmouseup="r('left')">◀</div>
    <div class="btn"
      ontouchstart="p('brake')" ontouchend="r('brake')"
      onmousedown="p('brake')" onmouseup="r('brake')">▼</div>
    <div class="btn"
      ontouchstart="p('right')" ontouchend="r('right')"
      onmousedown="p('right')" onmouseup="r('right')">▶</div>
  </div>
</div>

<!-- GYRO -->
<div id="gyro-panel">
  <div id="horizon"><div id="horizon-line"></div></div>
  <div id="tilt-txt">Tilt phone to steer<br>Fwd/back = thrust/brake</div>
  <div id="sens-row">
    Sensitivity:
    <input type="range" id="sens" min="1" max="10" value="5">
    <span id="sens-val">5</span>
  </div>
</div>

<button class="wide" id="dep"
  ontouchstart="once('deploy')" onmousedown="once('deploy')">
  📡 DEPLOY USBL BUOY
</button>
<button class="wide" id="rst"
  ontouchstart="doReset()" onmousedown="doReset()">
  🔄 NEW SURVEY RUN
</button>

<script>
// ── Mode switch ─────────────────────────────────────────────────────────────
let gyroMode = false;
function setMode(m) {
  gyroMode = (m === 'gyro');
  document.getElementById('dpad-panel').style.display = gyroMode?'none':'block';
  document.getElementById('gyro-panel').style.display = gyroMode?'flex':'none';
  document.querySelectorAll('.mode-btn').forEach((b,i)=>{
    b.classList.toggle('active', (i===0 && !gyroMode)||(i===1 && gyroMode));
  });
  if (gyroMode) requestGyro();
}

// ── D-pad ───────────────────────────────────────────────────────────────────
function p(k){post({k,v:true})}
function r(k){post({k,v:false})}
function once(k){post({k,v:true})}
function doReset(){fetch('/reset',{method:'POST'})}

// ── Gyro ─────────────────────────────────────────────────────────────────────
let gyroGranted = false;
function requestGyro() {
  if (typeof DeviceOrientationEvent !== 'undefined' &&
      typeof DeviceOrientationEvent.requestPermission === 'function') {
    DeviceOrientationEvent.requestPermission()
      .then(s=>{ if(s==='granted'){gyroGranted=true;startGyro();} })
      .catch(console.error);
  } else {
    gyroGranted = true; startGyro();
  }
}

let lastGyroSend = 0;
const GYRO_INTERVAL = 80;   // ms

function startGyro() {
  window.addEventListener('deviceorientation', (e) => {
    if (!gyroMode) return;
    const now = Date.now();
    if (now - lastGyroSend < GYRO_INTERVAL) return;
    lastGyroSend = now;

    const sens = parseInt(document.getElementById('sens').value) / 5.0;
    const roll  = e.gamma || 0;   // left/right tilt  (-90 to 90)
    const pitch = e.beta  || 0;   // fwd/back tilt   (-180 to 180)

    // Deadzone ±5 deg
    const deadzone = 5.0;
    const rollN  = Math.abs(roll)  > deadzone ? roll  / (30.0 / sens) : 0;
    const pitchN = Math.abs(pitch-45) > deadzone ? -(pitch-45) / (30.0/sens) : 0;
    // pitch-45: 45° is natural "hold phone upright" offset

    post({
      gyro: true,
      roll:  Math.max(-1, Math.min(1, rollN)),
      pitch: Math.max(-1, Math.min(1, pitchN)),
    });

    // Update horizon visual
    const hl = document.getElementById('horizon-line');
    hl.style.transform = `rotate(${roll}deg) translateY(${-(pitch-45)*2}px)`;
    document.getElementById('tilt-txt').textContent =
      `Roll: ${roll.toFixed(0)}°  Pitch: ${(pitch-45).toFixed(0)}°`;
  });
}

document.getElementById('sens').addEventListener('input', function(){
  document.getElementById('sens-val').textContent = this.value;
});

// ── Comms ────────────────────────────────────────────────────────────────────
function post(data) {
  fetch('/inp', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(data)});
}

// ── Poll state ───────────────────────────────────────────────────────────────
const qColors = {HIGH:'#2ecc71', MEDIUM:'#f39c12', LOW:'#e74c3c'};
setInterval(()=>{
  fetch('/state').then(r=>r.json()).then(d=>{
    document.getElementById('scorebox').textContent =
      'Score: '+d.score+' | Nodes: '+d.nodes+'/'+d.total+
      ' | Time: '+d.time.toFixed(0)+'s | Buoys: '+d.buoys;
    const qb = document.getElementById('quality-badge');
    qb.textContent = 'NAV QUALITY: '+d.quality;
    qb.style.color = qColors[d.quality]||'#fff';
    qb.style.borderColor = qColors[d.quality]||'#555';
    if (d.over) {
      document.getElementById('scorebox').textContent =
        '🏁 SURVEY COMPLETE  Final score: '+d.score;
    }
  }).catch(()=>{});
}, 400);
</script>
</body>
</html>"""


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except: return "127.0.0.1"


def start_flask(gs):
    if not FLASK_OK: return
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app = Flask(__name__); app.logger.disabled = True

    @app.route("/")
    def index(): return PHONE_HTML

    @app.route("/inp", methods=["POST"])
    def inp():
        d = request.get_json(silent=True) or {}
        if d.get("gyro"):
            # Gyro tilt → map to thrust/brake/left/right
            roll  = float(d.get("roll",  0))
            pitch = float(d.get("pitch", 0))
            THRESH = 0.15
            gs.inp["left"]   = roll  < -THRESH
            gs.inp["right"]  = roll  >  THRESH
            gs.inp["thrust"] = pitch >  THRESH
            gs.inp["brake"]  = pitch < -THRESH
        else:
            k, v = d.get("k",""), bool(d.get("v", False))
            if k == "deploy" and v:
                gs.inp["deploy"] = True
            elif k in gs.inp and k != "deploy":
                gs.inp[k] = v
        return jsonify(ok=True)

    @app.route("/reset", methods=["POST"])
    def reset():
        gs._reset_requested = True; return jsonify(ok=True)

    @app.route("/state")
    def state():
        return jsonify(
            score=gs.score(), nodes=len(gs.visited),
            total=N_NODES, time=gs.time_left,
            buoys=gs.buoys_left, over=gs.game_over,
            quality=gs.drift.quality())

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=FLASK_PORT, debug=False),
        daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _sub_pts(cx, cy, hdg, size=16):
    L, W = size, size*0.28
    pts = np.array([[L*.5,0],[L*.2,W*.5],[-L*.35,W*.5],
                    [-L*.5,W*.2],[-L*.5,-W*.2],
                    [-L*.35,-W*.5],[L*.2,-W*.5]])
    c, s = math.cos(hdg), math.sin(hdg)
    R = np.array([[c,-s],[s,c]])
    pts = (R @ pts.T).T
    pts[:,0]+=cx; pts[:,1]+=cy
    return pts


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    gs  = GameState()
    ip  = get_local_ip()
    start_flask(gs)

    print("\n" + "═"*58)
    print("  SNC SIMULATOR — The Oriel SURVEY MISSION  (Manual Mode)")
    print("═"*58)
    print("  OBJECTIVE: Survey the CCZ seabed.  Collect nodes.")
    print("  WARNING:   INS drift degrades data quality over time.")
    print("             Deploy USBL buoy to reset navigation.")
    print("             Node capture uses ESTIMATED position.")
    print("  NOTE:      The Oriel always moves at cruise speed.")
    print("             You can only control heading direction.")
    print()
    print("  KEYBOARD:  A/← D/→ Turn        SPACE  Deploy buoy")
    print("             I  Instructions      R  New run   Q  Quit")
    if PYGAME_OK:
        print(f"  PS4:       L-stick Turn  □/△ Deploy  Options Reset")
    if FLASK_OK:
        print(f"\n  PHONE CONTROL (same WiFi):")
        print(f"  ➜  Open in phone browser:  http://{ip}:{FLASK_PORT}")
    print("═"*58 + "\n")

    # ── Instructions panel state ──────────────────────────────────────────
    _show_instructions = [True]   # mutable container so closure can modify

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 9.5), facecolor=BG)
    fig.canvas.manager.set_window_title("SNC Simulator — The Oriel Survey Mission")

    gs_l = gridspec.GridSpec(
        2, 3, fig,
        height_ratios=[2.3, 1.0],
        width_ratios=[0.95, 1.6, 1.0],
        hspace=0.28, wspace=0.24,
        left=0.04, right=0.97, top=0.94, bottom=0.05)

    ax_3d  = fig.add_subplot(gs_l[0, 0], projection='3d')
    ax_map = fig.add_subplot(gs_l[0, 1:3])
    ax_par = fig.add_subplot(gs_l[1, 0])
    ax_err = fig.add_subplot(gs_l[1, 1])   # live error graph
    ax_lb  = fig.add_subplot(gs_l[1, 2])   # leaderboard

    def _s2d(ax, title=""):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e1e30"); sp.set_linewidth(0.6)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.8)
        if title: ax.set_title(title, color=TEXT, fontsize=8.5,
                                fontweight="bold", pad=4)

    # ── 3D panel ──────────────────────────────────────────────────────────
    ax_3d.set_facecolor("#05111f")
    ax_3d.set_title("The Oriel + Buoy (3D)", color=TEXT, fontsize=8.5,
                    fontweight="bold", pad=4)
    for lbl,fn in [("E[km]",ax_3d.set_xlabel),("N[km]",ax_3d.set_ylabel),
                   ("Depth[m]",ax_3d.set_zlabel)]:
        fn(lbl, color=TEXT, fontsize=5.5, labelpad=1)
    ax_3d.tick_params(colors=TEXT, labelsize=4.5)
    ax_3d.set_zlim(650, 0)
    ax_3d.view_init(elev=22, azim=-55)
    xx, yy = np.meshgrid([-1,1],[-1,1])
    ax_3d.plot_surface(xx,yy,np.zeros_like(xx),alpha=0.07,color="#1a6699")

    tr3t, = ax_3d.plot([],[],[], color=C_TRUE, lw=0.8, alpha=0.0)
    tr3d, = ax_3d.plot([],[],[], color=C_DRFT, lw=0.8, alpha=0.7, ls="--")
    sub3d  = ax_3d.scatter([0],[0],[600], c=C_SUB, s=40, zorder=5)
    buo3d  = ax_3d.scatter([],[],[], c=C_BUOY, s=25, zorder=5)
    tet3d, = ax_3d.plot([],[],[], color=C_BUOY, lw=0.7, alpha=0.8)
    lbl3d  = ax_3d.text2D(0.03, 0.97, "", transform=ax_3d.transAxes,
                           color=TEXT, fontsize=6, va="top")

    # ── Survey map ────────────────────────────────────────────────────────
    _s2d(ax_map)
    ax_map.set_title(
        "Survey Map  —  ESTIMATED trail shown  |  TRUTH hidden  "
        "(deploy buoy to reveal, score degrades with drift)",
        color=TEXT, fontsize=8, fontweight="bold", pad=4)
    ax_map.set_xlabel("East [km]", color=TEXT, fontsize=7)
    ax_map.set_ylabel("North [km]", color=TEXT, fontsize=7)

    # CCZ border
    bx=[CCZ_X[0],CCZ_X[1],CCZ_X[1],CCZ_X[0],CCZ_X[0]]
    by=[CCZ_Y[0],CCZ_Y[0],CCZ_Y[1],CCZ_Y[1],CCZ_Y[0]]
    ax_map.plot(bx, by, color=C_BDR, lw=1.0, ls="--", alpha=0.4, zorder=1)
    ax_map.fill(bx, by, color="#040d18", alpha=0.5, zorder=0)

    # Quality threshold rings (legend only — not per-node)
    ax_map.plot([],[], color="#2ecc71", lw=0, marker="D", ms=7,
                label=f"HIGH  (<{Q_HIGH_KM}km drift)")
    ax_map.plot([],[], color="#f39c12", lw=0, marker="D", ms=7,
                label=f"MED   (<{Q_MED_KM}km drift)")
    ax_map.plot([],[], color="#e74c3c", lw=0, marker="D", ms=7,
                label="LOW   (>15km drift)")
    ax_map.plot([],[], color=C_DRFT,  lw=1.5, ls="-",
                label="INS estimate (visible)")
    ax_map.plot([],[], color=C_TRUE,  lw=1.5, ls="-",
                label="True track (hidden)")
    leg = ax_map.legend(loc="lower left", fontsize=6.5, framealpha=0.4,
                        labelcolor=TEXT, facecolor=PANEL, ncol=2)

    # Nodes — colour updated live per quality on collection
    node_sc = ax_map.scatter(
        [n[0] for n in gs.nodes], [n[1] for n in gs.nodes],
        s=120, c=[C_ND_U]*N_NODES, zorder=4,
        marker="D", edgecolors="white", linewidths=0.35)
    node_circles = [
        ax_map.add_patch(Circle((nx,ny), NODE_RADIUS_KM,
                                color=C_ND_U, alpha=0.09, zorder=2))
        for nx,ny in gs.nodes]

    drift_ln, = ax_map.plot([], [], color=C_DRFT, lw=1.1, alpha=0.95,
                             zorder=5, solid_capstyle="round")
    truth_ln, = ax_map.plot([], [], color=C_TRUE, lw=1.1, alpha=0.0,
                             zorder=6, solid_capstyle="round")
    fix_mk,   = ax_map.plot([], [], "x", color=C_BUOY, ms=9, mew=2,
                             zorder=9, label="Fix point")
    buoy_mk,  = ax_map.plot([], [], "^", color=C_BUOY, ms=9, zorder=9)
    sub_poly  = plt.Polygon(_sub_pts(0,0,0), fc=C_SUB, ec="white", lw=0.5,
                             zorder=8, alpha=0.9)
    ax_map.add_patch(sub_poly)
    _arr = [None]

    def _hud(ax, x, y, ha="left", fs=9):
        return ax.text(x, y, "", transform=ax.transAxes,
                       color=TEXT, fontsize=fs, fontweight="bold",
                       va="top", ha=ha,
                       bbox=dict(fc=PANEL, ec=C_BDR, alpha=0.85, pad=3))

    hud_t  = _hud(ax_map, 0.01, 0.97)
    hud_s  = _hud(ax_map, 0.01, 0.87)
    hud_b  = _hud(ax_map, 0.99, 0.97, "right")
    hud_n  = _hud(ax_map, 0.99, 0.87, "right")
    hud_q  = _hud(ax_map, 0.01, 0.77, fs=8)
    hud_ov = ax_map.text(0.5, 0.5, "",
                          transform=ax_map.transAxes,
                          color="#f1c40f", fontsize=16, fontweight="bold",
                          ha="center", va="center", zorder=20,
                          bbox=dict(fc="#08080f", ec="#f1c40f",
                                    alpha=0.0, boxstyle="round,pad=0.8"))

    # ── START screen overlay ───────────────────────────────────────────────
    start_ov = ax_map.text(
        0.5, 0.5,
        "📖  READ THE INSTRUCTIONS FIRST\n\n"
        "Check the HOW TO PLAY panel on the right →\n\n"
        "  Keyboard:  A / ←  D / →  steer\n"
        "             SPACE  deploy buoy     R  restart\n\n"
        "  PS4:       Left thumbstick  steer\n"
        "             △ Triangle  deploy buoy\n"
        "             L1 / L2  restart\n\n"
        "  ▶  Click anywhere or press any key to BEGIN",
        transform=ax_map.transAxes,
        color="#f1c40f", fontsize=11, fontweight="bold",
        ha="center", va="center", zorder=25,
        linespacing=1.6,
        bbox=dict(fc="#06060f", ec="#f1c40f",
                  alpha=0.96, boxstyle="round,pad=1.0"))
    _start_ov = [start_ov]   # mutable for closure

    def _dismiss_start(*_):
        if not gs.game_started:
            gs.game_started = True
            gs.t_start = time.time()
            _start_ov[0].set_visible(False)
            plt.draw()

    fig.canvas.mpl_connect("button_press_event", _dismiss_start)

    # ── Parameter panel ───────────────────────────────────────────────────
    ax_par.set_facecolor(PANEL); ax_par.set_xticks([]); ax_par.set_yticks([])
    for sp in ax_par.spines.values(): sp.set_edgecolor("#1e1e30")
    ax_par.set_title("The Oriel Status", color=TEXT, fontsize=8.5,
                      fontweight="bold", pad=4)

    def _row(y, lbl, col=TEXT):
        ax_par.text(0.04, y, lbl, transform=ax_par.transAxes,
                    color="#6677aa", fontsize=7, va="top")
        return ax_par.text(0.52, y, "—", transform=ax_par.transAxes,
                           color=col, fontsize=7.5, fontweight="bold", va="top")

    p_x   = _row(0.91, "X (estimated):")
    p_y   = _row(0.82, "Y (estimated):")
    p_spd = _row(0.73, "Speed:")
    p_hdg = _row(0.64, "Heading:")
    p_err = _row(0.55, "Nav error:")
    p_q   = _row(0.46, "Data quality:")
    p_bst = _row(0.37, "Buoy status:")
    p_fix = _row(0.28, "Fixes used:")
    ax_par.text(0.04, 0.16,
                f"📱 http://{ip}:{FLASK_PORT}" if FLASK_OK else
                ("🎮 PS4 controller active" if PYGAME_OK else "⌨ keyboard only"),
                transform=ax_par.transAxes, color="#5dade2", fontsize=6, va="top")
    ax_par.text(0.04, 0.08,
                "A/← left  D/→ right\nSPC deploy buoy  I instructions",
                transform=ax_par.transAxes, color="#334455",
                fontsize=5.5, va="top")

    # ── Error timeline graph ───────────────────────────────────────────────
    _s2d(ax_err, "Navigation Error  (node capture fails above red line)")
    ax_err.set_xlabel("Time [s]", color=TEXT, fontsize=7)
    ax_err.set_ylabel("Error [km]", color=TEXT, fontsize=7)
    ax_err.axhline(NODE_RADIUS_KM, color="#e74c3c", lw=1.0, ls="--",
                   alpha=0.8, label=f"Capture radius {NODE_RADIUS_KM}km")
    ax_err.axhline(Q_HIGH_KM, color="#2ecc71", lw=0.8, ls=":",
                   alpha=0.6, label=f"HIGH quality {Q_HIGH_KM}km")
    ax_err.axhline(Q_MED_KM,  color="#f39c12", lw=0.8, ls=":",
                   alpha=0.6, label=f"MED quality {Q_MED_KM}km")
    ax_err.legend(fontsize=5.5, framealpha=0.3, labelcolor=TEXT,
                  facecolor=PANEL, loc="upper left")
    ax_err.set_xlim(0, GAME_DURATION_S)
    ax_err.set_ylim(-0.1, 8)
    err_ln, = ax_err.plot([], [], color="#5dade2", lw=1.2, zorder=4)
    fix_vlines = []

    # ── Instructions panel (replaces leaderboard) ─────────────────────────
    ax_lb.set_facecolor("#080814")
    ax_lb.set_xticks([]); ax_lb.set_yticks([])
    for sp in ax_lb.spines.values():
        sp.set_edgecolor(C_BDR); sp.set_linewidth(1.2)
    ax_lb.set_title("📖  HOW TO PLAY", color=C_SCORE, fontsize=9,
                     fontweight="bold", pad=5)

    INSTR_TEXT = (
        "YOUR MISSION\n"
        "You are piloting The Oriel, operating at 600 m below the\n"
        "Pacific Ocean, surveying the Clarion-\n"
        "Clipperton Zone — home to 21 billion\n"
        "tonnes of polymetallic nodules.\n"
        "\n"
        "Without GPS, your navigation system\n"
        "drifts over time. Collect ◆ survey nodes\n"
        "while your position is accurate — or the\n"
        "data you record will be worthless.\n"
        "\n"
        "WHY THIS MATTERS\n"
        "HIGH (green):  precise data  — x3 score.\n"
        "MEDIUM (orange): degraded    — x1.5.\n"
        "LOW (red): nodes GLITCH & DIM on your\n"
        "map — data nearly useless   — x0.5.\n"
        "\n"
        "KEYBOARD CONTROLS\n"
        "A / ←  D / →   Steer left / right\n"
        "SPACE           Deploy USBL buoy\n"
        "R               New survey run\n"
        "I               Toggle this panel\n"
        "\n"
        "PS4 CONTROLLER\n"
        "Left thumbstick  Steer\n"
        "△ (Triangle)     Deploy USBL buoy\n"
        "L1 / L2          New survey run\n"
        "\n"
        "The Oriel never stops — steer wisely.\n"
        "You have 3 USBL buoys. Use them well."
    )

    ax_lb.text(0.05, 0.96, INSTR_TEXT,
               transform=ax_lb.transAxes,
               color=TEXT, fontsize=6.2, va="top",
               fontfamily="monospace",
               linespacing=1.45)

    # Dismiss hint at bottom
    ax_lb.text(0.5, 0.02,
               "Press  I  to hide/show this panel",
               transform=ax_lb.transAxes,
               color="#334455", fontsize=6, ha="center", va="bottom")

    # Leaderboard texts (hidden — kept for data compat, won't display)
    lb_texts = [ax_lb.text(0.5, 0.5, "", transform=ax_lb.transAxes,
                            visible=False) for _ in range(5)]

    fig.suptitle(
        "SNC Simulator  |  CCZ Autonomous Survey — Manual Navigation Mode",
        color=TEXT, fontsize=11, fontweight="bold", y=0.998)

    # ── Keyboard ──────────────────────────────────────────────────────────
    def on_press(ev):
        _dismiss_start()   # any key also starts the game
        k = ev.key
        if k in ("a","left"):   gs.inp["left"]   = True
        elif k in ("d","right"):  gs.inp["right"]  = True
        elif k == " ":            gs.inp["deploy"] = True
        elif k in ("r","R"):      gs._reset_requested = True
        elif k in ("i","I"):
            _show_instructions[0] = not _show_instructions[0]
            ax_lb.set_visible(_show_instructions[0])
            plt.draw()
        elif k in ("q","escape"): plt.close(fig)

    def on_release(ev):
        k = ev.key
        if k in ("a","left"):   gs.inp["left"]   = False
        elif k in ("d","right"):  gs.inp["right"]  = False

    fig.canvas.mpl_connect("key_press_event",   on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)

    # ── Animation ─────────────────────────────────────────────────────────
    def _tick(frame):
        # ── PS4 / gamepad polling ─────────────────────────────────────────
        if PYGAME_OK and _joystick is not None:
            pygame.event.pump()
            DEAD = 0.25   # deadzone
            # Left stick X axis → steer (axis 0)
            lx = _joystick.get_axis(0)
            gs.inp["left"]  = lx < -DEAD
            gs.inp["right"] = lx >  DEAD
            # Triangle button (btn 3 on PS4 in pygame) → deploy buoy
            if _joystick.get_button(3):
                gs.inp["deploy"] = True
            # L1 (btn 4) or L2 (btn 6) → reset
            if _joystick.get_button(4) or _joystick.get_button(6):
                gs._reset_requested = True

        # ── Reset ─────────────────────────────────────────────────────────
        if gs._reset_requested:
            gs._reset_requested = False
            if gs.game_over and gs.final_score > 0:
                qb = gs.quality_breakdown()
                gs.leaderboard.append((
                    f"Run {len(gs.leaderboard)+1}",
                    gs.final_score, len(gs.visited),
                    qb["HIGH"], qb["MEDIUM"], qb["LOW"],
                    round(GAME_DURATION_S - gs.time_left, 1)))
                gs.leaderboard.sort(key=lambda x: -x[1])
                gs.leaderboard = gs.leaderboard[:5]
            lb = gs.leaderboard
            gs.reset(); gs.leaderboard = lb
            # Show start overlay again for new run
            _start_ov[0].set_visible(True)
            # Rebuild nodes
            node_sc.set_offsets(np.c_[[n[0] for n in gs.nodes],
                                       [n[1] for n in gs.nodes]])
            node_sc.set_color([C_ND_U]*N_NODES)
            for i,(nx,ny) in enumerate(gs.nodes):
                node_circles[i].center = (nx,ny)
                node_circles[i].set_facecolor(C_ND_U)
                node_circles[i].set_alpha(0.09)
            for vl in fix_vlines: vl.remove()
            fix_vlines.clear()
            fix_mk.set_data([], [])      # clear fix-point X markers from map
            drift_ln.set_data([], [])    # clear estimated trail
            truth_ln.set_data([], [])    # clear true trail
            err_ln.set_data([], [])
            hud_ov.set_alpha(0.0)
            hud_ov.get_bbox_patch().set_alpha(0.0)
            return

        dt = FRAME_MS / 1000.0
        gs.step(dt)

        err   = gs.error_km
        qual  = gs.drift.quality()

        # ── Camera ────────────────────────────────────────────────────────
        cx, cy = gs.x_est, gs.y_est
        ax_map.set_xlim(max(CCZ_X[0]-30, cx-VIEW_HW),
                        min(CCZ_X[1]+30, cx+VIEW_HW))
        ax_map.set_ylim(max(CCZ_Y[0]-30, cy-VIEW_HH),
                        min(CCZ_Y[1]+30, cy+VIEW_HH))

        # ── 3D ────────────────────────────────────────────────────────────
        R3 = 80.0
        ax_3d.set_xlim(gs.x_true-R3, gs.x_true+R3)
        ax_3d.set_ylim(gs.y_true-R3, gs.y_true+R3)
        ax_3d.view_init(elev=22, azim=ax_3d.azim+0.07)

        tail = 200
        tt = list(gs.trail_true)[-tail:]
        dt_ = list(gs.trail_est)[-tail:]
        if len(dt_) > 1:
            tr3d.set_data_3d([p[0] for p in dt_],
                              [p[1] for p in dt_], [600]*len(dt_))
        ta = 0.75 if gs._reveal_rem > 0 else 0.0
        if len(tt) > 1 and ta > 0:
            tr3t.set_data_3d([p[0] for p in tt],
                              [p[1] for p in tt], [600]*len(tt))
            tr3t.set_alpha(ta*0.6)
        else:
            tr3t.set_alpha(0.0)
        sub_col = C_WARN if gs.q_warn_flash > 0 else C_SUB
        sub3d._offsets3d = (np.array([gs.x_true]),
                             np.array([gs.y_true]), np.array([600.0]))
        sub3d.set_color(sub_col)
        if gs._buoy_phase in ("ascending","surface","descending"):
            bd = gs.buoy_depth_frac * 600.0
            buo3d._offsets3d = (np.array([gs.buoy_x]),
                                  np.array([gs.buoy_y]), np.array([bd]))
            buo3d.set_visible(True)
            tet3d.set_data_3d([gs.buoy_x,gs.buoy_x],
                               [gs.buoy_y,gs.buoy_y],[600,bd])
        else:
            buo3d.set_visible(False)
            tet3d.set_data_3d([],[],[])

        lbl_map = {"ascending": f"▲ BUOY {int(100*(1-gs.buoy_depth_frac))}%",
                   "surface":   "● SURFACE FIX!",
                   "descending":"▼ DESCENDING"}
        if gs._buoy_phase in lbl_map:
            lbl3d.set_text(lbl_map[gs._buoy_phase])
            lbl3d.set_color(C_BUOY if gs._buoy_phase!="surface" else "#5dade2")
        elif err > Q_MED_KM:
            lbl3d.set_text(f"⚠ DRIFT {err:.1f}km"); lbl3d.set_color(C_WARN)
        else:
            lbl3d.set_text(f"Depth 600m | {gs.speed:.1f}km/s")
            lbl3d.set_color(TEXT)

        # ── Map trails ────────────────────────────────────────────────────
        all_d = list(gs.trail_est)
        all_t = list(gs.trail_true)
        if all_d:
            drift_ln.set_data([p[0] for p in all_d],
                               [p[1] for p in all_d])
        truth_ln.set_alpha(0.75 if gs._reveal_rem > 0 else 0.0)
        if all_t and gs._reveal_rem > 0:
            truth_ln.set_data([p[0] for p in all_t],
                               [p[1] for p in all_t])

        # ── Nodes — colour by collection quality + glitch for MEDIUM/LOW ───
        cols = []
        node_positions = []
        gx_off, gy_off = gs.drift.glitch_offset()   # shared jitter this frame
        for i, (nx, ny) in enumerate(gs.nodes):
            if i in gs.visited:
                cols.append(Q_COLOURS[gs.visited[i]["quality"]])
                node_positions.append((nx, ny))
            else:
                cols.append(C_ND_U)
                # Unvisited nodes shimmer/scramble with nav error
                q = gs.drift.quality()
                if q == "HIGH":
                    node_positions.append((nx, ny))
                else:
                    # Each node gets its own jitter but correlated via shared offset
                    per = gs.drift.glitch_offset()
                    node_positions.append((nx + gx_off * 0.6 + per[0] * 0.4,
                                           ny + gy_off * 0.6 + per[1] * 0.4))
        node_sc.set_offsets(np.c_[[p[0] for p in node_positions],
                                   [p[1] for p in node_positions]])
        node_sc.set_color(cols)
        # Alpha for unvisited nodes: dim when nav quality degrades
        _node_alpha = {"HIGH": 0.90, "MEDIUM": 0.45, "LOW": 0.18}
        node_sc.set_alpha(_node_alpha[qual])
        for i,c in enumerate(node_circles):
            if i in gs.visited:
                c.set_alpha(0.0)
            else:
                nx, ny = node_positions[i]   # already glitched position
                c.center = (nx, ny)
                c.set_facecolor(Q_COLOURS[qual])
                c.set_alpha(0.12 if qual=="HIGH" else (0.06 if qual=="MEDIUM" else 0.03))

        # ── Sub polygon ───────────────────────────────────────────────────
        sub_poly.set_xy(_sub_pts(gs.x_est, gs.y_est, gs.heading, 16))
        sub_poly.set_facecolor(sub_col)
        if _arr[0] is not None:
            try: _arr[0].remove()
            except: pass
        _arr[0] = None
        if not gs.game_over:
            al = 18
            _arr[0] = FancyArrow(
                gs.x_est, gs.y_est,
                math.cos(gs.heading)*al, math.sin(gs.heading)*al,
                width=al*0.17, color="white", length_includes_head=True,
                head_width=al*0.38, head_length=al*0.4,
                zorder=12, alpha=0.85)
            ax_map.add_patch(_arr[0])

        if gs.fix_markers:
            fix_mk.set_data([p[0] for p in gs.fix_markers],
                             [p[1] for p in gs.fix_markers])
        if gs._buoy_phase in ("ascending","surface"):
            buoy_mk.set_data([gs.buoy_x],[gs.buoy_y])
        else:
            buoy_mk.set_data([],[])

        # ── HUD ───────────────────────────────────────────────────────────
        tl = gs.time_left
        tc = C_WARN if tl<20 else ("#f39c12" if tl<45 else TEXT)
        hud_t.set_text(f"⏱  {tl:.0f}s"); hud_t.set_color(tc)
        hud_s.set_text(f"⭐  {gs.score()}"); hud_s.set_color(C_SCORE)
        hud_b.set_text("BUOYS  "+"📡"*gs.buoys_left+"○"*(N_BUOYS-gs.buoys_left))
        hud_b.set_color(C_BUOY)
        hud_n.set_text(f"NODES  {len(gs.visited)}/{N_NODES}")
        hud_n.set_color(C_TRUE)
        qc = Q_COLOURS[qual]
        warn = "  ⚠ DEPLOY BUOY" if err > NODE_RADIUS_KM else ""
        hud_q.set_text(f"NAV QUALITY: {qual}  ({err:.1f}km error){warn}")
        hud_q.set_color(qc)

        if gs.game_over:
            qb = gs.quality_breakdown()
            hud_ov.set_text(
                f"🏁  SURVEY COMPLETE\n"
                f"Score: {gs.final_score}\n"
                f"H={qb['HIGH']}  M={qb['MEDIUM']}  L={qb['LOW']}  "
                f"/ {N_NODES} nodes\n"
                f"Press  R  for new run")
            hud_ov.set_alpha(1.0)
            hud_ov.get_bbox_patch().set_alpha(0.93)

        # ── Param panel ───────────────────────────────────────────────────
        p_x.set_text(f"{gs.x_est:+.1f} km")
        p_y.set_text(f"{gs.y_est:+.1f} km")
        p_spd.set_text(f"{gs.speed:.2f} km/s")
        p_hdg.set_text(f"{math.degrees(gs.heading)%360:.0f}°")
        p_err.set_text(f"{err:.1f} km"); p_err.set_color(qc)
        p_q.set_text(qual);  p_q.set_color(qc)
        bst_l={"idle":"IDLE","ascending":"ASCENDING ▲",
               "surface":"● FIX APPLIED","descending":"DESCENDING ▼"}
        bst_c={"idle":"#2ecc71","ascending":C_BUOY,
               "surface":"#5dade2","descending":"#aaaaaa"}
        p_bst.set_text(bst_l[gs._buoy_phase])
        p_bst.set_color(bst_c[gs._buoy_phase])
        p_fix.set_text(f"{gs.fix_count} / {N_BUOYS}")

        # ── Error graph ───────────────────────────────────────────────────
        if gs.quality_log:
            ts = [q[0] for q in gs.quality_log]
            es = [q[1] for q in gs.quality_log]
            err_ln.set_data(ts, es)
            # Colour the line segment by quality
            ax_err.set_ylim(-0.1, max(8, max(es)*1.1))
        # Fix vertical lines
        while len(fix_vlines) < gs.fix_count:
            t_fix = gs.quality_log[
                min(len(gs.quality_log)-1,
                    int(len(gs.quality_log)*
                        len(fix_vlines)/max(gs.fix_count,1)))][0]
            vl = ax_err.axvline(t_fix, color=C_BUOY, lw=1.2,
                                 ls="--", alpha=0.7)
            fix_vlines.append(vl)

        # ── Instructions panel — static, no per-frame update needed ─────────
        pass

        plt.draw()

    ani = animation.FuncAnimation(fig, _tick, interval=FRAME_MS,
                                   cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    run()
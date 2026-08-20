/* ══════════════════════════════════════════════════════════════════════
   SNC SIMULATOR — browser port
   Ported 1:1 (physics + scoring) from snc_manual.py
   ══════════════════════════════════════════════════════════════════════ */

// ── GAME / PHYSICS PARAMETERS ──────────────────────────────────────────
const GAME_DURATION_S = 90;

const N_NODES        = 14;
const NODE_RADIUS_KM = 8;

const N_BUOYS       = 5;
const BUOY_ASCENT_S = 2.0;

const CRUISE_SPEED  = 50.0;   // km/s
const TURN_RATE_DEG = 85.0;   // deg/s

const INS_SIGMA_BASE   = 0.55;
const INS_CORR_TIME    = 20.0;
const INS_ACCEL_FACTOR = 2.2;

const Q_HIGH_KM = 2.5;
const Q_MED_KM  = 4.0;
const Q_WEIGHTS = { HIGH: 3.0, MEDIUM: 1.5, LOW: 0.5 };
const Q_COLOURS = { HIGH: "#2ecc71", MEDIUM: "#f39c12", LOW: "#e74c3c" };

const CCZ_X = [-700.0, 700.0];
const CCZ_Y = [-280.0, 280.0];

const VIEW_HW = 180.0;
const VIEW_HH = 120.0;

// ── COLOURS ─────────────────────────────────────────────────────────────
const BG    = "#0b0b12";
const GRID  = "#161628";
const C_BDR = "#1a6699";
const C_SUB = "#3498db";
const C_WARN= "#e74c3c";
const C_TRUE= "#2ecc71";
const C_BUOY= "#f1c40f";
const C_ND_U= "#1a3a5c";

// ── RNG helpers ─────────────────────────────────────────────────────────
function gauss(mean = 0, std = 1) {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
function uniform(a, b) { return a + Math.random() * (b - a); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function hypot(dx, dy) { return Math.sqrt(dx * dx + dy * dy); }

// ══════════════════════════════════════════════════════════════════════
//  INS DRIFT MODEL
// ══════════════════════════════════════════════════════════════════════
class INSDrift {
  constructor() { this.reset(0, 0); }

  reset(xTrue, yTrue) {
    this.x_est = xTrue;
    this.y_est = yTrue;
    this._vxErr = 0.0;
    this._vyErr = 0.0;
    this._elapsed = 0.0;
    this._side = Math.random() < 0.5 ? -1.0 : 1.0;
    this.error_km = 0.0;
    this._scale = uniform(0.80, 1.20);
  }

  step(xTrue, yTrue, heading, speed, dt) {
    if (speed < 0.5) {
      this.x_est += gauss(0, 0.002);
      this.y_est += gauss(0, 0.002);
      this.error_km = hypot(this.x_est - xTrue, this.y_est - yTrue);
      return;
    }

    this._elapsed += dt;

    const alpha = dt / INS_CORR_TIME;
    const sigmaW = INS_SIGMA_BASE * Math.sqrt((2.0 * dt) / INS_CORR_TIME) * this._scale;

    this._vxErr += -alpha * this._vxErr + sigmaW * gauss(0, 1);
    this._vyErr += -alpha * this._vyErr + sigmaW * gauss(0, 1);

    const t = this._elapsed;
    const bias = INS_SIGMA_BASE * 0.6 * this._scale * Math.pow(t / 10.0, INS_ACCEL_FACTOR);
    const lx = -Math.sin(heading) * this._side;
    const ly = Math.cos(heading) * this._side;

    const vxTotal = this._vxErr + bias * lx;
    const vyTotal = this._vyErr + bias * ly;

    this.x_est += (speed * Math.cos(heading) + vxTotal) * dt;
    this.y_est += (speed * Math.sin(heading) + vyTotal) * dt;

    this.error_km = hypot(this.x_est - xTrue, this.y_est - yTrue);
  }

  quality() {
    if (this.error_km < Q_HIGH_KM) return "HIGH";
    if (this.error_km < Q_MED_KM) return "MEDIUM";
    return "LOW";
  }

  glitchOffset() {
    const q = this.quality();
    if (q === "HIGH") return [0, 0];
    if (q === "MEDIUM") {
      const amplitude = (this.error_km - Q_HIGH_KM) / (Q_MED_KM - Q_HIGH_KM);
      const jitter = amplitude * 0.4;
      return [gauss(0, jitter), gauss(0, jitter)];
    }
    const excess = Math.min(this.error_km - Q_MED_KM, 3.0);
    const jitter = 0.6 + excess * 0.5;
    return [gauss(0, jitter), gauss(0, jitter)];
  }
}

// ══════════════════════════════════════════════════════════════════════
//  GAME STATE
// ══════════════════════════════════════════════════════════════════════
class GameState {
  constructor() { this.reset(); }

  reset() {
    this.x_true = 0.0;
    this.y_true = 0.0;
    this.heading = Math.PI / 2; // North
    this.speed = CRUISE_SPEED;

    this.drift = new INSDrift();
    this.x_est = 0.0;
    this.y_est = 0.0;
    this.error_km = 0.0;

    this.trail_true = [[0, 0]];
    this.trail_est = [[0, 0]];

    this.nodes = [];
    this.visited = {}; // idx -> {quality, x_est, y_est, error}
    this._genNodes();

    this.buoys_left = N_BUOYS;
    this._buoyPhase = "idle";
    this._buoyTimer = 0.0;
    this.buoy_x = 0.0;
    this.buoy_y = 0.0;
    this.fix_count = 0;
    this._revealRem = 0;
    this.fix_markers = [];

    this.quality_log = []; // [t_real, error_km]
    this.q_warn_flash = 0;

    this.t_start = performance.now() / 1000;
    this.time_left = GAME_DURATION_S;
    this.game_over = false;
    this.final_score = 0;
    this.game_started = false;

    this.inp = { left: false, right: false, deploy: false };
  }

  _genNodes() {
    const margin = 60.0;
    this.nodes = [];
    let tries = 0;
    while (this.nodes.length < N_NODES && tries < 9999) {
      tries++;
      const nx = uniform(CCZ_X[0] + margin, CCZ_X[1] - margin);
      const ny = uniform(CCZ_Y[0] + margin, CCZ_Y[1] - margin);
      const close = this.nodes.some(([ex, ey]) => hypot(nx - ex, ny - ey) < NODE_RADIUS_KM * 3.5);
      if (!close && hypot(nx, ny) > 100) this.nodes.push([nx, ny]);
    }
  }

  score() {
    const vals = Object.values(this.visited);
    if (vals.length === 0) return 0;
    const raw = vals.reduce((s, v) => s + Q_WEIGHTS[v.quality] * 1000, 0);
    const tElapsed = GAME_DURATION_S - this.time_left;
    const spdBonus = Math.max(1.0, 2.5 - tElapsed / 50.0);
    const buoyBonus = 1.0 + (this.fix_count === 0 ? 0.5 : 0) + Math.max(0, N_BUOYS - this.fix_count) * 0.10;
    return Math.floor(raw * spdBonus * buoyBonus);
  }

  qualityBreakdown() {
    const counts = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    Object.values(this.visited).forEach((v) => counts[v.quality]++);
    return counts;
  }

  get buoyDepthFrac() {
    if (this._buoyPhase === "ascending") return Math.max(0.0, 1.0 - this._buoyTimer / BUOY_ASCENT_S);
    if (this._buoyPhase === "surface") return 0.0;
    if (this._buoyPhase === "descending") return Math.min(1.0, this._buoyTimer / BUOY_ASCENT_S);
    return 1.0;
  }

  step(dt) {
    if (this.game_over) return;
    if (!this.game_started) {
      this.t_start = performance.now() / 1000;
      return;
    }
    this.time_left = Math.max(0.0, GAME_DURATION_S - (performance.now() / 1000 - this.t_start));
    if (this.time_left <= 0) { this._end(); return; }

    const i = this.inp;

    // Turning — only control available
    const tr = (TURN_RATE_DEG * Math.PI / 180) * dt;
    if (i.left) this.heading += tr;
    if (i.right) this.heading -= tr;
    this.heading = ((this.heading % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);

    this.speed = CRUISE_SPEED;

    this.x_true += this.speed * Math.cos(this.heading) * dt;
    this.y_true += this.speed * Math.sin(this.heading) * dt;
    this.x_true = clamp(this.x_true, CCZ_X[0], CCZ_X[1]);
    this.y_true = clamp(this.y_true, CCZ_Y[0], CCZ_Y[1]);

    this.drift.step(this.x_true, this.y_true, this.heading, this.speed, dt);
    this.x_est = this.drift.x_est;
    this.y_est = this.drift.y_est;
    this.error_km = this.drift.error_km;

    if (this.drift.quality() === "LOW") this.q_warn_flash = Math.max(this.q_warn_flash, 6);
    if (this.q_warn_flash > 0) this.q_warn_flash--;

    this.trail_true.push([this.x_true, this.y_true]);
    this.trail_est.push([this.x_est, this.y_est]);
    if (this.trail_true.length > 1500) this.trail_true.shift();
    if (this.trail_est.length > 1500) this.trail_est.shift();
    this.quality_log.push([performance.now() / 1000 - this.t_start, this.error_km]);

    if (this._revealRem > 0) this._revealRem--;

    // Node capture — uses ESTIMATED position
    const qNow = this.drift.quality();
    for (let idx = 0; idx < this.nodes.length; idx++) {
      if (idx in this.visited) continue;
      const [nx, ny] = this.nodes[idx];
      if (hypot(this.x_est - nx, this.y_est - ny) < NODE_RADIUS_KM) {
        this.visited[idx] = { quality: qNow, x_est: this.x_est, y_est: this.y_est, error: this.error_km };
        if (Object.keys(this.visited).length === N_NODES) { this._end(); return; }
      }
    }

    // Deploy
    if (i.deploy) {
      i.deploy = false;
      if (this.buoys_left > 0 && this._buoyPhase === "idle") {
        this._buoyPhase = "ascending";
        this._buoyTimer = 0.0;
        this.buoy_x = this.x_true;
        this.buoy_y = this.y_true;
        this.buoys_left--;
      }
    }

    // Buoy state machine
    if (this._buoyPhase === "ascending") {
      this._buoyTimer += dt;
      if (this._buoyTimer >= BUOY_ASCENT_S) {
        this._buoyPhase = "surface";
        this._buoyTimer = 0.0;
        this.drift.reset(this.x_true, this.y_true);
        this.x_est = this.x_true;
        this.y_est = this.y_true;
        this.error_km = 0.0;
        this.fix_count++;
        this._revealRem = 100;
        this.fix_markers.push([this.x_true, this.y_true]);
      }
    } else if (this._buoyPhase === "surface") {
      this._buoyTimer += dt;
      if (this._buoyTimer >= 2.0) { this._buoyPhase = "descending"; this._buoyTimer = 0.0; }
    } else if (this._buoyPhase === "descending") {
      this._buoyTimer += dt;
      if (this._buoyTimer >= BUOY_ASCENT_S) this._buoyPhase = "idle";
    }
  }

  _end() {
    this.game_over = true;
    this.final_score = this.score();
  }
}

/* ══════════════════════════════════════════════════════════════════════
   RENDERING — 2D map canvas
   ══════════════════════════════════════════════════════════════════════ */
const mapCanvas = document.getElementById("map-canvas");
const mapCtx = mapCanvas.getContext("2d");

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

// world (km) -> screen (px), camera centred on (cx, cy), y flipped (screen y grows down)
function makeProjector(rect, cx, cy) {
  const scaleX = rect.width / (VIEW_HW * 2);
  const scaleY = rect.height / (VIEW_HH * 2);
  const scale = Math.min(scaleX, scaleY);
  return (wx, wy) => [
    rect.width / 2 + (wx - cx) * scale,
    rect.height / 2 - (wy - cy) * scale,
    scale,
  ];
}

function drawMap(gs) {
  const rect = resizeCanvas(mapCanvas);
  const ctx = mapCtx;
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, rect.width, rect.height);

  const cx = clamp(gs.x_est, CCZ_X[0] - 30 + VIEW_HW, CCZ_X[1] + 30 - VIEW_HW);
  const cy = clamp(gs.y_est, CCZ_Y[0] - 30 + VIEW_HH, CCZ_Y[1] + 30 - VIEW_HH);
  const proj = makeProjector(rect, gs.x_est, gs.y_est);

  // grid
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  const gridStep = 50;
  for (let gx = Math.floor((gs.x_est - VIEW_HW) / gridStep) * gridStep; gx < gs.x_est + VIEW_HW; gx += gridStep) {
    const [sx] = proj(gx, 0);
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, rect.height); ctx.stroke();
  }
  for (let gy = Math.floor((gs.y_est - VIEW_HH) / gridStep) * gridStep; gy < gs.y_est + VIEW_HH; gy += gridStep) {
    const [, sy] = proj(0, gy);
    ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(rect.width, sy); ctx.stroke();
  }

  // arena boundary
  ctx.strokeStyle = C_BDR;
  ctx.lineWidth = 2;
  const [bx0, by0] = proj(CCZ_X[0], CCZ_Y[0]);
  const [bx1, by1] = proj(CCZ_X[1], CCZ_Y[1]);
  ctx.strokeRect(Math.min(bx0, bx1), Math.min(by0, by1), Math.abs(bx1 - bx0), Math.abs(by1 - by0));

  // true trail (only while a fix reveal is active)
  if (gs._revealRem > 0 && gs.trail_true.length > 1) {
    ctx.strokeStyle = C_TRUE;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    gs.trail_true.forEach((p, idx) => {
      const [sx, sy] = proj(p[0], p[1]);
      idx === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  }

  // estimated trail
  if (gs.trail_est.length > 1) {
    ctx.strokeStyle = C_SUB;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    gs.trail_est.forEach((p, idx) => {
      const [sx, sy] = proj(p[0], p[1]);
      idx === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
    });
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  }

  // fix markers
  ctx.fillStyle = C_BUOY;
  gs.fix_markers.forEach(([fx, fy]) => {
    const [sx, sy] = proj(fx, fy);
    ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill();
  });

  // nodes
  const qual = gs.drift.quality();
  const nodeAlpha = { HIGH: 0.9, MEDIUM: 0.45, LOW: 0.18 }[qual];
  const [gxOff, gyOff] = gs.drift.glitchOffset();
  gs.nodes.forEach(([nx, ny], idx) => {
    let px = nx, py = ny, colour, alpha, radiusAlpha;
    if (idx in gs.visited) {
      colour = Q_COLOURS[gs.visited[idx].quality];
      alpha = 0.9;
    } else {
      colour = C_ND_U;
      alpha = nodeAlpha;
      if (qual !== "HIGH") {
        const [px2, py2] = gs.drift.glitchOffset();
        px = nx + gxOff * 0.6 + px2 * 0.4;
        py = ny + gyOff * 0.6 + py2 * 0.4;
      }
      // faint capture-radius circle
      const [sx, sy] = proj(nx, ny);
      const rPx = NODE_RADIUS_KM * proj(1, 0)[2];
      ctx.fillStyle = Q_COLOURS[qual];
      ctx.globalAlpha = qual === "HIGH" ? 0.12 : qual === "MEDIUM" ? 0.06 : 0.03;
      ctx.beginPath(); ctx.arc(sx, sy, rPx, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1.0;
    }
    const [sx, sy] = proj(px, py);
    ctx.fillStyle = colour;
    ctx.globalAlpha = alpha;
    ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1.0;
  });

  // buoy (surfacing marker)
  if (gs._buoyPhase === "ascending" || gs._buoyPhase === "surface") {
    const [sx, sy] = proj(gs.buoy_x, gs.buoy_y);
    ctx.fillStyle = C_BUOY;
    ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.fill();
  }

  // sub
  const subCol = gs.q_warn_flash > 0 ? C_WARN : C_SUB;
  const [sx, sy] = proj(gs.x_est, gs.y_est);
  const heading = gs.heading;
  const subLenPx = 16;
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(-heading);
  ctx.fillStyle = subCol;
  ctx.beginPath();
  ctx.moveTo(subLenPx, 0);
  ctx.lineTo(-subLenPx * 0.6, subLenPx * 0.55);
  ctx.lineTo(-subLenPx * 0.3, 0);
  ctx.lineTo(-subLenPx * 0.6, -subLenPx * 0.55);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // heading arrow (white, pointing forward)
  if (!gs.game_over) {
    ctx.strokeStyle = "white";
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 2;
    const al = 24;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(sx + Math.cos(heading) * al, sy - Math.sin(heading) * al);
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  }
}

/* ══════════════════════════════════════════════════════════════════════
   RENDERING — depth panel (three.js): buoy ascent/descent + tether
   ══════════════════════════════════════════════════════════════════════ */
const depthCanvas = document.getElementById("depth-canvas");
let three = { renderer: null, scene: null, camera: null, sub: null, buoy: null, tether: null, azim: 0 };

function initThree() {
  const rect = depthCanvas.getBoundingClientRect();
  three.renderer = new THREE.WebGLRenderer({ canvas: depthCanvas, antialias: true, alpha: true });
  three.renderer.setPixelRatio(window.devicePixelRatio || 1);
  three.renderer.setSize(rect.width, rect.height, false);

  three.scene = new THREE.Scene();
  three.scene.fog = new THREE.FogExp2(0x0b0b12, 0.012);

  three.camera = new THREE.PerspectiveCamera(45, rect.width / rect.height, 0.1, 1000);

  // seabed grid
  const grid = new THREE.GridHelper(160, 16, 0x1a6699, 0x161628);
  three.scene.add(grid);

  // ambient + point light
  three.scene.add(new THREE.AmbientLight(0x445566, 1.2));
  const pl = new THREE.PointLight(0xffffff, 0.8);
  pl.position.set(20, 40, 20);
  three.scene.add(pl);

  // sub (cone pointing "up" toward surface axis)
  const subGeo = new THREE.ConeGeometry(2.2, 7, 12);
  const subMat = new THREE.MeshPhongMaterial({ color: 0x3498db });
  three.sub = new THREE.Mesh(subGeo, subMat);
  three.sub.rotation.z = Math.PI / 2;
  three.scene.add(three.sub);

  // buoy
  const buoyGeo = new THREE.SphereGeometry(2.2, 16, 16);
  const buoyMat = new THREE.MeshPhongMaterial({ color: 0xf1c40f });
  three.buoy = new THREE.Mesh(buoyGeo, buoyMat);
  three.buoy.visible = false;
  three.scene.add(three.buoy);

  // tether line
  const tetherGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const tetherMat = new THREE.LineBasicMaterial({ color: 0xf1c40f, transparent: true, opacity: 0.7 });
  three.tether = new THREE.Line(tetherGeo, tetherMat);
  three.tether.visible = false;
  three.scene.add(three.tether);

  // surface plane (faint)
  const surfGeo = new THREE.PlaneGeometry(200, 200);
  const surfMat = new THREE.MeshBasicMaterial({ color: 0x5dade2, transparent: true, opacity: 0.06, side: THREE.DoubleSide });
  const surf = new THREE.Mesh(surfGeo, surfMat);
  surf.rotation.x = Math.PI / 2;
  surf.position.y = 45;
  three.scene.add(surf);
}

function drawDepth(gs) {
  const rect = depthCanvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  if (three.renderer.domElement.width !== rect.width * (window.devicePixelRatio || 1)) {
    three.renderer.setSize(rect.width, rect.height, false);
    three.camera.aspect = rect.width / rect.height;
    three.camera.updateProjectionMatrix();
  }

  three.azim += 0.006;

  // sub sits fixed near seabed (depth 600m -> y = -8)
  three.sub.position.set(0, -8, 0);
  const flashRed = gs.q_warn_flash > 0;
  three.sub.material.color.set(flashRed ? 0xe74c3c : 0x3498db);

  const buoyActive = ["ascending", "surface", "descending"].includes(gs._buoyPhase);
  let buoyY = -8;
  if (buoyActive) {
    const depthFrac = gs.buoyDepthFrac; // 1 = at sub depth, 0 = surface
    buoyY = -8 + (1 - depthFrac) * 53;
    three.buoy.position.set(0, buoyY, 0);
    three.buoy.visible = true;
    three.tether.visible = true;
    const pts = [new THREE.Vector3(0, -8, 0), new THREE.Vector3(0, buoyY, 0)];
    three.tether.geometry.setFromPoints(pts);
  } else {
    three.buoy.visible = false;
    three.tether.visible = false;
  }

  // Frame the camera to include both sub and buoy (when active)
  const midY = buoyActive ? (-8 + buoyY) / 2 : -5;
  const span = buoyActive ? Math.abs(buoyY - -8) : 10;
  const camR = 50 + span * 0.35;
  const camElev = midY + 14;
  three.camera.position.set(
    Math.cos(three.azim) * camR,
    camElev,
    Math.sin(three.azim) * camR
  );
  three.camera.lookAt(0, midY, 0);

  three.renderer.render(three.scene, three.camera);

  const lblMap = {
    ascending: `▲ BUOY ${Math.round(100 * (1 - gs.buoyDepthFrac))}%`,
    surface: "● SURFACE FIX!",
    descending: "▼ DESCENDING",
  };
  const caption = document.getElementById("depth-caption");
  if (lblMap[gs._buoyPhase]) {
    caption.textContent = lblMap[gs._buoyPhase];
    caption.style.color = gs._buoyPhase !== "surface" ? "#f1c40f" : "#5dade2";
  } else if (gs.error_km > Q_MED_KM) {
    caption.textContent = `⚠ DRIFT ${gs.error_km.toFixed(1)}km`;
    caption.style.color = "#e74c3c";
  } else {
    caption.textContent = `Depth 600m | ${gs.speed.toFixed(1)}km/s`;
    caption.style.color = "#d8dcea";
  }
}

/* ══════════════════════════════════════════════════════════════════════
   RENDERING — error-over-time graph
   ══════════════════════════════════════════════════════════════════════ */
const errCanvas = document.getElementById("err-canvas");
const errCtx = errCanvas.getContext("2d");

function drawErrGraph(gs) {
  const rect = resizeCanvas(errCanvas);
  const ctx = errCtx;
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, rect.width, rect.height);

  const log = gs.quality_log;
  if (log.length < 2) return;

  const padL = 34, padB = 18, padT = 10, padR = 10;
  const w = rect.width - padL - padR;
  const h = rect.height - padT - padB;

  const maxT = Math.max(GAME_DURATION_S, log[log.length - 1][0]);
  const maxE = Math.max(8, ...log.map((p) => p[1]) ) * 1.1;

  const X = (t) => padL + (t / maxT) * w;
  const Y = (e) => padT + h - (e / maxE) * h;

  // axes
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + h); ctx.lineTo(padL + w, padT + h); ctx.stroke();

  // threshold lines
  [[Q_HIGH_KM, "#2ecc71"], [Q_MED_KM, "#f39c12"]].forEach(([val, col]) => {
    if (val > maxE) return;
    ctx.strokeStyle = col;
    ctx.globalAlpha = 0.35;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(padL, Y(val)); ctx.lineTo(padL + w, Y(val)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1.0;
  });

  // fix vertical lines
  ctx.strokeStyle = C_BUOY;
  ctx.globalAlpha = 0.6;
  gs.fix_markers.forEach((_, idx) => {
    const frac = (idx + 1) / Math.max(gs.fix_count, 1);
    const pt = log[Math.min(log.length - 1, Math.floor(log.length * frac) - 1)];
    if (!pt) return;
    ctx.beginPath(); ctx.moveTo(X(pt[0]), padT); ctx.lineTo(X(pt[0]), padT + h); ctx.stroke();
  });
  ctx.globalAlpha = 1.0;

  // error line
  ctx.strokeStyle = C_WARN;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  log.forEach((p, idx) => {
    const x = X(p[0]), y = Y(Math.min(p[1], maxE));
    idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // labels
  ctx.fillStyle = "#7a84a0";
  ctx.font = "10px Consolas, monospace";
  ctx.fillText(`${maxE.toFixed(0)}km`, 2, padT + 8);
  ctx.fillText("0", 2, padT + h);
  ctx.fillText(`${maxT.toFixed(0)}s`, padL + w - 24, padT + h + 14);
}

/* ══════════════════════════════════════════════════════════════════════
   HUD / PARAM TEXT UPDATES
   ══════════════════════════════════════════════════════════════════════ */
function updateHud(gs) {
  const qual = gs.drift.quality();
  const qCol = Q_COLOURS[qual];

  document.getElementById("hud-time").textContent = `⏱ ${gs.time_left.toFixed(0)}s`;
  document.getElementById("hud-time").style.color = gs.time_left < 20 ? C_WARN : gs.time_left < 45 ? "#f39c12" : "#d8dcea";

  document.getElementById("hud-score").textContent = `⭐ ${gs.score()}`;
  document.getElementById("hud-buoys").textContent = "BUOYS " + "📡".repeat(gs.buoys_left) + "○".repeat(N_BUOYS - gs.buoys_left);
  document.getElementById("hud-nodes").textContent = `NODES ${Object.keys(gs.visited).length}/${N_NODES}`;

  const warn = gs.error_km > NODE_RADIUS_KM ? "  ⚠ DEPLOY BUOY" : "";
  const hq = document.getElementById("hud-quality");
  hq.textContent = `NAV QUALITY: ${qual} (${gs.error_km.toFixed(1)}km)${warn}`;
  hq.style.color = qCol;

  document.getElementById("p-pos").textContent = `${gs.x_est >= 0 ? "+" : ""}${gs.x_est.toFixed(1)}, ${gs.y_est >= 0 ? "+" : ""}${gs.y_est.toFixed(1)} km`;
  document.getElementById("p-spd").textContent = `${gs.speed.toFixed(2)} km/s`;
  const hdgDeg = (((gs.heading * 180 / Math.PI) % 360) + 360) % 360;
  document.getElementById("p-hdg").textContent = `${hdgDeg.toFixed(0)}°`;
  const pErr = document.getElementById("p-err");
  pErr.textContent = `${gs.error_km.toFixed(1)} km`; pErr.style.color = qCol;
  const pQ = document.getElementById("p-q");
  pQ.textContent = qual; pQ.style.color = qCol;

  const bstLabel = { idle: "IDLE", ascending: "ASCENDING ▲", surface: "● FIX APPLIED", descending: "DESCENDING ▼" };
  const bstColour = { idle: "#2ecc71", ascending: "#f1c40f", surface: "#5dade2", descending: "#aaaaaa" };
  const pBst = document.getElementById("p-bst");
  pBst.textContent = bstLabel[gs._buoyPhase]; pBst.style.color = bstColour[gs._buoyPhase];

  document.getElementById("p-fix").textContent = `${gs.fix_count} / ${N_BUOYS}`;
}

/* ══════════════════════════════════════════════════════════════════════
   INPUT — keyboard + gamepad
   ══════════════════════════════════════════════════════════════════════ */
let gs = new GameState();

function bindKeyboard() {
  window.addEventListener("keydown", (e) => {
    if (["ArrowLeft", "ArrowRight", " ", "Spacebar"].includes(e.key)) e.preventDefault();
    const k = e.key.toLowerCase();
    if (k === "a" || e.key === "ArrowLeft") gs.inp.left = true;
    if (k === "d" || e.key === "ArrowRight") gs.inp.right = true;
    if (e.key === " " || e.key === "Spacebar") gs.inp.deploy = true;
    if (k === "r") restartGame();
    if (k === "i") toggleInstructions();
  });
  window.addEventListener("keyup", (e) => {
    const k = e.key.toLowerCase();
    if (k === "a" || e.key === "ArrowLeft") gs.inp.left = false;
    if (k === "d" || e.key === "ArrowRight") gs.inp.right = false;
  });
}

let gamepadDeployLatch = false;
let gamepadRestartLatch = false;
function pollGamepad() {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  const pad = pads && pads[0];
  if (!pad) return;

  const axisX = pad.axes[0] || 0;
  const DEAD = 0.2;
  gs.inp.left = axisX < -DEAD;
  gs.inp.right = axisX > DEAD;

  const triangle = pad.buttons[3] && pad.buttons[3].pressed; // Y/Triangle
  if (triangle && !gamepadDeployLatch) gs.inp.deploy = true;
  gamepadDeployLatch = triangle;

  const l1 = pad.buttons[4] && pad.buttons[4].pressed;
  const l2 = pad.buttons[6] && pad.buttons[6].pressed;
  const restartPressed = l1 || l2;
  if (restartPressed && !gamepadRestartLatch) restartGame();
  gamepadRestartLatch = restartPressed;
}

/* ══════════════════════════════════════════════════════════════════════
   GAME LIFECYCLE
   ══════════════════════════════════════════════════════════════════════ */
function startGame() {
  document.getElementById("start-overlay").classList.add("hidden");
  document.getElementById("game-over-overlay").classList.add("hidden");
  gs.game_started = true;
  gs.t_start = performance.now() / 1000;
}

function restartGame() {
  gs = new GameState();
  document.getElementById("game-over-overlay").classList.add("hidden");
  document.getElementById("start-overlay").classList.remove("hidden");
}

function toggleInstructions() {
  document.getElementById("instructions-overlay").classList.toggle("hidden");
}

function showGameOver(g) {
  const qb = g.qualityBreakdown();
  document.getElementById("go-score").textContent = `Score: ${g.final_score}`;
  document.getElementById("go-breakdown").textContent = `H=${qb.HIGH}  M=${qb.MEDIUM}  L=${qb.LOW} / ${N_NODES} nodes`;
  document.getElementById("game-over-overlay").classList.remove("hidden");
}

document.getElementById("btn-start").addEventListener("click", startGame);
document.getElementById("btn-restart").addEventListener("click", restartGame);
document.getElementById("btn-instructions").addEventListener("click", toggleInstructions);
document.getElementById("btn-close-instructions").addEventListener("click", toggleInstructions);

bindKeyboard();
initThree();

/* ══════════════════════════════════════════════════════════════════════
   MAIN LOOP
   ══════════════════════════════════════════════════════════════════════ */
let lastT = performance.now();
let goShown = false;

function tick(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;

  pollGamepad();
  gs.step(dt);

  drawMap(gs);
  drawDepth(gs);
  drawErrGraph(gs);
  updateHud(gs);

  if (gs.game_over && !goShown) { showGameOver(gs); goShown = true; }
  if (!gs.game_over) goShown = false;

  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

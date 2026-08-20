// ============================================================
// SNC SIMULATOR — THE ORIEL
// Browser Game
// ============================================================

const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");


// ============================================================
// GAME PARAMETERS
// ============================================================

const GAME_DURATION = 90;

const CRUISE_SPEED = 50;

const TURN_RATE = 85 * Math.PI / 180;

const N_NODES = 14;

const NODE_RADIUS = 8;

const N_BUOYS = 3;


// ============================================================
// GAME STATE
// ============================================================

const game = {

    x: 0,
    y: 0,

    heading: Math.PI / 2,

    speed: CRUISE_SPEED,

    xEstimate: 0,
    yEstimate: 0,

    navError: 0,

    timeLeft: GAME_DURATION,

    score: 0,

    quality: "HIGH",

    buoys: N_BUOYS,

    nodes: [],

    collected: new Set(),

    running: false,

    lastTime: 0

};


// ============================================================
// INPUT
// ============================================================

const input = {

    left: false,
    right: false

};


document.addEventListener("keydown", event => {

    if (event.key === "a" || event.key === "ArrowLeft") {
        input.left = true;
    }

    if (event.key === "d" || event.key === "ArrowRight") {
        input.right = true;
    }

});


document.addEventListener("keyup", event => {

    if (event.key === "a" || event.key === "ArrowLeft") {
        input.left = false;
    }

    if (event.key === "d" || event.key === "ArrowRight") {
        input.right = false;
    }

});


// ============================================================
// CANVAS RESOLUTION
// ============================================================

function resizeCanvas() {

    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width;
    canvas.height = rect.height;

}

window.addEventListener("resize", resizeCanvas);

resizeCanvas();


// ============================================================
// GENERATE SURVEY NODES
// ============================================================

function generateNodes() {

    game.nodes = [];

    while (game.nodes.length < N_NODES) {

        const x = Math.random() * 1200 - 600;

        const y = Math.random() * 500 - 250;

        // Keep nodes reasonably separated

        const tooClose = game.nodes.some(node => {

            const dx = node.x - x;
            const dy = node.y - y;

            return Math.hypot(dx, dy) < NODE_RADIUS * 4;

        });

        if (!tooClose) {

            game.nodes.push({
                x,
                y
            });

        }

    }

}


// ============================================================
// NAVIGATION QUALITY
// ============================================================

function updateQuality() {

    if (game.navError < 2.5) {

        game.quality = "HIGH";

    } else if (game.navError < 4) {

        game.quality = "MEDIUM";

    } else {

        game.quality = "LOW";

    }

}


// ============================================================
// START GAME
// ============================================================

function startGame() {

    game.x = 0;
    game.y = 0;

    game.xEstimate = 0;
    game.yEstimate = 0;

    game.heading = Math.PI / 2;

    game.navError = 0;

    game.timeLeft = GAME_DURATION;

    game.score = 0;

    game.buoys = N_BUOYS;

    game.collected.clear();

    generateNodes();

    game.running = true;

    game.lastTime = performance.now();

    requestAnimationFrame(gameLoop);

}


// ============================================================
// UPDATE GAME
// ============================================================

function update(dt) {

    // --------------------------------------------------------
    // Turning
    // --------------------------------------------------------

    if (input.left) {

        game.heading += TURN_RATE * dt;

    }

    if (input.right) {

        game.heading -= TURN_RATE * dt;

    }


    // --------------------------------------------------------
    // True submarine position
    // --------------------------------------------------------

    game.x +=
        game.speed *
        Math.cos(game.heading) *
        dt;

    game.y +=
        game.speed *
        Math.sin(game.heading) *
        dt;


    // --------------------------------------------------------
    // VERY SIMPLE FIRST-PASS DRIFT MODEL
    //
    // We will replace this with your actual INS model later.
    // --------------------------------------------------------

    game.navError += 0.045 * dt;

    game.xEstimate =
        game.x +
        Math.sin(game.x / 100) * game.navError;

    game.yEstimate =
        game.y +
        Math.cos(game.y / 100) * game.navError;


    updateQuality();


    // --------------------------------------------------------
    // Survey node collection
    // --------------------------------------------------------

    game.nodes.forEach((node, index) => {

        if (game.collected.has(index)) {
            return;
        }

        const dx = game.xEstimate - node.x;
        const dy = game.yEstimate - node.y;

        const distance = Math.hypot(dx, dy);

        if (distance < NODE_RADIUS) {

            game.collected.add(index);

            const multiplier = {

                HIGH: 3.0,
                MEDIUM: 1.5,
                LOW: 0.5

            }[game.quality];

            game.score += Math.round(1000 * multiplier);

        }

    });


    // --------------------------------------------------------
    // Timer
    // --------------------------------------------------------

    game.timeLeft -= dt;

    if (game.timeLeft <= 0) {

        game.timeLeft = 0;

        game.running = false;

    }


    // --------------------------------------------------------
    // HUD
    // --------------------------------------------------------

    document.getElementById("time").textContent =
        Math.ceil(game.timeLeft);

    document.getElementById("score").textContent =
        game.score;

    const qualityElement =
        document.getElementById("quality");

    qualityElement.textContent =
        game.quality;

}


// ============================================================
// WORLD → SCREEN
// ============================================================

function worldToScreen(x, y) {

    const scale =
        Math.min(canvas.width / 700, canvas.height / 500);

    const screenX =
        canvas.width / 2 +
        (x - game.xEstimate) * scale;

    const screenY =
        canvas.height / 2 -
        (y - game.yEstimate) * scale;

    return {
        x: screenX,
        y: screenY
    };

}


// ============================================================
// DRAW BACKGROUND
// ============================================================

function drawBackground() {

    ctx.fillStyle = "#050912";

    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    // Grid

    ctx.strokeStyle = "#101a2a";

    ctx.lineWidth = 1;

    const gridSize = 50;

    for (
        let x = 0;
        x < canvas.width;
        x += gridSize
    ) {

        ctx.beginPath();

        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);

        ctx.stroke();

    }

    for (
        let y = 0;
        y < canvas.height;
        y += gridSize
    ) {

        ctx.beginPath();

        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);

        ctx.stroke();

    }

}


// ============================================================
// DRAW NODES
// ============================================================

function drawNodes() {

    game.nodes.forEach((node, index) => {

        if (game.collected.has(index)) {
            return;
        }

        const position =
            worldToScreen(node.x, node.y);


        let colour;

        if (game.quality === "HIGH") {

            colour = "#2ecc71";

        } else if (game.quality === "MEDIUM") {

            colour = "#f39c12";

        } else {

            colour = "#e74c3c";

        }


        ctx.fillStyle = colour;

        ctx.globalAlpha =
            game.quality === "LOW"
                ? 0.35
                : game.quality === "MEDIUM"
                    ? 0.65
                    : 1;


        ctx.beginPath();

        ctx.moveTo(
            position.x,
            position.y - 8
        );

        ctx.lineTo(
            position.x + 8,
            position.y
        );

        ctx.lineTo(
            position.x,
            position.y + 8
        );

        ctx.lineTo(
            position.x - 8,
            position.y
        );

        ctx.closePath();

        ctx.fill();

        ctx.globalAlpha = 1;

    });

}


// ============================================================
// DRAW SUBMARINE
// ============================================================

function drawSubmarine() {

    const x = canvas.width / 2;
    const y = canvas.height / 2;

    ctx.save();

    ctx.translate(x, y);

    ctx.rotate(-game.heading + Math.PI / 2);

    ctx.fillStyle = "#3498db";

    ctx.strokeStyle = "#ffffff";

    ctx.lineWidth = 1;

    ctx.beginPath();

    ctx.moveTo(0, -20);

    ctx.lineTo(10, 12);

    ctx.lineTo(0, 18);

    ctx.lineTo(-10, 12);

    ctx.closePath();

    ctx.fill();

    ctx.stroke();

    ctx.restore();

}


// ============================================================
// DRAW NAVIGATION ERROR
// ============================================================

function drawNavigationInfo() {

    ctx.fillStyle = "#d8dcea";

    ctx.font = "14px monospace";

    ctx.fillText(
        `NAV ERROR: ${game.navError.toFixed(1)} km`,
        20,
        30
    );

    ctx.fillText(
        `NODES: ${game.collected.size}/${N_NODES}`,
        20,
        52
    );

    ctx.fillText(
        `BUOYS: ${"●".repeat(game.buoys)}${"○".repeat(N_BUOYS - game.buoys)}`,
        20,
        74
    );

}


// ============================================================
// DRAW EVERYTHING
// ============================================================

function draw() {

    drawBackground();

    drawNodes();

    drawSubmarine();

    drawNavigationInfo();

}


// ============================================================
// MAIN GAME LOOP
// ============================================================

function gameLoop(timestamp) {

    if (!game.running) {

        draw();

        return;

    }


    const dt =
        Math.min(
            (timestamp - game.lastTime) / 1000,
            0.1
        );

    game.lastTime = timestamp;

    update(dt);

    draw();

    requestAnimationFrame(gameLoop);

}


// ============================================================
// START
// ============================================================

startGame();

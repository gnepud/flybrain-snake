/**
 * FlyBrain Snake — Client Controller & Telemetry Stream Manager
 * Connects to SSE /api/stream, synchronizes simulation state with UI,
 * dispatches control commands via POST /api/control, and drives 4-canvas rendering.
 */

(() => {
  // DOM Elements
  const snakeCanvas = document.getElementById('snakeCanvas');
  const brain3dCanvas = document.getElementById('brain3dCanvas');
  const compassCanvas = document.getElementById('compassCanvas');
  const steeringCanvas = document.getElementById('steeringCanvas');
  const motorCanvas = document.getElementById('motorCanvas');

  const brainViewFull = document.getElementById('brainViewFull');
  const brainViewBrain = document.getElementById('brainViewBrain');
  const brainViewVNC = document.getElementById('brainViewVNC');
  const brainViewSide = document.getElementById('brainViewSide');
  const brainViewDorsal = document.getElementById('brainViewDorsal');
  const brainRotateBtn = document.getElementById('brainRotateBtn');

  const statusBadge = document.getElementById('statusBadge');
  const scoreValue = document.getElementById('scoreValue');
  const stepValue = document.getElementById('stepValue');
  const timeValue = document.getElementById('timeValue');

  const playBtn = document.getElementById('playBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  const stepBtn = document.getElementById('stepBtn');
  const resetBtn = document.getElementById('resetBtn');
  const speedSlider = document.getElementById('speedSlider');
  const speedValue = document.getElementById('speedValue');

  const snakeStatusText = document.getElementById('snakeStatusText');
  const compassStatusText = document.getElementById('compassStatusText');
  const steeringStatusText = document.getElementById('steeringStatusText');
  const motorStatusText = document.getElementById('motorStatusText');

  // Client State
  let eventSource = null;
  let isPaused = false;
  let startTime = Date.now();
  let latestFrame = null;
  let isUsingSSE = false;
  let sseFallbackTimer = null;

  // --- Client-Side Autonomous Simulation Engine (Vercel & Offline Fallback) ---
  class ClientSimulation {
    constructor(gridSize = 16) {
      this.gridSize = gridSize;
      this.fps = 10;
      this.intervalId = null;
      this.head = [8, 8];
      this.body = [[8, 8], [8, 9], [8, 10]];
      this.direction = 0; // 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
      this.food = [8, 4];
      this.score = 0;
      this.step = 0;
      this.done = false;
      this.reason = null;
      this.lastOdorConc = null;
      this.active = false;
    }

    spawnFood() {
      const occupied = new Set(this.body.map(([x, y]) => `${x},${y}`));
      const free = [];
      for (let x = 0; x < this.gridSize; x++) {
        for (let y = 0; y < this.gridSize; y++) {
          if (!occupied.has(`${x},${y}`)) free.push([x, y]);
        }
      }
      if (free.length === 0) return [0, 0];
      return free[Math.floor(Math.random() * free.length)];
    }

    resetGame() {
      const mid = Math.floor(this.gridSize / 2);
      this.head = [mid, mid];
      this.body = [[mid, mid], [mid, mid + 1], [mid, mid + 2]];
      this.direction = 0; // UP
      this.score = 0;
      this.step = 0;
      this.done = false;
      this.reason = null;
      this.lastOdorConc = null;
      this.food = this.spawnFood();
      const frame = this.buildCurrentFrame(false, 'NONE', { pfl3_l: 0, pfl3_r: 0, threshold: 0.03, is_override: false }, { dna02_l: 0, dna02_r: 0, dna01: 0.85, dnp: 0.85, is_override: false });
      onNewFrame(frame);
      return frame;
    }

    start() {
      this.active = true;
      if (this.intervalId) clearInterval(this.intervalId);
      this.intervalId = setInterval(() => {
        if (!isPaused) {
          const frame = this.stepSimulation();
          onNewFrame(frame);
        }
      }, 1000 / this.fps);
    }

    stop() {
      this.active = false;
      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
    }

    pause() {
      // paused state handled via isPaused
    }

    play() {
      if (!this.active) {
        this.start();
      }
    }

    stepOnce() {
      const frame = this.stepSimulation();
      onNewFrame(frame);
    }

    setFps(fps) {
      this.fps = Math.max(1, Math.min(60, fps));
      if (this.active) {
        this.start();
      }
    }

    raycast(x, y, dir) {
      const vectors = [[0, -1], [1, 0], [0, 1], [-1, 0]];
      const [vx, vy] = vectors[dir];
      let dist = 0;
      let cx = x + vx;
      let cy = y + vy;
      const occupied = new Set(this.body.map(([bx, by]) => `${bx},${by}`));
      while (cx >= 0 && cx < this.gridSize && cy >= 0 && cy < this.gridSize) {
        if (occupied.has(`${cx},${cy}`)) break;
        dist++;
        cx += vx;
        cy += vy;
      }
      return dist;
    }

    stepSimulation() {
      if (this.done) {
        return this.resetGame();
      }

      this.step++;
      const [hx, hy] = this.head;
      const [fx, fy] = this.food;
      const headingAngles = [-Math.PI / 2, 0.0, Math.PI / 2, Math.PI];
      const headingAngle = headingAngles[this.direction];

      // 1. Raycast Obstacle Distances (Front, Left, Right)
      const turnLeftDir = (this.direction + 3) % 4;
      const turnRightDir = (this.direction + 1) % 4;

      const distFront = this.raycast(hx, hy, this.direction);
      const distLeft = this.raycast(hx, hy, turnLeftDir);
      const distRight = this.raycast(hx, hy, turnRightDir);

      // 2. Olfactory Antennal Field & Temporal Trend
      const antDist = 0.5;
      const leftAntX = hx + Math.cos(headingAngle - Math.PI / 2) * antDist;
      const leftAntY = hy + Math.sin(headingAngle - Math.PI / 2) * antDist;
      const rightAntX = hx + Math.cos(headingAngle + Math.PI / 2) * antDist;
      const rightAntY = hy + Math.sin(headingAngle + Math.PI / 2) * antDist;

      const dL = Math.hypot(fx - leftAntX, fy - leftAntY);
      const dR = Math.hypot(fx - rightAntX, fy - rightAntY);
      const cLeft = Math.exp(-dL / 6.0);
      const cRight = Math.exp(-dR / 6.0);
      const cDiff = cLeft - cRight;
      const avgC = (cLeft + cRight) / 2;
      const deltaC = this.lastOdorConc !== null ? avgC - this.lastOdorConc : 0.0;
      this.lastOdorConc = avgC;
      const gatingFactor = 1.0 + 1.5 * Math.max(cLeft, cRight);

      // 3. Egocentric Bearing & Steering Comparator (PFL3)
      const targetAngle = Math.atan2(fy - hy, fx - hx);
      let bearing = targetAngle - headingAngle;
      while (bearing > Math.PI) bearing -= 2 * Math.PI;
      while (bearing < -Math.PI) bearing += 2 * Math.PI;

      // Biological steering: bearing modulated by olfactory gating
      const turnSignal = bearing * gatingFactor + cDiff * 0.8;
      const pfl3L = Math.max(0, -turnSignal);
      const pfl3R = Math.max(0, turnSignal);

      // 4. Descending Motor Drive (DNa02)
      const dnaDiff = pfl3L - pfl3R;
      let dna02L = Math.max(0, dnaDiff);
      let dna02R = Math.max(0, -dnaDiff);
      const forwardVal = 0.85;

      // Upper connectome intent
      let action = 'STRAIGHT';
      let chosenDir = this.direction;
      if (dnaDiff > 0.03) {
        action = 'TURN_LEFT';
        chosenDir = turnLeftDir;
      } else if (dnaDiff < -0.03) {
        action = 'TURN_RIGHT';
        chosenDir = turnRightDir;
      }

      // 5. VNC Clearance Reflex Arc (Obstacle Avoidance FSM)
      let isOverride = false;
      if (distFront === 0) {
        isOverride = true;
        if (distLeft > distRight && distLeft > 0) {
          action = 'TURN_LEFT';
          chosenDir = turnLeftDir;
          dna02L = 1.2;
          dna02R = 0.0;
        } else if (distRight > 0) {
          action = 'TURN_RIGHT';
          chosenDir = turnRightDir;
          dna02L = 0.0;
          dna02R = 1.2;
        }
      } else {
        if (action === 'TURN_LEFT' && distLeft === 0) {
          action = 'STRAIGHT';
          chosenDir = this.direction;
          isOverride = true;
        } else if (action === 'TURN_RIGHT' && distRight === 0) {
          action = 'STRAIGHT';
          chosenDir = this.direction;
          isOverride = true;
        }
      }

      // 6. Kinematic Update
      this.direction = chosenDir;
      const vectors = [[0, -1], [1, 0], [0, 1], [-1, 0]];
      const [vx, vy] = vectors[this.direction];
      const newHead = [hx + vx, hy + vy];

      const steering = { pfl3_l: pfl3L, pfl3_r: pfl3R, threshold: 0.03, is_override: isOverride };
      const motor = { dna02_l: dna02L, dna02_r: dna02R, dna01: forwardVal, dnp: forwardVal, is_override: isOverride };

      // Wall Collision
      if (newHead[0] < 0 || newHead[0] >= this.gridSize || newHead[1] < 0 || newHead[1] >= this.gridSize) {
        this.done = true;
        this.reason = 'Wall Collision';
        return this.buildCurrentFrame(false, action, steering, motor);
      }

      // Self Body Collision (exclude tail which vacates)
      const bodySet = new Set(this.body.slice(0, -1).map(([x, y]) => `${x},${y}`));
      if (bodySet.has(`${newHead[0]},${newHead[1]}`)) {
        this.done = true;
        this.reason = 'Self Collision';
        return this.buildCurrentFrame(false, action, steering, motor);
      }

      // Food Eating
      let ateFood = false;
      if (newHead[0] === fx && newHead[1] === fy) {
        ateFood = true;
        this.score++;
        this.head = newHead;
        this.body.unshift(newHead);
        this.food = this.spawnFood();
      } else {
        this.head = newHead;
        this.body.unshift(newHead);
        this.body.pop();
      }

      // 7. E-PG 16-wedge bump
      const epg = [];
      const newHeadingAngle = headingAngles[this.direction];
      for (let i = 0; i < 16; i++) {
        const wedgeAngle = (i / 16) * 2 * Math.PI - Math.PI;
        let diffA = Math.abs(wedgeAngle - newHeadingAngle);
        if (diffA > Math.PI) diffA = 2 * Math.PI - diffA;
        epg.push(Math.exp(-(diffA * diffA) / (2 * 0.45 * 0.45)));
      }

      return this.buildCurrentFrame(
        ateFood,
        action,
        steering,
        motor,
        epg,
        { food_bearing: bearing, dist_front: distFront, dist_left: distLeft, dist_right: distRight },
        { c_left: parseFloat(cLeft.toFixed(3)), c_right: parseFloat(cRight.toFixed(3)), diff: parseFloat(cDiff.toFixed(3)), delta_c: parseFloat(deltaC.toFixed(4)), gating_factor: parseFloat(gatingFactor.toFixed(2)) }
      );
    }

    buildCurrentFrame(ateFood, action, steering, motor, epg, sensory, olfactory) {
      const headingAngles = [-Math.PI / 2, 0.0, Math.PI / 2, Math.PI];
      if (!epg) {
        epg = [];
        const headingAngle = headingAngles[this.direction];
        for (let i = 0; i < 16; i++) {
          const wedgeAngle = (i / 16) * 2 * Math.PI - Math.PI;
          let diffA = Math.abs(wedgeAngle - headingAngle);
          if (diffA > Math.PI) diffA = 2 * Math.PI - diffA;
          epg.push(Math.exp(-(diffA * diffA) / (2 * 0.45 * 0.45)));
        }
      }
      return {
        step: this.step,
        score: this.score,
        head: [...this.head],
        body: this.body.map(([x, y]) => [x, y]),
        food: [...this.food],
        direction: this.direction,
        done: this.done,
        reason: this.reason,
        epg: epg,
        steering: steering || { pfl3_l: 0, pfl3_r: 0, threshold: 0.03, is_override: false },
        motor: motor || { dna02_l: 0, dna02_r: 0, dna01: 0.85, dnp: 0.85, is_override: false },
        action: action || 'STRAIGHT',
        restrict_borders: false,
        sensory: sensory || { food_bearing: 0, dist_front: 5, dist_left: 5, dist_right: 5 },
        ate_food: !!ateFood,
        spikes: [],
        spikes_count: 0,
        olfactory: olfactory || { c_left: 0.5, c_right: 0.5, diff: 0.0, delta_c: 0.0, gating_factor: 1.0 }
      };
    }
  }

  const clientSim = new ClientSimulation(16);

  // Formatting helper
  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  // Send Control Command via POST /api/control (or local client simulation)
  async function sendCommand(command, value = null) {
    if (!isUsingSSE) {
      if (command === 'pause') {
        isPaused = true;
        clientSim.pause();
        updateStatusBadge('PAUSED');
      } else if (command === 'play') {
        isPaused = false;
        clientSim.play();
        updateStatusBadge('RUNNING');
      } else if (command === 'step') {
        isPaused = true;
        clientSim.stepOnce();
        updateStatusBadge('PAUSED');
      } else if (command === 'reset') {
        startTime = Date.now();
        isPaused = false;
        clientSim.resetGame();
        updateStatusBadge('RUNNING');
      } else if (command === 'set_speed') {
        const val = parseInt(value, 10) || 10;
        clientSim.setFps(val);
      }
    }

    try {
      const payload = { command };
      if (typeof value === 'object' && value !== null) {
        Object.assign(payload, value);
      } else if (value !== null) {
        payload.value = value;
      }

      const resp = await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        return null;
      }

      const result = await resp.json();
      if (isUsingSSE) {
        if (command === 'pause') {
          isPaused = true;
          updateStatusBadge('PAUSED');
        } else if (command === 'play') {
          isPaused = false;
          updateStatusBadge('RUNNING');
        } else if (command === 'reset') {
          startTime = Date.now();
        }

        if (result.frame) {
          onNewFrame(result.frame);
        }
      }
      return result;
    } catch (err) {
      return null;
    }
  }

  function updateStatusBadge(status) {
    if (!statusBadge) return;
    const badges = { 'RUNNING': 'badge-running', 'PAUSED': 'badge-paused', 'GAME OVER': 'badge-over' };
    statusBadge.className = `badge ${badges[status] || 'badge-action'}`;
    statusBadge.textContent = status;
  }

  // Handle Incoming Telemetry Frame
  function onNewFrame(frame) {
    latestFrame = frame;

    // Update Header Metrics
    if (scoreValue) scoreValue.textContent = frame.score !== undefined ? frame.score : 0;
    if (stepValue) stepValue.textContent = frame.step !== undefined ? frame.step : 0;
    if (timeValue) {
      const elapsed = (Date.now() - startTime) / 1000;
      timeValue.textContent = formatTime(elapsed);
    }

    if (frame.done) {
      isPaused = true;
      updateStatusBadge('GAME OVER');
    } else if (isPaused) {
      updateStatusBadge('PAUSED');
    } else {
      updateStatusBadge('RUNNING');
    }

    // Update Footers (Instrument Monospace Telemetry)
    if (snakeStatusText) {
      let statusStr = frame.done
        ? `CYCLE HALTED: ${frame.reason || 'BOUNDARY COLLISION'}`
        : `HEAD: (${frame.head ? frame.head.join(',') : '?'}) | TGT: (${frame.food ? frame.food.join(',') : '?'})`;
      if (frame.olfactory) {
        const trend = frame.olfactory.delta_c > 0 ? '+ CLOSER' : (frame.olfactory.delta_c < 0 ? '- FURTHER' : 'FLAT');
        const gateStr = frame.olfactory.gating_factor ? ` | GATE: ${frame.olfactory.gating_factor}x` : '';
        statusStr += ` | AL SENSING: L=${frame.olfactory.c_left} R=${frame.olfactory.c_right} (Δ=${frame.olfactory.diff}, ${trend})${gateStr}`;
      }
      snakeStatusText.textContent = statusStr;
    }

    if (compassStatusText && frame.epg) {
      const maxAct = Math.max(...frame.epg);
      compassStatusText.textContent = `BUMP PEAK: ${maxAct.toFixed(3)} | 16 E-PG GLOMERULI RESOLVING HEADING VECTOR`;
    }

    if (steeringStatusText && frame.steering) {
      const diff = (frame.steering.pfl3_l || 0) - (frame.steering.pfl3_r || 0);
      steeringStatusText.textContent = `Δ NULL: ${diff >= 0 ? '+' : ''}${diff.toFixed(3)} V | CH-L: ${(frame.steering.pfl3_l || 0).toFixed(3)} V | CH-R: ${(frame.steering.pfl3_r || 0).toFixed(3)} V`;
    }

    if (motorStatusText && frame.motor) {
      const forwardVal = (frame.motor.dna01 !== undefined ? frame.motor.dna01 : (frame.motor.dnp || 0));
      motorStatusText.textContent = `DNa02_L: ${(frame.motor.dna02_l || 0).toFixed(3)} V | DNa01: ${forwardVal.toFixed(3)} V | DNa02_R: ${(frame.motor.dna02_r || 0).toFixed(3)} V`;
    }

    // Dispatch to 2D Canvas Renderers (3D Brain rendered via requestAnimationFrame)
    if (window.renderSnake && snakeCanvas) {
      window.renderSnake(snakeCanvas, frame);
    }
    if (window.renderCompass && compassCanvas) {
      window.renderCompass(compassCanvas, frame);
    }
    if (window.renderSteering && steeringCanvas) {
      window.renderSteering(steeringCanvas, frame);
    }
    if (window.renderMotor && motorCanvas) {
      window.renderMotor(motorCanvas, frame);
    }
  }

  // Autonomous fallback activation
  function startClientFallback() {
    if (isUsingSSE) return;
    if (!clientSim.active) {
      console.log('Starting autonomous browser simulation engine (Vercel / Standalone mode)...');
      clientSim.start();
      updateStatusBadge(isPaused ? 'PAUSED' : 'RUNNING');
    }
  }

  // Initialize SSE Connection
  function connectStream() {
    if (eventSource) {
      eventSource.close();
    }

    if (sseFallbackTimer) clearTimeout(sseFallbackTimer);
    sseFallbackTimer = setTimeout(() => {
      if (!isUsingSSE) {
        startClientFallback();
      }
    }, 1200);

    try {
      eventSource = new EventSource('/api/stream');

      eventSource.onopen = () => {
        console.log('Connected to FlyBrain Snake SSE telemetry stream.');
      };

      eventSource.onmessage = (e) => {
        try {
          const frame = JSON.parse(e.data);
          if (frame && frame.head) {
            if (!isUsingSSE) {
              isUsingSSE = true;
              clientSim.stop();
              console.log('Live backend connectome telemetry stream active.');
            }
            onNewFrame(frame);
          } else if (frame && frame.serverless) {
            startClientFallback();
          }
        } catch (err) {
          console.error('Failed to parse SSE telemetry frame:', err);
        }
      };

      eventSource.onerror = (e) => {
        if (!isUsingSSE) {
          startClientFallback();
        } else {
          console.warn('SSE stream disconnected, falling back to client simulation in 2s...', e);
          setTimeout(() => {
            if (!eventSource || eventSource.readyState === EventSource.CLOSED) {
              isUsingSSE = false;
              startClientFallback();
            }
          }, 2000);
        }
      };
    } catch (err) {
      console.warn('EventSource initialization failed:', err);
      startClientFallback();
    }
  }

  // Check initial server status
  async function checkInitialStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        isPaused = !!data.paused;
        if (speedSlider) speedSlider.value = data.fps || 10;
        if (speedValue) speedValue.textContent = data.fps || 10;
        if (data.mode === 'vercel-serverless') {
          startClientFallback();
        } else if (data.done) {
          updateStatusBadge('GAME OVER');
        } else {
          updateStatusBadge(isPaused ? 'PAUSED' : 'RUNNING');
        }
      } else {
        startClientFallback();
      }
    } catch (e) {
      console.warn('Status check deferred, starting client simulation:', e);
      startClientFallback();
    }
  }

  // Event Listeners for UI Controls
  if (playBtn) {
    playBtn.addEventListener('click', () => sendCommand('play'));
  }
  if (pauseBtn) {
    pauseBtn.addEventListener('click', () => sendCommand('pause'));
  }
  if (stepBtn) {
    stepBtn.addEventListener('click', () => sendCommand('step'));
  }
  if (resetBtn) {
    resetBtn.addEventListener('click', () => sendCommand('reset'));
  }

  if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      if (speedValue) speedValue.textContent = val;
      sendCommand('set_speed', val);
    });
  }

  // 3D Brain View Controls (Neuroglancer Presets with active class highlighting)
  const presetMap = [
    { btn: brainViewFull, mode: 'full' },
    { btn: brainViewBrain, mode: 'brain' },
    { btn: brainViewVNC, mode: 'vnc' },
    { btn: brainViewSide, mode: 'side' },
    { btn: brainViewDorsal, mode: 'dorsal' }
  ];

  function setActivePresetBtn(activeBtn) {
    presetMap.forEach(({ btn }) => {
      if (btn) btn.classList.toggle('active', btn === activeBtn);
    });
  }

  presetMap.forEach(item => {
    if (item.btn) {
      item.btn.addEventListener('click', () => {
        setActivePresetBtn(item.btn);
        if (window.setBrain3DView) window.setBrain3DView(item.mode);
      });
    }
  });
  if (brainRotateBtn) {
    brainRotateBtn.addEventListener('click', () => {
      if (window.toggleBrain3DAutoRotate) {
        const isRotating = window.toggleBrain3DAutoRotate();
        if (isRotating) {
          brainRotateBtn.classList.add('active');
          brainRotateBtn.textContent = 'ROTATION: ACTIVE';
        } else {
          brainRotateBtn.classList.remove('active');
          brainRotateBtn.textContent = 'ROTATION: OFF';
        }
      }
    });
  }

  // Initialize 3D Brain Orbit Controls
  if (window.setupBrain3DInteraction && brain3dCanvas) {
    window.setupBrain3DInteraction(brain3dCanvas);
  }

  // Continuous 60fps Animation Loop for 3D Brain Orbit & Animated Odor Plume Waves
  const odorToggleBtn = document.getElementById('odorToggleBtn');
  const arenaOdorBtn = document.getElementById('arenaOdorBtn');

  window.showOdorField = false;

  function toggleOdorDisplay() {
    window.showOdorField = !window.showOdorField;
    const isShow = window.showOdorField;
    if (odorToggleBtn) {
      odorToggleBtn.classList.toggle('active', isShow);
      odorToggleBtn.textContent = isShow ? 'ODOR STIM: ACTIVE' : 'ODOR STIM: OFF';
    }
    if (arenaOdorBtn) {
      arenaOdorBtn.classList.toggle('active', isShow);
      arenaOdorBtn.textContent = isShow ? 'STIM: ACTIVE' : 'STIM: OFF';
    }
    if (latestFrame && window.renderSnake && snakeCanvas) {
      window.renderSnake(snakeCanvas, latestFrame);
    }
  }

  if (odorToggleBtn) odorToggleBtn.addEventListener('click', toggleOdorDisplay);
  if (arenaOdorBtn) arenaOdorBtn.addEventListener('click', toggleOdorDisplay);

  function animateViews() {
    if (window.renderBrain3D && brain3dCanvas) {
      window.renderBrain3D(brain3dCanvas, latestFrame);
    }
    if (window.renderSnake && snakeCanvas && latestFrame && window.showOdorField) {
      window.renderSnake(snakeCanvas, latestFrame);
    }
    requestAnimationFrame(animateViews);
  }
  requestAnimationFrame(animateViews);

  // Keyboard Shortcuts (Space: Play/Pause, S: Step, R: Reset, O: Toggle Odor)
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') {
      e.preventDefault();
      sendCommand(isPaused ? 'play' : 'pause');
    } else if (e.code === 'KeyS') {
      e.preventDefault();
      sendCommand('step');
    } else if (e.code === 'KeyR') {
      e.preventDefault();
      sendCommand('reset');
    } else if (e.code === 'KeyO') {
      e.preventDefault();
      toggleOdorDisplay();
    }
  });

  // Startup
  checkInitialStatus();
  connectStream();
})();

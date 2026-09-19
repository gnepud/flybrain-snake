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

  // Formatting helper
  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  // Send Control Command via POST /api/control
  async function sendCommand(command, value = null) {
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
        console.error(`Command ${command} failed with HTTP status ${resp.status}`);
        return null;
      }

      const result = await resp.json();
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
      return result;
    } catch (err) {
      console.error('Error dispatching control command:', err);
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

    // Update Footers
    if (snakeStatusText) {
      let statusStr = frame.done
        ? `Episode finished: ${frame.reason || 'Collision'}`
        : `Head at (${frame.head ? frame.head.join(',') : '?'}), Food at (${frame.food ? frame.food.join(',') : '?'})`;
      if (frame.olfactory) {
        const trend = frame.olfactory.delta_c > 0 ? '▲ Closer' : (frame.olfactory.delta_c < 0 ? '▼ Further' : '—');
        const gateStr = frame.olfactory.gating_factor ? ` | Visual Gate: ${frame.olfactory.gating_factor}x` : '';
        statusStr += ` | Olfactory AL: L=${frame.olfactory.c_left} R=${frame.olfactory.c_right} (Δ=${frame.olfactory.diff}, ${trend})${gateStr}`;
      }
      snakeStatusText.textContent = statusStr;
    }

    if (compassStatusText && frame.epg) {
      const maxAct = Math.max(...frame.epg);
      compassStatusText.textContent = `Ring bump peak: ${maxAct.toFixed(3)} | 16 E-PG wedges tracking heading.`;
    }

    if (steeringStatusText && frame.steering) {
      const diff = (frame.steering.pfl3_l || 0) - (frame.steering.pfl3_r || 0);
      steeringStatusText.textContent = `Differential Δ: ${diff.toFixed(3)} | Left: ${(frame.steering.pfl3_l || 0).toFixed(3)}, Right: ${(frame.steering.pfl3_r || 0).toFixed(3)}`;
    }

    if (motorStatusText && frame.motor) {
      const forwardVal = (frame.motor.dna01 !== undefined ? frame.motor.dna01 : (frame.motor.dnp || 0));
      motorStatusText.textContent = `DNa02_L: ${(frame.motor.dna02_l || 0).toFixed(3)} | DNa02_R: ${(frame.motor.dna02_r || 0).toFixed(3)} | DNa01: ${forwardVal.toFixed(3)}`;
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

  // Initialize SSE Connection
  function connectStream() {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource('/api/stream');

    eventSource.onopen = () => {
      console.log('Connected to FlyBrain Snake SSE telemetry stream.');
      if (latestFrame && latestFrame.done) {
        updateStatusBadge('GAME OVER');
      } else {
        updateStatusBadge(isPaused ? 'PAUSED' : 'RUNNING');
      }
    };

    eventSource.onmessage = (e) => {
      try {
        const frame = JSON.parse(e.data);
        onNewFrame(frame);
      } catch (err) {
        console.error('Failed to parse SSE telemetry frame:', err);
      }
    };

    eventSource.onerror = (e) => {
      console.warn('SSE stream disconnected, reconnecting in 2s...', e);
      if (statusBadge) {
        statusBadge.textContent = 'CONNECTING';
        statusBadge.className = 'badge badge-paused';
      }
    };
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
        if (data.done) {
          updateStatusBadge('GAME OVER');
        } else {
          updateStatusBadge(isPaused ? 'PAUSED' : 'RUNNING');
        }
      }
    } catch (e) {
      console.warn('Status check deferred:', e);
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
          brainRotateBtn.textContent = '⟳ Auto-Rotate: ON';
        } else {
          brainRotateBtn.classList.remove('active');
          brainRotateBtn.textContent = '⟳ Auto-Rotate: OFF';
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
      odorToggleBtn.textContent = isShow ? '♨ Odor Plume: ON' : '♨ Odor Plume: OFF';
    }
    if (arenaOdorBtn) {
      arenaOdorBtn.classList.toggle('active', isShow);
      arenaOdorBtn.textContent = isShow ? '♨ Odor: ON' : '♨ Odor: OFF';
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

/**
 * Snake Arena Canvas Renderer
 * Renders the 16x16 discrete grid, food target, snake body segments,
 * directional head with eyes, and terminal status overlays.
 */

function renderSnake(canvas, frame) {
  if (!canvas || !frame) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  const gridSize = 16;
  const cellSize = width / gridSize;

  // 1. Background
  ctx.fillStyle = '#0b0f14';
  ctx.fillRect(0, 0, width, height);

  // 2. Subtle Grid Lines
  ctx.strokeStyle = '#141c26';
  ctx.lineWidth = 1;
  for (let i = 0; i <= gridSize; i++) {
    const pos = i * cellSize;
    ctx.beginPath();
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(width, pos);
    ctx.stroke();
  }

  // 2.5 Food Odor Plume Field (if enabled)
  if (window.showOdorField && frame.food && frame.food.length >= 2) {
    const fx = frame.food[0] * cellSize + cellSize / 2;
    const fy = frame.food[1] * cellSize + cellSize / 2;
    const maxRadius = cellSize * 8.5;

    // A. Multi-stop radial chemical concentration field
    const plumeGrad = ctx.createRadialGradient(fx, fy, cellSize * 0.3, fx, fy, maxRadius);
    plumeGrad.addColorStop(0.0, 'rgba(245, 158, 11, 0.36)');     // Core warm amber
    plumeGrad.addColorStop(0.25, 'rgba(217, 119, 6, 0.20)');
    plumeGrad.addColorStop(0.55, 'rgba(52, 211, 153, 0.10)');    // Volatile mint/emerald diffusion
    plumeGrad.addColorStop(0.85, 'rgba(56, 189, 248, 0.04)');    // Trace edge
    plumeGrad.addColorStop(1.0, 'rgba(11, 15, 20, 0)');
    ctx.fillStyle = plumeGrad;
    ctx.beginPath();
    ctx.arc(fx, fy, maxRadius, 0, Math.PI * 2);
    ctx.fill();

    // B. Concentric pulsating odor diffusion wave rings
    const now = Date.now() / 1000;
    const ringCount = 3;
    ctx.lineWidth = 1.2;
    for (let r = 0; r < ringCount; r++) {
      const phase = ((now * 0.35 + r / ringCount) % 1.0);
      const ringRadius = cellSize * 0.6 + phase * cellSize * 7.5;
      const alpha = Math.sin(phase * Math.PI) * 0.30;
      ctx.strokeStyle = `rgba(245, 158, 11, ${alpha.toFixed(3)})`;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.arc(fx, fy, ringRadius, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // 3. Render Food
  if (frame.food && frame.food.length >= 2) {
    const fx = frame.food[0] * cellSize + cellSize / 2;
    const fy = frame.food[1] * cellSize + cellSize / 2;
    const radius = cellSize * 0.4;

    // Outer glow
    const grad = ctx.createRadialGradient(fx, fy, radius * 0.2, fx, fy, radius * 1.5);
    grad.addColorStop(0, '#f87171');
    grad.addColorStop(0.6, '#ef4444');
    grad.addColorStop(1, 'rgba(239, 68, 68, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(fx, fy, radius * 1.5, 0, Math.PI * 2);
    ctx.fill();

    // Solid core
    ctx.fillStyle = '#f85149';
    ctx.beginPath();
    ctx.arc(fx, fy, radius, 0, Math.PI * 2);
    ctx.fill();

    // Specular highlight
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(fx - radius * 0.3, fy - radius * 0.3, radius * 0.25, 0, Math.PI * 2);
    ctx.fill();
  }

  // 4. Render Snake Body
  if (frame.body && frame.body.length > 0) {
    const bodyLen = frame.body.length;
    for (let i = bodyLen - 1; i >= 1; i--) {
      const seg = frame.body[i];
      const bx = seg[0] * cellSize;
      const by = seg[1] * cellSize;
      const pad = 2;

      // Color fade from head to tail
      const t = i / Math.max(1, bodyLen);
      const r = Math.round(16 + (30 - 16) * t);
      const g = Math.round(185 + (130 - 185) * t);
      const b = Math.round(129 + (100 - 129) * t);

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      roundRect(ctx, bx + pad, by + pad, cellSize - pad * 2, cellSize - pad * 2, 4);
      ctx.fill();
    }
  }

  // 5. Render Snake Head
  if (frame.head && frame.head.length >= 2) {
    const hx = frame.head[0] * cellSize;
    const hy = frame.head[1] * cellSize;
    const pad = 1.5;

    ctx.fillStyle = '#38bdf8';
    roundRect(ctx, hx + pad, hy + pad, cellSize - pad * 2, cellSize - pad * 2, 6);
    ctx.fill();

    // Eyes indicating heading direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
    const dir = frame.direction !== undefined ? frame.direction : 1;
    const cx = hx + cellSize / 2;
    const cy = hy + cellSize / 2;
    const eyeOffset = cellSize * 0.22;
    const eyeRadius = cellSize * 0.1;
    let eye1 = [cx, cy];
    let eye2 = [cx, cy];

    if (dir === 0) { // UP
      eye1 = [cx - eyeOffset, cy - eyeOffset];
      eye2 = [cx + eyeOffset, cy - eyeOffset];
    } else if (dir === 1) { // RIGHT
      eye1 = [cx + eyeOffset, cy - eyeOffset];
      eye2 = [cx + eyeOffset, cy + eyeOffset];
    } else if (dir === 2) { // DOWN
      eye1 = [cx - eyeOffset, cy + eyeOffset];
      eye2 = [cx + eyeOffset, cy + eyeOffset];
    } else { // LEFT
      eye1 = [cx - eyeOffset, cy - eyeOffset];
      eye2 = [cx - eyeOffset, cy + eyeOffset];
    }

    const olf = frame.olfactory || {};
    const visualGain = parseFloat(olf.visual_gain || 3.5);
    const isVisionLocked = (visualGain >= 5.0 || (olf.gating_factor && olf.gating_factor > 1.25));

    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(eye1[0], eye1[1], eyeRadius, 0, Math.PI * 2);
    ctx.arc(eye2[0], eye2[1], eyeRadius, 0, Math.PI * 2);
    ctx.fill();

    // Specular / pupil highlight: radiant gold when vision is locked by odor gating!
    ctx.save();
    if (isVisionLocked) {
      ctx.fillStyle = '#fbbf24';
      ctx.shadowColor = '#f59e0b';
      ctx.shadowBlur = 8;
    } else {
      ctx.fillStyle = '#ffffff';
      ctx.shadowBlur = 0;
    }
    ctx.beginPath();
    ctx.arc(eye1[0], eye1[1], eyeRadius * 0.45, 0, Math.PI * 2);
    ctx.arc(eye2[0], eye2[1], eyeRadius * 0.45, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 5.5 Fruit Fly Bilateral Antennal Sensors & Attraction Gradient (if enabled)
    if (window.showOdorField) {
      const dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // UP, RIGHT, DOWN, LEFT
      const [dx, dy] = dirs[dir] || [1, 0];

      // True forward-left (al) and forward-right (ar) antenna grid coordinates
      const al_gx = frame.head[0] + dx + dy;
      const al_gy = frame.head[1] + dy - dx;
      const ar_gx = frame.head[0] + dx - dy;
      const ar_gy = frame.head[1] + dy + dx;

      const headPx = cx;
      const headPy = cy;
      const alPx = al_gx * cellSize + cellSize / 2;
      const alPy = al_gy * cellSize + cellSize / 2;
      const arPx = ar_gx * cellSize + cellSize / 2;
      const arPy = ar_gy * cellSize + cellSize / 2;

      const cL = olf.c_left !== undefined ? olf.c_left : 0;
      const cR = olf.c_right !== undefined ? olf.c_right : 0;

      // Attraction vector: dashed guidance line from head to food (high-voltage gold when locked)
      if (frame.food && frame.food.length >= 2) {
        const foodPx = frame.food[0] * cellSize + cellSize / 2;
        const foodPy = frame.food[1] * cellSize + cellSize / 2;

        ctx.save();
        if (isVisionLocked) {
          ctx.lineWidth = 2.0;
          ctx.setLineDash([4, 2]);
          ctx.strokeStyle = '#fbbf24';
          ctx.shadowColor = '#f59e0b';
          ctx.shadowBlur = 10;
        } else {
          ctx.lineWidth = 1.2;
          ctx.setLineDash([2, 4]);
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.40)';
          ctx.shadowBlur = 0;
        }
        ctx.beginPath();
        ctx.moveTo(headPx, headPy);
        ctx.lineTo(foodPx, foodPy);
        ctx.stroke();
        ctx.restore();
      }

      // Left & Right Antenna Filaments
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = cL >= cR ? 'rgba(251, 191, 36, 0.85)' : 'rgba(148, 163, 184, 0.45)';
      ctx.beginPath();
      ctx.moveTo(headPx + (dx + dy) * 3.5, headPy + (dy - dx) * 3.5);
      ctx.lineTo(alPx, alPy);
      ctx.stroke();

      ctx.strokeStyle = cR >= cL ? 'rgba(251, 191, 36, 0.85)' : 'rgba(148, 163, 184, 0.45)';
      ctx.beginPath();
      ctx.moveTo(headPx + (dx - dy) * 3.5, headPy + (dy + dx) * 3.5);
      ctx.lineTo(arPx, arPy);
      ctx.stroke();

      // Antenna Sensory Receptor Bulbs
      const baseBulb = cellSize * 0.16;
      const lBulb = baseBulb * (0.8 + Math.min(1.2, cL / 5.5));
      const rBulb = baseBulb * (0.8 + Math.min(1.2, cR / 5.5));

      ctx.save();
      // Left Bulb
      ctx.fillStyle = cL >= cR ? '#fbbf24' : '#64748b';
      ctx.shadowColor = cL >= cR ? '#f59e0b' : 'transparent';
      ctx.shadowBlur = cL >= cR ? 8 : 0;
      ctx.beginPath();
      ctx.arc(alPx, alPy, lBulb, 0, Math.PI * 2);
      ctx.fill();

      // Right Bulb
      ctx.fillStyle = cR >= cL ? '#fbbf24' : '#64748b';
      ctx.shadowColor = cR >= cL ? '#f59e0b' : 'transparent';
      ctx.shadowBlur = cR >= cL ? 8 : 0;
      ctx.beginPath();
      ctx.arc(arPx, arPy, rBulb, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Micro-labels for sensory concentration
      ctx.font = 'bold 9px monospace';
      ctx.fillStyle = cL >= cR ? '#fef08a' : '#94a3b8';
      ctx.fillText(`L:${cL.toFixed(1)}`, alPx - 10, alPy - (dy < 0 ? 5 : -14));

      ctx.fillStyle = cR >= cL ? '#fef08a' : '#94a3b8';
      ctx.fillText(`R:${cR.toFixed(1)}`, arPx - 10, arPy - (dy < 0 ? 5 : -14));
    }
  }

  // 6. Terminal Game Over Overlay
  if (frame.done) {
    ctx.fillStyle = 'rgba(10, 14, 20, 0.75)';
    ctx.fillRect(0, 0, width, height);

    ctx.font = 'bold 22px system-ui, sans-serif';
    ctx.fillStyle = '#f85149';
    ctx.textAlign = 'center';
    ctx.fillText('EPISODE CONCLUDED', width / 2, height / 2 - 15);

    ctx.font = '14px monospace';
    ctx.fillStyle = '#f0f6fc';
    const reasonText = frame.reason ? frame.reason.replace('_', ' ') : 'COMPLETE';
    ctx.fillText(`Reason: ${reasonText}`, width / 2, height / 2 + 15);
    ctx.fillText(`Score: ${frame.score}  •  Steps: ${frame.step}`, width / 2, height / 2 + 38);
  }
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

window.renderSnake = renderSnake;

/**
 * CH-01: 2D Behavioral Arena Monitor
 * Laboratory Electrophysiology Workstation Renderer
 * Renders calibrated 16x16 coordinate reticle, optical target reticle,
 * segmented biological probe, antennal sensing filaments, and chemical diffusion contours.
 */

function renderSnake(canvas, frame) {
  if (!canvas || !frame) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  const gridSize = 16;
  const cellSize = width / gridSize;

  // 1. Dark Substrate Well
  ctx.fillStyle = '#06090e';
  ctx.fillRect(0, 0, width, height);

  // 2. Uniform Board Grid (Clean 16x16 without 4x4 dividing lines)
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#101722';
  for (let i = 0; i <= gridSize; i++) {
    const pos = Math.floor(i * cellSize) + 0.5;

    ctx.beginPath();
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(width, pos);
    ctx.stroke();
  }

  // 4. Food Odor Plume Diffusion Contours (Chemical Stimulation Field)
  if (window.showOdorField && frame.food && frame.food.length >= 2) {
    const fx = frame.food[0] * cellSize + cellSize / 2;
    const fy = frame.food[1] * cellSize + cellSize / 2;
    const maxRadius = cellSize * 8.0;

    // A. Multi-zone calibrated diffusion field
    const plumeGrad = ctx.createRadialGradient(fx, fy, cellSize * 0.25, fx, fy, maxRadius);
    plumeGrad.addColorStop(0.0, 'rgba(245, 158, 11, 0.28)');
    plumeGrad.addColorStop(0.3, 'rgba(217, 119, 6, 0.15)');
    plumeGrad.addColorStop(0.65, 'rgba(16, 185, 129, 0.08)');
    plumeGrad.addColorStop(1.0, 'rgba(6, 9, 14, 0)');
    ctx.fillStyle = plumeGrad;
    ctx.beginPath();
    ctx.arc(fx, fy, maxRadius, 0, Math.PI * 2);
    ctx.fill();

    // B. Calibrated Isocontour Rings with Concentration Labels
    const isocontours = [
      { r: cellSize * 1.8, alpha: 0.35, label: 'c=0.75' },
      { r: cellSize * 3.8, alpha: 0.22, label: 'c=0.50' },
      { r: cellSize * 6.2, alpha: 0.14, label: 'c=0.25' }
    ];

    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    isocontours.forEach(c => {
      ctx.strokeStyle = `rgba(245, 158, 11, ${c.alpha})`;
      ctx.beginPath();
      ctx.arc(fx, fy, c.r, 0, Math.PI * 2);
      ctx.stroke();

      ctx.font = '7.5px "JetBrains Mono", monospace';
      ctx.fillStyle = `rgba(245, 158, 11, ${c.alpha * 1.4})`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(c.label, fx + c.r * 0.707 + 2, fy - c.r * 0.707 - 2);
    });
    ctx.setLineDash([]);
  }

  // 5. Target / Food Optical Reticle
  if (frame.food && frame.food.length >= 2) {
    const fx = frame.food[0] * cellSize + cellSize / 2;
    const fy = frame.food[1] * cellSize + cellSize / 2;
    const radius = cellSize * 0.38;

    // Reticle concentric target rings
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(fx, fy, radius * 1.4, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(fx, fy, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Fine crosshair
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(fx - radius * 1.6, fy);
    ctx.lineTo(fx - radius * 0.3, fy);
    ctx.moveTo(fx + radius * 0.3, fy);
    ctx.lineTo(fx + radius * 1.6, fy);
    ctx.moveTo(fx, fy - radius * 1.6);
    ctx.lineTo(fx, fy - radius * 0.3);
    ctx.moveTo(fx, fy + radius * 0.3);
    ctx.lineTo(fx, fy + radius * 1.6);
    ctx.stroke();

    // Center focal point
    ctx.fillStyle = '#f87171';
    ctx.beginPath();
    ctx.arc(fx, fy, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Coordinate tag
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`TGT[${frame.food[0]},${frame.food[1]}]`, fx, fy + radius * 1.5 + 2);
  }

  // 6. Biological Probe Body Segments
  if (frame.body && frame.body.length > 0) {
    const bodyLen = frame.body.length;
    for (let i = bodyLen - 1; i >= 1; i--) {
      const seg = frame.body[i];
      const bx = seg[0] * cellSize;
      const by = seg[1] * cellSize;
      const pad = 2.5;

      const t = i / Math.max(1, bodyLen);
      // Precision telemetry gradient: from emerald (#10b981) to slate (#334155)
      const r = Math.round(16 + (30 - 16) * t);
      const g = Math.round(160 + (70 - 160) * t);
      const b = Math.round(120 + (85 - 120) * t);

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.strokeStyle = '#0e1724';
      ctx.lineWidth = 1;
      ctx.fillRect(bx + pad, by + pad, cellSize - pad * 2, cellSize - pad * 2);
      ctx.strokeRect(bx + pad, by + pad, cellSize - pad * 2, cellSize - pad * 2);
    }
  }

  // 7. Cranial Sensor Head
  if (frame.head && frame.head.length >= 2) {
    const hx = frame.head[0] * cellSize;
    const hy = frame.head[1] * cellSize;
    const pad = 1.5;
    const cx = hx + cellSize / 2;
    const cy = hy + cellSize / 2;
    const dir = frame.direction !== undefined ? frame.direction : 1;

    // Head casing
    ctx.fillStyle = '#0284c7';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.2;
    ctx.fillRect(hx + pad, hy + pad, cellSize - pad * 2, cellSize - pad * 2);
    ctx.strokeRect(hx + pad, hy + pad, cellSize - pad * 2, cellSize - pad * 2);

    // Directional orientation reticle mark
    const dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // UP, RIGHT, DOWN, LEFT
    const [dx, dy] = dirs[dir] || [1, 0];

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + dx * (cellSize * 0.42), cy + dy * (cellSize * 0.42));
    ctx.stroke();

    const olf = frame.olfactory || {};
    const visualGain = parseFloat(olf.visual_gain || 3.5);
    const isVisionLocked = (visualGain >= 5.0 || (olf.gating_factor && olf.gating_factor > 1.25));

    // Guidance Vector to Target (Displayed strictly when Odor Plume is enabled)
    if (window.showOdorField && frame.food && frame.food.length >= 2) {
      const foodPx = frame.food[0] * cellSize + cellSize / 2;
      const foodPy = frame.food[1] * cellSize + cellSize / 2;

      ctx.save();
      if (isVisionLocked) {
        ctx.lineWidth = 1.4;
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = '#f59e0b';
      } else {
        ctx.lineWidth = 1.0;
        ctx.setLineDash([2, 5]);
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
      }
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(foodPx, foodPy);
      ctx.stroke();
      ctx.restore();
    }

    // Bilateral Antennal Sensors (AL Field Probe)
    if (window.showOdorField) {
      const al_gx = frame.head[0] + dx + dy;
      const al_gy = frame.head[1] + dy - dx;
      const ar_gx = frame.head[0] + dx - dy;
      const ar_gy = frame.head[1] + dy + dx;

      const alPx = al_gx * cellSize + cellSize / 2;
      const alPy = al_gy * cellSize + cellSize / 2;
      const arPx = ar_gx * cellSize + cellSize / 2;
      const arPy = ar_gy * cellSize + cellSize / 2;

      const cL = olf.c_left !== undefined ? olf.c_left : 0;
      const cR = olf.c_right !== undefined ? olf.c_right : 0;

      // Antenna filaments
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = cL >= cR ? 'rgba(245, 158, 11, 0.75)' : 'rgba(100, 116, 139, 0.45)';
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(alPx, alPy);
      ctx.stroke();

      ctx.strokeStyle = cR >= cL ? 'rgba(245, 158, 11, 0.75)' : 'rgba(100, 116, 139, 0.45)';
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(arPx, arPy);
      ctx.stroke();

      // Probe sensors
      const pR = cellSize * 0.14;
      ctx.fillStyle = cL >= cR ? '#f59e0b' : '#334155';
      ctx.beginPath();
      ctx.arc(alPx, alPy, pR, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = cR >= cL ? '#f59e0b' : '#334155';
      ctx.beginPath();
      ctx.arc(arPx, arPy, pR, 0, Math.PI * 2);
      ctx.fill();

      // Micro telemetry labels
      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.fillStyle = cL >= cR ? '#fbbf24' : '#64748b';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`L:${cL.toFixed(1)}`, alPx, alPy - 9);

      ctx.fillStyle = cR >= cL ? '#fbbf24' : '#64748b';
      ctx.fillText(`R:${cR.toFixed(1)}`, arPx, arPy - 9);
    }
  }

  // 8. Terminal Cycle Interruption Overlay (Laboratory Style)
  if (frame.done) {
    ctx.fillStyle = 'rgba(6, 9, 14, 0.85)';
    ctx.fillRect(0, 0, width, height);

    const boxW = width * 0.84;
    const boxH = 110;
    const boxX = (width - boxW) / 2;
    const boxY = (height - boxH) / 2;

    ctx.fillStyle = '#0f1724';
    ctx.fillRect(boxX, boxY, boxW, boxH);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(boxX, boxY, boxW, boxH);

    ctx.font = '600 11px "JetBrains Mono", monospace';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('[ SIGNAL HALTED // BOUNDARY INTERRUPT ]', width / 2, boxY + 16);

    ctx.font = '12px "Inter", sans-serif';
    ctx.fillStyle = '#f1f5f9';
    const reasonText = frame.reason ? frame.reason.replace('_', ' ').toUpperCase() : 'BOUNDARY COLLISION';
    ctx.fillText(`FAULT: ${reasonText}`, width / 2, boxY + 42);

    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`SCORE: ${frame.score}  |  CYCLE STEPS: ${frame.step}`, width / 2, boxY + 68);
    ctx.fillStyle = '#475569';
    ctx.fillText('AUTO-RESETTING SENSORIMOTOR REGISTERS...', width / 2, boxY + 86);
  }
}

window.renderSnake = renderSnake;

/**
 * CH-04: Fan-Shaped Body PFL3 Steering Comparator
 * Laboratory Electrophysiology Workstation Renderer
 * Visualizes the bilateral PFL3 differential steering comparator:
 * Center-zero differential null galvanometer and dual-channel activation columns.
 */

function renderSteering(canvas, frame) {
  if (!canvas || !frame) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // 1. Dark Substrate Well
  ctx.fillStyle = '#06090e';
  ctx.fillRect(0, 0, width, height);

  const steering = frame.steering || {};
  const pfl3_l = parseFloat(steering.pfl3_l || 0);
  const pfl3_r = parseFloat(steering.pfl3_r || 0);
  const diff = pfl3_l - pfl3_r;
  const steerThreshold = parseFloat(steering.threshold !== undefined ? steering.threshold : 0.03);

  // 2. Center-Zero Differential Null Galvanometer (Top Section)
  const meterX = 35;
  const meterY = 50;
  const meterWidth = width - 70;
  const meterHeight = 22;
  const centerX = meterX + meterWidth / 2;

  // Meter Track
  ctx.fillStyle = '#0a1018';
  ctx.fillRect(meterX, meterY, meterWidth, meterHeight);
  ctx.strokeStyle = '#1c2637';
  ctx.lineWidth = 1;
  ctx.strokeRect(meterX, meterY, meterWidth, meterHeight);

  // Deadband Gate Zone ([-steerThreshold, +steerThreshold])
  const maxDiff = 0.50;
  const deadbandPx = (meterWidth / 2) * (steerThreshold / maxDiff);
  ctx.fillStyle = 'rgba(16, 185, 129, 0.10)';
  ctx.fillRect(centerX - deadbandPx, meterY, deadbandPx * 2, meterHeight);
  ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
  ctx.lineWidth = 1;
  ctx.strokeRect(centerX - deadbandPx, meterY, deadbandPx * 2, meterHeight);

  // Active Differential Bar
  const clampedDiff = Math.max(-maxDiff, Math.min(maxDiff, diff));
  const diffPx = (clampedDiff / maxDiff) * (meterWidth / 2);

  if (Math.abs(diff) > steerThreshold) {
    ctx.fillStyle = diff > 0 ? '#0284c7' : '#9333ea';
    if (diff > 0) {
      ctx.fillRect(centerX - diffPx, meterY + 2, diffPx, meterHeight - 4);
    } else {
      ctx.fillRect(centerX, meterY + 2, -diffPx, meterHeight - 4);
    }
  }

  // Scale Ticks (-0.50, -0.25, 0.00, +0.25, +0.50)
  const ticks = [-0.5, -0.25, 0, 0.25, 0.5];
  ticks.forEach(tVal => {
    const tx = centerX + (tVal / maxDiff) * (meterWidth / 2);
    ctx.strokeStyle = tVal === 0 ? '#f1f5f9' : '#334155';
    ctx.lineWidth = tVal === 0 ? 1.5 : 1;
    ctx.beginPath();
    ctx.moveTo(tx, meterY);
    ctx.lineTo(tx, meterY + meterHeight);
    ctx.stroke();

    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = tVal === 0 ? '#94a3b8' : '#475569';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const label = tVal === 0 ? '0' : (tVal > 0 ? `+${tVal.toFixed(2)}` : tVal.toFixed(2));
    ctx.fillText(label, tx, meterY + meterHeight + 3);
  });

  // Header Labels for Differential Meter
  ctx.font = '600 8.5px "JetBrains Mono", monospace';
  ctx.fillStyle = '#38bdf8';
  ctx.textAlign = 'left';
  ctx.fillText('◄ STEER LEFT [CH-L]', meterX, meterY - 14);

  ctx.fillStyle = '#c084fc';
  ctx.textAlign = 'right';
  ctx.fillText('[CH-R] STEER RIGHT ►', meterX + meterWidth, meterY - 14);

  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'center';
  ctx.fillText(`Δ NULL: ${diff >= 0 ? '+' : ''}${diff.toFixed(3)} V  |  GATE: ±${steerThreshold.toFixed(2)} V`, centerX, meterY - 14);

  // 3. Bilateral Channel Columns (Bottom Section)
  const colWidth = 65;
  const colMaxHeight = 150;
  const colY = height - 42;

  const leftColX = centerX - 75 - colWidth / 2;
  const rightColX = centerX + 75 - colWidth / 2;

  const columns = [
    {
      name: 'CH-L: PFL3-L',
      tag: 'LEFT DRIVE',
      val: pfl3_l,
      x: leftColX,
      color: '#0284c7',
      borderColor: '#38bdf8'
    },
    {
      name: 'CH-R: PFL3-R',
      tag: 'RIGHT DRIVE',
      val: pfl3_r,
      x: rightColX,
      color: '#9333ea',
      borderColor: '#c084fc'
    }
  ];

  columns.forEach(col => {
    // Column track
    ctx.fillStyle = '#0a1018';
    ctx.fillRect(col.x, colY - colMaxHeight, colWidth, colMaxHeight);
    ctx.strokeStyle = '#1c2637';
    ctx.lineWidth = 1;
    ctx.strokeRect(col.x, colY - colMaxHeight, colWidth, colMaxHeight);

    // Horizontal grid ticks (0.2, 0.4, 0.6, 0.8)
    for (let step = 0.2; step < 1.0; step += 0.2) {
      const ty = colY - step * colMaxHeight;
      ctx.strokeStyle = '#16202e';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(col.x, ty);
      ctx.lineTo(col.x + colWidth, ty);
      ctx.stroke();
    }

    // Active fill
    const fillH = Math.min(colMaxHeight, Math.max(0, (col.val / 1.0) * colMaxHeight));
    ctx.fillStyle = col.color;
    ctx.fillRect(col.x + 2, colY - fillH, colWidth - 4, fillH);

    // Level top bar
    if (fillH > 2) {
      ctx.strokeStyle = col.borderColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(col.x + 2, colY - fillH);
      ctx.lineTo(col.x + colWidth - 2, colY - fillH);
      ctx.stroke();
    }

    // Numerical readout above bar
    ctx.font = '600 10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#f1f5f9';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`${col.val.toFixed(3)} V`, col.x + colWidth / 2, colY - fillH - 4);

    // Labels below column
    ctx.font = '600 9px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textBaseline = 'top';
    ctx.fillText(col.name, col.x + colWidth / 2, colY + 6);

    ctx.font = '8px "Inter", sans-serif';
    ctx.fillStyle = '#475569';
    ctx.fillText(col.tag, col.x + colWidth / 2, colY + 18);
  });
}

window.renderSteering = renderSteering;

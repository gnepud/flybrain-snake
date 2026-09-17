/**
 * PFL3 Steering Comparator Canvas Renderer
 * Visualizes the Fan-shaped Body bilateral comparator:
 * PFL3_L vs PFL3_R differential, steer threshold markers (+/- 0.05),
 * and bilateral activation level columns.
 */

function renderSteering(canvas, frame) {
  if (!canvas || !frame) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // 1. Background
  ctx.fillStyle = '#0b0f14';
  ctx.fillRect(0, 0, width, height);

  const steering = frame.steering || {};
  const pfl3_l = parseFloat(steering.pfl3_l || 0);
  const pfl3_r = parseFloat(steering.pfl3_r || 0);
  const diff = pfl3_l - pfl3_r;
  const steerThreshold = 0.05;

  // 2. Horizontal Differential Comparator Meter (Top Section)
  const meterX = 40;
  const meterY = 55;
  const meterWidth = width - 80;
  const meterHeight = 26;
  const centerX = meterX + meterWidth / 2;

  // Meter background track
  ctx.fillStyle = '#161e2a';
  ctx.fillRect(meterX, meterY, meterWidth, meterHeight);
  ctx.strokeStyle = '#2b394a';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(meterX, meterY, meterWidth, meterHeight);

  // Deadband zone (+/- 0.05) in center
  const deadbandPx = (meterWidth / 2) * (steerThreshold / 0.5);
  ctx.fillStyle = 'rgba(63, 185, 80, 0.15)';
  ctx.fillRect(centerX - deadbandPx, meterY, deadbandPx * 2, meterHeight);

  // Active differential bar deviating from center
  const maxDiff = 0.5;
  const clampedDiff = Math.max(-maxDiff, Math.min(maxDiff, diff));
  const diffPx = (clampedDiff / maxDiff) * (meterWidth / 2);

  if (Math.abs(diff) > steerThreshold) {
    ctx.fillStyle = diff > 0 ? '#38bdf8' : '#bc8cff';
    if (diff > 0) {
      // Left turn drive
      ctx.fillRect(centerX - diffPx, meterY, diffPx, meterHeight);
    } else {
      // Right turn drive
      ctx.fillRect(centerX, meterY, -diffPx, meterHeight);
    }
  }

  // Center reference line
  ctx.strokeStyle = '#f0f6fc';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(centerX, meterY - 4);
  ctx.lineTo(centerX, meterY + meterHeight + 4);
  ctx.stroke();

  // Threshold tick markers
  ctx.strokeStyle = '#d29922';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(centerX - deadbandPx, meterY);
  ctx.lineTo(centerX - deadbandPx, meterY + meterHeight);
  ctx.moveTo(centerX + deadbandPx, meterY);
  ctx.lineTo(centerX + deadbandPx, meterY + meterHeight);
  ctx.stroke();
  ctx.setLineDash([]);

  // Labels for meter
  ctx.font = '10px monospace';
  ctx.fillStyle = '#8b949e';
  ctx.textAlign = 'center';
  ctx.fillText('◄ STEER LEFT', meterX + 45, meterY - 12);
  ctx.fillText('STEER RIGHT ►', meterX + meterWidth - 45, meterY - 12);
  ctx.fillText(`Δ: ${diff >= 0 ? '+' : ''}${diff.toFixed(3)}`, centerX, meterY + meterHeight + 18);

  // 3. Bilateral PFL3 Level Columns (Bottom Section)
  const colWidth = 60;
  const colMaxHeight = 140;
  const colY = height - 50;

  const leftColX = centerX - 80 - colWidth / 2;
  const rightColX = centerX + 80 - colWidth / 2;

  // Background wells
  ctx.fillStyle = '#161e2a';
  ctx.fillRect(leftColX, colY - colMaxHeight, colWidth, colMaxHeight);
  ctx.fillRect(rightColX, colY - colMaxHeight, colWidth, colMaxHeight);
  ctx.strokeStyle = '#2b394a';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(leftColX, colY - colMaxHeight, colWidth, colMaxHeight);
  ctx.strokeRect(rightColX, colY - colMaxHeight, colWidth, colMaxHeight);

  // Fill heights
  const maxAct = 1.0;
  const leftH = Math.min(colMaxHeight, (pfl3_l / maxAct) * colMaxHeight);
  const rightH = Math.min(colMaxHeight, (pfl3_r / maxAct) * colMaxHeight);

  // Left column fill (PFL3_L)
  const leftGrad = ctx.createLinearGradient(0, colY, 0, colY - leftH);
  leftGrad.addColorStop(0, '#0284c7');
  leftGrad.addColorStop(1, '#38bdf8');
  ctx.fillStyle = leftGrad;
  ctx.fillRect(leftColX, colY - leftH, colWidth, leftH);

  // Right column fill (PFL3_R)
  const rightGrad = ctx.createLinearGradient(0, colY, 0, colY - rightH);
  rightGrad.addColorStop(0, '#7c3aed');
  rightGrad.addColorStop(1, '#bc8cff');
  ctx.fillStyle = rightGrad;
  ctx.fillRect(rightColX, colY - rightH, colWidth, rightH);

  // Values and titles
  ctx.font = 'bold 12px monospace';
  ctx.fillStyle = '#f0f6fc';
  ctx.textAlign = 'center';
  ctx.fillText(pfl3_l.toFixed(3), leftColX + colWidth / 2, colY - leftH - 8);
  ctx.fillText(pfl3_r.toFixed(3), rightColX + colWidth / 2, colY - rightH - 8);

  ctx.font = '11px system-ui, sans-serif';
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('PFL3 Left', leftColX + colWidth / 2, colY + 20);
  ctx.fillText('PFL3 Right', rightColX + colWidth / 2, colY + 20);

  // Decision summary badge in center between columns
  ctx.font = 'bold 12px system-ui, sans-serif';
  let decisionText = 'BALANCED';
  let decisionColor = '#3fb950';
  if (diff > steerThreshold) {
    decisionText = 'TURN LEFT';
    decisionColor = '#38bdf8';
  } else if (-diff > steerThreshold) {
    decisionText = 'TURN RIGHT';
    decisionColor = '#bc8cff';
  }
  ctx.fillStyle = decisionColor;
  ctx.fillText(decisionText, centerX, colY - colMaxHeight / 2);
}

window.renderSteering = renderSteering;

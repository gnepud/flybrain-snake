/**
 * CH-03: Central Complex E-PG Heading Resolver
 * Laboratory Electrophysiology Workstation Renderer
 * Calibrated 360-degree polar dial, 16-wedge Ellipsoid Body (EB) attractor,
 * and high-precision heading vector telemetry.
 */

function renderCompass(canvas, frame) {
  if (!canvas || !frame) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const cx = width / 2;
  const cy = height / 2;

  // 1. Dark Substrate Well
  ctx.fillStyle = '#06090e';
  ctx.fillRect(0, 0, width, height);

  const numWedges = 16;
  const dialRadius = Math.min(width, height) * 0.44;
  const outerR = dialRadius * 0.86;
  const innerR = dialRadius * 0.46;
  const anglePerWedge = (Math.PI * 2) / numWedges;

  const epg = frame.epg && frame.epg.length === 16 ? frame.epg : new Array(16).fill(0);
  const minAct = Math.min(...epg);
  const maxAct = Math.max(...epg);
  const actRange = maxAct - minAct;

  // 2. Outer Calibrated Scale Bezel & Tick Marks
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, dialRadius, 0, Math.PI * 2);
  ctx.stroke();

  // Tick marks every 22.5 deg (16 ticks) and sub-ticks (32 ticks)
  for (let t = 0; t < 32; t++) {
    const angle = (t * Math.PI * 2) / 32 - Math.PI / 2;
    const isMajor = (t % 2 === 0);
    const isCardinal = (t % 8 === 0);
    const tickLen = isCardinal ? 8 : (isMajor ? 5 : 3);

    const x1 = cx + Math.cos(angle) * dialRadius;
    const y1 = cy + Math.sin(angle) * dialRadius;
    const x2 = cx + Math.cos(angle) * (dialRadius - tickLen);
    const y2 = cy + Math.sin(angle) * (dialRadius - tickLen);

    ctx.strokeStyle = isCardinal ? '#64748b' : (isMajor ? '#334155' : '#1e293b');
    ctx.lineWidth = isCardinal ? 1.5 : 1;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  // Cardinal Degree Labels
  const cardinals = [
    { label: '000° [N]', angle: -Math.PI / 2 },
    { label: '090° [E]', angle: 0 },
    { label: '180° [S]', angle: Math.PI / 2 },
    { label: '270° [W]', angle: Math.PI }
  ];

  ctx.font = '8px "JetBrains Mono", monospace';
  ctx.fillStyle = '#64748b';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  cardinals.forEach(c => {
    const r = dialRadius - 14;
    ctx.fillText(c.label, cx + Math.cos(c.angle) * r, cy + Math.sin(c.angle) * r);
  });

  // 3. Render 16 E-PG Annular Wedges
  for (let i = 0; i < numWedges; i++) {
    const startAngle = i * anglePerWedge - Math.PI / 2 - anglePerWedge / 2;
    const endAngle = startAngle + anglePerWedge;
    const val = epg[i] || 0;
    const normVal = actRange > 0.01 ? Math.min(1.0, Math.max(0.0, (val - minAct) / actRange)) : 0.0;

    ctx.beginPath();
    ctx.arc(cx, cy, outerR, startAngle, endAngle, false);
    ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
    ctx.closePath();

    ctx.fillStyle = getCalibratedHeatmap(normVal);
    ctx.fill();

    ctx.strokeStyle = '#06090e';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Wedge ID (R1-8, L1-8)
    const midAngle = startAngle + anglePerWedge / 2;
    const labelR = (outerR + innerR) / 2;
    const lx = cx + Math.cos(midAngle) * labelR;
    const ly = cy + Math.sin(midAngle) * labelR;

    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = normVal > 0.6 ? '#f0fdf4' : 'rgba(148, 163, 184, 0.4)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${i + 1}`, lx, ly);
  }

  // 4. Center Well Hub & Digital Telemetry Readout
  ctx.fillStyle = '#080d14';
  ctx.beginPath();
  ctx.arc(cx, cy, innerR - 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.stroke();

  // 5. Circular Center-of-Mass Heading Vector
  let sumSin = 0;
  let sumCos = 0;
  let totalWeight = 0;
  for (let i = 0; i < numWedges; i++) {
    const angle = i * anglePerWedge - Math.PI / 2;
    const w = Math.max(0, epg[i] || 0);
    sumSin += Math.sin(angle) * w;
    sumCos += Math.cos(angle) * w;
    totalWeight += w;
  }

  let headingAngle = -Math.PI / 2;
  if (totalWeight > 0.01) {
    headingAngle = Math.atan2(sumSin, sumCos);
  } else if (frame.direction !== undefined) {
    const dMap = [-Math.PI / 2, 0, Math.PI / 2, Math.PI];
    headingAngle = dMap[frame.direction] || 0;
  }

  // Convert to compass degrees (0 = North, 90 = East, 180 = South, 270 = West)
  let deg = (headingAngle + Math.PI / 2) * (180 / Math.PI);
  while (deg < 0) deg += 360;
  while (deg >= 360) deg -= 360;

  // Digital Angle Readout inside Center Well
  ctx.font = '600 11px "JetBrains Mono", monospace';
  ctx.fillStyle = '#38bdf8';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`${deg.toFixed(1)}°`, cx, cy - 8);

  ctx.font = '8px "JetBrains Mono", monospace';
  ctx.fillStyle = '#64748b';
  ctx.fillText(`PK:${maxAct.toFixed(2)}`, cx, cy + 8);

  // 6. Precision Needle Indicator (Stylized Laboratory Galvanometer Needle)
  const needleLen = outerR * 0.95;
  const tailLen = innerR * 0.55;

  const nx = cx + Math.cos(headingAngle) * needleLen;
  const ny = cy + Math.sin(headingAngle) * needleLen;
  const tx = cx - Math.cos(headingAngle) * tailLen;
  const ty = cy - Math.sin(headingAngle) * tailLen;

  // Needle counterweight tail
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(tx, ty);
  ctx.stroke();

  // Forward indicator needle
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.stroke();

  // Diamond arrowhead
  const arrowSize = 6;
  ctx.fillStyle = '#38bdf8';
  ctx.beginPath();
  ctx.moveTo(nx, ny);
  ctx.lineTo(
    nx - arrowSize * Math.cos(headingAngle - Math.PI / 6),
    ny - arrowSize * Math.sin(headingAngle - Math.PI / 6)
  );
  ctx.lineTo(
    nx - (arrowSize * 0.6) * Math.cos(headingAngle),
    ny - (arrowSize * 0.6) * Math.sin(headingAngle)
  );
  ctx.lineTo(
    nx - arrowSize * Math.cos(headingAngle + Math.PI / 6),
    ny - arrowSize * Math.sin(headingAngle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();

  // Center pivot pin
  ctx.fillStyle = '#06090e';
  ctx.beginPath();
  ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1;
  ctx.stroke();
}

/**
 * Calibrated Electrophysiology Heatmap Scale
 * Subdued technical spectrum: Deep Slate -> Navy -> Cyan -> Phosphor Bright
 */
function getCalibratedHeatmap(t) {
  if (t <= 0.0) return '#0c141f';
  if (t < 0.3) {
    const s = t / 0.3;
    const r = Math.round(12 + (2 - 12) * s);
    const g = Math.round(20 + (80 - 20) * s);
    const b = Math.round(31 + (150 - 31) * s);
    return `rgb(${r}, ${g}, ${b})`;
  } else if (t < 0.7) {
    const s = (t - 0.3) / 0.4;
    const r = Math.round(2 + (56 - 2) * s);
    const g = Math.round(80 + (189 - 80) * s);
    const b = Math.round(150 + (248 - 150) * s);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const s = (t - 0.7) / 0.3;
    const r = Math.round(56 + (240 - 56) * s);
    const g = Math.round(189 + (253 - 189) * s);
    const b = Math.round(248 + (244 - 248) * s);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

window.renderCompass = renderCompass;

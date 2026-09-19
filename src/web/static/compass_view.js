/**
 * E-PG Neural Compass Canvas Renderer
 * Renders the 16-wedge Ellipsoid Body (EB) polar compass attractor,
 * heat-map color scale, heading vector, and glomerulus wedge numbers.
 */

function renderCompass(canvas, frame) {
  if (!canvas || !frame) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const cx = width / 2;
  const cy = height / 2;

  // 1. Background
  ctx.fillStyle = '#0b0f14';
  ctx.fillRect(0, 0, width, height);

  const numWedges = 16;
  const outerR = Math.min(width, height) * 0.40;
  const innerR = Math.min(width, height) * 0.18;
  const anglePerWedge = (Math.PI * 2) / numWedges;

  const epg = frame.epg && frame.epg.length === 16 ? frame.epg : new Array(16).fill(0);
  const minAct = Math.min(...epg);
  const maxAct = Math.max(...epg);
  const actRange = maxAct - minAct;

  // 2. Draw 16 E-PG Wedges
  // Wedge 0 points North (angle -PI/2)
  for (let i = 0; i < numWedges; i++) {
    const startAngle = i * anglePerWedge - Math.PI / 2 - anglePerWedge / 2;
    const endAngle = startAngle + anglePerWedge;
    const val = epg[i] || 0;
    const normVal = actRange > 0.01 ? Math.min(1.0, Math.max(0.0, (val - minAct) / actRange)) : 0.0;

    ctx.beginPath();
    ctx.arc(cx, cy, outerR, startAngle, endAngle, false);
    ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
    ctx.closePath();

    ctx.fillStyle = getHeatmapColor(normVal);
    ctx.fill();

    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Wedge number label near outer edge
    const midAngle = startAngle + anglePerWedge / 2;
    const labelR = outerR + 14;
    const lx = cx + Math.cos(midAngle) * labelR;
    const ly = cy + Math.sin(midAngle) * labelR;

    ctx.font = '9px monospace';
    ctx.fillStyle = normVal > 0.6 ? '#f0f6fc' : '#64748b';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${i + 1}`, lx, ly);
  }

  // 3. Central Donut Hole
  ctx.fillStyle = '#111827';
  ctx.beginPath();
  ctx.arc(cx, cy, innerR - 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 2;
  ctx.stroke();

  // 4. Calculate Compass Heading Vector (Circular Center-of-Mass)
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
    // Fallback to snake heading
    const dMap = [ -Math.PI / 2, 0, Math.PI / 2, Math.PI ];
    headingAngle = dMap[frame.direction] || 0;
  }

  // 5. Draw Heading Needle / Arrow
  const arrowLen = outerR * 0.90;
  const ax = cx + Math.cos(headingAngle) * arrowLen;
  const ay = cy + Math.sin(headingAngle) * arrowLen;

  // Arrow line
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(ax, ay);
  ctx.stroke();

  // Arrow head
  const headSize = 10;
  ctx.fillStyle = '#38bdf8';
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(
    ax - headSize * Math.cos(headingAngle - Math.PI / 6),
    ay - headSize * Math.sin(headingAngle - Math.PI / 6)
  );
  ctx.lineTo(
    ax - headSize * Math.cos(headingAngle + Math.PI / 6),
    ay - headSize * Math.sin(headingAngle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();

  // Center pivot dot
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fill();

  // 6. Cardinal Direction Markers
  const cardinals = [
    { label: 'N', x: cx, y: cy - outerR - 26 },
    { label: 'E', x: cx + outerR + 26, y: cy },
    { label: 'S', x: cx, y: cy + outerR + 26 },
    { label: 'W', x: cx - outerR - 26, y: cy }
  ];
  ctx.font = 'bold 11px system-ui, sans-serif';
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  cardinals.forEach(c => ctx.fillText(c.label, c.x, c.y));
}

function getHeatmapColor(t) {
  // Colormap: Navy (0.0) -> Blue (0.25) -> Cyan (0.5) -> Amber (0.75) -> Yellow/White (1.0)
  if (t <= 0.0) return '#0f172a';
  if (t < 0.25) {
    const s = t / 0.25;
    return `rgb(${Math.round(15 + 25 * s)}, ${Math.round(23 + 70 * s)}, ${Math.round(42 + 150 * s)})`;
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return `rgb(${Math.round(40 - 20 * s)}, ${Math.round(93 + 89 * s)}, ${Math.round(192 + 20 * s)})`;
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return `rgb(${Math.round(20 + 225 * s)}, ${Math.round(182 - 24 * s)}, ${Math.round(212 - 201 * s)})`;
  } else {
    const s = (t - 0.75) / 0.25;
    return `rgb(${Math.round(245 + 10 * s)}, ${Math.round(158 + 82 * s)}, ${Math.round(11 + 127 * s)})`;
  }
}

window.renderCompass = renderCompass;

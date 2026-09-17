/**
 * Descending Motor Drive Canvas Renderer
 * Visualizes descending premotor activations:
 * DNa02_L (left turn), DNa02_R (right turn), and DNp (forward propulsion),
 * along with the decoded discrete RelativeAction maneuver.
 */

function renderMotor(canvas, frame) {
  if (!canvas || !frame) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // 1. Background
  ctx.fillStyle = '#0b0f14';
  ctx.fillRect(0, 0, width, height);

  const motor = frame.motor || {};
  const dna_l = parseFloat(motor.dna02_l || 0);
  const dna_r = parseFloat(motor.dna02_r || 0);
  const dnp = parseFloat(motor.dnp || 0);
  const actionName = frame.action || 'STRAIGHT';

  // 2. Three Vertical Meters for DNa02_L, DNp, DNa02_R
  const meterWidth = 46;
  const meterMaxHeight = 150;
  const baseY = height - 100;
  const spacing = 75;
  const centerX = width / 2;

  const meters = [
    {
      name: 'DNa02_L',
      sub: 'Left Steer',
      val: dna_l,
      x: centerX - spacing,
      color1: '#0284c7',
      color2: '#38bdf8'
    },
    {
      name: 'DNp',
      sub: 'Forward',
      val: dnp,
      x: centerX,
      color1: '#15803d',
      color2: '#4ade80'
    },
    {
      name: 'DNa02_R',
      sub: 'Right Steer',
      val: dna_r,
      x: centerX + spacing,
      color1: '#7c3aed',
      color2: '#bc8cff'
    }
  ];

  meters.forEach(m => {
    // Meter track background
    ctx.fillStyle = '#161e2a';
    ctx.fillRect(m.x - meterWidth / 2, baseY - meterMaxHeight, meterWidth, meterMaxHeight);
    ctx.strokeStyle = '#2b394a';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(m.x - meterWidth / 2, baseY - meterMaxHeight, meterWidth, meterMaxHeight);

    // Active fill
    const fillH = Math.min(meterMaxHeight, Math.max(0, (m.val / 1.0) * meterMaxHeight));
    const grad = ctx.createLinearGradient(0, baseY, 0, baseY - fillH);
    grad.addColorStop(0, m.color1);
    grad.addColorStop(1, m.color2);
    ctx.fillStyle = grad;
    ctx.fillRect(m.x - meterWidth / 2, baseY - fillH, meterWidth, fillH);

    // Numeric readout
    ctx.font = 'bold 11px monospace';
    ctx.fillStyle = '#f0f6fc';
    ctx.textAlign = 'center';
    ctx.fillText(m.val.toFixed(3), m.x, baseY - fillH - 8);

    // Labels
    ctx.font = 'bold 11px system-ui, sans-serif';
    ctx.fillStyle = '#f0f6fc';
    ctx.fillText(m.name, m.x, baseY + 18);

    ctx.font = '10px system-ui, sans-serif';
    ctx.fillStyle = '#8b949e';
    ctx.fillText(m.sub, m.x, baseY + 32);
  });

  // 3. Decoded Motor Action Badge (Bottom Card)
  const badgeY = height - 42;
  const badgeW = 200;
  const badgeH = 32;
  const badgeX = centerX - badgeW / 2;

  let badgeBg = 'rgba(63, 185, 80, 0.2)';
  let badgeBorder = '#3fb950';
  let badgeText = '▲ STRAIGHT';

  if (actionName === 'TURN_LEFT') {
    badgeBg = 'rgba(56, 189, 248, 0.2)';
    badgeBorder = '#38bdf8';
    badgeText = '◄ TURN LEFT';
  } else if (actionName === 'TURN_RIGHT') {
    badgeBg = 'rgba(188, 140, 255, 0.2)';
    badgeBorder = '#bc8cff';
    badgeText = 'TURN RIGHT ►';
  }

  ctx.fillStyle = badgeBg;
  roundRect(ctx, badgeX, badgeY, badgeW, badgeH, 6);
  ctx.fill();
  ctx.strokeStyle = badgeBorder;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.font = 'bold 13px system-ui, sans-serif';
  ctx.fillStyle = badgeBorder;
  ctx.textAlign = 'center';
  ctx.fillText(badgeText, centerX, badgeY + 21);
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

window.renderMotor = renderMotor;

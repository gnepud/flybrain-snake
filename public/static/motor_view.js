/**
 * CH-05: Descending Premotor Drive Bus
 * Laboratory Electrophysiology Workstation Renderer
 * Calibrated 3-channel descending motor bus (DNa02_L, DNa01, DNa02_R),
 * discrete decoded action register, and VNC clearance reflex interlock indicator.
 */

function renderMotor(canvas, frame) {
  if (!canvas || !frame) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // 1. Dark Substrate Well
  ctx.fillStyle = '#06090e';
  ctx.fillRect(0, 0, width, height);

  const motor = frame.motor || {};
  const dna_l = parseFloat(motor.dna02_l || 0);
  const dna_r = parseFloat(motor.dna02_r || 0);
  const dna01 = parseFloat(motor.dna01 !== undefined ? motor.dna01 : (motor.dnp || 0));
  const actionName = frame.action || 'STRAIGHT';

  // 2. Three Vertical Channels
  const meterWidth = 50;
  const meterMaxHeight = 150;
  const baseY = height - 90;
  const spacing = 80;
  const centerX = width / 2;

  const meters = [
    {
      channel: 'CH-1',
      name: 'DNa02-L',
      sub: 'STEER L',
      val: dna_l,
      x: centerX - spacing,
      color: '#0284c7',
      borderColor: '#38bdf8'
    },
    {
      channel: 'CH-2',
      name: 'DNa01',
      sub: 'PROPEL',
      val: dna01,
      x: centerX,
      color: '#10b981',
      borderColor: '#34d399'
    },
    {
      channel: 'CH-3',
      name: 'DNa02-R',
      sub: 'STEER R',
      val: dna_r,
      x: centerX + spacing,
      color: '#9333ea',
      borderColor: '#c084fc'
    }
  ];

  // Draw Horizontal Scale Reference Lines Across All Channels
  const scaleLevels = [0.25, 0.50, 0.75, 1.00];
  scaleLevels.forEach(lvl => {
    const ly = baseY - lvl * meterMaxHeight;
    ctx.strokeStyle = '#121a26';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX - spacing - meterWidth, ly);
    ctx.lineTo(centerX + spacing + meterWidth, ly);
    ctx.stroke();

    ctx.font = '7px "JetBrains Mono", monospace';
    ctx.fillStyle = '#334155';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${lvl.toFixed(2)}V`, centerX - spacing - meterWidth / 2 - 4, ly);
  });

  meters.forEach(m => {
    // Channel Track Well
    ctx.fillStyle = '#0a1018';
    ctx.fillRect(m.x - meterWidth / 2, baseY - meterMaxHeight, meterWidth, meterMaxHeight);
    ctx.strokeStyle = '#1c2637';
    ctx.lineWidth = 1;
    ctx.strokeRect(m.x - meterWidth / 2, baseY - meterMaxHeight, meterWidth, meterMaxHeight);

    // Active Fill Bar
    const fillH = Math.min(meterMaxHeight, Math.max(0, (m.val / 1.0) * meterMaxHeight));
    ctx.fillStyle = m.color;
    ctx.fillRect(m.x - meterWidth / 2 + 2, baseY - fillH, meterWidth - 4, fillH);

    // Peak Level Line
    if (fillH > 2) {
      ctx.strokeStyle = m.borderColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(m.x - meterWidth / 2 + 2, baseY - fillH);
      ctx.lineTo(m.x + meterWidth / 2 - 2, baseY - fillH);
      ctx.stroke();
    }

    // Digital Readout above meter
    ctx.font = '600 10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#f1f5f9';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`${m.val.toFixed(3)} V`, m.x, baseY - fillH - 4);

    // Channel Labels below meter
    ctx.font = '600 8.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textBaseline = 'top';
    ctx.fillText(m.name, m.x, baseY + 6);

    ctx.font = '7.5px "Inter", sans-serif';
    ctx.fillStyle = '#475569';
    ctx.fillText(`${m.channel} // ${m.sub}`, m.x, baseY + 18);
  });

  // 3. Decoded Action Register & Interlock Status (Bottom Card)
  const isOverride = Boolean(motor.is_override || (frame.steering && frame.steering.is_override));
  const barY = height - 44;
  const barW = width - 40;
  const barH = 28;
  const barX = (width - barW) / 2;

  ctx.fillStyle = '#0a1018';
  ctx.fillRect(barX, barY, barW, barH);
  ctx.strokeStyle = isOverride ? '#b45309' : '#1c2637';
  ctx.lineWidth = 1;
  ctx.strokeRect(barX, barY, barW, barH);

  // Left: Decoded Maneuver
  ctx.font = '600 9px "JetBrains Mono", monospace';
  ctx.fillStyle = '#f1f5f9';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(`REGISTER: ${actionName}`, barX + 12, barY + barH / 2);

  // Right: VNC Clearance Reflex Interlock Indicator
  ctx.textAlign = 'right';
  if (isOverride) {
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('⚡ VNC REFLEX INTERLOCK: ENGAGED', barX + barW - 12, barY + barH / 2);
  } else {
    ctx.fillStyle = '#64748b';
    ctx.fillText('VNC INTERLOCK: NOMINAL', barX + barW - 12, barY + barH / 2);
  }
}

window.renderMotor = renderMotor;

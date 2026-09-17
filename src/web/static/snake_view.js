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

    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(eye1[0], eye1[1], eyeRadius, 0, Math.PI * 2);
    ctx.arc(eye2[0], eye2[1], eyeRadius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(eye1[0], eye1[1], eyeRadius * 0.4, 0, Math.PI * 2);
    ctx.arc(eye2[0], eye2[1], eyeRadius * 0.4, 0, Math.PI * 2);
    ctx.fill();
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

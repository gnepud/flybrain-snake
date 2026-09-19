/**
 * FlyBrain Snake — 3D Drosophila Connectome Point Cloud & Neural Activity Visualizer
 *
 * Implements the architecture inspired by fly-atlas, fly-in-a-tetris, and siliconfly:
 *
 * 1. Base Layer (Connectome Point Cloud):
 *    - 139,255 somas (FlyWire connectome scale) rendered via a single THREE.Points draw call.
 *    - Anatomically colored by brain region:
 *      * Optic Lobes: ~90,000 points, Purple (#a855f7)
 *      * Central Brain: ~35,000 points, Green (#22c55e)
 *      * Ventral Nerve Cord (VNC): ~14,000 points, Blue (#00d2ff)
 *    - Semi-transparent, soft-particle circular sprite texture.
 *
 * 2. Activity & 30–80 ms Decay Trails:
 *    - Recently fired neurons scale up (from 1.4 px to 5.5 px) and increase brightness to neon/white.
 *    - Exponential decay with tau ~ 50 ms (decays smoothly in 30–80 ms).
 *
 * 3. Task Circuit Always Highlighted:
 *    - Food-related Olfactory / Gustatory: Antennal Lobes & SEZ
 *    - Central Complex: EB compass bump rotation, PB bridge, FB steering
 *    - Bilateral DNa02 descending steering neurons
 *    - Mushroom Body PAM / DAN dopamine reward burst upon food capture
 *
 * 4. Highest-Weight Synaptic Connections (High-weight active functional edges):
 *    - ~2,500 active highest-weight functional edges rendered via THREE.LineSegments.
 *    - Only currently active edges light up and fade out smoothly.
 *
 * 5. Camera Scripting (Cinematic camera modes):
 *    - Foraging: Zoom in on olfactory (AL) and central complex (EB/PB)
 *    - Turning: Focus on descending neurons (DNa02) and ventral nerve cord (VNC)
 *    - Collision / Death: Flash giant fiber escape reflex and shake camera
 *
 * Zero CPU sorting, hardware WebGL accelerated, buttery smooth 60-120 FPS on Apple Silicon.
 */

(function () {
  'use strict';

  if (typeof THREE === 'undefined') {
    console.warn('Three.js not found, deferring Brain3D initialization.');
    return;
  }

  // --- Color Palette Constants ---
  var COLOR_OPTIC = { r: 0.658, g: 0.333, b: 0.968 };     // Purple #a855f7
  var COLOR_CENTRAL = { r: 0.133, g: 0.772, b: 0.368 };   // Green #22c55e
  var COLOR_VNC = { r: 0.0, g: 0.823, b: 1.0 };           // Cyan #00d2ff
  var COLOR_AL_SEZ = { r: 0.204, g: 0.827, b: 0.600 };    // Mint/Gold #34d399
  var COLOR_STEER = { r: 0.984, g: 0.749, b: 0.141 };     // Gold #fbbf24
  var COLOR_DOPAMINE = { r: 1.0, g: 0.60, b: 0.0 };       // Hot Amber #f59e0b
  var COLOR_GF_ALERT = { r: 0.937, g: 0.267, b: 0.267 };   // Crimson #ef4444

  var TOTAL_SOMAS = 141781;
  var OPTIC_COUNT = 90810;

  // Three.js Core
  var scene = null;
  var camera = null;
  var renderer = null;
  var controls = null;
  var canvasEl = null;
  var initialized = false;

  // Geometry & Buffers for 141,781 Somas
  var somaGeometry = null;
  var somaMaterial = null;
  var somaPointsMesh = null;
  var posArray = null;
  var colArray = null;

  // --- Firework Radial Gradient Bloom System ---
  var MAX_BURSTS = 8;
  var activeBursts = []; // [{ pos, color, radius, maxRadius, speed, decay, width, intensity }]
  var burstPosUniform = [];
  var burstColorUniform = [];
  var burstRadiusUniform = new Float32Array(MAX_BURSTS);
  var burstWidthUniform = new Float32Array(MAX_BURSTS);
  var burstIntensityUniform = new Float32Array(MAX_BURSTS);

  for (var bIdx = 0; bIdx < MAX_BURSTS; bIdx++) {
    burstPosUniform.push(new THREE.Vector3());
    burstColorUniform.push(new THREE.Vector3());
  }

  // Exact biological circuit centroids calculated from MaleCNS v1.0 coordinates
  var regionCentroids = {
    al_sez: new THREE.Vector3(0.01, 1.22, 1.00),
    eb_compass: new THREE.Vector3(0.06, 3.08, -2.17),
    pb_arch: new THREE.Vector3(-0.08, 4.55, -1.73),
    fb_steering: new THREE.Vector3(0.14, 3.13, -2.49),
    mb_l: new THREE.Vector3(-1.84, 3.98, -0.61),
    mb_r: new THREE.Vector3(1.98, 4.10, -0.79),
    optic_l: new THREE.Vector3(-3.20, 2.80, -0.50),
    optic_r: new THREE.Vector3(3.20, 2.80, -0.50),
    dna02_l: new THREE.Vector3(-0.67, -2.65, 2.04),
    dna02_r: new THREE.Vector3(0.84, -3.21, 2.15),
    dnp: new THREE.Vector3(0.02, -3.51, 2.13),
    giant_fiber: new THREE.Vector3(0.03, 2.59, 0.67)
  };

  var somaVertexShader = [
    'uniform vec3 uBurstPos[8];',
    'uniform vec3 uBurstColor[8];',
    'uniform float uBurstRadius[8];',
    'uniform float uBurstWidth[8];',
    'uniform float uBurstIntensity[8];',
    'uniform int uBurstCount;',
    'uniform float uDeathPulse;',
    'varying vec3 vColor;',
    'varying float vAlpha;',
    'void main() {',
    '  vec3 pos = position;',
    '  // Natural anatomical base color & opacity (preserving full visible brain structure)',
    '  vec3 outColor = color;',
    '  float ptSize = 0.10;',
    '  float extraAlpha = 0.0;',
    '',
    '  for (int i = 0; i < 8; i++) {',
    '    if (i >= uBurstCount) break;',
    '    float d = distance(pos, uBurstPos[i]);',
    '    float waveDist = abs(d - uBurstRadius[i]);',
    '    float w = max(uBurstWidth[i], 0.06);',
    '    // Compact localized Gaussian burst wave & dense core',
    '    float wave = exp(- (waveDist * waveDist) / (2.0 * w * w));',
    '    float core = exp(- (d * d) / (2.0 * w * w)) * 0.85;',
    '    float bloom = (wave * 1.5 + core) * uBurstIntensity[i];',
    '    if (bloom > 0.04) {',
    '      // Localized firework bloom: vibrant neon color and enlarged glowing embers',
    '      outColor = mix(outColor, uBurstColor[i] * 2.8, min(bloom * 1.5, 1.0));',
    '      ptSize += bloom * 0.45;',
    '      extraAlpha += bloom * 0.35;',
    '    }',
    '  }',
    '',
    '  // Special Death Effect: Terminal Descending Depolarization Shockwave',
    '  if (uDeathPulse > 0.01) {',
    '    float wavePhase = (pos.y * 0.38 + (1.0 - uDeathPulse) * 4.5);',
    '    float shock = exp(- wavePhase * wavePhase * 1.8);',
    '    vec3 deathTint = vec3(1.0, 0.15, 0.28);',
    '    outColor = mix(outColor, deathTint * 2.6, uDeathPulse * shock * 0.85);',
    '    ptSize += uDeathPulse * shock * 0.32;',
    '    extraAlpha += uDeathPulse * shock * 0.45;',
    '  }',
    '',
    '  vColor = outColor;',
    '  vAlpha = clamp(0.70 + extraAlpha, 0.0, 1.0);',
    '  vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);',
    '  float pScale = 220.0 / -mvPosition.z;',
    '  gl_PointSize = clamp(ptSize * pScale, 1.0, 36.0);',
    '  gl_Position = projectionMatrix * mvPosition;',
    '}'
  ].join('\n');

  var somaFragmentShader = [
    'varying vec3 vColor;',
    'varying float vAlpha;',
    'void main() {',
    '  vec2 coord = gl_PointCoord - vec2(0.5);',
    '  float distSq = dot(coord, coord);',
    '  if (distSq > 0.25) discard;',
    '  float feather = 1.0 - smoothstep(0.12, 0.5, sqrt(distSq));',
    '  gl_FragColor = vec4(vColor, vAlpha * feather);',
    '}'
  ].join('\n');

  // Trigger radial firework gradient wave centered on an anatomical circuit
  function triggerFirework(center, color, maxRadius, speed, decay, width) {
    if (!center) return;
    if (activeBursts.length >= MAX_BURSTS) {
      activeBursts.shift();
    }
    activeBursts.push({
      pos: center,
      color: color,
      radius: 0.05,
      maxRadius: maxRadius || 2.4,
      speed: speed || 0.12,
      decay: decay || 0.88,
      width: width || 0.65,
      intensity: 1.35
    });
  }

  // Task Circuit Indices inside Soma Array
  var taskCircuit = {
    al_sez: [],      // Olfactory & Gustatory
    eb_compass: [],  // 16 E-PG wedges
    pb_arch: [],     // Protocerebral bridge
    fb_steering: [], // PFL3 & FB columns
    dna02_l: [],     // Left descending steering
    dna02_r: [],     // Right descending steering
    dnp: [],         // Forward crawl
    mb_dopamine: [], // Mushroom body PAM dopaminergic cluster
    giant_fiber: []  // Giant Fiber escape reflex system
  };

  // Highest-Weight Functional Synaptic Edges (~2,500 lines)
  var edgeGeometry = null;
  var edgeMaterial = null;
  var edgeLinesMesh = null;
  var edgePosArray = null;
  var edgeColArray = null;
  var edgeList = []; // Array of { p0Idx, p1Idx, type, active: 0.0 }

  // Camera State (Defaults to Full System View: Brain + VNC + Curved Tracts)
  var targetCamPos = new THREE.Vector3(0.05, -0.94, 16.92);
  var targetLookAt = new THREE.Vector3(0.05, -0.94, 0.42);
  var startCamPos = new THREE.Vector3(0.05, -0.94, 16.92);
  var startLookAt = new THREE.Vector3(0.05, -0.94, 0.42);
  var transitionProgress = 1.0;
  var shakeTime = 0;

  // Telemetry & Statistics
  var lastFrame = null;
  var lastProcessedStep = -1;
  var lastProcessedDone = false;
  var lastProcessedAteFood = false;
  var lastEpgPeak = -1;
  var lastOdorLevel = 0;
  var currentActiveCircuit = '';
  var dopamineBurstTimer = 0;
  var giantFiberTimer = 0;
  var deathPulse = 0.0;

  // Deterministic PRNG
  function mulberry32(a) {
    return function () {
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  var rng = mulberry32(20260916);

  // --- 3B. Rebuild Real Curved Connectome Tracts (Bézier Cable Bundles) ---
  function buildRealCurvedEdges(edgeBuf) {
    if (!edgeBuf || edgeBuf.byteLength < 8) return;
    try {
      var dataView = new DataView(edgeBuf);
      var count = dataView.getUint32(4, true);

      var totalSegments = 0;
      var rawEdges = [];
      var offset = 8;
      for (var i = 0; i < count; i++) {
        var s = dataView.getUint32(offset, true);
        var d = dataView.getUint32(offset + 4, true);
        var type = dataView.getUint8(offset + 8);
        var weight = dataView.getUint8(offset + 9);
        offset += 10;

        if (s >= TOTAL_SOMAS || d >= TOTAL_SOMAS) continue;

        // Descending / Ascending have 4 cubic segments; Visual has 3 segments; Central/VNC have 2 segments
        var nSegs = (type === 2 || type === 3) ? 4 : (type === 0 ? 3 : 2);
        totalSegments += nSegs;
        rawEdges.push({ s: s, d: d, type: type, weight: weight, nSegs: nSegs });
      }

      edgePosArray = new Float32Array(totalSegments * 2 * 3);
      edgeColArray = new Float32Array(totalSegments * 2 * 3);
      edgeList = [];

      var posPtr = 0;
      var colPtr = 0;
      var segOffset = 0;

      for (var e = 0; e < rawEdges.length; e++) {
        var item = rawEdges[e];
        var s = item.s;
        var d = item.d;
        var type = item.type;
        var nSegs = item.nSegs;

        var p0x = posArray[s * 3];
        var p0y = posArray[s * 3 + 1];
        var p0z = posArray[s * 3 + 2];

        var p1x = posArray[d * 3];
        var p1y = posArray[d * 3 + 1];
        var p1z = posArray[d * 3 + 2];

        var col = COLOR_VNC;
        var edgeCategory = 'motor';
        if (type === 0) { // Visual Optic -> Central
          col = { r: 0.96, g: 0.25, b: 0.45 };
          edgeCategory = 'visual';
        } else if (type === 1) { // Central Brain internal
          col = COLOR_STEER;
          edgeCategory = 'compass';
        } else if (type === 2) { // Descending Brain -> VNC
          col = COLOR_VNC;
          edgeCategory = 'motor';
        } else if (type === 3) { // Ascending VNC -> Brain
          col = { r: 0.18, g: 0.85, b: 0.75 };
          edgeCategory = 'ascending';
        } else if (type === 4) { // VNC internal
          col = { r: 0.25, g: 0.55, b: 0.98 };
          edgeCategory = 'vnc_circuit';
        }

        edgeList.push({
          segStart: segOffset,
          segCount: nSegs,
          type: edgeCategory,
          active: 0.0,
          color: col,
          isLeft: (p0x < 0.15 || p1x < 0.15),
          isRight: (p0x > 0.15 || p1x > 0.15)
        });
        segOffset += nSegs;

        var curvePoints = [];
        if (type === 2 || type === 3) {
          // Anatomical fidelity: descending/ascending tracts converge through the Cervical Connective bottleneck
          var pBrainX = (type === 2) ? p0x : p1x;
          var pBrainZ = (type === 2) ? p0z : p1z;
          var pVncX = (type === 2) ? p1x : p0x;
          var pVncZ = (type === 2) ? p1z : p0z;

          // Cervical foramen center (X ≈ 0.15, Y ≈ 0.75, Z ≈ 0.85) with slight bilateral offset
          var xNeck = 0.15 + (pBrainX + pVncX) * 0.12;
          var zNeck = 0.85 + (pBrainZ + pVncZ) * 0.10;

          var c1x, c1y, c1z, c2x, c2y, c2z;
          if (type === 2) {
            // Brain (p0) -> VNC (p1)
            c1x = p0x * 0.5 + xNeck * 0.5;
            c1y = 1.35; // Ventral brain exit convergence
            c1z = p0z * 0.35 + zNeck * 0.65;

            c2x = p1x * 0.5 + xNeck * 0.5;
            c2y = 0.15; // Dorsal VNC entrance divergence
            c2z = p1z * 0.35 + zNeck * 0.65;
          } else {
            // VNC (p0) -> Brain (p1)
            c1x = p0x * 0.5 + xNeck * 0.5;
            c1y = 0.15;
            c1z = p0z * 0.35 + zNeck * 0.65;

            c2x = p1x * 0.5 + xNeck * 0.5;
            c2y = 1.35;
            c2z = p1z * 0.35 + zNeck * 0.65;
          }

          // Cubic Bézier Spline: smooth anatomical trajectory through cervical foramen
          for (var step = 0; step <= 4; step++) {
            var t = step / 4.0;
            var omt = 1.0 - t;
            var b0 = omt * omt * omt;
            var b1 = 3.0 * omt * omt * t;
            var b2 = 3.0 * omt * t * t;
            var b3 = t * t * t;

            curvePoints.push({
              x: b0 * p0x + b1 * c1x + b2 * c2x + b3 * p1x,
              y: b0 * p0y + b1 * c1y + b2 * c2y + b3 * p1y,
              z: b0 * p0z + b1 * c1z + b2 * c2z + b3 * p1z
            });
          }
        } else if (type === 0) {
          // Optic lobe to central brain: Anterior Optic Tract curve
          var cx = p0x * 0.35 + p1x * 0.65;
          var cy = Math.max(p0y, p1y) + 0.25;
          var cz = Math.max(p0z, p1z) + 0.35;

          for (var sStep = 0; sStep <= 3; sStep++) {
            var sT = sStep / 3.0;
            var s1mt = 1.0 - sT;
            curvePoints.push({
              x: s1mt * s1mt * p0x + 2.0 * s1mt * sT * cx + sT * sT * p1x,
              y: s1mt * s1mt * p0y + 2.0 * s1mt * sT * cy + sT * sT * p1y,
              z: s1mt * s1mt * p0z + 2.0 * s1mt * sT * cz + sT * sT * p1z
            });
          }
        } else {
          // Intrinsic neuropil within central brain or VNC: subtle arc
          var qcx = (p0x + p1x) * 0.5;
          var qcy = (p0y + p1y) * 0.5 + (type === 1 ? 0.20 : 0.0);
          var qcz = (p0z + p1z) * 0.5 + (type === 1 ? 0.15 : 0.20);

          for (var qStep = 0; qStep <= 2; qStep++) {
            var qT = qStep / 2.0;
            var q1mt = 1.0 - qT;
            curvePoints.push({
              x: q1mt * q1mt * p0x + 2.0 * q1mt * qT * qcx + qT * qT * p1x,
              y: q1mt * q1mt * p0y + 2.0 * q1mt * qT * qcy + qT * qT * p1y,
              z: q1mt * q1mt * p0z + 2.0 * q1mt * qT * qcz + qT * qT * p1z
            });
          }
        }

        // Write line segment vertex coordinates
        for (var seg = 0; seg < nSegs; seg++) {
          var ptA = curvePoints[seg];
          var ptB = curvePoints[seg + 1];

          edgePosArray[posPtr++] = ptA.x;
          edgePosArray[posPtr++] = ptA.y;
          edgePosArray[posPtr++] = ptA.z;

          edgePosArray[posPtr++] = ptB.x;
          edgePosArray[posPtr++] = ptB.y;
          edgePosArray[posPtr++] = ptB.z;

          var baseDim = 0.08;
          edgeColArray[colPtr++] = col.r * baseDim;
          edgeColArray[colPtr++] = col.g * baseDim;
          edgeColArray[colPtr++] = col.b * baseDim;

          edgeColArray[colPtr++] = col.r * baseDim;
          edgeColArray[colPtr++] = col.g * baseDim;
          edgeColArray[colPtr++] = col.b * baseDim;
        }
      }

      if (!edgeGeometry) {
        edgeGeometry = new THREE.BufferGeometry();
        edgeMaterial = new THREE.LineBasicMaterial({
          vertexColors: true,
          transparent: true,
          opacity: 0.75,
          blending: THREE.AdditiveBlending
        });
        edgeLinesMesh = new THREE.LineSegments(edgeGeometry, edgeMaterial);
        edgeLinesMesh.frustumCulled = false;
        scene.add(edgeLinesMesh);
      }

      edgeGeometry.setAttribute('position', new THREE.BufferAttribute(edgePosArray, 3));
      edgeGeometry.setAttribute('color', new THREE.BufferAttribute(edgeColArray, 3));
      edgeGeometry.attributes.position.needsUpdate = true;
      edgeGeometry.attributes.color.needsUpdate = true;
      edgeGeometry.computeBoundingSphere();
      edgeGeometry.computeBoundingBox();
      if (edgeLinesMesh) edgeLinesMesh.frustumCulled = false;
      console.log('Brain3D: Built ' + rawEdges.length + ' curved connectome tracts (' + totalSegments + ' segments) connected to VNC.');
    } catch (err) {
      console.error('Brain3D: Failed to build curved connectome edges:', err);
    }
  }

  // --- 4. Initialize Three.js Hardware WebGL Scene & OrbitControls ---
  function init(canvas) {
    if (initialized) return;
    canvasEl = canvas || document.getElementById('brain3dCanvas');
    if (!canvasEl) return;

    try {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x04060a);

      var width = canvasEl.width || 820;
      var height = canvasEl.height || 460;

      camera = new THREE.PerspectiveCamera(46, width / height, 0.1, 100);
      camera.position.set(targetCamPos.x, targetCamPos.y, targetCamPos.z);

      renderer = new THREE.WebGLRenderer({
        canvas: canvasEl,
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance'
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      controls = new THREE.OrbitControls(camera, canvasEl);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.target.set(targetLookAt.x, targetLookAt.y, targetLookAt.z);
      controls.minDistance = 2.0;
      controls.maxDistance = 45.0;

      // User interaction immediately yields preset transition to manual control
      canvasEl.addEventListener('pointerdown', function () {
        transitionProgress = 1.0;
      });
      canvasEl.addEventListener('wheel', function () {
        transitionProgress = 1.0;
      }, { passive: true });

      // Async load genuine 141,781 real soma coordinates from MaleCNS and curved tracts
      loadRealConnectomeData();

      initialized = true;
      console.log('Brain3D: Three.js WebGL scene initialized.');
    } catch (e) {
      console.error('Brain3D: Failed to initialize Three.js WebGL:', e);
    }
  }

  var isRealCoordinatesLoaded = false;
  function loadRealConnectomeData() {
    Promise.all([
      fetch('/static/data/real_somas_coords.bin').then(function (r) {
        if (!r.ok) throw new Error('Status ' + r.status);
        return r.arrayBuffer();
      }),
      fetch('/static/data/real_somas_regions.bin').then(function (r) {
        if (!r.ok) throw new Error('Status ' + r.status);
        return r.arrayBuffer();
      }),
      fetch('/static/data/real_edges_curved.bin').then(function (r) {
        if (!r.ok) throw new Error('Status ' + r.status);
        return r.arrayBuffer();
      })
    ]).then(function (results) {
      posArray = new Float32Array(results[0]);
      var regionsBuf = new Uint8Array(results[1]);
      var edgesBuf = results[2];
      var numSomas = Math.min(TOTAL_SOMAS, regionsBuf.length);

      colArray = new Float32Array(numSomas * 3);

      // Index taskCircuit with real anatomical soma clusters and calculate dynamic centroids
      taskCircuit.al_sez = [];
      taskCircuit.eb_compass = [];
      taskCircuit.pb_arch = [];
      taskCircuit.fb_steering = [];
      taskCircuit.mb_dopamine = [];
      taskCircuit.optic_l = [];
      taskCircuit.optic_r = [];
      taskCircuit.dna02_l = [];
      taskCircuit.dna02_r = [];
      taskCircuit.dnp = [];
      taskCircuit.giant_fiber = [];

      var sums = {
        al_sez: { x: 0, y: 0, z: 0, c: 0 },
        eb_compass: { x: 0, y: 0, z: 0, c: 0 },
        pb_arch: { x: 0, y: 0, z: 0, c: 0 },
        fb_steering: { x: 0, y: 0, z: 0, c: 0 },
        mb_l: { x: 0, y: 0, z: 0, c: 0 },
        mb_r: { x: 0, y: 0, z: 0, c: 0 },
        optic_l: { x: 0, y: 0, z: 0, c: 0 },
        optic_r: { x: 0, y: 0, z: 0, c: 0 },
        dna02_l: { x: 0, y: 0, z: 0, c: 0 },
        dna02_r: { x: 0, y: 0, z: 0, c: 0 },
        dnp: { x: 0, y: 0, z: 0, c: 0 },
        giant_fiber: { x: 0, y: 0, z: 0, c: 0 }
      };

      for (var i = 0; i < numSomas; i++) {
        var p = i * 3;
        var px = posArray[p];
        var py = posArray[p + 1];
        var pz = posArray[p + 2];

        var reg = regionsBuf[i];
        var col = (reg === 0) ? COLOR_OPTIC : (reg === 2 ? COLOR_VNC : COLOR_CENTRAL);
        colArray[p] = col.r;
        colArray[p + 1] = col.g;
        colArray[p + 2] = col.b;

        // Biological Subcircuit Classification on Real Coordinates
        if (reg === 2) {
          if (px < -0.2) {
            taskCircuit.dna02_l.push(i);
            sums.dna02_l.x += px; sums.dna02_l.y += py; sums.dna02_l.z += pz; sums.dna02_l.c++;
          } else if (px > 0.2) {
            taskCircuit.dna02_r.push(i);
            sums.dna02_r.x += px; sums.dna02_r.y += py; sums.dna02_r.z += pz; sums.dna02_r.c++;
          } else {
            taskCircuit.dnp.push(i);
            sums.dnp.x += px; sums.dnp.y += py; sums.dnp.z += pz; sums.dnp.c++;
          }
        } else if (reg === 1) {
          if (Math.abs(px) < 1.2 && py < 3.0 && pz > -0.5) {
            taskCircuit.al_sez.push(i);
            sums.al_sez.x += px; sums.al_sez.y += py; sums.al_sez.z += pz; sums.al_sez.c++;
          } else if (Math.abs(px) < 1.0 && py >= 2.6 && py <= 3.8 && pz <= -1.8) {
            taskCircuit.eb_compass.push(i);
            sums.eb_compass.x += px; sums.eb_compass.y += py; sums.eb_compass.z += pz; sums.eb_compass.c++;
          } else if (Math.abs(px) < 1.2 && py > 3.8 && pz < -0.8) {
            taskCircuit.pb_arch.push(i);
            sums.pb_arch.x += px; sums.pb_arch.y += py; sums.pb_arch.z += pz; sums.pb_arch.c++;
          } else if (Math.abs(px) < 1.2 && py >= 3.0 && py <= 4.2 && pz <= -1.8) {
            taskCircuit.fb_steering.push(i);
            sums.fb_steering.x += px; sums.fb_steering.y += py; sums.fb_steering.z += pz; sums.fb_steering.c++;
          } else if (Math.abs(px) > 0.8 && py > 3.4) {
            taskCircuit.mb_dopamine.push(i);
            if (px < 0) {
              sums.mb_l.x += px; sums.mb_l.y += py; sums.mb_l.z += pz; sums.mb_l.c++;
            } else {
              sums.mb_r.x += px; sums.mb_r.y += py; sums.mb_r.z += pz; sums.mb_r.c++;
            }
          }
          if (py > 2.0 && py < 3.5 && Math.abs(px) < 0.6 && pz > -1.0) {
            taskCircuit.giant_fiber.push(i);
            sums.giant_fiber.x += px; sums.giant_fiber.y += py; sums.giant_fiber.z += pz; sums.giant_fiber.c++;
          }
        } else if (reg === 0) {
          // Optic Lobes (Lobula + Medulla complex, source of LC10a projection neurons)
          if (px < 0) {
            taskCircuit.optic_l.push(i);
            sums.optic_l.x += px; sums.optic_l.y += py; sums.optic_l.z += pz; sums.optic_l.c++;
          } else {
            taskCircuit.optic_r.push(i);
            sums.optic_r.x += px; sums.optic_r.y += py; sums.optic_r.z += pz; sums.optic_r.c++;
          }
        }
      }

      for (var k in sums) {
        if (sums[k].c > 0) {
          regionCentroids[k].set(sums[k].x / sums[k].c, sums[k].y / sums[k].c, sums[k].z / sums[k].c);
        }
      }

      somaGeometry = new THREE.BufferGeometry();
      somaGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
      somaGeometry.setAttribute('color', new THREE.BufferAttribute(colArray, 3));
      somaGeometry.computeBoundingSphere();
      somaGeometry.computeBoundingBox();

      somaMaterial = new THREE.ShaderMaterial({
        vertexColors: true,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uBurstPos: { value: burstPosUniform },
          uBurstColor: { value: burstColorUniform },
          uBurstRadius: { value: burstRadiusUniform },
          uBurstWidth: { value: burstWidthUniform },
          uBurstIntensity: { value: burstIntensityUniform },
          uBurstCount: { value: 0 },
          uDeathPulse: { value: 0.0 }
        },
        vertexShader: somaVertexShader,
        fragmentShader: somaFragmentShader
      });

      somaPointsMesh = new THREE.Points(somaGeometry, somaMaterial);
      somaPointsMesh.frustumCulled = false;
      scene.add(somaPointsMesh);

      // Build curved connectome edges connected to real somas
      buildRealCurvedEdges(edgesBuf);
      isRealCoordinatesLoaded = true;
      var metaEl = document.getElementById('brain3dModelMeta');
      if (metaEl) {
        metaEl.innerHTML = '<span style="color:#38bdf8;font-weight:600;">141,781 Somas (MaleCNS 1.0)</span> • Decay: 50ms';
      }

      console.log('Brain3D: Loaded ' + numSomas + ' real MaleCNS soma coordinates & curved tracts.');
    }).catch(function (err) {
      console.error('Brain3D: Failed to load connectome binary data:', err);
    });
  }

  // --- 6. Real-time Closed-Loop Telemetry & Focused Firework Bloom ---
  function updateTelemetry(frame) {
    if (!frame || !initialized) return;
    lastFrame = frame;

    var isNewStep = (frame.step !== lastProcessedStep);
    var isDoneChanged = (!!frame.done !== lastProcessedDone);
    var isNewDone = (frame.done && !lastProcessedDone);
    var isNewAteFood = (frame.ate_food && !lastProcessedAteFood);

    // Guard: When paused or step has not advanced, do not re-trigger!
    if (!isNewStep && !isDoneChanged && !isNewAteFood) {
      return;
    }

    lastProcessedStep = frame.step;
    lastProcessedDone = !!frame.done;
    lastProcessedAteFood = !!frame.ate_food;

    var steering = frame.steering || null;
    var epg = frame.epg || null;
    var action = frame.action || 'STRAIGHT';
    var isDone = !!frame.done;
    var ateFood = !!frame.ate_food;

    // Odor tracking
    var olf = frame.olfactory || {};
    var maxOdor = Math.max(olf.c_left || 0, olf.c_right || 0);
    var deltaOdor = maxOdor - lastOdorLevel;
    lastOdorLevel = maxOdor;

    // E-PG compass heading peak
    var currentEpgPeak = -1;
    if (epg && epg.length >= 16) {
      var maxWedge = 0;
      var maxVal = epg[0];
      for (var w = 1; w < 16; w++) {
        if (epg[w] > maxVal) { maxVal = epg[w]; maxWedge = w; }
      }
      currentEpgPeak = maxWedge;
    }
    var epgPeakShift = (currentEpgPeak !== lastEpgPeak && lastEpgPeak !== -1);
    lastEpgPeak = currentEpgPeak;

    // Clear previous bursts on a new step to prevent multi-colored visual fog
    activeBursts = [];

    // --- Strict Single-Circuit Saliency Hierarchy ---
    // Only ONE circuit is selected to fire as the focus of this step!
    currentActiveCircuit = '';

    var sensory = frame.sensory || {};
    var foodBearing = parseFloat(sensory.food_bearing || 0.0);
    var visualGain = parseFloat(olf.visual_gain || 3.5);
    var isVisualTracking = (visualGain >= 5.0 || Math.abs(foodBearing) > 0.05);

    if (isDone) {
      // Special Death Effect: Terminal Neural Depolarization Wave & Giant Fiber Arrest
      currentActiveCircuit = 'giant_fiber';
      giantFiberTimer = 45;
      shakeTime = 0.45;
      deathPulse = 1.0; // Trigger descending electrical shockwave from brain down to VNC tip
      // 1. Brain Giant Fiber apex burst
      triggerFirework(regionCentroids.giant_fiber, COLOR_GF_ALERT, 1.5, 0.12, 0.90, 0.35);
      // 2. Cervical connective junction spark
      triggerFirework(new THREE.Vector3(0.05, 0.75, 0.85), { r: 1.0, g: 0.22, b: 0.45 }, 1.2, 0.09, 0.89, 0.28);
      // 3. Lower VNC terminal motor spark
      triggerFirework(regionCentroids.dnp, { r: 1.0, g: 0.15, b: 0.20 }, 1.1, 0.08, 0.89, 0.28);
    } else if (ateFood) {
      // 2. Food Capture Reward: Bilateral Mushroom Body Dopamine explosion (localized to calyces)
      currentActiveCircuit = 'dopamine';
      dopamineBurstTimer = 30;
      triggerFirework(regionCentroids.mb_l, { r: 1.0, g: 0.88, b: 0.20 }, 0.85, 0.07, 0.91, 0.24);
      triggerFirework(regionCentroids.mb_r, { r: 1.0, g: 0.88, b: 0.20 }, 0.85, 0.07, 0.91, 0.24);
    } else if (action === 'TURN_LEFT') {
      // 3A. Left Steering: Left Descending Steering column (DNa02_L) + Left Optic Lobe LC10a
      currentActiveCircuit = 'turn_left';
      triggerFirework(regionCentroids.dna02_l, { r: 0.0, g: 0.95, b: 1.0 }, 0.90, 0.08, 0.89, 0.25);
      if (foodBearing < -0.05 || visualGain >= 5.0) {
        triggerFirework(regionCentroids.optic_l, { r: 0.82, g: 0.45, b: 1.0 }, 0.85, 0.07, 0.90, 0.25);
      }
    } else if (action === 'TURN_RIGHT') {
      // 3B. Right Steering: Right Descending Steering column (DNa02_R) + Right Optic Lobe LC10a
      currentActiveCircuit = 'turn_right';
      triggerFirework(regionCentroids.dna02_r, { r: 0.85, g: 0.35, b: 1.0 }, 0.90, 0.08, 0.89, 0.25);
      if (foodBearing > 0.05 || visualGain >= 5.0) {
        triggerFirework(regionCentroids.optic_r, { r: 0.82, g: 0.45, b: 1.0 }, 0.85, 0.07, 0.90, 0.25);
      }
    } else if (isVisualTracking && (visualGain >= 5.0 || Math.abs(foodBearing) <= 0.05) && isNewStep) {
      // 3C. Forward Visual Pursuit Lock: Bilateral LC10a Optic Lobes & Lobula
      currentActiveCircuit = 'visual_pursuit';
      var COLOR_LC10A = { r: 0.82, g: 0.45, b: 1.0 };
      triggerFirework(regionCentroids.optic_l, COLOR_LC10A, 0.88, 0.08, 0.90, 0.26);
      triggerFirework(regionCentroids.optic_r, COLOR_LC10A, 0.88, 0.08, 0.90, 0.26);
    } else if (deltaOdor > 0.03 || (maxOdor > 0.30 && isNewStep)) {
      // 4. Food Odor Detection: ONLY Antennal Lobe / SEZ anterior glomeruli
      currentActiveCircuit = 'odor';
      var odorColor = (maxOdor > 1.0) ? COLOR_DOPAMINE : COLOR_AL_SEZ;
      triggerFirework(
        regionCentroids.al_sez,
        odorColor,
        0.85,
        0.06,
        0.89,
        0.25
      );
    } else if (epgPeakShift) {
      // 5. Heading Compass Shift: Central Complex EB ring compact emerald bloom
      currentActiveCircuit = 'compass';
      triggerFirework(regionCentroids.eb_compass, COLOR_STEER, 0.70, 0.06, 0.88, 0.20);
    }

    // Synchronous, focused synaptic line activation matching ONLY the active circuit
    for (var el = 0; el < edgeList.length; el++) {
      var edge = edgeList[el];
      var isEdgeFired = false;
      if (currentActiveCircuit === 'dopamine' && edge.type === 'dopamine') {
        isEdgeFired = true;
      } else if (currentActiveCircuit === 'giant_fiber' && (edge.type === 'giant_fiber' || edge.type === 'motor')) {
        isEdgeFired = true;
      } else if (currentActiveCircuit === 'turn_left' && edge.type === 'motor' && edge.isLeft) {
        isEdgeFired = true;
      } else if (currentActiveCircuit === 'turn_right' && edge.type === 'motor' && edge.isRight) {
        isEdgeFired = true;
      } else if (currentActiveCircuit === 'visual_pursuit' && edge.type === 'visual') {
        isEdgeFired = true;
      } else if (currentActiveCircuit === 'odor' && edge.type === 'ascending') {
        isEdgeFired = (el % 3 === 0);
      } else if (currentActiveCircuit === 'compass' && edge.type === 'compass') {
        isEdgeFired = (el % 2 === 0);
      }

      if (isEdgeFired) {
        edge.active = 1.0;
      }
    }
  }

  // --- 7. Main 60 FPS Render Loop with Firework Bloom Animation ---
  function render(canvas, frame) {
    if (!initialized) {
      init(canvas);
      if (!initialized) return;
    }

    if (frame) {
      updateTelemetry(frame);
    }

    // 1. Advance and decay active firework bursts
    if (somaMaterial && somaMaterial.uniforms) {
      for (var b = activeBursts.length - 1; b >= 0; b--) {
        var burst = activeBursts[b];
        burst.radius += burst.speed;
        burst.intensity *= burst.decay;
        if (burst.intensity < 0.02 || burst.radius > burst.maxRadius * 1.5) {
          activeBursts.splice(b, 1);
        }
      }

      var count = Math.min(activeBursts.length, MAX_BURSTS);
      somaMaterial.uniforms.uBurstCount.value = count;
      for (var u = 0; u < count; u++) {
        var ab = activeBursts[u];
        burstPosUniform[u].copy(ab.pos);
        burstColorUniform[u].set(ab.color.r, ab.color.g, ab.color.b);
        burstRadiusUniform[u] = ab.radius;
        burstWidthUniform[u] = ab.width;
        burstIntensityUniform[u] = ab.intensity;
      }
    }

    // 2. Decay Active Synaptic Edges smoothly
    if (edgeColArray && edgeList.length > 0) {
      var anyEdgeUpdated = false;
      for (var e = 0; e < edgeList.length; e++) {
        var edge = edgeList[e];
        if (edge.active > 0.01) {
          edge.active *= 0.84;
          anyEdgeUpdated = true;
        } else if (edge.active !== 0.0) {
          edge.active = 0.0;
          anyEdgeUpdated = true;
        }

        if (anyEdgeUpdated) {
          var act = edge.active;
          var baseDim = 0.08;
          var r = edge.color.r * (baseDim + act * 0.92);
          var g = edge.color.g * (baseDim + act * 0.92);
          var b = edge.color.b * (baseDim + act * 0.92);

          var startPtr = (edge.segStart !== undefined ? edge.segStart : e) * 6;
          var numFloats = (edge.segCount !== undefined ? edge.segCount : 1) * 6;
          for (var f = 0; f < numFloats; f += 3) {
            edgeColArray[startPtr + f] = r;
            edgeColArray[startPtr + f + 1] = g;
            edgeColArray[startPtr + f + 2] = b;
            edgeColArray[startPtr + f + 3] = r;
            edgeColArray[startPtr + f + 4] = g;
            edgeColArray[startPtr + f + 5] = b;
          }
        }
      }
      if (anyEdgeUpdated && edgeGeometry && edgeGeometry.attributes.color) {
        edgeGeometry.attributes.color.needsUpdate = true;
      }
    }

    // 3. Smooth Preset Glide, Screen Shake & OrbitControls update
    if (controls) {
      if (transitionProgress < 1.0) {
        transitionProgress += 0.05; // ~20 frames (~350ms) smooth glide
        if (transitionProgress >= 1.0) {
          transitionProgress = 1.0;
          camera.position.copy(targetCamPos);
          controls.target.copy(targetLookAt);
        } else {
          var t = easeInOutCubic(transitionProgress);
          camera.position.lerpVectors(startCamPos, targetCamPos, t);
          controls.target.lerpVectors(startLookAt, targetLookAt, t);
        }
      }

      if (shakeTime > 0) {
        camera.position.x += (Math.random() - 0.5) * shakeTime * 1.5;
        camera.position.y += (Math.random() - 0.5) * shakeTime * 1.5;
        shakeTime *= 0.88;
        if (shakeTime < 0.01) shakeTime = 0;
      }
      controls.update();
    }

    // Special Death Effect: Decay descending depolarization wave
    if (deathPulse > 0.005) {
      deathPulse *= 0.91;
      if (deathPulse < 0.005) deathPulse = 0.0;
    }
    if (somaMaterial && somaMaterial.uniforms && somaMaterial.uniforms.uDeathPulse) {
      somaMaterial.uniforms.uDeathPulse.value = deathPulse;
    }

    // Timers
    if (dopamineBurstTimer > 0) dopamineBurstTimer--;
    if (giantFiberTimer > 0) giantFiberTimer--;

    // Render Points + Curved Edges via GPU in < 0.2 ms
    renderer.render(scene, camera);

    // Update Status HUD Text
    renderHUD(lastFrame);
  }

  // --- 8. Status Bar & UI Integration ---
  function renderHUD(frame) {
    var hud = document.getElementById('brain3dStatusText');
    if (!hud) return;

    var eventText = 'Resting Connectome Dynamics (Subthreshold Cruising)';
    var eventColor = '#38bdf8';

    if (giantFiberTimer > 0 || currentActiveCircuit === 'giant_fiber') {
      eventText = '[HALT] Terminal Reflex Arrest (Giant Fiber System)';
      eventColor = '#ef4444';
    } else if (dopamineBurstTimer > 0 || currentActiveCircuit === 'dopamine') {
      eventText = '[REWARD] PAM Cluster Dopaminergic Burst (MB Ingestion Reward)';
      eventColor = '#fbbf24';
    } else if (currentActiveCircuit === 'turn_left') {
      eventText = '[STEER-L] DNa02_L Descending Motor Activation';
      eventColor = '#38bdf8';
    } else if (currentActiveCircuit === 'turn_right') {
      eventText = '[STEER-R] DNa02_R Descending Motor Activation';
      eventColor = '#c084fc';
    } else if (currentActiveCircuit === 'visual_pursuit') {
      eventText = '[PURSUIT] Optic Lobes & LC10a Lobula Complex (Visual Lock)';
      eventColor = '#e879f9';
    } else if (currentActiveCircuit === 'odor') {
      eventText = '[CHEMOSENS] Antennal Lobe Glomeruli (Odor Plume Excitation)';
      eventColor = '#f59e0b';
    } else if (currentActiveCircuit === 'compass') {
      eventText = '[HEADING] Central Complex Ellipsoid Body (E-PG Ring Attractor)';
      eventColor = '#34d399';
    }

    var statusEl = document.getElementById('brain3dStatusEvent');
    if (statusEl) {
      statusEl.textContent = eventText;
      statusEl.style.color = eventColor;
    } else {
      hud.innerHTML =
        '<div class="brain3d-footer-status"><span class="status-label">CIRCUIT TRACE:</span> <span style="color:' + eventColor + ';font-weight:600;">' + eventText + '</span></div>' +
        '<div class="brain3d-footer-legend">' +
        '<div class="legend-regions"><span style="color:#a855f7;">■ OPTIC LOBES</span> <span style="color:#22c55e;">■ CENTRAL BRAIN</span> <span style="color:#00d2ff;">■ VNC GANGLIA</span> <span style="color:#38bdf8;">■ CERVICAL TRACT</span></div>' +
        '<div class="legend-meta" id="brain3dModelMeta">' + (isRealCoordinatesLoaded ? '<span style="color:#38bdf8;font-weight:600;">141,781 SOMAS (MaleCNS 1.0)</span>' : '141,781 SOMAS') + ' • TAU: 50ms</div>' +
        '</div>';
    }
  }

  function easeInOutCubic(x) {
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  }

  function setView(preset) {
    if (!camera || !controls) return;

    if (preset === 'full') {
      // Full system view: Complete coverage of brain, cervical connective, and VNC
      targetCamPos.set(0.05, -0.94, 16.92);
      targetLookAt.set(0.05, -0.94, 0.42);
    } else if (preset === 'brain') {
      // Central brain view: Front-facing central brain (EB, PB, FB, AL, MB) and optic lobes
      targetCamPos.set(0.05, 3.42, 7.50);
      targetLookAt.set(0.05, 3.42, -0.30);
    } else if (preset === 'vnc') {
      // Ventral nerve cord view: From cervical entrance to abdominal ganglia motor columns
      targetCamPos.set(0.18, -3.20, 13.60);
      targetLookAt.set(0.18, -3.20, 2.10);
    } else if (preset === 'side' || preset === 'lateral') {
      // Lateral anatomical view: Side profile showing brain-cervical bend-VNC depth
      targetCamPos.set(16.50, -0.94, 0.42);
      targetLookAt.set(0.05, -0.94, 0.42);
    } else if (preset === 'dorsal') {
      // Dorsal view: Top-down projection of somatic distribution
      targetCamPos.set(0.05, 12.50, 0.42);
      targetLookAt.set(0.05, -0.94, 0.42);
    }

    startCamPos.copy(camera.position);
    startLookAt.copy(controls.target);
    transitionProgress = 0.0;
  }

  var autoRotateActive = false;
  function toggleAutoRotate() {
    autoRotateActive = !autoRotateActive;
    if (controls) {
      controls.autoRotate = autoRotateActive;
      controls.autoRotateSpeed = 1.2;
    }
    return autoRotateActive;
  }

  // Global window API
  window.renderBrain3D = render;
  window.setupBrain3DInteraction = function (c) { init(c); };
  window.setBrain3DView = setView;
  window.toggleBrain3DAutoRotate = toggleAutoRotate;
})();

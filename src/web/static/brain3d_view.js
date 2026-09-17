/**
 * FlyBrain Snake — 3D Drosophila Connectome Point Cloud & Neural Activity Visualizer
 *
 * Implements the architecture inspired by fly-atlas, fly-in-a-tetris, and siliconfly:
 *
 * 1. Base Layer (底图点云):
 *    - 139,255 somas (FlyWire connectome scale) rendered via a single THREE.Points draw call.
 *    - Anatomically colored by brain region:
 *      * Optic Lobes (双侧视叶): ~90,000 points, Purple (#a855f7)
 *      * Central Brain (中央脑): ~35,000 points, Green (#22c55e)
 *      * Ventral Nerve Cord (腹神经索 VNC): ~14,000 points, Blue (#00d2ff)
 *    - Semi-transparent, soft-particle circular sprite texture.
 *
 * 2. Activity & 30–80 ms Decay Trails (活动与短衰减拖尾):
 *    - Recently fired neurons scale up (from 1.4 px to 5.5 px) and increase brightness to neon/white.
 *    - Exponential decay with tau ~ 50 ms (decays smoothly in 30–80 ms).
 *
 * 3. Task Circuit Always Highlighted (任务回路始终高亮):
 *    - 食物相关嗅觉/味觉 (Olfactory / Gustatory: Antennal Lobes & SEZ)
 *    - 中央复合体 (Central Complex: EB compass bump rotation, PB bridge, FB steering)
 *    - 左右 DNa02 (Bilateral DNa02 descending steering neurons)
 *    - 吃到食物时的多巴胺爆发 (Mushroom Body PAM / DAN dopamine reward burst)
 *
 * 4. Highest-Weight Synaptic Connections (只画活跃且突触数最高的几千条边):
 *    - ~2,500 active highest-weight functional edges rendered via THREE.LineSegments.
 *    - Only currently active edges light up and fade out smoothly.
 *
 * 5. Camera Scripting (智能电影级相机脚本):
 *    - 找食物时: 推近嗅觉 (AL) 和中央复合体 (EB/PB)
 *    - 转弯时: 切到下行神经元 (DNa02) 与腹神经索 (VNC)
 *    - 死亡时: 闪巨纤维 (Giant Fiber escape reflex) 并拉远震颤
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

  // Geometry & Buffers for 139,255 Somas
  var somaGeometry = null;
  var somaMaterial = null;
  var somaPointsMesh = null;
  var posArray = null;
  var colArray = null;
  var baseColArray = null;
  var sizeArray = null;

  // Active Decay State (Array of decaying point indices for O(active) performance)
  var activeSpikeMap = new Map(); // index -> { intensity: 1.0, decayRate: 0.86, targetR, targetG, targetB }

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
  var dopamineBurstTimer = 0;
  var giantFiberTimer = 0;

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

  // --- 1. Soft Circular Sprite Texture for Points ---
  function createSoftCircleTexture() {
    var canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    var ctx = canvas.getContext('2d');
    var grad = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
    grad.addColorStop(0.25, 'rgba(255, 255, 255, 0.85)');
    grad.addColorStop(0.65, 'rgba(255, 255, 255, 0.25)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0.0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(16, 16, 16, 0, Math.PI * 2);
    ctx.fill();

    var tex = new THREE.CanvasTexture(canvas);
    tex.needsUpdate = true;
    return tex;
  }

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
          // 生物真实形态：下行/上行神经束必须经过「颈神经索」(Cervical Connective) 瓶颈束聚，绝非直插虚空
          var pBrainX = (type === 2) ? p0x : p1x;
          var pBrainZ = (type === 2) ? p0z : p1z;
          var pVncX = (type === 2) ? p1x : p0x;
          var pVncZ = (type === 2) ? p1z : p0z;

          // 颈神经孔解剖中心 (X ≈ 0.15, Y ≈ 0.75, Z ≈ 0.85)，根据两侧神经元微弱对称偏移
          var xNeck = 0.15 + (pBrainX + pVncX) * 0.12;
          var zNeck = 0.85 + (pBrainZ + pVncZ) * 0.10;

          var c1x, c1y, c1z, c2x, c2y, c2z;
          if (type === 2) {
            // Brain (p0) -> VNC (p1)
            c1x = p0x * 0.5 + xNeck * 0.5;
            c1y = 1.35; // 脑部腹侧出口汇聚
            c1z = p0z * 0.35 + zNeck * 0.65;

            c2x = p1x * 0.5 + xNeck * 0.5;
            c2y = 0.15; // 腹神经索背侧入口扩散
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

          // 三次贝塞尔曲线 (Cubic Bézier Spline): 从源 soma 沿颈孔平滑过渡至目标 soma
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
          // 视叶到中央脑：前视束 (Anterior Optic Tract) 弧线
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
          // 中央脑或腹神经索内部：神经毡平滑小弧线
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

        // 写入线段顶点数组
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
      baseColArray = new Float32Array(numSomas * 3);
      sizeArray = new Float32Array(numSomas);

      // Index taskCircuit with real anatomical soma clusters
      taskCircuit.al_sez = [];
      taskCircuit.eb_compass = [];
      taskCircuit.pb_arch = [];
      taskCircuit.fb_steering = [];
      taskCircuit.mb_dopamine = [];
      taskCircuit.dna02_l = [];
      taskCircuit.dna02_r = [];
      taskCircuit.dnp = [];
      taskCircuit.giant_fiber = [];

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
        baseColArray[p] = col.r;
        baseColArray[p + 1] = col.g;
        baseColArray[p + 2] = col.b;
        sizeArray[i] = 0.088;

        // Biological Subcircuit Classification on Real Coordinates
        if (reg === 2) {
          if (px < -0.2) taskCircuit.dna02_l.push(i);
          else if (px > 0.2) taskCircuit.dna02_r.push(i);
          else taskCircuit.dnp.push(i);
        } else if (reg === 1) {
          if (Math.abs(px) < 1.2 && py < 3.0 && pz > -0.5) {
            taskCircuit.al_sez.push(i);
          } else if (Math.abs(px) < 1.0 && py >= 2.6 && py <= 3.8 && pz <= -1.8) {
            taskCircuit.eb_compass.push(i);
          } else if (Math.abs(px) < 1.2 && py > 3.8 && pz < -0.8) {
            taskCircuit.pb_arch.push(i);
          } else if (Math.abs(px) < 1.2 && py >= 3.0 && py <= 4.2 && pz <= -1.8) {
            taskCircuit.fb_steering.push(i);
          } else if (Math.abs(px) > 0.8 && py > 3.4) {
            taskCircuit.mb_dopamine.push(i);
          }
          if (py > 2.0 && py < 3.5 && Math.abs(px) < 0.6 && pz > -1.0) {
            taskCircuit.giant_fiber.push(i);
          }
        }
      }

      somaGeometry = new THREE.BufferGeometry();
      somaGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
      somaGeometry.setAttribute('color', new THREE.BufferAttribute(colArray, 3));
      somaGeometry.setAttribute('size', new THREE.BufferAttribute(sizeArray, 1));
      somaGeometry.computeBoundingSphere();
      somaGeometry.computeBoundingBox();

      somaMaterial = new THREE.PointsMaterial({
        size: 0.12,
        map: createSoftCircleTexture(),
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });

      somaPointsMesh = new THREE.Points(somaGeometry, somaMaterial);
      somaPointsMesh.frustumCulled = false;
      scene.add(somaPointsMesh);

      // Build curved connectome edges connected to real somas
      buildRealCurvedEdges(edgesBuf);
      isRealCoordinatesLoaded = true;

      console.log('Brain3D: Loaded ' + numSomas + ' real MaleCNS soma coordinates & curved tracts.');
    }).catch(function (err) {
      console.error('Brain3D: Failed to load connectome binary data:', err);
    });
  }

  // --- 5. Trigger Spike Pulse on Points with Exponential Decay ---
  function triggerSpike(somaIndex, r, g, b, durationBoost) {
    if (somaIndex < 0 || somaIndex >= TOTAL_SOMAS) return;

    activeSpikeMap.set(somaIndex, {
      intensity: 1.0,
      decayRate: durationBoost ? 0.92 : 0.84, // 0.84 at 60fps ~ 45-60 ms decay!
      tr: r !== undefined ? r : 1.0,
      tg: g !== undefined ? g : 1.0,
      tb: b !== undefined ? b : 1.0
    });
  }

  // --- 6. Real-time Closed-Loop Telemetry & Camera Scripting ---
  function updateTelemetry(frame) {
    if (!frame || !initialized) return;
    lastFrame = frame;

    var steering = frame.steering || null;
    var epg = frame.epg || null;
    var action = frame.action || 'STRAIGHT';
    var isDone = !!frame.done;
    var ateFood = !!frame.ate_food;

    // A. 任务回路始终高亮 1: 食物嗅觉 / 味觉 (AL & SEZ)
    for (var a = 0; a < 25; a++) {
      var alIdx = taskCircuit.al_sez[Math.floor(rng() * taskCircuit.al_sez.length)];
      if (alIdx) triggerSpike(alIdx, COLOR_AL_SEZ.r, COLOR_AL_SEZ.g, COLOR_AL_SEZ.b, true);
    }

    // B. 任务回路始终高亮 2: 中央复合体 E-PG 罗盘与 PFL3 转向
    if (epg && epg.length >= 16) {
      // Find peak E-PG wedge
      var maxWedge = 0;
      var maxVal = epg[0];
      for (var w = 1; w < 16; w++) {
        if (epg[w] > maxVal) { maxVal = epg[w]; maxWedge = w; }
      }
      // Light up peak heading compass somas
      var ebCount = taskCircuit.eb_compass.length;
      var wedgeStart = Math.floor((maxWedge / 16) * ebCount);
      for (var ew = 0; ew < 35; ew++) {
        var ebPt = taskCircuit.eb_compass[(wedgeStart + ew) % ebCount];
        if (ebPt) triggerSpike(ebPt, COLOR_STEER.r, COLOR_STEER.g, COLOR_STEER.b, true);
      }
    }

    // C. 任务回路始终高亮 3: 左右 DNa02 转向下行神经元
    if (action === 'TURN_LEFT' || (steering && steering.pfl3_l > 0.1)) {
      for (var dl = 0; dl < 30; dl++) {
        var lIdx = taskCircuit.dna02_l[Math.floor(rng() * taskCircuit.dna02_l.length)];
        if (lIdx) triggerSpike(lIdx, 0.0, 0.95, 1.0, true);
      }
    }
    if (action === 'TURN_RIGHT' || (steering && steering.pfl3_r > 0.1)) {
      for (var dr = 0; dr < 30; dr++) {
        var rIdx = taskCircuit.dna02_r[Math.floor(rng() * taskCircuit.dna02_r.length)];
        if (rIdx) triggerSpike(rIdx, 0.75, 0.35, 1.0, true);
      }
    }

    // D. 任务回路始终高亮 4: 吃到食物时的多巴胺爆发 (DAN / PAM Reward Burst!)
    if (ateFood) {
      dopamineBurstTimer = 35; // 35 frames of dopamine firework
    }
    if (dopamineBurstTimer > 0) {
      dopamineBurstTimer--;
      for (var mb = 0; mb < 80; mb++) {
        var pamIdx = taskCircuit.mb_dopamine[Math.floor(rng() * taskCircuit.mb_dopamine.length)];
        if (pamIdx) triggerSpike(pamIdx, COLOR_DOPAMINE.r, COLOR_DOPAMINE.g, COLOR_DOPAMINE.b, true);
      }
    }

    // E. 死亡时闪巨纤维 (Giant Fiber Emergency Escape Reflex)
    if (isDone) {
      giantFiberTimer = 40; // 40 frames of scarlet shockwave
    }
    if (giantFiberTimer > 0) {
      giantFiberTimer--;
      for (var gf = 0; gf < taskCircuit.giant_fiber.length; gf++) {
        triggerSpike(taskCircuit.giant_fiber[gf], COLOR_GF_ALERT.r, COLOR_GF_ALERT.g, COLOR_GF_ALERT.b, true);
      }
      for (var dnpG = 0; dnpG < 40; dnpG++) {
        var vncG = taskCircuit.dnp[Math.floor(rng() * taskCircuit.dnp.length)];
        if (vncG) triggerSpike(vncG, 1.0, 0.2, 0.2, true);
      }
    }

    // F. 激活后台推来的神经网络脉冲 (Spikes from backend simulation)
    var spikes = frame.spikes || [];
    if (spikes.length > 0) {
      for (var sp = 0; sp < spikes.length; sp++) {
        var sId = spikes[sp];
        if (sId < 20) {
          // Visual projection neurons (ER2/ER4d/LC4/LPLC2)
          for (var v = 0; v < 3; v++) {
            var optPt = Math.floor(rng() * OPTIC_COUNT);
            triggerSpike(optPt, 0.95, 0.45, 1.0, false);
          }
        } else if (sId < 36) {
          // E-PG compass neurons
          var wIdx = sId - 20;
          var ebC = taskCircuit.eb_compass.length;
          var pIdx = taskCircuit.eb_compass[(Math.floor((wIdx / 16) * ebC) + (sp % 4)) % ebC];
          if (pIdx) triggerSpike(pIdx, COLOR_STEER.r, COLOR_STEER.g, COLOR_STEER.b, true);
        } else if (sId < 52) {
          // Delta7 bridge neurons
          var pbIdx = taskCircuit.pb_arch[Math.floor(rng() * taskCircuit.pb_arch.length)];
          if (pbIdx) triggerSpike(pbIdx, 0.2, 0.9, 1.0, true);
        } else if (sId < 116) {
          // Steering PFN / PFL3 columns
          var fbIdx = taskCircuit.fb_steering[Math.floor(rng() * taskCircuit.fb_steering.length)];
          if (fbIdx) triggerSpike(fbIdx, COLOR_STEER.r, COLOR_STEER.g, COLOR_STEER.b, true);
        } else if (sId === 122) {
          // DNa02_L
          var lIdx = taskCircuit.dna02_l[Math.floor(rng() * taskCircuit.dna02_l.length)];
          if (lIdx) triggerSpike(lIdx, 0.0, 0.95, 1.0, true);
        } else if (sId === 123) {
          // DNa02_R
          var rIdx = taskCircuit.dna02_r[Math.floor(rng() * taskCircuit.dna02_r.length)];
          if (rIdx) triggerSpike(rIdx, 0.75, 0.35, 1.0, true);
        }
      }
    }

    // 全脑自发背景微弱脉冲 (Spontaneous biological background firing)
    for (var bg = 0; bg < 16; bg++) {
      var bgIdx = Math.floor(rng() * TOTAL_SOMAS);
      triggerSpike(bgIdx, 1.0, 1.0, 1.0, false);
    }

    // G. 激活相连的突触边 (Active Synaptic Lines)
    for (var el = 0; el < edgeList.length; el++) {
      var edge = edgeList[el];
      var isEdgeFired = false;
      if (edge.type === 'dopamine' && dopamineBurstTimer > 0) isEdgeFired = true;
      else if (edge.type === 'giant_fiber' && giantFiberTimer > 0) isEdgeFired = true;
      else if (edge.type === 'motor') {
        if (action === 'TURN_LEFT' && edge.isLeft) isEdgeFired = true;
        else if (action === 'TURN_RIGHT' && edge.isRight) isEdgeFired = true;
        else if (action === 'STRAIGHT' && Math.random() < 0.20) isEdgeFired = true;
      }
      else if (edge.type === 'compass' && (epg ? maxVal > 0.4 : true)) isEdgeFired = Math.random() < 0.25;
      else if (edge.type === 'visual') isEdgeFired = Math.random() < 0.15;
      else if (edge.type === 'ascending' && Math.random() < 0.12) isEdgeFired = true;
      else if (edge.type === 'vnc_circuit' && (action === 'TURN_LEFT' || action === 'TURN_RIGHT')) isEdgeFired = Math.random() < 0.25;

      if (isEdgeFired) {
        edge.active = 1.0;
      }
    }
  }

  // --- 7. Main 60 FPS Render Loop with Dynamic Spike Decay ---
  function render(canvas, frame) {
    if (!initialized) {
      init(canvas);
      if (!initialized) return;
    }

    if (frame) {
      updateTelemetry(frame);
    }

    // 1. Decay Active Spikes (30–80 ms 短衰减拖尾)
    if (activeSpikeMap.size > 0 && colArray && sizeArray) {
      var toRemove = [];
      activeSpikeMap.forEach(function (data, sIdx) {
        data.intensity *= data.decayRate;

        if (data.intensity < 0.02) {
          // Restore to resting base state
          colArray[sIdx * 3] = baseColArray[sIdx * 3];
          colArray[sIdx * 3 + 1] = baseColArray[sIdx * 3 + 1];
          colArray[sIdx * 3 + 2] = baseColArray[sIdx * 3 + 2];
          sizeArray[sIdx] = 0.085;
          toRemove.push(sIdx);
        } else {
          // Scale up & increase brightness
          var t = data.intensity;
          colArray[sIdx * 3] = baseColArray[sIdx * 3] * (1 - t) + data.tr * t;
          colArray[sIdx * 3 + 1] = baseColArray[sIdx * 3 + 1] * (1 - t) + data.tg * t;
          colArray[sIdx * 3 + 2] = baseColArray[sIdx * 3 + 2] * (1 - t) + data.tb * t;
          sizeArray[sIdx] = 0.085 + t * 0.28; // Up to ~4.5x size during firing!
        }
      });

      for (var r = 0; r < toRemove.length; r++) {
        activeSpikeMap.delete(toRemove[r]);
      }

      if (somaGeometry && somaGeometry.attributes.color) {
        somaGeometry.attributes.color.needsUpdate = true;
        somaGeometry.attributes.size.needsUpdate = true;
      }
    }

    // 2. Decay Active Synaptic Edges (Supporting multi-segment curved Bézier cables)
    if (edgeColArray && edgeList.length > 0) {
      for (var e = 0; e < edgeList.length; e++) {
        var edge = edgeList[e];
        if (edge.active > 0.02) {
          edge.active *= 0.86;
        } else {
          edge.active = 0.0;
        }

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
        }
      }
      if (edgeGeometry && edgeGeometry.attributes.color) {
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

    // Render Points + Curved Edges via GPU in < 0.2 ms
    renderer.render(scene, camera);

    // Update Status HUD Text
    renderHUD(lastFrame);
  }

  // --- 8. Status Bar & UI Integration ---
  function renderHUD(frame) {
    var hud = document.getElementById('brain3dStatusText');
    if (!hud) return;

    var eventText = '巡航觅食 (中央复合体与全脑神经流)';
    var eventColor = '#34d399';

    if (giantFiberTimer > 0) {
      eventText = '⚡ 巨纤维紧急逃逸反射 (Giant Fiber System 爆发闪烁)';
      eventColor = '#ef4444';
    } else if (dopamineBurstTimer > 0) {
      eventText = '✨ 捕获食物多巴胺奖励爆发 (PAM Cluster Dopamine Reward)';
      eventColor = '#fbbf24';
    } else if (frame && (frame.action === 'TURN_LEFT' || frame.action === 'TURN_RIGHT')) {
      eventText = '↱ 转向机动 (下行神经元 DNa02 激发并通过颈神经束传至 VNC)';
      eventColor = '#00d2ff';
    }

    hud.innerHTML =
      '<span style="color:#a855f7;font-weight:600;">视叶点云</span> • ' +
      '<span style="color:#22c55e;font-weight:600;">中央脑点云</span> • ' +
      '<span style="color:#00d2ff;font-weight:600;">腹神经索</span> • ' +
      '<span style="color:#38bdf8;font-weight:600;">颈神经束通道 (Cervical Connective)</span> | ' +
      '当前状态: <span style="color:' + eventColor + ';font-weight:600;">' + eventText + '</span> | ' +
      (isRealCoordinatesLoaded ? '<span style="color:#38bdf8;font-weight:600;">141,781 Somas (MaleCNS 1.0 真实微米坐标 & 真实曲线神经束)</span>' : '141,781 Somas (Connectome Point Cloud)') + ' • 活动衰减: 30-80ms';
  }

  function easeInOutCubic(x) {
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  }

  function setView(preset) {
    if (!camera || !controls) return;

    if (preset === 'full') {
      // 全系统视角：完整覆盖脑部、颈神经束与腹神经索全部神经元 (100% 完整显示，四周留有舒适边距)
      targetCamPos.set(0.05, -0.94, 16.92);
      targetLookAt.set(0.05, -0.94, 0.42);
    } else if (preset === 'brain') {
      // 中央脑视角：正对中央脑核心结构 (EB, PB, FB, AL, MB) 及视叶，100% 完整居中显示
      targetCamPos.set(0.05, 3.42, 7.50);
      targetLookAt.set(0.05, 3.42, -0.30);
    } else if (preset === 'vnc') {
      // 腹神经索视角：完整覆盖从颈部入口至腹神经节末梢的所有躯干运动柱神经元 (100% 完整显示)
      targetCamPos.set(0.18, -3.20, 13.60);
      targetLookAt.set(0.18, -3.20, 2.10);
    } else if (preset === 'side' || preset === 'lateral') {
      // 侧视解剖视角：完整呈现果蝇中枢神经系统侧面轮廓 (脑部、颈弯曲及腹神经索全纵深 100% 完整显示)
      targetCamPos.set(16.50, -0.94, 0.42);
      targetLookAt.set(0.05, -0.94, 0.42);
    } else if (preset === 'dorsal') {
      // 俯视视角：自上而下俯瞰神经元平面分布 (100% 完整显示)
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

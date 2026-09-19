# FlyBrain Snake: Connectome-in-the-Loop Sensorimotor System

[![Connectome](https://img.shields.io/badge/Connectome-MaleCNS%20v1.0-blue.svg)](https://codex.flywire.ai)
[![Neurons](https://img.shields.io/badge/Neurons-166%2C700%20Spiking%20LIF-green.svg)](https://github.com/alextitonis/flybrain)
[![Synapses](https://img.shields.io/badge/Synapses-125M%20Dataset%20%7C%20~25.6M%20Runtime-purple.svg)](https://codex.flywire.ai)
[![License](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)

[**English**](README.md) | [**中文说明**](README_zh.md)

An embodied computational neuroscience exploration project coupling adult *Drosophila melanogaster* whole-brain connectome dynamics ([`flybrain`](https://github.com/alextitonis/flybrain)) with closed-loop sensorimotor navigation, peripheral clearance reflexes, and real-time neuro-telemetry HUD.

---

## 🏛️ System Architecture: Connectome-in-the-Loop Framework

The project is structured into a three-layer hierarchical sensorimotor architecture that reflects the classical division of labor in insect motor control (supraspinal orientation filtering + peripheral thoracic clearance reflexes). Closed-loop turning is governed exclusively by descending `DNa02` differentials and the clearance FSM, while Central Complex (CX) instruments serve as read-only telemetry monitors:

1. **Whole-Brain Dynamics Layer / Descending Intent Layer (Supraspinal Filter)**:  
   Simulated via a 4-step leaky integrate-and-fire (LIF) network across the **MaleCNS v1.0** connectome (166,700 neurons, ~25.58M runtime sparse synaptic edges). Driven by decoupled sensory inputs (encoder-side odor-gated visual pursuit), it functions as a biological nonlinear dynamical filter whose bilateral descending output (**`DNa02`**) conveys directional steering intent. At the start of each decision tick, membrane potentials are reset via `v.fill(0)`, eliminating cross-tick motor hysteresis so that each 4-step window operates as a clean, episodic dynamical filter without cross-tick memory leakage.
2. **Lower Reflexive Layer (VNC-Analogous Clearance FSM)**:  
   Functionally analogous to the insect **Ventral Nerve Cord (VNC)** and peripheral reflex arcs. A deterministic clearance finite-state machine (FSM) safeguards basic survival:
   - **Frontal Collision Override**: When facing an immediate fatal obstacle (`dist_front == 0`), it executes an emergency reflexive override to deflect away;
   - **Lateral Clearance Guard**: When contiguous with a lateral obstacle (`dist_left == 0` or `dist_right == 0`), it actively suppresses descending turn commands into that obstacle.
3. **Spatial Interface Layer (Sensory Transduction)**:  
   Serves as **virtual compound eyes and antennae**, translating 2D arena spatial geometry into biophysical voltage depolarizations (`brain.stimulate()`):
   - Antennal Lobe projection neurons (`AL`, DM1/VA2/DM2): bilateral chemical contrast $\Delta c$;
   - Lobula Columnar neurons (`LC10a`): target retinal azimuth $\theta_{\text{bearing}}$;
   - Looming proximity neurons (`LC4` / `LPLC2`): near-field obstacle voltage stimulation.

```text
======================= CLOSED-LOOP CONTROL PIPELINE =======================

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                      1. SPATIAL INTERFACE LAYER                         │
 │   Virtual Eyes (Ommatidia) ──► Target Bearing θ (LC10a) & Looming (LC4) │
 │   Virtual Antennae         ──► Bilateral Odor Contrast Δc (AL)          │
 └──────────────────────────────────┬──────────────────────────────────────┘
                                    │ Membrane Potential Depolarizations
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │        2. WHOLE-BRAIN DYNAMICS LAYER (MaleCNS 166.7k, ~25.6M Synapses)  │
 │                                                                         │
 │  Chemosensory (AL) + Odor-Gated Visual Pursuit (LC10a) + Looming (LC4)  │
 │                                  │                                      │
 │                                  ▼ (4-Step Whole-Brain LIF, v.fill(0))  │
 │  Descending Command Neurons: Bilateral Potential Diff (DNa02_L - DNa02_R│
 └──────────────────────────────────┬──────────────────────────────────────┘
                                    │ Descending Steering Intent
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │           3. LOWER REFLEXIVE LAYER (VNC-ANALOGOUS CLEARANCE FSM)        │
 │                                                                         │
 │   Clear Path (dist_front > 0)    ──► Execute DNa02 Steering / DNa01 Crawl│
 │   Front Hazard (dist_front == 0) ──► Emergency Reflexive Override (Turn)│
 │   Side Obstacle (dist_lat == 0)  ──► Suppress Turn into Adjacent Wall   │
 └──────────────────────────────────┬──────────────────────────────────────┘
                                    │ Motor Maneuver
                                    ▼
                         [ 2D ARENA EMBODIMENT ]

=================== READ-ONLY NEURO-TELEMETRY HUD ====================
(Extracted alongside simulation for live monitoring; does not drive actions)

  • E-PG Heading Ring : 16-wedge schematic reconstructed from current heading
  • PFL3 Comparator   : Fan-shaped Body bilateral balance gauge
  • PAM Dopamine      : Ingestion reward transient flash on food consumption
  • 3D Connectome View: WebGL event-triggered illumination of anatomical centroids
```

---

## 🧬 Sensorimotor Circuits & Signal Pathways

1. **Sensory Channel Decoupling**:
   - **Antennal Lobe (`AL`, DM1/VA2/DM2)**: Transduces exclusively bilateral chemical contrast $\Delta c = c_L - c_R$, free from retinal coordinates.
   - **Lobula Columnar (`LC10a`)**: Encodes pure retinal target yaw angle $\theta_{\text{bearing}}$ with a calibrated deadband ($\pm 0.05\text{ rad} \approx 2.9^\circ$).
   - **Near-field Looming (`LC4`/`LPLC2`)**: Depolarizes optical looming neurons when clearance $\le 1$.
2. **Encoder-Side Odor-Gated Visual Sensitization**:
   - Encodes sensory sensitization analogous to biological odor-gated visual pursuit:
     $$\text{visual\_gain} = \text{base\_gain} \times \left(1.0 + \alpha \cdot \frac{c_{\max}}{c_0}\right)$$
   - Smooth exploratory foraging in clean air (~4.5) transitioning to acute high-gain orienting saccades (~14.0) near food.
3. **Descending Motor Readout**:
   - Steering actions follow the bilateral membrane potential differential of descending command neurons **`DNa02`** ($\text{diff} = \text{DNa02\_L} - \text{DNa02\_R} > 0.03$), without constant left/right asymmetric bias (zero lateral bias; interface deadband $0.05\text{ rad}$, threshold $0.03$, and gain multipliers remain as explicit interface parameters).
   - Forward crawl baseline is coordinated through `DNa01`.
   - `brain.v.fill(0)` resets potential each tick to prevent inter-step motor drift.
4. **VNC-Analogous Clearance FSM**:
   - Evaluates immediate obstacle contact (`dist_front == 0`) using local grid clearance, triggering `⚡ REFLEX OVERRIDE (SPINAL COLLISION GUARD)` on the telemetry HUD when deflecting away from imminent wall crashes;
   - Evaluates lateral clearance (`dist_left == 0` or `dist_right == 0`) to suppress turning into adjacent obstructions.
5. **Read-Only Neuro-Telemetry HUD (Non-Decisional)**:
   - **E-PG Heading Display**: 16-wedge polar visualization reconstructed from current snake heading angle.
   - **PFL3 Bilateral Steering Gauge**: Read-only monitor of left vs. right Fan-shaped Body premotor potentials.
   - **PAM Dopaminergic Bursts**: Ingestion reward transients upon eating food.
   - **3D Whole-Brain View**: WebGL view with event-triggered regional illumination of anatomical centroids (Optic Lobes, Central Brain, VNC).

---

## 🚀 Quick Start

### 1. Requirements & Launch

The project runs with minimal dependencies (`flybrain` and `pytest`). The web server uses Python's built-in standard library `ThreadingHTTPServer` with Server-Sent Events (SSE).

```bash
# Install dependencies
pip install -r requirements.txt

# Launch simulation server
python3 run_server.py
```

Open your browser at [http://localhost:8080](http://localhost:8080).

### 2. Interactive Controls

- `Space`: Pause / Resume autonomous simulation
- `S`: Single-step advance (one 80 ms simulation window)
- `R`: Reset game arena and neural circuits
- `O`: Toggle 2D food odor diffusion plume
- Click + Drag / Mouse Wheel: Orbit, pan, and zoom the 3D connectome

---

## 🧪 Tests & Benchmark

```bash
# Run the automated test suite (65 passing unit & integration tests)
pytest

# Run the 50-episode sensorimotor benchmark
python3 -m src.benchmark.runner --episodes 50
```

---

## 📚 Technical Documentation

- [**Neural Circuits & Architecture Guide (EN)**](docs/NEURAL_CIRCUITS.md): In-depth connectomic sensorimotor pathways, odor-gated visual pursuit, and descending motor decoding.
- [**神经回路与工作原理指南 (中文)**](docs/NEURAL_CIRCUITS_zh.md): 详尽的果蝇全中枢连接组回路映射、多模态气味门控视觉追踪与下行控制说明。
- [**中文版项目说明 (README_zh.md)**](README_zh.md): 针对中文读者的完整项目介绍与三层分工指南。

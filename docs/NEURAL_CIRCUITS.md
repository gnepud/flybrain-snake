# FlyBrain Snake: Connectome Sensorimotor Circuits & Multimodal Odor-Gated Visual Pursuit Guide

> **System Architecture & Framework**:  
> **"This project explores a canonical Connectome-in-the-Loop Sensorimotor System:"**
> 
> 1. **Whole-Brain Dynamics Layer / Descending Intent Layer (Supraspinal Filter)**:  
>    Simulated via a 4-step leaky integrate-and-fire (LIF) network across the **MaleCNS v1.0** connectome (166,700 reconstructed neurons; dataset contains 125M synapses, runtime sparse array contains ~25.58M connections). Driven by decoupled sensory inputs (odor-gated visual pursuit), it acts as a nonlinear biological dynamics filter whose bilateral descending output (**`DNa02`**) spontaneously expresses steering intent. At the start of each decision tick, membrane potentials are reset via `brain.v.fill(0)` to prevent inter-step motor hysteresis (clean episodic settling with no cross-tick memory leakage).  
> 2. **Lower Reflexive Layer (VNC-Analogous Clearance FSM)**:  
>    Functionally analogous to the insect **Ventral Nerve Cord (VNC)** and peripheral reflex arcs. A deterministic clearance finite-state machine (FSM) safeguards basic survival: executing emergency reflexive overrides upon frontal hazards (`dist_front == 0`) and suppressing descending turn commands when contiguous with lateral obstacles (`dist_left/right == 0`).  
> 3. **Spatial Interface Layer (Sensory Transduction)**:  
>    Serves as **virtual compound eyes and antennae**, transducing 2D arena spatial geometry into biophysical voltage depolarizations (`brain.stimulate()`):
>    - Antennal Lobe projection neurons (`AL`, DM1/VA2/DM2): chemical contrast $\Delta c$;
>    - Lobula Columnar neurons (`LC10a`): target retinal azimuth $\theta_{\text{bearing}}$;
>    - Looming proximity neurons (`LC4` / `LPLC2`): near-field obstacle voltage stimulation.
>
> This tripartite division of labor reflects insect sensorimotor physiology: Central Complex (CX) instruments serve as read-only telemetry monitors, while closed-loop steering decisions come exclusively from descending `DNa02` differentials and the clearance FSM. Whole-brain computations run across 166.7k neurons through a 4-step LIF network with zero constant lateral bias, while the peripheral clearance FSM guards survival boundaries.

---

## 1. System Circuit Schematic & Signal Flow

```text
======================= CLOSED-LOOP SENSORIMOTOR PIPELINE =======================

[ 1. Spatial Interface Layer (Virtual Sensory Transduction) ]
  Bilateral Antenna Odor Sampling: c_L, c_R
  Bilateral Chemical Contrast: Δc = c_L − c_R ──► Antennal Lobe PNs (DM1/VA2/DM2)
  Egocentric Retinal Bearing: θ_bearing       ──► Lobula Columnar Neurons (LC10a)
  Near-Field Obstacle Hazard (Clearance <= 1) ──► Looming Neurons (LC4 / LPLC2)

[ 2. Encoder-Side Sensitization / Odor Gating ]
  Encountered Odor Peak: c_max = max(c_L, c_R)
  Dynamic Sensitization Gain (Analogous to Odor-Gated Visual Pursuit):
      visual_gain = base_gain × (1.0 + α · c_max / c_0)
      (α = 1.5, c_0 = 5.0, base_gain = 3.5; up to ~4× amplification near food)

[ 3. Pure Retinal Visual Channel (Lobula Columnar LC10a) ]
  Left Retinal Field (θ_bearing < −0.05): stim_vis_L = min(8, |θ_bearing| × visual_gain)
  Right Retinal Field (θ_bearing > +0.05): stim_vis_R = min(8, |θ_bearing| × visual_gain)
        └─► LC10a (~100+ per hemisphere) ──► AOTU ──► Drives Orienting Saccades

[ 4. Whole-Brain 4-Step LIF Dynamics & Descending Motor Readout ]
  DNa02_L / DNa02_R (1 bilateral pair, ipsilateral steering control)
      Signals: DNa02_L = V_L / 2.0,  DNa02_R = V_R / 2.0
      Readout Rule: diff = DNa02_L − DNa02_R
      Threshold: diff > 0.03 → TURN_LEFT; −diff > 0.03 → TURN_RIGHT (Zero lateral bias)
  DNa01_L / DNa01_R (1 bilateral pair, forward locomotor command neurons)
      Sustained baseline crawl drive (STRAIGHT); ΔCₜ monitored as temporal approach trend

[ 5. Lower Reflexive Layer (VNC-Analogous Clearance FSM) ]
  Path Clear (dist_front > 0)     ──► Follow DNa02 Steering Intent / DNa01 Crawl
  Immediate Hazard (dist_front == 0) ──► Reflexive Turn Override away from obstruction

=================== READ-ONLY NEURO-TELEMETRY HUD ===================
(Monitored alongside simulation for scientific visualization; non-decisional)

  • E-PG Heading Ring : 16-wedge schematic reconstructed from current heading
  • Δ7 Interneurons   : Protocerebral bridge cross-columnar inhibition reference
  • PFL3 Comparator   : Fan-shaped Body bilateral premotor balance gauge
  • PAM Dopamine      : Transient ingestion reward flashes upon eating food (35 frames)
  • 3D Connectome View: WebGL event-triggered illumination of anatomical centroids
  • DNp01 Arrest      : Terminal collision shockwave arrest HUD (non-decisional)
```

---

## 2. Anatomical Subsystems & Physiological Function

### I. Pure Chemosensory Olfactory Pathway (Antennal Lobe Glomeruli & PNs)

| Cell Class | Subtypes | Cell Count (MaleCNS) | Anatomical Region | Physiological & Behavioral Role |
| :--- | :--- | :--- | :--- | :--- |
| **ORN** (Olfactory Receptor Neurons) | `ORN_DM1`<br>`ORN_VA2` | DM1: ~30–40 / hem.<br>VA2: ~40 / hem. | Antennal Sensilla → AL Glomeruli DM1 / VA2 | **Primary Fermentation Attractants**: Transduces vinegar (ethyl acetate / acetic acid via Or42b/Or92a). Physical inter-antennal distance provides bilateral spatial contrast ($c_L \neq c_R$). |
| **ORN** (Olfactory Receptor Neurons) | `ORN_DM2` | ~25–30 / hem. | Antennal Sensilla → AL Glomerulus DM2 | **Fruity Ester Complement**: Transduces aliphatic fruit esters (via Or59b), broadening the appetitive foraging profile. |
| **PN** (Projection Neurons) | `DM1_lPN`<br>`VA2_adPN`<br>`DM2_lPN` | 1–3 cells per glomerulus | Antennal Lobe (AL) → Lateral Horn (LH) & MB Calyx | **Innate Valence Relay**: Projects to the Lateral Horn and Mushroom Body to supply chemotropic orientation bias and trigger neuromodulatory release. |

#### Physical Signal Decoupling for Antennal Lobe
* **Zero Spatial Bearing Contamination**: Antennal Lobe receptors respond exclusively to local chemical concentrations:
  $$\Delta c = c_L - c_R$$
  $$\text{stim}_{\text{AL\_L}} = \min(6.0, \max(0, \Delta c) \times 1.5), \quad \text{stim}_{\text{AL\_R}} = \min(6.0, \max(0, -\Delta c) \times 1.5)$$
* **Chemotactic Tropotaxis**: Bilateral concentration contrast drives lateralized chemotropism directly through premotor pathways without relying on visual coordinates.

#### Temporal Odor Telemetry & Biological Background (Design Rationale for Surge & Cast)
* **Actual Role of Temporal Odor Gradient ($\Delta C_t$)**:
  In the codebase, $\Delta C_t = c_{\max}(t) - c_{\max}(t-1)$ functions as a **Temporal Odor Trend telemetry signal**. It is packaged into agent telemetry and displayed in real time on the Web UI status bar to inform the user whether the snake is currently approaching or receding from food (`▲ Closer (Δc: +0.42)` / `▼ Further (Δc: -0.15)`).
* **Why Cast Oscillations Are Not Forcibly Imposed in 2D Grids**:
  - **Biological Turbulent Wind Tunnel**: Wild *Drosophila* navigating turbulent wind tunnels rely on wide crosswind cast sweeps when losing contact with intermittent odor plumes to re-acquire the fragmented filament;
  - **2D Compact Grid Environment**: In a 2D discrete environment, the food odor field is smooth and continuous. Bilateral spatial contrast ($c_L - c_R$) paired with odor-gated visual pursuit (LC10a target tracking) already yields a smoother, faster, and safer optimal path;
  - **Preventing Hazardous Self-Collisions**: Forcibly inducing open-loop lateral casting swings in a growing snake causes dangerous self-folding (self-intersection) and perimeter crashes. The architecture thus relies on spatial tropotaxis and sensitized visual pursuit, achieving rigorous theoretical alignment without engineering compromise.

---

### II. Multimodal Odor-Gated Visual Pursuit (LC10a Lobula Columnar Pathway)

| Cell Class | Subtypes | Cell Count (MaleCNS) | Anatomical Region | Physiological & Behavioral Role |
| :--- | :--- | :--- | :--- | :--- |
| **LC10a** (Lobula Columnar) | `LC10a` | ~100+ per hemisphere* | Lobula → Anterior Optic Tubercle (AOTU) | **Odor-Gated Visual Target Pursuit**: Encodes target azimuthal bearing on the retina; its gain is dynamically modulated by odor concentration. |
| **LC4 + LPLC2** (Looming Visual) | `LC4`<br>`LPLC2` | LC4: ~50–70 / hem.<br>LPLC2: ~90 / hem.* | Lobula → Posterior Ventrolateral Protocerebrum (PVLP) | **Near-Field Looming Injection**: Injected with optical looming current at the spatial interface layer when clearance $\le 1$. |

#### Encoder-Side Odor-Gated Visual Sensitization
An encoder-side sensitization formula is applied at the sensory interface, analogous to biological neuromodulatory sensitization where appetitive odors gate visual tracking circuits:

$$\text{visual\_gain} = \text{base\_gain} \times \left(1.0 + \alpha \cdot \frac{c_{\max}}{c_0}\right)$$

where:
- $\text{base\_gain} = 3.5$: Baseline visual tracking sensitivity in clean air.
- $c_{\max} = \max(c_L, c_R)$: Peak odor concentration at the antennae ($c = \frac{10}{1 + 0.3 d}$).
- $c_0 = 5.0$: Characteristic concentration normalization factor.
- $\alpha = 1.5$: Sensitization scaling coefficient.

The visual stimulation to LC10a is governed strictly by the target's retinal yaw angle and forward deadband threshold:
$$\text{stim}_{\text{LC10a\_L}} = \min(8.0, |\theta_{\text{bearing}}| \times \text{visual\_gain}) \quad (\text{if } \theta_{\text{bearing}} < -\theta_{\text{deadband}})$$
$$\text{stim}_{\text{LC10a\_R}} = \min(8.0, |\theta_{\text{bearing}}| \times \text{visual\_gain}) \quad (\text{if } \theta_{\text{bearing}} > +\theta_{\text{deadband}})$$

* **Retinal Deadband Refinement**:
  - **Parameter Tuning**: The straight-line deadband $\theta_{\text{deadband}}$ was refined from $0.15\text{ rad}$ (~$8.6^\circ$) down to $0.05\text{ rad}$ (~$2.86^\circ \approx 3^\circ$).
  - **Terminal Aiming Precision**: Under the previous $0.15\text{ rad}$ threshold, food items displaced by small offsets ($5^\circ \sim 8^\circ$) fell inside the dead zone and failed to elicit corrective visual turning, occasionally causing the snake to graze past the target. Narrowing the window to $0.05\text{ rad}$ (~$3^\circ$) ensures immediate, acute realignment during straight-line target rushes.
  - **Micro-Jitter Suppression**: Retaining an acute $\pm 3^\circ$ deadband effectively filters out whole-brain microvolt membrane potential noise, preventing alternating zig-zag wobble.
* **Far Field ($c_{\max} \to 1.0$)**: Visual gain remains near baseline (~$4.5$), allowing smooth cruise navigation.
* **Near Field ($c_{\max} \to 10.0$)**: Visual gain surges to ~$14.0$ (a $3\times$ to $4\times$ amplification), driving sharp, resolute orienting saccades directly into the target.

---

### III. Central Complex (CX) & Neuro-Telemetry HUD (Read-Only Stream)

| Cell Class | Source / Type | Count | Anatomical Region | Physiological & Telemetry Role |
| :--- | :--- | :--- | :--- | :--- |
| **E-PG Heading Ring** | Algorithmic Reconstruction | 16 wedges | Ellipsoid Body (EB) analogue | **Internal Heading Display**: 16-wedge polar visualization reconstructed from current snake heading angle (not read directly from EPG somas). |
| **Delta7** (Bridge Interneurons) | `Delta7` | ~16* | Protocerebral Bridge (PB) | **Cross-Columnar Inhibition Reference**: Stabilizes heading representation in biological systems. |
| **PFL3** (Steering Comparator) | `PFL3` | ~24* | Fan-shaped Body (FB) → Lateral Accessory Lobe (LAL) | **Premotor Balance Monitor**: Evaluates left vs. right Fan-shaped Body potentials for HUD gauge (read-only; non-decisional). |

> **Important Note**: Central Complex (CX) instruments are strictly non-decisional read-only telemetry monitors. Closed-loop turning decisions are governed exclusively by descending `DNa02` differentials and the peripheral clearance FSM.

---

### IV. Descending Command Neurons & Motor Decoding

| Cell Class | Bilateral Count | Connective Pathway | Physiological & Behavioral Role |
| :--- | :--- | :--- | :--- |
| **DNa02_L / DNa02_R** | 1 pair (1 left, 1 right) | Anterior Protocerebrum → Cervical Connective → VNC | **Ipsilateral Steering Drive**:<br>$\text{DNa02\_L} = \frac{V_L}{2.0}, \quad \text{DNa02\_R} = \frac{V_R}{2.0}$<br>$\text{diff} = \text{DNa02\_L} - \text{DNa02\_R}$<br>$\text{diff} > 0.03 \to \text{TURN\_LEFT}$; $-\text{diff} > 0.03 \to \text{TURN\_RIGHT}$ (Differential readout, zero constant lateral bias; retaining calibrated 0.05 rad deadband and 0.03 threshold). |
| **DNa01_L / DNa01_R** | 1 pair | Anterior Protocerebrum → Thoracic Leg/Crawling Circuits | **Forward Walking Drive**: Bilateral average membrane potential coordinates forward crawling propulsion (STRAIGHT). |
| **DNp01** (Giant Fiber System) | 1 pair (large-caliber axons) | Dorsal Protocerebrum → Tergotrochanteral & Flight Circuits | **Terminal Shockwave & Visualization**: Displays red arrest firework in WebGL and maintains terminal state upon fatal collision until manual reset (does not drive obstacle avoidance decisions). |

#### Supraspinal Motor Intent vs. VNC Spinal Reflex Arc
- **Supraspinal Biological Drive**: When navigating open space without imminent obstruction, steering actions are dictated entirely by the descending `DNa02` membrane potential differential (TURN_LEFT / TURN_RIGHT) or forward crawl drive from `DNa01` (STRAIGHT). This reflects emergent network dynamics propagating through 166.7k neurons under odor-gated visual stimulation, with zero constant lateral bias; membrane potentials are reset each tick via `brain.v.fill(0)` to prevent inter-step motor hysteresis.
- **VNC-Analogous Clearance Reflex Arc**: The system models the low-level peripheral reflex arc of the insect thoracic/abdominal Ventral Nerve Cord (VNC) via `MotorDecoder` with dual clearance guards: executing an emergency reflex override based on frontal spatial clearance (`dist_front == 0`, illuminating `⚡ REFLEX OVERRIDE (SPINAL COLLISION GUARD)` on the HUD), and suppressing descending turn commands when contiguous with lateral obstacles (`dist_left == 0` or `dist_right == 0`). Decoupling high-level navigational planning from low-level spinal reflex loops directly embodies standard bio-robotic and neuroethological control paradigms.

---

### V. Neuromodulatory Dynamics & Runtime Architecture

1. **PAM Dopaminergic Ingestion Signatures**:  
   Ingestion events (`ate_food == True`) trigger transient burst firing in the Mushroom Body protocerebral anterior medial (`PAM`) cluster, visualized over a 35-frame exponential decay window (~600 ms).
2. **Terminal State Preservation**:  
   Upon death, the simulation automatically pauses and holds the terminal neural collapse visual state indefinitely, preserving the 2D collision position, 3D Giant Fiber arrest HUD, and telemetry until manual reset or resume.
3. **Dynamic Connectome Indexing**:  
   Individual neuron indices are resolved dynamically at runtime via `brain.cells(["type_name"], side="L"|"R")`. Numerical indices in internal data buffers reflect specific array packings in the runtime binary rather than fixed morphological identifiers. Counts marked with `*` reflect canonical published MaleCNS v1.0 ranges.

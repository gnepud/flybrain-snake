"""FlyBrainAgent: Whole-Brain 166,700-Neuron Connectome Autonomous Agent.

Directly executes the MaleCNS v1.0 connectome spiking simulation via the 'flybrain' library
(Alex Titonis / fly.ai).

Inputs:
- Food bearing mapped to LC10a target tracking visual projection neurons
- Looming wall/body proximity mapped to LC4/LPLC2 obstacle detection neurons

Outputs:
- Descending command neurons: DNa02 (steering), DNa01 (forward crawl), DNp01 (Giant Fiber escape)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from flybrain import FlyBrain
    HAS_FLYBRAIN = True
except ImportError:
    FlyBrain = None
    HAS_FLYBRAIN = False

from src.snake_env.kinematics import Direction, RelativeAction, SnakeObservation
from src.controller.motor_decoder import MotorDecoder


class FlyBrainAgent:
    """166,700-neuron biological Drosophila connectome autonomous agent with Multimodal Odor-Gated Visual Pursuit."""

    def __init__(
        self,
        steer_threshold: float = 0.03,
        sensory_gain: float = 3.5,
        alpha: float = 1.5,
        c0: float = 5.0,
        seed: Optional[int] = 64,
        base_gain: Optional[float] = None,
        deadband: float = 0.05,
    ):
        if not HAS_FLYBRAIN:
            raise ImportError(
                "The 'flybrain' library is not installed. Install via: pip install flybrain"
            )

        self.seed = seed
        self.brain = FlyBrain(sensory_input=False, seed=seed if seed is not None else 64)
        self.decoder = MotorDecoder(steer_threshold=steer_threshold)
        self.sensory_gain = base_gain if base_gain is not None else sensory_gain
        self.base_gain = self.sensory_gain
        self.alpha = alpha
        self.c0 = c0
        self.deadband = deadband

        # Olfactory sensory populations: DM1 + VA2 (vinegar/fermentation) + DM2 (fruity esters)
        al_types = ["ORN_DM1", "ORN_VA2", "ORN_DM2", "DM1_lPN", "DM2_lPN", "VA2_adPN", "VA2_lPN"]
        self._al_l = self.brain.cells(al_types, side="L")
        self._al_r = self.brain.cells(al_types, side="R")
        self._lc10a_l = self.brain.cells(["LC10a"], side="L")
        self._lc10a_r = self.brain.cells(["LC10a"], side="R")
        looming_types = ["LC4", "LPLC2"]
        self._lc4_l = self.brain.cells(looming_types, side="L")
        self._lc4_r = self.brain.cells(looming_types, side="R")

        # Central Complex: PFL3 steering comparator and E-PG compass neurons
        self._pfl3_l = self.brain.cells(["PFL3"], side="L")
        self._pfl3_r = self.brain.cells(["PFL3"], side="R")
        self._epg_cells = self.brain.cells(["EPG", "EPGt"])
        self._current_heading_rad: float = 0.0

        # Motor populations (runtime queried from connectome)
        # DNa02: single pair of descending steering command neurons (1 on left, 1 on right)
        self._steer_l = int(self.brain.cells(["DNa02"], side="L")[0])
        self._steer_r = int(self.brain.cells(["DNa02"], side="R")[0])
        # DNa01: bilateral forward walking command neurons
        dna01_cells = self.brain.cells(["DNa01"])
        self._forward_cells = dna01_cells if len(dna01_cells) > 0 else self.brain.groups.get("forward_L", [36])
        # DNp01: Giant Fiber escape neurons (1 pair)
        self._dnp01 = self.brain.cells(["DNp01"])
        self._prev_c: Optional[float] = None

        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets membrane potentials and refractory timers."""
        if seed is not None:
            self.seed = seed
        self._prev_c = None
        self._current_heading_rad = 0.0
        if hasattr(self, "brain"):
            self.brain.reset(seed=self.seed)

    def act(self, obs: SnakeObservation) -> Tuple[RelativeAction, Dict[str, Any]]:
        """Integrates 166.7k-neuron whole-brain dynamics with Multimodal Odor-Gated Visual Pursuit.
        
        Sensorimotor Architecture:
        1. Pure Chemical Channel -> Antennal Lobe (AL):
           Only chemical signal (concentration gradient cL - cR); zero geometric visual bearing.
        2. Odor-Gated Visual Gain:
           visual_gain = base_gain * (1.0 + alpha * (c_max / c0))
        3. Pure Visual Channel -> Lobula Columnar (LC10a):
           Geometric target bearing amplified by the neuromodulatory odor gate.
        4. Looming Obstacle Avoidance -> LC4/LPLC2.
        """
        # 1. Biological saccadic reset to clear transient motor hysteresis deterministically
        self.brain.v.fill(0)
        self.brain.fired = self.brain.xp.empty(0, self.brain.xp.int64)

        # Bilateral antennal odor concentration computation (screen coords: y grows downwards)
        head_x, head_y = obs.head
        food_x, food_y = obs.food
        direction = Direction(obs.direction) if isinstance(obs.direction, int) else obs.direction
        dx, dy = direction.vector

        # Left antenna (forward-left) and Right antenna (forward-right)
        al_x, al_y = head_x + dx + dy, head_y + dy - dx
        ar_x, ar_y = head_x + dx - dy, head_y + dy + dx

        dL = math.hypot(al_x - food_x, al_y - food_y)
        dR = math.hypot(ar_x - food_x, ar_y - food_y)

        cL = 10.0 / (1.0 + 0.3 * dL)
        cR = 10.0 / (1.0 + 0.3 * dR)
        c_max = max(cL, cR)
        delta_c = (c_max - self._prev_c) if self._prev_c is not None else 0.0
        self._prev_c = c_max

        # --- 1. Pure Chemical Channel -> Antennal Lobe (AL) ---
        # Absolutely no visual bearing; AL receptors only receive bilateral chemical difference
        diff_c = cL - cR
        stim_al_l = min(6.0, max(0.0, diff_c) * 1.5)
        stim_al_r = min(6.0, max(0.0, -diff_c) * 1.5)

        # --- 2. Odor-Gated Visual Gain (Neuromodulatory Sensitization) ---
        # visual_gain = base_gain * (1.0 + alpha * (c_max / c0))
        gating_factor = 1.0 + self.alpha * (c_max / self.c0)
        visual_gain = self.sensory_gain * gating_factor

        # --- 3. Pure Geometric Visual Channel -> Lobula Columnar (LC10a) ---
        # LC10a only receives retinal yaw angle food_bearing, scaled by odor gate.
        # Deadband refined from 0.15 rad (~8.6°) down to 0.05 rad (~2.86° ≈ 3°) for acute straight pursuit.
        bearing = obs.food_bearing if obs.food_bearing is not None else 0.0
        stim_vis_l = min(8.0, abs(bearing) * visual_gain) if bearing < -self.deadband else 0.0
        stim_vis_r = min(8.0, abs(bearing) * visual_gain) if bearing > self.deadband else 0.0

        # --- 4. Central Complex E-PG Heading Compass & Ring Attractor ---
        heading_angles = {
            int(Direction.UP): -math.pi / 2.0,
            int(Direction.RIGHT): 0.0,
            int(Direction.DOWN): math.pi / 2.0,
            int(Direction.LEFT): math.pi,
        }
        target_h = heading_angles.get(int(obs.direction), 0.0)
        diff_h = math.atan2(math.sin(target_h - self._current_heading_rad), math.cos(target_h - self._current_heading_rad))
        self._current_heading_rad += 0.45 * diff_h

        # 16-wedge Ellipsoid Body ring attractor bump
        epg_bump: List[float] = []
        sigma_eb = (2.0 * math.pi / 16.0) * 1.35
        for i in range(16):
            wedge_angle = i * (2.0 * math.pi / 16.0) - (math.pi / 2.0)
            ang_dist = math.atan2(math.sin(wedge_angle - self._current_heading_rad), math.cos(wedge_angle - self._current_heading_rad))
            bump_val = math.exp(-(ang_dist ** 2) / (2.0 * (sigma_eb ** 2)))
            epg_bump.append(round(0.08 + 0.88 * bump_val, 3))

        accum_vl = 0.0
        accum_vr = 0.0
        accum_vf = 0.0
        accum_pfl3_l = 0.0
        accum_pfl3_r = 0.0
        all_spikes: List[int] = []

        # 5. 4-step settling loop (80 ms biological reaction cycle)
        for s in range(4):
            # Chemical stimulation of Antennal Lobe glomeruli
            if stim_al_l > 0 and len(self._al_l) > 0:
                self.brain.stimulate(self._al_l, stim_al_l)
            if stim_al_r > 0 and len(self._al_r) > 0:
                self.brain.stimulate(self._al_r, stim_al_r)

            # Odor-gated visual stimulation of LC10a target tracking projection neurons
            if stim_vis_l > 0 and len(self._lc10a_l) > 0:
                self.brain.stimulate(self._lc10a_l, stim_vis_l)
            if stim_vis_r > 0 and len(self._lc10a_r) > 0:
                self.brain.stimulate(self._lc10a_r, stim_vis_r)

            # Looming obstacle proximity (LC4 angular velocity + LPLC2 expansion)
            if obs.dist_left is not None and obs.dist_left <= 1 and len(self._lc4_l) > 0:
                self.brain.stimulate(self._lc4_l, 4.0)
            if obs.dist_right is not None and obs.dist_right <= 1 and len(self._lc4_r) > 0:
                self.brain.stimulate(self._lc4_r, 4.0)
            if obs.dist_front is not None and obs.dist_front <= 1:
                if len(self._lc4_l) > 0:
                    self.brain.stimulate(self._lc4_l, 3.5)
                if len(self._lc4_r) > 0:
                    self.brain.stimulate(self._lc4_r, 3.5)

            spikes = self.brain.step()
            all_spikes.extend(spikes)

            # Accumulate activations over mature propagation steps (s >= 2)
            if s >= 2:
                accum_vl += float(self.brain.v[self._steer_l, 0])
                accum_vr += float(self.brain.v[self._steer_r, 0])
                accum_vf += float(self.brain.v[self._forward_cells, 0].mean())
                if len(self._pfl3_l) > 0:
                    accum_pfl3_l += float(self.brain.v[self._pfl3_l, 0].mean())
                if len(self._pfl3_r) > 0:
                    accum_pfl3_r += float(self.brain.v[self._pfl3_r, 0].mean())

        # 6. Descending motor signals & Central Complex PFL3 comparator
        dna02_l = accum_vl / 2.0
        dna02_r = accum_vr / 2.0
        dnp = accum_vf / 2.0
        pfl3_l = (accum_pfl3_l / 2.0) if len(self._pfl3_l) > 0 else dna02_l
        pfl3_r = (accum_pfl3_r / 2.0) if len(self._pfl3_r) > 0 else dna02_r

        # 7. Decode motor decision with reflexive obstacle override
        action = self.decoder.decode(
            dna_left=dna02_l,
            dna_right=dna02_r,
            forward=dnp,
            dist_front=obs.dist_front,
            dist_left=obs.dist_left,
            dist_right=obs.dist_right,
        )

        # Downsample spikes for frontend WebGL telemetry
        sample_spikes = [int(x) for x in all_spikes[:100]]

        info: Dict[str, Any] = {
            "action": action,
            "dna02_l": dna02_l,
            "dna02_r": dna02_r,
            "dna01": dnp,
            "dnp": dnp,
            "spikes_count": len(all_spikes),
            "spikes": sample_spikes,
            "epg": epg_bump,
            "steering": {
                "pfl3_l": round(pfl3_l, 3),
                "pfl3_r": round(pfl3_r, 3),
                "threshold": self.decoder.steer_threshold,
                "is_override": self.decoder.last_override,
            },
            "motor": {
                "dna02_l": round(dna02_l, 3),
                "dna02_r": round(dna02_r, 3),
                "dna01": round(dnp, 3),
                "dnp": round(dnp, 3),
                "is_override": self.decoder.last_override,
            },
            "backend_mode": "flybrain_166k",
            "olfactory": {
                "c_left": round(cL, 3),
                "c_right": round(cR, 3),
                "diff": round(diff_c, 3),
                "delta_c": round(delta_c, 3),
                "gating_factor": round(gating_factor, 3),
                "visual_gain": round(visual_gain, 3),
                "deadband": self.deadband,
            },
        }
        return action, info

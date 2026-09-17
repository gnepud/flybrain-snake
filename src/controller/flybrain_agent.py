"""FlyBrainAgent: Whole-Brain 166,700-Neuron Connectome Autonomous Agent.

Directly executes the MaleCNS v1.0 connectome spiking simulation via the 'flybrain' library
(Alex Titonis / fly.ai).

Inputs:
- Food bearing mapped to LC10a target tracking visual projection neurons
- Looming wall/body proximity mapped to LC4/LPLC2 obstacle detection neurons

Outputs:
- Descending command neurons: DNa02 (steering), DNg100 (forward crawl), DNp01 (Giant Fiber escape)
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    from flybrain import FlyBrain
    HAS_FLYBRAIN = True
except ImportError:
    FlyBrain = None
    HAS_FLYBRAIN = False

from src.snake_env.kinematics import RelativeAction, SnakeObservation
from src.controller.motor_decoder import MotorDecoder


class FlyBrainAgent:
    """166,700-neuron biological Drosophila connectome autonomous agent."""

    def __init__(
        self,
        steer_threshold: float = 0.05,
        sensory_gain: float = 3.5,
        seed: Optional[int] = 64,
    ):
        if not HAS_FLYBRAIN:
            raise ImportError(
                "The 'flybrain' library is not installed. Install via: pip install flybrain"
            )

        self.seed = seed
        self.brain = FlyBrain(sensory_input=False, seed=seed if seed is not None else 64)
        self.decoder = MotorDecoder(steer_threshold=steer_threshold)
        self.sensory_gain = sensory_gain

        # Sensory populations (Visual projection neurons)
        self._lc10a_l = self.brain.cells(["LC10a"], side="L")
        self._lc10a_r = self.brain.cells(["LC10a"], side="R")
        self._lc4_l = self.brain.cells(["LC4"], side="L")
        self._lc4_r = self.brain.cells(["LC4"], side="R")

        # Motor populations
        self._steer_l = int(self.brain.groups["steer_L"][0])
        self._steer_r = int(self.brain.groups["steer_R"][0])
        self._forward_l = int(self.brain.groups["forward_L"][0])

        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets membrane potentials and refractory timers."""
        if seed is not None:
            self.seed = seed
        if hasattr(self, "brain"):
            self.brain.reset(seed=self.seed)

    def act(self, obs: SnakeObservation) -> Tuple[RelativeAction, Dict[str, Any]]:
        """Integrates 166.7k-neuron whole-brain dynamics and decodes steering actions.
        
        Uses a 4-step biological saccade cycle (80 ms) with transient membrane reset:
        - Allows sensory inputs (LC10a, LC4) to propagate through the 166k connectome
        - Prevents perpetual limit-cycle reverberations and spinning in circles
        """
        # 1. Biological saccadic reset to clear transient motor hysteresis deterministically
        self.brain.v.fill(0)
        self.brain.fired = self.brain.xp.empty(0, self.brain.xp.int64)

        bearing = obs.food_bearing if obs.food_bearing is not None else 0.0
        accum_vl = 0.0
        accum_vr = 0.0
        accum_vf = 0.0
        all_spikes: List[int] = []

        # 2. 4-step settling loop (80 ms biological reaction cycle)
        for s in range(4):
            # Dead-band of ~11 degrees (|bearing| < 0.20): food is directly in front -> continue straight
            if bearing < -0.20 and len(self._lc10a_l) > 0:
                stim_l = min(7.0, abs(bearing) * self.sensory_gain)
                self.brain.stimulate(self._lc10a_l, stim_l)
            elif bearing > 0.20 and len(self._lc10a_r) > 0:
                stim_r = min(7.0, abs(bearing) * self.sensory_gain)
                self.brain.stimulate(self._lc10a_r, stim_r)

            # Looming obstacle proximity to LC4 / LPLC2
            if obs.dist_left is not None and obs.dist_left <= 1 and len(self._lc4_l) > 0:
                self.brain.stimulate(self._lc4_l, 4.0)
            if obs.dist_right is not None and obs.dist_right <= 1 and len(self._lc4_r) > 0:
                self.brain.stimulate(self._lc4_r, 4.0)

            spikes = self.brain.step()
            all_spikes.extend(spikes)

            # Accumulate descending activations over mature propagation steps (s >= 2)
            if s >= 2:
                accum_vl += float(self.brain.v[self._steer_l, 0])
                accum_vr += float(self.brain.v[self._steer_r, 0])
                accum_vf += float(self.brain.v[self._forward_l, 0])

        # 3. Descending motor signals
        dna02_l = accum_vl / 2.0
        dna02_r = accum_vr / 2.0
        dnp = accum_vf / 2.0

        # 4. Decode motor decision with reflexive obstacle override
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
            "dnp": dnp,
            "spikes_count": len(all_spikes),
            "spikes": sample_spikes,
            "steering": {"pfl3_l": dna02_l, "pfl3_r": dna02_r},
            "backend_mode": "flybrain_166k",
        }
        return action, info

"""Motor Decoder: Descending Premotor Activations to Relative Turning Actions.

Translates bilateral descending motor neuron signals (DNa02_L, DNa02_R, DNp) into
discrete RelativeAction decisions (STRAIGHT, TURN_LEFT, TURN_RIGHT) with biological
reflexive obstacle override.
"""

from typing import Optional

from src.snake_env.kinematics import RelativeAction


class MotorDecoder:
    """Decodes bilateral descending motor activations into discrete snake maneuvers."""

    def __init__(self, steer_threshold: float = 0.05):
        self.steer_threshold = steer_threshold

    def decode(
        self,
        dna_left: float,
        dna_right: float,
        forward: float = 0.5,
        dist_front: Optional[int] = None,
        dist_left: Optional[int] = None,
        dist_right: Optional[int] = None,
    ) -> RelativeAction:
        """Decodes descending motor activations into a discrete RelativeAction.
        
        Args:
            dna_left: Left descending motor neuron activation (DNa02_L).
            dna_right: Right descending motor neuron activation (DNa02_R).
            forward: Forward locomotion drive (DNp).
            dist_front: Clearance in current heading.
            dist_left: Clearance to the left.
            dist_right: Clearance to the right.
            
        Returns:
            RelativeAction (STRAIGHT, TURN_LEFT, or TURN_RIGHT).
        """
        # 1. Reflexive Collision Override:
        # If immediate obstacle in front (dist_front == 0), moving straight would crash.
        if dist_front is not None and dist_front == 0:
            left_ok = (dist_left is None or dist_left > 0)
            right_ok = (dist_right is None or dist_right > 0)
            if left_ok and not right_ok:
                return RelativeAction.TURN_LEFT
            elif right_ok and not left_ok:
                return RelativeAction.TURN_RIGHT
            elif left_ok and right_ok:
                # Both lateral directions open: follow descending motor command
                if dna_left > dna_right:
                    return RelativeAction.TURN_LEFT
                elif dna_right > dna_left:
                    return RelativeAction.TURN_RIGHT
                else:
                    return RelativeAction.TURN_LEFT if (dist_left or 0) >= (dist_right or 0) else RelativeAction.TURN_RIGHT
            else:
                return RelativeAction.STRAIGHT

        # 2. Descending Steering Differential Decoding
        diff = dna_left - dna_right
        if diff > self.steer_threshold:
            # Lateral clearance guard: suppress left turn if obstacle is immediately adjacent on the left
            if dist_left is not None and dist_left == 0:
                if dist_front is not None and dist_front == 0:
                    return RelativeAction.TURN_RIGHT if (dist_right is None or dist_right > 0) else RelativeAction.STRAIGHT
                return RelativeAction.STRAIGHT
            return RelativeAction.TURN_LEFT
        elif -diff > self.steer_threshold:
            # Lateral clearance guard: suppress right turn if obstacle is immediately adjacent on the right
            if dist_right is not None and dist_right == 0:
                if dist_front is not None and dist_front == 0:
                    return RelativeAction.TURN_LEFT if (dist_left is None or dist_left > 0) else RelativeAction.STRAIGHT
                return RelativeAction.STRAIGHT
            return RelativeAction.TURN_RIGHT
        else:
            return RelativeAction.STRAIGHT


def decode_motor(
    dna_left: float,
    dna_right: float,
    forward: float = 0.5,
    steer_threshold: float = 0.05,
    dist_front: Optional[int] = None,
    dist_left: Optional[int] = None,
    dist_right: Optional[int] = None,
) -> RelativeAction:
    """Functional convenience wrapper for motor decoding."""
    decoder = MotorDecoder(steer_threshold=steer_threshold)
    return decoder.decode(
        dna_left=dna_left,
        dna_right=dna_right,
        forward=forward,
        dist_front=dist_front,
        dist_left=dist_left,
        dist_right=dist_right,
    )

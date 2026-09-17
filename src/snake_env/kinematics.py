"""Snake Environment Kinematics, Directions, Actions, and Observation Models.

Provides:
- Direction: Cardinal grid headings (UP, RIGHT, DOWN, LEFT) with displacement vectors.
- RelativeAction: Egocentric discrete turns (STRAIGHT, TURN_LEFT, TURN_RIGHT).
- SnakeObservation: Immutable snapshot of board state and sensory features.
- Kinematic helper functions: Egocentric bearing calculation, distance raycasting, turn geometry.
"""

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, List, Optional, Tuple


class Direction(IntEnum):
    """Cardinal grid directions with standard screen coordinate vectors (y grows downwards)."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def vector(self) -> Tuple[int, int]:
        """Displacement step vector (dx, dy)."""
        return DIRECTION_VECTORS[self]

    def opposite(self) -> "Direction":
        """180-degree reverse direction."""
        return OPPOSITE_DIRECTIONS[self]


class RelativeAction(IntEnum):
    """Egocentric turning actions relative to the snake's current heading."""

    STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


DIRECTION_VECTORS: dict[Direction, Tuple[int, int]] = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}

HEADING_ANGLES: dict[int, float] = {
    int(Direction.UP): -math.pi / 2,   # UP (0, -1)
    int(Direction.RIGHT): 0.0,         # RIGHT (1, 0)
    int(Direction.DOWN): math.pi / 2,  # DOWN (0, 1)
    int(Direction.LEFT): math.pi,      # LEFT (-1, 0)
}

OPPOSITE_DIRECTIONS: dict[Direction, Direction] = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


def turn_direction(current: Direction, action: RelativeAction | int) -> Direction:
    """Computes the new absolute heading given a relative turning action."""
    act = RelativeAction(action)
    if act == RelativeAction.STRAIGHT:
        return current
    elif act == RelativeAction.TURN_LEFT:
        return Direction((int(current) - 1) % 4)
    elif act == RelativeAction.TURN_RIGHT:
        return Direction((int(current) + 1) % 4)
    raise ValueError(f"Invalid relative action: {action}")


def calculate_food_bearing(
    head: Tuple[int, int],
    food: Tuple[int, int],
    direction: Direction | int,
) -> float:
    """Calculates egocentric food bearing angle in radians normalized to (-pi, pi].
    
    Returns 0.0 if head and food are co-located or straight ahead,
    negative values for food to the left (-pi/2 for 90 deg left),
    positive values for food to the right (+pi/2 for 90 deg right).
    """
    dx = food[0] - head[0]
    dy = food[1] - head[1]
    if dx == 0 and dy == 0:
        return 0.0

    target_angle = math.atan2(dy, dx)
    heading_angle = HEADING_ANGLES[int(direction)]
    diff = target_angle - heading_angle

    # Normalize to (-pi, pi] matching reference oracle exactly
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff <= -math.pi:
        diff += 2.0 * math.pi

    return diff


def calculate_distance_rays(
    head: Tuple[int, int],
    body: List[Tuple[int, int]],
    direction: Direction | int,
    width: int,
    height: int,
) -> Tuple[int, int, int]:
    """Calculates clearance distances (front, left, right) to the nearest obstacle.
    
    Considers grid boundaries and body segments as obstacles.
    In accordance with the tail vacation rule, the tail segment is not treated as
    an immediate obstacle since it vacates on non-food steps.
    
    Returns:
        (dist_front, dist_left, dist_right) in grid units.
    """
    # Safe obstacle set: if head is already outside, return 0 clearance
    if not (0 <= head[0] < width and 0 <= head[1] < height):
        return (0, 0, 0)

    # All body segments except the tail segment (which vacates on the next move)
    body_obstacles = set(body[:-1]) if len(body) > 1 else set()

    dir_val = int(direction)
    front_dir = Direction(dir_val)
    left_dir = Direction((dir_val - 1) % 4)
    right_dir = Direction((dir_val + 1) % 4)

    def _cast_ray(d: Direction) -> int:
        dx, dy = DIRECTION_VECTORS[d]
        k = 1
        while True:
            nx = head[0] + k * dx
            ny = head[1] + k * dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height or (nx, ny) in body_obstacles:
                return k - 1
            k += 1

    return (_cast_ray(front_dir), _cast_ray(left_dir), _cast_ray(right_dir))


@dataclass(frozen=True)
class SnakeObservation:
    """Immutable state observation and egocentric sensory features for the Snake game."""

    head: Tuple[int, int]
    body: List[Tuple[int, int]]
    food: Tuple[int, int]
    direction: Direction
    grid_size: Tuple[int, int]
    food_bearing: float         # relative angle in radians [-pi, pi]
    food_distance: float        # Euclidean distance
    dist_front: int             # clearance to nearest wall/body in current heading
    dist_left: int              # clearance to nearest wall/body to the left
    dist_right: int             # clearance to nearest wall/body to the right
    score: int
    step_count: int
    done: bool
    reason: Optional[str] = None  # None, "WALL_COLLISION", "BODY_COLLISION", "VICTORY", "TIMEOUT"

    def to_dict(self) -> dict[str, Any]:
        """Serializes observation to a JSON-compatible dictionary."""
        return {
            "head": list(self.head),
            "body": [list(p) for p in self.body],
            "food": list(self.food),
            "direction": int(self.direction),
            "grid_size": list(self.grid_size),
            "food_bearing": self.food_bearing,
            "food_distance": self.food_distance,
            "dist_front": self.dist_front,
            "dist_left": self.dist_left,
            "dist_right": self.dist_right,
            "score": self.score,
            "step_count": self.step_count,
            "done": self.done,
            "reason": self.reason,
        }

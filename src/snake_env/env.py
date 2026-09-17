"""Snake Environment Engine.

Implements the high-performance 2D discrete grid Snake simulation:
- Deterministic PRNG seeded food generation
- Contiguous body deque management with tail vacation rules
- Boundary and body collision detection
- Closed-loop egocentric sensory observation generation
"""

import math
import random
from typing import Any, List, Optional, Tuple, Union

from src.snake_env.kinematics import (
    Direction,
    RelativeAction,
    SnakeObservation,
    DIRECTION_VECTORS,
    turn_direction,
    calculate_food_bearing,
    calculate_distance_rays,
)

_SENTINEL = object()


class SnakeEnv:
    """Discrete 2D Snake game environment with deterministic seeding and full telemetry."""

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        seed: Optional[int] = None,
        restrict_borders: bool = False,
    ):
        if width < 4 or height < 4:
            raise ValueError(f"Grid dimensions must be at least 4x4, got {width}x{height}")

        self.width = int(width)
        self.height = int(height)
        self.restrict_borders: bool = bool(restrict_borders)
        self._initial_seed = seed
        self._rng: random.Random = random.Random(seed) if seed is not None else random.Random()

        self._head: Tuple[int, int] = (0, 0)
        self._body: List[Tuple[int, int]] = []
        self._food: Tuple[int, int] = (0, 0)
        self._direction: Direction = Direction.RIGHT
        self._score: int = 0
        self._step_count: int = 0
        self._done: bool = False
        self._reason: Optional[str] = None

        self.reset(seed=seed)

    def reset(self, seed: Any = _SENTINEL) -> SnakeObservation:
        """Resets board to initial centered state and binds PRNG sequence."""
        if seed is not _SENTINEL:
            self._initial_seed = seed
            if seed is not None:
                self._rng = random.Random(seed)
            else:
                self._rng = random.Random()
        else:
            if self._initial_seed is not None:
                self._rng = random.Random(self._initial_seed)
            else:
                self._rng = random.Random()

        self._score = 0
        self._step_count = 0
        self._done = False
        self._reason = None

        # Position snake horizontally near center facing RIGHT
        cx = self.width // 2
        cy = self.height // 2
        self._head = (cx, cy)
        self._body = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self._direction = Direction.RIGHT

        self._spawn_food()
        return self._get_obs()

    def _spawn_food(self) -> None:
        """Uniformly samples a vacant grid coordinate for new food."""
        occupied = set(self._body)
        margin = 2 if self.restrict_borders else 0

        # Candidate vacant coordinates, excluding margin cells if restrict_borders is enabled
        vacant = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied and (
                not self.restrict_borders or (
                    margin <= x < self.width - margin and
                    margin <= y < self.height - margin
                )
            )
        ]

        if not vacant:
            # Fall back to any vacant coordinate on the board if inner zone is full or unavailable
            vacant = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if (x, y) not in occupied
            ]

        if not vacant:
            # Full grid victory condition
            self._done = True
            self._reason = "VICTORY"
        else:
            self._food = self._rng.choice(vacant)

    def step(
        self,
        action: Union[RelativeAction, Direction, int],
    ) -> Tuple[SnakeObservation, float, bool, dict[str, Any]]:
        """Executes one simulation step given an action.
        
        Args:
            action: RelativeAction (STRAIGHT, TURN_LEFT, TURN_RIGHT) or Direction (UP, RIGHT, DOWN, LEFT).
            
        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            return (
                self._get_obs(),
                0.0,
                True,
                {"reason": self._reason, "score": self._score, "step_count": self._step_count},
            )

        # 1. Action decoding and self-reversal blocking
        if isinstance(action, Direction):
            # Absolute direction command
            if len(self._body) > 1 and (int(action) + 2) % 4 == int(self._direction):
                # Block 180-degree instant reversal; maintain current heading
                pass
            else:
                self._direction = action
        elif isinstance(action, RelativeAction) or action in (0, 1, 2):
            self._direction = turn_direction(self._direction, action)
        elif action == 3:
            # Raw int 3 corresponds to Direction.LEFT
            if len(self._body) > 1 and (3 + 2) % 4 == int(self._direction):
                pass
            else:
                self._direction = Direction.LEFT
        else:
            raise ValueError(f"Invalid action: {action}")

        # 2. Compute candidate head position
        dx, dy = DIRECTION_VECTORS[self._direction]
        next_head = (self._head[0] + dx, self._head[1] + dy)

        # 3. Boundary / wall collision check
        if (
            next_head[0] < 0
            or next_head[0] >= self.width
            or next_head[1] < 0
            or next_head[1] >= self.height
        ):
            self._head = next_head
            self._step_count += 1
            self._done = True
            self._reason = "WALL_COLLISION"
            return (
                self._get_obs(),
                0.0,
                True,
                {"reason": self._reason, "score": self._score, "step_count": self._step_count},
            )

        # 4. Food eating status
        eating_food = (next_head == self._food)

        # 5. Body collision check with tail vacation rule
        # If not eating food, the tail segment will pop on this step, so entering
        # the vacated tail cell is completely valid.
        if eating_food:
            obstacle_segments = set(self._body)
        else:
            obstacle_segments = set(self._body[:-1]) if len(self._body) > 1 else set()

        if next_head in obstacle_segments:
            self._head = next_head
            self._step_count += 1
            self._done = True
            self._reason = "BODY_COLLISION"
            return (
                self._get_obs(),
                0.0,
                True,
                {"reason": self._reason, "score": self._score, "step_count": self._step_count},
            )

        # 6. Advance snake body
        self._head = next_head
        self._body.insert(0, next_head)
        self._step_count += 1

        if eating_food:
            self._score += 1
            reward = 1.0
            self._spawn_food()
        else:
            self._body.pop()
            reward = 0.0

        return (
            self._get_obs(),
            reward,
            self._done,
            {"reason": self._reason, "score": self._score, "step_count": self._step_count},
        )

    def _get_obs(self) -> SnakeObservation:
        """Constructs egocentric observation dataclass."""
        bearing = calculate_food_bearing(self._head, self._food, self._direction)
        dist = math.hypot(self._food[0] - self._head[0], self._food[1] - self._head[1])
        df, dl, dr = calculate_distance_rays(
            self._head, self._body, self._direction, self.width, self.height
        )

        return SnakeObservation(
            head=self._head,
            body=list(self._body),
            food=self._food,
            direction=self._direction,
            grid_size=(self.width, self.height),
            food_bearing=bearing,
            food_distance=dist,
            dist_front=df,
            dist_left=dl,
            dist_right=dr,
            score=self._score,
            step_count=self._step_count,
            done=self._done,
            reason=self._reason,
        )

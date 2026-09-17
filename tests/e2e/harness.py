"""Opaque-box E2E Test Harness for FlyBrain Snake.

Provides high-level test fixtures, episode runners, trajectory validators,
and server telemetry inspectors without relying on internal module private state.
"""

import json
import math
import socket
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple
from tests.helpers import safe_import, require_symbols, reference_relative_bearing


class E2ETestHarness:
    """Opaque-box test harness orchestrating end-to-end sessions and evaluations."""

    def __init__(self):
        # Dynamically discover and bind public classes
        env_cls = safe_import("src.snake_env.env", "SnakeEnv")[0]
        if env_cls is None:
            env_cls = safe_import("src.snake_env", "SnakeEnv")[0]
        self.SnakeEnv = env_cls

        agent_cls = safe_import("src.controller.agent", "FlyBrainAgent", "RandomAgent")
        if agent_cls[0] is None:
            agent_cls = safe_import("src.controller.flybrain_agent", "FlyBrainAgent", "RandomAgent")
        self.ConnectomeAgent = agent_cls[0]
        self.FlyBrainAgent = agent_cls[0]
        self.RandomAgent = agent_cls[1]

        server_symbols = safe_import(
            "src.web.server",
            "SnakeServer",
            "create_server",
        )
        self.SnakeServer = server_symbols[0]
        self.create_server = server_symbols[1]

    def has_environment(self) -> bool:
        return self.SnakeEnv is not None

    def has_controller(self) -> bool:
        return self.ConnectomeAgent is not None

    def has_server(self) -> bool:
        return self.SnakeServer is not None or self.create_server is not None

    def run_session(
        self,
        grid_width: int = 16,
        grid_height: int = 16,
        seed: int = 42,
        max_steps: int = 500,
        agent_type: str = "connectome",
    ) -> Dict[str, Any]:
        """Execute a complete opaque-box game session and return structured trajectory telemetry."""
        if not self.has_environment():
            require_symbols(None, feature_desc="SnakeEnv (Milestone 2)")

        env = self.SnakeEnv(width=grid_width, height=grid_height, seed=seed)
        if agent_type == "connectome":
            if not self.has_controller():
                require_symbols(None, feature_desc="ConnectomeAgent (Milestone 2)")
            agent = self.ConnectomeAgent()
        elif agent_type == "random":
            if self.RandomAgent:
                agent = self.RandomAgent(seed=seed)
            else:
                # Fallback simple random agent
                class _FallbackRandom:
                    def __init__(self, s):
                        import random
                        self.rng = random.Random(s)
                    def act(self, obs):
                        return self.rng.choice([0, 1, 2]), {}
                agent = _FallbackRandom(seed)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        obs = env.reset(seed=seed)
        if hasattr(agent, "reset"):
            agent.reset()

        trajectory = {
            "steps": 0,
            "score": 0,
            "done": False,
            "reason": None,
            "head_history": [obs.head],
            "actions": [],
            "food_history": [obs.food],
            "wall_collisions": 0,
            "body_collisions": 0,
        }

        while not obs.done and trajectory["steps"] < max_steps:
            action, act_info = agent.act(obs)
            next_obs, reward, done, info = env.step(action)

            trajectory["steps"] += 1
            trajectory["score"] = next_obs.score
            trajectory["head_history"].append(next_obs.head)
            trajectory["actions"].append(action)
            trajectory["food_history"].append(next_obs.food)

            if done:
                trajectory["done"] = True
                trajectory["reason"] = next_obs.reason or info.get("reason")
                if "WALL" in str(trajectory["reason"]).upper():
                    trajectory["wall_collisions"] += 1
                elif "BODY" in str(trajectory["reason"]).upper():
                    trajectory["body_collisions"] += 1
                break

            obs = next_obs

        return trajectory

    @staticmethod
    def validate_trajectory_integrity(trajectory: Dict[str, Any], grid_size: Tuple[int, int]) -> List[str]:
        """Verify physics and kinematic invariants along the trajectory.
        
        Returns a list of violation descriptions (empty if 100% valid).
        """
        violations = []
        heads = trajectory["head_history"]
        W, H = grid_size

        for i in range(len(heads) - 1):
            h1 = heads[i]
            h2 = heads[i + 1]
            dist = abs(h1[0] - h2[0]) + abs(h1[1] - h2[1])
            # If step didn't end in death, head must move exactly 1 Manhattan unit
            if i < len(heads) - 2 or not trajectory["done"]:
                if dist != 1:
                    violations.append(f"Non-contiguous head jump between step {i} {h1} and {i+1} {h2}")

        # Check bounds for all steps except the terminal crash coordinate
        for i, h in enumerate(heads[:-1]):
            if not (0 <= h[0] < W and 0 <= h[1] < H):
                violations.append(f"Premature out-of-bounds at step {i}: {h}")

        return violations

    def run_paired_benchmark(self, episodes: int = 50, seed_base: int = 1000) -> Dict[str, Any]:
        """Execute paired multi-episode evaluation for Connectome vs Random baseline."""
        results = {
            "connectome": {"survival": [], "food": []},
            "random": {"survival": [], "food": []},
        }

        for ep in range(episodes):
            seed = seed_base + ep
            res_c = self.run_session(seed=seed, agent_type="connectome")
            res_r = self.run_session(seed=seed, agent_type="random")

            results["connectome"]["survival"].append(res_c["steps"])
            results["connectome"]["food"].append(res_c["score"])
            results["random"]["survival"].append(res_r["steps"])
            results["random"]["food"].append(res_r["score"])

        return results

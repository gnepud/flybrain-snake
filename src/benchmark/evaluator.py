"""Multi-episode comparative benchmark evaluator for FlyBrain Snake.

Provides:
- BenchmarkEvaluator: Comparative evaluation engine comparing ConnectomeAgent
  against RandomAgent across identical seeded environments.
- run_benchmark / run_headless_benchmark: High-level headless benchmark entry points.
"""

from datetime import datetime, timezone
import random
import time
import tracemalloc
from typing import Any, Dict, List, Optional

from src.snake_env.env import SnakeEnv
from src.controller.flybrain_agent import FlyBrainAgent
from src.controller.agent import RandomAgent
from src.benchmark.stats import (
    welch_t_test,
    mann_whitney_u,
    mean,
    std_dev,
    median,
)


class BenchmarkEvaluator:
    """Evaluates and compares autonomous Snake agents under reproducible conditions."""

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        max_steps: int = 2000,
    ):
        """Initialize BenchmarkEvaluator with default board dimensions and step limit."""
        self.width = width
        self.height = height
        self.max_steps = max_steps

    def _resolve_agent(self, agent_spec: Any, seed: int) -> Any:
        """Instantiate or reconfigure an agent for a given seed."""
        if agent_spec in ("flybrain", "connectome", None):
            return FlyBrainAgent(seed=seed)
        elif agent_spec == "random":
            return RandomAgent(seed=seed)
        elif callable(agent_spec) and not hasattr(agent_spec, "act"):
            try:
                return agent_spec(seed=seed)
            except TypeError:
                return agent_spec()
        elif isinstance(agent_spec, RandomAgent):
            agent_spec._rng = random.Random(seed)
            return agent_spec
        elif hasattr(agent_spec, "act"):
            if hasattr(agent_spec, "reset"):
                agent_spec.reset()
            return agent_spec
        else:
            raise ValueError(f"Unsupported agent specification: {agent_spec}")

    def evaluate_agent(
        self,
        agent: Any,
        episodes: int = 50,
        seed_base: int = 1000,
        max_steps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute multi-episode evaluation for a single agent.
        
        Args:
            agent: Agent instance, type, or string ("connectome" | "random").
            episodes: Number of episodes to simulate.
            seed_base: Starting seed for reproducible sequence.
            max_steps: Maximum allowable steps per episode.
            width: Grid width (defaults to self.width).
            height: Grid height (defaults to self.height).
            
        Returns:
            Dictionary containing per-episode trajectories, aggregate metrics, and timing.
        """
        w = width or self.width
        h = height or self.height
        step_limit = max_steps or self.max_steps

        survival: List[int] = []
        food: List[int] = []
        reasons: List[str] = []
        wall_collisions = 0
        body_collisions = 0
        timeouts = 0

        start_time = time.perf_counter()

        for ep in range(episodes):
            seed = seed_base + ep
            env = SnakeEnv(width=w, height=h, seed=seed)
            current_agent = self._resolve_agent(agent, seed)
            obs = env.reset(seed=seed)

            while not obs.done and obs.step_count < step_limit:
                action, _ = current_agent.act(obs)
                obs, _, _, step_info = env.step(action)

            steps = obs.step_count
            score = obs.score
            reason = obs.reason or step_info.get("reason")
            if not reason:
                reason = "TIMEOUT" if steps >= step_limit else "TERMINATED"

            survival.append(steps)
            food.append(score)
            reasons.append(str(reason))

            reason_str = str(reason).upper()
            if "WALL" in reason_str:
                wall_collisions += 1
            elif "BODY" in reason_str:
                body_collisions += 1
            elif "TIMEOUT" in reason_str or steps >= step_limit:
                timeouts += 1

        elapsed = max(time.perf_counter() - start_time, 1e-6)
        total_steps = sum(survival)
        steps_per_sec = total_steps / elapsed

        return {
            "episodes": episodes,
            "survival": survival,
            "food": food,
            "reasons": reasons,
            "mean_survival": mean(survival),
            "std_survival": std_dev(survival),
            "median_survival": median(survival),
            "mean_food": mean(food),
            "std_food": std_dev(food),
            "median_food": median(food),
            "wall_collisions": wall_collisions,
            "body_collisions": body_collisions,
            "timeouts": timeouts,
            "total_steps": total_steps,
            "total_time_seconds": elapsed,
            "steps_per_second": steps_per_sec,
        }

    def measure_memory_stability(
        self,
        steps: int = 1000,
        warmup: int = 100,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Dict[str, float]:
        """Verify memory stability across continuous closed-loop simulation steps."""
        w = width or self.width
        h = height or self.height

        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()

        env = SnakeEnv(width=w, height=h, seed=42)
        agent = FlyBrainAgent()
        obs = env.reset(seed=42)
        agent.reset()

        # Warm-up phase
        for _ in range(warmup):
            action, _ = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
                agent.reset()

        mem_start, _ = tracemalloc.get_traced_memory()

        # Continuous test phase
        for _ in range(steps):
            action, _ = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
                agent.reset()

        mem_end, mem_peak = tracemalloc.get_traced_memory()

        if not was_tracing:
            tracemalloc.stop()

        delta_mb = (mem_end - mem_start) / (1024.0 * 1024.0)
        peak_mb = mem_peak / (1024.0 * 1024.0)

        return {
            "start_mb": mem_start / (1024.0 * 1024.0),
            "end_mb": mem_end / (1024.0 * 1024.0),
            "delta_mb": delta_mb,
            "peak_mb": peak_mb,
        }

    def run_paired_benchmark(
        self,
        episodes: int = 50,
        seed_base: int = 1000,
        max_steps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        assert_thresholds: bool = True,
        memory_steps: int = 1000,
    ) -> Dict[str, Any]:
        """Execute paired multi-episode benchmark between ConnectomeAgent and RandomAgent.
        
        Args:
            episodes: Number of benchmark episodes (default 50).
            seed_base: Base seed for reproducible runs (default 1000).
            max_steps: Maximum step limit per episode (default 2000).
            width: Grid width (default 16).
            height: Grid height (default 16).
            assert_thresholds: If True, asserts speed >= 50 steps/s and memory delta < 5 MB.
            
        Returns:
            JSON-serializable report dictionary containing metrics, statistical hypothesis
            tests, and acceptance criteria verification.
        """
        w = width or self.width
        h = height or self.height
        step_limit = max_steps or self.max_steps

        # 1. Evaluate ConnectomeAgent
        c_res = self.evaluate_agent(
            "connectome",
            episodes=episodes,
            seed_base=seed_base,
            max_steps=step_limit,
            width=w,
            height=h,
        )

        # 2. Evaluate RandomAgent on identical seeds
        r_res = self.evaluate_agent(
            "random",
            episodes=episodes,
            seed_base=seed_base,
            max_steps=step_limit,
            width=w,
            height=h,
        )

        # 3. Statistical hypothesis tests
        t_stat_s, p_val_s, df_s = welch_t_test(c_res["survival"], r_res["survival"])
        u_stat_s, p_u_s = mann_whitney_u(c_res["survival"], r_res["survival"])

        t_stat_f, p_val_f, df_f = welch_t_test(c_res["food"], r_res["food"])
        u_stat_f, p_u_f = mann_whitney_u(c_res["food"], r_res["food"])

        # 4. Simulation throughput speed
        total_steps = c_res["total_steps"] + r_res["total_steps"]
        total_time = c_res["total_time_seconds"] + r_res["total_time_seconds"]
        overall_speed = total_steps / max(total_time, 1e-6)

        # 5. Memory stability over continuous steps
        if memory_steps > 0:
            mem_metrics = self.measure_memory_stability(
                steps=memory_steps,
                warmup=min(100, memory_steps // 2),
                width=w,
                height=h,
            )
            delta_mem_mb = mem_metrics["delta_mb"]
            memory_ok = delta_mem_mb < 5.0
        else:
            mem_metrics = {"start_mb": 0.0, "end_mb": 0.0, "delta_mb": 0.0, "peak_mb": 0.0}
            delta_mem_mb = 0.0
            memory_ok = True

        # 6. Acceptance criteria verification
        survival_superior = c_res["mean_survival"] > r_res["mean_survival"]
        stat_significant = p_val_s < 0.05
        speed_ok = overall_speed >= 50.0
        memory_ok = delta_mem_mb < 5.0

        all_passed = bool(survival_superior and stat_significant and speed_ok and memory_ok)

        # Optional strict threshold enforcement
        if assert_thresholds:
            assert speed_ok, f"Simulation speed {overall_speed:.1f} steps/s is below threshold 50.0 steps/s"
            assert memory_ok, f"Memory growth {delta_mem_mb:.2f} MB exceeds stability threshold 5.0 MB"
            if episodes >= 10:
                assert survival_superior, (
                    f"Connectome survival ({c_res['mean_survival']:.1f}) not greater than "
                    f"random baseline ({r_res['mean_survival']:.1f})"
                )
                assert stat_significant, (
                    f"Survival difference is not statistically significant (p = {p_val_s:.4f} >= 0.05)"
                )

        report = {
            "benchmark_metadata": {
                "episodes": episodes,
                "seed_base": seed_base,
                "grid_size": [w, h],
                "max_steps": step_limit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "connectome": c_res,
            "random": r_res,
            "metrics": {
                "connectome": {
                    "mean_survival": c_res["mean_survival"],
                    "std_survival": c_res["std_survival"],
                    "median_survival": c_res["median_survival"],
                    "mean_food": c_res["mean_food"],
                    "std_food": c_res["std_food"],
                    "median_food": c_res["median_food"],
                },
                "random_baseline": {
                    "mean_survival": r_res["mean_survival"],
                    "std_survival": r_res["std_survival"],
                    "median_survival": r_res["median_survival"],
                    "mean_food": r_res["mean_food"],
                    "std_food": r_res["std_food"],
                    "median_food": r_res["median_food"],
                },
            },
            "statistical_tests": {
                "survival": {
                    "welch_t": t_stat_s,
                    "welch_p": p_val_s,
                    "welch_df": df_s,
                    "mann_whitney_u": u_stat_s,
                    "mann_whitney_p": p_u_s,
                    "significant": stat_significant,
                },
                "food": {
                    "welch_t": t_stat_f,
                    "welch_p": p_val_f,
                    "welch_df": df_f,
                    "mann_whitney_u": u_stat_f,
                    "mann_whitney_p": p_u_f,
                    "significant": bool(p_val_f < 0.05),
                },
            },
            "performance": {
                "steps_per_second": overall_speed,
                "speed_threshold_met": speed_ok,
                "memory_growth_mb": delta_mem_mb,
                "memory_peak_mb": mem_metrics["peak_mb"],
                "memory_threshold_met": memory_ok,
            },
            "acceptance_criteria": {
                "connectome_survival_superiority": survival_superior,
                "survival_statistical_significance": stat_significant,
                "speed_threshold_satisfied": speed_ok,
                "memory_stability_satisfied": memory_ok,
            },
            "passed": all_passed,
        }

        return report


def run_benchmark(
    episodes: int = 50,
    seed_base: int = 1000,
    width: int = 16,
    height: int = 16,
    max_steps: int = 2000,
    **kwargs,
) -> Dict[str, Any]:
    """Run comparative benchmark between ConnectomeAgent and RandomAgent."""
    evaluator = BenchmarkEvaluator(width=width, height=height, max_steps=max_steps)
    return evaluator.run_paired_benchmark(
        episodes=episodes,
        seed_base=seed_base,
        width=width,
        height=height,
        max_steps=max_steps,
        **kwargs,
    )


def run_headless_benchmark(
    episodes: int = 50,
    seed_base: int = 1000,
    width: int = 16,
    height: int = 16,
    max_steps: int = 2000,
    **kwargs,
) -> Dict[str, Any]:
    """Headless benchmark execution alias for test compatibility."""
    return run_benchmark(
        episodes=episodes,
        seed_base=seed_base,
        width=width,
        height=height,
        max_steps=max_steps,
        **kwargs,
    )

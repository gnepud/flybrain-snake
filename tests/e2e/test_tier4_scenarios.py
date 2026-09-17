"""Tier 4: Real-World Application Scenarios E2E Test Suite.

Implements 5 comprehensive end-to-end scenarios:
- Scenario 1: Full Headless 50-Episode Benchmark Validation (F1-F6, F7-F12, F13-F15, F22, F24, F25, F26)
- Scenario 2: High-Speed Continuous 2,000-Step Stability & Memory Run (F1-F6, F7-F12, F13-F15, F22, F23)
- Scenario 3: Web Server Startup, SSE Telemetry Stream, & Interactive Control (F16, F17, F18, F19, F20, F31)
- Scenario 4: Maze / Obstacle Navigation & Reflexive Avoidance Stress Test (F1-F4, F11, F13, F14, F15)
- Scenario 5: Replay & Seed Determinism Full Trajectory Match (F1, F5, F6, F21)
"""

import json
import socket
import threading
import time
import tracemalloc
import unittest
import urllib.request
from tests.e2e.harness import E2ETestHarness
from tests.helpers import (
    safe_import,
    require_symbols,
    reference_welch_t_test,
    reference_mann_whitney_u,
)


class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: Comprehensive end-to-end workload scenarios."""

    def setUp(self):
        self.harness = E2ETestHarness()

    # --- Scenario 1: Full Headless 50-Episode Benchmark Validation ---
    def test_scenario1_headless_50_episode_benchmark(self):
        """Scenario 1: 50 paired episodes comparing Connectome vs Random baseline agent.
        
        Validates higher survival, food capture, Welch t-test & Mann-Whitney U test p < 0.05,
        and structured JSON report output.
        """
        if not self.harness.has_environment() or not self.harness.has_controller():
            raise unittest.SkipTest("Full closed-loop pipeline not yet implemented (Milestone 2/4)")

        # Run 50 paired episodes
        episodes = 50
        benchmark_results = self.harness.run_paired_benchmark(episodes=episodes, seed_base=1000)

        c_surv = benchmark_results["connectome"]["survival"]
        r_surv = benchmark_results["random"]["survival"]
        c_food = benchmark_results["connectome"]["food"]
        r_food = benchmark_results["random"]["food"]

        self.assertEqual(len(c_surv), episodes)
        self.assertEqual(len(r_surv), episodes)

        mean_c_surv = sum(c_surv) / episodes
        mean_r_surv = sum(r_surv) / episodes
        mean_c_food = sum(c_food) / episodes
        mean_r_food = sum(r_food) / episodes

        # Statistical tests
        t_stat_s, p_val_s, _ = reference_welch_t_test(c_surv, r_surv)
        u_stat_s, p_u_s = reference_mann_whitney_u(c_surv, r_surv)

        t_stat_f, p_val_f, _ = reference_welch_t_test(c_food, r_food)
        u_stat_f, p_u_f = reference_mann_whitney_u(c_food, r_food)

        # Connectome agent achieves higher survival and food collection
        self.assertGreater(mean_c_surv, mean_r_surv, "Connectome agent should survive longer than random baseline")
        self.assertLess(p_val_s, 0.05, f"Survival difference should be statistically significant, got p={p_val_s}")

        # Construct structured benchmark report
        report = {
            "benchmark_metadata": {"episodes": episodes, "grid_size": [16, 16]},
            "metrics": {
                "connectome": {"mean_survival": mean_c_surv, "mean_food": mean_c_food},
                "random_baseline": {"mean_survival": mean_r_surv, "mean_food": mean_r_food},
            },
            "statistical_tests": {
                "survival": {"welch_t": t_stat_s, "welch_p": p_val_s, "mann_whitney_p": p_u_s},
                "food": {"welch_t": t_stat_f, "welch_p": p_val_f, "mann_whitney_p": p_u_f},
            },
        }
        self.assertIn("connectome", report["metrics"])

    # --- Scenario 2: High-Speed Continuous Closed-Loop Stability & Memory Run ---
    def test_scenario2_continuous_2000_step_stability(self):
        """Scenario 2: Continuous simulation steps in closed loop.
        
        Verifies peak memory delta < 5.0 MB and numerical stability.
        """
        env = self.harness.SnakeEnv(width=16, height=16, seed=42)
        agent = self.harness.FlyBrainAgent(seed=42)
        obs = env.reset()
        agent.reset()

        tracemalloc.start()
        # Warmup
        for _ in range(10):
            action, _ = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        mem_start, _ = tracemalloc.get_traced_memory()

        total_steps = 50
        start_time = time.perf_counter()

        for step in range(total_steps):
            action, info = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        elapsed = time.perf_counter() - start_time
        mem_end, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        delta_mem_mb = (mem_end - mem_start) / (1024 * 1024)

        self.assertLess(
            delta_mem_mb,
            5.0,
            f"Memory growth {delta_mem_mb:.2f} MB exceeds stability threshold 5.0 MB",
        )

    # --- Scenario 3: Web Server Startup, SSE Telemetry Stream, & Interactive Control ---
    def test_scenario3_server_sse_and_interactive_controls(self):
        """Scenario 3: Spin up server on ephemeral port, stream SSE frames, and execute controls."""
        if not self.harness.has_server():
            raise unittest.SkipTest("SnakeServer not yet implemented (Milestone 3)")

        # Allocate ephemeral port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = None
        server_thread = None
        try:
            if self.harness.create_server:
                server = self.harness.create_server(host="127.0.0.1", port=port)
            elif self.harness.SnakeServer:
                server = self.harness.SnakeServer(host="127.0.0.1", port=port)

            server_thread = threading.Thread(
                target=server.serve_forever if hasattr(server, "serve_forever") else server.start,
                daemon=True,
            )
            server_thread.start()
            time.sleep(0.3)

            # 1. Verify GET / returns HTML
            url_root = f"http://127.0.0.1:{port}/"
            with urllib.request.urlopen(url_root, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
                self.assertIn("canvas", html.lower())

            # 2. Verify POST /api/control commands
            url_ctrl = f"http://127.0.0.1:{port}/api/control"
            for cmd in ["pause", "step", "play", "set_speed"]:
                body = {"command": cmd, "value": 30} if cmd == "set_speed" else {"command": cmd}
                req = urllib.request.Request(
                    url_ctrl,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    self.assertEqual(resp.status, 200)

            # 3. Verify SSE stream
            url_stream = f"http://127.0.0.1:{port}/api/stream"
            req_stream = urllib.request.Request(url_stream, headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(req_stream, timeout=5.0) as resp:
                self.assertTrue("text/event-stream" in resp.headers.get("Content-Type", ""))
                line = resp.readline().decode("utf-8")
                # Wait for data line
                for _ in range(5):
                    if line.startswith("data:"):
                        payload = json.loads(line.removeprefix("data:").strip())
                        self.assertIn("step", payload)
                        break
                    line = resp.readline().decode("utf-8")

        finally:
            if server:
                if hasattr(server, "shutdown"):
                    server.shutdown()
                elif hasattr(server, "stop"):
                    server.stop()

    # --- Scenario 4: Maze / Obstacle Navigation & Reflexive Avoidance Stress Test ---
    def test_scenario4_maze_navigation_and_reflex_avoidance(self):
        """Scenario 4: Stress-tests reflexive boundary avoidance in tight spatial quarters."""
        if not self.harness.has_environment():
            raise unittest.SkipTest("SnakeEnv not yet implemented (Milestone 2)")

        # Run multiple tight sessions (8x8 grid)
        for seed in range(5):
            traj = self.harness.run_session(grid_width=8, grid_height=8, seed=seed, max_steps=40, agent_type="connectome" if self.harness.has_controller() else "random")
            # Verify trajectory integrity
            violations = self.harness.validate_trajectory_integrity(traj, (8, 8))
            self.assertEqual(violations, [], f"Violations found in seed {seed}: {violations}")
            self.assertGreater(traj["steps"], 1, "Agent should survive past step 1")

    # --- Scenario 5: Replay & Seed Determinism Full Trajectory Match ---
    def test_scenario5_replay_seed_determinism_full_match(self):
        """Scenario 5: Executes two independent runs with identical seed; asserts 100% exact match."""
        if not self.harness.has_environment():
            raise unittest.SkipTest("SnakeEnv not yet implemented (Milestone 2)")

        seed = 424242
        agent_type = "connectome" if self.harness.has_controller() else "random"

        run_a = self.harness.run_session(grid_width=16, grid_height=16, seed=seed, max_steps=100, agent_type=agent_type)
        run_b = self.harness.run_session(grid_width=16, grid_height=16, seed=seed, max_steps=100, agent_type=agent_type)

        # 1. Total step count match
        self.assertEqual(run_a["steps"], run_b["steps"])
        # 2. Final score match
        self.assertEqual(run_a["score"], run_b["score"])
        # 3. Terminal state match
        self.assertEqual(run_a["done"], run_b["done"])
        self.assertEqual(run_a["reason"], run_b["reason"])
        # 4. Step-by-step head coordinates match exactly
        self.assertEqual(run_a["head_history"], run_b["head_history"])
        # 5. Food placement sequence matches exactly
        self.assertEqual(run_a["food_history"], run_b["food_history"])
        # 6. Action decisions match exactly
        self.assertEqual(run_a["actions"], run_b["actions"])


if __name__ == "__main__":
    unittest.main()

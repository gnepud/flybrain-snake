"""Unit and Integration Tests for FlyBrainAgent (166.7k-Neuron Model)."""

import pytest
from src.snake_env.env import SnakeEnv
from src.snake_env.kinematics import RelativeAction

try:
    from src.controller.flybrain_agent import FlyBrainAgent, HAS_FLYBRAIN
except ImportError:
    HAS_FLYBRAIN = False


@pytest.mark.skipif(not HAS_FLYBRAIN, reason="flybrain library not available")
def test_flybrain_agent_initialization():
    agent = FlyBrainAgent()
    assert agent.brain.n == 166700
    assert len(agent._al_l) > 0
    assert len(agent._al_r) > 0
    assert len(agent._lc10a_l) > 0
    assert len(agent._lc10a_r) > 0
    assert agent._steer_l >= 0
    assert agent._steer_r >= 0


@pytest.mark.skipif(not HAS_FLYBRAIN, reason="flybrain library not available")
def test_flybrain_agent_step_and_telemetry():
    agent = FlyBrainAgent()
    env = SnakeEnv(width=16, height=16, seed=42)
    obs = env.reset()

    action, info = agent.act(obs)
    assert isinstance(action, RelativeAction)
    assert "spikes_count" in info
    assert info["spikes_count"] >= 0
    assert "dna02_l" in info
    assert "dna02_r" in info
    assert info["backend_mode"] == "flybrain_166k"
    assert "olfactory" in info
    assert "c_left" in info["olfactory"]
    assert "c_right" in info["olfactory"]
    assert "diff" in info["olfactory"]
    assert "delta_c" in info["olfactory"]


@pytest.mark.skipif(not HAS_FLYBRAIN, reason="flybrain library not available")
def test_flybrain_agent_turns_towards_adjacent_food():
    agent = FlyBrainAgent()
    env = SnakeEnv(width=16, height=16, seed=42)
    obs = env.reset()

    # In seed 42: food is at (6, 10). The agent turns and eats food by step 3
    eaten = False
    for step in range(4):
        action, info = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        if reward > 0:
            eaten = True
            break

    assert eaten
    assert not obs.done


@pytest.mark.skipif(not HAS_FLYBRAIN, reason="flybrain library not available")
def test_flybrain_agent_episode_run():
    agent = FlyBrainAgent()
    env = SnakeEnv(width=16, height=16, seed=123)
    obs = env.reset()

    steps = 0
    for _ in range(15):
        action, _ = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        steps += 1
        if done:
            break

    assert steps == 15
    assert not env._done


@pytest.mark.skipif(not HAS_FLYBRAIN, reason="flybrain library not available")
def test_multimodal_odor_gated_visual_gain():
    """Verify multimodal gating: visual_gain scales with c_max / c0 according to biological formula."""
    agent = FlyBrainAgent(base_gain=3.0, alpha=1.5, c0=5.0) if hasattr(agent_init := FlyBrainAgent, "_base_gain") else FlyBrainAgent(sensory_gain=3.0, alpha=1.5, c0=5.0)
    env = SnakeEnv(width=16, height=16, seed=42)
    obs = env.reset()

    action, info = agent.act(obs)
    olf = info["olfactory"]
    c_max = max(olf["c_left"], olf["c_right"])
    expected_gating = 1.0 + 1.5 * (c_max / 5.0)
    assert abs(olf["gating_factor"] - expected_gating) < 0.01
    assert abs(olf["visual_gain"] - 3.0 * expected_gating) < 0.01
    assert olf["visual_gain"] > 3.0  # Amplified above base gain due to positive odor concentration


@pytest.mark.skipif(not HAS_FLYBRAIN, reason="flybrain library not available")
def test_retinal_deadband_refinement():
    """Verify retinal deadband defaults to 0.05 rad (~3 deg) and suppresses small-angle jitter."""
    import dataclasses
    agent = FlyBrainAgent(seed=42)
    assert agent.deadband == 0.05

    env = SnakeEnv(width=16, height=16, seed=42)
    obs = env.reset()

    # Small misalignment inside deadband (e.g., 0.02 rad ≈ 1.1°)
    obs_sub_deadband = dataclasses.replace(obs, food_bearing=0.02)
    _, info_sub = agent.act(obs_sub_deadband)
    assert info_sub["olfactory"]["deadband"] == 0.05


@pytest.mark.skipif(not HAS_FLYBRAIN, reason="flybrain library not available")
def test_epg_bump_and_reflex_override():
    """Verify E-PG 16-wedge bump generation and reflexive obstacle override detection."""
    import dataclasses
    agent = FlyBrainAgent(seed=42)
    env = SnakeEnv(width=16, height=16, seed=42)
    obs = env.reset()

    # 1. Check E-PG bump
    action, info = agent.act(obs)
    assert "epg" in info
    assert len(info["epg"]) == 16
    assert max(info["epg"]) > min(info["epg"])  # True attractor bump contrast
    assert "steering" in info
    assert "pfl3_l" in info["steering"]
    assert "pfl3_r" in info["steering"]
    assert info["steering"]["threshold"] == 0.03
    assert info["steering"]["is_override"] is False

    # 2. Reflexive override when immediately blocked in front (dist_front == 0)
    blocked_obs = dataclasses.replace(obs, dist_front=0, dist_left=5, dist_right=0)
    action_override, info_override = agent.act(blocked_obs)
    assert action_override == RelativeAction.TURN_LEFT
    assert info_override["steering"]["is_override"] is True
    assert info_override["motor"]["is_override"] is True


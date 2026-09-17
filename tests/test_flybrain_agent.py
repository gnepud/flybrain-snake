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

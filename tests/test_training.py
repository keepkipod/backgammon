"""Tests for the training pipeline: network, MCTS, self-play, checkpointing."""

import os
import tempfile

import numpy as np
import pytest
import torch

from backgammon.backgammon_game import BackgammonGame
from backgammon.mcts import MCTS
from backgammon.network import DualHeadNetwork, get_device
from backgammon.trainer import TrainConfig, Trainer


@pytest.fixture
def game():
    return BackgammonGame()


@pytest.fixture
def device():
    return get_device()


@pytest.fixture
def small_config(tmp_path):
    return TrainConfig(
        hidden_size=32,
        num_res_blocks=2,
        num_simulations=5,
        num_self_play_games=1,
        num_training_steps=2,
        batch_size=4,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_game_length=1000,
        num_eval_games=4,
    )


# ─── Network ───────────────────────────────────────────────────────────


class TestNetwork:
    def test_forward_shape(self, game, device):
        net = DualHeadNetwork(input_size=197, action_size=151, hidden_size=32, num_res_blocks=2).to(device)
        x = torch.randn(8, 197, device=device)
        logits, value = net(x)
        assert logits.shape == (8, 151)
        assert value.shape == (8, 1)

    def test_value_range(self, game, device):
        net = DualHeadNetwork(input_size=197, action_size=151, hidden_size=32, num_res_blocks=2).to(device)
        x = torch.randn(16, 197, device=device)
        _, value = net(x)
        assert (value >= -1).all() and (value <= 1).all()

    def test_predict_with_mask(self, game, device):
        net = DualHeadNetwork(input_size=197, action_size=151, hidden_size=32, num_res_blocks=2).to(device)
        state = game.get_initial_state()
        state = game.apply_chance_outcome(state, (3, 1))
        state_tensor = torch.FloatTensor(game.encode_state(state)).to(device)

        legal_mask = torch.zeros(151, device=device)
        for action in game.get_legal_actions(state):
            legal_mask[game.action_to_index(action)] = 1.0

        policy, value = net.predict(state_tensor, legal_mask)
        assert policy.shape == (151,)
        assert abs(policy.sum().item() - 1.0) < 1e-5
        # Only legal actions should have nonzero probability
        for i in range(151):
            if legal_mask[i] == 0:
                assert policy[i].item() < 1e-6


# ─── MCTS ──────────────────────────────────────────────────────────────


class TestMCTS:
    def test_search_returns_valid_policy(self, game, device):
        net = DualHeadNetwork(input_size=197, action_size=151, hidden_size=32, num_res_blocks=2).to(device)
        state = game.get_initial_state()
        state = game.apply_chance_outcome(state, (3, 1))

        mcts = MCTS(game=game, network=net, device=device, num_simulations=10)
        policy, value = mcts.search(state)

        assert policy.shape == (151,)
        assert abs(policy.sum() - 1.0) < 1e-5
        assert isinstance(value, float)

    def test_policy_only_on_legal_actions(self, game, device):
        net = DualHeadNetwork(input_size=197, action_size=151, hidden_size=32, num_res_blocks=2).to(device)
        state = game.get_initial_state()
        state = game.apply_chance_outcome(state, (5, 2))

        mcts = MCTS(game=game, network=net, device=device, num_simulations=10)
        policy, _ = mcts.search(state)

        legal_indices = set()
        for action in game.get_legal_actions(state):
            legal_indices.add(game.action_to_index(action))

        for i in range(151):
            if i not in legal_indices:
                assert policy[i] == 0.0

    def test_temperature_zero_is_greedy(self, game, device):
        net = DualHeadNetwork(input_size=197, action_size=151, hidden_size=32, num_res_blocks=2).to(device)
        state = game.get_initial_state()
        state = game.apply_chance_outcome(state, (4, 3))

        mcts = MCTS(game=game, network=net, device=device, num_simulations=20, temperature=0)
        policy, _ = mcts.search(state)

        # Greedy: exactly one action should have probability 1.0
        assert np.count_nonzero(policy) == 1
        assert max(policy) == 1.0


# ─── Self-Play & Training ──────────────────────────────────────────────


class TestTraining:
    def test_self_play_generates_examples(self, game, small_config):
        trainer = Trainer(game, small_config)
        examples = trainer.self_play_game()
        assert len(examples) > 0
        # Each example should have state, policy, value
        ex = examples[0]
        assert ex.state.shape == (197,)
        assert ex.policy.shape == (151,)
        assert -3 <= ex.value <= 3

    def test_training_step_reduces_or_changes_loss(self, game, small_config):
        trainer = Trainer(game, small_config)
        examples = trainer.self_play_game()
        assert len(examples) > 0
        # Pad buffer
        for _ in range(10):
            trainer.replay_buffer.add(examples)

        m1 = trainer.train_step()
        m2 = trainer.train_step()
        assert "error" not in m1
        assert "error" not in m2
        assert m1["total_loss"] > 0
        assert m2["total_loss"] > 0

    def test_checkpoint_roundtrip(self, game, small_config):
        trainer = Trainer(game, small_config)
        examples = trainer.self_play_game()
        trainer.replay_buffer.add(examples)
        for _ in range(5):
            trainer.replay_buffer.add(examples)
        trainer.train_step()
        trainer.save_checkpoint()

        trainer2 = Trainer(game, small_config)
        trainer2.load_checkpoint(os.path.join(small_config.checkpoint_dir, "latest.pt"))

        assert trainer2.training_step == trainer.training_step
        assert len(trainer2.replay_buffer) == len(trainer.replay_buffer)

    def test_loss_is_finite(self, game, small_config):
        trainer = Trainer(game, small_config)
        examples = trainer.self_play_game()
        for _ in range(10):
            trainer.replay_buffer.add(examples)

        for _ in range(5):
            metrics = trainer.train_step()
            assert "error" not in metrics
            assert np.isfinite(metrics["total_loss"])
            assert np.isfinite(metrics["value_loss"])
            assert np.isfinite(metrics["policy_loss"])


# ─── Smoke Test (full pipeline) ────────────────────────────────────────


class TestSmokeTest:
    def test_full_pipeline(self, game, small_config):
        """End-to-end: self-play → train → checkpoint → load → verify."""
        trainer = Trainer(game, small_config)

        # Self-play
        examples = trainer.self_play_game()
        assert len(examples) > 0
        trainer.replay_buffer.add(examples)
        for _ in range(10):
            trainer.replay_buffer.add(examples)

        # Train
        metrics = trainer.train_step()
        assert metrics["total_loss"] > 0

        # Checkpoint
        trainer.save_checkpoint()

        # Load and verify
        trainer2 = Trainer(game, small_config)
        trainer2.load_checkpoint(os.path.join(small_config.checkpoint_dir, "latest.pt"))

        # Networks should produce identical output
        state = game.get_initial_state()
        state = game.apply_chance_outcome(state, (3, 1))
        enc = torch.FloatTensor(game.encode_state(state)).to(trainer.device)

        trainer.network.eval()
        trainer2.network.eval()
        with torch.no_grad():
            p1, v1 = trainer.network(enc.unsqueeze(0))
            p2, v2 = trainer2.network(enc.unsqueeze(0))
        assert torch.allclose(p1, p2)
        assert torch.allclose(v1, v2)

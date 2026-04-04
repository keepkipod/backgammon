"""Tests for the Backgammon Gymnasium environment."""

import random

import numpy as np
import pytest

from backgammon.backgammon_game import ACTION_SPACE_SIZE, BackgammonGame
from backgammon.env import BackgammonEnv, random_opponent


@pytest.fixture
def env():
    return BackgammonEnv(opponent=random_opponent, seed=42)


@pytest.fixture
def self_play_env():
    return BackgammonEnv(opponent=None, seed=42)


# ─── Basic Interface ────────────────────────────────────────────────────


class TestBasicInterface:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert obs.shape == (197,)
        assert obs.dtype == np.float32
        assert "current_player" in info
        assert "legal_actions" in info

    def test_observation_space(self, env):
        assert env.observation_space.shape == (197,)

    def test_action_space(self, env):
        assert env.action_space.n == ACTION_SPACE_SIZE

    def test_player_0_after_reset(self, env):
        obs, info = env.reset()
        assert info["current_player"] == 0

    def test_reset_with_seed(self, env):
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        assert np.array_equal(obs1, obs2)

    def test_dice_rolled_after_reset(self, env):
        obs, info = env.reset()
        assert len(info["dice_remaining"]) > 0


# ─── Action Masking ─────────────────────────────────────────────────────


class TestActionMasking:
    def test_mask_shape(self, env):
        env.reset()
        mask = env.legal_action_mask()
        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == np.int8

    def test_mask_has_legal_moves(self, env):
        env.reset()
        mask = env.legal_action_mask()
        assert mask.sum() > 0

    def test_mask_matches_legal_actions(self, env):
        env.reset()
        mask = env.legal_action_mask()
        info = env._info()
        game = env.game

        for action in info["legal_actions"]:
            idx = game.action_to_index(action)
            assert mask[idx] == 1

    def test_illegal_action_penalized(self, env):
        env.reset()
        mask = env.legal_action_mask()
        # Find an illegal action
        illegal_idx = None
        for i in range(ACTION_SPACE_SIZE):
            if mask[i] == 0:
                illegal_idx = i
                break
        assert illegal_idx is not None

        obs, reward, term, trunc, info = env.step(illegal_idx)
        assert reward == -0.1
        assert info.get("illegal_move") is True
        assert not term


# ─── Step Mechanics ─────────────────────────────────────────────────────


class TestStepMechanics:
    def test_step_returns_correct_shape(self, env):
        env.reset()
        mask = env.legal_action_mask()
        legal_idx = np.where(mask == 1)[0][0]
        obs, reward, term, trunc, info = env.step(int(legal_idx))
        assert obs.shape == (197,)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)

    def test_step_advances_state(self, env):
        obs1, _ = env.reset(seed=123)
        mask = env.legal_action_mask()
        legal_idx = np.where(mask == 1)[0][0]
        obs2, _, _, _, _ = env.step(int(legal_idx))
        assert not np.array_equal(obs1, obs2)

    def test_multiple_submoves_in_turn(self, env):
        """Agent can make multiple sub-moves before opponent plays."""
        env.reset(seed=50)
        steps_taken = 0
        for _ in range(10):
            mask = env.legal_action_mask()
            if mask.sum() == 0:
                break
            legal_idx = np.where(mask == 1)[0][0]
            obs, reward, term, trunc, info = env.step(int(legal_idx))
            steps_taken += 1
            if term:
                break
        assert steps_taken >= 1


# ─── Self-Play Mode ────────────────────────────────────────────────────


class TestSelfPlay:
    def test_self_play_alternates_players(self, self_play_env):
        obs, info = self_play_env.reset()
        player_before = info["current_player"]

        # Play all sub-moves for this player's turn
        while True:
            mask = self_play_env.legal_action_mask()
            if mask.sum() == 0:
                break
            legal_idx = np.where(mask == 1)[0][0]
            obs, _, term, _, info = self_play_env.step(int(legal_idx))
            if term:
                return
            if info["current_player"] != player_before:
                break

        assert info["current_player"] != player_before


# ─── Random Agent Simulation ───────────────────────────────────────────


class TestRandomSimulation:
    def _play_random_game(self, seed: int) -> dict:
        env = BackgammonEnv(opponent=random_opponent, seed=seed)
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        max_steps = 5000

        while steps < max_steps:
            mask = env.legal_action_mask()
            legal_indices = np.where(mask == 1)[0]
            if len(legal_indices) == 0:
                # Shouldn't happen with opponent mode, but safety valve
                break
            action = int(np.random.choice(legal_indices))
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            if term:
                break

        return {
            "terminated": term if steps < max_steps else False,
            "reward": total_reward,
            "steps": steps,
        }

    def test_single_random_game(self):
        result = self._play_random_game(seed=42)
        assert result["terminated"] is True

    @pytest.mark.parametrize("seed", range(50))
    def test_50_random_games_no_crash(self, seed):
        result = self._play_random_game(seed)
        # Game should terminate (random games typically end well under 5000 steps)
        assert result["terminated"] is True

    def test_reward_distribution(self):
        """Check that wins and losses both occur with random play."""
        wins = 0
        losses = 0
        for seed in range(200):
            result = self._play_random_game(seed)
            if result["reward"] > 0:
                wins += 1
            elif result["reward"] < 0:
                losses += 1

        # Both outcomes should happen (random vs random is roughly 50/50)
        assert wins > 20, f"Too few wins: {wins}/200"
        assert losses > 20, f"Too few losses: {losses}/200"

    def test_gammon_backgammon_occur(self):
        """Gammons and backgammons should occasionally happen in random play."""
        gammons = 0
        for seed in range(500):
            result = self._play_random_game(seed)
            if abs(result["reward"]) >= 2:
                gammons += 1

        assert gammons > 0, "No gammons/backgammons in 500 random games"

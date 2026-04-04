"""Gymnasium environment for Backgammon.

Wraps the BackgammonGame engine so any RL agent can plug in via the
standard Gymnasium reset() / step() interface.

The environment handles:
  - Dice rolling (chance events) internally
  - Opponent turns (via a pluggable opponent policy)
  - Action masking for illegal moves
  - Sub-move sequencing within a turn

Observation: 197-dim float32 vector (see BackgammonGame.encode_state)
Action: Discrete(151) — index into the sub-move action space
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backgammon.backgammon_game import (
    ACTION_SPACE_SIZE,
    BackgammonGame,
    BackgammonState,
)

# Opponent policy type: given a game, state, and legal actions, return an action
OpponentPolicy = Callable[[BackgammonGame, BackgammonState, list[tuple[int, int]]], tuple[int, int]]


def random_opponent(game: BackgammonGame, state: BackgammonState, actions: list[tuple[int, int]]) -> tuple[int, int]:
    """Random opponent — picks a legal action uniformly at random."""
    return random.choice(actions)


class BackgammonEnv(gym.Env):
    """Gymnasium environment for Backgammon.

    The agent always plays as player 0. The opponent (player 1) is controlled
    by a pluggable policy function. Set opponent=None for self-play mode
    where step() controls both players.

    Args:
        opponent: Policy function for player 1. None for self-play mode.
        seed: Random seed for dice rolls.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: Optional[OpponentPolicy] = random_opponent,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.game = BackgammonGame()
        self.opponent = opponent
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        # 197-dim observation from encode_state
        self.observation_space = spaces.Box(
            low=-1.0, high=15.0, shape=(197,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self.state: Optional[BackgammonState] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.RandomState(seed)

        self.state = self.game.get_initial_state()

        if self.opponent is not None:
            self._advance_to_agent_turn()
        else:
            self._roll_dice()

        obs = self.game.encode_state(self.state)
        return obs, self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a sub-move action.

        In self-play mode (opponent=None), step() alternates between both players.
        With an opponent, step() handles the agent's sub-moves and then plays
        the full opponent turn automatically before returning.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.state is not None, "Call reset() first"

        # Decode action index to (source, die)
        src_die = self.game.index_to_action(action)

        # Validate it's a legal action
        legal = self.game.get_legal_actions(self.state)
        if src_die not in legal:
            # Illegal action — return negative reward signal, no state change
            obs = self.game.encode_state(self.state)
            return obs, -0.1, False, False, {**self._info(), "illegal_move": True}

        # Apply the sub-move
        self.state = self.game.apply_action(self.state, src_die)

        # Check terminal
        if self.game.is_terminal(self.state):
            reward = self.game.get_reward(self.state, 0)
            obs = self.game.encode_state(self.state)
            return obs, float(reward), True, False, self._info()

        # If still player 0's turn with dice remaining, return for next sub-move
        if (
            not self.state.needs_dice
            and self.state.current_player == 0
            and self.state.dice_remaining
        ):
            obs = self.game.encode_state(self.state)
            return obs, 0.0, False, False, self._info()

        # Turn ended or switched — advance through opponent/no-move turns
        if self.opponent is not None:
            self._advance_to_agent_turn()

            if self.game.is_terminal(self.state):
                reward = self.game.get_reward(self.state, 0)
                obs = self.game.encode_state(self.state)
                return obs, float(reward), True, False, self._info()
        else:
            # Self-play: just roll dice if needed
            if self.state.needs_dice:
                self._roll_dice()

        obs = self.game.encode_state(self.state)
        return obs, 0.0, False, False, self._info()

    def legal_action_mask(self) -> np.ndarray:
        """Return a binary mask over the action space (1 = legal, 0 = illegal)."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if self.state is None:
            return mask
        legal = self.game.get_legal_actions(self.state)
        for action in legal:
            idx = self.game.action_to_index(action)
            mask[idx] = 1
        return mask

    def _roll_dice(self):
        """Roll dice for the current chance node."""
        if self.state is not None and self.game.is_chance_node(self.state):
            outcomes = self.game.get_chance_outcomes(self.state)
            rolls = [o for o, _ in outcomes]
            probs = [p for _, p in outcomes]
            idx = self._np_rng.choice(len(rolls), p=probs)
            self.state = self.game.apply_chance_outcome(self.state, rolls[idx])

    def _advance_to_agent_turn(self):
        """Advance game state until player 0 has legal moves or game is over.

        Handles dice rolling, opponent sub-moves, and skipping turns where
        either player has no legal moves. Loops until player 0 is ready to act.
        """
        max_iterations = 500  # safety cap against infinite loops
        for _ in range(max_iterations):
            if self.game.is_terminal(self.state):
                return

            # Roll dice if needed
            if self.state.needs_dice:
                self._roll_dice()
                if self.game.is_terminal(self.state):
                    return

            # If it's the opponent's turn, play one sub-move
            if self.state.current_player == 1:
                legal = self.game.get_legal_actions(self.state)
                if legal:
                    action = self.opponent(self.game, self.state, legal)
                    self.state = self.game.apply_action(self.state, action)
                else:
                    # Opponent has no moves — skip their turn
                    self.state.dice_remaining = []
                    self.state.current_player = 0
                    self.state.needs_dice = True
                continue

            # It's player 0's turn
            legal = self.game.get_legal_actions(self.state)
            if legal:
                return  # Player 0 has moves — done

            # Player 0 has no legal moves — skip their turn
            self.state.dice_remaining = []
            self.state.current_player = 1
            self.state.needs_dice = True

    def _info(self) -> dict[str, Any]:
        """Build the info dict."""
        info: dict[str, Any] = {}
        if self.state is not None:
            info["current_player"] = self.state.current_player
            info["dice_remaining"] = list(self.state.dice_remaining)
            info["legal_actions"] = self.game.get_legal_actions(self.state)
            info["borne_off"] = list(self.state.borne_off)
            info["bar"] = list(self.state.bar)
        return info

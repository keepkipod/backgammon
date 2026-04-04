"""Abstract GameState interface for multi-game support.

All future games (Backgammon, Chess, Othello, etc.) implement this interface.
The MCTS and neural network modules depend only on this abstraction.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# Type aliases — concrete games define what these actually are
State = Any
Action = Any
Player = int
ChanceOutcome = Any


class GameState(ABC):
    """Abstract base class for a game's rules and state transitions.

    A game implementation must define every method here. The interface
    separates player decisions from stochastic events (chance outcomes)
    to support both deterministic and stochastic games.
    """

    @abstractmethod
    def get_initial_state(self) -> State:
        """Return the starting state of the game."""

    @abstractmethod
    def get_current_player(self, state: State) -> Player:
        """Return the id of the player whose turn it is (0 or 1)."""

    @abstractmethod
    def get_legal_actions(self, state: State) -> list[Action]:
        """Return all legal actions for the current player in this state."""

    @abstractmethod
    def apply_action(self, state: State, action: Action) -> State:
        """Apply an action and return the resulting state. Must not mutate the input."""

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Return True if the game is over."""

    @abstractmethod
    def get_reward(self, state: State, player: Player) -> float:
        """Return the reward for the given player in a terminal state.

        Only meaningful when is_terminal(state) is True.
        """

    @abstractmethod
    def is_chance_node(self, state: State) -> bool:
        """Return True if the next transition is a chance event (e.g., dice roll).

        Deterministic games should always return False.
        """

    @abstractmethod
    def get_chance_outcomes(self, state: State) -> list[tuple[ChanceOutcome, float]]:
        """Return all possible chance outcomes with their probabilities.

        Each entry is (outcome, probability). Probabilities must sum to 1.
        Only meaningful when is_chance_node(state) is True.
        """

    @abstractmethod
    def apply_chance_outcome(self, state: State, outcome: ChanceOutcome) -> State:
        """Apply a chance outcome and return the resulting state."""

    @abstractmethod
    def encode_state(self, state: State) -> np.ndarray:
        """Encode the state as a numpy array suitable for neural network input."""

    @abstractmethod
    def get_action_space_size(self) -> int:
        """Return the size of the fixed action space for the policy head."""

    @abstractmethod
    def action_to_index(self, action: Action) -> int:
        """Map an action to its index in the fixed action space."""

    @abstractmethod
    def index_to_action(self, index: int) -> Action:
        """Map an index in the fixed action space back to an action."""

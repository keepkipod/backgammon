"""Backgammon game implementation.

Board layout (player 0's perspective):
  Points 0-23 represent the 24 points on the board.
  Point 0 is player 0's home board (bearing off destination).
  Point 23 is player 1's home board.

  Positive values = player 0's checkers.
  Negative values = player 1's checkers.

  Player 0 moves from high points toward point 0 (descending).
  Player 1 moves from low points toward point 23 (ascending).

Sub-move action encoding:
  Each sub-move is (source, die_value) where:
    source: 0-23 (board point) or 24 (bar)
    die_value: 1-6
  Total action space: 25 sources × 6 die values = 150
  Plus a special NO_MOVE action (index 150) for when no legal moves exist.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from backgammon.game_state import GameState

# Constants
NUM_POINTS = 24
NUM_CHECKERS = 15
BAR = 24  # Virtual "point" index for the bar
ACTION_SPACE_SIZE = 151  # 25 sources * 6 dice + 1 no-move
NO_MOVE_INDEX = 150

# All 21 distinct dice rolls with probabilities
DICE_ROLLS: list[tuple[tuple[int, int], float]] = []
for d1 in range(1, 7):
    for d2 in range(d1, 7):
        if d1 == d2:
            DICE_ROLLS.append(((d1, d2), 1 / 36))
        else:
            DICE_ROLLS.append(((d1, d2), 2 / 36))


@dataclass
class BackgammonState:
    """Complete state of a backgammon game."""

    # Board: points[i] > 0 means player 0 has that many checkers, < 0 means player 1
    points: list[int] = field(default_factory=lambda: [0] * NUM_POINTS)

    # Checkers on the bar for each player
    bar: list[int] = field(default_factory=lambda: [0, 0])

    # Checkers borne off for each player
    borne_off: list[int] = field(default_factory=lambda: [0, 0])

    # Current player (0 or 1)
    current_player: int = 0

    # Remaining dice values to use this turn (list of ints)
    dice_remaining: list[int] = field(default_factory=list)

    # Whether we're waiting for a dice roll (chance node)
    needs_dice: bool = True

    # Game over flag
    game_over: bool = False
    winner: int = -1


def _initial_points() -> list[int]:
    """Standard backgammon starting position."""
    points = [0] * NUM_POINTS
    # Player 0 (positive) — moves toward point 0
    points[23] = 2
    points[12] = 5
    points[7] = 3
    points[5] = 5
    # Player 1 (negative) — moves toward point 23
    points[0] = -2
    points[11] = -5
    points[16] = -3
    points[18] = -5
    return points


def _checker_count(state: BackgammonState, player: int) -> int:
    """Count total checkers for a player (on board + bar + borne off)."""
    if player == 0:
        on_board = sum(v for v in state.points if v > 0)
    else:
        on_board = sum(-v for v in state.points if v < 0)
    return on_board + state.bar[player] + state.borne_off[player]


def _own_checker_at(state: BackgammonState, player: int, point: int) -> int:
    """How many of player's own checkers are at a point."""
    if player == 0:
        return max(0, state.points[point])
    else:
        return max(0, -state.points[point])


def _opponent_checker_at(state: BackgammonState, player: int, point: int) -> int:
    """How many of opponent's checkers are at a point."""
    return _own_checker_at(state, 1 - player, point)


def _destination(player: int, source: int, die: int) -> int:
    """Calculate destination point for a sub-move. Returns -1 if bearing off."""
    if player == 0:
        dest = source - die
    else:
        dest = source + die

    if dest < 0 or dest >= NUM_POINTS:
        return -1  # bearing off
    return dest


def _can_bear_off(state: BackgammonState, player: int) -> bool:
    """Check if all of a player's checkers are in their home board."""
    if state.bar[player] > 0:
        return False
    if player == 0:
        # Home board is points 0-5
        for i in range(6, NUM_POINTS):
            if state.points[i] > 0:
                return False
    else:
        # Home board is points 18-23
        for i in range(0, 18):
            if state.points[i] < 0:
                return False
    return True


def _farthest_checker(state: BackgammonState, player: int) -> int:
    """Return the farthest point from home that has the player's checker.

    For player 0: highest point index with positive checkers.
    For player 1: lowest point index with negative checkers.
    Returns -1 if no checkers on board.
    """
    if player == 0:
        for i in range(NUM_POINTS - 1, -1, -1):
            if state.points[i] > 0:
                return i
        return -1
    else:
        for i in range(NUM_POINTS):
            if state.points[i] < 0:
                return i
        return -1


def _get_sub_moves_for_die(state: BackgammonState, player: int, die: int) -> list[tuple[int, int]]:
    """Get all legal sub-moves for a single die value.

    Returns list of (source, die) tuples.
    """
    moves = []

    # Must enter from bar first
    if state.bar[player] > 0:
        if player == 0:
            dest = NUM_POINTS - die  # entering from opponent's home board
        else:
            dest = die - 1

        if 0 <= dest < NUM_POINTS:
            opp_count = _opponent_checker_at(state, player, dest)
            if opp_count <= 1:
                moves.append((BAR, die))
        return moves  # When on bar, can ONLY enter — no other moves

    # Regular moves from board points
    for src in range(NUM_POINTS):
        if _own_checker_at(state, player, src) == 0:
            continue

        dest = _destination(player, src, die)

        if dest == -1:
            # Bearing off
            if not _can_bear_off(state, player):
                continue
            # Can bear off exactly, or if this is the farthest checker and die is larger
            farthest = _farthest_checker(state, player)
            if player == 0:
                if src == farthest or src - die >= 0:
                    # src - die >= 0 means exact bear off; src == farthest allows larger die
                    if src - die < 0 and src != farthest:
                        continue
                    moves.append((src, die))
            else:
                if src == farthest or src + die <= 23:
                    if src + die > 23 and src != farthest:
                        continue
                    moves.append((src, die))
        else:
            # Regular move — destination must not be blocked
            if _opponent_checker_at(state, player, dest) <= 1:
                moves.append((src, die))

    return moves


def _apply_sub_move(state: BackgammonState, player: int, source: int, die: int) -> BackgammonState:
    """Apply a single sub-move and return a new state. Does not mutate input."""
    new_state = copy.deepcopy(state)

    # Remove checker from source
    if source == BAR:
        new_state.bar[player] -= 1
        if player == 0:
            dest = NUM_POINTS - die
        else:
            dest = die - 1
    else:
        if player == 0:
            new_state.points[source] -= 1
        else:
            new_state.points[source] += 1
        dest = _destination(player, source, die)

    if dest == -1:
        # Bearing off
        new_state.borne_off[player] += 1
    else:
        # Hit opponent's blot
        if _opponent_checker_at(state, player, dest) == 1:
            if player == 0:
                new_state.points[dest] = 0  # remove opponent
            else:
                new_state.points[dest] = 0
            new_state.bar[1 - player] += 1

        # Place checker at destination
        if player == 0:
            new_state.points[dest] += 1
        else:
            new_state.points[dest] -= 1

    # Remove used die
    new_state.dice_remaining.remove(die)

    return new_state


def _generate_all_turn_sequences(
    state: BackgammonState, player: int, dice: list[int]
) -> list[list[tuple[int, int]]]:
    """Generate all possible complete turn sequences (lists of sub-moves).

    Backgammon rules require:
    1. Use both dice if possible.
    2. If only one die can be used, must use the larger one.
    3. With doubles, use as many of the 4 dice as possible.

    Returns a list of move sequences, where each sequence is a list of (source, die) tuples.
    """
    if not dice:
        return [[]]

    # Try all possible sub-moves for each remaining die value
    results: list[list[tuple[int, int]]] = []

    # Use unique dice values to avoid duplicate branches
    tried_dice = set()
    for die in dice:
        if die in tried_dice:
            continue
        tried_dice.add(die)

        sub_moves = _get_sub_moves_for_die(state, player, die)
        for src, d in sub_moves:
            new_state = _apply_sub_move(state, player, src, d)
            remaining = list(dice)
            remaining.remove(d)
            continuations = _generate_all_turn_sequences(new_state, player, remaining)
            for cont in continuations:
                results.append([(src, d)] + cont)

    if not results:
        return [[]]  # No moves possible — empty sequence

    # Enforce maximization: use as many dice as possible
    max_len = max(len(seq) for seq in results)
    results = [seq for seq in results if len(seq) == max_len]

    # If only one die can be used (max_len == 1) and dice are different,
    # must use the larger die if possible
    if max_len == 1 and len(set(dice)) > 1:
        larger_die = max(set(dice))
        larger_moves = [seq for seq in results if seq[0][1] == larger_die]
        if larger_moves:
            results = larger_moves

    return results


def _get_legal_full_turns(state: BackgammonState) -> list[list[tuple[int, int]]]:
    """Get all legal complete turns (sequences of sub-moves) for the current position."""
    player = state.current_player
    dice = list(state.dice_remaining)

    sequences = _generate_all_turn_sequences(state, player, dice)

    # Deduplicate: keep unique sequences, but preserve different orderings
    # since get_legal_actions extracts first moves from these sequences.
    # Two sequences are identical only if they have the same moves in the same order.
    seen = set()
    unique = []
    for seq in sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique.append(seq)

    return unique


def _check_terminal(state: BackgammonState) -> BackgammonState:
    """Check if the game is over and set winner/game_over accordingly."""
    for player in (0, 1):
        if state.borne_off[player] == NUM_CHECKERS:
            state.game_over = True
            state.winner = player
            break
    return state


def _get_outcome_multiplier(state: BackgammonState, winner: int) -> int:
    """Return 1 for normal, 2 for gammon, 3 for backgammon."""
    loser = 1 - winner
    if state.borne_off[loser] == 0:
        # Gammon or backgammon
        if state.bar[loser] > 0:
            return 3  # Backgammon: loser has checker on bar
        # Check if loser has checker in winner's home board
        if winner == 0:
            # Winner's home is points 0-5; check if loser has checkers there
            for i in range(6):
                if state.points[i] < 0:
                    return 3
        else:
            for i in range(18, 24):
                if state.points[i] > 0:
                    return 3
        return 2  # Gammon
    return 1  # Normal win


class BackgammonGame(GameState):
    """Backgammon implementation of the GameState interface."""

    def get_initial_state(self) -> BackgammonState:
        state = BackgammonState()
        state.points = _initial_points()
        state.needs_dice = True
        return state

    def get_current_player(self, state: BackgammonState) -> int:
        return state.current_player

    def get_legal_actions(self, state: BackgammonState) -> list[tuple[int, int]]:
        """Return legal sub-move actions for the current step.

        In backgammon, a full turn is a sequence of sub-moves. This method
        returns the legal first sub-moves. The caller applies one, then calls
        again for the next sub-move, until dice_remaining is empty.

        Returns list of (source, die) tuples, or [(NO_MOVE_INDEX, 0)] if no moves.
        """
        if state.game_over or state.needs_dice:
            return []

        player = state.current_player
        dice = state.dice_remaining

        if not dice:
            return []

        # Get all complete turn sequences to determine which first sub-moves
        # are part of a maximal-length sequence
        full_turns = _get_legal_full_turns(state)

        if not full_turns or full_turns == [[]]:
            return []

        # Extract the legal first sub-moves from valid turn sequences
        first_moves = set()
        for seq in full_turns:
            if seq:
                first_moves.add(seq[0])

        return sorted(first_moves)

    def apply_action(self, state: BackgammonState, action: tuple[int, int]) -> BackgammonState:
        """Apply a sub-move action (source, die) and return new state."""
        source, die = action
        player = state.current_player

        new_state = _apply_sub_move(state, player, source, die)
        new_state = _check_terminal(new_state)

        if new_state.game_over:
            return new_state

        # If no more dice remaining, switch turns
        if not new_state.dice_remaining:
            new_state.current_player = 1 - player
            new_state.needs_dice = True
        else:
            # Check if there are any legal moves with remaining dice
            remaining_moves = _get_sub_moves_for_die(new_state, player, new_state.dice_remaining[0])
            # Check all remaining dice
            has_moves = False
            tried = set()
            for d in new_state.dice_remaining:
                if d in tried:
                    continue
                tried.add(d)
                if _get_sub_moves_for_die(new_state, player, d):
                    has_moves = True
                    break
            if not has_moves:
                new_state.dice_remaining = []
                new_state.current_player = 1 - player
                new_state.needs_dice = True

        return new_state

    def is_terminal(self, state: BackgammonState) -> bool:
        return state.game_over

    def get_reward(self, state: BackgammonState, player: int) -> float:
        if not state.game_over:
            return 0.0
        multiplier = _get_outcome_multiplier(state, state.winner)
        if state.winner == player:
            return float(multiplier)
        return float(-multiplier)

    def is_chance_node(self, state: BackgammonState) -> bool:
        return state.needs_dice and not state.game_over

    def get_chance_outcomes(self, state: BackgammonState) -> list[tuple[tuple[int, int], float]]:
        return DICE_ROLLS

    def apply_chance_outcome(
        self, state: BackgammonState, outcome: tuple[int, int]
    ) -> BackgammonState:
        """Apply a dice roll outcome."""
        new_state = copy.deepcopy(state)
        d1, d2 = outcome
        if d1 == d2:
            new_state.dice_remaining = [d1, d1, d1, d1]  # doubles
        else:
            new_state.dice_remaining = [d1, d2]
        new_state.needs_dice = False

        # If no legal moves with this roll, skip turn
        has_moves = False
        tried = set()
        for d in new_state.dice_remaining:
            if d in tried:
                continue
            tried.add(d)
            if _get_sub_moves_for_die(new_state, new_state.current_player, d):
                has_moves = True
                break

        if not has_moves:
            new_state.dice_remaining = []
            new_state.current_player = 1 - new_state.current_player
            new_state.needs_dice = True

        return new_state

    def encode_state(self, state: BackgammonState) -> np.ndarray:
        """Encode board state as neural network input.

        Encoding per point (4 features per point per player = 8 per point):
          For each player:
            - 1 checker: [1, 0, 0, 0]
            - 2 checkers: [1, 1, 0, 0]
            - 3 checkers: [1, 1, 1, 0]
            - 4+ checkers: [1, 1, 1, (n-3)/2]  (scaled count)

        Additional features:
          - Bar count for each player (2)
          - Borne off count for each player (2)
          - Current player indicator (1)

        Total: 24 * 8 + 5 = 197 features
        """
        features = []

        for point in range(NUM_POINTS):
            for player in (0, 1):
                count = _own_checker_at(state, player, point)
                if count == 0:
                    features.extend([0, 0, 0, 0])
                elif count == 1:
                    features.extend([1, 0, 0, 0])
                elif count == 2:
                    features.extend([1, 1, 0, 0])
                elif count == 3:
                    features.extend([1, 1, 1, 0])
                else:
                    features.extend([1, 1, 1, (count - 3) / 2.0])

        features.append(state.bar[0] / 2.0)
        features.append(state.bar[1] / 2.0)
        features.append(state.borne_off[0] / NUM_CHECKERS)
        features.append(state.borne_off[1] / NUM_CHECKERS)
        features.append(float(state.current_player))

        return np.array(features, dtype=np.float32)

    def get_action_space_size(self) -> int:
        return ACTION_SPACE_SIZE

    def action_to_index(self, action: tuple[int, int]) -> int:
        source, die = action
        return source * 6 + (die - 1)

    def index_to_action(self, index: int) -> tuple[int, int]:
        if index == NO_MOVE_INDEX:
            return (NO_MOVE_INDEX, 0)
        source = index // 6
        die = (index % 6) + 1
        return (source, die)

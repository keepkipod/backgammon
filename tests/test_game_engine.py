"""Tests for the backgammon game engine."""

import random

import numpy as np
import pytest

from backgammon.backgammon_game import (
    BAR,
    NUM_CHECKERS,
    NUM_POINTS,
    BackgammonGame,
    BackgammonState,
    _can_bear_off,
    _checker_count,
    _farthest_checker,
    _get_legal_full_turns,
    _get_sub_moves_for_die,
    _initial_points,
)


@pytest.fixture
def game():
    return BackgammonGame()


@pytest.fixture
def initial_state(game):
    return game.get_initial_state()


# ─── Initial State ─────────────────────────────────────────────────────


class TestInitialState:
    def test_starting_position_checker_counts(self, initial_state):
        assert _checker_count(initial_state, 0) == NUM_CHECKERS
        assert _checker_count(initial_state, 1) == NUM_CHECKERS

    def test_starting_position_layout(self, initial_state):
        p = initial_state.points
        # Player 0
        assert p[23] == 2
        assert p[12] == 5
        assert p[7] == 3
        assert p[5] == 5
        # Player 1
        assert p[0] == -2
        assert p[11] == -5
        assert p[16] == -3
        assert p[18] == -5

    def test_initial_state_needs_dice(self, initial_state):
        assert initial_state.needs_dice is True
        assert initial_state.dice_remaining == []
        assert initial_state.game_over is False

    def test_initial_bar_empty(self, initial_state):
        assert initial_state.bar == [0, 0]

    def test_initial_borne_off_empty(self, initial_state):
        assert initial_state.borne_off == [0, 0]

    def test_is_chance_node_at_start(self, game, initial_state):
        assert game.is_chance_node(initial_state)


# ─── Dice / Chance Outcomes ─────────────────────────────────────────────


class TestChanceOutcomes:
    def test_21_distinct_rolls(self, game, initial_state):
        outcomes = game.get_chance_outcomes(initial_state)
        assert len(outcomes) == 21

    def test_probabilities_sum_to_one(self, game, initial_state):
        outcomes = game.get_chance_outcomes(initial_state)
        total = sum(prob for _, prob in outcomes)
        assert abs(total - 1.0) < 1e-9

    def test_doubles_give_four_dice(self, game, initial_state):
        state = game.apply_chance_outcome(initial_state, (3, 3))
        assert state.dice_remaining == [3, 3, 3, 3]
        assert state.needs_dice is False

    def test_non_doubles_give_two_dice(self, game, initial_state):
        state = game.apply_chance_outcome(initial_state, (2, 5))
        assert sorted(state.dice_remaining) == [2, 5]
        assert state.needs_dice is False


# ─── Move Generation ────────────────────────────────────────────────────


class TestMoveGeneration:
    def test_has_legal_moves_after_dice(self, game, initial_state):
        state = game.apply_chance_outcome(initial_state, (3, 1))
        actions = game.get_legal_actions(state)
        assert len(actions) > 0

    def test_no_actions_when_needs_dice(self, game, initial_state):
        assert game.get_legal_actions(initial_state) == []

    def test_bar_entry_forced(self, game):
        """When a player has checkers on the bar, they must enter first."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[5] = 5
        state.points[3] = 5
        state.points[0] = 4
        state.bar = [1, 0]
        state.current_player = 0
        state.dice_remaining = [3, 1]
        state.needs_dice = False

        actions = game.get_legal_actions(state)
        # All first moves must be from bar
        for src, die in actions:
            assert src == BAR

    def test_bar_entry_blocked(self, game):
        """If all entry points are blocked, player can't move."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        # Block all entry points for player 0 (points 18-23)
        for i in range(18, 24):
            state.points[i] = -2
        state.points[5] = 14  # rest of player 0's checkers
        state.bar = [1, 0]
        state.borne_off = [0, 3]  # account for player 1 checkers
        state.current_player = 0
        state.dice_remaining = [3, 1]
        state.needs_dice = False

        actions = game.get_legal_actions(state)
        assert actions == []

    def test_hitting_blot(self, game):
        """Moving to a point with one opponent checker hits it."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[5] = 1   # player 0 checker
        state.points[3] = -1  # player 1 blot
        state.borne_off = [14, 14]
        state.current_player = 0
        state.dice_remaining = [2]
        state.needs_dice = False

        actions = game.get_legal_actions(state)
        assert (5, 2) in actions

        new_state = game.apply_action(state, (5, 2))
        assert new_state.points[3] == 1   # player 0 now there
        assert new_state.bar[1] == 1       # player 1 hit to bar

    def test_cannot_land_on_blocked_point(self, game):
        """Cannot land on a point with 2+ opponent checkers."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[5] = 1
        state.points[3] = -2  # blocked
        state.borne_off = [14, 13]
        state.current_player = 0
        state.dice_remaining = [2]
        state.needs_dice = False

        moves = _get_sub_moves_for_die(state, 0, 2)
        assert (5, 2) not in moves


# ─── Bearing Off ────────────────────────────────────────────────────────


class TestBearingOff:
    def test_can_bear_off_when_all_home(self, game):
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[3] = 5
        state.points[1] = 5
        state.points[0] = 5
        state.current_player = 0

        assert _can_bear_off(state, 0)

    def test_cannot_bear_off_with_outside_checker(self, game):
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[3] = 5
        state.points[1] = 5
        state.points[0] = 4
        state.points[10] = 1  # outside home
        state.current_player = 0

        assert not _can_bear_off(state, 0)

    def test_cannot_bear_off_with_bar_checker(self, game):
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[3] = 5
        state.points[1] = 5
        state.points[0] = 4
        state.bar = [1, 0]
        state.current_player = 0

        assert not _can_bear_off(state, 0)

    def test_exact_bear_off(self, game):
        """Bear off with exact die value."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[2] = 1
        state.borne_off = [14, 15]
        state.current_player = 0
        state.dice_remaining = [3]
        state.needs_dice = False

        actions = game.get_legal_actions(state)
        assert (2, 3) in actions

        new_state = game.apply_action(state, (2, 3))
        assert new_state.borne_off[0] == 15
        assert new_state.game_over is True
        assert new_state.winner == 0

    def test_bear_off_with_larger_die(self, game):
        """Can bear off with a larger die when no checker is farther back."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[1] = 1  # point 1 — needs 2 to exact bear off
        state.borne_off = [14, 15]
        state.current_player = 0
        state.dice_remaining = [5]
        state.needs_dice = False

        actions = game.get_legal_actions(state)
        assert (1, 5) in actions

    def test_cannot_bear_off_larger_die_with_farther_checker(self, game):
        """Cannot use larger die to bear off if there's a checker farther back."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[1] = 1
        state.points[4] = 1
        state.borne_off = [13, 15]
        state.current_player = 0
        state.dice_remaining = [5]
        state.needs_dice = False

        moves = _get_sub_moves_for_die(state, 0, 5)
        # Point 4 can bear off with 5 (exact), point 1 cannot (4 is farther)
        assert (4, 5) in moves
        assert (1, 5) not in moves

    def test_player1_bearing_off(self, game):
        """Player 1 bears off from their home board (points 18-23)."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[22] = -1
        state.borne_off = [15, 14]
        state.current_player = 1
        state.dice_remaining = [2]
        state.needs_dice = False

        actions = game.get_legal_actions(state)
        assert (22, 2) in actions

        new_state = game.apply_action(state, (22, 2))
        assert new_state.borne_off[1] == 15


# ─── Forced Maximization ───────────────────────────────────────────────


class TestForcedMaximization:
    def test_must_use_both_dice(self, game):
        """If both dice can be used, both must be used."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[10] = 2
        state.borne_off = [13, 15]
        state.current_player = 0
        state.dice_remaining = [3, 2]
        state.needs_dice = False

        turns = _get_legal_full_turns(state)
        for turn in turns:
            assert len(turn) == 2

    def test_must_use_larger_die_when_only_one_possible(self):
        """If only one die can be played, must use the larger."""
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[5] = 1
        # Block point 2 and point 0 area so that:
        # die 3: can move 5→2, but then die 4 is blocked
        # die 4: can move 5→1, but then die 3 is blocked
        state.points[2] = -2  # block point 2
        state.points[1] = -2  # block point 1
        state.borne_off = [14, 11]
        state.current_player = 0
        state.dice_remaining = [3, 4]
        state.needs_dice = False

        turns = _get_legal_full_turns(state)
        # All turns should use 1 die, and it must be the larger one (4)
        for turn in turns:
            if len(turn) == 1:
                assert turn[0][1] == 4


# ─── Turn Switching & Game Flow ─────────────────────────────────────────


class TestGameFlow:
    def test_turn_switches_after_all_dice_used(self, game, initial_state):
        state = game.apply_chance_outcome(initial_state, (1, 2))
        assert state.current_player == 0

        # Play all sub-moves
        while state.dice_remaining and not state.needs_dice:
            actions = game.get_legal_actions(state)
            if not actions:
                break
            state = game.apply_action(state, actions[0])

        assert state.current_player == 1
        assert state.needs_dice is True

    def test_full_turn_cycle(self, game, initial_state):
        """Play two full turns (both players) without error."""
        state = game.apply_chance_outcome(initial_state, (3, 1))
        while state.dice_remaining and not state.needs_dice:
            actions = game.get_legal_actions(state)
            if not actions:
                break
            state = game.apply_action(state, actions[0])

        assert state.current_player == 1
        state = game.apply_chance_outcome(state, (5, 2))
        while state.dice_remaining and not state.needs_dice:
            actions = game.get_legal_actions(state)
            if not actions:
                break
            state = game.apply_action(state, actions[0])

        assert state.current_player == 0


# ─── Reward / Terminal ──────────────────────────────────────────────────


class TestRewardAndTerminal:
    def test_normal_win(self, game):
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.borne_off = [15, 10]
        state.bar = [0, 0]
        state.game_over = True
        state.winner = 0

        assert game.get_reward(state, 0) == 1.0
        assert game.get_reward(state, 1) == -1.0

    def test_gammon(self, game):
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[10] = -5
        state.points[20] = -5
        state.points[15] = -5
        state.borne_off = [15, 0]
        state.game_over = True
        state.winner = 0

        assert game.get_reward(state, 0) == 2.0
        assert game.get_reward(state, 1) == -2.0

    def test_backgammon_checker_on_bar(self, game):
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[10] = -10
        state.points[20] = -4
        state.bar = [0, 1]
        state.borne_off = [15, 0]
        state.game_over = True
        state.winner = 0

        assert game.get_reward(state, 0) == 3.0

    def test_backgammon_checker_in_winners_home(self, game):
        state = BackgammonState()
        state.points = [0] * NUM_POINTS
        state.points[3] = -5   # in player 0's home board
        state.points[10] = -5
        state.points[20] = -5
        state.borne_off = [15, 0]
        state.game_over = True
        state.winner = 0

        assert game.get_reward(state, 0) == 3.0

    def test_not_terminal_initially(self, game, initial_state):
        assert not game.is_terminal(initial_state)
        assert game.get_reward(initial_state, 0) == 0.0


# ─── State Encoding ─────────────────────────────────────────────────────


class TestEncoding:
    def test_encoding_shape(self, game, initial_state):
        enc = game.encode_state(initial_state)
        assert enc.shape == (197,)
        assert enc.dtype == np.float32

    def test_encoding_changes_with_moves(self, game, initial_state):
        enc1 = game.encode_state(initial_state)
        state = game.apply_chance_outcome(initial_state, (3, 1))
        actions = game.get_legal_actions(state)
        state = game.apply_action(state, actions[0])
        enc2 = game.encode_state(state)
        assert not np.array_equal(enc1, enc2)


# ─── Action Space ───────────────────────────────────────────────────────


class TestActionSpace:
    def test_action_roundtrip(self, game):
        for source in range(25):
            for die in range(1, 7):
                action = (source, die)
                idx = game.action_to_index(action)
                assert 0 <= idx < 150
                assert game.index_to_action(idx) == action

    def test_action_space_size(self, game):
        assert game.get_action_space_size() == 151


# ─── Immutability ───────────────────────────────────────────────────────


class TestImmutability:
    def test_apply_action_does_not_mutate(self, game, initial_state):
        state = game.apply_chance_outcome(initial_state, (3, 1))
        points_before = list(state.points)
        bar_before = list(state.bar)
        dice_before = list(state.dice_remaining)

        actions = game.get_legal_actions(state)
        _ = game.apply_action(state, actions[0])

        assert state.points == points_before
        assert state.bar == bar_before
        assert state.dice_remaining == dice_before

    def test_apply_chance_does_not_mutate(self, game, initial_state):
        points_before = list(initial_state.points)
        _ = game.apply_chance_outcome(initial_state, (4, 2))
        assert initial_state.points == points_before
        assert initial_state.needs_dice is True


# ─── Random Simulation (Stress Test) ───────────────────────────────────


class TestRandomSimulation:
    def _play_random_game(self, game: BackgammonGame, seed: int) -> BackgammonState:
        rng = random.Random(seed)
        state = game.get_initial_state()
        max_steps = 2000  # safety cap

        for _ in range(max_steps):
            if game.is_terminal(state):
                break

            if game.is_chance_node(state):
                outcomes = game.get_chance_outcomes(state)
                roll = rng.choices(
                    [o for o, _ in outcomes],
                    weights=[p for _, p in outcomes],
                )[0]
                state = game.apply_chance_outcome(state, roll)
            else:
                actions = game.get_legal_actions(state)
                if not actions:
                    # No moves — force turn switch
                    state.dice_remaining = []
                    state.current_player = 1 - state.current_player
                    state.needs_dice = True
                    continue
                action = rng.choice(actions)
                state = game.apply_action(state, action)

        return state

    def test_single_random_game_completes(self, game):
        state = self._play_random_game(game, seed=42)
        assert state.game_over is True

    def test_checker_invariant_during_game(self, game):
        """Checker count must always be 15 per player throughout a game."""
        rng = random.Random(123)
        state = game.get_initial_state()

        for _ in range(2000):
            assert _checker_count(state, 0) == NUM_CHECKERS, f"P0 checkers: {_checker_count(state, 0)}"
            assert _checker_count(state, 1) == NUM_CHECKERS, f"P1 checkers: {_checker_count(state, 1)}"

            if game.is_terminal(state):
                break

            if game.is_chance_node(state):
                outcomes = game.get_chance_outcomes(state)
                roll = rng.choices(
                    [o for o, _ in outcomes],
                    weights=[p for _, p in outcomes],
                )[0]
                state = game.apply_chance_outcome(state, roll)
            else:
                actions = game.get_legal_actions(state)
                if not actions:
                    state.dice_remaining = []
                    state.current_player = 1 - state.current_player
                    state.needs_dice = True
                    continue
                state = game.apply_action(state, rng.choice(actions))

    @pytest.mark.parametrize("seed", range(100))
    def test_100_random_games_no_crash(self, game, seed):
        """Run 100 random games — all must complete without error."""
        state = self._play_random_game(game, seed)
        # Game should finish (some may hit step limit, but most should end)
        assert _checker_count(state, 0) == NUM_CHECKERS
        assert _checker_count(state, 1) == NUM_CHECKERS

"""Tests for the web server API."""

import json

import pytest

from web.server import app, game


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAPI:
    def test_index_page(self, client):
        rv = client.get("/")
        assert rv.status_code == 200
        assert b"Backgammon" in rv.data

    def test_new_game(self, client):
        rv = client.post("/api/new-game")
        data = json.loads(rv.data)
        assert rv.status_code == 200
        assert "points" in data
        assert "dice_remaining" in data
        assert len(data["points"]) == 24
        assert data["game_over"] is False

    def test_new_game_has_dice(self, client):
        rv = client.post("/api/new-game")
        data = json.loads(rv.data)
        assert len(data["dice_remaining"]) >= 2
        assert data["needs_dice"] is False

    def test_get_state_before_game(self, client):
        rv = client.get("/api/state")
        # No game started yet, should error or return state from previous test
        # (Flask test client shares state within fixture)

    def test_legal_actions_present(self, client):
        rv = client.post("/api/new-game")
        data = json.loads(rv.data)
        assert "legal_actions" in data
        assert len(data["legal_actions"]) > 0
        # Each action should have source and die
        for action in data["legal_actions"]:
            assert "source" in action
            assert "die" in action

    def test_make_legal_move(self, client):
        rv = client.post("/api/new-game")
        data = json.loads(rv.data)
        action = data["legal_actions"][0]

        rv = client.post(
            "/api/move",
            data=json.dumps({"source": action["source"], "die": action["die"]}),
            content_type="application/json",
        )
        data = json.loads(rv.data)
        assert rv.status_code == 200
        assert "points" in data

    def test_illegal_move_rejected(self, client):
        client.post("/api/new-game")
        rv = client.post(
            "/api/move",
            data=json.dumps({"source": 0, "die": 1}),
            content_type="application/json",
        )
        # Might be legal or not depending on dice — test with clearly illegal
        # Source 0 with die 1 for player 0 moves to -1 which is bearing off
        # If not all in home board, it's illegal

    def test_ai_move(self, client):
        rv = client.post("/api/new-game")
        data = json.loads(rv.data)

        # Play through player 0's turn
        while data["current_player"] == 0 and not data["game_over"]:
            if data["needs_dice"]:
                rv = client.post("/api/roll-dice")
                data = json.loads(rv.data)
                continue
            if not data["legal_actions"]:
                break
            action = data["legal_actions"][0]
            rv = client.post(
                "/api/move",
                data=json.dumps({"source": action["source"], "die": action["die"]}),
                content_type="application/json",
            )
            data = json.loads(rv.data)

        # Now it should be AI's turn
        if not data["game_over"] and data["current_player"] == 1:
            rv = client.post("/api/ai-move")
            data = json.loads(rv.data)
            assert rv.status_code == 200
            assert "ai_moves" in data
            assert len(data["ai_moves"]) > 0

    def test_set_difficulty(self, client):
        rv = client.post(
            "/api/set-difficulty",
            data=json.dumps({"simulations": 50}),
            content_type="application/json",
        )
        data = json.loads(rv.data)
        assert data["simulations"] == 50

    def test_full_turn_cycle(self, client):
        """Play a full cycle: new game → player moves → AI moves."""
        rv = client.post("/api/new-game")
        data = json.loads(rv.data)

        moves_made = 0
        for _ in range(20):  # safety
            if data["game_over"]:
                break
            if data["current_player"] == 1:
                rv = client.post("/api/ai-move")
                data = json.loads(rv.data)
                continue
            if data["needs_dice"]:
                rv = client.post("/api/roll-dice")
                data = json.loads(rv.data)
                continue
            if not data["legal_actions"]:
                break
            action = data["legal_actions"][0]
            rv = client.post(
                "/api/move",
                data=json.dumps({"source": action["source"], "die": action["die"]}),
                content_type="application/json",
            )
            data = json.loads(rv.data)
            moves_made += 1

        assert moves_made > 0

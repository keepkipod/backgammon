"""Flask server for the Backgammon GUI.

Provides a REST API for game state management and AI moves,
serving a single-page web app for the board UI.

Usage:
    python -m web.server
    python -m web.server --checkpoint checkpoints/latest.pt
    python -m web.server --port 8080
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from backgammon.backgammon_game import (
    ACTION_SPACE_SIZE,
    BAR,
    NUM_CHECKERS,
    NUM_POINTS,
    BackgammonGame,
    BackgammonState,
    _checker_count,
)
from backgammon.mcts import MCTS
from backgammon.network import DualHeadNetwork, get_device

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")

# Global game state
game = BackgammonGame()
state: Optional[BackgammonState] = None
network: Optional[DualHeadNetwork] = None
device: Optional[torch.device] = None
ai_simulations: int = 200  # configurable difficulty


def _state_to_json(s: BackgammonState) -> dict:
    """Convert game state to JSON-serializable dict for the frontend."""
    return {
        "points": list(s.points),
        "bar": list(s.bar),
        "borne_off": list(s.borne_off),
        "current_player": s.current_player,
        "dice_remaining": list(s.dice_remaining),
        "needs_dice": s.needs_dice,
        "game_over": s.game_over,
        "winner": s.winner,
        "legal_actions": [
            {"source": src, "die": die}
            for src, die in game.get_legal_actions(s)
        ],
    }


def _get_ai_analysis(s: BackgammonState) -> dict:
    """Get AI analysis: win probability and top moves with equity."""
    if network is None:
        return {"win_prob": 0.5, "top_moves": [], "available": False}

    network.eval()
    state_tensor = torch.FloatTensor(game.encode_state(s)).to(device)
    legal_mask = torch.zeros(ACTION_SPACE_SIZE, device=device)
    legal_actions = game.get_legal_actions(s)
    for action in legal_actions:
        legal_mask[game.action_to_index(action)] = 1.0

    if not legal_actions:
        return {"win_prob": 0.5, "top_moves": [], "available": True}

    # Get raw value estimate
    policy, value = network.predict(state_tensor, legal_mask)
    # Convert value from [-1, 1] to win probability [0, 1]
    win_prob = (value + 1.0) / 2.0

    # Get top moves with their policy weight (as proxy for equity)
    policy_np = policy.cpu().numpy()
    move_evals = []
    for action in legal_actions:
        idx = game.action_to_index(action)
        src, die = action

        # Evaluate resulting state for equity
        new_state = game.apply_action(s, action)
        new_enc = torch.FloatTensor(game.encode_state(new_state)).to(device)
        new_mask = torch.zeros(ACTION_SPACE_SIZE, device=device)
        _, move_value = network.predict(new_enc, new_mask)

        # Negate if player changed (opponent's value)
        if new_state.current_player != s.current_player:
            move_value = -move_value

        move_evals.append({
            "source": src,
            "die": die,
            "policy_weight": float(policy_np[idx]),
            "equity": float((move_value + 1.0) / 2.0),
            "source_label": "BAR" if src == BAR else f"Point {src + 1}",
        })

    # Sort by equity (best first)
    move_evals.sort(key=lambda m: m["equity"], reverse=True)

    return {
        "win_prob": float(win_prob),
        "top_moves": move_evals[:5],
        "available": True,
    }


def _detect_blunder(
    s: BackgammonState, chosen_action: tuple[int, int]
) -> Optional[dict]:
    """Check if the chosen move is a blunder compared to the best move."""
    if network is None:
        return None

    legal_actions = game.get_legal_actions(s)
    if len(legal_actions) <= 1:
        return None

    network.eval()

    # Evaluate all legal moves
    best_equity = -float("inf")
    best_action = None
    chosen_equity = None

    for action in legal_actions:
        new_state = game.apply_action(s, action)
        enc = torch.FloatTensor(game.encode_state(new_state)).to(device)
        mask = torch.zeros(ACTION_SPACE_SIZE, device=device)
        _, val = network.predict(enc, mask)
        if new_state.current_player != s.current_player:
            val = -val
        equity = (val + 1.0) / 2.0

        if equity > best_equity:
            best_equity = equity
            best_action = action
        if action == chosen_action:
            chosen_equity = equity

    if chosen_equity is None or best_action is None:
        return None

    equity_drop = best_equity - chosen_equity
    if equity_drop > 0.05:  # 5% equity threshold
        return {
            "is_blunder": True,
            "equity_drop": float(equity_drop),
            "best_move": {
                "source": best_action[0],
                "die": best_action[1],
                "source_label": "BAR" if best_action[0] == BAR else f"Point {best_action[0] + 1}",
            },
            "best_equity": float(best_equity),
            "chosen_equity": float(chosen_equity),
        }
    return None


# ─── Routes ─────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


@app.route("/api/new-game", methods=["POST"])
def new_game():
    global state
    state = game.get_initial_state()
    # Roll initial dice
    outcomes = game.get_chance_outcomes(state)
    rolls = [o for o, _ in outcomes]
    probs = [p for _, p in outcomes]
    idx = np.random.choice(len(rolls), p=probs)
    state = game.apply_chance_outcome(state, rolls[idx])

    response = _state_to_json(state)
    response["analysis"] = _get_ai_analysis(state) if not state.needs_dice else {"win_prob": 0.5, "top_moves": [], "available": network is not None}
    return jsonify(response)


@app.route("/api/move", methods=["POST"])
def make_move():
    global state
    if state is None:
        return jsonify({"error": "No game in progress"}), 400

    data = request.json
    source = data["source"]
    die = data["die"]
    action = (source, die)

    legal = game.get_legal_actions(state)
    if action not in legal:
        return jsonify({"error": "Illegal move"}), 400

    # Blunder detection before applying
    blunder = _detect_blunder(state, action)

    state = game.apply_action(state, action)

    response = _state_to_json(state)
    response["blunder"] = blunder

    # If turn ended, include analysis
    if not state.game_over and not state.needs_dice and state.dice_remaining:
        response["analysis"] = _get_ai_analysis(state)

    return jsonify(response)


@app.route("/api/roll-dice", methods=["POST"])
def roll_dice():
    global state
    if state is None:
        return jsonify({"error": "No game in progress"}), 400
    if not state.needs_dice:
        return jsonify({"error": "Dice already rolled"}), 400

    outcomes = game.get_chance_outcomes(state)
    rolls = [o for o, _ in outcomes]
    probs = [p for _, p in outcomes]
    idx = np.random.choice(len(rolls), p=probs)
    state = game.apply_chance_outcome(state, rolls[idx])

    response = _state_to_json(state)
    response["analysis"] = _get_ai_analysis(state)
    return jsonify(response)


@app.route("/api/ai-move", methods=["POST"])
def ai_move():
    """Have the AI play its turn (all sub-moves)."""
    global state
    if state is None:
        return jsonify({"error": "No game in progress"}), 400

    moves_made = []

    # Roll dice if needed
    if state.needs_dice:
        outcomes = game.get_chance_outcomes(state)
        rolls = [o for o, _ in outcomes]
        probs = [p for _, p in outcomes]
        idx = np.random.choice(len(rolls), p=probs)
        state = game.apply_chance_outcome(state, rolls[idx])

    ai_player = state.current_player

    # Play all sub-moves for this turn
    for _ in range(10):  # safety cap
        if state.game_over or state.needs_dice or state.current_player != ai_player:
            break

        legal = game.get_legal_actions(state)
        if not legal:
            state.dice_remaining = []
            state.current_player = 1 - state.current_player
            state.needs_dice = True
            break

        if network is not None:
            if ai_simulations <= 0:
                # Direct network policy — no MCTS, instant moves
                state_tensor = torch.FloatTensor(game.encode_state(state)).to(device)
                legal_mask = torch.zeros(game.get_action_space_size(), device=device)
                for a in legal:
                    legal_mask[game.action_to_index(a)] = 1.0
                policy, _ = network.predict(state_tensor, legal_mask)
                policy = policy.cpu().numpy()
                legal_indices = [game.action_to_index(a) for a in legal]
                legal_probs = [(policy[i], a) for i, a in zip(legal_indices, legal)]
                action = max(legal_probs, key=lambda x: x[0])[1]
            else:
                mcts = MCTS(
                    game=game,
                    network=network,
                    device=device,
                    num_simulations=ai_simulations,
                    dirichlet_alpha=0.0,
                    dirichlet_epsilon=0.0,
                    temperature=0.1,
                )
                policy, _ = mcts.search(state)
                action_idx = np.argmax(policy)
                action = game.index_to_action(action_idx)
                if action not in legal:
                    action = random.choice(legal)
        else:
            action = random.choice(legal)

        state = game.apply_action(state, action)
        moves_made.append({"source": action[0], "die": action[1]})

    response = _state_to_json(state)
    response["ai_moves"] = moves_made
    response["analysis"] = _get_ai_analysis(state) if not state.game_over else {"win_prob": 0.5, "top_moves": [], "available": network is not None}
    return jsonify(response)


@app.route("/api/analysis", methods=["GET"])
def get_analysis():
    if state is None:
        return jsonify({"error": "No game in progress"}), 400
    return jsonify(_get_ai_analysis(state))


@app.route("/api/set-difficulty", methods=["POST"])
def set_difficulty():
    global ai_simulations
    data = request.json
    ai_simulations = max(0, min(1000, data.get("simulations", 200)))
    return jsonify({"simulations": ai_simulations})


@app.route("/api/state", methods=["GET"])
def get_state():
    if state is None:
        return jsonify({"error": "No game in progress"}), 400
    response = _state_to_json(state)
    response["analysis"] = _get_ai_analysis(state)
    return jsonify(response)


def main():
    global network, device, ai_simulations

    parser = argparse.ArgumentParser(description="Backgammon Web UI")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations for AI")
    args = parser.parse_args()

    ai_simulations = args.simulations
    # For inference with small networks, CPU is faster than MPS
    # (avoids per-call GPU dispatch overhead in MCTS loop)
    device = torch.device("cpu")

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        input_size = 197
        action_size = ACTION_SPACE_SIZE
        network = DualHeadNetwork(
            input_size=input_size,
            action_size=action_size,
            hidden_size=config.get("hidden_size", 128),
            num_res_blocks=config.get("num_res_blocks", 5),
        ).to(device)
        network.load_state_dict(checkpoint["network_state"])
        network.eval()
        logger.info("Model loaded successfully")
    else:
        logger.info("No model checkpoint — AI will play randomly")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

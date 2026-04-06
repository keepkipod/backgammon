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
    _get_sub_moves_for_die,
)
from backgammon.mcts import MCTS
from backgammon.network import DualHeadNetwork, get_device

logger = logging.getLogger(__name__)

from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    """JSON provider that handles numpy types."""
    @staticmethod
    def default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return DefaultJSONProvider.default(obj)

app = Flask(__name__, static_folder="static")
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # disable static file caching

# Global game state
game = BackgammonGame()
state: Optional[BackgammonState] = None
network: Optional[DualHeadNetwork] = None
device: Optional[torch.device] = None
ai_simulations: int = 200  # configurable difficulty
human_player: int = 0  # which player the human controls (0=white, 1=brown)


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
        "human_player": human_player,
        "legal_actions": [
            {"source": src, "die": die}
            for src, die in game.get_legal_actions(s)
        ],
    }


def _get_ai_analysis(s: BackgammonState) -> dict:
    """Get AI analysis: win probability and top moves with equity.

    Uses batched network evaluation for all legal moves in a single forward pass.
    """
    if network is None:
        return {"win_prob": 0.5, "top_moves": [], "available": False}

    network.eval()
    legal_actions = game.get_legal_actions(s)

    if not legal_actions:
        # Just get win probability for current state
        state_tensor = torch.FloatTensor(game.encode_state(s)).to(device)
        legal_mask = torch.zeros(ACTION_SPACE_SIZE, device=device)
        _, value = network.predict(state_tensor, legal_mask)
        return {"win_prob": float((value + 1.0) / 2.0), "top_moves": [], "available": True}

    # Build batch: current state + all resulting states
    encodings = [game.encode_state(s)]
    result_states = []
    for action in legal_actions:
        new_state = game.apply_action(s, action)
        encodings.append(game.encode_state(new_state))
        result_states.append(new_state)

    # Single batched forward pass
    batch_tensor = torch.FloatTensor(np.array(encodings)).to(device)
    batch_mask = torch.zeros(len(encodings), ACTION_SPACE_SIZE, device=device)
    # Set legal mask only for the current state (index 0)
    for action in legal_actions:
        batch_mask[0][game.action_to_index(action)] = 1.0

    policies, values = network.predict_batch(batch_tensor, batch_mask)
    policy_np = policies[0].cpu().numpy()
    values_np = values.cpu().numpy()

    win_prob = (float(values_np[0]) + 1.0) / 2.0

    move_evals = []
    for i, action in enumerate(legal_actions):
        idx = game.action_to_index(action)
        src, die = action
        move_value = float(values_np[i + 1])
        if result_states[i].current_player != s.current_player:
            move_value = -move_value

        move_evals.append({
            "source": int(src),
            "die": int(die),
            "policy_weight": float(policy_np[idx]),
            "equity": float((move_value + 1.0) / 2.0),
            "source_label": "BAR" if src == BAR else f"Point {src + 1}",
        })

    move_evals.sort(key=lambda m: m["equity"], reverse=True)

    return {
        "win_prob": float(win_prob),
        "top_moves": move_evals[:5],
        "available": True,
    }


def _detect_blunder(
    s: BackgammonState, chosen_action: tuple[int, int]
) -> Optional[dict]:
    """Check if the chosen move is a blunder compared to the best move.

    Uses batched network evaluation for all legal moves.
    """
    if network is None:
        return None

    legal_actions = game.get_legal_actions(s)
    if len(legal_actions) <= 1:
        return None

    network.eval()

    # Batch evaluate all resulting states
    encodings = []
    result_states = []
    for action in legal_actions:
        new_state = game.apply_action(s, action)
        encodings.append(game.encode_state(new_state))
        result_states.append(new_state)

    batch_tensor = torch.FloatTensor(np.array(encodings)).to(device)
    batch_mask = torch.zeros(len(encodings), ACTION_SPACE_SIZE, device=device)
    _, values = network.predict_batch(batch_tensor, batch_mask)
    values_np = values.cpu().numpy()

    best_equity = -float("inf")
    best_action = None
    chosen_equity = None

    for i, action in enumerate(legal_actions):
        val = float(values_np[i])
        if result_states[i].current_player != s.current_player:
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
                "source": int(best_action[0]),
                "die": int(best_action[1]),
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
    global state, human_player
    data = request.get_json(silent=True) or {}
    human_player = data.get("color", 0)  # 0=white, 1=brown

    state = game.get_initial_state()

    # Opening roll: each player rolls one die, higher goes first
    while True:
        human_die = random.randint(1, 6)
        ai_die = random.randint(1, 6)
        if human_die != ai_die:
            break  # re-roll on ties

    if human_die > ai_die:
        first_player = human_player
    else:
        first_player = 1 - human_player

    # Apply the opening dice to the starting player
    state.current_player = first_player
    d1, d2 = max(human_die, ai_die), min(human_die, ai_die)
    state.dice_remaining = [d1, d2]
    state.needs_dice = False

    # Check if first player has any legal moves
    has_moves = False
    tried = set()
    for d in state.dice_remaining:
        if d in tried:
            continue
        tried.add(d)
        if _get_sub_moves_for_die(state, first_player, d):
            has_moves = True
            break
    if not has_moves:
        state.dice_remaining = []
        state.current_player = 1 - first_player
        state.needs_dice = True

    response = _state_to_json(state)
    response["opening_roll"] = {
        "human_die": human_die,
        "ai_die": ai_die,
        "first_player": first_player,
    }
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


@app.route("/api/compound-moves", methods=["POST"])
def compound_moves():
    """Get all reachable destinations from a source, including multi-die compounds.

    Returns single-die moves and compound moves (2+ dice on the same checker).
    Each result has a dest and the action sequence to get there.
    """
    if state is None:
        return jsonify({"error": "No game in progress"}), 400

    data = request.json
    source = data["source"]
    player = state.current_player

    def calc_dest(src, die_val, p):
        if src == BAR:
            return die_val - 1 if p == 1 else NUM_POINTS - die_val
        d = src + die_val if p == 1 else src - die_val
        return -1 if d < 0 or d >= NUM_POINTS else d

    results = []  # [{dest, actions: [{source, die}, ...], is_compound}]
    seen = set()  # (dest, tuple of actions) for dedup

    def explore(s, current_src, actions_so_far):
        legal = game.get_legal_actions(s)
        for src, die in legal:
            if src != current_src:
                continue
            dest = calc_dest(src, die, player)
            new_actions = actions_so_far + [{"source": int(src), "die": int(die)}]
            action_key = (dest, tuple((a["source"], a["die"]) for a in new_actions))
            if action_key not in seen:
                seen.add(action_key)
                results.append({
                    "dest": int(dest),
                    "actions": new_actions,
                    "is_compound": len(new_actions) > 1,
                })
            # Recurse if the checker can continue moving
            if dest >= 0 and dest < NUM_POINTS:
                new_state = game.apply_action(s, (src, die))
                if (not new_state.game_over
                        and not new_state.needs_dice
                        and new_state.current_player == player):
                    explore(new_state, dest, new_actions)

    explore(state, source, [])

    # Deduplicate: keep shortest action sequence per destination
    best = {}
    for r in results:
        d = r["dest"]
        if d not in best or len(r["actions"]) < len(best[d]["actions"]):
            best[d] = r
    # Also add compound-only entries (longer paths to same dest)
    compound_results = list(best.values())

    return jsonify({"moves": compound_results})


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

        # Compute destination for the UI trace
        src, die_val = action
        ai_player = 1 - human_player
        if src == BAR:
            dest = die_val - 1 if ai_player == 1 else NUM_POINTS - die_val
        else:
            if ai_player == 1:
                dest = src + die_val
                if dest >= NUM_POINTS:
                    dest = -1  # bearing off
            else:
                dest = src - die_val
                if dest < 0:
                    dest = -1  # bearing off

        state = game.apply_action(state, action)
        moves_made.append({"source": int(src), "die": int(die_val), "dest": int(dest)})

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

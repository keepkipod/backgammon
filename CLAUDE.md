# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Backgammon AI with AlphaZero-style architecture — a "Superhuman Backgammon Tutor." The project uses Reinforcement Learning (self-play + neural network evaluation) to train a world-class Backgammon agent, with a GUI layer for human play and coaching. This is the first in a series of game AI projects; the architecture is designed so future games (Chess, Go, Othello, etc.) reuse the same MCTS and neural network infrastructure with only a new `GameState` implementation.

**Tech stack:** Python, PyTorch, NumPy, Gymnasium, Flask (web UI), PyYAML (config).

**Hardware targets:** Must run on Apple M3 (MPS) for local dev/testing and Cloud GPUs (CUDA) for heavy training. Code dynamically detects `cuda` > `mps` > `cpu`. Training targets free-tier Colab/Kaggle with checkpoint-resume.

## Current Status

**All 4 phases are implemented and tested.** No model has been trained yet — the next step is cloud training.

| Phase | Status | Tests |
|---|---|---|
| 1. Game Engine | Complete | 140 tests (test_game_engine.py) |
| 2. Gymnasium Env | Complete | 67 tests (test_env.py) |
| 3. AI Training Pipeline | Complete | 11 tests (test_training.py) |
| 4. Web GUI & Tutor | Complete | 10 tests (test_server.py) |

**Next steps:** Train the model on Colab/Kaggle (see WORKPLAN.md "Cloud Training Strategy"), then run the web UI with the trained checkpoint.

## Commands

```bash
# Run all tests (fast — excludes env stress tests)
python3 -m pytest tests/test_game_engine.py tests/test_training.py tests/test_server.py -v

# Run env tests (slower — 50-game random simulation takes ~6 min)
python3 -m pytest tests/test_env.py -v

# Run a single test class
python3 -m pytest tests/test_game_engine.py::TestBearingOff -v

# Smoke test (full pipeline: self-play -> train -> checkpoint -> load -> verify)
python3 train.py --smoke-test

# Start training with default config
python3 train.py --config config.yaml

# Resume training from checkpoint
python3 train.py --resume-from checkpoints/latest.pt

# Start web UI (no trained model — AI plays randomly)
python3 -m web.server

# Start web UI with trained model
python3 -m web.server --checkpoint checkpoints/latest.pt

# Start web UI with custom difficulty (MCTS simulations)
python3 -m web.server --checkpoint checkpoints/latest.pt --simulations 50

# Install dependencies
pip3 install -r requirements.txt
```

## Code Structure

### Game Engine (`backgammon/`)
- `game_state.py` — Abstract `GameState` interface. All future games implement this. MCTS and neural network depend only on this abstraction. Key methods: `get_legal_actions`, `apply_action`, `is_chance_node`, `get_chance_outcomes`, `encode_state`, `action_to_index`.
- `backgammon_game.py` — Backgammon implementation of `GameState`. Board is a 24-int array (positive = player 0, negative = player 1). Action space is 151 discrete sub-moves: 25 sources x 6 die values + 1 no-move. State encoding is 197 floats. Key internal functions: `_generate_all_turn_sequences` (handles forced maximization rules), `_get_sub_moves_for_die`, `_can_bear_off`.

### RL Environment (`backgammon/`)
- `env.py` — Gymnasium wrapper. Agent is always player 0. Opponent is pluggable (`random_opponent` default, `None` for self-play). Critical method: `_advance_to_agent_turn()` — unified loop that handles dice rolling, opponent sub-moves, and no-move turn skipping. This was the trickiest part (both players can have no-move rolls back-to-back).

### AI Training (`backgammon/`, `train.py`, `config.yaml`)
- `network.py` — `DualHeadNetwork`: ResNet with configurable residual blocks, policy head (logits over 151 actions), value head (tanh scalar). `predict()` method handles legal move masking + softmax.
- `mcts.py` — `MCTS` class with `DecisionNode` and `ChanceNode`. Chance nodes lazily expand dice outcomes. PUCT selection at decision nodes, probability-weighted sampling at chance nodes. Dirichlet noise at root for self-play exploration.
- `trainer.py` — `Trainer` class: self-play game generation, `ReplayBuffer`, training loop (MSE value + cross-entropy policy loss), champion evaluation gating (55% win rate), checkpoint save/load (includes network, optimizer, buffer, history). Health checks: NaN/Inf detection, entropy collapse warning.
- `train.py` — CLI entry point. `--smoke-test` validates full pipeline in ~6s. `--config` loads YAML. `--resume-from` restores from checkpoint.
- `config.yaml` — All hyperparameters with documented defaults.

### Web UI (`web/`)
- `server.py` — Flask REST API. Endpoints: `/api/new-game`, `/api/move`, `/api/roll-dice`, `/api/ai-move`, `/api/analysis`, `/api/set-difficulty`. AI analysis includes: win probability (value head), top-5 move equity (evaluates each resulting state), blunder detection (>5% equity drop threshold). Loads model checkpoint on startup if provided.
- `static/index.html` — Single-page canvas app. Click-to-move: click source checker, then click destination (or borne-off tray). Green highlights show legal destinations. UI panels: dice, win probability bar, top moves with equity %, blunder alerts, difficulty slider (controls MCTS simulation count).

### Tests (`tests/`)
- `test_game_engine.py` — 140 tests: initial state, dice, move generation (bar entry, hitting, blocking), bearing off (exact, larger die, farthest checker rule), forced maximization, game flow, rewards (normal/gammon/backgammon), encoding, action space roundtrip, immutability, 100 random game simulations with checker count invariant.
- `test_env.py` — 67 tests: interface, action masking, step mechanics, self-play alternation, 50 random games, reward distribution (wins and losses both occur), gammon/backgammon occurrence.
- `test_training.py` — 11 tests: network forward/value range/masked prediction, MCTS policy validity/legal-only/greedy temperature, self-play examples, training step, checkpoint roundtrip, finite loss, full pipeline smoke test.
- `test_server.py` — 10 tests: index page, new game, dice, legal actions, legal/illegal moves, AI move, difficulty, full turn cycle.

## Key Design Decisions

- **AlphaZero over TD-Gammon:** TD-Gammon doesn't transfer to deterministic games. AlphaZero's policy + value + MCTS is the universal framework for future game projects.
- **Sequential sub-moves:** Action space is individual checker moves (source, die), not full turns. Keeps action space at 151 vs combinatorial explosion. A turn is 2-4 sequential step() calls.
- **No doubling cube:** Simplifies engine, action space, and value output.
- **Stochastic MCTS with chance nodes:** 21 distinct dice outcomes as probability-weighted branches. Lazily expanded to save memory.
- **State encoding:** 197 floats — 4 features per point per player (thresholded counts) + bar + borne-off + current player.
- **Checkpoint-resume design:** `train.py --resume-from` reconstructs full state. Aggressive checkpointing for cloud sessions that die unexpectedly.
- **Multi-game modularity:** `GameState` interface → `MCTS` module → `Network` module. Adding a new game = implement `GameState` only.

# Backgammon AI Tutor

**A superhuman Backgammon agent powered by AlphaZero-style reinforcement learning, with a real-time coaching web interface.**

Train an AI from scratch through pure self-play — no human game data, no handcrafted heuristics — then play against it while it teaches you optimal strategy in real time.

```
                    AlphaZero Self-Play
                    ┌───────────────┐
                    │  Neural Net   │
                    │  (policy +    │◄──── Training Loop
                    │   value)      │       (MSE + CE loss)
                    └──────┬────────┘
                           │
                    ┌──────▼────────┐
                    │  Stochastic   │
                    │    MCTS       │◄──── Chance Nodes
                    │  (with PUCT)  │       (21 dice outcomes)
                    └──────┬────────┘
                           │
                    ┌──────▼────────┐
                    │  Game Engine  │
                    │  (151 action  │◄──── Full rules:
                    │   space)      │       hitting, bar,
                    └──────┬────────┘       bearing off,
                           │                forced moves
                    ┌──────▼────────┐
                    │   Web Tutor   │
                    │  Win%, equity,│◄──── Real-time
                    │  blunder alert│       coaching
                    └───────────────┘
```

---

## Why This Exists

Most Backgammon programs rely on TD-Gammon-style temporal difference learning — effective, but limited to a single game. This project uses **AlphaZero's approach** (deep neural network + Monte Carlo Tree Search), which generalizes across games. The architecture is modular: swap in a new `GameState` implementation and the entire training pipeline (MCTS, neural network, self-play, evaluation) works for Chess, Go, Othello, or any other game.

This is the first project in a planned series of game AIs sharing the same infrastructure.

---

## Features

### Game Engine
- Complete Backgammon rules: hitting, blocking, bar entry, bearing off (exact and oversize dice), forced move maximization
- 151-action discrete space (25 sources x 6 die values + no-move) — avoids combinatorial explosion by decomposing turns into sequential sub-moves
- 197-dimensional state encoding optimized for neural network input
- 228+ unit tests covering edge cases, invariants, and random game simulations

### AI Training Pipeline
- **Dual-head ResNet** — policy head (action probabilities) and value head (win prediction) with configurable depth and width
- **Stochastic MCTS** — chance nodes model all 21 distinct dice outcomes with lazy expansion for memory efficiency
- **Self-play loop** with replay buffer, champion evaluation gating (55% win rate threshold), and Dirichlet noise for exploration
- **Health monitoring** — NaN/Inf detection, entropy collapse warnings, periodic evaluation against random baseline
- **Checkpoint-resume** — aggressive saving for cloud sessions that die unexpectedly; full state reconstruction from any checkpoint

### Web Tutor Interface
- Interactive canvas board with click-to-move and legal move highlighting
- **Real-time coaching panel:**
  - Win probability bar (neural network value head)
  - Top-5 moves ranked by equity (evaluates each resulting position)
  - Blunder detection with alerts (>5% equity drop)
  - Adjustable difficulty slider (controls MCTS simulation count)

### Hardware Flexibility
- Automatic device detection: CUDA > MPS > CPU
- Designed for local development on Apple Silicon and heavy training on cloud GPUs (Colab/Kaggle free tier)

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/keepkipod/backgammon.git
cd backgammon
pip install -r requirements.txt
```

### Run the Web UI (no trained model — AI plays randomly)

```bash
python -m web.server
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Run with a Trained Model

```bash
python -m web.server --checkpoint checkpoints/latest.pt --simulations 50
```

### Train from Scratch

```bash
# Smoke test first (~6 seconds, validates full pipeline)
python train.py --smoke-test

# Full training run
python train.py --config config.yaml

# Resume from checkpoint
python train.py --resume-from checkpoints/latest.pt
```

### Run Tests

```bash
# Core tests (fast)
python -m pytest tests/test_game_engine.py tests/test_training.py tests/test_server.py -v

# Environment tests (slower — includes 50-game random simulations)
python -m pytest tests/test_env.py -v
```

---

## Architecture

```
backgammon/
├── game_state.py          # Abstract GameState interface (implement for any game)
├── backgammon_game.py     # Backgammon rules, state transitions, encoding
├── env.py                 # Gymnasium RL environment wrapper
├── network.py             # Dual-head ResNet (policy + value)
├── mcts.py                # Stochastic MCTS with chance nodes
└── trainer.py             # Self-play, training loop, evaluation, checkpointing

web/
├── server.py              # Flask REST API + AI analysis endpoints
└── static/
    └── index.html         # Single-page canvas board + coaching UI

train.py                   # CLI entry point for training
config.yaml                # All hyperparameters with documented defaults
tests/                     # 228+ tests across 4 test files
```

### How the Pieces Fit Together

```
GameState (abstract)  ◄── Any game implements this interface
     │
     ├──► MCTS          Uses GameState for tree expansion + neural net for leaf eval
     │      │
     │      ▼
     ├──► Trainer        Orchestrates self-play (MCTS) → collect data → train network
     │      │
     │      ▼
     ├──► Network        ResNet evaluated by MCTS; trained by Trainer
     │
     ├──► Env            Gymnasium wrapper for standardized RL interaction
     │
     └──► Server         Hosts game + uses MCTS/Network for AI moves & analysis
```

The key abstraction is `GameState`. MCTS, the neural network, and the training loop never import Backgammon-specific code — they operate on the abstract interface. To add a new game, you implement `GameState` and everything else works.

---

## Configuration

All hyperparameters live in `config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `hidden_size` | 128 | ResNet hidden layer width |
| `num_res_blocks` | 5 | Number of residual blocks |
| `num_simulations` | 200 | MCTS simulations per move |
| `c_puct` | 1.5 | Exploration constant for PUCT |
| `dirichlet_alpha` | 0.3 | Noise parameter for root exploration |
| `batch_size` | 256 | Training batch size |
| `learning_rate` | 0.001 | Adam learning rate |
| `num_self_play_games` | 100 | Games per self-play iteration |
| `replay_buffer_size` | 100,000 | Max training examples stored |
| `win_rate_threshold` | 0.55 | New model must beat champion by this margin |

---

## How It Works

### 1. State Encoding

The board is encoded as **197 floats**: for each of the 24 points and each player, 4 threshold features capture checker density (1, 2, 3, 4+). Additional features encode checkers on the bar, checkers borne off, and the current player.

### 2. Action Space

Instead of enumerating all possible full turns (combinatorial explosion), each turn is decomposed into **sequential sub-moves**. A sub-move is `(source_point, die_value)`, yielding 25 x 6 + 1 = **151 discrete actions**. A full turn consists of 2-4 sub-moves.

### 3. Monte Carlo Tree Search

MCTS handles Backgammon's stochastic nature through **chance nodes** — after a player acts, the tree branches into 21 possible dice outcomes (weighted by probability). Decision nodes use PUCT selection; chance nodes sample proportionally. The neural network evaluates leaf positions instead of random rollouts.

### 4. Self-Play Training Loop

```
repeat:
  1. Generate N games of self-play using MCTS
  2. Store (state, MCTS_policy, game_outcome) in replay buffer
  3. Train network on random batches (policy CE + value MSE)
  4. Evaluate new network vs. current champion (40 games)
  5. If win rate > 55%: promote to champion, save checkpoint
```

### 5. Real-Time Coaching

The web UI runs MCTS on each position to provide:
- **Win probability** from the value head
- **Move equity** by evaluating the resulting position of each legal move
- **Blunder detection** when your move drops equity by more than 5%

---

## Adding a New Game

1. Create `your_game.py` implementing the `GameState` abstract class
2. Define: initial state, legal actions, state transitions, terminal conditions, encoding, action indexing
3. Handle chance nodes if the game has randomness (optional — deterministic games skip this)
4. Update `config.yaml` with appropriate hyperparameters
5. Train: `python train.py --config config.yaml`

The MCTS, neural network, training loop, and evaluation pipeline require zero modifications.

---

## Tech Stack

| Component | Technology |
|---|---|
| Game Engine | Python, NumPy |
| Neural Network | PyTorch |
| RL Environment | Gymnasium |
| Training | PyTorch + custom self-play loop |
| Web Server | Flask |
| Web UI | Vanilla JS + Canvas |
| Configuration | PyYAML |
| Testing | pytest |

---

## License

MIT

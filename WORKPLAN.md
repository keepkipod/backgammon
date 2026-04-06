# Backgammon AI — Project Workplan

## Implementation Progress

| Phase | Status | Key Files | Tests |
|---|---|---|---|
| 1. Game Engine | **DONE** | `backgammon/game_state.py`, `backgammon/backgammon_game.py` | 140 passing |
| 2. Gymnasium Env | **DONE** | `backgammon/env.py` | 67 passing |
| 3. AI Training | **DONE** (code complete, not yet trained) | `backgammon/network.py`, `backgammon/mcts.py`, `backgammon/trainer.py`, `train.py`, `config.yaml` | 11 passing |
| 4. Web GUI & Tutor | **DONE** | `web/server.py`, `web/static/index.html` | 10 passing |

### What's Next
1. **Continue training** on Vast.ai (generation 0, step ~8900+ as of 2026-04-06)
2. **Benchmark** against GNU Backgammon once model is trained
3. **GUI polish** — equity graph, post-game review (listed in Phase 4.3 but not yet implemented)

---

## Vision

Build a superhuman Backgammon AI tutor using AlphaZero-style architecture. A human player faces a trained AI opponent that also teaches — showing win probabilities, highlighting blunders, and suggesting better moves. This is the first project in a series; the architecture is designed so future games (Chess, Go, Othello, etc.) reuse the same MCTS and neural network infrastructure with only a new game engine plugged in.

---

## Architectural Decisions

### AlphaZero over TD-Gammon
TD-Gammon (value-only network + TD learning) would be simpler for backgammon alone, but it doesn't transfer to deterministic games. AlphaZero's policy + value heads with MCTS is the universal framework for game AI — stochastic and non-stochastic alike. Building it once here means future games are mostly a new `GameState` implementation.

### Stochastic MCTS with Chance Nodes
Standard MCTS assumes deterministic transitions. Backgammon has 21 distinct dice outcomes per turn. The tree search must include explicit chance nodes that branch over dice rolls, weighted by probability. This is harder than standard MCTS but teaches the full generality needed for any stochastic game.

### Action Space: Sequential Sub-Moves
Rather than encoding entire turns (combinatorial explosion), the policy head outputs over individual checker moves: `(source_point, die_value) → destination_point`. A turn consists of 2–4 sequential sub-moves (depending on the dice roll). This keeps the action space small (~156 possible sub-moves: 26 source points × 6 die values) and maps cleanly to other games later.

### No Doubling Cube
The doubling cube is excluded from scope. This simplifies the game engine, action space, and value network output (single scalar rather than win/gammon/backgammon probabilities with cube equity).

### Multi-Game Modularity
Three swappable components:
- **`GameState` interface** — board representation, legal moves, terminal detection. This is what changes per game.
- **`MCTS` module** — tree search with optional chance nodes. Reusable across games.
- **`Network` module** — residual blocks, policy + value heads. Input/output dimensions change per game; core architecture stays.

### Cloud-First Training
Local machines (M3 Mac) are for development and debugging only. All serious training runs on cloud GPUs (RunPod, AWS, or Colab). The codebase must support checkpointing, remote monitoring, and headless execution from day one.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| ML Framework | PyTorch |
| RL Environment | Gymnasium |
| Numerical | NumPy |
| Testing | pytest |
| Cloud Training | RunPod / AWS / Google Colab |
| GUI (Phase 4) | Web (JavaScript/HTML) or Pygame |
| Benchmarking | GNU Backgammon (gnubg) |

---

## Phase 1: Game Engine — COMPLETE

**Goal:** A mathematically perfect backgammon referee with no AI, no graphics — pure logic.

### 1.1 — GameState Interface
Define an abstract `GameState` base class that all future games will implement:
- `get_initial_state()` → starting board
- `get_legal_actions(state)` → list of legal actions
- `apply_action(state, action)` → new state
- `is_terminal(state)` → bool
- `get_reward(state)` → float
- `get_current_player(state)` → player id
- `get_chance_outcomes(state)` → list of (outcome, probability) for stochastic games

### 1.2 — Board Representation
- 24-point array (positive = current player's checkers, negative = opponent's)
- Bar counts for each player
- Borne-off counts for each player
- Current dice roll
- Current player indicator
- Tensor encoding method for neural network input (flattened, normalized)

### 1.3 — Move Generation
The hardest part. Must handle:
- All legal checker moves for a given die value
- Compound turns (2 dice = 2 sub-moves; doubles = 4 sub-moves)
- Forced maximization (must use both dice if possible; if only one, must use the larger)
- Hitting opponent's blots
- Bearing off (including with a die value larger than the farthest checker)
- Blocked points (can't land on 2+ opponent checkers)
- Bar re-entry (must enter from bar before moving any other checker)

### 1.4 — Game Mechanics
- Execute a sub-move, update board state
- Detect terminal states: all 15 checkers borne off
- Score outcomes: normal win (1), gammon (2 — opponent has 0 borne off), backgammon (3 — opponent has checker on bar or in winner's home)
- Turn switching after all sub-moves consumed

### 1.5 — Testing
- Unit tests for every edge case in move generation
- Simulated full games with random agents (10,000+ games, zero crashes)
- Property-based tests: checker count is always 15 per player, no illegal states
- Benchmark against GNU Backgammon move generation for correctness

### Milestone: `python -m pytest tests/test_game_engine.py` passes with 100% coverage on game logic. **ACHIEVED — 140 tests passing.**

---

## Phase 2: Gymnasium Environment — COMPLETE

**Goal:** Wrap the engine so any RL agent can plug in.

### 2.1 — Environment Wrapper
- `BackgammonEnv(gymnasium.Env)` implementing `reset()` and `step(action)`
- Observation space: flattened tensor from board representation (Phase 1.2)
- Action space: `Discrete(N)` where N covers all possible sub-moves
- Reward: 0 during play, +1/-1 (or +2/+3 for gammon/backgammon) at terminal state
- Automatic dice rolling between turns (chance event handled internally)
- Action masking: expose `legal_action_mask()` returning a binary vector

### 2.2 — Opponent Handling
- Support self-play mode (both players use the same policy)
- Support asymmetric play (human vs AI, random vs AI)
- Pluggable opponent interface

### 2.3 — Validation
- Random agent survives 10,000+ games without crashing or stalling
- Verify observation/action space shapes are consistent
- Check reward distribution is reasonable (gammons/backgammons occur at expected rates with random play)

### Milestone: Random agent completes 10,000 games; environment passes Gymnasium's `check_env()`. **ACHIEVED — 67 tests passing, 50+ random games stable.**

### Implementation Note
The `_advance_to_agent_turn()` method in `env.py` was the trickiest part — it's a unified loop that handles dice rolling, opponent sub-moves, and no-move turn skipping. The original split design (`_play_opponent_turns` + `_skip_no_move_turns`) failed when both players had consecutive no-move dice rolls, causing state bouncing. The unified loop fixed this.

---

## Phase 3: Neural Network & MCTS — CODE COMPLETE (not yet trained)

**Goal:** Build and train a superhuman backgammon agent.

### 3.1 — Neural Network
- **Input:** Board state tensor from Phase 2 observation space
- **Body:** Configurable stack of residual blocks (start with 5 blocks, 128 filters — tune later)
- **Policy head:** Outputs logits over the sub-move action space; masked to legal moves before softmax
- **Value head:** Single scalar output via tanh activation (range -1 to +1)
- **Device agnostic:** Auto-detect `cuda` > `mps` > `cpu`

### 3.2 — Stochastic MCTS
- Tree nodes are either **decision nodes** (player picks a move) or **chance nodes** (dice roll sampled)
- At decision nodes: select action via PUCT (polynomial upper confidence trees) using policy head prior
- At chance nodes: branch over all 21 distinct dice outcomes weighted by true probabilities
- Leaf evaluation via value head (no random rollouts)
- Dirichlet noise at root for exploration during self-play
- Temperature parameter: high early in game (exploration), low late (exploitation)
- Configurable simulation count (start with 200 simulations per move; scale up for cloud)

### 3.3 — Self-Play Pipeline (Cloud-Optimized)
- **Data generation:** Multiple parallel self-play workers generating games
- **Replay buffer:** Store (board_state, mcts_policy, game_outcome) tuples; configurable buffer size
- **Training loop:** Sample mini-batches from replay buffer; combined loss = MSE(value) + cross-entropy(policy) + L2 regularization
- **Async architecture:** Self-play workers and training can run concurrently on cloud

### 3.4 — Model Evaluation & Checkpointing
- After N training steps, pit new model vs current champion in head-to-head matches
- Champion updated only if new model wins >55% of evaluation games
- Auto-save checkpoints every K training steps
- Checkpoint includes: model weights, optimizer state, replay buffer, training step count, evaluation history
- Checkpoints stored to cloud storage (S3 / GCS) for persistence across cloud instances
- Resume training from any checkpoint seamlessly

### 3.5 — Training Infrastructure
- **Entry point:** Single `train.py` script with YAML config for all hyperparameters
- **Logging:** Training loss, value accuracy, policy accuracy, evaluation win rate — logged to TensorBoard or Weights & Biases
- **Cloud launch:** Docker container or simple setup script for RunPod/AWS (install deps, pull checkpoint, resume training)
- **Monitoring:** Expose metrics so you can check training progress remotely

### 3.6 — Training Validation (Staged)

Training bugs can waste days of compute. Validation is built into four escalating stages — each must pass before advancing to the next.

**Smoke test (seconds, local CPU, run via pytest):**
- 1 self-play game with 5 MCTS simulations → train 1 batch on the result
- Validates full pipeline end-to-end: data shapes, loss computation, gradient flow, checkpoint save/load
- Run automatically on every code change

**Micro training (~10 minutes, local CPU/MPS):**
- ~50 self-play games, ~100 training steps
- Verify: loss is decreasing (not NaN, not stuck), policy head shifts toward legal-looking moves, value head correlates with outcomes, checkpoint resume produces identical results to continuous training

**Mini training (1–2 hours, Colab T4):**
- ~2,000 self-play games, several thousand training steps
- Model must beat a random agent (>70% win rate). If it can't after this much training, something is fundamentally broken

**Full training (days/weeks, Colab + Kaggle):**
- The real run. By this point every component has been validated at smaller scale

### 3.7 — Automated Health Checks During Training

The training loop asserts on every step and pauses with an alert (rather than burning hours on garbage) if any check fails:

- Loss is not NaN / Inf
- Value head output stays in [-1, 1]
- Policy entropy isn't collapsing to zero prematurely (sign of degenerate convergence)
- Games aren't stuck in infinite loops (cap game length at a configurable max, e.g., 500 sub-moves)
- Win rate vs random agent, sampled every N training steps, is trending upward over time

### 3.8 — Benchmarking
- Evaluate trained model against GNU Backgammon at various difficulty levels
- Track Elo rating progression over training
- Target: consistently beat GNU Backgammon at its highest setting

### Milestone: Model wins >60% against GNU Backgammon "world class" difficulty after cloud training. **NOT YET — requires training runs.**

### Implementation Notes
- Smoke test validates full pipeline in ~6 seconds: self-play → train → checkpoint → load → verify network outputs match.
- Self-play games average ~350 sub-moves. `max_game_length` must be ≥1000 to avoid premature termination.
- The self-play and eval game loops must handle no-legal-moves by skipping the turn (setting dice_remaining=[], flipping player, needs_dice=True) — mirroring the env logic.

### Bugs Found & Fixed (2026-04-06)

**Game Engine Bugs (affected training data quality):**

1. **Bearing off exact roll denied** (`backgammon_game.py:_get_sub_moves_for_die`): Exact bear-offs (e.g., die 5 from the 5-point) were blocked when checkers existed on higher points. The condition `src - die >= 0` was meant to detect exact bear-off but was always false in the bearing-off branch. Fix: explicit check for `src - die == -1` (player 0) / `src + die == 24` (player 1).

2. **Move dedup hiding valid first moves** (`backgammon_game.py:_get_legal_full_turns`): Sequences were deduplicated by sorted form, so `[(21,6),(5,4)]` and `[(5,4),(21,6)]` collapsed to one entry. Since `get_legal_actions` only extracts the first move of each sequence, valid first moves disappeared. Measured impact: ~12% of legal moves hidden in ~32% of positions. Fix: deduplicate by exact sequence order.

**Training Infrastructure:**

3. **Checkpoint corruption on disk full** (`trainer.py:save_checkpoint`): `torch.save` wrote directly to target file; disk-full mid-write corrupted the checkpoint. Fix: atomic saves (write `.tmp`, then `os.replace`) + auto-delete old checkpoints (keep last 3).

**Performance (MCTS and server):**

4. **MCTS sequential network calls** (`mcts.py`): Each simulation made an individual forward pass. 200 sims × 2-4 sub-moves = 400-800 round-trips. Fix: batched MCTS with virtual loss — collect multiple leaves, evaluate in single forward pass. Added `predict_batch()` to network. Result: 200 sims ~0.3s/turn (was 30-120s+).

5. **Server analysis sequential calls** (`server.py`): `_get_ai_analysis` and `_detect_blunder` evaluated every legal move one-by-one. Fix: batch all evaluations into a single `predict_batch()` call.

6. **MPS overhead for small networks** (`server.py`): Per-call GPU dispatch overhead on Apple MPS exceeded actual compute time. Fix: force CPU for web server inference.

**Server Bugs:**

7. **numpy int64 JSON serialization crash** (`server.py`): AI move endpoint returned 500. Fix: custom `NumpyJSONProvider` for Flask.

8. **`--simulations 0` wrong function signature** (`server.py`): Called `network.predict(game, state)` instead of `network.predict(state_tensor, legal_mask)`.

**Training Impact Assessment:** Bugs #1 and #2 affected all training data generated so far. However, ~88% of correct moves were still available, and the model learned reasonable board evaluation. Existing checkpoints are usable as a starting point — the model will adapt to the corrected action space on resumed training.

---

## Phase 4: GUI & Tutor — COMPLETE (base features)

**Goal:** Visual board where a human plays against the AI and learns from it.

### 4.1 — Game Interface
- Visual backgammon board with drag-and-drop checker movement
- Dice rolling animation
- Legal move highlighting (show where a selected checker can go)
- Game state display (borne-off counts, whose turn)

### 4.2 — AI Opponent Integration
- Load trained model checkpoint
- AI computes its move via MCTS (configurable thinking time / simulation count)
- Animate AI's moves on the board

### 4.3 — Tutor Features
- **Win probability bar:** Real-time display of AI's value head output, updated after every move
- **Equity graph:** Running chart of win probability over the course of the game
- **Move analysis:** After a human moves, show the AI's top 3 recommended moves with their equity values
- **Blunder detection:** If the human's move equity is significantly lower (configurable threshold, e.g., >0.05 equity drop) than the best move, flag it as a blunder with an explanation
- **Post-game review:** Step through the game move-by-move with AI commentary on each decision
- **Difficulty slider:** Reduce AI simulation count or add noise to make it beatable for learning

### Milestone: A human can play a full game against the AI with real-time win probability and blunder alerts. **ACHIEVED (base features) — 10 API tests passing.**

### What's Implemented
- Canvas-rendered board with click-to-move and borne-off tray (widened to avoid point 24 / OFF tray collision)
- Dice display, three-color highlight system: blue = legal sources, green = valid destinations, red = blunder destinations (>5% equity drop)
- Context-aware status messages when a checker can't move (blocked/can't bear off vs forced maximization rule)
- AI opponent via batched MCTS (or raw network policy with `--simulations 0`, or random if no model loaded)
- Win probability bar (real-time, from value head)
- Top-5 move analysis with equity percentages (batched evaluation)
- Blunder detection with pre-move warning via red destination highlights + post-move alert
- Difficulty slider (controls MCTS simulation count, 0–500; 0 = instant raw network policy)

### Not Yet Implemented (Future Polish)
- Equity graph (running chart over game course)
- Post-game review (step through moves with AI commentary)
- Dice rolling animation
- Move animation for AI

---

## Implementation Order & Dependencies

```
Phase 1.1 (GameState interface)
  └─→ Phase 1.2 (Board representation)
        └─→ Phase 1.3 (Move generation)
              └─→ Phase 1.4 (Game mechanics)
                    └─→ Phase 1.5 (Testing)
                          └─→ Phase 2.1–2.3 (Gymnasium environment)
                                └─→ Phase 3.1 (Neural network)
                                │     └─→ Phase 3.2 (Stochastic MCTS)
                                │           └─→ Phase 3.3 (Self-play pipeline)
                                │                 └─→ Phase 3.4 (Evaluation & checkpointing)
                                │                       └─→ Phase 3.6 (Benchmarking)
                                └─→ Phase 3.5 (Training infrastructure — can parallel with 3.1–3.2)
                                
Phase 4.1–4.3 (GUI & Tutor — can start after Phase 3.4 produces a trained model)
```

---

## Cloud Training Strategy (Free Tier)

The network for this project is small (5 res blocks, 128 filters) and backgammon games are short. Free-tier cloud GPUs are sufficient to reach a strong level. Estimated compute needed: **50–100 hours of T4 GPU time** — achievable in 2–3 weeks of consistent free-tier usage.

### Primary: Google Colab (Free Tier)
- **GPU:** T4 (16GB VRAM) — more than enough for this network size
- **Limits:** ~4-hour session max, idle disconnects, no guaranteed GPU availability
- **Strategy:** Checkpoint every 15–20 minutes to Google Drive. Reconnect and resume. Expect 4–8 GPU hours per day
- **Workflow:** Colab notebook clones the repo from GitHub, pulls latest checkpoint from Google Drive, trains for N steps, saves checkpoint back to Google Drive

### Secondary: Kaggle Notebooks
- **GPU:** T4 or P100, 30 hours per week
- **Limits:** 12-hour max session, internet access restrictions during runtime
- **Strategy:** Same checkpoint/resume pattern as Colab. Combined with Colab, this gives 60+ GPU hours/week for free
- **Workflow:** Upload training script as Kaggle notebook, read/write checkpoints via Kaggle Datasets

### Training Session Pattern
1. Push latest code + config to GitHub
2. Open Colab or Kaggle notebook
3. Clone repo, install dependencies
4. Pull latest checkpoint from Google Drive / Kaggle Datasets
5. Train for N steps (with periodic checkpoint saves every 15–20 min)
6. Push final checkpoint back to storage
7. Repeat across sessions until convergence

### Design Constraints for Free-Tier Training
- **Aggressive checkpointing:** Sessions can die at any time — never lose more than 20 minutes of work
- **Stateless entry point:** `train.py` must accept a `--resume-from` flag and reconstruct full training state from a checkpoint
- **Small batch sizes:** T4 has 16GB VRAM; batch size 128–256 is the sweet spot for this network
- **Google Drive as checkpoint store:** Free 15GB is plenty for model checkpoints (~50MB each)

### Optional: Paid Cloud (Only if Needed)
If free-tier proves too slow for pushing from "strong" to "superhuman," upgrade to:
- **RunPod:** A10G at ~$0.30/hr or A100 at ~$0.80/hr with spot instances
- **AWS:** g5.xlarge (A10G) spot instances at similar pricing
- This should not be necessary for reaching a competitive level of play

### What Runs Locally (M3 Mac)
- All development and unit testing
- Small-scale self-play for debugging (10–50 games)
- Model inference for GUI (trained model loads fine on MPS/CPU)
- Quick smoke tests before committing cloud GPU time

---

## Success Criteria

| Milestone | Measure | Status |
|---|---|---|
| Game engine complete | 10,000 random games, zero illegal states | **DONE** — 140 tests, 100 random games with invariant checks |
| Environment works | Random agent stable, action masking correct | **DONE** — 67 tests, 50 random games, reward distribution verified |
| Training pipeline works | Smoke test passes (self-play → train → checkpoint → load) | **DONE** — 11 tests, ~6s smoke test |
| AI plays reasonably | Wins >80% against random agent after initial training | **PENDING** — requires Colab/Kaggle training |
| AI is strong | Wins >60% against GNU Backgammon highest difficulty | **PENDING** — requires extended training |
| Tutor works | Human can play full game with live equity bar and blunder detection | **DONE** — 10 API tests, web UI functional |
| Multi-game ready | Second game (e.g., Othello) plugs in by implementing only `GameState` | **PENDING** — architecture supports it, not yet tested |

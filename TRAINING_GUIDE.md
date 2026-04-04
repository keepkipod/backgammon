# Training Guide: From Zero to Superhuman Backgammon

This guide walks through training the backgammon AI from scratch, starting with local validation and scaling up to cloud GPUs.

---

## Step 0: Local Prerequisites

Make sure everything works on your M3 Mac before touching cloud resources.

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the smoke test (should pass in ~6 seconds)
python3 train.py --smoke-test

# Run the full test suite (skip env stress tests for speed)
python3 -m pytest tests/test_game_engine.py tests/test_training.py tests/test_server.py -v
```

If the smoke test passes, the full pipeline (self-play → train → checkpoint → load) is working correctly.

---

## Step 1: Local Micro Training (~10 minutes, M3 Mac)

**Purpose:** Verify that loss decreases and the model actually learns. Catches training bugs before spending cloud GPU time.

Create `config_micro.yaml`:

```yaml
# Micro training config — local validation only
hidden_size: 128
num_res_blocks: 5
num_simulations: 25
c_puct: 1.5
dirichlet_alpha: 0.3
dirichlet_epsilon: 0.25
temperature_moves: 15

num_self_play_games: 10
max_game_length: 1000

batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
num_training_steps: 50
replay_buffer_size: 10000

num_eval_games: 10
win_rate_threshold: 0.55

checkpoint_dir: checkpoints_micro
checkpoint_interval: 25
max_policy_entropy_drop: 0.5
eval_vs_random_interval: 50
```

Run it:

```bash
python3 train.py --config config_micro.yaml
```

**What to check:**
- Loss prints every 25 steps — it should be **decreasing**
- No NaN/Inf errors
- No entropy collapse warnings
- Self-play games complete without hanging

If loss is stuck or increasing, something is wrong — do NOT proceed to cloud training.

---

## Step 2: Google Colab Setup (Mini Training)

### 2.1 — Push Code to GitHub

Make sure your latest code is pushed:

```bash
git add -A
git commit -m "Ready for cloud training"
git push origin main
```

### 2.2 — Create the Colab Notebook

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook. **Paste and run each cell one at a time, in order** — each cell depends on the previous one completing before you proceed.

**Cell 1 — Setup & Clone Repo:**

```python
# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Clone the repo
!git clone https://github.com/keepkipod/backgammon.git
%cd backgammon

# Install dependencies
!pip install -r requirements.txt -q
```

**Cell 2 — Mount Google Drive (for checkpoint persistence):**

```python
from google.colab import drive
drive.mount('/content/drive')

# Create checkpoint directory on Google Drive
import os
DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/backgammon_checkpoints'
os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)

# Check for existing checkpoint to resume from
latest = os.path.join(DRIVE_CHECKPOINT_DIR, 'latest.pt')
if os.path.exists(latest):
    print(f"Found existing checkpoint: {latest}")
    print("Will resume from this checkpoint")
else:
    print("No existing checkpoint — starting fresh")
```

**Cell 3 — Smoke Test (verify everything works on Colab):**

```python
!python train.py --smoke-test
```

**Cell 4 — Mini Training Config:**

```python
%%writefile config_colab.yaml
# Colab mini training config
# Aim: beat random agent >70% in 1-2 hours

hidden_size: 128
num_res_blocks: 5

num_simulations: 100
c_puct: 1.5
dirichlet_alpha: 0.3
dirichlet_epsilon: 0.25
temperature_moves: 30

num_self_play_games: 50
max_game_length: 1000

batch_size: 256
learning_rate: 0.001
weight_decay: 0.0001
num_training_steps: 500
replay_buffer_size: 100000

num_eval_games: 20
win_rate_threshold: 0.55

checkpoint_dir: /content/drive/MyDrive/backgammon_checkpoints
checkpoint_interval: 50

max_policy_entropy_drop: 0.5
eval_vs_random_interval: 100
```

**Cell 5 — Run Training:**

```python
import os

# Check for existing checkpoint
DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/backgammon_checkpoints'
latest = os.path.join(DRIVE_CHECKPOINT_DIR, 'latest.pt')

if os.path.exists(latest):
    print("Resuming from checkpoint...")
    !python train.py --config config_colab.yaml --resume-from {latest}
else:
    print("Starting fresh training...")
    !python train.py --config config_colab.yaml
```

**Cell 6 — Check Progress (run anytime):**

```python
import torch
import os

DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/backgammon_checkpoints'
latest = os.path.join(DRIVE_CHECKPOINT_DIR, 'latest.pt')

if os.path.exists(latest):
    ckpt = torch.load(latest, map_location='cpu', weights_only=False)
    print(f"Training step: {ckpt['training_step']}")
    print(f"Generation (champion updates): {ckpt['generation']}")
    print(f"Replay buffer size: {len(ckpt['replay_buffer'])}")
    if ckpt.get('training_history'):
        last = ckpt['training_history'][-1]
        print(f"Last loss: {last.get('total_loss', 'N/A'):.4f}")
        print(f"  Value loss: {last.get('value_loss', 'N/A'):.4f}")
        print(f"  Policy loss: {last.get('policy_loss', 'N/A'):.4f}")
else:
    print("No checkpoint found yet")
```

**Cell 7 — Quick Eval vs Random (run after some training):**

```python
import torch
import sys
sys.path.insert(0, '/content/backgammon')

from backgammon.backgammon_game import BackgammonGame
from backgammon.trainer import Trainer, TrainConfig

game = BackgammonGame()
config = TrainConfig(
    checkpoint_dir='/content/drive/MyDrive/backgammon_checkpoints',
    num_simulations=50,
)
trainer = Trainer(game, config)
trainer.load_checkpoint('/content/drive/MyDrive/backgammon_checkpoints/latest.pt')
win_rate = trainer.evaluate_vs_random(num_games=50)
print(f"\nWin rate vs random: {win_rate:.1%}")
```

### 2.3 — What to Expect (Mini Training Timeline)

| Time | Expected State |
|---|---|
| 0-15 min | First self-play batch generating (slow — MCTS is expensive) |
| 15-30 min | First training steps, loss should be high but finite |
| 30-60 min | Loss decreasing, maybe first champion update |
| 1-2 hrs | Should beat random agent >60-70% |

If after 2 hours the model can't beat random >50%, something is wrong.

---

## Step 3: Full Training (Days/Weeks)

Once mini training proves the model learns, scale up for full training.

### 3.1 — Full Training Config

```yaml
# Full training config — for serious cloud training
hidden_size: 128
num_res_blocks: 5

num_simulations: 200
c_puct: 1.5
dirichlet_alpha: 0.3
dirichlet_epsilon: 0.25
temperature_moves: 30

num_self_play_games: 100
max_game_length: 1000

batch_size: 256
learning_rate: 0.001
weight_decay: 0.0001
num_training_steps: 1000
replay_buffer_size: 100000

num_eval_games: 40
win_rate_threshold: 0.55

checkpoint_dir: /content/drive/MyDrive/backgammon_checkpoints
checkpoint_interval: 100

max_policy_entropy_drop: 0.5
eval_vs_random_interval: 200
```

### 3.2 — Daily Routine (Free Tier)

Colab free tier gives ~4-hour sessions with idle disconnects. Here's the workflow:

1. Open the Colab notebook
2. Run Cell 1 (setup) — takes ~1 min
3. Run Cell 2 (mount Drive) — auto-detects existing checkpoint
4. Run Cell 5 (training) — resumes automatically from latest checkpoint
5. **Keep the tab active** (Colab disconnects idle sessions)
6. When the session dies or GPU quota runs out, close and try again later

**Tips to maximize free GPU time:**
- Colab resets GPU quota roughly every 12-24 hours
- Train in morning and evening sessions
- Use Kaggle (Step 3.3) as supplemental compute when Colab is unavailable

### 3.3 — Kaggle as Supplemental Compute

Kaggle gives 30 GPU hours/week with 12-hour max sessions.

1. Go to [Kaggle](https://www.kaggle.com/) → New Notebook
2. Turn on GPU: Settings → Accelerator → GPU T4 x2
3. Turn on Internet: Settings → Internet → On

**Kaggle notebook cells:**

```python
# Cell 1: Setup
!git clone https://github.com/keepkipod/backgammon.git
%cd backgammon
!pip install -r requirements.txt -q

# Cell 2: Upload/download checkpoints via Kaggle Datasets
# Option A: Upload your latest.pt as a Kaggle Dataset, then:
# import shutil
# shutil.copy('/kaggle/input/backgammon-checkpoints/latest.pt', 'checkpoints/latest.pt')

# Option B: Use Google Drive via gdown (install first)
!pip install gdown -q
# Share your checkpoint file on Google Drive, get the link, then:
# !gdown "https://drive.google.com/uc?id=YOUR_FILE_ID" -O checkpoints/latest.pt

# Cell 3: Train
!python train.py --config config.yaml --resume-from checkpoints/latest.pt

# Cell 4: After training, download the checkpoint
# From the Kaggle output files, or copy to a Kaggle Dataset for next session
```

**Transferring checkpoints between Colab and Kaggle:**
- Easiest: Share a Google Drive folder between both
- Alternative: Upload checkpoints as Kaggle Datasets between sessions

### 3.4 — Training Milestones

| Milestone | Estimated Time | Check |
|---|---|---|
| Loss is decreasing steadily | 1-2 hours | Verify in Cell 6 |
| Beats random >70% | 5-10 hours | Run Cell 7 |
| Beats random >90% | 20-30 hours | Model is learning real strategy |
| Multiple champion updates | 30-50 hours | Generations increasing in Cell 6 |
| Competitive play | 50-100 hours | Benchmark vs GNU Backgammon |

---

## Step 4: Using the Trained Model

### 4.1 — Download Checkpoint Locally

```bash
# From Google Drive, download latest.pt to your local machine
# Place it in the checkpoints/ directory
mkdir -p checkpoints
# (copy latest.pt here from Google Drive)
```

### 4.2 — Run the Web UI

```bash
# With trained model
python3 -m web.server --checkpoint checkpoints/latest.pt

# With custom difficulty (fewer simulations = weaker AI)
python3 -m web.server --checkpoint checkpoints/latest.pt --simulations 50
```

Open `http://localhost:5000` in your browser.

### 4.3 — Benchmark Against GNU Backgammon

Install GNU Backgammon (`gnubg`) and run head-to-head evaluation:

```bash
# macOS
brew install gnubg
```

(Automated benchmarking integration is not yet built — this is a future task.)

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Loss is NaN | Numeric overflow, bad learning rate | Reduce `learning_rate` to 0.0001 |
| Loss not decreasing | Learning rate too low, or data issue | Check self-play is generating diverse games |
| Entropy collapse warning | Policy head converging prematurely | Increase `dirichlet_epsilon` to 0.35 |
| Self-play games hang | `max_game_length` too low | Set `max_game_length: 1500` |
| Colab session dies | Normal — free tier has ~4hr limit | Resume from checkpoint (automatic) |
| Out of VRAM on T4 | Batch size too large | Reduce `batch_size` to 128 |
| Model doesn't beat random after hours | Training bug or bad hyperparams | Re-run smoke test, check loss curve |
| Checkpoint file too large | Replay buffer is huge | Reduce `replay_buffer_size` to 50000 |

---

## Summary: Minimum Viable Training Run

1. **Local:** Run smoke test → run micro training (10 min) → verify loss decreases
2. **Colab:** Create notebook → mount Drive → run mini training (1-2 hrs) → verify beats random >70%
3. **Scale:** Switch to full config → train across Colab + Kaggle sessions → target 50-100 GPU hours
4. **Play:** Download checkpoint → run web UI → enjoy your AI tutor

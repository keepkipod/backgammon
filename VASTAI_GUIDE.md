# Training on Vast.ai — Step by Step

## Overview

Finish the backgammon AI training on a rented GPU. You already have a checkpoint from Colab — this guide resumes from it on Vast.ai so nothing is lost.

**Expected cost:** $5–15 total
**Expected time:** 24–48 hours of continuous training on a 4090

---

## Step 1: Download Your Colab Checkpoint

1. Go to Google Drive → `backgammon_checkpoints/`
2. Download `latest.pt` to your local machine (e.g., `~/Downloads/latest.pt`)

---

## Step 2: Create a Vast.ai Account

1. Go to [vast.ai](https://vast.ai/) and sign up
2. Add billing — $10–15 credit is enough
3. Go to **Console** → **Templates** → search for **PyTorch** (pick one with CUDA 12.x + PyTorch 2.x)

---

## Step 3: Rent a GPU Instance

1. Go to **Search** (the marketplace)
2. Filter by:
   - **GPU:** RTX 4090 or RTX 3090 (best price/performance for this workload)
   - **Price:** sort by $/hr, look for $0.15–0.30/hr
   - **Disk:** 10 GB is plenty
   - **Internet:** make sure upload/download is enabled
3. Click **Rent** on your chosen instance
4. Wait for it to start (usually 1–2 minutes)

---

## Step 4: Connect via SSH

Vast.ai shows SSH connection details on the instance page. It will look something like:

```bash
ssh -p 12345 root@<vast-ip-address>
```

You may need to add your SSH key first in Vast.ai account settings. Alternatively, use the **web terminal** from the Vast.ai dashboard (no SSH setup needed).

---

## Step 5: Upload Your Checkpoint

From your **local machine** (not the Vast.ai instance), run:

```bash
scp -P <port> ~/Downloads/latest.pt root@<vast-ip>:/root/latest.pt
```

Replace `<port>` and `<vast-ip>` with the values from your instance's SSH details.

---

## Step 6: Set Up the Training Environment

SSH into the instance, then run:

```bash
# Clone the repo
git clone https://github.com/keepkipod/backgammon.git
cd backgammon

# Install dependencies
pip install -r requirements.txt

# Move checkpoint into place
mkdir -p checkpoints
mv /root/latest.pt checkpoints/latest.pt

# Verify it works
python train.py --smoke-test
```

---

## Step 7: Create the Full Training Config

```bash
cat > config_vastai.yaml << 'EOF'
# Full training config for Vast.ai
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

checkpoint_dir: checkpoints
checkpoint_interval: 100

max_policy_entropy_drop: 0.5
eval_vs_random_interval: 200
EOF
```

---

## Step 8: Start Training in tmux

tmux keeps the training running even if your SSH connection drops.

```bash
# Start a tmux session
tmux new -s training

# Start training (resumes from your Colab checkpoint)
python train.py --config config_vastai.yaml --resume-from checkpoints/latest.pt

# To detach (training keeps running): press Ctrl+B, then D
# To reattach later: tmux attach -t training
```

---

## Step 9: Monitor Progress

SSH in anytime and check on training:

```bash
# Reattach to see live output
tmux attach -t training

# Or check the latest checkpoint without interrupting
python -c "
import torch
ckpt = torch.load('checkpoints/latest.pt', map_location='cpu', weights_only=False)
print(f'Training step: {ckpt[\"training_step\"]}')
print(f'Generation: {ckpt[\"generation\"]}')
print(f'Buffer size: {len(ckpt[\"replay_buffer\"])}')
if ckpt.get('training_history'):
    last = ckpt['training_history'][-1]
    print(f'Last loss: {last.get(\"total_loss\", \"N/A\"):.4f}')
"
```

### Quick eval vs random (run in a separate SSH session):

```bash
cd /root/backgammon
python -c "
from backgammon.backgammon_game import BackgammonGame
from backgammon.trainer import Trainer, TrainConfig
game = BackgammonGame()
config = TrainConfig(num_simulations=50, checkpoint_dir='checkpoints')
trainer = Trainer(game, config)
trainer.load_checkpoint('checkpoints/latest.pt')
win_rate = trainer.evaluate_vs_random(num_games=50)
print(f'Win rate vs random: {win_rate:.1%}')
"
```

---

## Step 10: Know When to Stop

Training runs in an infinite loop (self-play → train → evaluate → repeat). There is no automatic stop — you decide when the model is strong enough.

### Milestones to check:

| Sign | What it means | Action |
|---|---|---|
| Win rate vs random > 70% | Model learned fundamentals | Keep going |
| Win rate vs random > 90% | Strong basic play | Getting close |
| Multiple champion updates (generation > 3) | Model is improving consistently | Keep going |
| Win rate vs random plateaus at 90%+ | Diminishing returns | Good stopping point |
| Loss stops decreasing across generations | Converged | Stop training |

### When you're satisfied, stop training:

```bash
# In the tmux session, press Ctrl+C to stop training
# The latest checkpoint is already saved (auto-saves every 100 steps)
```

---

## Step 11: Download the Trained Checkpoint

From your **local machine**:

```bash
mkdir -p checkpoints
scp -P <port> root@<vast-ip>:/root/backgammon/checkpoints/latest.pt ./checkpoints/latest.pt
```

---

## Step 12: Destroy the Instance

**Important — billing continues until you destroy it.**

1. Go to the Vast.ai dashboard
2. Click **Destroy** on your instance
3. Confirm

---

## Step 13: Play Against Your AI Locally

```bash
# Start the web UI with the trained model
python3 -m web.server --checkpoint checkpoints/latest.pt

# Optional: set difficulty (fewer simulations = weaker/faster AI)
python3 -m web.server --checkpoint checkpoints/latest.pt --simulations 50
```

Open http://localhost:5000 in your browser.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| SSH connection refused | Instance may still be starting — wait 1–2 min and retry |
| `torch.cuda.is_available()` returns False | You picked a CPU instance — destroy it and rent a GPU one |
| Out of VRAM | Reduce `batch_size` to 128 in the config |
| Training seems stuck | Check `tmux attach -t training` — self-play games are slow (~2–5 min each with 200 sims) |
| Lost SSH but training is running | Just reconnect and `tmux attach -t training` |
| Forgot to download before destroying | Checkpoint is gone — always download first |

---

## Cost Estimate

| GPU | $/hr | 50 hrs | 100 hrs |
|---|---|---|---|
| RTX 3090 | ~$0.15 | $7.50 | $15 |
| RTX 4090 | ~$0.25 | $12.50 | $25 |
| A10G | ~$0.30 | $15 | $30 |

You're resuming from ~1500 training steps, so you likely need 30–60 more hours depending on GPU speed. Budget $10–15.

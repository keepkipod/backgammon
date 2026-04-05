# When to Stop Training

The training loop runs forever. You decide when to stop.

---

## How to Check Progress

Run these in a **separate SSH session** (don't interrupt the training).

### Check win rate vs random (primary metric):

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

### Check generation and training step:

```bash
python -c "
import torch
ckpt = torch.load('checkpoints/latest.pt', map_location='cpu', weights_only=False)
print(f'Training step: {ckpt[\"training_step\"]}')
print(f'Generation (champion updates): {ckpt[\"generation\"]}')
print(f'Buffer size: {len(ckpt[\"replay_buffer\"])}')
if ckpt.get('training_history'):
    last = ckpt['training_history'][-1]
    print(f'Last loss: {last.get(\"total_loss\", \"N/A\"):.4f}')
    print(f'  Value loss: {last.get(\"value_loss\", \"N/A\"):.4f}')
    print(f'  Policy loss: {last.get(\"policy_loss\", \"N/A\"):.4f}')
"
```

---

## What the Numbers Mean

### Win Rate vs Random

| Win Rate | Meaning | Action |
|---|---|---|
| < 50% | Model is broken or too early | Check logs for errors, keep training |
| 50–70% | Learning basic moves | Keep going |
| 70–85% | Understands real strategy | Keep going — getting useful |
| 85–95% | Strong play, makes smart moves | **Good stopping point for a tutor** |
| 95%+ | Near-optimal vs random | Diminishing returns — stop here |

### Generation Count

| Generation | Meaning |
|---|---|
| 0 | No champion update yet — still early |
| 1–3 | Model is improving, new versions beating old ones |
| 4+ | Consistent improvement across multiple evaluations |

### Loss

| Trend | Meaning |
|---|---|
| Decreasing over time | Healthy — model is learning |
| Flat for a long time | Model may have converged — check win rate |
| Increasing or NaN | Something is wrong — stop and investigate |

---

## Recommended Stopping Point

**85%+ win rate vs random** is enough for a backgammon tutor. At this level, the AI:

- Plays meaningfully better than a beginner
- Gives useful win probability estimates
- Detects real blunders (not noise)
- Suggests moves that are genuinely better than yours

You don't need a superhuman AI to learn from it.

---

## How to Stop

```bash
# Reattach to training
tmux attach -t training

# Press Ctrl+C to stop
# The latest checkpoint is already saved (auto-saves every 100 training steps)
```

Then download `checkpoints/latest.pt` to your local machine before destroying the instance.

---

## Something Looks Wrong?

| Symptom | Likely Cause | Fix |
|---|---|---|
| Win rate stuck at ~50% after hours | Not enough self-play data or learning rate too low | Let it run longer — first few generations are slow |
| Win rate was improving, now dropping | Overfitting or entropy collapse | Check for entropy collapse warnings in training logs |
| Loss is NaN | Numeric overflow | Restart from last good checkpoint, reduce learning rate to 0.0001 |
| Generation stays at 0 forever | New models can't beat champion by 55% | This is normal early on — the threshold is strict, keep training |

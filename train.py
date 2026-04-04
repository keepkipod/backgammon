"""Entry point for training the backgammon AI.

Usage:
    # Train with defaults
    python train.py

    # Train with custom config
    python train.py --config config.yaml

    # Resume from checkpoint
    python train.py --resume-from checkpoints/latest.pt

    # Quick smoke test
    python train.py --smoke-test
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from backgammon.backgammon_game import BackgammonGame
from backgammon.trainer import TrainConfig, Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> TrainConfig:
    """Load training config from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    config = TrainConfig()
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config key: {key}")
    return config


def smoke_test():
    """Quick pipeline validation: 1 game, 1 training step, checkpoint roundtrip."""
    logger.info("=== SMOKE TEST ===")

    config = TrainConfig(
        hidden_size=32,
        num_res_blocks=2,
        num_simulations=5,
        num_self_play_games=1,
        num_training_steps=1,
        batch_size=4,
        checkpoint_dir="/tmp/backgammon_smoke_test",
        max_game_length=1000,
    )

    game = BackgammonGame()
    trainer = Trainer(game, config)

    # Self-play
    logger.info("Self-play...")
    examples = trainer.self_play_game()
    logger.info(f"Generated {len(examples)} examples")
    if not examples:
        # Try again with a different seed — rare case where game was too short
        logger.info("Retrying self-play...")
        examples = trainer.self_play_game()
    assert len(examples) > 0, "Self-play produced no examples"
    trainer.replay_buffer.add(examples)

    # Pad buffer to batch size
    while len(trainer.replay_buffer) < config.batch_size:
        trainer.replay_buffer.add(examples)

    # Training step
    logger.info("Training step...")
    metrics = trainer.train_step()
    assert "error" not in metrics, f"Training error: {metrics}"
    assert metrics["total_loss"] > 0, "Loss should be positive"
    logger.info(f"Loss: {metrics['total_loss']:.4f}")

    # Checkpoint save/load roundtrip
    logger.info("Checkpoint roundtrip...")
    trainer.save_checkpoint()

    trainer2 = Trainer(game, config)
    trainer2.load_checkpoint(f"{config.checkpoint_dir}/latest.pt")
    assert trainer2.training_step == trainer.training_step
    assert len(trainer2.replay_buffer) == len(trainer.replay_buffer)

    # Verify network produces same output after loading
    import torch
    import numpy as np
    test_state = game.get_initial_state()
    test_state = game.apply_chance_outcome(test_state, (3, 1))
    state_tensor = torch.FloatTensor(game.encode_state(test_state)).to(trainer.device)
    mask = torch.zeros(game.get_action_space_size(), device=trainer.device)

    trainer.network.eval()
    trainer2.network.eval()
    with torch.no_grad():
        p1, v1 = trainer.network(state_tensor.unsqueeze(0))
        p2, v2 = trainer2.network(state_tensor.unsqueeze(0))
    assert torch.allclose(p1, p2), "Network outputs differ after checkpoint load"
    assert torch.allclose(v1, v2), "Value outputs differ after checkpoint load"

    logger.info("=== SMOKE TEST PASSED ===")


def main():
    parser = argparse.ArgumentParser(description="Train Backgammon AI")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--resume-from", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick pipeline validation")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainConfig()

    logger.info(f"Config: {config}")

    # Create trainer
    game = BackgammonGame()
    trainer = Trainer(game, config)

    # Resume if requested
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    # Run training
    trainer.run_training_loop()


if __name__ == "__main__":
    main()

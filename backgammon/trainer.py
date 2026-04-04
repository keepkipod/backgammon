"""Self-play training pipeline for AlphaZero-style game AI.

Components:
  - Self-play: generate training data via MCTS
  - Replay buffer: store and sample training examples
  - Training loop: optimize network on replay data
  - Champion evaluation: gate model updates on win rate
  - Checkpointing: save/resume full training state
  - Health checks: detect training failures early
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from backgammon.game_state import GameState
from backgammon.mcts import MCTS
from backgammon.network import DualHeadNetwork, get_device

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example from self-play."""

    state: np.ndarray  # encoded state
    policy: np.ndarray  # MCTS policy distribution
    value: float  # actual game outcome from this player's perspective


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Network
    hidden_size: int = 128
    num_res_blocks: int = 5

    # MCTS
    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 30  # use temp=1 for first N moves, then temp=0

    # Self-play
    num_self_play_games: int = 100
    max_game_length: int = 1000  # sub-moves per game safety cap

    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_training_steps: int = 1000
    replay_buffer_size: int = 100_000

    # Evaluation
    num_eval_games: int = 40
    win_rate_threshold: float = 0.55

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 100  # save every N training steps

    # Health checks
    max_policy_entropy_drop: float = 0.5  # alert if entropy drops more than this between checks
    eval_vs_random_interval: int = 200  # evaluate vs random every N steps


class ReplayBuffer:
    """Fixed-size buffer of training examples with uniform sampling."""

    def __init__(self, max_size: int):
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, examples: list[TrainingExample]):
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class Trainer:
    """Orchestrates the full AlphaZero training pipeline."""

    def __init__(self, game: GameState, config: TrainConfig):
        self.game = game
        self.config = config
        self.device = get_device()

        # Network
        input_size = game.encode_state(game.get_initial_state()).shape[0]
        action_size = game.get_action_space_size()
        self.network = DualHeadNetwork(
            input_size=input_size,
            action_size=action_size,
            hidden_size=config.hidden_size,
            num_res_blocks=config.num_res_blocks,
        ).to(self.device)

        # Champion network (for evaluation gating)
        self.champion_network = DualHeadNetwork(
            input_size=input_size,
            action_size=action_size,
            hidden_size=config.hidden_size,
            num_res_blocks=config.num_res_blocks,
        ).to(self.device)
        self.champion_network.load_state_dict(self.network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        # Tracking
        self.training_step = 0
        self.generation = 0
        self.champion_wins = 0
        self.last_entropy = None
        self.training_history: list[dict] = []

    def self_play_game(self) -> list[TrainingExample]:
        """Play a single self-play game and return training examples."""
        mcts = MCTS(
            game=self.game,
            network=self.network,
            device=self.device,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
        )

        state = self.game.get_initial_state()
        examples = []  # (state, policy, current_player)
        move_count = 0

        for _ in range(self.config.max_game_length):
            if self.game.is_terminal(state):
                break

            # Handle chance nodes (dice rolls)
            if self.game.is_chance_node(state):
                outcomes = self.game.get_chance_outcomes(state)
                rolls = [o for o, _ in outcomes]
                probs = [p for _, p in outcomes]
                idx = np.random.choice(len(rolls), p=probs)
                state = self.game.apply_chance_outcome(state, rolls[idx])
                continue

            # Check for legal moves
            legal_actions = self.game.get_legal_actions(state)
            if not legal_actions:
                # No legal moves — skip turn
                state.dice_remaining = []
                state.current_player = 1 - state.current_player
                state.needs_dice = True
                continue

            # Temperature schedule
            temperature = 1.0 if move_count < self.config.temperature_moves else 0.1
            mcts.temperature = temperature

            # MCTS search
            policy, _ = mcts.search(state)

            # Store example
            encoded = self.game.encode_state(state)
            current_player = self.game.get_current_player(state)
            examples.append((encoded, policy, current_player))

            # Select action from MCTS policy
            action_idx = np.random.choice(len(policy), p=policy)
            action = self.game.index_to_action(action_idx)

            # Validate action is legal (policy should only have weight on legal actions)
            if action not in legal_actions:
                # Fallback: pick random legal action
                action = random.choice(legal_actions)

            state = self.game.apply_action(state, action)
            move_count += 1

        # Assign values based on game outcome
        training_examples = []
        if self.game.is_terminal(state):
            for encoded, policy, player in examples:
                value = self.game.get_reward(state, player)
                training_examples.append(
                    TrainingExample(state=encoded, policy=policy, value=float(value))
                )

        return training_examples

    def generate_self_play_data(self, num_games: int) -> int:
        """Run multiple self-play games and add data to replay buffer.

        Returns the number of examples generated.
        """
        total_examples = 0
        for i in range(num_games):
            examples = self.self_play_game()
            if examples:
                self.replay_buffer.add(examples)
                total_examples += len(examples)
            if (i + 1) % 10 == 0:
                logger.info(f"Self-play: {i + 1}/{num_games} games, {total_examples} examples")
        return total_examples

    def train_step(self) -> dict:
        """Run a single training step on a mini-batch from the replay buffer.

        Returns dict with loss metrics.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {"error": "Not enough data in replay buffer"}

        self.network.train()
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Prepare tensors
        states = torch.FloatTensor(np.array([ex.state for ex in batch])).to(self.device)
        target_policies = torch.FloatTensor(np.array([ex.policy for ex in batch])).to(self.device)
        target_values = torch.FloatTensor([[ex.value] for ex in batch]).to(self.device)

        # Forward pass
        pred_logits, pred_values = self.network(states)

        # Value loss: MSE
        value_loss = F.mse_loss(pred_values, target_values)

        # Policy loss: cross-entropy with MCTS policy
        log_probs = F.log_softmax(pred_logits, dim=1)
        policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()

        # Total loss
        loss = value_loss + policy_loss

        # Health check: loss must be finite
        if not torch.isfinite(loss):
            logger.error(f"Non-finite loss at step {self.training_step}: {loss.item()}")
            return {"error": "non-finite loss", "value_loss": float("nan"), "policy_loss": float("nan")}

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_step += 1

        # Compute policy entropy for health monitoring
        with torch.no_grad():
            probs = F.softmax(pred_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()

        metrics = {
            "total_loss": loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "policy_entropy": entropy,
            "step": self.training_step,
        }

        # Health check: entropy collapse
        if self.last_entropy is not None:
            if self.last_entropy - entropy > self.config.max_policy_entropy_drop:
                logger.warning(
                    f"Policy entropy dropped sharply: {self.last_entropy:.3f} -> {entropy:.3f}"
                )
        self.last_entropy = entropy

        return metrics

    def train(self, num_steps: int) -> list[dict]:
        """Run multiple training steps. Returns list of metrics per step."""
        history = []
        for i in range(num_steps):
            metrics = self.train_step()
            history.append(metrics)

            if "error" in metrics:
                logger.error(f"Training error at step {self.training_step}: {metrics['error']}")
                break

            # Checkpoint
            if self.training_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
                logger.info(
                    f"Step {self.training_step}: loss={metrics['total_loss']:.4f} "
                    f"(v={metrics['value_loss']:.4f}, p={metrics['policy_loss']:.4f}), "
                    f"entropy={metrics['policy_entropy']:.3f}"
                )

        self.training_history.extend(history)
        return history

    def evaluate_vs_random(self, num_games: int = 50) -> float:
        """Evaluate current network against a random player.

        Returns win rate of the network.
        """
        wins = 0
        for _ in range(num_games):
            result = self._play_eval_game(use_champion=False, opponent="random")
            if result > 0:
                wins += 1
        win_rate = wins / num_games
        logger.info(f"Eval vs random: {win_rate:.1%} ({wins}/{num_games})")
        return win_rate

    def evaluate_vs_champion(self) -> float:
        """Evaluate current network against the champion.

        Returns win rate of the current network.
        """
        wins = 0
        for _ in range(self.config.num_eval_games):
            result = self._play_eval_game(use_champion=False, opponent="champion")
            if result > 0:
                wins += 1
        win_rate = wins / self.config.num_eval_games
        logger.info(f"Eval vs champion: {win_rate:.1%} ({wins}/{self.config.num_eval_games})")
        return win_rate

    def maybe_update_champion(self) -> bool:
        """Evaluate and update champion if current network is better.

        Returns True if champion was updated.
        """
        win_rate = self.evaluate_vs_champion()
        if win_rate >= self.config.win_rate_threshold:
            self.champion_network.load_state_dict(self.network.state_dict())
            self.generation += 1
            logger.info(f"New champion! Generation {self.generation}, win rate: {win_rate:.1%}")
            return True
        logger.info(f"Champion retained. Win rate: {win_rate:.1%} < {self.config.win_rate_threshold:.1%}")
        return False

    def _play_eval_game(self, use_champion: bool, opponent: str) -> float:
        """Play a single evaluation game. Returns reward for the network player."""
        net = self.champion_network if use_champion else self.network
        net.eval()

        mcts = MCTS(
            game=self.game,
            network=net,
            device=self.device,
            num_simulations=max(self.config.num_simulations // 4, 10),
            c_puct=self.config.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            temperature=0.1,
        )

        if opponent == "champion":
            opp_mcts = MCTS(
                game=self.game,
                network=self.champion_network,
                device=self.device,
                num_simulations=max(self.config.num_simulations // 4, 10),
                c_puct=self.config.c_puct,
                dirichlet_alpha=0.0,
                dirichlet_epsilon=0.0,
                temperature=0.1,
            )

        state = self.game.get_initial_state()
        network_player = 0  # network plays as player 0

        for _ in range(self.config.max_game_length):
            if self.game.is_terminal(state):
                break

            if self.game.is_chance_node(state):
                outcomes = self.game.get_chance_outcomes(state)
                rolls = [o for o, _ in outcomes]
                probs = [p for _, p in outcomes]
                idx = np.random.choice(len(rolls), p=probs)
                state = self.game.apply_chance_outcome(state, rolls[idx])
                continue

            legal = self.game.get_legal_actions(state)
            if not legal:
                state.dice_remaining = []
                state.current_player = 1 - state.current_player
                state.needs_dice = True
                continue

            current_player = self.game.get_current_player(state)

            if current_player == network_player:
                policy, _ = mcts.search(state)
                action_idx = np.argmax(policy)
                action = self.game.index_to_action(action_idx)
            elif opponent == "champion":
                policy, _ = opp_mcts.search(state)
                action_idx = np.argmax(policy)
                action = self.game.index_to_action(action_idx)
            else:
                # Random opponent
                action = random.choice(legal)

            if action not in legal:
                action = random.choice(legal)

            state = self.game.apply_action(state, action)

        if self.game.is_terminal(state):
            return self.game.get_reward(state, network_player)
        return 0.0

    def save_checkpoint(self, path: Optional[str] = None):
        """Save full training state to disk."""
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.config.checkpoint_dir, f"checkpoint_{self.training_step:06d}.pt"
            )

        checkpoint = {
            "network_state": self.network.state_dict(),
            "champion_state": self.champion_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "generation": self.generation,
            "replay_buffer": list(self.replay_buffer.buffer),
            "config": self.config.__dict__,
            "training_history": self.training_history[-1000:],  # keep last 1000 entries
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        # Also save a "latest" symlink/copy for easy resume
        latest_path = os.path.join(self.config.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint["network_state"])
        self.champion_network.load_state_dict(checkpoint["champion_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.training_step = checkpoint["training_step"]
        self.generation = checkpoint["generation"]

        # Restore replay buffer
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        self.replay_buffer.buffer = deque(checkpoint["replay_buffer"], maxlen=self.config.replay_buffer_size)

        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]

        logger.info(
            f"Checkpoint loaded: step={self.training_step}, "
            f"generation={self.generation}, "
            f"buffer={len(self.replay_buffer)}"
        )

    def run_training_loop(self):
        """Full training pipeline: self-play → train → evaluate → repeat."""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Config: {self.config}")

        while True:
            # Phase 1: Generate self-play data
            logger.info(f"--- Generation {self.generation} ---")
            logger.info("Generating self-play data...")
            num_examples = self.generate_self_play_data(self.config.num_self_play_games)
            logger.info(f"Generated {num_examples} examples, buffer size: {len(self.replay_buffer)}")

            if len(self.replay_buffer) < self.config.batch_size:
                logger.warning("Not enough data, running more self-play...")
                continue

            # Phase 2: Train
            logger.info("Training...")
            history = self.train(self.config.num_training_steps)
            if history and "error" not in history[-1]:
                last = history[-1]
                logger.info(
                    f"Training complete. Final loss: {last['total_loss']:.4f}"
                )

            # Phase 3: Evaluate and maybe update champion
            logger.info("Evaluating...")
            updated = self.maybe_update_champion()
            if updated:
                self.save_checkpoint()

            # Phase 4: Quick eval vs random
            if self.training_step % self.config.eval_vs_random_interval == 0:
                self.evaluate_vs_random()

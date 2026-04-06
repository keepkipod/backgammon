"""Neural network for AlphaZero-style game AI.

ResNet architecture with two heads:
  - Policy head: probability distribution over actions
  - Value head: scalar game outcome prediction [-1, +1]

Designed to be game-agnostic — input/output sizes are configurable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    """Single residual block: two linear layers with batch norm and skip connection."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = F.relu(out + residual)
        return out


class DualHeadNetwork(nn.Module):
    """ResNet with policy and value heads.

    Args:
        input_size: Dimension of the state encoding vector.
        action_size: Number of possible actions (policy head output size).
        hidden_size: Width of residual blocks.
        num_res_blocks: Number of residual blocks.
    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_size: int = 128,
        num_res_blocks: int = 5,
    ):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size

        # Input projection
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_fc1 = nn.Linear(hidden_size, hidden_size)
        self.policy_bn = nn.BatchNorm1d(hidden_size)
        self.policy_fc2 = nn.Linear(hidden_size, action_size)

        # Value head
        self.value_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_bn = nn.BatchNorm1d(hidden_size // 2)
        self.value_fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Batch of encoded states, shape (batch_size, input_size).

        Returns:
            policy_logits: Raw logits, shape (batch_size, action_size).
                           Caller applies masking + softmax.
            value: Predicted game value, shape (batch_size, 1), range [-1, 1].
        """
        # Shared trunk
        x = F.relu(self.input_bn(self.input_fc(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_fc1(x)))
        policy_logits = self.policy_fc2(p)

        # Value head
        v = F.relu(self.value_bn(self.value_fc1(x)))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def predict(
        self, state_tensor: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Single-state inference with legal move masking.

        Args:
            state_tensor: Encoded state, shape (input_size,).
            legal_mask: Binary mask, shape (action_size,). 1 = legal.

        Returns:
            policy: Probability distribution over legal actions, shape (action_size,).
            value: Scalar value prediction.
        """
        self.eval()
        with torch.no_grad():
            x = state_tensor.unsqueeze(0)  # add batch dim
            logits, value = self(x)
            logits = logits.squeeze(0)
            value = value.item()

            # Mask illegal actions and compute softmax
            logits[legal_mask == 0] = float("-inf")
            policy = F.softmax(logits, dim=0)

        return policy, value

    def predict_batch(
        self, state_tensors: torch.Tensor, legal_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch inference with legal move masking.

        Args:
            state_tensors: shape (batch, input_size).
            legal_masks: shape (batch, action_size). 1 = legal.

        Returns:
            policies: shape (batch, action_size).
            values: shape (batch,).
        """
        self.eval()
        with torch.no_grad():
            logits, values = self(state_tensors)
            values = values.squeeze(-1)

            logits[legal_masks == 0] = float("-inf")
            policies = F.softmax(logits, dim=1)

        return policies, values

"""Stochastic Monte Carlo Tree Search for games with chance nodes.

Supports both deterministic and stochastic games via the GameState interface.
Uses a neural network for leaf evaluation instead of random rollouts.

Tree structure:
  - Decision nodes: player chooses an action (PUCT selection)
  - Chance nodes: environment samples an outcome (weighted by probability)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from backgammon.game_state import GameState
from backgammon.network import DualHeadNetwork


@dataclass
class DecisionNode:
    """Node where a player makes a decision."""

    state: Any
    parent: Optional[Any] = None  # DecisionNode or ChanceNode
    parent_action: Any = None
    children: dict = field(default_factory=dict)  # action -> ChanceNode or DecisionNode
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0


@dataclass
class ChanceNode:
    """Node where the environment produces a random outcome (e.g., dice roll)."""

    state: Any
    parent: DecisionNode = None
    children: dict = field(default_factory=dict)  # outcome -> DecisionNode
    visit_count: int = 0
    value_sum: float = 0.0
    outcome_probs: dict = field(default_factory=dict)  # outcome -> probability

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """Stochastic MCTS with neural network evaluation.

    Args:
        game: GameState implementation.
        network: Neural network for state evaluation.
        device: Torch device for inference.
        num_simulations: Number of MCTS simulations per move.
        c_puct: Exploration constant for PUCT formula.
        dirichlet_alpha: Alpha parameter for Dirichlet noise at root.
        dirichlet_epsilon: Weight of Dirichlet noise (0 = no noise).
        temperature: Controls exploration in action selection.
            1.0 = proportional to visit counts.
            0.0 = greedy (pick most visited).
    """

    def __init__(
        self,
        game: GameState,
        network: DualHeadNetwork,
        device: torch.device,
        num_simulations: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
    ):
        self.game = game
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature

    def search(self, state: Any) -> tuple[np.ndarray, float]:
        """Run MCTS from the given state.

        Uses batched neural network evaluation: collects multiple leaf nodes
        per iteration and evaluates them in a single forward pass.

        Returns:
            policy: Probability distribution over action space, shape (action_size,).
            value: Estimated value of the root state.
        """
        root = DecisionNode(state=state)
        self._expand_decision_node(root)

        if not root.children:
            policy = np.zeros(self.game.get_action_space_size(), dtype=np.float32)
            return policy, 0.0

        if self.dirichlet_epsilon > 0:
            self._add_dirichlet_noise(root)

        root_player = self.game.get_current_player(root.state)
        batch_size = min(8, max(1, self.num_simulations // 4))
        sims_done = 0

        while sims_done < self.num_simulations:
            current_batch = min(batch_size, self.num_simulations - sims_done)

            # Phase 1: SELECT — traverse tree to find leaves, apply virtual loss
            leaves = []  # (node, search_path, needs_nn)
            for _ in range(current_batch):
                node = root
                search_path = [node]

                while node.is_expanded and not self.game.is_terminal(node.state):
                    if self.game.is_chance_node(node.state):
                        chance = node.children.get("chance")
                        if chance is None:
                            break
                        node = self._select_chance_outcome(chance)
                        search_path.append(chance)
                        search_path.append(node)
                    else:
                        action, node = self._select_child(node)
                        search_path.append(node)

                # Apply virtual loss to encourage diverse paths
                for n in search_path:
                    n.visit_count += 1
                    if isinstance(n, DecisionNode):
                        n.value_sum -= 1  # pessimistic bias

                if self.game.is_terminal(node.state):
                    value = self.game.get_reward(node.state, root_player)
                    leaves.append((node, search_path, False, value))
                else:
                    leaves.append((node, search_path, True, None))

            # Phase 2: EXPAND & EVALUATE — batch all leaf nodes needing NN
            nn_leaves = [(i, node, path) for i, (node, path, needs_nn, _) in enumerate(leaves) if needs_nn]

            if nn_leaves:
                # Prepare batch tensors
                state_tensors = []
                legal_masks = []
                leaf_infos = []  # (leaf_index, node, legal_actions or None)

                for leaf_idx, node, path in nn_leaves:
                    s = node.state
                    enc = self.game.encode_state(s)
                    state_tensors.append(enc)

                    if self.game.is_chance_node(s):
                        # Chance node — just need value, no policy
                        legal_masks.append(np.zeros(self.game.get_action_space_size(), dtype=np.float32))
                        leaf_infos.append((leaf_idx, node, None))
                    else:
                        legal_actions = self.game.get_legal_actions(s)
                        mask = np.zeros(self.game.get_action_space_size(), dtype=np.float32)
                        if legal_actions:
                            for action in legal_actions:
                                mask[self.game.action_to_index(action)] = 1.0
                        legal_masks.append(mask)
                        leaf_infos.append((leaf_idx, node, legal_actions))

                # Single batched forward pass
                batch_states = torch.FloatTensor(np.array(state_tensors)).to(self.device)
                batch_masks = torch.FloatTensor(np.array(legal_masks)).to(self.device)
                policies_batch, values_batch = self.network.predict_batch(batch_states, batch_masks)
                policies_np = policies_batch.cpu().numpy()
                values_np = values_batch.cpu().numpy()

                # Expand each leaf
                for j, (leaf_idx, node, legal_actions) in enumerate(leaf_infos):
                    value = float(values_np[j])
                    policy = policies_np[j]

                    if legal_actions is None:
                        # Chance node
                        if not node.is_expanded:
                            chance = ChanceNode(state=node.state, parent=node)
                            outcomes = self.game.get_chance_outcomes(node.state)
                            chance.outcome_probs = {o: p for o, p in outcomes}
                            node.children["chance"] = chance
                    elif legal_actions:
                        if not node.is_expanded:
                            for action in legal_actions:
                                idx = self.game.action_to_index(action)
                                child_state = self.game.apply_action(node.state, action)
                                child = DecisionNode(
                                    state=child_state,
                                    parent=node,
                                    parent_action=action,
                                    prior=policy[idx],
                                )
                                node.children[action] = child

                    # Store value back
                    leaves[leaf_idx] = (leaves[leaf_idx][0], leaves[leaf_idx][1], False, value)

            # Phase 3: BACKPROPAGATE — undo virtual loss and apply real values
            for node, search_path, _, value in leaves:
                # Undo virtual loss
                for n in search_path:
                    n.visit_count -= 1
                    if isinstance(n, DecisionNode):
                        n.value_sum += 1
                # Real backprop
                self._backpropagate(search_path, value, root_player)

            sims_done += current_batch

        policy = self._get_policy(root)
        return policy, root.value

    def _expand_decision_node(self, node: DecisionNode) -> float:
        """Expand a decision node using the neural network.

        Returns the value estimate for the node's state.
        """
        state = node.state

        if self.game.is_terminal(state):
            return 0.0

        # If it's a chance node, create a chance child
        if self.game.is_chance_node(state):
            chance = ChanceNode(state=state, parent=node)
            outcomes = self.game.get_chance_outcomes(state)
            chance.outcome_probs = {o: p for o, p in outcomes}
            node.children["chance"] = chance
            # Evaluate current state
            return self._evaluate(state)

        # Get legal actions and network evaluation
        legal_actions = self.game.get_legal_actions(state)
        if not legal_actions:
            return self._evaluate(state)

        # Neural network inference
        state_tensor = torch.FloatTensor(self.game.encode_state(state)).to(self.device)
        legal_mask = torch.zeros(self.game.get_action_space_size(), device=self.device)
        for action in legal_actions:
            legal_mask[self.game.action_to_index(action)] = 1.0

        policy, value = self.network.predict(state_tensor, legal_mask)
        policy = policy.cpu().numpy()

        # Create children for each legal action
        for action in legal_actions:
            idx = self.game.action_to_index(action)
            child_state = self.game.apply_action(state, action)
            child = DecisionNode(
                state=child_state,
                parent=node,
                parent_action=action,
                prior=policy[idx],
            )
            node.children[action] = child

        return value

    def _evaluate(self, state: Any) -> float:
        """Get value estimate from neural network."""
        state_tensor = torch.FloatTensor(self.game.encode_state(state)).to(self.device)
        legal_mask = torch.zeros(self.game.get_action_space_size(), device=self.device)
        # We only need value here, not policy
        _, value = self.network.predict(state_tensor, legal_mask)
        return value

    def _select_child(self, node: DecisionNode) -> tuple[Any, DecisionNode]:
        """Select child using PUCT formula."""
        best_score = -float("inf")
        best_action = None
        best_child = None

        sqrt_parent = math.sqrt(node.visit_count)

        for action, child in node.children.items():
            if action == "chance":
                continue

            # PUCT score
            if child.visit_count > 0:
                q_value = child.value
            else:
                q_value = 0.0

            # Flip value if players alternate (opponent's value is negated)
            if self.game.get_current_player(child.state) != self.game.get_current_player(node.state):
                q_value = -q_value

            exploration = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _select_chance_outcome(self, chance: ChanceNode) -> DecisionNode:
        """Sample a chance outcome proportional to its probability."""
        outcomes = list(chance.outcome_probs.keys())
        probs = [chance.outcome_probs[o] for o in outcomes]
        idx = np.random.choice(len(outcomes), p=probs)
        outcome = outcomes[idx]

        if outcome not in chance.children:
            # Lazily expand chance outcome
            child_state = self.game.apply_chance_outcome(chance.state, outcome)
            child = DecisionNode(state=child_state, parent=chance)
            chance.children[outcome] = child

        return chance.children[outcome]

    def _backpropagate(
        self, search_path: list, value: float, root_player: int
    ):
        """Backpropagate value through the search path."""
        for node in search_path:
            node.visit_count += 1
            if isinstance(node, DecisionNode):
                # Value is from root_player's perspective
                if self.game.get_current_player(node.state) == root_player:
                    node.value_sum += value
                else:
                    node.value_sum -= value
            else:
                # Chance nodes just accumulate
                node.value_sum += value

    def _add_dirichlet_noise(self, root: DecisionNode):
        """Add Dirichlet noise to root priors for exploration."""
        actions = [a for a in root.children if a != "chance"]
        if not actions:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * n
            )

    def _get_policy(self, root: DecisionNode) -> np.ndarray:
        """Convert root visit counts to a policy distribution."""
        action_size = self.game.get_action_space_size()
        visits = np.zeros(action_size, dtype=np.float32)

        for action, child in root.children.items():
            if action == "chance":
                continue
            idx = self.game.action_to_index(action)
            visits[idx] = child.visit_count

        if visits.sum() == 0:
            return visits

        if self.temperature == 0:
            # Greedy: all weight on most visited
            policy = np.zeros_like(visits)
            policy[np.argmax(visits)] = 1.0
            return policy

        # Apply temperature
        visits_temp = visits ** (1.0 / self.temperature)
        total = visits_temp.sum()
        if total > 0:
            return visits_temp / total
        return visits

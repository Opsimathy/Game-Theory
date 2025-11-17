"""
Monte Carlo Tree Search (MCTS)

MCTS is a best-first search algorithm that uses random sampling to evaluate positions.
It balances exploration and exploitation using the UCT (Upper Confidence Bound for Trees) formula.

Algorithm:
1. Selection: Start at root, select child with highest UCB until reaching a leaf
2. Expansion: Add new child node to the tree
3. Simulation: Play out the game randomly from the new node
4. Backpropagation: Update statistics of all nodes in the path

UCB1 formula: UCB(node) = value(node) + C * sqrt(ln(parent_visits) / node_visits)

Success stories:
- AlphaGo (2016): First program to beat world champion at Go
- AlphaZero (2017): Mastered chess, shogi, and Go through self-play
- MuZero (2019): Extended to games without known rules

Reference:
- Kocsis & Szepesvári (2006). "Bandit based Monte-Carlo Planning"
- Browne et al. (2012). "A Survey of Monte Carlo Tree Search Methods"
- Silver et al. (2016). "Mastering the game of Go with deep neural networks"
"""

import numpy as np
import math
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class GameState:
    """
    Abstract game state interface.

    Implementations should provide:
    - get_legal_actions(): Return list of legal actions
    - take_action(action): Return new state after taking action
    - is_terminal(): Check if game is over
    - get_reward(player): Get reward for a player
    - get_current_player(): Get player whose turn it is
    """
    pass


class MCTSNode:
    """
    Node in the MCTS tree.

    Attributes:
        state: Game state at this node
        parent: Parent node
        action: Action that led to this node
        children: Child nodes
        visits: Number of times this node was visited
        value: Total value accumulated
        untried_actions: Actions not yet explored
    """

    def __init__(
        self,
        state: Any,
        parent: Optional['MCTSNode'] = None,
        action: Optional[Any] = None
    ):
        """
        Initialize MCTS node.

        Args:
            state: Game state
            parent: Parent node
            action: Action that led to this state
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self._get_untried_actions()

    def _get_untried_actions(self) -> List[Any]:
        """Get actions that haven't been tried yet."""
        # This should be implemented based on game interface
        # For now, return empty list
        return []

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        # Should be implemented based on game interface
        return False

    def best_child(self, exploration_weight: float = 1.41) -> 'MCTSNode':
        """
        Select best child using UCB1 formula.

        Args:
            exploration_weight: Exploration constant (typically sqrt(2))

        Returns:
            Child node with highest UCB value
        """
        def ucb_score(child: 'MCTSNode') -> float:
            if child.visits == 0:
                return float('inf')

            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )
            return exploitation + exploration

        return max(self.children, key=ucb_score)

    def expand(self) -> 'MCTSNode':
        """
        Expand tree by adding a child for an untried action.

        Returns:
            New child node
        """
        action = self.untried_actions.pop()
        # This should create new state based on game interface
        # For now, we'll create placeholder
        next_state = None  # Should be: self.state.take_action(action)
        child = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, reward: float) -> None:
        """
        Update node statistics.

        Args:
            reward: Reward from simulation
        """
        self.visits += 1
        self.value += reward

    def backpropagate(self, reward: float) -> None:
        """
        Backpropagate reward up the tree.

        Args:
            reward: Reward to propagate
        """
        self.update(reward)
        if self.parent is not None:
            # Flip reward for opponent
            self.parent.backpropagate(-reward)


class MCTSAgent:
    """
    MCTS agent for game playing.

    Uses UCT (Upper Confidence Bound for Trees) for action selection.
    """

    def __init__(
        self,
        simulations: int = 1000,
        exploration_weight: float = 1.41
    ):
        """
        Initialize MCTS agent.

        Args:
            simulations: Number of MCTS simulations per move
            exploration_weight: UCT exploration constant
        """
        self.simulations = simulations
        self.exploration_weight = exploration_weight

    def search(self, root_state: Any) -> Any:
        """
        Perform MCTS search to find best action.

        Args:
            root_state: Current game state

        Returns:
            Best action to take
        """
        root = MCTSNode(root_state)

        for _ in range(self.simulations):
            node = root

            # Selection: Descend tree using UCB
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)

            # Expansion: Add new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation: Play out randomly
            reward = self._simulate(node.state)

            # Backpropagation: Update all nodes in path
            node.backpropagate(reward)

        # Return action of best child (exploitation only)
        best_child = root.best_child(exploration_weight=0)
        return best_child.action

    def _simulate(self, state: Any) -> float:
        """
        Simulate a random playout from the given state.

        Args:
            state: Starting state

        Returns:
            Reward from terminal state
        """
        # This should be implemented based on game interface
        # Random playout until terminal state
        current_state = state

        # Placeholder: return random reward
        return np.random.uniform(-1, 1)

    def get_action_probabilities(self, root_state: Any) -> Tuple[List[Any], np.ndarray]:
        """
        Get action probabilities based on visit counts.

        Useful for AlphaZero-style training.

        Args:
            root_state: Current game state

        Returns:
            (actions, probabilities) based on visit counts
        """
        root = MCTSNode(root_state)

        # Run simulations
        for _ in range(self.simulations):
            node = root

            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)

            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            reward = self._simulate(node.state)
            node.backpropagate(reward)

        # Compute probabilities from visit counts
        actions = [child.action for child in root.children]
        visits = np.array([child.visits for child in root.children])

        if visits.sum() > 0:
            probabilities = visits / visits.sum()
        else:
            probabilities = np.ones(len(visits)) / len(visits)

        return actions, probabilities


# Example game implementation for Tic-Tac-Toe
class TicTacToeState:
    """Simple Tic-Tac-Toe implementation for MCTS demonstration."""

    def __init__(self, board: Optional[np.ndarray] = None, player: int = 1):
        """
        Initialize Tic-Tac-Toe state.

        Args:
            board: 3x3 board (1 for X, -1 for O, 0 for empty)
            player: Current player (1 or -1)
        """
        self.board = board if board is not None else np.zeros((3, 3))
        self.player = player

    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """Get list of legal moves."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def take_action(self, action: Tuple[int, int]) -> 'TicTacToeState':
        """Return new state after taking action."""
        new_board = self.board.copy()
        new_board[action] = self.player
        return TicTacToeState(new_board, -self.player)

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.get_winner() is not None or len(self.get_legal_actions()) == 0

    def get_winner(self) -> Optional[int]:
        """Get winner (1, -1, or None)."""
        # Check rows
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return int(self.board[i, 0])

        # Check columns
        for j in range(3):
            if abs(self.board[:, j].sum()) == 3:
                return int(self.board[0, j])

        # Check diagonals
        if abs(np.trace(self.board)) == 3:
            return int(self.board[0, 0])
        if abs(np.trace(np.fliplr(self.board))) == 3:
            return int(self.board[0, 2])

        return None

    def get_reward(self, player: int) -> float:
        """Get reward for a player."""
        winner = self.get_winner()
        if winner is None:
            return 0.0
        return 1.0 if winner == player else -1.0


if __name__ == "__main__":
    print("=== MCTS for Tic-Tac-Toe ===")
    print("This is a conceptual demonstration.")
    print("\nMCTS is used in:")
    print("- AlphaGo: Defeated world champion Lee Sedol (2016)")
    print("- AlphaZero: Superhuman play in chess, shogi, Go (2017)")
    print("- MuZero: Planning without knowing the rules (2019)")
    print("\nKey parameters:")
    print("- Simulations: More = stronger play, but slower")
    print("- Exploration weight: Balance exploration vs exploitation")
    print("- Common values: 1.0 - 2.0 (sqrt(2) ≈ 1.41 is theoretical optimum)")

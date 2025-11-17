"""
Minimax and Alpha-Beta Pruning

Minimax is a classic algorithm for perfect-information two-player zero-sum games.
It assumes both players play optimally and computes the best move by:
1. Maximizing player tries to maximize the outcome
2. Minimizing player tries to minimize it
3. Recursively evaluate game tree to find optimal moves

Alpha-Beta Pruning:
- Optimization that reduces the number of nodes evaluated
- Maintains bounds (alpha, beta) to prune branches that can't affect final decision
- Can reduce complexity from O(b^d) to O(b^(d/2)) with good move ordering

Applications:
- Chess engines (Deep Blue, Stockfish)
- Checkers (Chinook - solved the game)
- Othello/Reversi

Reference:
- Shannon, C. (1950). "Programming a Computer for Playing Chess"
- Knuth & Moore (1975). "An Analysis of Alpha-Beta Pruning"
- Russell & Norvig, "Artificial Intelligence: A Modern Approach", Chapter 5
"""

from typing import Tuple, Optional, Callable, Any
import math


def minimax(
    state: Any,
    depth: int,
    maximizing_player: bool,
    evaluate_fn: Callable[[Any], float],
    get_actions_fn: Callable[[Any], list],
    take_action_fn: Callable[[Any, Any], Any],
    is_terminal_fn: Callable[[Any], bool]
) -> Tuple[float, Optional[Any]]:
    """
    Minimax algorithm for perfect-information games.

    Args:
        state: Current game state
        depth: Maximum search depth
        maximizing_player: True if current player is maximizing
        evaluate_fn: Function to evaluate terminal/leaf states
        get_actions_fn: Function to get legal actions from state
        take_action_fn: Function to get next state given action
        is_terminal_fn: Function to check if state is terminal

    Returns:
        (value, best_action) tuple
    """
    # Base case: terminal state or depth limit
    if depth == 0 or is_terminal_fn(state):
        return evaluate_fn(state), None

    best_action = None

    if maximizing_player:
        max_eval = -math.inf

        for action in get_actions_fn(state):
            next_state = take_action_fn(state, action)
            eval_score, _ = minimax(
                next_state,
                depth - 1,
                False,
                evaluate_fn,
                get_actions_fn,
                take_action_fn,
                is_terminal_fn
            )

            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action

        return max_eval, best_action

    else:  # Minimizing player
        min_eval = math.inf

        for action in get_actions_fn(state):
            next_state = take_action_fn(state, action)
            eval_score, _ = minimax(
                next_state,
                depth - 1,
                True,
                evaluate_fn,
                get_actions_fn,
                take_action_fn,
                is_terminal_fn
            )

            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action

        return min_eval, best_action


def alpha_beta(
    state: Any,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    evaluate_fn: Callable[[Any], float],
    get_actions_fn: Callable[[Any], list],
    take_action_fn: Callable[[Any, Any], Any],
    is_terminal_fn: Callable[[Any], bool]
) -> Tuple[float, Optional[Any]]:
    """
    Minimax with alpha-beta pruning.

    Alpha-beta pruning eliminates branches that cannot affect the final decision.

    Args:
        state: Current game state
        depth: Maximum search depth
        alpha: Best value maximizer can guarantee (lower bound)
        beta: Best value minimizer can guarantee (upper bound)
        maximizing_player: True if current player is maximizing
        evaluate_fn: Function to evaluate terminal/leaf states
        get_actions_fn: Function to get legal actions from state
        take_action_fn: Function to get next state given action
        is_terminal_fn: Function to check if state is terminal

    Returns:
        (value, best_action) tuple
    """
    # Base case
    if depth == 0 or is_terminal_fn(state):
        return evaluate_fn(state), None

    best_action = None

    if maximizing_player:
        max_eval = -math.inf

        for action in get_actions_fn(state):
            next_state = take_action_fn(state, action)
            eval_score, _ = alpha_beta(
                next_state,
                depth - 1,
                alpha,
                beta,
                False,
                evaluate_fn,
                get_actions_fn,
                take_action_fn,
                is_terminal_fn
            )

            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action

            alpha = max(alpha, eval_score)

            # Beta cutoff
            if beta <= alpha:
                break

        return max_eval, best_action

    else:  # Minimizing player
        min_eval = math.inf

        for action in get_actions_fn(state):
            next_state = take_action_fn(state, action)
            eval_score, _ = alpha_beta(
                next_state,
                depth - 1,
                alpha,
                beta,
                True,
                evaluate_fn,
                get_actions_fn,
                take_action_fn,
                is_terminal_fn
            )

            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action

            beta = min(beta, eval_score)

            # Alpha cutoff
            if beta <= alpha:
                break

        return min_eval, best_action


# Example: Tic-Tac-Toe with minimax
class TicTacToe:
    """Simple Tic-Tac-Toe for demonstrating minimax."""

    def __init__(self):
        self.board = [[0 for _ in range(3)] for _ in range(3)]
        self.current_player = 1  # 1 or -1

    def get_legal_actions(self):
        """Get legal moves."""
        return [(i, j) for i in range(3) for j in range(3)
                if self.board[i][j] == 0]

    def make_move(self, action):
        """Make a move."""
        i, j = action
        self.board[i][j] = self.current_player
        self.current_player = -self.current_player

    def is_terminal(self):
        """Check if game is over."""
        return self.get_winner() is not None or len(self.get_legal_actions()) == 0

    def get_winner(self):
        """Get winner (1, -1, or 0 for draw)."""
        # Check rows and columns
        for i in range(3):
            if abs(sum(self.board[i])) == 3:
                return self.board[i][0]
            if abs(sum(self.board[j][i] for j in range(3))) == 3:
                return self.board[0][i]

        # Check diagonals
        if abs(sum(self.board[i][i] for i in range(3))) == 3:
            return self.board[0][0]
        if abs(sum(self.board[i][2-i] for i in range(3))) == 3:
            return self.board[0][2]

        # Check for draw
        if len(self.get_legal_actions()) == 0:
            return 0

        return None

    def evaluate(self):
        """Evaluate current position."""
        winner = self.get_winner()
        if winner is None:
            return 0
        return winner

    def copy(self):
        """Create a copy of the game state."""
        new_game = TicTacToe()
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        return new_game

    def print_board(self):
        """Print the board."""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()


if __name__ == "__main__":
    print("=== Minimax for Tic-Tac-Toe ===")
    print("Demonstrating minimax with alpha-beta pruning\n")

    game = TicTacToe()

    # Helper functions for minimax
    def evaluate_fn(g): return g.evaluate()
    def get_actions_fn(g): return g.get_legal_actions()
    def is_terminal_fn(g): return g.is_terminal()

    def take_action_fn(g, action):
        g_copy = g.copy()
        g_copy.make_move(action)
        return g_copy

    print("Initial board:")
    game.print_board()

    # Computer plays first move
    value, action = alpha_beta(
        game, depth=9, alpha=-math.inf, beta=math.inf,
        maximizing_player=True,
        evaluate_fn=evaluate_fn,
        get_actions_fn=get_actions_fn,
        take_action_fn=take_action_fn,
        is_terminal_fn=is_terminal_fn
    )

    print(f"Best move: {action}, Expected value: {value}")
    print("\nWith optimal play from both sides:")
    print("- Tic-Tac-Toe always results in a draw")
    print("- This was proven by exhaustive search")
    print("\nAlpha-beta pruning benefits:")
    print("- Reduces search space significantly")
    print("- With good move ordering, can search ~2x deeper")
    print("- Critical for games like chess (b ≈ 35, typical depth 8-12)")

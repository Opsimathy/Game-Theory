"""
Linear Programming Solver for Zero-Sum Games

For two-player zero-sum games, Nash equilibria can be computed efficiently
using linear programming. This is based on von Neumann's minimax theorem.

The row player (maximizer) solves:
    max v
    s.t. sum_i p_i * A[i,j] >= v  for all j
         sum_i p_i = 1
         p_i >= 0 for all i

The column player (minimizer) solves the dual problem.

Complexity: Polynomial time (LP is solvable in polynomial time)

Reference:
- von Neumann, J. (1928). "Zur Theorie der Gesellschaftsspiele"
- Dantzig, G. B. (1951). "A Proof of the Equivalence of the Programming Problem and the Game Problem"
- Shoham & Leyton-Brown (2009), Section 3.5.3
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.normal_form import NormalFormGame


class LinearProgrammingSolver:
    """
    Solver for computing Nash equilibria in two-player zero-sum games.

    Uses linear programming to find the minimax equilibrium.
    """

    def __init__(self, game: NormalFormGame, tolerance: float = 1e-6):
        """
        Initialize the LP solver.

        Args:
            game: A two-player zero-sum game
            tolerance: Numerical tolerance
        """
        if game.n_players != 2:
            raise ValueError("LP solver only works for two-player games")

        # Check if game is zero-sum
        test_profile = tuple([0] * game.n_players)
        payoffs = game.get_payoff(test_profile)
        if abs(sum(payoffs)) > tolerance:
            print("Warning: Game may not be zero-sum")

        self.game = game
        self.tolerance = tolerance

    def solve(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve for the Nash equilibrium using linear programming.

        Returns:
            (strategy_p1, strategy_p2, value) where value is the game value
        """
        payoff_matrix = self.game.payoffs[0]
        m, n = payoff_matrix.shape

        # Solve for player 1 (row player, maximizer)
        # Variables: [p_1, ..., p_m, v]
        # Objective: max v  =>  min -v
        c = np.zeros(m + 1)
        c[-1] = -1  # Minimize -v

        # Inequality constraints: -sum_i p_i * A[i,j] + v <= 0 for all j
        # i.e., sum_i p_i * A[i,j] >= v
        A_ub = np.zeros((n, m + 1))
        for j in range(n):
            A_ub[j, :m] = -payoff_matrix[:, j]
            A_ub[j, m] = 1  # coefficient of v

        b_ub = np.zeros(n)

        # Equality constraint: sum_i p_i = 1
        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1
        b_eq = np.array([1])

        # Bounds: p_i >= 0, v unbounded
        bounds = [(0, None) for _ in range(m)] + [(None, None)]

        # Solve
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs'
        )

        if not result.success:
            raise ValueError(f"LP solver failed: {result.message}")

        strategy_p1 = result.x[:m]
        value = result.x[m]

        # Solve for player 2 (column player, minimizer)
        # Variables: [q_1, ..., q_n, u]
        # Objective: min u
        c = np.zeros(n + 1)
        c[-1] = 1  # Minimize u

        # Inequality constraints: sum_j q_j * A[i,j] - u <= 0 for all i
        # i.e., sum_j q_j * A[i,j] <= u
        A_ub = np.zeros((m, n + 1))
        for i in range(m):
            A_ub[i, :n] = payoff_matrix[i, :]
            A_ub[i, n] = -1  # coefficient of u

        b_ub = np.zeros(m)

        # Equality constraint: sum_j q_j = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1
        b_eq = np.array([1])

        # Bounds: q_j >= 0, u unbounded
        bounds = [(0, None) for _ in range(n)] + [(None, None)]

        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs'
        )

        if not result.success:
            raise ValueError(f"LP solver failed: {result.message}")

        strategy_p2 = result.x[:n]
        value_p2 = result.x[n]

        # Verify that values match (minimax theorem)
        if abs(value - value_p2) > self.tolerance:
            print(f"Warning: Game values don't match: {value} vs {value_p2}")

        # Clean up numerical errors
        strategy_p1 = np.maximum(strategy_p1, 0)
        strategy_p2 = np.maximum(strategy_p2, 0)
        strategy_p1 /= strategy_p1.sum()
        strategy_p2 /= strategy_p2.sum()

        return strategy_p1, strategy_p2, value


if __name__ == "__main__":
    print("=== Rock-Paper-Scissors ===")
    game = NormalFormGame.rock_paper_scissors()
    solver = LinearProgrammingSolver(game)
    s1, s2, value = solver.solve()
    print(f"Nash Equilibrium:")
    print(f"  Player 1: {s1}")
    print(f"  Player 2: {s2}")
    print(f"  Game value: {value}")

    print("\n=== Matching Pennies ===")
    game = NormalFormGame.matching_pennies()
    solver = LinearProgrammingSolver(game)
    s1, s2, value = solver.solve()
    print(f"Nash Equilibrium:")
    print(f"  Player 1: {s1}")
    print(f"  Player 2: {s2}")
    print(f"  Game value: {value}")

    print("\n=== Custom Zero-Sum Game ===")
    # A simple 2x2 zero-sum game
    payoff_p1 = np.array([[3, -1], [-2, 4]])
    payoff_p2 = -payoff_p1
    game = NormalFormGame(payoff_p1, payoff_p2)
    solver = LinearProgrammingSolver(game)
    s1, s2, value = solver.solve()
    print(f"Nash Equilibrium:")
    print(f"  Player 1: {s1}")
    print(f"  Player 2: {s2}")
    print(f"  Game value: {value}")

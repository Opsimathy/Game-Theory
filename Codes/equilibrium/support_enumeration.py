"""
Support Enumeration Algorithm for Computing Nash Equilibria

The support enumeration algorithm finds all Nash equilibria by:
1. Enumerating all possible support sets (subsets of actions)
2. For each support pair, solving for a completely mixed equilibrium
3. Checking if the solution satisfies equilibrium conditions

Complexity: O(2^m × 2^n) for an m×n game
Practical only for small games (< 10 actions per player)

Reference:
- Porter, Nudelman, & Shoham (2004). "Simple search methods for finding a Nash equilibrium"
- Shoham & Leyton-Brown (2009), "Multiagent Systems", Section 3.4.4
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.normal_form import NormalFormGame


class SupportEnumerationSolver:
    """
    Solver for finding all Nash equilibria using support enumeration.

    Works for two-player games by enumerating all possible support combinations.
    """

    def __init__(self, game: NormalFormGame, tolerance: float = 1e-6):
        """
        Initialize the support enumeration solver.

        Args:
            game: A two-player normal-form game
            tolerance: Numerical tolerance for comparisons
        """
        if game.n_players != 2:
            raise ValueError("Support enumeration only works for two-player games")

        self.game = game
        self.tolerance = tolerance

    def solve(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Find all Nash equilibria using support enumeration.

        Returns:
            List of Nash equilibria, each as (strategy_p1, strategy_p2)
        """
        equilibria = []

        n_actions_p1, n_actions_p2 = self.game.n_actions

        # Enumerate all possible support sizes
        for support_size_p1 in range(1, n_actions_p1 + 1):
            for support_size_p2 in range(1, n_actions_p2 + 1):
                # Enumerate all supports of this size
                for support_p1 in combinations(range(n_actions_p1), support_size_p1):
                    for support_p2 in combinations(range(n_actions_p2), support_size_p2):
                        # Try to find equilibrium with this support
                        eq = self._find_equilibrium_with_support(
                            list(support_p1), list(support_p2)
                        )
                        if eq is not None:
                            # Check if we already found this equilibrium
                            is_duplicate = False
                            for existing_eq in equilibria:
                                if (np.allclose(eq[0], existing_eq[0], atol=self.tolerance) and
                                    np.allclose(eq[1], existing_eq[1], atol=self.tolerance)):
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                equilibria.append(eq)

        return equilibria

    def _find_equilibrium_with_support(
        self,
        support_p1: List[int],
        support_p2: List[int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Try to find a Nash equilibrium with given supports.

        For an equilibrium to exist with these supports:
        1. All actions in support must give equal (best) payoff
        2. All actions outside support must give weakly lower payoff
        3. Probabilities must be non-negative and sum to 1

        Args:
            support_p1: Support set for player 1
            support_p2: Support set for player 2

        Returns:
            Nash equilibrium (strategy_p1, strategy_p2) or None
        """
        payoff_p1 = self.game.payoffs[0]
        payoff_p2 = self.game.payoffs[1]

        n_actions_p1, n_actions_p2 = self.game.n_actions

        # Build system of equations for player 2's strategy
        # All actions in player 1's support must give equal payoff
        if len(support_p1) > 1:
            # Equations: payoff from action i = payoff from action j for all i, j in support
            A_p2 = []
            b_p2 = []

            base_action = support_p1[0]
            for action in support_p1[1:]:
                # payoff_p1[base_action, :] @ strategy_p2 = payoff_p1[action, :] @ strategy_p2
                eq = payoff_p1[base_action, :] - payoff_p1[action, :]
                A_p2.append(eq)
                b_p2.append(0)

            # Add constraint: probabilities sum to 1
            A_p2.append(np.ones(n_actions_p2))
            b_p2.append(1)

            A_p2 = np.array(A_p2)
            b_p2 = np.array(b_p2)

            # Solve for player 2's strategy
            try:
                strategy_p2_full, residuals, rank, s = np.linalg.lstsq(A_p2, b_p2, rcond=None)

                # Check if solution is valid
                if rank < len(support_p1):  # System is underdetermined
                    return None

                # Check residuals
                if len(residuals) > 0 and residuals[0] > self.tolerance:
                    return None

            except np.linalg.LinAlgError:
                return None
        else:
            # Pure strategy for player 1, player 2 must best-respond
            strategy_p2_full = np.zeros(n_actions_p2)
            # Find best responses for player 2
            action_p1 = support_p1[0]
            best_payoff = payoff_p2[action_p1, :].max()
            best_actions = np.where(payoff_p2[action_p1, :] >= best_payoff - self.tolerance)[0]

            # Player 2 can play any mixture over best responses
            # For simplicity, we'll use uniform over best responses
            strategy_p2_full[best_actions] = 1.0 / len(best_actions)

        # Similarly for player 1's strategy
        if len(support_p2) > 1:
            A_p1 = []
            b_p1 = []

            base_action = support_p2[0]
            for action in support_p2[1:]:
                eq = payoff_p2[:, base_action] - payoff_p2[:, action]
                A_p1.append(eq)
                b_p1.append(0)

            A_p1.append(np.ones(n_actions_p1))
            b_p1.append(1)

            A_p1 = np.array(A_p1)
            b_p1 = np.array(b_p1)

            try:
                strategy_p1_full, residuals, rank, s = np.linalg.lstsq(A_p1, b_p1, rcond=None)

                if rank < len(support_p2):
                    return None

                if len(residuals) > 0 and residuals[0] > self.tolerance:
                    return None

            except np.linalg.LinAlgError:
                return None
        else:
            strategy_p1_full = np.zeros(n_actions_p1)
            action_p2 = support_p2[0]
            best_payoff = payoff_p1[:, action_p2].max()
            best_actions = np.where(payoff_p1[:, action_p2] >= best_payoff - self.tolerance)[0]
            strategy_p1_full[best_actions] = 1.0 / len(best_actions)

        # Verify that strategies are valid probability distributions
        if (np.any(strategy_p1_full < -self.tolerance) or
            np.any(strategy_p2_full < -self.tolerance)):
            return None

        if (abs(strategy_p1_full.sum() - 1.0) > self.tolerance or
            abs(strategy_p2_full.sum() - 1.0) > self.tolerance):
            return None

        # Verify that support matches
        support_p1_actual = np.where(strategy_p1_full > self.tolerance)[0]
        support_p2_actual = np.where(strategy_p2_full > self.tolerance)[0]

        if (not set(support_p1).issubset(set(support_p1_actual)) or
            not set(support_p2).issubset(set(support_p2_actual))):
            return None

        # Verify Nash equilibrium conditions
        if not self.game.is_nash_equilibrium([strategy_p1_full, strategy_p2_full], self.tolerance):
            return None

        # Clean up numerical errors
        strategy_p1_full = np.maximum(strategy_p1_full, 0)
        strategy_p2_full = np.maximum(strategy_p2_full, 0)
        strategy_p1_full /= strategy_p1_full.sum()
        strategy_p2_full /= strategy_p2_full.sum()

        return (strategy_p1_full, strategy_p2_full)


if __name__ == "__main__":
    # Example: Find Nash equilibria for classic games
    print("=== Matching Pennies ===")
    game = NormalFormGame.matching_pennies()
    solver = SupportEnumerationSolver(game)
    equilibria = solver.solve()
    print(f"Found {len(equilibria)} equilibria:")
    for i, (s1, s2) in enumerate(equilibria):
        print(f"Equilibrium {i+1}: P1={s1}, P2={s2}")

    print("\n=== Rock-Paper-Scissors ===")
    game = NormalFormGame.rock_paper_scissors()
    solver = SupportEnumerationSolver(game)
    equilibria = solver.solve()
    print(f"Found {len(equilibria)} equilibria:")
    for i, (s1, s2) in enumerate(equilibria):
        print(f"Equilibrium {i+1}: P1={s1}, P2={s2}")

    print("\n=== Battle of the Sexes ===")
    game = NormalFormGame.battle_of_sexes()
    solver = SupportEnumerationSolver(game)
    equilibria = solver.solve()
    print(f"Found {len(equilibria)} equilibria:")
    for i, (s1, s2) in enumerate(equilibria):
        print(f"Equilibrium {i+1}: P1={s1}, P2={s2}")

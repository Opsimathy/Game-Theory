"""
Correlated Equilibrium Computation

A correlated equilibrium is a generalization of Nash equilibrium where players can
coordinate using a mediator (correlation device) that recommends actions.

Key Properties:
- Always exists (even in finite games)
- Can achieve higher social welfare than Nash equilibrium
- Can be computed efficiently using linear programming
- Set of correlated equilibria contains all Nash equilibria

Mathematical Definition:
A probability distribution π over action profiles is a correlated equilibrium if
for each player i and each pair of actions (a_i, a'_i):

  Σ π(a) * u_i(a) ≥ Σ π(a_{-i}, a_i) * u_i(a_{-i}, a'_i)
  a: a_i=a_i           a_{-i}

This says: following the recommendation is better than deviating.

Reference:
- Aumann, R. (1974). "Subjectivity and Correlation in Randomized Strategies"
- Aumann, R. (1987). "Correlated Equilibrium as an Expression of Bayesian Rationality"
- Nisan et al. (2007). "Algorithmic Game Theory", Chapter 1
- Papadimitriou & Roughgarden (2008). "Computing Correlated Equilibria in Multi-Player Games"
"""

import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple, Optional, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.normal_form import NormalFormGame


class CorrelatedEquilibriumSolver:
    """
    Solver for computing correlated equilibria using linear programming.

    Can compute:
    - Any correlated equilibrium (feasibility LP)
    - Social welfare maximizing correlated equilibrium
    - Uniform correlated equilibrium
    """

    def __init__(self, game: NormalFormGame, tolerance: float = 1e-8):
        """
        Initialize correlated equilibrium solver.

        Args:
            game: Two-player normal-form game
            tolerance: Numerical tolerance
        """
        if game.n_players != 2:
            raise ValueError("Current implementation supports 2-player games")

        self.game = game
        self.tolerance = tolerance

    def solve_welfare_maximizing(self) -> np.ndarray:
        """
        Find the correlated equilibrium that maximizes social welfare.

        Returns:
            Probability distribution over action profiles (matrix)
        """
        n1, n2 = self.game.n_actions
        n_outcomes = n1 * n2

        # Variables: π(a1, a2) for each action profile
        # Objective: maximize sum of payoffs
        payoff_p1 = self.game.payoffs[0].flatten()
        payoff_p2 = self.game.payoffs[1].flatten()
        social_welfare = payoff_p1 + payoff_p2

        # Maximize social welfare => minimize negative social welfare
        c = -social_welfare

        # Incentive constraints for player 1
        # For each action a1 and deviation a'1:
        # Σ_{a2} π(a1,a2)[u1(a1,a2) - u1(a'1,a2)] ≥ 0
        A_ub = []
        b_ub = []

        for a1 in range(n1):
            for a1_prime in range(n1):
                if a1 == a1_prime:
                    continue

                constraint = np.zeros(n_outcomes)
                for a2 in range(n2):
                    idx = a1 * n2 + a2
                    idx_prime = a1_prime * n2 + a2

                    # u1(a1,a2) - u1(a'1,a2) when recommended a1
                    payoff_follow = self.game.payoffs[0][a1, a2]
                    payoff_deviate = self.game.payoffs[0][a1_prime, a2]

                    # Constraint: -[π(a1,a2) * (payoff_follow - payoff_deviate)] ≤ 0
                    constraint[idx] = -(payoff_follow - payoff_deviate)

                A_ub.append(constraint)
                b_ub.append(0)

        # Incentive constraints for player 2
        for a2 in range(n2):
            for a2_prime in range(n2):
                if a2 == a2_prime:
                    continue

                constraint = np.zeros(n_outcomes)
                for a1 in range(n1):
                    idx = a1 * n2 + a2
                    idx_prime = a1 * n2 + a2_prime

                    payoff_follow = self.game.payoffs[1][a1, a2]
                    payoff_deviate = self.game.payoffs[1][a1, a2_prime]

                    constraint[idx] = -(payoff_follow - payoff_deviate)

                A_ub.append(constraint)
                b_ub.append(0)

        A_ub = np.array(A_ub) if A_ub else np.zeros((0, n_outcomes))
        b_ub = np.array(b_ub) if b_ub else np.zeros(0)

        # Equality constraint: probabilities sum to 1
        A_eq = np.ones((1, n_outcomes))
        b_eq = np.array([1.0])

        # Bounds: probabilities are non-negative
        bounds = [(0, None) for _ in range(n_outcomes)]

        # Solve LP
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs'
        )

        if not result.success:
            raise ValueError(f"LP solver failed: {result.message}")

        # Reshape to matrix form
        distribution = result.x.reshape(n1, n2)

        # Clean up numerical errors
        distribution = np.maximum(distribution, 0)
        distribution /= distribution.sum()

        return distribution

    def verify_correlated_equilibrium(self, distribution: np.ndarray) -> bool:
        """
        Verify that a distribution is a correlated equilibrium.

        Args:
            distribution: Probability distribution over action profiles

        Returns:
            True if distribution is a correlated equilibrium
        """
        n1, n2 = self.game.n_actions

        # Check player 1 incentive constraints
        for a1 in range(n1):
            for a1_prime in range(n1):
                if a1 == a1_prime:
                    continue

                # Expected payoff from following recommendation a1
                payoff_follow = 0
                # Expected payoff from deviating to a1_prime when recommended a1
                payoff_deviate = 0

                for a2 in range(n2):
                    prob = distribution[a1, a2]
                    payoff_follow += prob * self.game.payoffs[0][a1, a2]
                    payoff_deviate += prob * self.game.payoffs[0][a1_prime, a2]

                if payoff_deviate > payoff_follow + self.tolerance:
                    return False

        # Check player 2 incentive constraints
        for a2 in range(n2):
            for a2_prime in range(n2):
                if a2 == a2_prime:
                    continue

                payoff_follow = 0
                payoff_deviate = 0

                for a1 in range(n1):
                    prob = distribution[a1, a2]
                    payoff_follow += prob * self.game.payoffs[1][a1, a2]
                    payoff_deviate += prob * self.game.payoffs[1][a1, a2_prime]

                if payoff_deviate > payoff_follow + self.tolerance:
                    return False

        return True

    def get_expected_payoffs(self, distribution: np.ndarray) -> Tuple[float, float]:
        """
        Compute expected payoffs under a correlated equilibrium.

        Args:
            distribution: Probability distribution over action profiles

        Returns:
            Expected payoffs for each player
        """
        payoff_p1 = np.sum(distribution * self.game.payoffs[0])
        payoff_p2 = np.sum(distribution * self.game.payoffs[1])
        return payoff_p1, payoff_p2


def compare_nash_vs_correlated(game: NormalFormGame):
    """
    Compare Nash equilibrium and correlated equilibrium welfare.

    Args:
        game: Game to analyze
    """
    from equilibrium.support_enumeration import SupportEnumerationSolver

    print("Computing Nash equilibria...")
    nash_solver = SupportEnumerationSolver(game)
    nash_equilibria = nash_solver.solve()

    print(f"Found {len(nash_equilibria)} Nash equilibria\n")

    best_nash_welfare = -float('inf')
    for i, (s1, s2) in enumerate(nash_equilibria):
        payoffs = game.get_expected_payoff([s1, s2])
        welfare = sum(payoffs)
        print(f"Nash Equilibrium {i+1}:")
        print(f"  Strategies: P1={s1}, P2={s2}")
        print(f"  Payoffs: {payoffs}")
        print(f"  Social Welfare: {welfare:.4f}\n")
        best_nash_welfare = max(best_nash_welfare, welfare)

    print("Computing welfare-maximizing correlated equilibrium...")
    ce_solver = CorrelatedEquilibriumSolver(game)
    ce_distribution = ce_solver.solve_welfare_maximizing()

    print("Correlated Equilibrium:")
    print(f"  Distribution:\n{ce_distribution}\n")

    is_ce = ce_solver.verify_correlated_equilibrium(ce_distribution)
    print(f"  Is valid CE: {is_ce}")

    ce_payoffs = ce_solver.get_expected_payoffs(ce_distribution)
    ce_welfare = sum(ce_payoffs)
    print(f"  Expected payoffs: {ce_payoffs}")
    print(f"  Social Welfare: {ce_welfare:.4f}\n")

    print(f"Welfare improvement: {ce_welfare - best_nash_welfare:.4f}")
    print(f"  ({(ce_welfare / best_nash_welfare - 1) * 100:.2f}% better than best Nash)")


if __name__ == "__main__":
    print("="*60)
    print("CORRELATED EQUILIBRIUM EXAMPLES")
    print("="*60)
    print()

    # Example 1: Chicken Game (benefits from correlation)
    print("Example 1: Chicken Game")
    print("-"*60)
    print("Two drivers head toward each other. Each can swerve or stay.")
    print("If both stay, they crash (worst outcome)")
    print("If one swerves, the other gets prestige")
    print()

    # Payoff matrix:
    #           Swerve  Stay
    # Swerve    (0,0)   (-1,1)
    # Stay      (1,-1)  (-10,-10)

    payoff_p1 = np.array([[0, -1], [1, -10]])
    payoff_p2 = np.array([[0, 1], [-1, -10]])
    chicken_game = NormalFormGame(payoff_p1, payoff_p2)

    compare_nash_vs_correlated(chicken_game)

    print("\n" + "="*60)
    print("Example 2: Battle of the Sexes")
    print("-"*60)
    game = NormalFormGame.battle_of_sexes()
    compare_nash_vs_correlated(game)

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("✓ Correlated equilibria can achieve higher social welfare")
    print("✓ Computed efficiently via linear programming")
    print("✓ Represents coordination through a mediator/signal")
    print("✓ Used in traffic routing, spectrum allocation, etc.")
    print()
    print("Historical note:")
    print("- Introduced by Aumann (1974, 1987)")
    print("- Generalizes Nash equilibrium")
    print("- Price of Anarchy often better for correlated equilibria")

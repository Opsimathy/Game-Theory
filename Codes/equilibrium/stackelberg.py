"""
Stackelberg Equilibrium (Leader-Follower Games)

In Stackelberg games, one player (leader) commits to a strategy first,
then the other player (follower) responds optimally.

This represents:
- First-mover advantage in competitive markets
- Security games (defender commits, attacker responds)
- Pricing in monopolistic competition

Key Differences from Nash Equilibrium:
- Sequential vs simultaneous moves
- Leader can commit credibly
- Often leads to higher payoff for leader than Nash
- Used in security applications (airport security, wildlife protection)

Applications:
- Cybersecurity: defender allocates resources, attacker responds
- Wildlife protection: rangers patrol, poachers respond
- Market competition: incumbent sets price, entrant responds

Reference:
- von Stackelberg, H. (1934). "Marktform und Gleichgewicht"
- Conitzer & Sandholm (2006). "Computing the Optimal Strategy to Commit to"
- Tambe, M. (2011). "Security and Game Theory" (game-theoretic security)
- Korzhyk et al. (2010). "Stackelberg vs. Nash in Security Games"
"""

import numpy as np
from scipy.optimize import linprog, minimize
from typing import Tuple, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.normal_form import NormalFormGame


class StackelbergSolver:
    """
    Solver for Stackelberg equilibrium in two-player games.

    The leader commits to a (possibly mixed) strategy, and the follower
    best-responds to it.
    """

    def __init__(self, game: NormalFormGame, leader: int = 0):
        """
        Initialize Stackelberg solver.

        Args:
            game: Two-player normal-form game
            leader: Which player is the leader (0 or 1)
        """
        if game.n_players != 2:
            raise ValueError("Stackelberg solver requires 2-player game")

        self.game = game
        self.leader = leader
        self.follower = 1 - leader

    def solve(self) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """
        Compute Strong Stackelberg Equilibrium.

        In SSE, the follower breaks ties in favor of the leader.

        Returns:
            (leader_strategy, follower_strategy, payoffs)
        """
        if self.leader == 0:
            return self._solve_leader_p1()
        else:
            return self._solve_leader_p2()

    def _solve_leader_p1(self) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """
        Solve when player 1 is the leader.

        This is a bi-level optimization:
        - Outer: maximize leader payoff over leader strategies
        - Inner: for each leader strategy, compute follower's best response
        """
        n1, n2 = self.game.n_actions

        # We'll solve this using linear programming
        # Variables: [p1_1, ..., p1_n1, q2_1, ..., q2_n2, v]
        # p1: leader's strategy
        # q2: follower's strategy (best response)
        # v: leader's payoff

        n_vars = n1 + n2 + 1
        p1_vars = slice(0, n1)
        q2_vars = slice(n1, n1 + n2)
        v_var = n1 + n2

        # Objective: maximize v (leader's payoff)
        c = np.zeros(n_vars)
        c[v_var] = -1  # Minimize -v

        # Constraints:
        A_ub = []
        b_ub = []

        # 1. Follower best response: for all actions a2, a2'
        #    Σ p1(a1) u2(a1, a2) ≥ Σ p1(a1) u2(a1, a2') for all a2 in support of q2
        #    This is hard to encode directly, so we use a different formulation

        # Alternative: Use strong duality
        # For each follower action a2, if q2(a2) > 0, then it must be a best response

        # We'll use the MILP formulation, but for now, use an approximation
        # Enumerate over follower pure strategies and find best

        best_leader_payoff = -np.inf
        best_leader_strategy = None
        best_follower_strategy = None

        # Try different leader mixed strategies
        # For computational feasibility, we'll use optimization

        def leader_objective(leader_strategy):
            """Compute leader's payoff when follower best-responds."""
            # Normalize to ensure it's a probability distribution
            p1 = np.abs(leader_strategy)
            if p1.sum() > 0:
                p1 = p1 / p1.sum()
            else:
                p1 = np.ones(n1) / n1

            # Compute follower's payoff for each action
            follower_payoffs = np.zeros(n2)
            for a2 in range(n2):
                for a1 in range(n1):
                    follower_payoffs[a2] += p1[a1] * self.game.payoffs[1][a1, a2]

            # Follower best responds (breaks ties in favor of leader in SSE)
            best_follower_payoff = follower_payoffs.max()
            best_follower_actions = np.where(
                follower_payoffs >= best_follower_payoff - 1e-9
            )[0]

            # Among best responses, follower chooses one that maximizes leader's payoff (SSE)
            best_leader_payoff_among_br = -np.inf
            for a2 in best_follower_actions:
                leader_payoff = 0
                for a1 in range(n1):
                    leader_payoff += p1[a1] * self.game.payoffs[0][a1, a2]
                best_leader_payoff_among_br = max(best_leader_payoff_among_br, leader_payoff)

            return -best_leader_payoff_among_br  # Minimize negative

        # Optimize leader's strategy
        from scipy.optimize import differential_evolution

        # Constraints: strategy sums to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
        bounds = [(0, 1) for _ in range(n1)]

        result = minimize(
            leader_objective,
            x0=np.ones(n1) / n1,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            leader_strategy = result.x
            leader_strategy = np.maximum(leader_strategy, 0)
            leader_strategy /= leader_strategy.sum()

            # Compute follower's best response
            follower_payoffs = np.zeros(n2)
            for a2 in range(n2):
                for a1 in range(n1):
                    follower_payoffs[a2] += leader_strategy[a1] * self.game.payoffs[1][a1, a2]

            best_follower_payoff = follower_payoffs.max()
            best_follower_actions = np.where(
                follower_payoffs >= best_follower_payoff - 1e-9
            )[0]

            # SSE: follower picks action that maximizes leader's payoff among best responses
            best_a2 = None
            best_leader_value = -np.inf
            for a2 in best_follower_actions:
                leader_value = sum(
                    leader_strategy[a1] * self.game.payoffs[0][a1, a2]
                    for a1 in range(n1)
                )
                if leader_value > best_leader_value:
                    best_leader_value = leader_value
                    best_a2 = a2

            follower_strategy = np.zeros(n2)
            follower_strategy[best_a2] = 1.0

            payoffs = self.game.get_expected_payoff([leader_strategy, follower_strategy])

            return leader_strategy, follower_strategy, tuple(payoffs)
        else:
            raise ValueError("Optimization failed")

    def _solve_leader_p2(self) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """Solve when player 2 is the leader (symmetric to above)."""
        n1, n2 = self.game.n_actions

        def leader_objective(leader_strategy):
            p2 = np.abs(leader_strategy)
            if p2.sum() > 0:
                p2 = p2 / p2.sum()
            else:
                p2 = np.ones(n2) / n2

            follower_payoffs = np.zeros(n1)
            for a1 in range(n1):
                for a2 in range(n2):
                    follower_payoffs[a1] += p2[a2] * self.game.payoffs[0][a1, a2]

            best_follower_payoff = follower_payoffs.max()
            best_follower_actions = np.where(
                follower_payoffs >= best_follower_payoff - 1e-9
            )[0]

            best_leader_payoff_among_br = -np.inf
            for a1 in best_follower_actions:
                leader_payoff = sum(
                    p2[a2] * self.game.payoffs[1][a1, a2]
                    for a2 in range(n2)
                )
                best_leader_payoff_among_br = max(best_leader_payoff_among_br, leader_payoff)

            return -best_leader_payoff_among_br

        constraints = {'type': 'eq', 'fun': lambda x: x.sum() - 1}
        bounds = [(0, 1) for _ in range(n2)]

        result = minimize(
            leader_objective,
            x0=np.ones(n2) / n2,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            leader_strategy = result.x
            leader_strategy = np.maximum(leader_strategy, 0)
            leader_strategy /= leader_strategy.sum()

            follower_payoffs = np.zeros(n1)
            for a1 in range(n1):
                for a2 in range(n2):
                    follower_payoffs[a1] += leader_strategy[a2] * self.game.payoffs[0][a1, a2]

            best_follower_payoff = follower_payoffs.max()
            best_follower_actions = np.where(
                follower_payoffs >= best_follower_payoff - 1e-9
            )[0]

            best_a1 = None
            best_leader_value = -np.inf
            for a1 in best_follower_actions:
                leader_value = sum(
                    leader_strategy[a2] * self.game.payoffs[1][a1, a2]
                    for a2 in range(n2)
                )
                if leader_value > best_leader_value:
                    best_leader_value = leader_value
                    best_a1 = a1

            follower_strategy = np.zeros(n1)
            follower_strategy[best_a1] = 1.0

            payoffs = self.game.get_expected_payoff([follower_strategy, leader_strategy])

            return leader_strategy, follower_strategy, tuple(payoffs)
        else:
            raise ValueError("Optimization failed")


if __name__ == "__main__":
    print("="*60)
    print("STACKELBERG EQUILIBRIUM EXAMPLES")
    print("="*60)
    print()

    # Example: Security Game
    print("Example 1: Simple Security Game")
    print("-"*60)
    print("Defender (leader) chooses patrol route")
    print("Attacker (follower) chooses target knowing defender's strategy")
    print()

    # Payoff matrix (Defender payoffs, Attacker payoffs)
    # Actions: Defend Location A, Defend Location B
    #          Attack A, Attack B
    #
    #              Attack A    Attack B
    # Defend A     (1,-1)      (-2,2)
    # Defend B     (-2,2)      (1,-1)

    defender_payoffs = np.array([[1, -2], [-2, 1]])
    attacker_payoffs = np.array([[-1, 2], [2, -1]])

    security_game = NormalFormGame(defender_payoffs, attacker_payoffs)

    # Defender as leader
    solver = StackelbergSolver(security_game, leader=0)
    leader_strat, follower_strat, payoffs = solver.solve()

    print("Stackelberg Equilibrium (Defender leads):")
    print(f"  Defender strategy: {leader_strat}")
    print(f"  Attacker strategy: {follower_strat}")
    print(f"  Payoffs (Defender, Attacker): {payoffs}")
    print()

    # Compare with Nash equilibrium
    from equilibrium.support_enumeration import SupportEnumerationSolver
    nash_solver = SupportEnumerationSolver(security_game)
    nash_eq = nash_solver.solve()

    print("Nash Equilibria for comparison:")
    for i, (s1, s2) in enumerate(nash_eq):
        nash_payoffs = security_game.get_expected_payoff([s1, s2])
        print(f"  Nash {i+1}: Defender={s1}, Attacker={s2}")
        print(f"  Payoffs: {nash_payoffs}")

    print()
    print("="*60)
    print("Example 2: Market Entry Game")
    print("-"*60)

    # Incumbent (leader) sets price
    # Entrant (follower) decides whether to enter
    # High price: (5, -1) if enter, (8, 0) if stay out
    # Low price: (2, -3) if enter, (3, 0) if stay out

    incumbent_payoffs = np.array([[5, 8], [2, 3]])
    entrant_payoffs = np.array([[-1, 0], [-3, 0]])

    entry_game = NormalFormGame(incumbent_payoffs, entrant_payoffs)

    solver = StackelbergSolver(entry_game, leader=0)
    leader_strat, follower_strat, payoffs = solver.solve()

    print("Stackelberg Equilibrium (Incumbent leads):")
    print(f"  Incumbent strategy: {leader_strat}")
    print(f"  Entrant strategy: {follower_strat}")
    print(f"  Payoffs (Incumbent, Entrant): {payoffs}")

    print()
    print("="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("✓ First-mover advantage: Leader can commit credibly")
    print("✓ Often yields higher payoff for leader than Nash equilibrium")
    print("✓ Used in security (LAX, TSA), wildlife protection, cybersecurity")
    print("✓ Strong Stackelberg Equilibrium: follower breaks ties favorably")
    print()
    print("Applications:")
    print("- Airport security: ARMOR system at LAX (since 2007)")
    print("- Wildlife protection: PAWS system")
    print("- Network security: resource allocation")
    print("- Market competition: pricing and entry deterrence")

"""
Replicator Dynamics and Evolutionary Stable Strategies

Replicator dynamics models how strategy frequencies evolve in populations
where more successful strategies reproduce at higher rates.

The Replicator Equation:
    dx_i/dt = x_i * [u_i(x) - u_avg(x)]

where:
- x_i = frequency of strategy i
- u_i(x) = payoff of strategy i against population x
- u_avg(x) = average payoff in population

Key Properties:
- Nash equilibria are fixed points
- Evolutionarily Stable Strategies (ESS) are asymptotically stable
- Models biological evolution, cultural transmission, learning

Applications:
- Biological evolution (Hawk-Dove, Prisoner's Dilemma)
- Cultural evolution and social norms
- Learning in games
- Population dynamics

Reference:
- Maynard Smith & Price (1973). "The Logic of Animal Conflict"
- Taylor & Jonker (1978). "Evolutionary Stable Strategies and Game Dynamics"
- Hofbauer & Sigmund (1998). "Evolutionary Games and Population Dynamics"
- Weibull (1995). "Evolutionary Game Theory"
- Nowak (2006). "Evolutionary Dynamics"
"""

import numpy as np
from scipy.integrate import odeint
from typing import List, Tuple, Optional, Callable
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.normal_form import NormalFormGame


class ReplicatorDynamics:
    """
    Replicator dynamics for symmetric games.

    Models evolution of strategy frequencies in a population.
    """

    def __init__(self, payoff_matrix: np.ndarray):
        """
        Initialize replicator dynamics.

        Args:
            payoff_matrix: Symmetric payoff matrix A where A[i,j] is
                          payoff of strategy i against strategy j
        """
        self.payoff_matrix = payoff_matrix
        self.n_strategies = payoff_matrix.shape[0]

    def fitness(self, strategy_freq: np.ndarray) -> np.ndarray:
        """
        Compute fitness of each strategy.

        Args:
            strategy_freq: Current frequency distribution

        Returns:
            Fitness of each strategy
        """
        return self.payoff_matrix @ strategy_freq

    def average_fitness(self, strategy_freq: np.ndarray) -> float:
        """
        Compute average fitness in population.

        Args:
            strategy_freq: Current frequency distribution

        Returns:
            Population average fitness
        """
        fitness_values = self.fitness(strategy_freq)
        return strategy_freq @ fitness_values

    def replicator_equation(self, x: np.ndarray, t: float = 0) -> np.ndarray:
        """
        Replicator dynamics differential equation.

        dx_i/dt = x_i * (u_i(x) - u_avg(x))

        Args:
            x: Strategy frequencies
            t: Time (unused, for ODE solver compatibility)

        Returns:
            Derivative dx/dt
        """
        fitness_values = self.fitness(x)
        avg_fitness = x @ fitness_values

        # Replicator equation
        dx_dt = x * (fitness_values - avg_fitness)

        return dx_dt

    def simulate(
        self,
        initial_freq: np.ndarray,
        timesteps: np.ndarray
    ) -> np.ndarray:
        """
        Simulate replicator dynamics over time.

        Args:
            initial_freq: Initial strategy frequencies
            timesteps: Array of time points to evaluate

        Returns:
            Strategy frequencies at each timestep (timesteps × strategies)
        """
        # Normalize initial frequencies
        initial_freq = initial_freq / initial_freq.sum()

        # Integrate replicator equation
        trajectory = odeint(self.replicator_equation, initial_freq, timesteps)

        return trajectory

    def find_fixed_points(self, n_trials: int = 100) -> List[np.ndarray]:
        """
        Find fixed points (Nash equilibria) of replicator dynamics.

        Uses random restarts to find multiple fixed points.

        Args:
            n_trials: Number of random initializations

        Returns:
            List of fixed points found
        """
        from scipy.optimize import fsolve

        fixed_points = []
        tolerance = 1e-6

        for _ in range(n_trials):
            # Random initial point
            x0 = np.random.dirichlet(np.ones(self.n_strategies))

            # Solve for fixed point
            solution = fsolve(self.replicator_equation, x0)

            # Normalize
            solution = np.maximum(solution, 0)
            if solution.sum() > 0:
                solution = solution / solution.sum()

            # Check if it's actually a fixed point
            residual = np.linalg.norm(self.replicator_equation(solution))
            if residual < tolerance:
                # Check if we already found this one
                is_new = True
                for fp in fixed_points:
                    if np.linalg.norm(fp - solution) < tolerance:
                        is_new = False
                        break

                if is_new:
                    fixed_points.append(solution)

        return fixed_points


class ESS:
    """
    Evolutionarily Stable Strategy analysis.

    An ESS is a strategy that, if adopted by a population, cannot be invaded
    by any alternative (mutant) strategy.
    """

    def __init__(self, payoff_matrix: np.ndarray):
        """
        Initialize ESS analyzer.

        Args:
            payoff_matrix: Symmetric payoff matrix
        """
        self.payoff_matrix = payoff_matrix
        self.n_strategies = payoff_matrix.shape[0]

    def is_ess(self, strategy: np.ndarray, tolerance: float = 1e-8) -> bool:
        """
        Check if a strategy is an ESS.

        A strategy x* is an ESS if for all mutant strategies y ≠ x*:
        1. u(x*, x*) ≥ u(y, x*) (Nash equilibrium condition)
        2. If u(x*, x*) = u(y, x*), then u(x*, y) > u(y, y) (stability condition)

        Args:
            strategy: Strategy to test
            tolerance: Numerical tolerance

        Returns:
            True if strategy is an ESS
        """
        # Normalize strategy
        strategy = strategy / strategy.sum()

        # Fitness of strategy against itself
        fitness_self = strategy @ self.payoff_matrix @ strategy

        # Check all possible deviations
        # For pure strategies as mutants
        for i in range(self.n_strategies):
            mutant = np.zeros(self.n_strategies)
            mutant[i] = 1.0

            # Skip if mutant equals the strategy
            if np.allclose(mutant, strategy, atol=tolerance):
                continue

            # Condition 1: u(x*, x*) ≥ u(y, x*)
            fitness_mutant_vs_population = mutant @ self.payoff_matrix @ strategy

            if fitness_mutant_vs_population > fitness_self + tolerance:
                return False

            # Condition 2: If equal, check u(x*, y) > u(y, y)
            if abs(fitness_mutant_vs_population - fitness_self) < tolerance:
                fitness_strategy_vs_mutant = strategy @ self.payoff_matrix @ mutant
                fitness_mutant_vs_mutant = mutant @ self.payoff_matrix @ mutant

                if fitness_strategy_vs_mutant <= fitness_mutant_vs_mutant + tolerance:
                    return False

        return True

    def find_all_ess(self) -> List[np.ndarray]:
        """
        Find all pure and mixed ESS.

        Returns:
            List of ESS found
        """
        ess_strategies = []

        # Check all pure strategies
        for i in range(self.n_strategies):
            strategy = np.zeros(self.n_strategies)
            strategy[i] = 1.0

            if self.is_ess(strategy):
                ess_strategies.append(strategy)

        # Try to find mixed ESS using Nash equilibria
        from games.normal_form import NormalFormGame
        game = NormalFormGame(self.payoff_matrix, self.payoff_matrix)

        try:
            from equilibrium.support_enumeration import SupportEnumerationSolver
            solver = SupportEnumerationSolver(game)
            equilibria = solver.solve()

            for s1, _ in equilibria:
                if self.is_ess(s1):
                    # Check if not already in list
                    is_new = True
                    for ess in ess_strategies:
                        if np.allclose(ess, s1):
                            is_new = False
                            break
                    if is_new:
                        ess_strategies.append(s1)
        except:
            pass

        return ess_strategies


def hawk_dove_game():
    """
    Create the classic Hawk-Dove game.

    Scenario: Two animals compete for a resource of value V
    - Hawk: Always fights
    - Dove: Displays but retreats if opponent fights

    Payoffs (V=2, C=3 for injury cost):
    - Hawk vs Hawk: (V-C)/2 = -0.5 (fight, both get injured)
    - Hawk vs Dove: V = 2 (hawk gets resource)
    - Dove vs Hawk: 0 (dove retreats)
    - Dove vs Dove: V/2 = 1 (share resource)
    """
    V = 2  # Resource value
    C = 3  # Cost of injury

    payoff_matrix = np.array([
        [(V - C) / 2, V],      # Hawk vs (Hawk, Dove)
        [0, V / 2]             # Dove vs (Hawk, Dove)
    ])

    return payoff_matrix


if __name__ == "__main__":
    print("="*60)
    print("REPLICATOR DYNAMICS AND ESS")
    print("="*60)
    print()

    # Example 1: Hawk-Dove Game
    print("Example 1: Hawk-Dove Game")
    print("-"*60)
    print("Two strategies: Hawk (aggressive), Dove (peaceful)")
    print("Resource value V=2, injury cost C=3")
    print()

    payoff_matrix = hawk_dove_game()
    print("Payoff Matrix:")
    print("       Hawk  Dove")
    print(f"Hawk   {payoff_matrix[0,0]:.2f}   {payoff_matrix[0,1]:.2f}")
    print(f"Dove   {payoff_matrix[1,0]:.2f}   {payoff_matrix[1,1]:.2f}")
    print()

    # Analyze ESS
    ess_analyzer = ESS(payoff_matrix)
    ess_strategies = ess_analyzer.find_all_ess()

    print(f"Found {len(ess_strategies)} ESS:")
    for i, ess in enumerate(ess_strategies):
        print(f"  ESS {i+1}: Hawk={ess[0]:.3f}, Dove={ess[1]:.3f}")

    # Simulate replicator dynamics
    print("\nSimulating Replicator Dynamics:")
    rd = ReplicatorDynamics(payoff_matrix)

    # Try different initial conditions
    initial_conditions = [
        np.array([0.9, 0.1]),  # Mostly hawks
        np.array([0.5, 0.5]),  # Equal mix
        np.array([0.1, 0.9]),  # Mostly doves
    ]

    timesteps = np.linspace(0, 50, 1000)

    for i, init_freq in enumerate(initial_conditions):
        trajectory = rd.simulate(init_freq, timesteps)
        final_freq = trajectory[-1]

        print(f"\n  Initial: Hawk={init_freq[0]:.2f}, Dove={init_freq[1]:.2f}")
        print(f"  Final:   Hawk={final_freq[0]:.3f}, Dove={final_freq[1]:.3f}")

    print()
    print("="*60)
    print("Example 2: Rock-Paper-Scissors")
    print("-"*60)

    # Rock-Paper-Scissors: cyclic dynamics
    rps_payoff = np.array([
        [0, -1, 1],   # Rock vs (Rock, Paper, Scissors)
        [1, 0, -1],   # Paper vs (Rock, Paper, Scissors)
        [-1, 1, 0]    # Scissors vs (Rock, Paper, Scissors)
    ])

    rd_rps = ReplicatorDynamics(rps_payoff)
    init_freq = np.array([0.5, 0.3, 0.2])
    timesteps = np.linspace(0, 20, 1000)

    trajectory = rd_rps.simulate(init_freq, timesteps)

    print("Initial frequencies: Rock=0.50, Paper=0.30, Scissors=0.20")
    print("\nFrequencies at different times:")
    for t_idx in [100, 300, 500, 700, 999]:
        t = timesteps[t_idx]
        freq = trajectory[t_idx]
        print(f"  t={t:.1f}: Rock={freq[0]:.3f}, Paper={freq[1]:.3f}, Scissors={freq[2]:.3f}")

    print()
    print("="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("✓ Replicator dynamics models evolutionary strategy change")
    print("✓ ESS are asymptotically stable under replicator dynamics")
    print("✓ Not all Nash equilibria are ESS")
    print("✓ Cycles possible (e.g., Rock-Paper-Scissors)")
    print()
    print("Applications:")
    print("- Biological evolution (Hawk-Dove, Prisoner's Dilemma)")
    print("- Cultural evolution and norm emergence")
    print("- Learning in games (EWA, reinforcement learning)")
    print("- Population genetics and ecology")
    print()
    print("Historical notes:")
    print("- Maynard Smith & Price (1973): Introduced ESS")
    print("- Taylor & Jonker (1978): Connected to replicator dynamics")
    print("- Nowak (2006): Extended to evolutionary graph theory")

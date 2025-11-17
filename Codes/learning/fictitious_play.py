"""
Fictitious Play

Fictitious Play is a classic learning dynamics in games:
1. Each player maintains beliefs about opponent's strategy (empirical frequency)
2. Each round, play best response to these beliefs
3. Update beliefs based on observed actions

Convergence Properties:
- Converges to Nash equilibrium in:
  * Zero-sum games
  * Games with unique Nash equilibrium
  * Potential games
- May not converge in general games (can cycle)

Applications:
- Modeling learning in repeated games
- Finding Nash equilibria iteratively
- Understanding strategic behavior evolution

Reference:
- Brown, G. W. (1951). "Iterative Solution of Games by Fictitious Play"
- Robinson, J. (1951). "An Iterative Method of Solving a Game"
- Fudenberg & Levine (1998). "The Theory of Learning in Games"
"""

import numpy as np
from typing import List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.normal_form import NormalFormGame


class FictitiousPlay:
    """
    Fictitious play learning algorithm for games.

    Each player:
    - Tracks empirical frequency of opponent actions
    - Plays best response to empirical frequency
    - Updates beliefs after each round
    """

    def __init__(self, game: NormalFormGame):
        """
        Initialize fictitious play.

        Args:
            game: The game to learn in
        """
        if game.n_players != 2:
            raise ValueError("Fictitious play implementation supports 2-player games")

        self.game = game
        self.n_actions_p1, self.n_actions_p2 = game.n_actions

        # Track action counts for each player
        self.action_counts_p1 = np.zeros(self.n_actions_p1)
        self.action_counts_p2 = np.zeros(self.n_actions_p2)

        # Initialize with uniform beliefs
        self.action_counts_p1[:] = 1
        self.action_counts_p2[:] = 1

        self.iteration = 0

    def get_beliefs(self, player: int) -> np.ndarray:
        """
        Get current beliefs (empirical frequency) about a player's strategy.

        Args:
            player: Player index (0 or 1)

        Returns:
            Probability distribution over actions
        """
        if player == 0:
            counts = self.action_counts_p1
        else:
            counts = self.action_counts_p2

        return counts / counts.sum()

    def best_response(self, player: int, opponent_beliefs: np.ndarray) -> int:
        """
        Compute best response to opponent's empirical frequency.

        Args:
            player: Player computing best response
            opponent_beliefs: Believed strategy of opponent

        Returns:
            Best response action
        """
        payoff_matrix = self.game.payoffs[player]
        n_actions = self.game.n_actions[player]

        # Compute expected payoff for each action
        expected_payoffs = np.zeros(n_actions)

        if player == 0:
            # Player 1: rows
            for action in range(n_actions):
                expected_payoffs[action] = payoff_matrix[action, :] @ opponent_beliefs
        else:
            # Player 2: columns
            for action in range(n_actions):
                expected_payoffs[action] = payoff_matrix[:, action] @ opponent_beliefs

        # Return best action (break ties by smallest index)
        return int(np.argmax(expected_payoffs))

    def step(self) -> Tuple[int, int]:
        """
        Execute one step of fictitious play.

        Returns:
            (action_p1, action_p2) chosen in this round
        """
        # Player 1's beliefs about player 2
        beliefs_p2 = self.get_beliefs(1)
        action_p1 = self.best_response(0, beliefs_p2)

        # Player 2's beliefs about player 1
        beliefs_p1 = self.get_beliefs(0)
        action_p2 = self.best_response(1, beliefs_p1)

        # Update counts
        self.action_counts_p1[action_p1] += 1
        self.action_counts_p2[action_p2] += 1

        self.iteration += 1

        return action_p1, action_p2

    def train(self, iterations: int) -> List[np.ndarray]:
        """
        Run fictitious play for multiple iterations.

        Args:
            iterations: Number of iterations to run

        Returns:
            Empirical frequencies (beliefs) for each player
        """
        for _ in range(iterations):
            self.step()

        return [self.get_beliefs(0), self.get_beliefs(1)]

    def get_current_strategies(self) -> List[np.ndarray]:
        """
        Get current empirical frequency strategies.

        Returns:
            List of strategies for each player
        """
        return [self.get_beliefs(0), self.get_beliefs(1)]


def analyze_convergence(
    game: NormalFormGame,
    iterations: int = 1000,
    check_interval: int = 100
) -> None:
    """
    Analyze convergence of fictitious play.

    Args:
        game: Game to analyze
        iterations: Number of iterations
        check_interval: How often to print status
    """
    fp = FictitiousPlay(game)

    print(f"Running fictitious play for {iterations} iterations...")
    print()

    for i in range(0, iterations, check_interval):
        fp.train(check_interval)

        strategies = fp.get_current_strategies()
        payoffs = game.get_expected_payoff(strategies)
        is_nash = game.is_nash_equilibrium(strategies)

        print(f"Iteration {fp.iteration}:")
        print(f"  Player 1 strategy: {strategies[0]}")
        print(f"  Player 2 strategy: {strategies[1]}")
        print(f"  Expected payoffs: {payoffs}")
        print(f"  Is Nash equilibrium: {is_nash}")
        print()


if __name__ == "__main__":
    print("=== Fictitious Play Examples ===\n")

    print("Example 1: Matching Pennies (does not converge)")
    print("-" * 50)
    game = NormalFormGame.matching_pennies()
    analyze_convergence(game, iterations=500, check_interval=100)

    print("\n" + "="*50 + "\n")

    print("Example 2: Rock-Paper-Scissors (converges to mixed equilibrium)")
    print("-" * 50)
    game = NormalFormGame.rock_paper_scissors()
    analyze_convergence(game, iterations=1000, check_interval=200)

    print("\n" + "="*50 + "\n")

    print("Example 3: Battle of the Sexes (may cycle)")
    print("-" * 50)
    game = NormalFormGame.battle_of_sexes()
    analyze_convergence(game, iterations=500, check_interval=100)

    print("\n=== Key Insights ===")
    print("Fictitious Play:")
    print("✓ Simple and intuitive learning rule")
    print("✓ Converges in zero-sum games")
    print("✓ Provides iterative Nash equilibrium finder")
    print("✗ May not converge in general games")
    print("✗ Can cycle in coordination games")
    print()
    print("Historical note:")
    print("- Introduced by Brown (1951)")
    print("- Robinson (1951) proved convergence for zero-sum games")
    print("- Basis for many modern learning algorithms")

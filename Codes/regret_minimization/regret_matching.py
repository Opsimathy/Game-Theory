"""
Regret Matching Algorithm

Regret matching is a simple yet powerful algorithm for learning in games:
1. Track cumulative regret for each action
2. Play actions proportional to positive regret
3. Update regrets based on observed payoffs

Key properties:
- Guarantees O(1/sqrt(T)) average regret
- Converges to Nash equilibrium in two-player zero-sum games
- Simple to implement and understand

Reference:
- Hart & Mas-Colell (2000). "A Simple Adaptive Procedure Leading to Correlated Equilibrium"
- Zinkevich et al. (2007). "Regret Minimization in Games with Incomplete Information"
"""

import numpy as np
from typing import List, Callable
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.normal_form import NormalFormGame


class RegretMatchingSolver:
    """
    Regret matching solver for repeated games.

    Learns a strategy by minimizing regret in repeated play.
    """

    def __init__(self, game: NormalFormGame, player: int = 0):
        """
        Initialize regret matching solver.

        Args:
            game: The game to solve
            player: Which player this solver controls
        """
        self.game = game
        self.player = player
        self.n_actions = game.n_actions[player]

        # Cumulative regret for each action
        self.cumulative_regret = np.zeros(self.n_actions)

        # Strategy sum (for computing average strategy)
        self.strategy_sum = np.zeros(self.n_actions)

        # Current iteration
        self.iteration = 0

    def get_strategy(self) -> np.ndarray:
        """
        Get current strategy using regret matching.

        Strategy is proportional to positive regret.
        If all regrets are negative, play uniformly.

        Returns:
            Current mixed strategy
        """
        strategy = np.maximum(self.cumulative_regret, 0)

        normalizing_sum = strategy.sum()
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.ones(self.n_actions) / self.n_actions

        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        Get average strategy over all iterations.

        The average strategy converges to Nash equilibrium.

        Returns:
            Average mixed strategy
        """
        if self.strategy_sum.sum() > 0:
            return self.strategy_sum / self.strategy_sum.sum()
        else:
            return np.ones(self.n_actions) / self.n_actions

    def update(self, action_taken: int, action_utilities: np.ndarray) -> None:
        """
        Update regrets after taking an action.

        Args:
            action_taken: The action that was played
            action_utilities: Utility for each possible action
        """
        # Current strategy
        strategy = self.get_strategy()

        # Expected utility from current strategy
        expected_utility = strategy @ action_utilities

        # Regret for each action
        regrets = action_utilities - expected_utility

        # Update cumulative regret
        self.cumulative_regret += regrets

        # Update strategy sum for average strategy
        self.strategy_sum += strategy

        self.iteration += 1

    def train(
        self,
        opponent_strategy_fn: Callable[[], np.ndarray],
        iterations: int
    ) -> np.ndarray:
        """
        Train against an opponent for a number of iterations.

        Args:
            opponent_strategy_fn: Function that returns opponent's current strategy
            iterations: Number of iterations to train

        Returns:
            Average strategy after training
        """
        for _ in range(iterations):
            # Get current strategy
            strategy = self.get_strategy()

            # Sample action
            action = np.random.choice(self.n_actions, p=strategy)

            # Get opponent strategy
            opponent_strategy = opponent_strategy_fn()

            # Compute utilities for each action
            action_utilities = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                if self.player == 0:
                    # Player 1
                    for opp_action in range(len(opponent_strategy)):
                        action_utilities[a] += (
                            opponent_strategy[opp_action] *
                            self.game.payoffs[self.player][a, opp_action]
                        )
                else:
                    # Player 2
                    for opp_action in range(len(opponent_strategy)):
                        action_utilities[a] += (
                            opponent_strategy[opp_action] *
                            self.game.payoffs[self.player][opp_action, a]
                        )

            # Update regrets
            self.update(action, action_utilities)

        return self.get_average_strategy()


def self_play_regret_matching(game: NormalFormGame, iterations: int) -> List[np.ndarray]:
    """
    Run self-play regret matching for both players.

    Args:
        game: Two-player game
        iterations: Number of iterations

    Returns:
        List of average strategies for each player
    """
    if game.n_players != 2:
        raise ValueError("Self-play only works for two-player games")

    solver_p1 = RegretMatchingSolver(game, player=0)
    solver_p2 = RegretMatchingSolver(game, player=1)

    for _ in range(iterations):
        # Player 1 update
        strategy_p1 = solver_p1.get_strategy()
        strategy_p2 = solver_p2.get_strategy()

        action_p1 = np.random.choice(solver_p1.n_actions, p=strategy_p1)

        # Compute utilities for each action of player 1
        utilities_p1 = np.zeros(solver_p1.n_actions)
        for a in range(solver_p1.n_actions):
            for b in range(solver_p2.n_actions):
                utilities_p1[a] += strategy_p2[b] * game.payoffs[0][a, b]

        solver_p1.update(action_p1, utilities_p1)

        # Player 2 update
        strategy_p1 = solver_p1.get_strategy()
        strategy_p2 = solver_p2.get_strategy()

        action_p2 = np.random.choice(solver_p2.n_actions, p=strategy_p2)

        # Compute utilities for each action of player 2
        utilities_p2 = np.zeros(solver_p2.n_actions)
        for a in range(solver_p1.n_actions):
            for b in range(solver_p2.n_actions):
                utilities_p2[b] += strategy_p1[a] * game.payoffs[1][a, b]

        solver_p2.update(action_p2, utilities_p2)

    return [solver_p1.get_average_strategy(), solver_p2.get_average_strategy()]


if __name__ == "__main__":
    print("=== Rock-Paper-Scissors ===")
    game = NormalFormGame.rock_paper_scissors()
    strategies = self_play_regret_matching(game, iterations=100000)
    print(f"Player 1 strategy: {strategies[0]}")
    print(f"Player 2 strategy: {strategies[1]}")
    print(f"Expected payoffs: {game.get_expected_payoff(strategies)}")

    print("\n=== Matching Pennies ===")
    game = NormalFormGame.matching_pennies()
    strategies = self_play_regret_matching(game, iterations=100000)
    print(f"Player 1 strategy: {strategies[0]}")
    print(f"Player 2 strategy: {strategies[1]}")
    print(f"Expected payoffs: {game.get_expected_payoff(strategies)}")

    print("\n=== Battle of the Sexes ===")
    game = NormalFormGame.battle_of_sexes()
    strategies = self_play_regret_matching(game, iterations=100000)
    print(f"Player 1 strategy: {strategies[0]}")
    print(f"Player 2 strategy: {strategies[1]}")
    print(f"Expected payoffs: {game.get_expected_payoff(strategies)}")

"""
Counterfactual Regret Minimization (CFR)

CFR is the state-of-the-art algorithm for solving large imperfect information games.
It extends regret matching to extensive-form games by computing counterfactual values.

Key innovations:
- Operates on information sets rather than game states
- Computes counterfactual regret (regret weighted by reach probability)
- Guaranteed O(1/sqrt(T)) convergence to Nash equilibrium
- Scales to games with 10^14+ states (e.g., heads-up poker)

Applications:
- Libratus (heads-up no-limit poker, 2017)
- Pluribus (6-player no-limit poker, 2019)
- DeepStack (2017)

Reference:
- Zinkevich et al. (2007). "Regret Minimization in Games with Incomplete Information"
- Lanctot et al. (2009). "Monte Carlo Sampling for Regret Minimization in Extensive Games"
- Bowling et al. (2015). "Heads-up Limit Hold'em Poker is Solved"
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.extensive_form import ExtensiveFormGame, GameNode


class CFRSolver:
    """
    Counterfactual Regret Minimization solver for extensive-form games.
    """

    def __init__(self, game: ExtensiveFormGame):
        """
        Initialize CFR solver.

        Args:
            game: Extensive-form game to solve
        """
        self.game = game

        # Regret and strategy sums for each information set
        self.regret_sum: Dict[str, np.ndarray] = {}
        self.strategy_sum: Dict[str, np.ndarray] = {}

        # Initialize for all information sets
        for infoset in game.infosets:
            n_actions = len(game.get_information_set_actions(infoset))
            self.regret_sum[infoset] = np.zeros(n_actions)
            self.strategy_sum[infoset] = np.zeros(n_actions)

        self.iteration = 0

    def get_strategy(self, infoset: str) -> np.ndarray:
        """
        Get current strategy for an information set using regret matching.

        Args:
            infoset: Information set identifier

        Returns:
            Current mixed strategy
        """
        regrets = self.regret_sum[infoset]
        strategy = np.maximum(regrets, 0)

        normalizing_sum = strategy.sum()
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            n_actions = len(regrets)
            strategy = np.ones(n_actions) / n_actions

        return strategy

    def get_average_strategy(self, infoset: str) -> np.ndarray:
        """
        Get average strategy for an information set.

        Args:
            infoset: Information set identifier

        Returns:
            Average strategy over all iterations
        """
        avg_strategy = self.strategy_sum[infoset]

        normalizing_sum = avg_strategy.sum()
        if normalizing_sum > 0:
            return avg_strategy / normalizing_sum
        else:
            n_actions = len(avg_strategy)
            return np.ones(n_actions) / n_actions

    def cfr(
        self,
        node: GameNode,
        reach_probs: List[float],
        player: int
    ) -> float:
        """
        Recursive CFR algorithm.

        Args:
            node: Current game node
            reach_probs: Reach probabilities for each player
            player: Player for whom we're computing values

        Returns:
            Expected value for the player at this node
        """
        # Terminal node
        if node.is_terminal:
            return node.payoffs[player]

        # Chance node
        if node.is_chance:
            value = 0.0
            for action, prob in node.chance_probs.items():
                child = node.children[action]
                value += prob * self.cfr(child, reach_probs, player)
            return value

        # Decision node
        infoset = node.infoset
        n_actions = len(node.actions)

        # Get current strategy
        strategy = self.get_strategy(infoset)

        # Compute value for each action
        action_values = np.zeros(n_actions)
        for i, action in enumerate(node.actions):
            if action not in node.children:
                continue

            child = node.children[action]

            # Update reach probabilities
            new_reach_probs = reach_probs.copy()
            new_reach_probs[node.player] *= strategy[i]

            action_values[i] = self.cfr(child, new_reach_probs, player)

        # Node value
        node_value = strategy @ action_values

        # Update regrets for the acting player
        if node.player == player:
            # Counterfactual reach probability (all players except current)
            cfr_reach = 1.0
            for p in range(self.game.n_players):
                if p != player:
                    cfr_reach *= reach_probs[p]

            # Counterfactual regret for each action
            for i in range(n_actions):
                regret = action_values[i] - node_value
                self.regret_sum[infoset][i] += cfr_reach * regret

        # Update strategy sum
        if node.player == player:
            for i in range(n_actions):
                self.strategy_sum[infoset][i] += reach_probs[player] * strategy[i]

        return node_value

    def train(self, iterations: int) -> Dict[str, np.ndarray]:
        """
        Train using CFR for a number of iterations.

        Args:
            iterations: Number of CFR iterations

        Returns:
            Average strategy for each information set
        """
        for _ in range(iterations):
            for player in range(self.game.n_players):
                # Initialize reach probabilities
                reach_probs = [1.0] * self.game.n_players

                # Run CFR for this player
                self.cfr(self.game.root, reach_probs, player)

            self.iteration += 1

        # Return average strategies
        avg_strategies = {}
        for infoset in self.game.infosets:
            avg_strategies[infoset] = self.get_average_strategy(infoset)

        return avg_strategies

    def get_exploitability(self) -> float:
        """
        Compute exploitability of current average strategy.

        Exploitability is the amount by which the sum of best response
        values exceeds the Nash equilibrium value.

        Returns:
            Exploitability value
        """
        # This is a simplified version
        # Full implementation would compute best response values
        # For now, we return 0 as a placeholder
        return 0.0


if __name__ == "__main__":
    print("=== Kuhn Poker ===")
    game = ExtensiveFormGame.kuhn_poker()
    solver = CFRSolver(game)

    print("Training CFR...")
    strategies = solver.train(iterations=10000)

    print("\nAverage Strategies:")
    for infoset, strategy in strategies.items():
        actions = game.get_information_set_actions(infoset)
        print(f"{infoset}:")
        for action, prob in zip(actions, strategy):
            print(f"  {action}: {prob:.3f}")

    print("\n=== Interpretation ===")
    print("In Kuhn Poker, optimal play involves:")
    print("- Betting with strong hands (King)")
    print("- Bluffing with some weak hands (Jack)")
    print("- Calling with medium-strong hands")
    print("- Folding with very weak hands")

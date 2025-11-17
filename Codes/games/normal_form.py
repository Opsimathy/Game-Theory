"""
Normal-Form (Strategic-Form) Game Representation

A normal-form game is represented by:
- A set of players
- A set of actions (pure strategies) for each player
- A payoff function mapping action profiles to payoffs

Reference:
- Osborne & Rubinstein, "A Course in Game Theory" (1994), Chapter 2
- Shoham & Leyton-Brown, "Multiagent Systems" (2009), Chapter 3
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import product


class NormalFormGame:
    """
    Represents a normal-form game with n players.

    For two-player games, payoffs are represented as matrices.
    For n-player games (n > 2), payoffs are n-dimensional tensors.

    Attributes:
        n_players: Number of players
        n_actions: List of number of actions for each player
        payoffs: List of payoff tensors, one per player
    """

    def __init__(self, *payoff_matrices):
        """
        Initialize a normal-form game.

        Args:
            *payoff_matrices: Payoff matrix/tensor for each player.
                For 2-player games: two matrices of shape (n_actions_p1, n_actions_p2)
                For n-player games: n tensors of shape (n_actions_p1, ..., n_actions_pn)

        Example:
            # Prisoner's Dilemma
            payoff_p1 = np.array([[-1, -3], [0, -2]])
            payoff_p2 = np.array([[-1, 0], [-3, -2]])
            game = NormalFormGame(payoff_p1, payoff_p2)
        """
        self.payoffs = [np.array(p) for p in payoff_matrices]
        self.n_players = len(self.payoffs)

        # Validate that all payoff tensors have the same shape
        shapes = [p.shape for p in self.payoffs]
        if len(set(shapes)) != 1:
            raise ValueError("All payoff tensors must have the same shape")

        self.n_actions = list(self.payoffs[0].shape)

    def get_payoff(self, action_profile: Tuple[int, ...]) -> List[float]:
        """
        Get payoffs for all players given an action profile.

        Args:
            action_profile: Tuple of actions, one per player

        Returns:
            List of payoffs for each player
        """
        return [self.payoffs[i][action_profile] for i in range(self.n_players)]

    def get_expected_payoff(self, strategies: List[np.ndarray]) -> List[float]:
        """
        Compute expected payoffs given mixed strategies for all players.

        Args:
            strategies: List of mixed strategy profiles (probability distributions)
                Each strategy is a numpy array summing to 1.

        Returns:
            Expected payoff for each player
        """
        expected_payoffs = []

        for player in range(self.n_players):
            expected_payoff = 0.0

            # Iterate over all action profiles
            for action_profile in product(*[range(n) for n in self.n_actions]):
                # Probability of this action profile
                prob = np.prod([strategies[p][action_profile[p]]
                               for p in range(self.n_players)])
                # Add to expected payoff
                expected_payoff += prob * self.payoffs[player][action_profile]

            expected_payoffs.append(expected_payoff)

        return expected_payoffs

    def best_response(self, player: int, opponent_strategies: List[np.ndarray]) -> np.ndarray:
        """
        Compute best response for a player against opponent strategies.

        Args:
            player: Index of the player
            opponent_strategies: List of strategies for all other players

        Returns:
            Best response mixed strategy (may be pure or mixed if multiple best responses)
        """
        n_actions = self.n_actions[player]
        expected_payoffs = np.zeros(n_actions)

        # Compute expected payoff for each pure strategy
        for action in range(n_actions):
            # Build action profile with this action for the player
            for opponent_profile in product(*[range(self.n_actions[p])
                                             for p in range(self.n_players) if p != player]):
                # Insert player's action into the profile
                action_profile = list(opponent_profile)
                action_profile.insert(player, action)
                action_profile = tuple(action_profile)

                # Probability of opponent profile
                opponent_idx = [p for p in range(self.n_players) if p != player]
                prob = np.prod([opponent_strategies[i][opponent_profile[i]]
                               for i in range(len(opponent_idx))])

                expected_payoffs[action] += prob * self.payoffs[player][action_profile]

        # Return uniform distribution over best actions
        best_actions = np.where(expected_payoffs == expected_payoffs.max())[0]
        best_response = np.zeros(n_actions)
        best_response[best_actions] = 1.0 / len(best_actions)

        return best_response

    def is_nash_equilibrium(self, strategies: List[np.ndarray], tolerance: float = 1e-6) -> bool:
        """
        Check if a strategy profile is a Nash equilibrium.

        Args:
            strategies: List of mixed strategies, one per player
            tolerance: Numerical tolerance for comparison

        Returns:
            True if the profile is a Nash equilibrium
        """
        for player in range(self.n_players):
            opponent_strategies = [strategies[p] for p in range(self.n_players) if p != player]
            best_resp = self.best_response(player, opponent_strategies)

            # Check if current strategy is a best response
            current_payoff = self.get_expected_payoff(strategies)[player]

            # Compute payoff from best response
            br_strategies = strategies.copy()
            br_strategies[player] = best_resp
            br_payoff = self.get_expected_payoff(br_strategies)[player]

            if br_payoff - current_payoff > tolerance:
                return False

        return True

    @classmethod
    def prisoners_dilemma(cls):
        """Create the classic Prisoner's Dilemma game."""
        payoff_p1 = np.array([[-1, -3], [0, -2]])
        payoff_p2 = np.array([[-1, 0], [-3, -2]])
        return cls(payoff_p1, payoff_p2)

    @classmethod
    def matching_pennies(cls):
        """Create the Matching Pennies game."""
        payoff_p1 = np.array([[1, -1], [-1, 1]])
        payoff_p2 = np.array([[-1, 1], [1, -1]])
        return cls(payoff_p1, payoff_p2)

    @classmethod
    def rock_paper_scissors(cls):
        """Create the Rock-Paper-Scissors game."""
        # Rock, Paper, Scissors
        payoff_p1 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        payoff_p2 = -payoff_p1
        return cls(payoff_p1, payoff_p2)

    @classmethod
    def battle_of_sexes(cls):
        """Create the Battle of the Sexes game."""
        payoff_p1 = np.array([[2, 0], [0, 1]])
        payoff_p2 = np.array([[1, 0], [0, 2]])
        return cls(payoff_p1, payoff_p2)

    def __repr__(self):
        return f"NormalFormGame(players={self.n_players}, actions={self.n_actions})"

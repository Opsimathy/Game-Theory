"""
Shapley Value for Cooperative Games

The Shapley value is a solution concept for fairly distributing the total
gains from cooperation among players in a coalitional game.

Key Properties (Uniqueness Axioms):
1. Efficiency: Sum of payoffs equals value of grand coalition
2. Symmetry: Identical players receive identical payoffs
3. Dummy player: Player who adds nothing gets nothing
4. Additivity: Value of sum of games equals sum of values

Formula:
    φ_i(v) = Σ_{S⊆N\{i}} |S|!(n-|S|-1)!/n! * [v(S∪{i}) - v(S)]

Interpretation: Average marginal contribution across all orderings

Applications:
- Fair cost/profit allocation
- Voting power indices (Shapley-Shubik index)
- Machine learning (feature importance, SHAP values)
- Network centrality
- Resource allocation

Reference:
- Shapley, L. S. (1953). "A Value for n-Person Games"
- Roth, A. E. (1988). "The Shapley Value: Essays in Honor of Lloyd S. Shapley"
- Winter, E. (2002). "The Shapley Value" in Handbook of Game Theory
- Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP)
"""

import numpy as np
from itertools import combinations, permutations
from typing import Dict, Set, FrozenSet, Callable, List, Tuple
from collections import defaultdict


class CoalitionalGame:
    """
    Represents a coalitional (cooperative) game.

    A coalitional game is defined by:
    - Set of players N = {1, 2, ..., n}
    - Characteristic function v: 2^N → R
      v(S) = value that coalition S can achieve
    """

    def __init__(self, n_players: int, value_function: Callable[[FrozenSet[int]], float]):
        """
        Initialize a coalitional game.

        Args:
            n_players: Number of players
            value_function: Function that maps coalitions to values
        """
        self.n_players = n_players
        self.players = set(range(n_players))
        self.value_function = value_function

        # Cache for computed values
        self._value_cache: Dict[FrozenSet[int], float] = {}

    def value(self, coalition: Set[int]) -> float:
        """
        Get the value of a coalition.

        Args:
            coalition: Set of player indices

        Returns:
            Value that coalition can achieve
        """
        frozen_coalition = frozenset(coalition)

        if frozen_coalition not in self._value_cache:
            self._value_cache[frozen_coalition] = self.value_function(frozen_coalition)

        return self._value_cache[frozen_coalition]

    def is_superadditive(self) -> bool:
        """
        Check if game is superadditive.

        A game is superadditive if v(S ∪ T) ≥ v(S) + v(T) for all disjoint S, T.
        This means cooperation never hurts.

        Returns:
            True if game is superadditive
        """
        # Check all pairs of disjoint coalitions
        for size_s in range(self.n_players):
            for S in combinations(self.players, size_s):
                S_set = set(S)
                remaining = self.players - S_set

                for size_t in range(len(remaining) + 1):
                    for T in combinations(remaining, size_t):
                        T_set = set(T)

                        if not S_set or not T_set:  # Skip empty coalitions
                            continue

                        v_S = self.value(S_set)
                        v_T = self.value(T_set)
                        v_union = self.value(S_set | T_set)

                        if v_union < v_S + v_T - 1e-9:
                            return False

        return True


class ShapleyValue:
    """
    Computes Shapley value for coalitional games.
    """

    def __init__(self, game: CoalitionalGame):
        """
        Initialize Shapley value computer.

        Args:
            game: Coalitional game
        """
        self.game = game

    def compute(self) -> np.ndarray:
        """
        Compute Shapley value for all players.

        Uses the permutation formula:
        φ_i = (1/n!) * Σ_{π} [v(S_i^π ∪ {i}) - v(S_i^π)]

        where S_i^π is the set of players before i in permutation π.

        Returns:
            Shapley values for each player
        """
        n = self.game.n_players
        shapley_values = np.zeros(n)

        # Enumerate all permutations
        all_perms = list(permutations(range(n)))
        n_perms = len(all_perms)

        for perm in all_perms:
            # For each player in this permutation
            for idx, player in enumerate(perm):
                # Players before this player in the permutation
                players_before = set(perm[:idx])

                # Marginal contribution
                v_with = self.game.value(players_before | {player})
                v_without = self.game.value(players_before) if players_before else 0

                marginal_contribution = v_with - v_without

                shapley_values[player] += marginal_contribution

        # Average over all permutations
        shapley_values /= n_perms

        return shapley_values

    def compute_efficient(self) -> np.ndarray:
        """
        Compute Shapley value using the efficient formula.

        φ_i = Σ_{S⊆N\{i}} |S|!(n-|S|-1)!/n! * [v(S∪{i}) - v(S)]

        More efficient than enumerating all permutations for small games.

        Returns:
            Shapley values for each player
        """
        n = self.game.n_players
        shapley_values = np.zeros(n)

        # For each player
        for player in range(n):
            other_players = self.game.players - {player}

            # For each subset of other players
            for size in range(n):
                for subset in combinations(other_players, size):
                    S = set(subset)
                    size_S = len(S)

                    # Weight: |S|!(n-|S|-1)!/n!
                    import math
                    weight = (math.factorial(size_S) *
                             math.factorial(n - size_S - 1) /
                             math.factorial(n))

                    # Marginal contribution
                    v_with = self.game.value(S | {player})
                    v_without = self.game.value(S) if S else 0

                    marginal_contribution = v_with - v_without

                    shapley_values[player] += weight * marginal_contribution

        return shapley_values

    def verify_axioms(self, shapley_values: np.ndarray) -> Dict[str, bool]:
        """
        Verify that computed values satisfy Shapley axioms.

        Args:
            shapley_values: Computed Shapley values

        Returns:
            Dictionary of axiom verification results
        """
        results = {}

        # Efficiency: sum equals value of grand coalition
        grand_coalition_value = self.game.value(self.game.players)
        results['efficiency'] = abs(shapley_values.sum() - grand_coalition_value) < 1e-9

        # Check if any player is a dummy (adds 0 to all coalitions)
        # This is harder to check automatically, skip for now

        return results


# Example games

def glove_game(n_left: int, n_right: int) -> CoalitionalGame:
    """
    Glove game: players have left or right gloves.
    A coalition's value is the number of pairs they can make.

    Args:
        n_left: Number of players with left gloves
        n_right: Number of players with right gloves

    Returns:
        Coalitional game
    """
    n_players = n_left + n_right

    def value_function(coalition: FrozenSet[int]) -> float:
        # Players 0 to n_left-1 have left gloves
        # Players n_left to n_players-1 have right gloves
        left_gloves = sum(1 for p in coalition if p < n_left)
        right_gloves = sum(1 for p in coalition if p >= n_left)
        return min(left_gloves, right_gloves)

    return CoalitionalGame(n_players, value_function)


def weighted_voting_game(quota: int, weights: List[int]) -> CoalitionalGame:
    """
    Weighted voting game.

    A coalition wins (value=1) if total weight ≥ quota, else value=0.
    Used to model voting power (e.g., UN Security Council, EU Council).

    Args:
        quota: Votes needed to win
        weights: Weight (number of votes) for each player

    Returns:
        Coalitional game
    """
    n_players = len(weights)

    def value_function(coalition: FrozenSet[int]) -> float:
        total_weight = sum(weights[p] for p in coalition)
        return 1.0 if total_weight >= quota else 0.0

    return CoalitionalGame(n_players, value_function)


def airport_game(costs: List[float]) -> CoalitionalGame:
    """
    Airport game: planes of different sizes need runways.

    Each plane type needs a runway of certain length (cost).
    Cost-sharing problem: how to split runway construction cost?

    Args:
        costs: Runway cost needed by each plane type (sorted)

    Returns:
        Coalitional game
    """
    n_players = len(costs)

    def value_function(coalition: FrozenSet[int]) -> float:
        if not coalition:
            return 0.0
        # Cost is the maximum runway needed
        return -max(costs[p] for p in coalition)  # Negative because it's a cost

    return CoalitionalGame(n_players, value_function)


if __name__ == "__main__":
    print("="*60)
    print("SHAPLEY VALUE EXAMPLES")
    print("="*60)
    print()

    # Example 1: Glove Game
    print("Example 1: Glove Game (2 left, 1 right)")
    print("-"*60)
    print("Players 0,1 have left gloves; Player 2 has right glove")
    print("Value = number of pairs that can be made")
    print()

    game1 = glove_game(n_left=2, n_right=1)

    print("Some coalition values:")
    print(f"  v({{0}}) = {game1.value({0})}")
    print(f"  v({{2}}) = {game1.value({2})}")
    print(f"  v({{0,2}}) = {game1.value({0, 2})}")
    print(f"  v({{0,1,2}}) = {game1.value({0, 1, 2})}")
    print()

    sv1 = ShapleyValue(game1)
    values1 = sv1.compute_efficient()

    print("Shapley values:")
    print(f"  Player 0 (left): {values1[0]:.3f}")
    print(f"  Player 1 (left): {values1[1]:.3f}")
    print(f"  Player 2 (right): {values1[2]:.3f}")
    print(f"  Sum: {values1.sum():.3f}")

    print("\nInterpretation: Player with right glove has more power")
    print("(they're essential for making pairs)")
    print()

    # Example 2: Weighted Voting
    print("="*60)
    print("Example 2: Weighted Voting Game")
    print("-"*60)
    print("Simplified UN Security Council:")
    print("  3 permanent members (veto power): 7 votes each")
    print("  2 non-permanent members: 1 vote each")
    print("  Quota: 10 votes to pass")
    print()

    weights = [7, 7, 7, 1, 1]  # 3 permanent, 2 non-permanent
    quota = 10
    game2 = weighted_voting_game(quota, weights)

    sv2 = ShapleyValue(game2)
    values2 = sv2.compute_efficient()

    print("Shapley-Shubik power indices:")
    for i in range(5):
        member_type = "Permanent" if i < 3 else "Non-permanent"
        print(f"  Player {i} ({member_type}): {values2[i]:.4f}")

    print(f"  Sum: {values2.sum():.3f}")
    print()

    # Example 3: Airport Game
    print("="*60)
    print("Example 3: Airport Cost Sharing")
    print("-"*60)
    print("3 plane types need runways costing $1M, $2M, $3M")
    print("How to fairly split the $3M cost?")
    print()

    costs = [1.0, 2.0, 3.0]  # in millions
    game3 = airport_game(costs)

    sv3 = ShapleyValue(game3)
    values3 = sv3.compute_efficient()

    print("Shapley cost allocation (millions):")
    for i in range(3):
        print(f"  Plane type {i+1}: ${-values3[i]:.3f}M")

    print(f"  Total: ${-values3.sum():.3f}M")
    print()

    print("="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("✓ Shapley value uniquely characterized by axioms")
    print("✓ Fair allocation based on marginal contributions")
    print("✓ Satisfies efficiency, symmetry, dummy, additivity")
    print("✓ Computationally expensive: O(2^n) coalitions")
    print()
    print("Applications:")
    print("- Cost/profit sharing (airports, utilities)")
    print("- Voting power indices (UN, EU, shareholders)")
    print("- ML interpretability (SHAP values)")
    print("- Network centrality and influence")
    print("- Matching markets and kidney exchange")
    print()
    print("Historical note:")
    print("- Shapley (1953): Introduced the value")
    print("- Won Nobel Prize in Economics (2012)")
    print("- SHAP (2017): Brought Shapley values to ML interpretability")

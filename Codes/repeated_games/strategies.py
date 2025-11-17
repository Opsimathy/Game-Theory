"""
Strategies for Repeated Games

Repeated games model long-term interactions where players can condition
their behavior on past history. This enables cooperation through:
- Reputation and trust building
- Punishment of defectors
- Reciprocity

Classic Application: Iterated Prisoner's Dilemma (IPD)
- Single-shot: mutual defection is Nash equilibrium
- Repeated: cooperation can emerge through strategic behavior

Folk Theorem:
In infinitely repeated games with sufficient patience, any individually
rational payoff can be sustained as a Nash equilibrium through appropriate
punishment strategies.

Famous Strategies:
- Tit-for-Tat: Copy opponent's last move (Axelrod tournaments winner)
- Grim Trigger: Cooperate until opponent defects, then defect forever
- Pavlov (Win-Stay-Lose-Shift): Repeat if did well, change if did poorly

Reference:
- Axelrod, R. (1984). "The Evolution of Cooperation"
- Axelrod, R. (1980). "Effective Choice in the Prisoner's Dilemma"
- Fudenberg & Maskin (1986). "The Folk Theorem in Repeated Games"
- Nowak & Sigmund (1993). "A Strategy of Win-Stay, Lose-Shift"
- Press & Dyson (2012). "Iterated Prisoner's Dilemma contains strategies that dominate"
"""

import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


# Action encoding
COOPERATE = 1
DEFECT = 0


class Strategy(ABC):
    """
    Base class for repeated game strategies.
    """

    def __init__(self, name: str = "Unknown"):
        """
        Initialize strategy.

        Args:
            name: Name of the strategy
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset strategy state for a new game."""
        self.history_self = []
        self.history_opponent = []

    @abstractmethod
    def next_action(self) -> int:
        """
        Decide next action based on history.

        Returns:
            COOPERATE or DEFECT
        """
        pass

    def update_history(self, own_action: int, opponent_action: int) -> None:
        """
        Update history after a round.

        Args:
            own_action: Action taken by this strategy
            opponent_action: Action taken by opponent
        """
        self.history_self.append(own_action)
        self.history_opponent.append(opponent_action)

    def __str__(self) -> str:
        return self.name


class AlwaysCooperate(Strategy):
    """Always cooperates, regardless of opponent's actions."""

    def __init__(self):
        super().__init__("Always Cooperate")

    def next_action(self) -> int:
        return COOPERATE


class AlwaysDefect(Strategy):
    """Always defects, regardless of opponent's actions."""

    def __init__(self):
        super().__init__("Always Defect")

    def next_action(self) -> int:
        return DEFECT


class TitForTat(Strategy):
    """
    Tit-for-Tat: Cooperate initially, then copy opponent's last move.

    Properties:
    - Nice: never defects first
    - Retaliating: punishes defection
    - Forgiving: returns to cooperation if opponent does
    - Clear: easy for opponent to understand

    Winner of Axelrod's tournaments (1980).
    """

    def __init__(self):
        super().__init__("Tit-for-Tat")

    def next_action(self) -> int:
        if len(self.history_opponent) == 0:
            # Cooperate on first move
            return COOPERATE
        else:
            # Copy opponent's last move
            return self.history_opponent[-1]


class GrimTrigger(Strategy):
    """
    Grim Trigger: Cooperate until opponent defects, then defect forever.

    Properties:
    - Nice: never defects first
    - Unforgiving: permanent retaliation
    - Used to enforce cooperation in theory
    - Can sustain cooperation in infinitely repeated games
    """

    def __init__(self):
        super().__init__("Grim Trigger")
        self.triggered = False

    def reset(self) -> None:
        super().reset()
        self.triggered = False

    def next_action(self) -> int:
        if self.triggered:
            return DEFECT

        # Check if opponent ever defected
        if DEFECT in self.history_opponent:
            self.triggered = True
            return DEFECT
        else:
            return COOPERATE


class Pavlov(Strategy):
    """
    Pavlov (Win-Stay, Lose-Shift):
    - If last round was good (both cooperated or both defected), repeat
    - If last round was bad (only one cooperated), switch

    Alternative name: Win-Stay-Lose-Shift
    Introduced by Nowak & Sigmund (1993)

    Properties:
    - Corrects mistakes (unlike Tit-for-Tat)
    - Can achieve mutual cooperation from mutual defection
    - Exploits always-cooperate
    """

    def __init__(self):
        super().__init__("Pavlov")

    def next_action(self) -> int:
        if len(self.history_self) == 0:
            # Cooperate initially
            return COOPERATE

        last_own = self.history_self[-1]
        last_opponent = self.history_opponent[-1]

        # Win-Stay: if both cooperated or both defected, repeat
        # Lose-Shift: otherwise, switch

        if last_own == last_opponent:
            # Both cooperated or both defected -> stay
            return last_own
        else:
            # One cooperated, one defected -> shift
            return 1 - last_own


class RandomStrategy(Strategy):
    """
    Random: Cooperate or defect with equal probability.

    Useful for comparison/benchmarking.
    """

    def __init__(self, p_cooperate: float = 0.5):
        """
        Initialize random strategy.

        Args:
            p_cooperate: Probability of cooperation
        """
        super().__init__("Random")
        self.p_cooperate = p_cooperate

    def next_action(self) -> int:
        return COOPERATE if np.random.random() < self.p_cooperate else DEFECT


class TitForTwoTats(Strategy):
    """
    Tit-for-Two-Tats: More forgiving than Tit-for-Tat.
    Only retaliates after opponent defects twice in a row.
    """

    def __init__(self):
        super().__init__("Tit-for-Two-Tats")

    def next_action(self) -> int:
        if len(self.history_opponent) < 2:
            return COOPERATE

        # Defect only if opponent defected in last two rounds
        if (self.history_opponent[-1] == DEFECT and
            self.history_opponent[-2] == DEFECT):
            return DEFECT
        else:
            return COOPERATE


def play_repeated_game(
    strategy1: Strategy,
    strategy2: Strategy,
    payoff_matrix: np.ndarray,
    n_rounds: int
) -> Tuple[float, float, List[Tuple[int, int]]]:
    """
    Play a repeated game between two strategies.

    Args:
        strategy1: First player's strategy
        strategy2: Second player's strategy
        payoff_matrix: 2x2 payoff matrix [CC, CD; DC, DD]
                      where payoff_matrix[i,j] is payoff when
                      player 1 plays i and player 2 plays j
        n_rounds: Number of rounds to play

    Returns:
        (total_payoff_1, total_payoff_2, history)
    """
    strategy1.reset()
    strategy2.reset()

    total_payoff_1 = 0
    total_payoff_2 = 0
    history = []

    for _ in range(n_rounds):
        # Get actions
        action1 = strategy1.next_action()
        action2 = strategy2.next_action()

        # Get payoffs
        payoff_1 = payoff_matrix[action1, action2]
        payoff_2 = payoff_matrix[action2, action1]  # Symmetric game

        # Update totals
        total_payoff_1 += payoff_1
        total_payoff_2 += payoff_2

        # Update histories
        strategy1.update_history(action1, action2)
        strategy2.update_history(action2, action1)

        # Record
        history.append((action1, action2))

    return total_payoff_1, total_payoff_2, history


def tournament(strategies: List[Strategy], payoff_matrix: np.ndarray, n_rounds: int) -> np.ndarray:
    """
    Run round-robin tournament between strategies.

    Args:
        strategies: List of strategies to compete
        payoff_matrix: Payoff matrix for the game
        n_rounds: Rounds per match

    Returns:
        Score matrix where scores[i,j] is total payoff of strategy i vs strategy j
    """
    n_strategies = len(strategies)
    scores = np.zeros((n_strategies, n_strategies))

    for i in range(n_strategies):
        for j in range(n_strategies):
            if i != j:
                payoff_i, payoff_j, _ = play_repeated_game(
                    strategies[i],
                    strategies[j],
                    payoff_matrix,
                    n_rounds
                )
                scores[i, j] = payoff_i

    return scores


if __name__ == "__main__":
    print("="*60)
    print("REPEATED GAMES: ITERATED PRISONER'S DILEMMA")
    print("="*60)
    print()

    # Prisoner's Dilemma payoff matrix
    # Rows/Cols: Cooperate, Defect
    #              C    D
    #        C   3,3  0,5
    #        D   5,0  1,1

    payoff = np.array([
        [3, 0],  # Cooperate vs (Cooperate, Defect)
        [5, 1],  # Defect vs (Cooperate, Defect)
    ])

    print("Prisoner's Dilemma Payoffs:")
    print("  Mutual cooperation: (3, 3)")
    print("  Mutual defection:   (1, 1)")
    print("  One defects:        (5, 0) or (0, 5)")
    print()

    # Create strategies
    strategies = [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat(),
        GrimTrigger(),
        Pavlov(),
        TitForTwoTats(),
    ]

    n_rounds = 100

    print(f"Round-Robin Tournament ({n_rounds} rounds per match)")
    print("="*60)
    print()

    # Run tournament
    scores = tournament(strategies, payoff, n_rounds)

    # Compute average scores
    avg_scores = scores.sum(axis=1) / (len(strategies) - 1)

    # Sort by average score
    ranking = np.argsort(-avg_scores)

    print("Rankings:")
    for rank, idx in enumerate(ranking):
        strategy = strategies[idx]
        avg_score = avg_scores[idx]
        print(f"  {rank+1}. {strategy.name:20s} - Avg: {avg_score:.2f}")

    print()

    # Detailed matchups
    print("Selected Matchups:")
    print("-"*60)

    matchups = [
        (TitForTat(), AlwaysCooperate()),
        (TitForTat(), AlwaysDefect()),
        (TitForTat(), TitForTat()),
        (Pavlov(), TitForTat()),
        (GrimTrigger(), AlwaysDefect()),
    ]

    for s1, s2 in matchups:
        p1, p2, history = play_repeated_game(s1, s2, payoff, 10)

        # Count cooperation
        coop_count_1 = sum(1 for a, _ in history if a == COOPERATE)
        coop_count_2 = sum(1 for _, a in history if a == COOPERATE)

        print(f"\n{s1.name} vs {s2.name}:")
        print(f"  Scores: {p1:.1f} - {p2:.1f}")
        print(f"  Cooperation: {s1.name}={coop_count_1}/10, {s2.name}={coop_count_2}/10")

    print()
    print("="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("✓ Cooperation can emerge in repeated interactions")
    print("✓ Tit-for-Tat is simple, robust, and successful")
    print("✓ Nice strategies (never defect first) do well")
    print("✓ Forgiveness important (Tit-for-Two-Tats vs Tit-for-Tat)")
    print("✓ Context matters: no universally best strategy")
    print()
    print("Axelrod's Tournaments (1980):")
    print("- Tit-for-Tat won against 14 other strategies")
    print("- Properties: Nice, Retaliating, Forgiving, Clear")
    print("- \"Be nice, but don't be a pushover\"")
    print()
    print("Applications:")
    print("- International relations (cooperation between nations)")
    print("- Business partnerships (repeated contracts)")
    print("- Social norms and reciprocity")
    print("- Biological evolution (vampire bats, cleaner fish)")
    print()
    print("Folk Theorem:")
    print("In infinitely repeated games with sufficient patience,")
    print("any individually rational outcome can be sustained as equilibrium")
    print("through appropriate reward/punishment strategies.")

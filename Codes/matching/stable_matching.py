"""
Stable Matching: Gale-Shapley Algorithm

The stable matching problem asks: given preference lists of two groups,
find a matching such that no pair would prefer to be matched with each other
over their current partners.

Classic Example: Marriage problem with men and women

Key Properties:
- Always produces a stable matching
- Terminates in O(n²) time
- Strategy-proof for proposing side
- Not strategy-proof for receiving side
- Multiple stable matchings may exist

Applications:
- Medical residency matching (NRMP since 1952)
- School choice and college admissions
- Kidney exchange
- Labor market matching

Reference:
- Gale & Shapley (1962). "College Admissions and the Stability of Marriage"
- Roth & Sotomayor (1990). "Two-Sided Matching: A Study in Game-Theoretic Modeling"
- Roth, A. E. (2015). "Who Gets What and Why" (Nobel Prize 2012)
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import deque


class StableMarriage:
    """
    Represents an instance of the stable marriage problem.
    """

    def __init__(
        self,
        men_preferences: List[List[int]],
        women_preferences: List[List[int]]
    ):
        """
        Initialize stable marriage problem.

        Args:
            men_preferences: men_preferences[m] = list of women in order of preference for man m
            women_preferences: women_preferences[w] = list of men in order of preference for woman w
        """
        self.n = len(men_preferences)

        if len(women_preferences) != self.n:
            raise ValueError("Must have equal number of men and women")

        self.men_preferences = [list(prefs) for prefs in men_preferences]
        self.women_preferences = [list(prefs) for prefs in women_preferences]

        # Build ranking dictionaries for efficient lookups
        # men_rankings[m][w] = rank of woman w in man m's preference list
        self.men_rankings = []
        for prefs in men_preferences:
            ranking = {}
            for rank, woman in enumerate(prefs):
                ranking[woman] = rank
            self.men_rankings.append(ranking)

        # women_rankings[w][m] = rank of man m in woman w's preference list
        self.women_rankings = []
        for prefs in women_preferences:
            ranking = {}
            for rank, man in enumerate(prefs):
                ranking[man] = rank
            self.women_rankings.append(ranking)

    def is_stable(self, matching: Dict[int, int]) -> bool:
        """
        Check if a matching is stable.

        A matching is stable if there's no blocking pair:
        a man and woman who both prefer each other to their current partners.

        Args:
            matching: Dictionary mapping men to women

        Returns:
            True if matching is stable
        """
        # Build reverse matching (women to men)
        reverse_matching = {w: m for m, w in matching.items()}

        # Check all pairs
        for man in range(self.n):
            current_woman = matching[man]

            # Check if man prefers any other woman to current partner
            for woman in self.men_preferences[man]:
                # If we've reached current partner, no blocking pair with this man
                if woman == current_woman:
                    break

                # Man prefers 'woman' to current partner
                # Check if woman prefers this man to her current partner
                current_man = reverse_matching[woman]

                woman_rank_current = self.women_rankings[woman][current_man]
                woman_rank_man = self.women_rankings[woman][man]

                if woman_rank_man < woman_rank_current:
                    # Blocking pair found!
                    return False

        return True


class GaleShapley:
    """
    Gale-Shapley algorithm for stable matching.

    Algorithm:
    1. All men are initially unmatched
    2. While there exists an unmatched man who hasn't proposed to all women:
       a. Man proposes to highest-ranked woman he hasn't proposed to
       b. If woman is unmatched, she accepts
       c. If woman is matched, she accepts if she prefers new proposer
          (and her current partner becomes unmatched)
    """

    def __init__(self, problem: StableMarriage):
        """
        Initialize Gale-Shapley solver.

        Args:
            problem: Stable marriage problem instance
        """
        self.problem = problem
        self.n = problem.n

    def solve(self, proposers: str = "men") -> Dict[int, int]:
        """
        Run Gale-Shapley algorithm.

        Args:
            proposers: "men" or "women" - which side proposes

        Returns:
            Stable matching (proposers -> receivers)
        """
        if proposers == "men":
            return self._solve_men_propose()
        else:
            return self._solve_women_propose()

    def _solve_men_propose(self) -> Dict[int, int]:
        """
        Solve with men proposing.

        Returns:
            Matching from men to women
        """
        # Track which woman each man is matched to (-1 if unmatched)
        man_partner = [-1] * self.n
        # Track which man each woman is matched to (-1 if unmatched)
        woman_partner = [-1] * self.n

        # Track next woman to propose to for each man
        next_proposal = [0] * self.n

        # Queue of unmatched men
        free_men = deque(range(self.n))

        while free_men:
            man = free_men.popleft()

            # Man has proposed to all women, must remain unmatched
            # (This shouldn't happen in valid input)
            if next_proposal[man] >= self.n:
                continue

            # Next woman to propose to
            woman = self.problem.men_preferences[man][next_proposal[man]]
            next_proposal[man] += 1

            # If woman is unmatched, accept
            if woman_partner[woman] == -1:
                man_partner[man] = woman
                woman_partner[woman] = man
            else:
                # Woman is matched, check if she prefers new proposer
                current_partner = woman_partner[woman]

                rank_current = self.problem.women_rankings[woman][current_partner]
                rank_new = self.problem.women_rankings[woman][man]

                if rank_new < rank_current:
                    # Woman prefers new proposer
                    # Current partner becomes free
                    free_men.append(current_partner)
                    man_partner[current_partner] = -1

                    # Accept new proposer
                    man_partner[man] = woman
                    woman_partner[woman] = man
                else:
                    # Woman rejects, man remains free
                    free_men.append(man)

        return {m: w for m, w in enumerate(man_partner)}

    def _solve_women_propose(self) -> Dict[int, int]:
        """
        Solve with women proposing.

        Returns:
            Matching from women to men
        """
        # Symmetric to men proposing
        woman_partner = [-1] * self.n
        man_partner = [-1] * self.n

        next_proposal = [0] * self.n
        free_women = deque(range(self.n))

        while free_women:
            woman = free_women.popleft()

            if next_proposal[woman] >= self.n:
                continue

            man = self.problem.women_preferences[woman][next_proposal[woman]]
            next_proposal[woman] += 1

            if man_partner[man] == -1:
                woman_partner[woman] = man
                man_partner[man] = woman
            else:
                current_partner = man_partner[man]
                rank_current = self.problem.men_rankings[man][current_partner]
                rank_new = self.problem.men_rankings[man][woman]

                if rank_new < rank_current:
                    free_women.append(current_partner)
                    woman_partner[current_partner] = -1

                    woman_partner[woman] = man
                    man_partner[man] = woman
                else:
                    free_women.append(woman)

        return {w: m for w, m in enumerate(woman_partner)}


if __name__ == "__main__":
    print("="*60)
    print("STABLE MATCHING: GALE-SHAPLEY ALGORITHM")
    print("="*60)
    print()

    # Example 1: Small instance
    print("Example 1: 3 men, 3 women")
    print("-"*60)

    # Men's preferences (each man ranks all women)
    men_prefs = [
        [0, 1, 2],  # Man 0 prefers: Woman 0 > Woman 1 > Woman 2
        [1, 0, 2],  # Man 1 prefers: Woman 1 > Woman 0 > Woman 2
        [0, 1, 2],  # Man 2 prefers: Woman 0 > Woman 1 > Woman 2
    ]

    # Women's preferences
    women_prefs = [
        [1, 0, 2],  # Woman 0 prefers: Man 1 > Man 0 > Man 2
        [0, 1, 2],  # Woman 1 prefers: Man 0 > Man 1 > Man 2
        [0, 1, 2],  # Woman 2 prefers: Man 0 > Man 1 > Man 2
    ]

    print("Men's preferences:")
    for m, prefs in enumerate(men_prefs):
        print(f"  Man {m}: {' > '.join(f'W{w}' for w in prefs)}")

    print("\nWomen's preferences:")
    for w, prefs in enumerate(women_prefs):
        print(f"  Woman {w}: {' > '.join(f'M{m}' for m in prefs)}")

    print()

    problem = StableMarriage(men_prefs, women_prefs)
    solver = GaleShapley(problem)

    # Men propose
    print("Matching (Men propose):")
    matching_men_propose = solver.solve(proposers="men")
    for m, w in matching_men_propose.items():
        print(f"  Man {m} - Woman {w}")

    is_stable_men = problem.is_stable(matching_men_propose)
    print(f"  Is stable: {is_stable_men}")

    print()

    # Women propose
    print("Matching (Women propose):")
    matching_women_propose = solver.solve(proposers="women")

    # Convert to men->women format for display
    matching_women_reverse = {w: m for m, w in matching_women_propose.items()}
    for m in range(3):
        w = matching_women_reverse.get(m, -1)
        if w != -1:
            print(f"  Man {m} - Woman {w}")

    is_stable_women = problem.is_stable(matching_women_reverse)
    print(f"  Is stable: {is_stable_women}")

    print()

    # Compare matchings
    if matching_men_propose != matching_women_reverse:
        print("Note: Different matchings (both stable)!")
        print("This demonstrates that multiple stable matchings can exist.")
    else:
        print("Both proposing sides yield the same matching.")

    print()

    # Example 2: Illustrate strategic incentives
    print("="*60)
    print("Example 2: Strategic Manipulation")
    print("-"*60)
    print("Showing that receiving side can benefit from misreporting")
    print()

    # Simple example where woman can benefit from lying
    men_prefs2 = [
        [0, 1],  # Man 0: W0 > W1
        [0, 1],  # Man 1: W0 > W1
    ]

    women_prefs2 = [
        [1, 0],  # Woman 0: M1 > M0 (true preferences)
        [0, 1],  # Woman 1: M0 > M1
    ]

    print("True preferences:")
    print("  Man 0: W0 > W1")
    print("  Man 1: W0 > W1")
    print("  Woman 0: M1 > M0")
    print("  Woman 1: M0 > M1")
    print()

    problem2 = StableMarriage(men_prefs2, women_prefs2)
    solver2 = GaleShapley(problem2)
    matching2_true = solver2.solve(proposers="men")

    print("Matching (truthful):")
    for m, w in matching2_true.items():
        print(f"  Man {m} - Woman {w}")
    print()

    # Woman 0 lies about preferences
    women_prefs2_lie = [
        [0, 1],  # Woman 0 LIES: claims M0 > M1 (actually prefers M1)
        [0, 1],
    ]

    problem2_lie = StableMarriage(men_prefs2, women_prefs2_lie)
    solver2_lie = GaleShapley(problem2_lie)
    matching2_lie = solver2_lie.solve(proposers="men")

    print("Matching (Woman 0 misreports):")
    for m, w in matching2_lie.items():
        print(f"  Man {m} - Woman {w}")

    print()
    if matching2_true != matching2_lie:
        print("Woman 0 benefits from lying!")
        print("She gets her preferred partner M1 instead of M0")

    print()
    print("="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("✓ Gale-Shapley always finds a stable matching")
    print("✓ Runs in O(n²) time (worst case: all men propose to all women)")
    print("✓ Strategy-proof for proposing side (truthful is optimal)")
    print("✗ NOT strategy-proof for receiving side (can benefit from lying)")
    print("✓ Proposing side gets best possible stable match")
    print("✓ Receiving side gets worst possible stable match")
    print()
    print("Applications:")
    print("- Medical Residency Matching (NRMP): ~40,000 residents/year since 1952")
    print("- School Choice: Boston, NYC school assignment")
    print("- College Admissions: in many countries")
    print("- Kidney Exchange: matching donors and recipients")
    print()
    print("Nobel Prize in Economics (2012):")
    print("- Alvin Roth & Lloyd Shapley")
    print("- For theory and practice of stable allocations")

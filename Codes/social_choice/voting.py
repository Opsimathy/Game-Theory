"""
Voting Systems and Social Choice

Implementation of various voting rules and their properties.

Classic Results:
- Arrow's Impossibility Theorem: No voting system can satisfy all desirable properties
- Gibbard-Satterthwaite: All non-dictatorial voting rules are manipulable
- Condorcet Paradox: Majority preferences can be cyclic

Voting Rules Implemented:
1. Plurality (First-Past-the-Post)
2. Borda Count
3. Approval Voting
4. Instant Runoff (IRV/Alternative Vote)
5. Condorcet Methods

Reference:
- Arrow, K. (1951). "Social Choice and Individual Values"
- Gibbard, A. (1973). "Manipulation of Voting Schemes"
- Brandt et al. (2016). "Handbook of Computational Social Choice"
"""

from typing import List, Dict, Optional, Set, Tuple
from collections import Counter, defaultdict


def plurality_vote(preferences: List[List[int]]) -> int:
    """
    Plurality (First-Past-the-Post) voting.

    Each voter's top choice gets 1 point.
    Winner is candidate with most first-place votes.

    Pros:
    - Simple to understand and implement
    - Fast to compute

    Cons:
    - Can violate majority criterion
    - Susceptible to vote splitting
    - Can elect candidate disliked by majority

    Args:
        preferences: List of preference orderings (each is list of candidates)
                    preferences[i][0] is voter i's top choice

    Returns:
        Winning candidate ID

    Example:
        >>> preferences = [[0, 1, 2], [1, 0, 2], [1, 2, 0]]
        >>> plurality_vote(preferences)
        1  # Candidate 1 has 2 first-place votes
    """
    if not preferences:
        raise ValueError("No preferences provided")

    first_choices = [pref[0] for pref in preferences]
    vote_counts = Counter(first_choices)

    # Return candidate with most votes
    return vote_counts.most_common(1)[0][0]


def borda_count(preferences: List[List[int]]) -> int:
    """
    Borda Count voting.

    Points based on position in each ranking:
    - Last place: 0 points
    - Second to last: 1 point
    - ...
    - First place: (n-1) points

    Winner is candidate with most total points.

    Pros:
    - Considers full preference ordering
    - Less susceptible to vote splitting

    Cons:
    - Can violate Condorcet criterion
    - Susceptible to "cloning" manipulation
    - Can be strategically manipulated

    Args:
        preferences: List of preference orderings

    Returns:
        Winning candidate ID

    Example:
        >>> preferences = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
        >>> borda_count(preferences)
        1  # Candidate 1 has most points
    """
    if not preferences:
        raise ValueError("No preferences provided")

    n_candidates = len(preferences[0])
    scores = defaultdict(int)

    for pref in preferences:
        for rank, candidate in enumerate(pref):
            # Score = n_candidates - rank - 1
            # First place (rank 0) gets n_candidates - 1 points
            scores[candidate] += n_candidates - rank - 1

    return max(scores, key=scores.get)


def approval_voting(approvals: List[Set[int]]) -> int:
    """
    Approval Voting.

    Each voter approves any number of candidates.
    Winner is candidate approved by most voters.

    Pros:
    - Simple and expressive
    - Resistant to vote splitting
    - Can elect broadly acceptable candidates

    Cons:
    - Requires voters to make approval/disapproval decision
    - Strategic voting still possible

    Args:
        approvals: List of approval sets (each voter approves subset of candidates)

    Returns:
        Winning candidate ID

    Example:
        >>> approvals = [{0, 1}, {1, 2}, {1, 2}]
        >>> approval_voting(approvals)
        1  # Candidate 1 approved by all 3 voters
    """
    if not approvals:
        raise ValueError("No approvals provided")

    approval_counts = Counter()
    for approval_set in approvals:
        for candidate in approval_set:
            approval_counts[candidate] += 1

    return approval_counts.most_common(1)[0][0]


def instant_runoff(preferences: List[List[int]]) -> int:
    """
    Instant Runoff Voting (IRV) / Alternative Vote.

    Iterative elimination process:
    1. Count first-place votes
    2. If someone has majority, they win
    3. Otherwise, eliminate candidate with fewest first-place votes
    4. Redistribute their votes to next choice
    5. Repeat

    Pros:
    - Ensures winner has majority support
    - Reduces strategic voting incentives
    - Eliminates vote splitting

    Cons:
    - Can violate monotonicity (getting more votes can cause you to lose)
    - Computationally more complex
    - Can eliminate Condorcet winner

    Args:
        preferences: List of preference orderings

    Returns:
        Winning candidate ID

    Example:
        >>> preferences = [[0, 1, 2], [1, 2, 0], [2, 1, 0], [1, 0, 2]]
        >>> instant_runoff(preferences)
        1  # After eliminating candidate with fewest votes
    """
    if not preferences:
        raise ValueError("No preferences provided")

    # Create mutable copy of preferences
    active_preferences = [list(pref) for pref in preferences]
    n_voters = len(preferences)
    eliminated = set()

    while True:
        # Count first-place votes among non-eliminated candidates
        first_place_votes = defaultdict(int)

        for pref in active_preferences:
            # Find first non-eliminated candidate
            for candidate in pref:
                if candidate not in eliminated:
                    first_place_votes[candidate] += 1
                    break

        if not first_place_votes:
            raise ValueError("No candidates remaining")

        # Check for majority winner
        for candidate, votes in first_place_votes.items():
            if votes > n_voters / 2:
                return candidate

        # Eliminate candidate with fewest votes
        min_votes = min(first_place_votes.values())
        # Break ties by choosing smallest ID
        to_eliminate = min(
            candidate for candidate, votes in first_place_votes.items()
            if votes == min_votes
        )

        eliminated.add(to_eliminate)

        # If only one candidate left, they win
        remaining = set(first_place_votes.keys()) - eliminated
        if len(remaining) == 1:
            return remaining.pop()


def condorcet_winner(preferences: List[List[int]]) -> Optional[int]:
    """
    Find Condorcet winner if one exists.

    A Condorcet winner beats every other candidate in pairwise majority vote.

    Properties:
    - Condorcet winner doesn't always exist (Condorcet paradox)
    - When it exists, it's a strong notion of winner
    - Many voting rules can fail to elect Condorcet winner

    Args:
        preferences: List of preference orderings

    Returns:
        Condorcet winner ID, or None if no Condorcet winner exists

    Example:
        >>> # A beats B beats C beats A (cycle - no Condorcet winner)
        >>> preferences = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        >>> condorcet_winner(preferences)
        None
    """
    if not preferences:
        return None

    # Get all candidates
    candidates = set()
    for pref in preferences:
        candidates.update(pref)

    # Compute pairwise victories
    # wins[a][b] = number of voters who prefer a to b
    wins = defaultdict(lambda: defaultdict(int))

    for pref in preferences:
        # For each pair of candidates
        for i, cand_a in enumerate(pref):
            for cand_b in pref[i+1:]:
                # cand_a is preferred to cand_b
                wins[cand_a][cand_b] += 1

    # Find candidate who beats all others
    for candidate in candidates:
        is_condorcet_winner = True

        for opponent in candidates:
            if opponent == candidate:
                continue

            # Check if candidate beats opponent
            if wins[candidate][opponent] <= wins[opponent][candidate]:
                is_condorcet_winner = False
                break

        if is_condorcet_winner:
            return candidate

    return None


if __name__ == "__main__":
    print("=== Voting Systems Comparison ===\n")

    # Example election with 3 candidates, 9 voters
    preferences = [
        [0, 1, 2],  # 4 voters: A > B > C
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [1, 2, 0],  # 3 voters: B > C > A
        [1, 2, 0],
        [1, 2, 0],
        [2, 1, 0],  # 2 voters: C > B > A
        [2, 1, 0],
    ]

    candidate_names = {0: 'A', 1: 'B', 2: 'C'}

    print("Voter preferences:")
    print("4 voters: A > B > C")
    print("3 voters: B > C > A")
    print("2 voters: C > B > A")
    print()

    # Plurality
    winner_plurality = plurality_vote(preferences)
    print(f"Plurality winner: {candidate_names[winner_plurality]}")

    # Borda
    winner_borda = borda_count(preferences)
    print(f"Borda count winner: {candidate_names[winner_borda]}")

    # Approval (assume voters approve top 2)
    approvals = [{0, 1}, {0, 1}, {0, 1}, {0, 1},
                 {1, 2}, {1, 2}, {1, 2},
                 {2, 1}, {2, 1}]
    winner_approval = approval_voting(approvals)
    print(f"Approval voting winner: {candidate_names[winner_approval]}")

    # IRV
    winner_irv = instant_runoff(preferences)
    print(f"Instant runoff winner: {candidate_names[winner_irv]}")

    # Condorcet
    winner_condorcet = condorcet_winner(preferences)
    if winner_condorcet is not None:
        print(f"Condorcet winner: {candidate_names[winner_condorcet]}")
    else:
        print("Condorcet winner: None (cycle exists)")

    print("\n=== Condorcet Paradox Example ===")
    paradox_prefs = [
        [0, 1, 2],  # A > B > C
        [1, 2, 0],  # B > C > A
        [2, 0, 1],  # C > A > B
    ]

    print("3 voters with preferences:")
    print("Voter 1: A > B > C")
    print("Voter 2: B > C > A")
    print("Voter 3: C > A > B")
    print()
    print("Pairwise comparisons:")
    print("A vs B: A wins 2-1")
    print("B vs C: B wins 2-1")
    print("C vs A: C wins 2-1")
    print("Result: Cycle (A > B > C > A)")
    print(f"Condorcet winner: {condorcet_winner(paradox_prefs)}")

    print("\n=== Key Theorems ===")
    print("Arrow's Impossibility Theorem:")
    print("  No voting system can simultaneously satisfy:")
    print("  1. Unanimity (if everyone prefers A to B, A should win)")
    print("  2. Independence of Irrelevant Alternatives")
    print("  3. Non-dictatorship")
    print()
    print("Gibbard-Satterthwaite Theorem:")
    print("  Every non-dictatorial voting rule with ≥3 candidates")
    print("  is manipulable (strategic voting can change outcome)")

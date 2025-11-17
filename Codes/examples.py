"""
Comprehensive Examples for Game Theory Algorithms

This file demonstrates usage of all implemented algorithms.
"""

import numpy as np
import sys

# Game representations
from games.normal_form import NormalFormGame
from games.extensive_form import ExtensiveFormGame

# Equilibrium solvers
from equilibrium.support_enumeration import SupportEnumerationSolver
from equilibrium.linear_programming import LinearProgrammingSolver

# Regret minimization
from regret_minimization.regret_matching import self_play_regret_matching
from regret_minimization.cfr import CFRSolver

# Auctions
from auctions.single_item import FirstPriceAuction, SecondPriceAuction
from auctions.vcg import VCGMechanism

# Social choice
from social_choice.voting import (
    plurality_vote, borda_count, approval_voting,
    instant_runoff, condorcet_winner
)

# Learning
from learning.fictitious_play import FictitiousPlay
from learning.q_learning import QLearningAgent


def example_nash_equilibrium():
    """Demonstrate Nash equilibrium computation."""
    print("="*60)
    print("EXAMPLE 1: Computing Nash Equilibria")
    print("="*60)
    print()

    # Create Battle of the Sexes game
    game = NormalFormGame.battle_of_sexes()
    print("Game: Battle of the Sexes")
    print("Player 1 payoffs:")
    print(game.payoffs[0])
    print("Player 2 payoffs:")
    print(game.payoffs[1])
    print()

    # Solve using support enumeration
    solver = SupportEnumerationSolver(game)
    equilibria = solver.solve()

    print(f"Found {len(equilibria)} Nash equilibria:")
    for i, (s1, s2) in enumerate(equilibria):
        print(f"\nEquilibrium {i+1}:")
        print(f"  Player 1: {s1}")
        print(f"  Player 2: {s2}")
        payoffs = game.get_expected_payoff([s1, s2])
        print(f"  Expected payoffs: {payoffs}")

    print()


def example_regret_minimization():
    """Demonstrate regret matching."""
    print("="*60)
    print("EXAMPLE 2: Regret Matching")
    print("="*60)
    print()

    game = NormalFormGame.rock_paper_scissors()
    print("Game: Rock-Paper-Scissors")
    print("Training regret matching for 50000 iterations...")
    print()

    strategies = self_play_regret_matching(game, iterations=50000)

    print("Converged strategies:")
    print(f"Player 1: {strategies[0]}")
    print(f"Player 2: {strategies[1]}")
    print(f"Expected payoffs: {game.get_expected_payoff(strategies)}")
    print()


def example_cfr():
    """Demonstrate CFR on Kuhn Poker."""
    print("="*60)
    print("EXAMPLE 3: Counterfactual Regret Minimization (CFR)")
    print("="*60)
    print()

    game = ExtensiveFormGame.kuhn_poker()
    print("Game: Kuhn Poker (simplified)")
    print("Training CFR for 10000 iterations...")
    print()

    solver = CFRSolver(game)
    strategies = solver.train(iterations=10000)

    print("Learned strategies:")
    for infoset, strategy in strategies.items():
        actions = game.get_information_set_actions(infoset)
        print(f"\n{infoset}:")
        for action, prob in zip(actions, strategy):
            print(f"  {action}: {prob:.3f}")

    print()


def example_auctions():
    """Demonstrate auction mechanisms."""
    print("="*60)
    print("EXAMPLE 4: Auction Mechanisms")
    print("="*60)
    print()

    values = [100, 80, 60]
    print(f"Bidders have values: {values}")
    print()

    # Second-price auction
    print("Second-Price (Vickrey) Auction:")
    auction = SecondPriceAuction()
    for i, value in enumerate(values):
        auction.submit_bid(i, value, value)  # Truthful bidding

    winner, payment, utility = auction.run_auction()
    print(f"  Winner: Bidder {winner}")
    print(f"  Payment: ${payment:.2f}")
    print(f"  Winner's utility: ${utility:.2f}")
    print()

    # VCG mechanism
    print("VCG Mechanism (2 items, 3 bidders):")
    mechanism = VCGMechanism(items={0, 1})

    # Bidder valuations
    mechanism.submit_bid(1, {0}, 50)
    mechanism.submit_bid(1, {1}, 30)
    mechanism.submit_bid(2, {0}, 40)
    mechanism.submit_bid(2, {1}, 45)
    mechanism.submit_bid(3, {0, 1}, 80)

    allocation, payments = mechanism.run_mechanism()
    print("  Allocation:")
    for bidder, items in allocation.items():
        print(f"    Bidder {bidder}: items {items}")
    print("  Payments:")
    for bidder, payment in payments.items():
        print(f"    Bidder {bidder}: ${payment:.2f}")
    print()


def example_voting():
    """Demonstrate voting systems."""
    print("="*60)
    print("EXAMPLE 5: Voting Systems")
    print("="*60)
    print()

    # Election with 3 candidates
    preferences = [
        [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],  # 4 voters: A > B > C
        [1, 2, 0], [1, 2, 0], [1, 2, 0],              # 3 voters: B > C > A
        [2, 1, 0], [2, 1, 0],                         # 2 voters: C > B > A
    ]

    candidate_names = {0: 'A', 1: 'B', 2: 'C'}

    print("Voter preferences:")
    print("  4 voters: A > B > C")
    print("  3 voters: B > C > A")
    print("  2 voters: C > B > A")
    print()

    print("Results under different voting systems:")
    print(f"  Plurality: {candidate_names[plurality_vote(preferences)]}")
    print(f"  Borda Count: {candidate_names[borda_count(preferences)]}")

    winner_cond = condorcet_winner(preferences)
    if winner_cond is not None:
        print(f"  Condorcet: {candidate_names[winner_cond]}")
    else:
        print(f"  Condorcet: No winner (cycle)")

    print()


def example_learning():
    """Demonstrate learning algorithms."""
    print("="*60)
    print("EXAMPLE 6: Learning Algorithms")
    print("="*60)
    print()

    game = NormalFormGame.matching_pennies()
    print("Game: Matching Pennies")
    print("Running Fictitious Play for 1000 iterations...")
    print()

    fp = FictitiousPlay(game)
    strategies = fp.train(1000)

    print("Empirical frequencies:")
    print(f"  Player 1: {strategies[0]}")
    print(f"  Player 2: {strategies[1]}")
    print(f"  Is Nash: {game.is_nash_equilibrium(strategies)}")
    print()


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*60)
    print("GAME THEORY ALGORITHMS - COMPREHENSIVE EXAMPLES")
    print("="*60)
    print()

    try:
        example_nash_equilibrium()
    except Exception as e:
        print(f"Error in Nash equilibrium example: {e}\n")

    try:
        example_regret_minimization()
    except Exception as e:
        print(f"Error in regret minimization example: {e}\n")

    try:
        example_cfr()
    except Exception as e:
        print(f"Error in CFR example: {e}\n")

    try:
        example_auctions()
    except Exception as e:
        print(f"Error in auctions example: {e}\n")

    try:
        example_voting()
    except Exception as e:
        print(f"Error in voting example: {e}\n")

    try:
        example_learning()
    except Exception as e:
        print(f"Error in learning example: {e}\n")

    print("="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    run_all_examples()

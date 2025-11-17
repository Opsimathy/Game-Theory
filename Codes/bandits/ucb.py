"""
Upper Confidence Bound (UCB) Algorithms for Multi-Armed Bandits

The multi-armed bandit problem models the exploration-exploitation tradeoff:
- K slot machines (arms) with unknown reward distributions
- Goal: maximize total reward over T rounds
- Tradeoff: exploit best-known arm vs explore to learn

UCB algorithms select arms based on:
- Estimated mean reward (exploitation)
- Confidence bound (exploration)

UCB1 Formula:
    Select arm i = argmax_j [μ_j + sqrt(2 ln t / n_j)]

where:
- μ_j = empirical mean reward of arm j
- t = total number of rounds
- n_j = number of times arm j was played

Key Properties:
- Logarithmic regret: O(log T)
- No-regret learning
- Optimal up to constant factors

Applications:
- Clinical trials (adaptive experimentation)
- Online advertising (ad selection)
- Recommendation systems
- Hyperparameter tuning (Bayesian optimization)
- Resource allocation

Reference:
- Auer et al. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem"
- Lai & Robbins (1985). "Asymptotically Efficient Adaptive Allocation Rules"
- Bubeck & Cesa-Bianchi (2012). "Regret Analysis of Stochastic and Nonstochastic MAB"
"""

import numpy as np
from typing import List, Tuple, Callable


class UCB:
    """
    Base class for UCB algorithms.
    """

    def __init__(self, n_arms: int, exploration_constant: float = 2.0):
        """
        Initialize UCB algorithm.

        Args:
            n_arms: Number of arms
            exploration_constant: Exploration parameter (typically sqrt(2))
        """
        self.n_arms = n_arms
        self.c = exploration_constant

        # Empirical means
        self.means = np.zeros(n_arms)

        # Number of times each arm was pulled
        self.counts = np.zeros(n_arms)

        # Total number of rounds
        self.t = 0

    def select_arm(self) -> int:
        """
        Select an arm to pull.

        Returns:
            Arm index
        """
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        """
        Update statistics after pulling an arm.

        Args:
            arm: Arm that was pulled
            reward: Observed reward
        """
        self.counts[arm] += 1
        self.t += 1

        # Update empirical mean incrementally
        n = self.counts[arm]
        old_mean = self.means[arm]
        self.means[arm] = old_mean + (reward - old_mean) / n


class UCB1(UCB):
    """
    UCB1 algorithm.

    Selects arm with highest upper confidence bound:
        UCB(i) = μ_i + c * sqrt(ln t / n_i)
    """

    def select_arm(self) -> int:
        """
        Select arm using UCB1 rule.

        Returns:
            Arm index
        """
        # Initially, pull each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # Compute UCB for each arm
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            exploitation = self.means[arm]
            exploration = np.sqrt(np.log(self.t) / self.counts[arm])
            ucb_values[arm] = exploitation + self.c * exploration

        return int(np.argmax(ucb_values))


def simulate_ucb(
    reward_functions: List[Callable[[], float]],
    n_rounds: int,
    exploration_constant: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate UCB1 algorithm.

    Args:
        reward_functions: List of functions that return rewards for each arm
        n_rounds: Number of rounds to simulate
        exploration_constant: UCB exploration parameter

    Returns:
        (cumulative_rewards, cumulative_regret, final_regret)
    """
    n_arms = len(reward_functions)

    # Compute optimal arm (highest expected reward)
    true_means = np.array([
        np.mean([reward_functions[i]() for _ in range(10000)])
        for i in range(n_arms)
    ])
    optimal_arm = np.argmax(true_means)
    optimal_mean = true_means[optimal_arm]

    # Run UCB1
    ucb = UCB1(n_arms, exploration_constant)

    cumulative_reward = np.zeros(n_rounds)
    cumulative_regret = np.zeros(n_rounds)

    total_reward = 0
    total_regret = 0

    for t in range(n_rounds):
        # Select and pull arm
        arm = ucb.select_arm()
        reward = reward_functions[arm]()

        # Update
        ucb.update(arm, reward)

        # Track metrics
        total_reward += reward
        total_regret += optimal_mean - true_means[arm]

        cumulative_reward[t] = total_reward
        cumulative_regret[t] = total_regret

    return cumulative_reward, cumulative_regret, total_regret


if __name__ == "__main__":
    print("="*60)
    print("UCB (UPPER CONFIDENCE BOUND) ALGORITHMS")
    print("="*60)
    print()

    # Example: 3 arms with different reward distributions
    print("Example: 3 slot machines")
    print("-"*60)

    # Arm 0: Bernoulli(0.3)
    # Arm 1: Bernoulli(0.5)  <- Optimal
    # Arm 2: Bernoulli(0.4)

    reward_functions = [
        lambda: np.random.binomial(1, 0.3),  # Arm 0
        lambda: np.random.binomial(1, 0.5),  # Arm 1 (best)
        lambda: np.random.binomial(1, 0.4),  # Arm 2
    ]

    print("True reward probabilities:")
    print("  Arm 0: 0.3")
    print("  Arm 1: 0.5 (optimal)")
    print("  Arm 2: 0.4")
    print()

    n_rounds = 1000
    print(f"Running UCB1 for {n_rounds} rounds...")
    print()

    np.random.seed(42)
    cum_reward, cum_regret, final_regret = simulate_ucb(
        reward_functions,
        n_rounds,
        exploration_constant=2.0
    )

    # Run UCB1
    ucb = UCB1(n_arms=3)

    arm_counts = np.zeros(3)

    for t in range(n_rounds):
        arm = ucb.select_arm()
        reward = reward_functions[arm]()
        ucb.update(arm, reward)
        arm_counts[arm] += 1

    print("Final statistics:")
    print(f"  Total reward: {cum_reward[-1]:.1f}")
    print(f"  Total regret: {cum_regret[-1]:.1f}")
    print(f"  Average regret per round: {cum_regret[-1]/n_rounds:.4f}")
    print()

    print("Empirical means:")
    for arm in range(3):
        print(f"  Arm {arm}: {ucb.means[arm]:.3f} (pulled {int(ucb.counts[arm])} times)")

    print()

    print("Pull distribution:")
    for arm in range(3):
        percentage = arm_counts[arm] / n_rounds * 100
        print(f"  Arm {arm}: {int(arm_counts[arm])} pulls ({percentage:.1f}%)")

    print()

    # Demonstrate regret growth
    print("="*60)
    print("REGRET ANALYSIS")
    print("="*60)
    print()

    print("Cumulative regret at different time steps:")
    checkpoints = [100, 250, 500, 750, 1000]
    for t in checkpoints:
        regret = cum_regret[t-1]
        avg_regret = regret / t
        print(f"  t={t:4d}: Regret={regret:6.1f}, Avg={avg_regret:.4f}")

    print()
    print("Note: Regret grows logarithmically O(log T)")
    print("This is optimal for stochastic bandits!")

    print()
    print("="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("✓ UCB1 achieves logarithmic regret: O(log T)")
    print("✓ Optimal up to constant factors")
    print("✓ No-regret learning (average regret → 0)")
    print("✓ Balances exploration and exploitation automatically")
    print("✓ No tuning needed (unlike ε-greedy)")
    print()
    print("Applications:")
    print("- Clinical trials: adaptive patient allocation")
    print("- Online advertising: ad selection")
    print("- Recommendation systems: content selection")
    print("- Hyperparameter tuning: Bayesian optimization")
    print("- Network routing: path selection")
    print()
    print("Variants:")
    print("- UCB-V: accounts for variance")
    print("- KL-UCB: uses KL divergence")
    print("- Bayes-UCB: Bayesian version")
    print("- Thompson Sampling: Bayesian alternative")

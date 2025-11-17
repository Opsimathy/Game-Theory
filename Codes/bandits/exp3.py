"""
EXP3: Exponential-weight algorithm for Exploration and Exploitation

EXP3 is a bandit algorithm for adversarial (non-stochastic) settings where
rewards can be chosen adversarially.

Key Differences from UCB:
- UCB assumes stochastic rewards
- EXP3 works even if rewards are adversarially chosen
- EXP3 uses exponential weighting of estimated rewards

Algorithm:
1. Maintain weights w_i for each arm
2. Select arm i with probability proportional to w_i
3. Update weights: w_i ← w_i * exp(γ * r̃_i / p_i)

where r̃_i is the estimated reward (importance-weighted)

Regret Bound:
- O(sqrt(K T log K)) worst-case regret
- Minimax optimal for adversarial bandits

Applications:
- Online advertising with adversarial users
- Game playing against adaptive opponents
- Cybersecurity (attacker adapts to defender)
- Portfolio selection with adversarial markets

Reference:
- Auer et al. (2002). "The Nonstochastic Multiarmed Bandit Problem"
- Bubeck & Cesa-Bianchi (2012). "Regret Analysis of Stochastic and Nonstochastic MAB"
- Cesa-Bianchi & Lugosi (2006). "Prediction, Learning, and Games"
"""

import numpy as np
from typing import List, Tuple, Callable, Optional


class EXP3:
    """
    EXP3 algorithm for adversarial multi-armed bandits.

    Uses exponential weighting to select arms based on past performance.
    """

    def __init__(self, n_arms: int, gamma: Optional[float] = None):
        """
        Initialize EXP3 algorithm.

        Args:
            n_arms: Number of arms
            gamma: Exploration parameter. If None, uses theory-optimal value.
        """
        self.n_arms = n_arms

        # Weights for each arm
        self.weights = np.ones(n_arms)

        # Exploration parameter
        if gamma is None:
            # Theory-optimal: γ = min(1, sqrt(K log K / (e-1) T))
            # But T is unknown, so use a reasonable default
            self.gamma = min(1.0, np.sqrt(n_arms * np.log(n_arms) / 1000))
        else:
            self.gamma = gamma

        # Total rounds
        self.t = 0

        # Track cumulative rewards (for analysis)
        self.cumulative_rewards = np.zeros(n_arms)

    def get_probabilities(self) -> np.ndarray:
        """
        Compute probability distribution over arms.

        Returns:
            Probability of selecting each arm
        """
        # Mix uniform distribution with weighted distribution
        n = self.n_arms
        weighted_probs = self.weights / self.weights.sum()
        probs = (1 - self.gamma) * weighted_probs + self.gamma / n

        return probs

    def select_arm(self) -> int:
        """
        Select an arm according to current probabilities.

        Returns:
            Selected arm index
        """
        probs = self.get_probabilities()
        return np.random.choice(self.n_arms, p=probs)

    def update(self, arm: int, reward: float) -> None:
        """
        Update weights after observing reward.

        Args:
            arm: Arm that was pulled
            reward: Observed reward (assumed in [0, 1])
        """
        probs = self.get_probabilities()

        # Importance-weighted reward estimate
        estimated_reward = reward / probs[arm]

        # Update weight (exponential weighting)
        self.weights[arm] *= np.exp(self.gamma * estimated_reward / self.n_arms)

        # Track for analysis
        self.cumulative_rewards[arm] += reward
        self.t += 1


def simulate_exp3_adversarial(
    n_arms: int,
    n_rounds: int,
    adversary: Callable[[int, np.ndarray], np.ndarray],
    gamma: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Simulate EXP3 against an adversarial reward function.

    Args:
        n_arms: Number of arms
        n_rounds: Number of rounds
        adversary: Function(round, history) -> rewards for this round
        gamma: EXP3 exploration parameter

    Returns:
        (cumulative_rewards, final_cumulative_reward)
    """
    exp3 = EXP3(n_arms, gamma)

    cumulative_reward = np.zeros(n_rounds)
    total_reward = 0

    arm_history = []

    for t in range(n_rounds):
        # Get adversarial rewards for this round
        rewards = adversary(t, np.array(arm_history))

        # Select and pull arm
        arm = exp3.select_arm()
        reward = rewards[arm]

        # Update
        exp3.update(arm, reward)

        # Track
        arm_history.append(arm)
        total_reward += reward
        cumulative_reward[t] = total_reward

    return cumulative_reward, total_reward


if __name__ == "__main__":
    print("="*60)
    print("EXP3 (EXPONENTIAL-WEIGHT ALGORITHM)")
    print("="*60)
    print()

    # Example 1: Stochastic rewards (EXP3 still works)
    print("Example 1: Stochastic Setting")
    print("-"*60)

    n_arms = 3
    n_rounds = 1000

    # Stochastic adversary (same as UCB example)
    def stochastic_adversary(t, history):
        return np.array([
            np.random.binomial(1, 0.3),  # Arm 0
            np.random.binomial(1, 0.5),  # Arm 1 (best)
            np.random.binomial(1, 0.4),  # Arm 2
        ])

    print("Reward distributions:")
    print("  Arm 0: Bernoulli(0.3)")
    print("  Arm 1: Bernoulli(0.5) (best)")
    print("  Arm 2: Bernoulli(0.4)")
    print()

    np.random.seed(42)
    cum_rewards, total_reward = simulate_exp3_adversarial(
        n_arms,
        n_rounds,
        stochastic_adversary,
        gamma=0.1
    )

    print(f"Total reward: {total_reward:.1f}")
    print(f"Average reward per round: {total_reward/n_rounds:.3f}")
    print()

    # Example 2: Adversarial rewards
    print("="*60)
    print("Example 2: Adversarial Setting")
    print("-"*60)
    print("Adversary adapts to algorithm's choices")
    print()

    # Adversarial strategy: give lower rewards to frequently-chosen arms
    def adversarial_adversary(t, history):
        if len(history) < 10:
            # Random rewards initially
            return np.random.rand(3)

        # Count recent arm selections
        recent = history[-100:] if len(history) >= 100 else history
        counts = np.bincount(recent, minlength=3)

        # Give lower rewards to frequently-chosen arms
        total = counts.sum()
        if total > 0:
            frequencies = counts / total
            rewards = 1.0 - frequencies + np.random.rand(3) * 0.1
            return np.clip(rewards, 0, 1)
        else:
            return np.random.rand(3)

    np.random.seed(42)
    cum_rewards_adv, total_reward_adv = simulate_exp3_adversarial(
        n_arms,
        n_rounds,
        adversarial_adversary,
        gamma=0.1
    )

    print(f"Total reward: {total_reward_adv:.1f}")
    print(f"Average reward per round: {total_reward_adv/n_rounds:.3f}")
    print()

    # Compare regret growth
    print("="*60)
    print("REGRET COMPARISON")
    print("="*60)
    print()

    print("Cumulative reward (stochastic adversary):")
    checkpoints = [100, 250, 500, 750, 1000]
    for t in checkpoints:
        reward = cum_rewards[t-1]
        avg = reward / t
        print(f"  t={t:4d}: Total={reward:6.1f}, Avg={avg:.3f}")

    print()
    print("Cumulative reward (adaptive adversary):")
    for t in checkpoints:
        reward = cum_rewards_adv[t-1]
        avg = reward / t
        print(f"  t={t:4d}: Total={reward:6.1f}, Avg={avg:.3f}")

    print()
    print("="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("✓ EXP3 works in adversarial settings (UCB may fail)")
    print("✓ Regret bound: O(sqrt(KT log K))")
    print("✓ Minimax optimal for adversarial bandits")
    print("✓ Uses importance weighting for unbiased estimates")
    print("✓ Exploration via mixing with uniform distribution")
    print()
    print("Comparison with UCB:")
    print("  UCB (stochastic):   O(log T) regret")
    print("  EXP3 (adversarial): O(sqrt(T log K)) regret")
    print("  Trade-off: EXP3 works in harder settings but slower convergence")
    print()
    print("Applications:")
    print("- Online advertising with click manipulation")
    print("- Game playing vs adaptive opponents")
    print("- Cybersecurity (adversarial attackers)")
    print("- Spam filtering (adversarial spammers)")
    print("- Financial trading (adversarial markets)")
    print()
    print("Variants:")
    print("- EXP3.P: adds forced exploration")
    print("- EXP3-IX: improved for bounded rewards")
    print("- EXP3.S: for time-varying rewards")

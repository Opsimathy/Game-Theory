"""
Q-Learning in Multi-Agent Settings

Q-Learning is a model-free reinforcement learning algorithm that learns action values.
In multi-agent settings, agents learn simultaneously, leading to non-stationary environment.

Standard Q-Learning Update:
Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

Multi-Agent Challenges:
- Non-stationarity: Other agents are learning too
- Credit assignment: Hard to know if outcome is due to your action or others'
- Convergence: May not converge to Nash equilibrium

Variants for Games:
- Nash Q-Learning: Update toward Nash equilibrium of Q-values
- Friend-or-Foe Q-Learning: Distinguish cooperative/adversarial agents
- Minimax Q-Learning: For zero-sum games

Reference:
- Watkins (1989). "Learning from Delayed Rewards"
- Littman (1994). "Markov Games as a Framework for Multi-Agent RL"
- Hu & Wellman (2003). "Nash Q-Learning for General-Sum Stochastic Games"
"""

import numpy as np
from typing import Tuple, Optional, Dict
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games.normal_form import NormalFormGame


class QLearningAgent:
    """
    Q-Learning agent for multi-agent games.

    Learns Q-values for state-action pairs through interaction.
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1
    ):
        """
        Initialize Q-learning agent.

        Args:
            n_actions: Number of actions available
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy
        """
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q-values: Q[state][action]
        self.Q: Dict[any, np.ndarray] = defaultdict(lambda: np.zeros(n_actions))

        # Track visits for initialization
        self.visits: Dict[any, np.ndarray] = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state: any, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            explore: Whether to explore (False for evaluation)

        Returns:
            Selected action
        """
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action
            return int(np.argmax(self.Q[state]))

    def update(
        self,
        state: any,
        action: int,
        reward: float,
        next_state: any,
        done: bool
    ) -> None:
        """
        Update Q-value using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.visits[state][action] += 1

        # Q-learning update
        current_q = self.Q[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        self.Q[state][action] = current_q + self.alpha * (target - current_q)

    def get_greedy_action(self, state: any) -> int:
        """Get best action without exploration."""
        return int(np.argmax(self.Q[state]))

    def get_policy(self, state: any) -> np.ndarray:
        """
        Get policy as probability distribution.

        Returns greedy policy (probability 1 on best action).

        Args:
            state: State to get policy for

        Returns:
            Probability distribution over actions
        """
        policy = np.zeros(self.n_actions)
        best_action = self.get_greedy_action(state)
        policy[best_action] = 1.0
        return policy


class MinimaxQLearning(QLearningAgent):
    """
    Minimax Q-Learning for two-player zero-sum games.

    Instead of max_a' Q(s', a'), use value of minimax equilibrium.

    Reference:
    - Littman, M. (1994). "Markov Games as a Framework for Multi-Agent RL"
    """

    def __init__(
        self,
        n_actions_self: int,
        n_actions_opponent: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1
    ):
        """
        Initialize Minimax Q-learning agent.

        Args:
            n_actions_self: Number of actions for this agent
            n_actions_opponent: Number of actions for opponent
            learning_rate: Learning rate
            discount_factor: Discount factor
            epsilon: Exploration rate
        """
        super().__init__(n_actions_self, learning_rate, discount_factor, epsilon)
        self.n_actions_opponent = n_actions_opponent

        # Q-values are now Q[state][action_self][action_opponent]
        self.Q: Dict[any, np.ndarray] = defaultdict(
            lambda: np.zeros((n_actions_self, n_actions_opponent))
        )

    def get_minimax_value(self, state: any) -> Tuple[float, np.ndarray]:
        """
        Compute minimax value and policy for a state.

        Solve: max_p min_q p^T Q q

        Args:
            state: State to compute value for

        Returns:
            (value, policy) where policy is optimal mixed strategy
        """
        from scipy.optimize import linprog

        Q_matrix = self.Q[state]
        m, n = Q_matrix.shape

        # Solve for row player (maximizer)
        # Variables: [p_1, ..., p_m, v]
        # Objective: max v  =>  min -v
        c = np.zeros(m + 1)
        c[-1] = -1

        # Inequality constraints: -sum_i p_i * Q[i,j] + v <= 0 for all j
        A_ub = np.zeros((n, m + 1))
        for j in range(n):
            A_ub[j, :m] = -Q_matrix[:, j]
            A_ub[j, m] = 1

        b_ub = np.zeros(n)

        # Equality constraint: sum_i p_i = 1
        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1
        b_eq = np.array([1])

        # Bounds
        bounds = [(0, None) for _ in range(m)] + [(None, None)]

        # Solve
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs'
        )

        if result.success:
            policy = result.x[:m]
            value = result.x[m]
            return value, policy
        else:
            # Fallback to uniform policy
            return 0.0, np.ones(m) / m

    def update(
        self,
        state: any,
        action_self: int,
        action_opponent: int,
        reward: float,
        next_state: any,
        done: bool
    ) -> None:
        """
        Update Q-value using minimax Q-learning.

        Args:
            state: Current state
            action_self: Our action
            action_opponent: Opponent's action
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        current_q = self.Q[state][action_self, action_opponent]

        if done:
            target = reward
        else:
            # Use minimax value instead of max
            next_value, _ = self.get_minimax_value(next_state)
            target = reward + self.gamma * next_value

        self.Q[state][action_self, action_opponent] += self.alpha * (target - current_q)

    def get_action(self, state: any, explore: bool = True) -> int:
        """
        Select action using minimax policy with exploration.

        Args:
            state: Current state
            explore: Whether to explore

        Returns:
            Selected action
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            _, policy = self.get_minimax_value(state)
            return np.random.choice(self.n_actions, p=policy)


if __name__ == "__main__":
    print("=== Q-Learning in Matrix Games ===\n")

    print("Demonstrating Q-learning for repeated matrix game")
    print("Game: Rock-Paper-Scissors")
    print()

    # Create game
    game = NormalFormGame.rock_paper_scissors()

    # Create agents
    agent1 = QLearningAgent(n_actions=3, learning_rate=0.1, epsilon=0.1)
    agent2 = QLearningAgent(n_actions=3, learning_rate=0.1, epsilon=0.1)

    # Train through repeated play
    n_episodes = 10000
    state = 0  # Single state (stateless game)

    print(f"Training for {n_episodes} episodes...")

    for episode in range(n_episodes):
        # Both agents select actions
        action1 = agent1.get_action(state)
        action2 = agent2.get_action(state)

        # Get rewards
        rewards = game.get_payoff((action1, action2))

        # Update both agents
        agent1.update(state, action1, rewards[0], state, done=False)
        agent2.update(state, action2, rewards[1], state, done=False)

        # Print progress
        if (episode + 1) % 2000 == 0:
            policy1 = agent1.get_policy(state)
            policy2 = agent2.get_policy(state)
            print(f"Episode {episode + 1}:")
            print(f"  Agent 1 policy: {policy1}")
            print(f"  Agent 2 policy: {policy2}")

    print("\nFinal policies:")
    print(f"Agent 1: {agent1.get_policy(state)}")
    print(f"Agent 2: {agent2.get_policy(state)}")
    print()

    print("=== Key Points ===")
    print("Multi-Agent Q-Learning challenges:")
    print("- Non-stationarity: Opponent is learning too")
    print("- May not converge to Nash equilibrium")
    print("- Sensitive to learning rates and exploration")
    print()
    print("Better approaches for games:")
    print("- Nash Q-Learning: Update toward Nash equilibrium")
    print("- Minimax Q-Learning: For zero-sum games")
    print("- Policy gradient methods: REINFORCE, PPO")
    print("- Self-play with frozen opponents")

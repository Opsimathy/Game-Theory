# Game Theory Algorithms - Implementation Collection

This directory contains well-documented implementations of fundamental and advanced algorithms in computational game theory, mechanism design, and multi-agent learning.

## Directory Structure

### 1. `equilibrium/` - Equilibrium Computation
Algorithms for computing various equilibrium concepts:
- **Support Enumeration**: Exact algorithm for finding all Nash equilibria
- **Linear Programming**: Efficient solver for zero-sum games
- **Correlated Equilibrium**: LP-based solver for welfare-maximizing CE
- **Stackelberg Equilibrium**: Leader-follower game solver (security games)

### 2. `regret_minimization/` - Regret Minimization Algorithms
Core algorithms for online learning and large game solving:
- **Regret Matching**: Basic regret-based strategy updates
- **CFR (Counterfactual Regret Minimization)**: Standard CFR algorithm
- **CFR+**: Improved variant with faster convergence
- **External Regret Minimization**: No-regret learning algorithms

### 3. `search/` - Game Tree Search
Search algorithms for sequential games:
- **Monte Carlo Tree Search (MCTS)**: UCT-based tree search
- **Minimax with Alpha-Beta Pruning**: Classic adversarial search
- **Expectimax**: Search under uncertainty

### 4. `auctions/` - Auction Mechanisms
Implementation of various auction formats:
- **First-Price Sealed-Bid Auction**
- **Second-Price (Vickrey) Auction**
- **VCG (Vickrey-Clarke-Groves) Mechanism**
- **Ascending/Descending Auctions**

### 5. `social_choice/` - Voting and Social Choice
Voting systems and fair division algorithms:
- **Plurality, Borda Count, Approval Voting**
- **Condorcet Methods**
- **Single Transferable Vote (STV)**
- **Fair Division Algorithms**

### 6. `learning/` - Multi-Agent Learning
Learning algorithms for game-theoretic settings:
- **Fictitious Play**: Classic learning dynamics
- **Q-Learning in Games**: Value-based learning
- **Policy Gradient Methods**: Direct policy optimization
- **Self-Play Training**

### 7. `games/` - Game Representations
Common game implementations and utilities:
- **Normal-Form Games**: Matrix game representation
- **Extensive-Form Games**: Game tree representation
- **Poker Games**: Simplified poker variants (Kuhn Poker)
- **Utility Functions**: Common game utilities

### 8. `evolutionary/` - Evolutionary Game Theory
Evolutionary dynamics and stable strategies:
- **Replicator Dynamics**: Population strategy evolution
- **ESS (Evolutionarily Stable Strategy)**: Stability analysis
- **Hawk-Dove and other biological games**

### 9. `cooperative/` - Cooperative Games
Coalitional game theory and solution concepts:
- **Shapley Value**: Fair profit/cost allocation
- **Core**: Stable coalition structures
- **Examples**: Glove game, voting games, airport game

### 10. `matching/` - Matching Markets
Stable matching algorithms:
- **Gale-Shapley Algorithm**: Deferred acceptance algorithm
- **Stable Marriage Problem**: Two-sided matching
- **Applications**: Medical residency, school choice

### 11. `bandits/` - Multi-Armed Bandits
Exploration-exploitation algorithms:
- **UCB1**: Upper Confidence Bound for stochastic bandits
- **EXP3**: Exponential-weight algorithm for adversarial bandits
- **Applications**: Clinical trials, online advertising

### 12. `repeated_games/` - Repeated Game Strategies
Strategies for infinitely repeated games:
- **Tit-for-Tat**: Copy opponent's last move
- **Grim Trigger**: Permanent punishment
- **Pavlov**: Win-stay, lose-shift
- **Tournament simulation**: Round-robin competitions

## Requirements

```bash
pip install numpy scipy
```

For visualization (optional):
```bash
pip install matplotlib networkx
```

## Usage Examples

Each subdirectory contains a README with specific usage examples. Here's a quick overview:

```python
# Computing Nash equilibrium
from equilibrium.support_enumeration import SupportEnumerationSolver
from games.normal_form import NormalFormGame

game = NormalFormGame(payoff_matrix_p1, payoff_matrix_p2)
solver = SupportEnumerationSolver(game)
equilibria = solver.solve()

# Running CFR on a poker game
from regret_minimization.cfr import CFRSolver
from games.kuhn_poker import KuhnPoker

game = KuhnPoker()
solver = CFRSolver(game)
strategy = solver.train(iterations=10000)

# MCTS for game tree search
from search.mcts import MCTSAgent
from games.tictactoe import TicTacToe

game = TicTacToe()
agent = MCTSAgent(simulations=1000)
action = agent.get_action(game.get_state())
```

## Key Algorithms Implemented

### Nash Equilibrium Computation
- **Support Enumeration**: O(2^n × 2^m) - Enumerates all possible support sets
- **Linear Programming**: Polynomial for zero-sum games
- **Complexity**: Finding Nash equilibrium is PPAD-complete in general

### Regret Minimization
- **CFR**: O(|I| × |A|) per iteration, where |I| is information sets, |A| is actions
- **Convergence**: Average regret approaches 0 at rate O(1/√T)
- **Applications**: Solved heads-up limit poker (10^14 game states)

### MCTS
- **UCT (UCB for Trees)**: Balances exploration and exploitation
- **Time Complexity**: O(simulations × depth)
- **Success Stories**: AlphaGo, AlphaZero, MuZero

## Testing

Run tests for all modules:
```bash
python -m pytest tests/
```

Run specific algorithm tests:
```bash
python -m pytest tests/test_cfr.py
python -m pytest tests/test_nash.py
```

## Performance Notes

- Implementations prioritize clarity and correctness over performance
- For production use, consider optimized libraries like OpenSpiel
- NumPy operations are vectorized where possible
- Some algorithms include both naive and optimized versions

## References

Each implementation includes references to:
- Original papers introducing the algorithm
- Textbook chapters with detailed explanations
- Modern variants and improvements

## Contributing

When adding new algorithms:
1. Follow the existing code structure
2. Include comprehensive docstrings
3. Add unit tests
4. Update the relevant README
5. Include complexity analysis and references

## License

Educational and research use. See individual file headers for specific attributions.

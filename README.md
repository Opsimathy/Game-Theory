# Game Theory

A curated collection of resources on game theory, mechanism design, and social choice theory, with a focus on their algorithmic, computational, and multi-agent learning aspects.

## Purpose and Motivation

This repository serves as a comprehensive knowledge base for understanding game theory from a computational perspective. It bridges classical game-theoretic concepts with modern algorithmic approaches and their applications in artificial intelligence, economics, and multi-agent systems.

Game theory provides the mathematical foundation for analyzing strategic interactions between rational agents. As AI systems become increasingly sophisticated and autonomous, understanding equilibrium concepts, mechanism design, and computational approaches to solving games becomes crucial for:

- **AI and Multi-Agent Systems**: Developing intelligent agents that can reason strategically in competitive and cooperative environments
- **Algorithmic Economics**: Designing markets, auctions, and mechanisms that achieve desirable outcomes
- **Social Choice**: Understanding voting systems, fair division, and collective decision-making
- **Online Learning**: Adapting strategies in dynamic environments with incomplete information

This collection emphasizes the computational aspects of game theory, including algorithms for computing equilibria, game abstraction techniques, and modern approaches like counterfactual regret minimization (CFR) that have led to superhuman performance in complex games.

## Repository Structure

### 📚 Books

The `Books/` directory contains foundational and advanced textbooks covering:

#### Game Theory Fundamentals
- **Osborne & Rubinstein** - A Course in Game Theory (with Solutions)
- **Steven Tadelis** - Game Theory: An Introduction
- **Fudenberg & Tirole** - Game Theory
- **Maschler, Solan & Zamir** - Game Theory

#### Algorithmic and Computational Perspectives
- **Algorithmic Game Theory** - Comprehensive coverage of computational aspects
- **Economics and Computation** - Intersection of computer science and economics
- **Networks, Crowds, and Markets** - Network effects and strategic behavior
- **Computational Social Choice** - Voting, fair division, and collective decisions

#### Specialized Topics
- **Auction Theory** - Mechanism design for auctions
- **Microeconomic Theory** - Economic foundations
- **Population Games and Evolutionary Dynamics** - Evolutionary game theory
- **Multi-Agent Reinforcement Learning** - Learning in strategic environments

#### Supporting Materials
- **Introduction to Algorithms** - Algorithmic foundations
- **Fourier Analysis** - Mathematical tools

### 📝 Notes

The `Notes/` directory contains lecture materials from advanced courses:

#### CS15-888F24: Computational Game Theory
Topics covered include:
- Tree-form and extensive-form games
- Monte Carlo Tree Search (MCTS)
- AlphaGo and AlphaZero
- Normal-form games and Nash equilibria
- Counterfactual Regret Minimization (CFR) and speedups
- Correlation and correlated equilibria
- Game abstraction techniques
- Subgame solving (Libratus, Pluribus)
- Policy Space Response Oracles (PSRO)
- Deep learning in games
- Mechanism design
- Team games and certificates

#### CS364B: Advanced Topics
Lecture materials covering advanced mechanism design and algorithmic game theory topics (Lectures 21-40)

#### Additional Resources
- **A Modern Introduction to Online Learning** - Online learning algorithms and regret minimization

### 🎓 Seminar

The `Seminar/` directory contains research papers on cutting-edge topics in:
- Online Markov Decision Processes
- Policy optimization for Markov games
- Learning algorithms (V-learning, Q-learning)
- Game-theoretic analysis and applications
- Advanced algorithmic techniques

## Key Topics Covered

### Core Game Theory Concepts
- **Solution Concepts**: Nash equilibrium, correlated equilibrium, subgame perfect equilibrium
- **Game Representations**: Normal-form, extensive-form, tree-form games
- **Information Structures**: Perfect/imperfect information, complete/incomplete information
- **Game Types**: Zero-sum, general-sum, cooperative, repeated games

### Computational Methods
- **Equilibrium Computation**: Linear programming, support enumeration, fictitious play
- **Regret Minimization**: CFR, CFR+, MCCFR, Deep CFR
- **Search Algorithms**: MCTS, minimax, alpha-beta pruning
- **Abstraction Techniques**: Lossless abstraction, information abstraction, action abstraction
- **Subgame Solving**: Safe subgame solving, depth-limited solving

### Applications and Modern Developments
- **AI for Games**: Poker (Libratus, Pluribus), Go (AlphaGo), Chess (AlphaZero), Stratego
- **Mechanism Design**: Auctions, voting systems, incentive compatibility
- **Multi-Agent Learning**: Independent learning, opponent modeling, meta-game analysis
- **Online Learning**: Bandit algorithms, no-regret learning, expert algorithms

## Recommended Additional Resources

### Online Courses
- [Tim Roughgarden's Game Theory Course (Coursera/Stanford)](https://www.coursera.org/learn/game-theory-1)
- [AGT Course by Tim Roughgarden](http://timroughgarden.org/f13/f13.html)
- [Algorithmic Game Theory (CMU 15-888)](https://www.cs.cmu.edu/~sandholm/cs15-888F24/)

### Key Research Groups and Researchers
- **CMU**: Tuomas Sandholm (computational game theory, poker AI)
- **DeepMind**: David Silver (AlphaGo, AlphaZero, reinforcement learning)
- **Stanford**: Tim Roughgarden (algorithmic game theory)
- **UC Berkeley**: Christos Papadimitriou, Alistair Sinclair (computational complexity)

### Important Conferences
- **EC** (Economics and Computation) - Premier venue for algorithmic game theory
- **AAAI/IJCAI** - AI applications of game theory
- **NeurIPS/ICML** - Machine learning in games
- **WINE** (Web and Internet Economics)
- **AAMAS** (Autonomous Agents and Multiagent Systems)

### Software and Tools
- **OpenSpiel** - Framework for research in game theory and reinforcement learning
- **Gambit** - Software tools for game theory analysis
- **PokerKit** - Python library for poker game development

### Online Resources
- [Game Theory Online](https://www.gametheory.net/) - Educational resources and tutorials
- [Algorithmic Game Theory Book (Free Online)](https://www.cambridge.org/us/academic/subjects/computer-science/algorithmics-complexity-computer-algebra-and-computational-g/algorithmic-game-theory)
- [Tim Roughgarden's YouTube Channel](https://www.youtube.com/user/algorithmicgametheory)

### Classic Papers
- Von Neumann & Morgenstern - Theory of Games and Economic Behavior (1944)
- Nash - Non-Cooperative Games (1951)
- Lemke & Howson - Equilibrium Points of Bimatrix Games (1964)
- Zinkevich et al. - Regret Minimization in Games with Incomplete Information (2007)
- Silver et al. - Mastering the Game of Go with Deep Neural Networks (2016)
- Brown & Sandholm - Superhuman AI for Multiplayer Poker (2019)

## Contributing

This is a personal collection of resources. If you find it useful and have suggestions for additional high-quality resources, feel free to open an issue or submit a pull request.

## Related Topics

For those interested in exploring related areas:
- **Reinforcement Learning**: Single-agent sequential decision making
- **Optimization Theory**: Mathematical programming and duality theory
- **Probability and Statistics**: Stochastic processes, Bayesian inference
- **Complexity Theory**: Computational hardness of equilibrium problems
- **Distributed Systems**: Consensus protocols and Byzantine agreement

## License

This repository contains links to and collections of academic resources. Please respect the copyright and licensing terms of individual resources. Use for educational and research purposes.

---

*Last Updated: November 2025*

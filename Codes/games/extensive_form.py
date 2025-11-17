"""
Extensive-Form Game Representation

An extensive-form game represents sequential games with:
- Game tree structure
- Information sets (which nodes a player cannot distinguish)
- Chance nodes (random events)
- Payoffs at terminal nodes

Reference:
- Osborne & Rubinstein, "A Course in Game Theory" (1994), Chapter 6
- Shoham & Leyton-Brown, "Multiagent Systems" (2009), Chapter 5
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class GameNode:
    """
    Represents a node in an extensive-form game tree.

    Attributes:
        node_id: Unique identifier for this node
        player: Player who acts at this node (None for terminal/chance nodes)
        actions: Available actions at this node
        children: Mapping from actions to child nodes
        payoffs: Terminal payoffs if this is a terminal node
        infoset: Information set identifier
        is_terminal: Whether this is a terminal node
        is_chance: Whether this is a chance node
        chance_probs: Probability distribution over actions at chance nodes
    """
    node_id: str
    player: Optional[int] = None
    actions: List[Any] = None
    children: Dict[Any, 'GameNode'] = None
    payoffs: Optional[List[float]] = None
    infoset: Optional[str] = None
    is_terminal: bool = False
    is_chance: bool = False
    chance_probs: Optional[Dict[Any, float]] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.children is None:
            self.children = {}


class ExtensiveFormGame:
    """
    Represents an extensive-form game.

    Attributes:
        n_players: Number of players
        root: Root node of the game tree
        infosets: Mapping from information set IDs to lists of nodes
    """

    def __init__(self, n_players: int):
        """
        Initialize an extensive-form game.

        Args:
            n_players: Number of players in the game
        """
        self.n_players = n_players
        self.root: Optional[GameNode] = None
        self.infosets: Dict[str, List[GameNode]] = {}

    def add_node(self, node: GameNode) -> None:
        """
        Add a node to the game tree.

        Args:
            node: GameNode to add
        """
        if self.root is None:
            self.root = node

        # Add to information set mapping
        if node.infoset is not None:
            if node.infoset not in self.infosets:
                self.infosets[node.infoset] = []
            self.infosets[node.infoset].append(node)

    def get_terminal_payoffs(self, action_sequence: List[Tuple[GameNode, Any]]) -> List[float]:
        """
        Get terminal payoffs following a sequence of actions.

        Args:
            action_sequence: List of (node, action) pairs

        Returns:
            Payoffs for all players at the terminal node
        """
        current = self.root

        for node, action in action_sequence:
            if current != node:
                raise ValueError("Invalid action sequence")
            if action not in current.children:
                raise ValueError(f"Invalid action {action} at node {current.node_id}")
            current = current.children[action]

        if not current.is_terminal:
            raise ValueError("Action sequence does not lead to terminal node")

        return current.payoffs

    def get_information_set_actions(self, infoset: str) -> List[Any]:
        """
        Get available actions at an information set.

        Args:
            infoset: Information set identifier

        Returns:
            List of available actions (should be same for all nodes in infoset)
        """
        if infoset not in self.infosets:
            raise ValueError(f"Information set {infoset} not found")

        nodes = self.infosets[infoset]
        if len(nodes) == 0:
            return []

        # All nodes in an information set must have the same actions
        return nodes[0].actions

    @classmethod
    def kuhn_poker(cls):
        """
        Create a Kuhn Poker game.

        Kuhn Poker is the simplest poker game:
        - 3 cards: King, Queen, Jack
        - 2 players, each antes 1 chip
        - Each player gets 1 card
        - Player 1 can check or bet (1 chip)
        - If P1 checks, P2 can check or bet
        - If P1 bets, P2 can fold or call
        - If both check, higher card wins
        - If someone bets and opponent folds, bettor wins
        - If bet is called, higher card wins

        Reference: Kuhn, H. W. (1950). "A simplified two-person poker"
        """
        game = cls(n_players=2)

        # We'll build a simplified version for demonstration
        # In practice, you'd enumerate all chance outcomes

        # Root is chance node (dealing cards)
        root = GameNode(
            node_id="root",
            is_chance=True,
            actions=["KQ", "KJ", "QK", "QJ", "JK", "JQ"],
            chance_probs={
                "KQ": 1/6, "KJ": 1/6, "QK": 1/6,
                "QJ": 1/6, "JK": 1/6, "JQ": 1/6
            }
        )
        game.root = root

        # For brevity, we'll show the structure for one deal: P1 has K, P2 has Q
        deal_kq = GameNode(
            node_id="KQ",
            player=0,
            actions=["check", "bet"],
            infoset="P1_K"
        )
        root.children["KQ"] = deal_kq
        game.add_node(deal_kq)

        # P1 checks
        p1_check = GameNode(
            node_id="KQ_check",
            player=1,
            actions=["check", "bet"],
            infoset="P2_Q_P1check"
        )
        deal_kq.children["check"] = p1_check
        game.add_node(p1_check)

        # P1 checks, P2 checks (showdown, P1 wins with K)
        terminal_cc = GameNode(
            node_id="KQ_check_check",
            is_terminal=True,
            payoffs=[1, -1]  # P1 has higher card
        )
        p1_check.children["check"] = terminal_cc
        game.add_node(terminal_cc)

        # P1 checks, P2 bets
        p1_check_p2_bet = GameNode(
            node_id="KQ_check_bet",
            player=0,
            actions=["fold", "call"],
            infoset="P1_K_P2bet"
        )
        p1_check.children["bet"] = p1_check_p2_bet
        game.add_node(p1_check_p2_bet)

        # P1 folds
        terminal_cbf = GameNode(
            node_id="KQ_check_bet_fold",
            is_terminal=True,
            payoffs=[-1, 1]  # P2 wins the ante
        )
        p1_check_p2_bet.children["fold"] = terminal_cbf
        game.add_node(terminal_cbf)

        # P1 calls
        terminal_cbc = GameNode(
            node_id="KQ_check_bet_call",
            is_terminal=True,
            payoffs=[2, -2]  # P1 wins with higher card
        )
        p1_check_p2_bet.children["call"] = terminal_cbc
        game.add_node(terminal_cbc)

        # P1 bets
        p1_bet = GameNode(
            node_id="KQ_bet",
            player=1,
            actions=["fold", "call"],
            infoset="P2_Q_P1bet"
        )
        deal_kq.children["bet"] = p1_bet
        game.add_node(p1_bet)

        # P2 folds
        terminal_bf = GameNode(
            node_id="KQ_bet_fold",
            is_terminal=True,
            payoffs=[1, -1]
        )
        p1_bet.children["fold"] = terminal_bf
        game.add_node(terminal_bf)

        # P2 calls
        terminal_bc = GameNode(
            node_id="KQ_bet_call",
            is_terminal=True,
            payoffs=[2, -2]  # P1 wins with higher card
        )
        p1_bet.children["call"] = terminal_bc
        game.add_node(terminal_bc)

        return game

    def __repr__(self):
        return f"ExtensiveFormGame(players={self.n_players}, infosets={len(self.infosets)})"

"""
Single-Item Auction Mechanisms

Classical auction formats for selling a single indivisible item.

Auction Formats:
1. First-Price Sealed-Bid: Highest bidder wins, pays their bid
2. Second-Price (Vickrey): Highest bidder wins, pays second-highest bid

Key Properties:
- Vickrey auction is truthful (dominant strategy to bid true value)
- First-price requires strategic bidding (shade below true value)
- Revenue equivalence: Both yield same expected revenue under certain conditions

Reference:
- Vickrey, W. (1961). "Counterspeculation, Auctions, and Competitive Sealed Tenders"
- Krishna, V. (2009). "Auction Theory" (2nd edition)
- Milgrom, P. (2004). "Putting Auction Theory to Work"
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Bid:
    """
    Represents a bid in an auction.

    Attributes:
        bidder_id: Unique identifier for the bidder
        amount: Bid amount
        value: True value (private information, used for analysis)
    """
    bidder_id: int
    amount: float
    value: Optional[float] = None


class FirstPriceAuction:
    """
    First-Price Sealed-Bid Auction.

    - Bidders submit sealed bids
    - Highest bidder wins
    - Winner pays their own bid

    Strategic considerations:
    - Bidding true value yields zero utility if you win
    - Optimal to shade bid below true value
    - Equilibrium bid depends on value distribution
    """

    def __init__(self):
        """Initialize first-price auction."""
        self.bids: List[Bid] = []

    def submit_bid(self, bidder_id: int, amount: float, value: Optional[float] = None) -> None:
        """
        Submit a bid.

        Args:
            bidder_id: Bidder identifier
            amount: Bid amount
            value: True value (optional, for analysis)
        """
        self.bids.append(Bid(bidder_id, amount, value))

    def run_auction(self) -> Tuple[int, float, float]:
        """
        Run the auction and determine winner.

        Returns:
            (winner_id, payment, winner_utility)
        """
        if not self.bids:
            raise ValueError("No bids submitted")

        # Find highest bid
        winning_bid = max(self.bids, key=lambda b: b.amount)

        winner_id = winning_bid.bidder_id
        payment = winning_bid.amount

        # Calculate utility if true value is known
        utility = 0.0
        if winning_bid.value is not None:
            utility = winning_bid.value - payment

        return winner_id, payment, utility

    def get_revenue(self) -> float:
        """Get auction revenue (payment from winner)."""
        if not self.bids:
            return 0.0
        _, payment, _ = self.run_auction()
        return payment


class SecondPriceAuction:
    """
    Second-Price (Vickrey) Auction.

    - Bidders submit sealed bids
    - Highest bidder wins
    - Winner pays second-highest bid

    Key Properties:
    - Truthful: Dominant strategy to bid true value
    - Strategy-proof: No incentive to misreport
    - Efficient: Item goes to bidder with highest value

    Proof of truthfulness:
    - If you have value v and bid b:
      - If b > v: Risk paying more than value
      - If b < v: Risk losing when you'd want to win
      - Bidding b = v is always optimal
    """

    def __init__(self):
        """Initialize second-price auction."""
        self.bids: List[Bid] = []

    def submit_bid(self, bidder_id: int, amount: float, value: Optional[float] = None) -> None:
        """
        Submit a bid.

        Args:
            bidder_id: Bidder identifier
            amount: Bid amount
            value: True value (optional, for analysis)
        """
        self.bids.append(Bid(bidder_id, amount, value))

    def run_auction(self) -> Tuple[int, float, float]:
        """
        Run the auction and determine winner.

        Returns:
            (winner_id, payment, winner_utility)
        """
        if not self.bids:
            raise ValueError("No bids submitted")

        # Sort bids by amount (descending)
        sorted_bids = sorted(self.bids, key=lambda b: b.amount, reverse=True)

        # Winner is highest bidder
        winning_bid = sorted_bids[0]
        winner_id = winning_bid.bidder_id

        # Payment is second-highest bid
        if len(sorted_bids) > 1:
            payment = sorted_bids[1].amount
        else:
            payment = 0.0  # No competition

        # Calculate utility if true value is known
        utility = 0.0
        if winning_bid.value is not None:
            utility = winning_bid.value - payment

        return winner_id, payment, utility

    def get_revenue(self) -> float:
        """Get auction revenue (payment from winner)."""
        if not self.bids:
            return 0.0
        _, payment, _ = self.run_auction()
        return payment


def optimal_bid_first_price(value: float, n_bidders: int, distribution: str = "uniform") -> float:
    """
    Compute optimal bid in first-price auction.

    For symmetric bidders with values drawn from [0, 1]:
    - Uniform distribution: b(v) = (n-1)/n * v
    - General distribution: b(v) = v - ∫[0 to v] F(x)^(n-1) dx / F(v)^(n-1)

    Args:
        value: Bidder's true value
        n_bidders: Number of bidders
        distribution: Value distribution ("uniform" only for now)

    Returns:
        Optimal bid amount
    """
    if distribution == "uniform":
        # For uniform [0, 1]: b(v) = (n-1)/n * v
        return ((n_bidders - 1) / n_bidders) * value
    else:
        raise NotImplementedError(f"Distribution {distribution} not implemented")


if __name__ == "__main__":
    print("=== First-Price Auction ===")
    auction1 = FirstPriceAuction()

    # Bidders with values [100, 80, 60]
    # In equilibrium, they should shade their bids
    values = [100, 80, 60]
    n_bidders = 3

    print(f"True values: {values}")
    print(f"Optimal bids (assuming uniform values on [0, 100]):")

    for i, value in enumerate(values):
        optimal_bid = optimal_bid_first_price(value / 100, n_bidders) * 100
        auction1.submit_bid(i, optimal_bid, value)
        print(f"  Bidder {i}: value={value}, bid={optimal_bid:.2f}")

    winner, payment, utility = auction1.run_auction()
    print(f"\nResult: Bidder {winner} wins")
    print(f"Payment: ${payment:.2f}")
    print(f"Utility: ${utility:.2f}")
    print(f"Revenue: ${auction1.get_revenue():.2f}")

    print("\n=== Second-Price (Vickrey) Auction ===")
    auction2 = SecondPriceAuction()

    # In Vickrey auction, bidding true value is optimal
    print(f"True values: {values}")
    print(f"Optimal strategy: Bid true value")

    for i, value in enumerate(values):
        auction2.submit_bid(i, value, value)
        print(f"  Bidder {i}: bid={value}")

    winner, payment, utility = auction2.run_auction()
    print(f"\nResult: Bidder {winner} wins")
    print(f"Payment: ${payment:.2f}")
    print(f"Utility: ${utility:.2f}")
    print(f"Revenue: ${auction2.get_revenue():.2f}")

    print("\n=== Key Insights ===")
    print("1. Vickrey auction is truthful (optimal to bid true value)")
    print("2. First-price requires strategic bidding (shade below value)")
    print("3. Revenue Equivalence Theorem: Both auctions yield same expected revenue")
    print("   under symmetric independent private values")
    print("4. Vickrey is easier for bidders (no need to strategize)")
    print("5. First-price might generate higher revenue in practice (cursed)")

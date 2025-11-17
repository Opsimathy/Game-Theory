"""
Vickrey-Clarke-Groves (VCG) Mechanism

VCG is a general mechanism for multi-item auctions and combinatorial auctions.
It generalizes the Vickrey auction to arbitrary valuation functions.

Key Properties:
- Truthful (incentive compatible): Dominant strategy to report true values
- Efficient: Maximizes social welfare
- Individual rationality: Participation is beneficial

VCG Payment Rule:
- Each winner pays the "harm" they cause to others
- Payment = (welfare without you) - (welfare of others with you)

Applications:
- Combinatorial auctions (spectrum, procurement)
- Sponsored search auctions (with modifications)
- Resource allocation in cloud computing

Reference:
- Vickrey (1961), Clarke (1971), Groves (1973)
- Nisan et al. (2007). "Algorithmic Game Theory", Chapter 9
- Cramton et al. (2006). "Combinatorial Auctions"
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Callable, Any
from itertools import combinations
from dataclasses import dataclass


@dataclass
class VCGBid:
    """
    Bid in a VCG auction.

    Attributes:
        bidder_id: Unique bidder identifier
        bundle: Set of items bidder wants
        value: Reported value for the bundle
    """
    bidder_id: int
    bundle: Set[int]
    value: float


class VCGMechanism:
    """
    Vickrey-Clarke-Groves mechanism for combinatorial auctions.

    Supports:
    - Multiple items
    - Combinatorial valuations (value for bundles)
    - Truthful allocation and pricing
    """

    def __init__(self, items: Set[int]):
        """
        Initialize VCG mechanism.

        Args:
            items: Set of item identifiers
        """
        self.items = items
        self.bids: List[VCGBid] = []
        self.allocation: Dict[int, Set[int]] = {}
        self.payments: Dict[int, float] = {}

    def submit_bid(self, bidder_id: int, bundle: Set[int], value: float) -> None:
        """
        Submit a bid for a bundle of items.

        Args:
            bidder_id: Bidder identifier
            bundle: Set of items
            value: Value for this bundle
        """
        if not bundle.issubset(self.items):
            raise ValueError("Bundle contains invalid items")

        self.bids.append(VCGBid(bidder_id, bundle, value))

    def _compute_optimal_allocation(
        self,
        bids: List[VCGBid]
    ) -> Tuple[Dict[int, Set[int]], float]:
        """
        Compute welfare-maximizing allocation.

        This is an NP-hard problem in general (combinatorial auction).
        For small instances, we use brute force.

        Args:
            bids: List of bids to consider

        Returns:
            (allocation, total_welfare) where allocation maps bidder_id to bundle
        """
        if not bids:
            return {}, 0.0

        # Get unique bidders
        bidders = list(set(bid.bidder_id for bid in bids))

        best_allocation = {}
        best_welfare = 0.0

        # Try all possible allocations (exponential in number of bidders)
        # For each bidder, decide which bid (if any) to accept

        def is_feasible(allocation: Dict[int, VCGBid]) -> bool:
            """Check if allocation doesn't assign same item to multiple bidders."""
            assigned_items = set()
            for bid in allocation.values():
                if bid.bundle & assigned_items:  # Overlap
                    return False
                assigned_items |= bid.bundle
            return True

        # Group bids by bidder
        bids_by_bidder = {b: [] for b in bidders}
        for bid in bids:
            bids_by_bidder[bid.bidder_id].append(bid)

        # Enumerate all combinations
        # For each bidder, choose 0 or 1 bid
        def enumerate_allocations(bidder_idx: int, current_allocation: Dict[int, VCGBid]):
            nonlocal best_allocation, best_welfare

            if bidder_idx == len(bidders):
                # Check feasibility
                if is_feasible(current_allocation):
                    welfare = sum(bid.value for bid in current_allocation.values())
                    if welfare > best_welfare:
                        best_welfare = welfare
                        best_allocation = {
                            bidder_id: bid.bundle
                            for bidder_id, bid in current_allocation.items()
                        }
                return

            bidder = bidders[bidder_idx]

            # Option 1: Don't allocate to this bidder
            enumerate_allocations(bidder_idx + 1, current_allocation)

            # Option 2: Allocate one of their bids
            for bid in bids_by_bidder[bidder]:
                enumerate_allocations(
                    bidder_idx + 1,
                    {**current_allocation, bidder: bid}
                )

        enumerate_allocations(0, {})

        return best_allocation, best_welfare

    def run_mechanism(self) -> Tuple[Dict[int, Set[int]], Dict[int, float]]:
        """
        Run VCG mechanism: compute allocation and payments.

        Returns:
            (allocation, payments) where:
                allocation: Maps bidder_id to set of items
                payments: Maps bidder_id to payment amount
        """
        # Compute optimal allocation
        allocation, total_welfare = self._compute_optimal_allocation(self.bids)

        # Compute VCG payments
        payments = {}

        for bidder_id in allocation:
            # Welfare without this bidder
            bids_without = [bid for bid in self.bids if bid.bidder_id != bidder_id]
            _, welfare_without = self._compute_optimal_allocation(bids_without)

            # Welfare of others in actual allocation
            welfare_others = total_welfare - next(
                bid.value for bid in self.bids
                if bid.bidder_id == bidder_id and bid.bundle == allocation[bidder_id]
            )

            # VCG payment = harm to others
            payment = welfare_without - welfare_others
            payments[bidder_id] = payment

        self.allocation = allocation
        self.payments = payments

        return allocation, payments

    def get_total_revenue(self) -> float:
        """Get total revenue from all payments."""
        return sum(self.payments.values())

    def get_social_welfare(self) -> float:
        """Get total social welfare (sum of values)."""
        _, welfare = self._compute_optimal_allocation(self.bids)
        return welfare


if __name__ == "__main__":
    print("=== VCG Mechanism Example ===")
    print("\nScenario: Selling items {A, B} to 3 bidders")
    print()

    # Create mechanism for items {0, 1} (representing A and B)
    mechanism = VCGMechanism(items={0, 1})

    # Bidder 1: values A at $10, B at $5, {A,B} at $20
    mechanism.submit_bid(bidder_id=1, bundle={0}, value=10)
    mechanism.submit_bid(bidder_id=1, bundle={1}, value=5)
    mechanism.submit_bid(bidder_id=1, bundle={0, 1}, value=20)

    # Bidder 2: values A at $8, B at $12
    mechanism.submit_bid(bidder_id=2, bundle={0}, value=8)
    mechanism.submit_bid(bidder_id=2, bundle={1}, value=12)

    # Bidder 3: values B at $6
    mechanism.submit_bid(bidder_id=3, bundle={1}, value=6)

    print("Bids submitted:")
    print("Bidder 1: A=$10, B=$5, {A,B}=$20")
    print("Bidder 2: A=$8, B=$12")
    print("Bidder 3: B=$6")
    print()

    # Run mechanism
    allocation, payments = mechanism.run_mechanism()

    print("=== Results ===")
    print("\nAllocation:")
    item_names = {0: 'A', 1: 'B'}
    for bidder_id, bundle in allocation.items():
        items_str = '{' + ','.join(item_names[i] for i in bundle) + '}'
        print(f"  Bidder {bidder_id} receives: {items_str}")

    print("\nPayments:")
    for bidder_id, payment in payments.items():
        print(f"  Bidder {bidder_id} pays: ${payment:.2f}")

    print(f"\nTotal Revenue: ${mechanism.get_total_revenue():.2f}")
    print(f"Social Welfare: ${mechanism.get_social_welfare():.2f}")

    print("\n=== VCG Payment Explanation ===")
    print("Each winner pays the 'harm' they cause to others:")
    print("- Payment = (welfare without you) - (welfare of others with you)")
    print()
    print("Example: Bidder 1 receives {A,B} for value $20")
    print("- Without Bidder 1: Best allocation is Bidder 2 gets A ($8) and B ($12) = $20")
    print("- With Bidder 1: Others get nothing = $0")
    print("- Payment: $20 - $0 = $20")
    print()
    print("Key Properties:")
    print("✓ Truthful: Dominant strategy to report true values")
    print("✓ Efficient: Maximizes social welfare")
    print("✓ Individual Rational: Winners get non-negative utility")

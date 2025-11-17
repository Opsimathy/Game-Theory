"""
Social Choice and Voting Systems
"""

from .voting import (
    plurality_vote,
    borda_count,
    approval_voting,
    instant_runoff,
    condorcet_winner
)

__all__ = [
    'plurality_vote',
    'borda_count',
    'approval_voting',
    'instant_runoff',
    'condorcet_winner'
]

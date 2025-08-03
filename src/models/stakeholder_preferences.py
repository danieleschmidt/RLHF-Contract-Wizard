"""
Stakeholder preference management and aggregation.

Implements multi-stakeholder governance patterns for RLHF contracts
with democratic voting and preference aggregation mechanisms.
"""

import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import jax.numpy as jnp
import logging
from datetime import datetime


class VotingStrategy(Enum):
    """Voting strategies for stakeholder decisions."""
    SIMPLE_MAJORITY = "simple_majority"
    SUPERMAJORITY = "supermajority"
    UNANIMOUS = "unanimous"
    WEIGHTED_MAJORITY = "weighted_majority"


class ProposalStatus(Enum):
    """Status of contract amendment proposals."""
    DRAFT = "draft"
    ACTIVE = "active"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Vote:
    """Represents a single stakeholder vote."""
    stakeholder: str
    support: bool
    weight: float
    timestamp: float
    reasoning: Optional[str] = None


@dataclass
class Amendment:
    """Represents a proposed contract amendment."""
    id: str
    proposer: str
    title: str
    description: str
    changes: Dict[str, Any]
    created_at: float
    expires_at: float
    status: ProposalStatus = ProposalStatus.DRAFT
    votes: List[Vote] = field(default_factory=list)
    required_consensus: float = 0.66


class StakeholderPreferences:
    """
    Manages stakeholder preferences and democratic governance.
    
    Handles preference registration, voting, and consensus mechanisms
    for RLHF contract governance.
    """
    
    def __init__(self, voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED_MAJORITY):
        """
        Initialize stakeholder preferences manager.
        
        Args:
            voting_strategy: Strategy for counting votes
        """
        self.voting_strategy = voting_strategy
        self.stakeholder_rewards: Dict[str, Callable] = {}
        self.stakeholder_weights: Dict[str, float] = {}
        self.amendments: Dict[str, Amendment] = {}
        self._preference_history: List[Dict[str, Any]] = []
    
    def add_stakeholder(
        self, 
        name: str, 
        weight: float,
        reward_function: Optional[Callable] = None
    ) -> 'StakeholderPreferences':
        """
        Add a new stakeholder to the governance system.
        
        Args:
            name: Stakeholder identifier
            weight: Voting weight (should sum to 1.0 across all stakeholders)
            reward_function: Optional custom reward function for this stakeholder
            
        Returns:
            Self for method chaining
        """
        if weight <= 0:
            raise ValueError("Stakeholder weight must be positive")
        
        self.stakeholder_weights[name] = weight
        if reward_function:
            self.stakeholder_rewards[name] = reward_function
        
        # Normalize weights
        self._normalize_weights()
        
        logging.info(f"Added stakeholder: {name} with weight {weight}")
        return self
    
    def register_reward_function(self, stakeholder: str):
        """Decorator for registering stakeholder-specific reward functions."""
        def decorator(func: Callable):
            if stakeholder not in self.stakeholder_weights:
                raise ValueError(f"Unknown stakeholder: {stakeholder}")
            
            self.stakeholder_rewards[stakeholder] = func
            logging.info(f"Registered reward function for stakeholder: {stakeholder}")
            return func
        return decorator
    
    def propose_amendment(
        self,
        proposer: str,
        title: str,
        description: str,
        changes: Dict[str, Any],
        duration_hours: float = 168.0  # 1 week default
    ) -> Amendment:
        """
        Propose an amendment to the contract.
        
        Args:
            proposer: Stakeholder proposing the amendment
            title: Amendment title
            description: Detailed description
            changes: Dictionary of proposed changes
            duration_hours: Voting period duration in hours
            
        Returns:
            Created amendment
        """
        if proposer not in self.stakeholder_weights:
            raise ValueError(f"Unknown proposer: {proposer}")
        
        amendment_id = f"amend_{int(time.time())}_{proposer}"
        
        amendment = Amendment(
            id=amendment_id,
            proposer=proposer,
            title=title,
            description=description,
            changes=changes,
            created_at=time.time(),
            expires_at=time.time() + (duration_hours * 3600),
            status=ProposalStatus.ACTIVE
        )
        
        self.amendments[amendment_id] = amendment
        
        logging.info(f"Amendment proposed: {amendment_id} by {proposer}")
        return amendment
    
    def vote(
        self,
        amendment_id: str,
        stakeholder: str,
        support: bool,
        reasoning: Optional[str] = None
    ) -> bool:
        """
        Cast a vote on an amendment.
        
        Args:
            amendment_id: ID of amendment to vote on
            stakeholder: Voting stakeholder
            support: True for support, False for opposition
            reasoning: Optional reasoning for the vote
            
        Returns:
            True if vote was recorded successfully
        """
        if amendment_id not in self.amendments:
            raise ValueError(f"Unknown amendment: {amendment_id}")
        
        if stakeholder not in self.stakeholder_weights:
            raise ValueError(f"Unknown stakeholder: {stakeholder}")
        
        amendment = self.amendments[amendment_id]
        
        # Check if amendment is still active
        if amendment.status != ProposalStatus.ACTIVE:
            raise ValueError(f"Amendment {amendment_id} is not active")
        
        if time.time() > amendment.expires_at:
            amendment.status = ProposalStatus.EXPIRED
            raise ValueError(f"Amendment {amendment_id} has expired")
        
        # Remove any existing vote from this stakeholder
        amendment.votes = [v for v in amendment.votes if v.stakeholder != stakeholder]
        
        # Add new vote
        vote = Vote(
            stakeholder=stakeholder,
            support=support,
            weight=self.stakeholder_weights[stakeholder],
            timestamp=time.time(),
            reasoning=reasoning
        )
        
        amendment.votes.append(vote)
        
        # Check if consensus is reached
        self._check_consensus(amendment)
        
        logging.info(f"Vote recorded: {stakeholder} {'supports' if support else 'opposes'} {amendment_id}")
        return True
    
    def get_consensus_status(self, amendment_id: str) -> Dict[str, Any]:
        """
        Get current consensus status for an amendment.
        
        Args:
            amendment_id: Amendment to check
            
        Returns:
            Dictionary with consensus information
        """
        if amendment_id not in self.amendments:
            raise ValueError(f"Unknown amendment: {amendment_id}")
        
        amendment = self.amendments[amendment_id]
        
        total_weight = sum(self.stakeholder_weights.values())
        support_weight = sum(v.weight for v in amendment.votes if v.support)
        opposition_weight = sum(v.weight for v in amendment.votes if not v.support)
        
        support_ratio = support_weight / total_weight if total_weight > 0 else 0
        participation_ratio = (support_weight + opposition_weight) / total_weight if total_weight > 0 else 0
        
        return {
            'amendment_id': amendment_id,
            'status': amendment.status.value,
            'support_ratio': support_ratio,
            'opposition_ratio': opposition_weight / total_weight if total_weight > 0 else 0,
            'participation_ratio': participation_ratio,
            'required_consensus': amendment.required_consensus,
            'has_consensus': support_ratio >= amendment.required_consensus,
            'time_remaining': max(0, amendment.expires_at - time.time()),
            'votes_cast': len(amendment.votes),
            'total_stakeholders': len(self.stakeholder_weights)
        }
    
    def apply_amendment(self, amendment_id: str) -> Dict[str, Any]:
        """
        Apply an approved amendment to the contract.
        
        Args:
            amendment_id: Amendment to apply
            
        Returns:
            Dictionary with application results
        """
        if amendment_id not in self.amendments:
            raise ValueError(f"Unknown amendment: {amendment_id}")
        
        amendment = self.amendments[amendment_id]
        
        if amendment.status != ProposalStatus.APPROVED:
            raise ValueError(f"Amendment {amendment_id} is not approved")
        
        # Record the change in history
        change_record = {
            'amendment_id': amendment_id,
            'applied_at': time.time(),
            'changes': amendment.changes,
            'applied_by': 'system'
        }
        
        self._preference_history.append(change_record)
        
        logging.info(f"Amendment applied: {amendment_id}")
        return change_record
    
    def aggregate_rewards(
        self,
        individual_rewards: Dict[str, float],
        strategy: Optional[str] = None
    ) -> float:
        """
        Aggregate individual stakeholder rewards.
        
        Args:
            individual_rewards: Dict mapping stakeholder names to rewards
            strategy: Optional aggregation strategy override
            
        Returns:
            Aggregated reward value
        """
        if not individual_rewards:
            return 0.0
        
        # Use weighted average by default
        total_weight = 0.0
        weighted_sum = 0.0
        
        for stakeholder, reward in individual_rewards.items():
            if stakeholder in self.stakeholder_weights:
                weight = self.stakeholder_weights[stakeholder]
                weighted_sum += weight * reward
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _normalize_weights(self):
        """Normalize stakeholder weights to sum to 1.0."""
        if not self.stakeholder_weights:
            return
        
        total_weight = sum(self.stakeholder_weights.values())
        if total_weight > 0:
            for stakeholder in self.stakeholder_weights:
                self.stakeholder_weights[stakeholder] /= total_weight
    
    def _check_consensus(self, amendment: Amendment):
        """Check if an amendment has reached consensus."""
        total_weight = sum(self.stakeholder_weights.values())
        support_weight = sum(v.weight for v in amendment.votes if v.support)
        
        support_ratio = support_weight / total_weight if total_weight > 0 else 0
        
        if support_ratio >= amendment.required_consensus:
            amendment.status = ProposalStatus.APPROVED
            logging.info(f"Amendment {amendment.id} approved with {support_ratio:.1%} support")
        elif time.time() > amendment.expires_at:
            amendment.status = ProposalStatus.EXPIRED
            logging.info(f"Amendment {amendment.id} expired without consensus")
    
    def get_active_amendments(self) -> List[Amendment]:
        """Get all active amendments."""
        return [a for a in self.amendments.values() if a.status == ProposalStatus.ACTIVE]
    
    def get_amendment_history(self) -> List[Dict[str, Any]]:
        """Get history of all amendments."""
        return [
            {
                'id': a.id,
                'proposer': a.proposer,
                'title': a.title,
                'status': a.status.value,
                'created_at': a.created_at,
                'votes': len(a.votes)
            }
            for a in self.amendments.values()
        ]
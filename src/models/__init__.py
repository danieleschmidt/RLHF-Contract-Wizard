"""
Models package for RLHF-Contract-Wizard.

Contains core data models and domain objects for the contract system.
"""

from .reward_contract import (
    RewardContract,
    Stakeholder,
    Constraint,
    ContractMetadata,
    AggregationStrategy
)

from .legal_blocks import (
    LegalBlocks,
    LegalBlock,
    ConstraintType,
    LegalBlocksParser,
    RLHFConstraints
)

from .stakeholder_preferences import (
    StakeholderPreferences,
    VotingStrategy,
    ProposalStatus,
    Vote,
    Amendment
)

__all__ = [
    # Reward Contract
    'RewardContract',
    'Stakeholder', 
    'Constraint',
    'ContractMetadata',
    'AggregationStrategy',
    
    # Legal Blocks
    'LegalBlocks',
    'LegalBlock',
    'ConstraintType',
    'LegalBlocksParser',
    'RLHFConstraints',
    
    # Stakeholder Preferences
    'StakeholderPreferences',
    'VotingStrategy',
    'ProposalStatus',
    'Vote',
    'Amendment'
]
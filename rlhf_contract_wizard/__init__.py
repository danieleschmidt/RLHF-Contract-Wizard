"""
RLHF Contract Wizard — encode reward functions as auditable model card contracts.
"""

from .contract import RewardContract, RewardClause, ClauseType
from .verifier import ContractVerifier, VerificationResult
from .generator import ContractGenerator

__all__ = [
    "RewardContract",
    "RewardClause",
    "ClauseType",
    "ContractVerifier",
    "VerificationResult",
    "ContractGenerator",
]

__version__ = "0.1.0"

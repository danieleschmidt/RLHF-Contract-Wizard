"""
RLHF-Contract-Wizard: Machine-Readable Contracts for AI Alignment

A JAX library that encodes RLHF reward functions directly in OpenChain 
machine-readable model cards, implementing Stanford's 2025 "Legal-Blocks" 
white paper for verifiable AI alignment.
"""

__version__ = "0.1.0-alpha"
__author__ = "Daniel Schmidt"
__email__ = "contact@rlhf-contracts.org"

# Core imports for easy access
from .models import (
    RewardContract,
    LegalBlocks,
    Stakeholder,
    AggregationStrategy,
    RLHFConstraints
)

from .services import (
    ContractService,
    VerificationService,
    BlockchainService
)

# Quantum task planner imports
from .quantum_planner import (
    QuantumTaskPlanner,
    QuantumTask,
    TaskState,
    ContractualTaskPlanner,
    QuantumPlannerVisualizer
)

# Convenience functions
def create_contract(
    name: str,
    stakeholders: dict,
    **kwargs
) -> RewardContract:
    """
    Create a new reward contract.
    
    Args:
        name: Contract name
        stakeholders: Dictionary of stakeholder weights
        **kwargs: Additional contract parameters
        
    Returns:
        New RewardContract instance
    """
    from .services import ContractService
    service = ContractService()
    return service.create_contract(name, stakeholders=stakeholders, **kwargs)


def verify_contract(contract: RewardContract) -> dict:
    """
    Verify contract properties.
    
    Args:
        contract: Contract to verify
        
    Returns:
        Verification results
    """
    from .services import VerificationService
    service = VerificationService()
    return service.verify_contract(contract).to_dict()


__all__ = [
    # Core classes
    'RewardContract',
    'LegalBlocks', 
    'Stakeholder',
    'AggregationStrategy',
    'RLHFConstraints',
    
    # Services
    'ContractService',
    'VerificationService',
    'BlockchainService',
    
    # Quantum planning
    'QuantumTaskPlanner',
    'QuantumTask',
    'TaskState',
    'ContractualTaskPlanner',
    'QuantumPlannerVisualizer',
    
    # Convenience functions
    'create_contract',
    'verify_contract',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__'
]
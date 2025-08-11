"""
RLHF-Contract-Wizard: Machine-Readable Contracts for AI Alignment

A JAX library that encodes RLHF reward functions directly in OpenChain 
machine-readable model cards, implementing Stanford's 2025 "Legal-Blocks" 
white paper for verifiable AI alignment.
"""

__version__ = "0.1.0-alpha"
__author__ = "Daniel Schmidt"
__email__ = "contact@rlhf-contracts.org"

# Core imports for easy access - with error handling for missing dependencies
try:
    from .models.reward_contract import (
        RewardContract,
        Stakeholder,
        AggregationStrategy,
        Constraint
    )
    _MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Models import failed: {e}")
    _MODELS_AVAILABLE = False

try:
    from .models.legal_blocks import LegalBlocks
except ImportError:
    LegalBlocks = None

try:
    from .services.contract_service import ContractService
except ImportError:
    ContractService = None

try:
    from .services.verification_service import VerificationService
except ImportError:
    VerificationService = None

try:
    from .services.blockchain_service import BlockchainService
except ImportError:
    BlockchainService = None

# Quantum task planner imports - simplified
try:
    from .quantum_planner.core import (
        QuantumTaskPlanner,
        QuantumTask,
        TaskState
    )
    _QUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Quantum planner import failed: {e}")
    QuantumTaskPlanner = None
    QuantumTask = None 
    TaskState = None
    _QUANTUM_AVAILABLE = False

# Convenience functions - only if models available
if _MODELS_AVAILABLE:
    def create_contract(
        name: str,
        stakeholders: dict,
        **kwargs
    ):
        """Create a new reward contract."""
        if ContractService:
            service = ContractService()
            return service.create_contract(name, stakeholders=stakeholders, **kwargs)
        else:
            return RewardContract(name, stakeholders=stakeholders, **kwargs)

    def verify_contract(contract):
        """Verify contract properties."""
        if VerificationService:
            service = VerificationService()
            return service.verify_contract(contract).to_dict()
        else:
            return {"status": "verification_service_unavailable"}

# Build exports based on what's available
__all__ = ['__version__', '__author__', '__email__']

if _MODELS_AVAILABLE:
    __all__.extend(['RewardContract', 'Stakeholder', 'AggregationStrategy', 'Constraint'])

if LegalBlocks:
    __all__.append('LegalBlocks')

if ContractService:
    __all__.append('ContractService')
    
if VerificationService:
    __all__.append('VerificationService')
    
if BlockchainService:
    __all__.append('BlockchainService')

if _QUANTUM_AVAILABLE:
    __all__.extend(['QuantumTaskPlanner', 'QuantumTask', 'TaskState'])

if _MODELS_AVAILABLE:
    __all__.extend(['create_contract', 'verify_contract'])
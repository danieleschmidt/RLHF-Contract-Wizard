"""
Services package for RLHF-Contract-Wizard.

Contains business logic and application services for contract management,
verification, and blockchain integration.
"""

# Import services with graceful error handling
try:
    from .contract_service import ContractService, ContractValidationError
    _CONTRACT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Contract service not available: {e}")
    ContractService = None
    ContractValidationError = None
    _CONTRACT_AVAILABLE = False

try:
    from .verification_service import (
        VerificationService,
        VerificationBackend,
        ProofResult,
        VerificationResult,
        Z3VerificationBackend,
        MockVerificationBackend
    )
    _VERIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Verification service not available: {e}")
    VerificationService = None
    VerificationBackend = None
    ProofResult = None
    VerificationResult = None
    Z3VerificationBackend = None
    MockVerificationBackend = None
    _VERIFICATION_AVAILABLE = False

try:
    from .blockchain_service import (
        BlockchainService,
        NetworkType,
        TransactionResult,
        ContractDeployment,
        Web3Backend,
        MockBlockchainBackend
    )
    _BLOCKCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Blockchain service not available: {e}")
    BlockchainService = None
    NetworkType = None
    TransactionResult = None
    ContractDeployment = None
    Web3Backend = None
    MockBlockchainBackend = None
    _BLOCKCHAIN_AVAILABLE = False

# Build __all__ based on what's available
__all__ = []

if _CONTRACT_AVAILABLE:
    __all__.extend(['ContractService', 'ContractValidationError'])

if _VERIFICATION_AVAILABLE:
    __all__.extend([
        'VerificationService',
        'VerificationBackend', 
        'ProofResult',
        'VerificationResult',
        'Z3VerificationBackend',
        'MockVerificationBackend'
    ])

if _BLOCKCHAIN_AVAILABLE:
    __all__.extend([
        'BlockchainService',
        'NetworkType',
        'TransactionResult', 
        'ContractDeployment',
        'Web3Backend',
        'MockBlockchainBackend'
    ])
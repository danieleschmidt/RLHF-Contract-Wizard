"""
Research Module - Novel Algorithms for RLHF Contract Optimization

This module contains breakthrough research implementations:
1. Quantum-Contract Hybrid Optimizer
2. ML-Based Security Vulnerability Prediction  
3. Comprehensive Performance Validation Framework

Author: Terry (Terragon Labs Research Division)
"""

from .quantum_contract_optimizer import (
    QuantumContractOptimizer,
    QuantumContractConfig,
    QuantumContractResult,
    QuantumState,
    VerificationMode
)

from .ml_security_predictor import (
    MLSecurityPredictor,
    VulnerabilityPrediction,
    VulnerabilityRiskLevel,
    AttackVector,
    SecurityFeatures,
    create_synthetic_training_data
)

from .performance_validation import (
    PerformanceValidator,
    ComparativeStudyResults,
    BenchmarkResult,
    ExperimentConfig,
    BenchmarkMetric,
    run_validation_demo
)

__version__ = "1.0.0"
__author__ = "Terry (Terragon Labs)"

# Research exports for integration
__all__ = [
    # Quantum optimization
    "QuantumContractOptimizer",
    "QuantumContractConfig", 
    "QuantumContractResult",
    "QuantumState",
    "VerificationMode",
    
    # ML security prediction
    "MLSecurityPredictor",
    "VulnerabilityPrediction",
    "VulnerabilityRiskLevel",
    "AttackVector",
    "SecurityFeatures",
    "create_synthetic_training_data",
    
    # Performance validation
    "PerformanceValidator",
    "ComparativeStudyResults",
    "BenchmarkResult",
    "ExperimentConfig",
    "BenchmarkMetric",
    "run_validation_demo"
]
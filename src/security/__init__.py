"""
Security module for RLHF Contract Wizard.

This module provides comprehensive security services including:
- Advanced threat detection and mitigation
- Cryptographic security framework
- Access control and authentication
- Security policy enforcement
- Audit logging and compliance monitoring
"""

from .comprehensive_security_framework import (
    ComprehensiveSecurityFramework,
    SecurityLevel,
    PermissionType,
    SecurityContext,
    CryptographicService,
    AccessControlService,
    AuditService,
    SecurityPolicyEngine
)

try:
    from .advanced_threat_detection import (
        AdvancedThreatDetectionSystem,
        ThreatSignature,
        SecurityEvent,
        AnomalyScore,
        StatisticalAnomalyDetector,
        BehavioralAnomalyDetector,
        CryptographicIntegrityChecker
    )
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Advanced threat detection not available: {e}")
    
    # Provide placeholder classes
    class AdvancedThreatDetectionSystem:
        def __init__(self, *args, **kwargs):
            raise ImportError("Advanced threat detection requires additional dependencies")

__all__ = [
    # Core security framework
    'ComprehensiveSecurityFramework',
    'SecurityLevel',
    'PermissionType', 
    'SecurityContext',
    'CryptographicService',
    'AccessControlService',
    'AuditService',
    'SecurityPolicyEngine',
    
    # Advanced threat detection (if available)
    'AdvancedThreatDetectionSystem',
    'ThreatSignature',
    'SecurityEvent',
    'AnomalyScore',
    'StatisticalAnomalyDetector',
    'BehavioralAnomalyDetector',
    'CryptographicIntegrityChecker'
]

# Version information
__version__ = '1.0.0'
__author__ = 'RLHF Contract Wizard Security Team'
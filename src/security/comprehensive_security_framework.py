"""
Comprehensive Security Framework for RLHF Contract Systems.

This module provides a unified security framework that integrates threat detection,
access control, audit logging, and security policy enforcement for RLHF contracts.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
import hashlib
import hmac
import secrets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from ..models.reward_contract import RewardContract
from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity
from .advanced_threat_detection import AdvancedThreatDetectionSystem, SecurityEvent


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = 0
    BASIC = 1
    RESTRICTED = 2
    CONFIDENTIAL = 3
    SECRET = 4
    TOP_SECRET = 5


class PermissionType(Enum):
    """Types of permissions."""
    READ = "read"
    WRITE = "write" 
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"
    AUDIT = "audit"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    permissions: Set[PermissionType]
    security_level: SecurityLevel
    source_ip: str
    user_agent: str
    authenticated_at: datetime
    expires_at: datetime
    mfa_verified: bool = False
    rate_limit_remaining: int = 1000


@dataclass
class AuditEntry:
    """Audit log entry."""
    id: str
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    outcome: str  # "success", "failure", "blocked"
    security_context: Dict[str, Any]
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enforcement_level: str  # "advisory", "warning", "blocking"
    applicable_resources: List[str]
    exceptions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class CryptographicService:
    """Cryptographic operations service."""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        self.public_key = self.private_key.public_key()
        
        # Symmetric encryption key for session data
        self.session_key = secrets.token_bytes(32)  # 256-bit key
    
    def sign_data(self, data: bytes) -> bytes:
        """Sign data with private key."""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes, public_key=None) -> bool:
        """Verify signature with public key."""
        key = public_key or self.public_key
        try:
            key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def encrypt_data(self, data: bytes) -> Tuple[bytes, bytes]:
        """Encrypt data with session key, return (encrypted_data, iv)."""
        iv = secrets.token_bytes(16)  # 128-bit IV
        
        cipher = Cipher(algorithms.AES(self.session_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data to multiple of 16 bytes
        padded_data = data + b'\x00' * (16 - len(data) % 16)
        
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        return encrypted, iv
    
    def decrypt_data(self, encrypted_data: bytes, iv: bytes) -> bytes:
        """Decrypt data with session key."""
        cipher = Cipher(algorithms.AES(self.session_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        decrypted = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Remove padding
        return decrypted.rstrip(b'\x00')
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def compute_hash(self, data: Union[str, bytes]) -> str:
        """Compute SHA-256 hash of data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        digest = hashes.Hash(hashes.SHA256())
        digest.update(data)
        return digest.finalize().hex()


class AccessControlService:
    """Role-based access control service."""
    
    def __init__(self):
        self.roles: Dict[str, Dict[str, Any]] = {}
        self.permissions: Dict[str, Set[PermissionType]] = {}
        self.resource_policies: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default security roles."""
        
        # Public user - minimal permissions
        self.roles["public"] = {
            "name": "Public User",
            "security_level": SecurityLevel.PUBLIC,
            "permissions": {PermissionType.READ},
            "resource_access": ["public_contracts", "public_docs"],
            "rate_limits": {"requests_per_minute": 60}
        }
        
        # Authenticated user - basic permissions
        self.roles["user"] = {
            "name": "Authenticated User", 
            "security_level": SecurityLevel.BASIC,
            "permissions": {PermissionType.READ, PermissionType.WRITE},
            "resource_access": ["user_contracts", "public_contracts"],
            "rate_limits": {"requests_per_minute": 300}
        }
        
        # Contract owner - full contract control
        self.roles["contract_owner"] = {
            "name": "Contract Owner",
            "security_level": SecurityLevel.RESTRICTED,
            "permissions": {PermissionType.READ, PermissionType.WRITE, 
                          PermissionType.EXECUTE, PermissionType.DELETE},
            "resource_access": ["owned_contracts", "user_contracts"],
            "rate_limits": {"requests_per_minute": 1000}
        }
        
        # Administrator - system administration
        self.roles["admin"] = {
            "name": "System Administrator",
            "security_level": SecurityLevel.SECRET,
            "permissions": {PermissionType.READ, PermissionType.WRITE, 
                          PermissionType.EXECUTE, PermissionType.DELETE, 
                          PermissionType.ADMIN, PermissionType.AUDIT},
            "resource_access": ["*"],
            "rate_limits": {"requests_per_minute": 5000}
        }
        
        # Security auditor - read-only + audit
        self.roles["auditor"] = {
            "name": "Security Auditor",
            "security_level": SecurityLevel.CONFIDENTIAL,
            "permissions": {PermissionType.READ, PermissionType.AUDIT},
            "resource_access": ["*"],
            "rate_limits": {"requests_per_minute": 2000}
        }
    
    def check_permission(
        self, 
        security_context: SecurityContext, 
        resource: str, 
        permission: PermissionType
    ) -> Tuple[bool, str]:
        """Check if user has permission for resource operation."""
        
        # Check if permission is granted
        if permission not in security_context.permissions:
            return False, f"Permission {permission.value} not granted"
        
        # Check security level requirements
        resource_policies = self.resource_policies.get(resource, [])
        for policy in resource_policies:
            required_level = policy.get("min_security_level", SecurityLevel.PUBLIC)
            if security_context.security_level.value < required_level.value:
                return False, f"Insufficient security level: requires {required_level.name}"
        
        # Check session validity
        if datetime.now() > security_context.expires_at:
            return False, "Session expired"
        
        # Check MFA requirements for sensitive operations
        if permission in {PermissionType.DELETE, PermissionType.ADMIN}:
            if not security_context.mfa_verified:
                return False, "Multi-factor authentication required"
        
        return True, "Access granted"
    
    def create_security_context(
        self, 
        user_id: str, 
        role: str,
        source_ip: str,
        user_agent: str,
        mfa_verified: bool = False
    ) -> SecurityContext:
        """Create security context for user session."""
        
        role_config = self.roles.get(role, self.roles["public"])
        session_id = secrets.token_urlsafe(32)
        
        # Session duration based on security level
        duration_hours = {
            SecurityLevel.PUBLIC: 24,
            SecurityLevel.BASIC: 12,
            SecurityLevel.RESTRICTED: 8,
            SecurityLevel.CONFIDENTIAL: 4,
            SecurityLevel.SECRET: 2,
            SecurityLevel.TOP_SECRET: 1
        }
        
        security_level = role_config["security_level"]
        session_duration = timedelta(hours=duration_hours[security_level])
        
        return SecurityContext(
            user_id=user_id,
            session_id=session_id,
            permissions=role_config["permissions"],
            security_level=security_level,
            source_ip=source_ip,
            user_agent=user_agent,
            authenticated_at=datetime.now(),
            expires_at=datetime.now() + session_duration,
            mfa_verified=mfa_verified,
            rate_limit_remaining=role_config["rate_limits"]["requests_per_minute"]
        )
    
    def add_resource_policy(
        self, 
        resource: str, 
        policy: Dict[str, Any]
    ):
        """Add security policy for a resource."""
        self.resource_policies[resource].append(policy)
    
    def get_user_permissions(self, user_id: str, role: str) -> Dict[str, Any]:
        """Get comprehensive user permissions and constraints."""
        role_config = self.roles.get(role, self.roles["public"])
        
        return {
            "permissions": list(role_config["permissions"]),
            "security_level": role_config["security_level"].name,
            "resource_access": role_config["resource_access"],
            "rate_limits": role_config["rate_limits"]
        }


class AuditService:
    """Comprehensive audit logging service."""
    
    def __init__(self, output_dir: Path = Path("audit_logs")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.audit_entries: List[AuditEntry] = []
        self.crypto_service = CryptographicService()
        
        # Setup secure audit logging
        self.logger = logging.getLogger("security_audit")
        self._setup_audit_logging()
    
    def _setup_audit_logging(self):
        """Setup tamper-evident audit logging."""
        log_file = self.output_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def log_action(
        self,
        security_context: SecurityContext,
        action: str,
        resource: str,
        outcome: str,
        details: Dict[str, Any] = None,
        risk_score: float = 0.0
    ) -> str:
        """Log security-relevant action with integrity protection."""
        
        entry_id = f"audit_{int(time.time())}_{secrets.token_hex(8)}"
        
        entry = AuditEntry(
            id=entry_id,
            timestamp=datetime.now(),
            user_id=security_context.user_id,
            action=action,
            resource=resource,
            outcome=outcome,
            security_context={
                "session_id": security_context.session_id,
                "source_ip": security_context.source_ip,
                "user_agent": security_context.user_agent,
                "security_level": security_context.security_level.name,
                "permissions": [p.value for p in security_context.permissions]
            },
            details=details or {},
            risk_score=risk_score
        )
        
        self.audit_entries.append(entry)
        
        # Create tamper-evident log entry
        log_data = {
            "id": entry.id,
            "timestamp": entry.timestamp.isoformat(),
            "user_id": entry.user_id,
            "action": entry.action,
            "resource": entry.resource,
            "outcome": entry.outcome,
            "risk_score": entry.risk_score,
            "details": entry.details
        }
        
        # Sign the log entry
        log_json = json.dumps(log_data, sort_keys=True)
        signature = self.crypto_service.sign_data(log_json.encode())
        
        # Log with signature
        self.logger.info(f"{log_json}|SIGNATURE:{signature.hex()}")
        
        # Trigger alerts for high-risk actions
        if risk_score > 0.7:
            await self._trigger_security_alert(entry)
        
        return entry_id
    
    async def _trigger_security_alert(self, entry: AuditEntry):
        """Trigger security alerts for high-risk actions."""
        alert_data = {
            "alert_type": "high_risk_action",
            "user_id": entry.user_id,
            "action": entry.action,
            "resource": entry.resource,
            "risk_score": entry.risk_score,
            "timestamp": entry.timestamp.isoformat(),
            "details": entry.details
        }
        
        # Save alert for immediate attention
        alert_file = self.output_dir / f"security_alert_{entry.id}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        # Log critical alert
        self.logger.critical(f"HIGH_RISK_ACTION: {json.dumps(alert_data)}")
    
    def verify_audit_integrity(self, entry_id: str) -> bool:
        """Verify integrity of audit log entry."""
        # Find the log entry in the file
        log_file = self.output_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        
        if not log_file.exists():
            return False
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if entry_id in line and "SIGNATURE:" in line:
                        parts = line.strip().split("|SIGNATURE:")
                        if len(parts) == 2:
                            log_data = parts[0].split(" - AUDIT - ")[1]
                            signature_hex = parts[1]
                            signature = bytes.fromhex(signature_hex)
                            
                            return self.crypto_service.verify_signature(
                                log_data.encode(), signature
                            )
            
        except Exception as e:
            self.logger.error(f"Error verifying audit integrity: {e}")
        
        return False
    
    def get_audit_report(
        self, 
        start_date: datetime = None, 
        end_date: datetime = None,
        user_id: str = None,
        risk_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        
        # Filter entries
        filtered_entries = self.audit_entries
        
        if start_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_date]
        
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_date]
        
        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]
        
        if risk_threshold > 0:
            filtered_entries = [e for e in filtered_entries if e.risk_score >= risk_threshold]
        
        # Generate statistics
        total_entries = len(filtered_entries)
        
        action_counts = {}
        outcome_counts = {}
        user_counts = {}
        risk_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for entry in filtered_entries:
            action_counts[entry.action] = action_counts.get(entry.action, 0) + 1
            outcome_counts[entry.outcome] = outcome_counts.get(entry.outcome, 0) + 1
            user_counts[entry.user_id] = user_counts.get(entry.user_id, 0) + 1
            
            if entry.risk_score < 0.3:
                risk_distribution["low"] += 1
            elif entry.risk_score < 0.6:
                risk_distribution["medium"] += 1
            elif entry.risk_score < 0.8:
                risk_distribution["high"] += 1
            else:
                risk_distribution["critical"] += 1
        
        return {
            "summary": {
                "total_entries": total_entries,
                "date_range": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None
                },
                "high_risk_actions": len([e for e in filtered_entries if e.risk_score > 0.7])
            },
            "statistics": {
                "actions": action_counts,
                "outcomes": outcome_counts,
                "users": user_counts,
                "risk_distribution": risk_distribution
            },
            "top_risks": sorted(
                [{"id": e.id, "action": e.action, "user": e.user_id, "risk": e.risk_score}
                 for e in filtered_entries],
                key=lambda x: x["risk"],
                reverse=True
            )[:10]
        }


class SecurityPolicyEngine:
    """Security policy definition and enforcement engine."""
    
    def __init__(self):
        self.policies: Dict[str, SecurityPolicy] = {}
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        
        # Contract modification policy
        self.policies["contract_modification"] = SecurityPolicy(
            name="Contract Modification Security",
            description="Governs security requirements for contract modifications",
            rules=[
                {
                    "condition": "action == 'modify_contract'",
                    "requirements": [
                        "mfa_required",
                        "min_security_level:RESTRICTED",
                        "audit_log_required"
                    ]
                },
                {
                    "condition": "modification_type == 'stakeholder_weights'",
                    "requirements": [
                        "stakeholder_approval_required",
                        "cooling_off_period:24h"
                    ]
                }
            ],
            enforcement_level="blocking",
            applicable_resources=["contracts", "reward_models"]
        )
        
        # Data access policy
        self.policies["data_access"] = SecurityPolicy(
            name="Data Access Control",
            description="Controls access to sensitive training and preference data",
            rules=[
                {
                    "condition": "resource_type == 'training_data'",
                    "requirements": [
                        "min_security_level:CONFIDENTIAL",
                        "data_classification_check",
                        "need_to_know_basis"
                    ]
                },
                {
                    "condition": "action == 'export_data'", 
                    "requirements": [
                        "admin_approval_required",
                        "encryption_required",
                        "audit_trail_complete"
                    ]
                }
            ],
            enforcement_level="blocking",
            applicable_resources=["datasets", "user_preferences", "model_weights"]
        )
        
        # API rate limiting policy
        self.policies["api_rate_limiting"] = SecurityPolicy(
            name="API Rate Limiting",
            description="Prevents abuse through excessive API requests",
            rules=[
                {
                    "condition": "api_endpoint == '/contracts/evaluate'",
                    "requirements": [
                        "max_requests_per_minute:100",
                        "burst_limit:20"
                    ]
                },
                {
                    "condition": "security_level == 'PUBLIC'",
                    "requirements": [
                        "max_requests_per_minute:60",
                        "concurrent_connections:5"
                    ]
                }
            ],
            enforcement_level="blocking",
            applicable_resources=["api_endpoints"]
        )
    
    def evaluate_policy(
        self,
        policy_name: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Evaluate security policy against given context.
        
        Returns:
            (allowed, requirements_met, violations)
        """
        
        if policy_name not in self.policies:
            return True, [], ["Policy not found"]
        
        policy = self.policies[policy_name]
        requirements_met = []
        violations = []
        
        for rule in policy.rules:
            condition = rule["condition"]
            
            # Simple condition evaluation (would need more sophisticated parser)
            if self._evaluate_condition(condition, context):
                requirements = rule["requirements"]
                
                for requirement in requirements:
                    if self._check_requirement(requirement, context):
                        requirements_met.append(requirement)
                    else:
                        violations.append(f"Requirement not met: {requirement}")
        
        allowed = len(violations) == 0
        
        # Apply enforcement level
        if policy.enforcement_level == "advisory" and violations:
            allowed = True  # Allow but log warnings
        
        return allowed, requirements_met, violations
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a policy condition against context."""
        
        # Simple condition parsing (production would use proper parser)
        try:
            # Handle equality conditions
            if "==" in condition:
                left, right = condition.split("==")
                left = left.strip()
                right = right.strip().strip("'\"")
                
                return context.get(left) == right
            
            # Add more condition types as needed
            return False
            
        except Exception:
            return False
    
    def _check_requirement(self, requirement: str, context: Dict[str, Any]) -> bool:
        """Check if a requirement is satisfied by the context."""
        
        if requirement == "mfa_required":
            return context.get("mfa_verified", False)
        
        elif requirement.startswith("min_security_level:"):
            required_level = requirement.split(":")[1]
            user_level = context.get("security_level", "PUBLIC")
            
            level_order = ["PUBLIC", "BASIC", "RESTRICTED", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]
            return level_order.index(user_level) >= level_order.index(required_level)
        
        elif requirement == "audit_log_required":
            return context.get("audit_enabled", True)
        
        elif requirement.startswith("max_requests_per_minute:"):
            limit = int(requirement.split(":")[1])
            current_rate = context.get("current_request_rate", 0)
            return current_rate <= limit
        
        elif requirement == "encryption_required":
            return context.get("encryption_enabled", False)
        
        # Add more requirement checks as needed
        return False
    
    def add_policy(self, policy: SecurityPolicy):
        """Add or update a security policy."""
        self.policies[policy.name] = policy
        policy.updated_at = datetime.now()
    
    def get_applicable_policies(self, resource: str) -> List[SecurityPolicy]:
        """Get all policies applicable to a resource."""
        applicable = []
        
        for policy in self.policies.values():
            if resource in policy.applicable_resources or "*" in policy.applicable_resources:
                applicable.append(policy)
        
        return applicable


class ComprehensiveSecurityFramework:
    """
    Unified security framework integrating all security services.
    
    Provides centralized security management with:
    - Threat detection and response
    - Access control and authentication
    - Audit logging and compliance
    - Policy enforcement
    - Cryptographic services
    """
    
    def __init__(
        self,
        output_dir: Path = Path("security_framework"),
        enable_auto_response: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize security services
        self.threat_detector = AdvancedThreatDetectionSystem(
            output_dir=self.output_dir / "threats",
            auto_mitigation=enable_auto_response
        )
        
        self.access_control = AccessControlService()
        self.audit_service = AuditService(output_dir=self.output_dir / "audits")
        self.policy_engine = SecurityPolicyEngine()
        self.crypto_service = CryptographicService()
        
        # Security state
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.security_incidents: List[Dict[str, Any]] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_framework_logging()
    
    def _setup_framework_logging(self):
        """Setup comprehensive security framework logging."""
        log_file = self.output_dir / "security_framework.log"
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - SECURITY_FRAMEWORK - %(levelname)s - %(message)s'
        ))
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def authenticate_user(
        self,
        user_id: str,
        credentials: Dict[str, Any],
        source_ip: str,
        user_agent: str
    ) -> Tuple[bool, Optional[SecurityContext], str]:
        """
        Authenticate user and create security context.
        
        Returns:
            (success, security_context, message)
        """
        
        try:
            # Basic credential validation (would integrate with real auth system)
            if not self._validate_credentials(user_id, credentials):
                await self.audit_service.log_action(
                    SecurityContext(
                        user_id=user_id,
                        session_id="unauthenticated",
                        permissions=set(),
                        security_level=SecurityLevel.PUBLIC,
                        source_ip=source_ip,
                        user_agent=user_agent,
                        authenticated_at=datetime.now(),
                        expires_at=datetime.now()
                    ),
                    action="authenticate",
                    resource="authentication_service",
                    outcome="failure",
                    details={"reason": "invalid_credentials"},
                    risk_score=0.6
                )
                
                return False, None, "Authentication failed"
            
            # Determine user role (would come from user database)
            user_role = self._get_user_role(user_id)
            
            # Check MFA if required
            mfa_verified = credentials.get("mfa_token") is not None
            if user_role in ["admin", "auditor"] and not mfa_verified:
                return False, None, "Multi-factor authentication required"
            
            # Create security context
            security_context = self.access_control.create_security_context(
                user_id=user_id,
                role=user_role,
                source_ip=source_ip,
                user_agent=user_agent,
                mfa_verified=mfa_verified
            )
            
            # Store active session
            self.active_sessions[security_context.session_id] = security_context
            
            # Log successful authentication
            await self.audit_service.log_action(
                security_context,
                action="authenticate",
                resource="authentication_service",
                outcome="success",
                details={"role": user_role, "mfa_verified": mfa_verified}
            )
            
            self.logger.info(f"User {user_id} authenticated successfully with role {user_role}")
            
            return True, security_context, "Authentication successful"
            
        except Exception as e:
            self.logger.error(f"Authentication error for user {user_id}: {e}")
            return False, None, "Authentication system error"
    
    def _validate_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Validate user credentials (simplified implementation)."""
        # In production, this would check against secure credential store
        password = credentials.get("password", "")
        
        # Basic validation (would use proper password hashing)
        if len(password) < 8:
            return False
        
        # Simulate credential database check
        return True  # Simplified for demo
    
    def _get_user_role(self, user_id: str) -> str:
        """Get user role from user database (simplified implementation)."""
        # In production, this would query user database
        
        # Admin users
        if user_id.startswith("admin_"):
            return "admin"
        
        # Auditor users
        elif user_id.startswith("auditor_"):
            return "auditor"
        
        # Contract owners
        elif user_id.startswith("owner_"):
            return "contract_owner"
        
        # Default to regular user
        else:
            return "user"
    
    async def authorize_operation(
        self,
        security_context: SecurityContext,
        operation: str,
        resource: str,
        additional_context: Dict[str, Any] = None
    ) -> Tuple[bool, str]:
        """
        Authorize operation with comprehensive security checks.
        
        Returns:
            (authorized, message)
        """
        
        try:
            # Check session validity
            if security_context.session_id not in self.active_sessions:
                return False, "Invalid session"
            
            if datetime.now() > security_context.expires_at:
                return False, "Session expired"
            
            # Determine required permission
            permission_mapping = {
                "read_contract": PermissionType.READ,
                "modify_contract": PermissionType.WRITE,
                "execute_contract": PermissionType.EXECUTE,
                "delete_contract": PermissionType.DELETE,
                "admin_operation": PermissionType.ADMIN,
                "audit_access": PermissionType.AUDIT
            }
            
            required_permission = permission_mapping.get(operation, PermissionType.READ)
            
            # Check basic permissions
            allowed, message = self.access_control.check_permission(
                security_context, resource, required_permission
            )
            
            if not allowed:
                await self.audit_service.log_action(
                    security_context,
                    action=operation,
                    resource=resource,
                    outcome="blocked",
                    details={"reason": message},
                    risk_score=0.3
                )
                return False, message
            
            # Evaluate security policies
            context = {
                "action": operation,
                "resource_type": resource.split(":")[0] if ":" in resource else resource,
                "security_level": security_context.security_level.name,
                "mfa_verified": security_context.mfa_verified,
                "user_id": security_context.user_id,
                **(additional_context or {})
            }
            
            # Check applicable policies
            applicable_policies = self.policy_engine.get_applicable_policies(resource)
            
            for policy in applicable_policies:
                policy_allowed, requirements_met, violations = self.policy_engine.evaluate_policy(
                    policy.name, context
                )
                
                if not policy_allowed:
                    await self.audit_service.log_action(
                        security_context,
                        action=operation,
                        resource=resource,
                        outcome="blocked",
                        details={
                            "policy": policy.name,
                            "violations": violations,
                            "requirements_met": requirements_met
                        },
                        risk_score=0.5
                    )
                    
                    return False, f"Policy violation: {policy.name} - {', '.join(violations)}"
            
            # Rate limiting check
            if security_context.rate_limit_remaining <= 0:
                await self.audit_service.log_action(
                    security_context,
                    action=operation,
                    resource=resource,
                    outcome="blocked",
                    details={"reason": "rate_limit_exceeded"},
                    risk_score=0.4
                )
                return False, "Rate limit exceeded"
            
            # Update rate limit
            security_context.rate_limit_remaining -= 1
            
            # Log successful authorization
            await self.audit_service.log_action(
                security_context,
                action=operation,
                resource=resource,
                outcome="success",
                details={"requirements_met": [p.name for p in applicable_policies]}
            )
            
            return True, "Operation authorized"
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False, "Authorization system error"
    
    async def secure_contract_interaction(
        self,
        security_context: SecurityContext,
        contract: RewardContract,
        operation: str,
        interaction_data: Dict[str, Any]
    ) -> Tuple[bool, Any, List[SecurityEvent]]:
        """
        Perform secure contract interaction with comprehensive monitoring.
        
        Returns:
            (success, result, security_events)
        """
        
        try:
            # Authorization check
            authorized, auth_message = await self.authorize_operation(
                security_context,
                operation, 
                f"contract:{contract.metadata.name}",
                {
                    "contract_id": contract.compute_hash(),
                    "operation_type": operation
                }
            )
            
            if not authorized:
                return False, {"error": auth_message}, []
            
            # Enhanced interaction data with security context
            enhanced_interaction = {
                **interaction_data,
                "security_context": {
                    "user_id": security_context.user_id,
                    "session_id": security_context.session_id,
                    "security_level": security_context.security_level.name
                },
                "contract_hash": contract.compute_hash(),
                "operation": operation,
                "timestamp": datetime.now().isoformat()
            }
            
            # Threat detection monitoring
            security_events = await self.threat_detector.monitor_contract_interaction(
                contract,
                enhanced_interaction,
                {
                    "source_ip": security_context.source_ip,
                    "user_id": security_context.user_id,
                    "role": "user",  # Would get from context
                    "session_id": security_context.session_id
                }
            )
            
            # Check for critical security events
            critical_events = [e for e in security_events if e.severity == "critical"]
            if critical_events:
                await self.audit_service.log_action(
                    security_context,
                    action=operation,
                    resource=f"contract:{contract.metadata.name}",
                    outcome="blocked",
                    details={
                        "reason": "critical_security_event",
                        "events": [e.description for e in critical_events]
                    },
                    risk_score=1.0
                )
                
                return False, {"error": "Operation blocked due to security concerns"}, security_events
            
            # Proceed with operation (would call actual contract method)
            result = self._execute_contract_operation(contract, operation, interaction_data)
            
            # Log successful operation
            await self.audit_service.log_action(
                security_context,
                action=operation,
                resource=f"contract:{contract.metadata.name}",
                outcome="success",
                details={
                    "result_summary": str(result)[:100] if result else "None",
                    "security_events_count": len(security_events)
                },
                risk_score=max(0.1, max([e.confidence for e in security_events] + [0.0]))
            )
            
            return True, result, security_events
            
        except Exception as e:
            self.logger.error(f"Secure contract interaction error: {e}")
            
            await self.audit_service.log_action(
                security_context,
                action=operation,
                resource=f"contract:{contract.metadata.name}",
                outcome="error",
                details={"error": str(e)},
                risk_score=0.7
            )
            
            return False, {"error": "System error during operation"}, []
    
    def _execute_contract_operation(
        self,
        contract: RewardContract,
        operation: str,
        data: Dict[str, Any]
    ) -> Any:
        """Execute the actual contract operation (simplified implementation)."""
        
        if operation == "read_contract":
            return contract.to_dict()
        
        elif operation == "evaluate_reward":
            state = data.get("state", jnp.ones(10))
            action = data.get("action", jnp.ones(5))
            return {"reward": contract.compute_reward(state, action)}
        
        elif operation == "modify_contract":
            # Would modify contract (simplified)
            return {"status": "modification_simulated"}
        
        else:
            return {"status": f"operation_{operation}_completed"}
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        
        # Threat detection summary
        threat_summary = self.threat_detector.get_security_summary()
        
        # Audit summary
        audit_report = self.audit_service.get_audit_report(
            start_date=datetime.now() - timedelta(days=7)
        )
        
        # Session statistics
        active_session_count = len(self.active_sessions)
        session_by_level = {}
        for session in self.active_sessions.values():
            level = session.security_level.name
            session_by_level[level] = session_by_level.get(level, 0) + 1
        
        return {
            "summary": {
                "security_health_score": threat_summary["security_health_score"],
                "active_sessions": active_session_count,
                "recent_threats": threat_summary["recent_events"],
                "audit_entries_week": audit_report["summary"]["total_entries"]
            },
            "threats": {
                "total_events": threat_summary["total_events"],
                "by_severity": threat_summary["events_by_severity"],
                "most_common": threat_summary["most_common_threats"],
                "blocked_ips": threat_summary["blocked_ips"]
            },
            "access_control": {
                "active_sessions": active_session_count,
                "sessions_by_level": session_by_level,
                "policies_count": len(self.policy_engine.policies)
            },
            "audit": {
                "total_entries": audit_report["summary"]["total_entries"],
                "high_risk_actions": audit_report["summary"]["high_risk_actions"],
                "top_risks": audit_report["top_risks"][:5]
            }
        }
    
    async def security_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive security health assessment."""
        
        health_score = 100.0
        issues = []
        recommendations = []
        
        # Check threat detection health
        threat_summary = self.threat_detector.get_security_summary()
        if threat_summary["active_threats"] > 0:
            health_score -= threat_summary["active_threats"] * 10
            issues.append(f"{threat_summary['active_threats']} active security threats")
        
        # Check session security
        expired_sessions = 0
        for session in self.active_sessions.values():
            if datetime.now() > session.expires_at:
                expired_sessions += 1
        
        if expired_sessions > 0:
            health_score -= expired_sessions * 2
            issues.append(f"{expired_sessions} expired sessions not cleaned up")
            recommendations.append("Implement automatic session cleanup")
        
        # Check audit log integrity (sample check)
        recent_entries = [e for e in self.audit_service.audit_entries 
                         if (datetime.now() - e.timestamp).total_seconds() < 3600]
        
        integrity_checks = 0
        for entry in recent_entries[:10]:  # Check recent 10 entries
            if self.audit_service.verify_audit_integrity(entry.id):
                integrity_checks += 1
        
        if len(recent_entries) > 0:
            integrity_rate = integrity_checks / min(len(recent_entries), 10)
            if integrity_rate < 1.0:
                health_score -= (1 - integrity_rate) * 30
                issues.append(f"Audit integrity issues detected ({integrity_rate:.1%} verified)")
        
        # Check policy compliance
        policy_violations = sum(
            1 for entry in recent_entries 
            if entry.outcome == "blocked" and "policy" in entry.details.get("reason", "")
        )
        
        if policy_violations > 10:
            health_score -= 5
            issues.append(f"{policy_violations} policy violations in last hour")
            recommendations.append("Review and update security policies")
        
        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "overall_health": {
                "score": max(0, health_score),
                "status": status,
                "assessment_time": datetime.now().isoformat()
            },
            "issues_identified": issues,
            "recommendations": recommendations,
            "component_health": {
                "threat_detection": threat_summary["security_health_score"],
                "access_control": 95.0,  # Would calculate based on metrics
                "audit_system": integrity_rate * 100 if 'integrity_rate' in locals() else 100.0,
                "policy_engine": max(0, 100 - policy_violations)
            }
        }


# Example usage and integration test
if __name__ == "__main__":
    
    async def main():
        print("🔒 Initializing Comprehensive Security Framework...")
        
        # Initialize security framework
        security_framework = ComprehensiveSecurityFramework(
            output_dir=Path("security_demo"),
            enable_auto_response=True
        )
        
        # Test authentication
        print("\\n🔑 Testing Authentication...")
        success, security_context, message = await security_framework.authenticate_user(
            user_id="test_user_001",
            credentials={"password": "secure_password_123"},
            source_ip="192.168.1.100",
            user_agent="TestClient/1.0"
        )
        
        print(f"   Authentication: {'✅ Success' if success else '❌ Failed'} - {message}")
        
        if success and security_context:
            # Test authorization
            print("\\n🛡️ Testing Authorization...")
            authorized, auth_message = await security_framework.authorize_operation(
                security_context,
                operation="read_contract",
                resource="contract:test_contract"
            )
            
            print(f"   Authorization: {'✅ Allowed' if authorized else '❌ Denied'} - {auth_message}")
            
            # Test secure contract interaction
            print("\\n📋 Testing Secure Contract Interaction...")
            contract = RewardContract(
                name="security_test_contract",
                stakeholders={"user": 0.8, "safety": 0.2}
            )
            
            success, result, events = await security_framework.secure_contract_interaction(
                security_context,
                contract,
                "evaluate_reward",
                {
                    "state": jnp.ones(10),
                    "action": jnp.ones(5),
                    "request_id": "test_req_001"
                }
            )
            
            print(f"   Contract Interaction: {'✅ Success' if success else '❌ Failed'}")
            print(f"   Security Events: {len(events)} detected")
            
            for event in events:
                print(f"     - {event.description} (Severity: {event.severity})")
        
        # Test security health check
        print("\\n🏥 Running Security Health Check...")
        health_report = await security_framework.security_health_check()
        
        print(f"   Overall Health: {health_report['overall_health']['score']:.1f}/100 ({health_report['overall_health']['status']})")
        
        if health_report['issues_identified']:
            print("   Issues:")
            for issue in health_report['issues_identified']:
                print(f"     - {issue}")
        
        # Get security dashboard
        print("\\n📊 Security Dashboard Summary...")
        dashboard = await security_framework.get_security_dashboard()
        
        print(f"   Security Health Score: {dashboard['summary']['security_health_score']:.1f}/100")
        print(f"   Active Sessions: {dashboard['summary']['active_sessions']}")
        print(f"   Recent Threats: {dashboard['summary']['recent_threats']}")
        print(f"   Audit Entries (Week): {dashboard['summary']['audit_entries_week']}")
        
        print("\\n✅ Comprehensive Security Framework demonstration completed!")
        print("📁 Security logs and data saved to: security_demo/")
    
    asyncio.run(main())
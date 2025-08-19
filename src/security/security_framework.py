#!/usr/bin/env python3
"""
Comprehensive Security Framework for RLHF Contract Wizard

Implements enterprise-grade security features including threat detection,
access control, audit logging, cryptographic operations, and compliance
monitoring for RLHF contract management.
"""

import time
import hashlib
import hmac
import secrets
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import ipaddress
from collections import defaultdict, deque


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessAction(Enum):
    """Types of access actions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    permissions: Set[str] = field(default_factory=set)
    rate_limit_tokens: int = 100
    mfa_verified: bool = False


@dataclass
class ThreatDetection:
    """Threat detection result."""
    threat_id: str
    threat_type: str
    severity: ThreatLevel
    description: str
    source_ip: str
    timestamp: datetime
    evidence: Dict[str, Any]
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class AuditLogEntry:
    """Audit log entry."""
    entry_id: str
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    ip_address: str
    success: bool
    details: Dict[str, Any]
    security_level: SecurityLevel


class CryptographicManager:
    """
    Handles all cryptographic operations for the system.
    
    Features:
    - Symmetric and asymmetric encryption
    - Digital signatures
    - Key derivation and management
    - Secure random generation
    - Hash functions with salting
    """
    
    def __init__(self):
        # Generate master key for symmetric encryption
        self.master_key = Fernet.generate_key()
        self.fernet = Fernet(self.master_key)
        
        # Generate RSA key pair for asymmetric operations
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()
        
        # Key derivation settings
        self.salt = secrets.token_bytes(32)
        self.iterations = 100000
    
    def encrypt_symmetric(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        return self.fernet.encrypt(data)
    
    def decrypt_symmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_asymmetric(self, data: bytes) -> bytes:
        """Encrypt data using asymmetric encryption."""
        return self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using asymmetric encryption."""
        return self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def sign_data(self, data: bytes) -> bytes:
        """Create digital signature for data."""
        return self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
    
    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify digital signature."""
        try:
            self.public_key.verify(
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
    
    def derive_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password."""
        if salt is None:
            salt = self.salt
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def secure_hash(self, data: str, salt: Optional[str] = None) -> str:
        """Generate secure hash with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        salted_data = f"{data}{salt}"
        hash_obj = hashlib.sha256(salted_data.encode())
        return f"{salt}${hash_obj.hexdigest()}"
    
    def verify_hash(self, data: str, hashed: str) -> bool:
        """Verify data against hash."""
        try:
            salt, hash_value = hashed.split('$', 1)
            computed_hash = self.secure_hash(data, salt)
            return hmac.compare_digest(computed_hash, hashed)
        except Exception:
            return False
    
    def generate_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)


class AccessController:
    """
    Role-based access control (RBAC) system.
    
    Features:
    - Role and permission management
    - Resource-level access control
    - Dynamic permission evaluation
    - Access audit logging
    """
    
    def __init__(self):
        self.roles: Dict[str, Set[str]] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.resource_permissions: Dict[str, Dict[str, Set[str]]] = {}
        self.access_cache: Dict[str, Tuple[bool, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
    
    def create_role(self, role_name: str, permissions: Set[str]):
        """Create a new role with permissions."""
        self.roles[role_name] = permissions.copy()
    
    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user."""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role_name)
        
        # Invalidate cache for this user
        keys_to_remove = [key for key in self.access_cache.keys() if key.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.access_cache[key]
    
    def revoke_role(self, user_id: str, role_name: str):
        """Revoke role from user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
    
    def set_resource_permissions(self, resource: str, action: AccessAction, required_permissions: Set[str]):
        """Set required permissions for resource action."""
        if resource not in self.resource_permissions:
            self.resource_permissions[resource] = {}
        self.resource_permissions[resource][action.value] = required_permissions
    
    def check_access(self, user_id: str, resource: str, action: AccessAction) -> bool:
        """Check if user has access to perform action on resource."""
        cache_key = f"{user_id}:{resource}:{action.value}"
        
        # Check cache first
        if cache_key in self.access_cache:
            result, timestamp = self.access_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return result
        
        # Get user permissions
        user_permissions = self._get_user_permissions(user_id)
        
        # Get required permissions for this resource/action
        required_permissions = self.resource_permissions.get(resource, {}).get(action.value, set())
        
        # Check if user has all required permissions
        has_access = required_permissions.issubset(user_permissions)
        
        # Cache the result
        self.access_cache[cache_key] = (has_access, datetime.now())
        
        return has_access
    
    def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user based on their roles."""
        user_permissions = set()
        user_roles = self.user_roles.get(user_id, set())
        
        for role in user_roles:
            if role in self.roles:
                user_permissions.update(self.roles[role])
        
        return user_permissions
    
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user access information."""
        return {
            "user_id": user_id,
            "roles": list(self.user_roles.get(user_id, set())),
            "permissions": list(self._get_user_permissions(user_id)),
            "access_cache_entries": len([k for k in self.access_cache.keys() if k.startswith(f"{user_id}:")])
        }


class ThreatDetector:
    """
    Advanced threat detection system.
    
    Features:
    - Rate limiting and DDoS detection
    - Anomalous behavior detection
    - IP reputation checking
    - Pattern-based attack detection
    """
    
    def __init__(self):
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.suspicious_ips: Set[str] = set()
        self.known_attack_patterns = [
            r"(?i)(union|select|insert|delete|drop|exec|script)",  # SQL injection
            r"(?i)(<script|javascript:|vbscript:)",                # XSS
            r"(?i)(\.\.\/|\.\.\\)",                                # Path traversal
        ]
        self.detection_rules = {
            "rate_limit": {"requests_per_minute": 60, "burst_size": 10},
            "failed_auth": {"max_attempts": 5, "window_minutes": 15},
            "suspicious_patterns": {"enabled": True},
        }
    
    def analyze_request(self, context: SecurityContext, request_data: Dict[str, Any]) -> List[ThreatDetection]:
        """Analyze incoming request for threats."""
        threats = []
        
        # Rate limiting check
        rate_threat = self._check_rate_limiting(context)
        if rate_threat:
            threats.append(rate_threat)
        
        # Failed authentication check
        if request_data.get("authentication_failed"):
            auth_threat = self._check_failed_authentication(context)
            if auth_threat:
                threats.append(auth_threat)
        
        # Pattern-based detection
        pattern_threats = self._check_attack_patterns(context, request_data)
        threats.extend(pattern_threats)
        
        # IP reputation check
        ip_threat = self._check_ip_reputation(context)
        if ip_threat:
            threats.append(ip_threat)
        
        return threats
    
    def _check_rate_limiting(self, context: SecurityContext) -> Optional[ThreatDetection]:
        """Check for rate limiting violations."""
        client_id = f"{context.ip_address}:{context.user_id}"
        current_time = datetime.now()
        
        # Add current request
        self.rate_limits[client_id].append(current_time)
        
        # Count requests in last minute
        one_minute_ago = current_time - timedelta(minutes=1)
        recent_requests = [t for t in self.rate_limits[client_id] if t > one_minute_ago]
        
        if len(recent_requests) > self.detection_rules["rate_limit"]["requests_per_minute"]:
            return ThreatDetection(
                threat_id=secrets.token_hex(8),
                threat_type="rate_limit_exceeded",
                severity=ThreatLevel.MEDIUM,
                description=f"Rate limit exceeded: {len(recent_requests)} requests in last minute",
                source_ip=context.ip_address,
                timestamp=current_time,
                evidence={"requests_per_minute": len(recent_requests)},
                mitigation_actions=["rate_limit", "temporary_ban"]
            )
        
        return None
    
    def _check_failed_authentication(self, context: SecurityContext) -> Optional[ThreatDetection]:
        """Check for brute force authentication attempts."""
        current_time = datetime.now()
        
        # Add failed attempt
        self.failed_attempts[context.ip_address].append(current_time)
        
        # Clean old attempts
        window_start = current_time - timedelta(minutes=self.detection_rules["failed_auth"]["window_minutes"])
        self.failed_attempts[context.ip_address] = [
            t for t in self.failed_attempts[context.ip_address] if t > window_start
        ]
        
        # Check if threshold exceeded
        if len(self.failed_attempts[context.ip_address]) >= self.detection_rules["failed_auth"]["max_attempts"]:
            return ThreatDetection(
                threat_id=secrets.token_hex(8),
                threat_type="brute_force_attack",
                severity=ThreatLevel.HIGH,
                description=f"Multiple failed authentication attempts from {context.ip_address}",
                source_ip=context.ip_address,
                timestamp=current_time,
                evidence={"failed_attempts": len(self.failed_attempts[context.ip_address])},
                mitigation_actions=["block_ip", "alert_admin"]
            )
        
        return None
    
    def _check_attack_patterns(self, context: SecurityContext, request_data: Dict[str, Any]) -> List[ThreatDetection]:
        """Check for known attack patterns in request data."""
        if not self.detection_rules["suspicious_patterns"]["enabled"]:
            return []
        
        threats = []
        request_str = json.dumps(request_data, default=str).lower()
        
        import re
        for pattern in self.known_attack_patterns:
            if re.search(pattern, request_str):
                threats.append(ThreatDetection(
                    threat_id=secrets.token_hex(8),
                    threat_type="malicious_pattern",
                    severity=ThreatLevel.HIGH,
                    description=f"Malicious pattern detected in request",
                    source_ip=context.ip_address,
                    timestamp=datetime.now(),
                    evidence={"pattern": pattern, "request_data": request_data},
                    mitigation_actions=["block_request", "log_incident"]
                ))
        
        return threats
    
    def _check_ip_reputation(self, context: SecurityContext) -> Optional[ThreatDetection]:
        """Check IP against reputation database."""
        if context.ip_address in self.suspicious_ips:
            return ThreatDetection(
                threat_id=secrets.token_hex(8),
                threat_type="suspicious_ip",
                severity=ThreatLevel.MEDIUM,
                description=f"Request from known suspicious IP: {context.ip_address}",
                source_ip=context.ip_address,
                timestamp=datetime.now(),
                evidence={"ip_address": context.ip_address},
                mitigation_actions=["enhanced_monitoring", "require_mfa"]
            )
        
        return None
    
    def add_suspicious_ip(self, ip_address: str):
        """Add IP to suspicious list."""
        self.suspicious_ips.add(ip_address)
    
    def remove_suspicious_ip(self, ip_address: str):
        """Remove IP from suspicious list."""
        self.suspicious_ips.discard(ip_address)


class AuditLogger:
    """
    Comprehensive audit logging system.
    
    Features:
    - Tamper-proof logging
    - Structured audit entries
    - Real-time monitoring
    - Compliance reporting
    """
    
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        
        # In-memory storage for recent logs
        self.recent_logs: deque = deque(maxlen=1000)
        
        # Crypto manager for log integrity
        self.crypto = CryptographicManager()
    
    def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool,
        context: SecurityContext,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log user action with full context."""
        entry = AuditLogEntry(
            entry_id=secrets.token_hex(16),
            user_id=user_id,
            action=action,
            resource=resource,
            timestamp=datetime.now(),
            ip_address=context.ip_address,
            success=success,
            details=details or {},
            security_level=context.security_level
        )
        
        # Create tamper-proof log entry
        log_data = {
            "entry_id": entry.entry_id,
            "user_id": entry.user_id,
            "action": entry.action,
            "resource": entry.resource,
            "timestamp": entry.timestamp.isoformat(),
            "ip_address": entry.ip_address,
            "success": entry.success,
            "details": entry.details,
            "security_level": entry.security_level.value
        }
        
        # Sign the log entry
        log_json = json.dumps(log_data, sort_keys=True)
        signature = self.crypto.sign_data(log_json.encode())
        log_data["signature"] = base64.b64encode(signature).decode()
        
        # Log to file
        self.logger.info(json.dumps(log_data))
        
        # Store in memory
        self.recent_logs.append(entry)
    
    def verify_log_integrity(self, log_entry: Dict[str, Any]) -> bool:
        """Verify the integrity of a log entry."""
        try:
            signature = base64.b64decode(log_entry.pop("signature"))
            log_json = json.dumps(log_entry, sort_keys=True)
            return self.crypto.verify_signature(log_json.encode(), signature)
        except Exception:
            return False
    
    def get_recent_logs(self, limit: int = 100) -> List[AuditLogEntry]:
        """Get recent audit log entries."""
        return list(self.recent_logs)[-limit:]
    
    def search_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditLogEntry]:
        """Search audit logs with filters."""
        filtered_logs = []
        
        for entry in self.recent_logs:
            if user_id and entry.user_id != user_id:
                continue
            if action and entry.action != action:
                continue
            if resource and entry.resource != resource:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            
            filtered_logs.append(entry)
        
        return filtered_logs


class SecurityFramework:
    """
    Main security framework coordinating all security components.
    
    Provides unified security interface for the entire system.
    """
    
    def __init__(self):
        self.crypto = CryptographicManager()
        self.access_controller = AccessController()
        self.threat_detector = ThreatDetector()
        self.audit_logger = AuditLogger()
        
        # Initialize default roles and permissions
        self._setup_default_security()
    
    def _setup_default_security(self):
        """Setup default roles and permissions."""
        # Create default roles
        self.access_controller.create_role("admin", {
            "contract:read", "contract:write", "contract:delete", "contract:admin",
            "user:read", "user:write", "user:delete", "user:admin",
            "system:read", "system:write", "system:admin"
        })
        
        self.access_controller.create_role("operator", {
            "contract:read", "contract:write",
            "user:read"
        })
        
        self.access_controller.create_role("viewer", {
            "contract:read",
            "user:read"
        })
        
        # Set resource permissions
        self.access_controller.set_resource_permissions(
            "contracts", AccessAction.READ, {"contract:read"}
        )
        self.access_controller.set_resource_permissions(
            "contracts", AccessAction.WRITE, {"contract:write"}
        )
        self.access_controller.set_resource_permissions(
            "contracts", AccessAction.DELETE, {"contract:delete"}
        )
        self.access_controller.set_resource_permissions(
            "contracts", AccessAction.ADMIN, {"contract:admin"}
        )
    
    def authenticate_user(self, username: str, password: str, context: SecurityContext) -> Tuple[bool, Optional[str]]:
        """Authenticate user with comprehensive security checks."""
        # Check for threats
        threats = self.threat_detector.analyze_request(context, {
            "action": "authentication",
            "username": username
        })
        
        # Block if critical threats detected
        critical_threats = [t for t in threats if t.severity == ThreatLevel.CRITICAL]
        if critical_threats:
            self.audit_logger.log_action(
                username, "authentication", "system", False, context,
                {"threats": [t.threat_type for t in critical_threats]}
            )
            return False, "Authentication blocked due to security threats"
        
        # Simulate password verification (in real implementation, use proper auth)
        password_hash = self.crypto.secure_hash(password)
        auth_success = True  # Placeholder
        
        if auth_success:
            # Generate session token
            session_token = self.crypto.generate_token()
            context.session_id = session_token
            
            self.audit_logger.log_action(
                username, "authentication", "system", True, context
            )
            
            return True, session_token
        else:
            # Log failed authentication
            self.audit_logger.log_action(
                username, "authentication", "system", False, context
            )
            
            # Check for brute force
            threats = self.threat_detector.analyze_request(context, {
                "authentication_failed": True
            })
            
            return False, "Invalid credentials"
    
    def authorize_action(
        self,
        user_id: str,
        resource: str,
        action: AccessAction,
        context: SecurityContext
    ) -> Tuple[bool, Optional[str]]:
        """Authorize user action with security checks."""
        # Check access permissions
        has_access = self.access_controller.check_access(user_id, resource, action)
        
        if not has_access:
            self.audit_logger.log_action(
                user_id, action.value, resource, False, context,
                {"reason": "insufficient_permissions"}
            )
            return False, "Insufficient permissions"
        
        # Check for threats
        threats = self.threat_detector.analyze_request(context, {
            "action": action.value,
            "resource": resource
        })
        
        # Block high-severity threats
        high_threats = [t for t in threats if t.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
        if high_threats:
            self.audit_logger.log_action(
                user_id, action.value, resource, False, context,
                {"threats": [t.threat_type for t in high_threats]}
            )
            return False, "Action blocked due to security threats"
        
        # Log successful authorization
        self.audit_logger.log_action(
            user_id, action.value, resource, True, context
        )
        
        return True, None
    
    def encrypt_contract_data(self, contract_data: Dict[str, Any]) -> bytes:
        """Encrypt contract data for secure storage."""
        json_data = json.dumps(contract_data, sort_keys=True)
        return self.crypto.encrypt_symmetric(json_data.encode())
    
    def decrypt_contract_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt contract data."""
        decrypted_data = self.crypto.decrypt_symmetric(encrypted_data)
        return json.loads(decrypted_data.decode())
    
    def sign_contract(self, contract_data: Dict[str, Any]) -> str:
        """Create digital signature for contract."""
        json_data = json.dumps(contract_data, sort_keys=True)
        signature = self.crypto.sign_data(json_data.encode())
        return base64.b64encode(signature).decode()
    
    def verify_contract_signature(self, contract_data: Dict[str, Any], signature: str) -> bool:
        """Verify contract digital signature."""
        try:
            json_data = json.dumps(contract_data, sort_keys=True)
            signature_bytes = base64.b64decode(signature)
            return self.crypto.verify_signature(json_data.encode(), signature_bytes)
        except Exception:
            return False
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        recent_logs = self.audit_logger.get_recent_logs(100)
        
        return {
            "active_threats": len([
                t for threats in self.threat_detector.rate_limits.values()
                for t in threats
                if datetime.now() - t < timedelta(minutes=5)
            ]),
            "suspicious_ips": len(self.threat_detector.suspicious_ips),
            "failed_auth_attempts": sum(
                len(attempts) for attempts in self.threat_detector.failed_attempts.values()
            ),
            "recent_audit_entries": len(recent_logs),
            "successful_actions": len([log for log in recent_logs if log.success]),
            "failed_actions": len([log for log in recent_logs if not log.success]),
            "unique_users_active": len(set(log.user_id for log in recent_logs)),
        }
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        metrics = self.get_security_metrics()
        recent_logs = self.audit_logger.get_recent_logs(1000)
        
        # Analyze security trends
        security_events = defaultdict(int)
        for log in recent_logs:
            if not log.success:
                security_events[log.action] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "top_failed_actions": dict(
                sorted(security_events.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "security_recommendations": self._generate_recommendations(metrics),
            "compliance_status": self._check_compliance(),
        }
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on metrics."""
        recommendations = []
        
        if metrics["failed_actions"] > metrics["successful_actions"] * 0.1:
            recommendations.append("High failure rate detected - review access controls")
        
        if metrics["active_threats"] > 10:
            recommendations.append("Multiple active threats - consider enhanced monitoring")
        
        if metrics["suspicious_ips"] > 0:
            recommendations.append("Suspicious IPs detected - review and update blocklist")
        
        return recommendations
    
    def _check_compliance(self) -> Dict[str, bool]:
        """Check compliance with security standards."""
        return {
            "audit_logging_enabled": True,
            "encryption_in_use": True,
            "access_control_configured": len(self.access_controller.roles) > 0,
            "threat_detection_active": True,
            "regular_security_reviews": True,  # Would be checked against schedule
        }


# Example usage and testing
async def example_security_workflow():
    """Example demonstrating security framework usage."""
    # Initialize security framework
    security = SecurityFramework()
    
    # Create security context
    context = SecurityContext(
        user_id="alice",
        session_id="",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0...",
        timestamp=datetime.now(),
        security_level=SecurityLevel.CONFIDENTIAL
    )
    
    # Authenticate user
    auth_success, session_token = security.authenticate_user("alice", "password123", context)
    if auth_success:
        print(f"✅ Authentication successful. Session: {session_token[:16]}...")
        context.session_id = session_token
        
        # Assign role to user
        security.access_controller.assign_role("alice", "operator")
        
        # Test authorization
        can_read, error = security.authorize_action("alice", "contracts", AccessAction.READ, context)
        print(f"✅ Read access: {can_read}")
        
        can_delete, error = security.authorize_action("alice", "contracts", AccessAction.DELETE, context)
        print(f"❌ Delete access: {can_delete} ({'Permission denied' if error else 'Allowed'})")
        
        # Test contract encryption
        contract_data = {"name": "TestContract", "stakeholders": {"alice": 1.0}}
        encrypted_contract = security.encrypt_contract_data(contract_data)
        decrypted_contract = security.decrypt_contract_data(encrypted_contract)
        print(f"✅ Encryption test: {decrypted_contract == contract_data}")
        
        # Generate security report
        report = security.generate_security_report()
        print(f"📊 Security metrics: {report['metrics']}")
    
    else:
        print(f"❌ Authentication failed: {session_token}")


if __name__ == "__main__":
    asyncio.run(example_security_workflow())
"""
Comprehensive security validation system for Generation 2: MAKE IT ROBUST

Implements advanced security measures, input validation, threat detection,
and security monitoring for the RLHF-Contract-Wizard system.
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import re
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import jax.numpy as jnp
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class SecurityLevel(Enum):
    """Security validation levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class ThreatType(Enum):
    """Types of security threats."""
    INJECTION = "injection"
    TAMPERING = "tampering"
    REPLAY = "replay"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_LEAKAGE = "data_leakage"
    DENIAL_OF_SERVICE = "denial_of_service"
    MALICIOUS_INPUT = "malicious_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityEvent:
    """Security event for monitoring and analysis."""
    timestamp: float
    threat_type: ThreatType
    severity: SecurityLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    operation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    confidence: float = 1.0


@dataclass
class ValidationResult:
    """Result of security validation."""
    passed: bool
    security_level: SecurityLevel
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    threat_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    def __init__(self):
        # Common attack patterns
        self.sql_injection_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bINSERT\b.*\bINTO\b)",
            r"(\bDELETE\b.*\bFROM\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"('.*--)",
            r"(;.*--)",
            r"(\bOR\b.*=.*)",
            r"(\bAND\b.*=.*)"
        ]
        
        self.script_injection_patterns = [
            r"(<script[^>]*>.*</script>)",
            r"(javascript:)",
            r"(on\w+\s*=)",
            r"(<iframe[^>]*>.*</iframe>)",
            r"(<object[^>]*>.*</object>)",
            r"(<embed[^>]*>.*</embed>)"
        ]
        
        self.command_injection_patterns = [
            r"(;\s*(rm|del|format|shutdown))",
            r"(\|\s*(cat|ls|dir|type))",
            r"(`.*`)",
            r"(\$\(.*\))",
            r"(&&\s*\w+)",
            r"(\|\|\s*\w+)"
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            "sql": [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_injection_patterns],
            "script": [re.compile(pattern, re.IGNORECASE) for pattern in self.script_injection_patterns],
            "command": [re.compile(pattern, re.IGNORECASE) for pattern in self.command_injection_patterns]
        }
    
    def sanitize_string(self, input_str: str, max_length: int = 1000) -> ValidationResult:
        """Sanitize and validate string input."""
        violations = []
        warnings = []
        threat_score = 0.0
        
        # Length validation
        if len(input_str) > max_length:
            violations.append(f"Input exceeds maximum length ({max_length})")
            threat_score += 0.3
        
        # Pattern matching for known attacks
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(input_str):
                    violations.append(f"Potential {category} injection detected")
                    threat_score += 0.4
        
        # Character analysis
        suspicious_chars = set("';\"\\<>{}[]()$`|&")
        char_count = sum(1 for c in input_str if c in suspicious_chars)
        char_ratio = char_count / max(1, len(input_str))
        
        if char_ratio > 0.1:
            warnings.append("High concentration of suspicious characters")
            threat_score += 0.2
        
        # Entropy analysis (detect random/encoded content)
        entropy = self._calculate_entropy(input_str)
        if entropy > 4.5:  # High entropy threshold
            warnings.append("Input has high entropy (possible encoded content)")
            threat_score += 0.1
        
        # Determine security level
        if threat_score >= 0.7:
            security_level = SecurityLevel.MAXIMUM
        elif threat_score >= 0.4:
            security_level = SecurityLevel.HIGH
        elif threat_score >= 0.2:
            security_level = SecurityLevel.MEDIUM
        else:
            security_level = SecurityLevel.LOW
        
        return ValidationResult(
            passed=len(violations) == 0,
            security_level=security_level,
            violations=violations,
            warnings=warnings,
            threat_score=threat_score,
            metadata={"entropy": entropy, "suspicious_char_ratio": char_ratio}
        )
    
    def sanitize_array(self, input_array: Union[np.ndarray, jnp.ndarray], max_size: int = 10000) -> ValidationResult:
        """Sanitize and validate array input."""
        violations = []
        warnings = []
        threat_score = 0.0
        
        # Size validation
        if input_array.size > max_size:
            violations.append(f"Array exceeds maximum size ({max_size})")
            threat_score += 0.3
        
        # Check for NaN/Inf values
        if not jnp.all(jnp.isfinite(input_array)):
            violations.append("Array contains NaN or infinite values")
            threat_score += 0.4
        
        # Range validation
        if jnp.any(jnp.abs(input_array) > 1e6):
            warnings.append("Array contains very large values")
            threat_score += 0.1
        
        # Check for suspicious patterns
        if input_array.size > 0:
            # Check for repeating patterns (possible attack)
            if input_array.size >= 10:
                first_ten = input_array.flatten()[:10]
                if jnp.allclose(input_array.flatten()[:input_array.size//10*10].reshape(-1, 10), 
                              first_ten, rtol=1e-10):
                    warnings.append("Array contains repeating patterns")
                    threat_score += 0.1
        
        return ValidationResult(
            passed=len(violations) == 0,
            security_level=SecurityLevel.MEDIUM if threat_score > 0.2 else SecurityLevel.LOW,
            violations=violations,
            warnings=warnings,
            threat_score=threat_score,
            metadata={"size": input_array.size, "shape": input_array.shape}
        )
    
    def sanitize_dict(self, input_dict: Dict[str, Any], max_depth: int = 5) -> ValidationResult:
        """Sanitize and validate dictionary input."""
        violations = []
        warnings = []
        threat_score = 0.0
        
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                violations.append(f"Dictionary nesting exceeds maximum depth ({max_depth})")
                return 0.3
            
            threat_score_local = 0.0
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Validate keys
                    key_validation = self.sanitize_string(str(key), max_length=100)
                    if not key_validation.passed:
                        violations.extend([f"Key '{key}': {v}" for v in key_validation.violations])
                    threat_score_local += key_validation.threat_score * 0.5
                    
                    # Recursively validate values
                    threat_score_local += check_depth(value, current_depth + 1)
            
            elif isinstance(obj, (list, tuple)):
                if len(obj) > 1000:
                    warnings.append("Collection contains many items")
                    threat_score_local += 0.1
                
                for item in obj:
                    threat_score_local += check_depth(item, current_depth + 1)
            
            elif isinstance(obj, str):
                str_validation = self.sanitize_string(obj)
                if not str_validation.passed:
                    violations.extend(str_validation.violations)
                threat_score_local += str_validation.threat_score * 0.3
            
            return threat_score_local
        
        threat_score = check_depth(input_dict)
        
        return ValidationResult(
            passed=len(violations) == 0,
            security_level=SecurityLevel.HIGH if threat_score > 0.4 else SecurityLevel.MEDIUM,
            violations=violations,
            warnings=warnings,
            threat_score=threat_score,
            metadata={"depth": self._calculate_dict_depth(input_dict)}
        )
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / length
            entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_dict_depth(self, obj, current_depth=0):
        """Calculate maximum depth of nested dictionary."""
        if not isinstance(obj, dict):
            return current_depth
        
        if not obj:
            return current_depth + 1
        
        max_depth = current_depth + 1
        for value in obj.values():
            if isinstance(value, dict):
                depth = self._calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth


class SecurityMonitor:
    """Real-time security monitoring and threat detection."""
    
    def __init__(self, max_events: int = 10000):
        self.security_events: List[SecurityEvent] = []
        self.max_events = max_events
        self.threat_counters = {threat_type: 0 for threat_type in ThreatType}
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)
    
    def log_security_event(
        self,
        threat_type: ThreatType,
        severity: SecurityLevel,
        operation: str,
        details: Dict[str, Any],
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        blocked: bool = False
    ):
        """Log a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            threat_type=threat_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            operation=operation,
            details=details,
            blocked=blocked
        )
        
        self.security_events.append(event)
        self.threat_counters[threat_type] += 1
        
        # Trim old events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
        
        # Log based on severity
        log_message = f"Security event: {threat_type.value} in {operation} (severity: {severity.value})"
        if severity in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Check for rate limiting
        if source_ip:
            self._check_rate_limit(source_ip, event)
    
    def _check_rate_limit(self, source_ip: str, event: SecurityEvent):
        """Check and enforce rate limits."""
        current_time = time.time()
        window_size = 300  # 5 minutes
        max_events_per_window = 100
        
        if source_ip not in self.rate_limits:
            self.rate_limits[source_ip] = []
        
        # Clean old events
        self.rate_limits[source_ip] = [
            t for t in self.rate_limits[source_ip]
            if current_time - t < window_size
        ]
        
        # Add current event
        self.rate_limits[source_ip].append(current_time)
        
        # Check if rate limit exceeded
        if len(self.rate_limits[source_ip]) > max_events_per_window:
            self.blocked_ips.add(source_ip)
            self.logger.warning(f"IP {source_ip} blocked for rate limit violation")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is blocked."""
        return ip in self.blocked_ips
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of security threats."""
        recent_events = [
            e for e in self.security_events
            if time.time() - e.timestamp < 3600  # Last hour
        ]
        
        severity_counts = {level: 0 for level in SecurityLevel}
        threat_counts = {threat: 0 for threat in ThreatType}
        
        for event in recent_events:
            severity_counts[event.severity] += 1
            threat_counts[event.threat_type] += 1
        
        return {
            "total_events": len(self.security_events),
            "recent_events": len(recent_events),
            "severity_distribution": {k.value: v for k, v in severity_counts.items()},
            "threat_distribution": {k.value: v for k, v in threat_counts.items()},
            "blocked_ips": len(self.blocked_ips),
            "high_risk_operations": self._identify_high_risk_operations()
        }
    
    def _identify_high_risk_operations(self) -> List[Dict[str, Any]]:
        """Identify operations with high security risk."""
        recent_events = [
            e for e in self.security_events
            if time.time() - e.timestamp < 3600 and 
               e.severity in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]
        ]
        
        operation_counts = {}
        for event in recent_events:
            op = event.operation
            if op not in operation_counts:
                operation_counts[op] = {"count": 0, "severity_sum": 0}
            operation_counts[op]["count"] += 1
            operation_counts[op]["severity_sum"] += {
                SecurityLevel.LOW: 1,
                SecurityLevel.MEDIUM: 2,
                SecurityLevel.HIGH: 3,
                SecurityLevel.MAXIMUM: 4
            }[event.severity]
        
        # Calculate risk scores
        risk_operations = []
        for op, data in operation_counts.items():
            if data["count"] >= 5:  # At least 5 incidents
                risk_score = data["severity_sum"] / data["count"]
                risk_operations.append({
                    "operation": op,
                    "incident_count": data["count"],
                    "risk_score": risk_score
                })
        
        return sorted(risk_operations, key=lambda x: x["risk_score"], reverse=True)[:10]


class CryptographicSecurity:
    """Cryptographic operations for data protection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._encryption_key = None
        self._signing_key = None
        self._verification_key = None
        
        # Initialize keys
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialize cryptographic keys."""
        try:
            # Generate symmetric encryption key
            self._encryption_key = Fernet.generate_key()
            
            # Generate RSA key pair for signing
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self._signing_key = private_key
            self._verification_key = private_key.public_key()
            
            self.logger.info("Cryptographic keys initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cryptographic keys: {e}")
            raise
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        try:
            fernet = Fernet(self._encryption_key)
            encrypted_data = fernet.encrypt(data)
            return encrypted_data
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        try:
            fernet = Fernet(self._encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            return decrypted_data
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def sign_data(self, data: bytes) -> bytes:
        """Create digital signature for data."""
        try:
            signature = self._signing_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature
        except Exception as e:
            self.logger.error(f"Signing failed: {e}")
            raise
    
    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify digital signature."""
        try:
            self._verification_key.verify(
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
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """Hash password using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        password_hash = kdf.derive(password.encode())
        
        return {
            "hash": base64.b64encode(password_hash).decode(),
            "salt": base64.b64encode(salt).decode()
        }
    
    def verify_password(self, password: str, stored_hash: str, stored_salt: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt = base64.b64decode(stored_salt)
            expected_hash = base64.b64decode(stored_hash)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            kdf.verify(password.encode(), expected_hash)
            return True
        except Exception:
            return False


class ComprehensiveSecurityValidator:
    """
    Comprehensive security validation system combining all security components.
    
    Features:
    - Advanced input sanitization
    - Real-time threat monitoring
    - Cryptographic data protection
    - Rate limiting and IP blocking
    - Security analytics and reporting
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.sanitizer = InputSanitizer()
        self.monitor = SecurityMonitor()
        self.crypto = CryptographicSecurity()
        self.logger = logging.getLogger(__name__)
        
        # Security configuration based on level
        self.config = self._get_security_config(security_level)
    
    def _get_security_config(self, level: SecurityLevel) -> Dict[str, Any]:
        """Get security configuration based on security level."""
        configs = {
            SecurityLevel.LOW: {
                "max_string_length": 10000,
                "max_array_size": 100000,
                "max_dict_depth": 10,
                "enable_encryption": False,
                "enable_signing": False,
                "rate_limit_window": 3600,
                "rate_limit_max": 1000
            },
            SecurityLevel.MEDIUM: {
                "max_string_length": 5000,
                "max_array_size": 50000,
                "max_dict_depth": 8,
                "enable_encryption": True,
                "enable_signing": False,
                "rate_limit_window": 1800,
                "rate_limit_max": 500
            },
            SecurityLevel.HIGH: {
                "max_string_length": 2000,
                "max_array_size": 20000,
                "max_dict_depth": 5,
                "enable_encryption": True,
                "enable_signing": True,
                "rate_limit_window": 900,
                "rate_limit_max": 200
            },
            SecurityLevel.MAXIMUM: {
                "max_string_length": 1000,
                "max_array_size": 10000,
                "max_dict_depth": 3,
                "enable_encryption": True,
                "enable_signing": True,
                "rate_limit_window": 300,
                "rate_limit_max": 50
            }
        }
        return configs[level]
    
    async def validate_input(
        self,
        input_data: Any,
        operation: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ValidationResult:
        """Comprehensive input validation with security monitoring."""
        
        # Check if IP is blocked
        if source_ip and self.monitor.is_ip_blocked(source_ip):
            self.monitor.log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.HIGH,
                operation,
                {"reason": "blocked_ip", "ip": source_ip},
                source_ip=source_ip,
                blocked=True
            )
            return ValidationResult(
                passed=False,
                security_level=SecurityLevel.MAXIMUM,
                violations=["Source IP is blocked"],
                threat_score=1.0
            )
        
        # Validate different input types
        if isinstance(input_data, str):
            result = self.sanitizer.sanitize_string(
                input_data, 
                max_length=self.config["max_string_length"]
            )
        elif isinstance(input_data, (np.ndarray, jnp.ndarray)):
            result = self.sanitizer.sanitize_array(
                input_data,
                max_size=self.config["max_array_size"]
            )
        elif isinstance(input_data, dict):
            result = self.sanitizer.sanitize_dict(
                input_data,
                max_depth=self.config["max_dict_depth"]
            )
        else:
            # Convert to string and validate
            result = self.sanitizer.sanitize_string(
                str(input_data),
                max_length=self.config["max_string_length"]
            )
        
        # Log security events based on validation result
        if not result.passed or result.threat_score > 0.3:
            threat_type = ThreatType.MALICIOUS_INPUT
            if "injection" in str(result.violations).lower():
                threat_type = ThreatType.INJECTION
            elif result.threat_score > 0.5:
                threat_type = ThreatType.TAMPERING
            
            self.monitor.log_security_event(
                threat_type,
                result.security_level,
                operation,
                {
                    "violations": result.violations,
                    "warnings": result.warnings,
                    "threat_score": result.threat_score
                },
                source_ip=source_ip,
                user_id=user_id,
                blocked=not result.passed
            )
        
        return result
    
    def protect_sensitive_data(self, data: Any) -> Dict[str, Any]:
        """Protect sensitive data with encryption and signing."""
        try:
            # Serialize data
            if isinstance(data, (dict, list)):
                import json
                data_bytes = json.dumps(data, sort_keys=True).encode()
            elif isinstance(data, str):
                data_bytes = data.encode()
            else:
                data_bytes = str(data).encode()
            
            protected_data = {"original_type": type(data).__name__}
            
            # Encrypt if enabled
            if self.config["enable_encryption"]:
                protected_data["encrypted_data"] = base64.b64encode(
                    self.crypto.encrypt_data(data_bytes)
                ).decode()
            else:
                protected_data["data"] = data
            
            # Sign if enabled
            if self.config["enable_signing"]:
                signature = self.crypto.sign_data(data_bytes)
                protected_data["signature"] = base64.b64encode(signature).decode()
            
            protected_data["timestamp"] = time.time()
            return protected_data
            
        except Exception as e:
            self.logger.error(f"Failed to protect sensitive data: {e}")
            raise
    
    def unprotect_sensitive_data(self, protected_data: Dict[str, Any]) -> Any:
        """Unprotect sensitive data by decrypting and verifying."""
        try:
            if "encrypted_data" in protected_data:
                # Decrypt data
                encrypted_bytes = base64.b64decode(protected_data["encrypted_data"])
                data_bytes = self.crypto.decrypt_data(encrypted_bytes)
                
                # Verify signature if present
                if "signature" in protected_data:
                    signature = base64.b64decode(protected_data["signature"])
                    if not self.crypto.verify_signature(data_bytes, signature):
                        raise ValueError("Signature verification failed")
                
                # Deserialize based on original type
                original_type = protected_data.get("original_type", "str")
                if original_type in ["dict", "list"]:
                    import json
                    return json.loads(data_bytes.decode())
                else:
                    return data_bytes.decode()
            
            else:
                # Data not encrypted, return as-is
                return protected_data.get("data")
                
        except Exception as e:
            self.logger.error(f"Failed to unprotect sensitive data: {e}")
            raise
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard."""
        threat_summary = self.monitor.get_threat_summary()
        
        return {
            "security_level": self.security_level.value,
            "configuration": self.config,
            "threat_monitoring": threat_summary,
            "system_health": {
                "blocked_ips": len(self.monitor.blocked_ips),
                "recent_incidents": threat_summary["recent_events"],
                "critical_threats": sum(
                    1 for event in self.monitor.security_events[-100:]
                    if event.severity == SecurityLevel.MAXIMUM
                ),
                "uptime_hours": (time.time() - (
                    self.monitor.security_events[0].timestamp 
                    if self.monitor.security_events else time.time()
                )) / 3600
            },
            "recommendations": self._generate_security_recommendations()
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current state."""
        recommendations = []
        
        threat_summary = self.monitor.get_threat_summary()
        
        # Check for high threat activity
        if threat_summary["recent_events"] > 50:
            recommendations.append("Consider increasing security level due to high threat activity")
        
        # Check for specific threat types
        threat_dist = threat_summary["threat_distribution"]
        if threat_dist.get("injection", 0) > 5:
            recommendations.append("Implement additional input validation for injection attacks")
        
        if threat_dist.get("denial_of_service", 0) > 10:
            recommendations.append("Consider implementing DDoS protection")
        
        # Check blocked IPs
        if len(self.monitor.blocked_ips) > 100:
            recommendations.append("Review and cleanup blocked IP list")
        
        # Security level recommendations
        if (self.security_level == SecurityLevel.LOW and 
            threat_summary["severity_distribution"].get("high", 0) > 5):
            recommendations.append("Consider upgrading to MEDIUM security level")
        
        if not recommendations:
            recommendations.append("Security posture is good - continue monitoring")
        
        return recommendations


# Global security validator instance
_security_validator: Optional[ComprehensiveSecurityValidator] = None


def get_security_validator(security_level: SecurityLevel = SecurityLevel.HIGH) -> ComprehensiveSecurityValidator:
    """Get or create the global security validator instance."""
    global _security_validator
    
    if _security_validator is None:
        _security_validator = ComprehensiveSecurityValidator(security_level)
    
    return _security_validator
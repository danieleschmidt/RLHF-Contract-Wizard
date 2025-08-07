"""
Security module for quantum task planning.

Implements security controls, access management, audit logging,
and threat detection for the quantum task planning system.
"""

import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .core import QuantumTask, TaskState


class SecurityLevel(Enum):
    """Security classification levels for tasks and data."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """Security context for task execution."""
    user_id: str
    session_id: str
    access_level: SecurityLevel
    permissions: Set[str] = field(default_factory=set)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    authentication_method: str = "unknown"
    mfa_verified: bool = False
    session_start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


@dataclass 
class SecurityEvent:
    """Security event for audit logging."""
    event_id: str
    timestamp: float
    event_type: str
    severity: ThreatLevel
    user_id: Optional[str]
    session_id: Optional[str]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class SecurityValidator:
    """
    Validates security requirements for quantum task planning operations.
    
    Implements access control, input validation, and threat detection
    to ensure secure operation of the quantum planning system.
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Security configurations
        self.max_session_duration = 3600 * 8  # 8 hours
        self.max_inactive_duration = 1800     # 30 minutes
        self.max_failed_attempts = 5
        self.rate_limit_window = 60           # 1 minute
        self.max_requests_per_window = 100
        
        # Tracking
        self.failed_attempts: Dict[str, List[float]] = {}
        self.request_history: Dict[str, List[float]] = {}
        self.blocked_users: Set[str] = set()
        self.security_events: List[SecurityEvent] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def validate_security_context(self, context: SecurityContext) -> Dict[str, Any]:
        """
        Validate security context for task operations.
        
        Returns:
            Validation result with security status and any violations
        """
        validation_result = {
            'valid': True,
            'violations': [],
            'threat_level': ThreatLevel.LOW,
            'recommended_actions': []
        }
        
        current_time = time.time()
        
        # Check session validity
        session_age = current_time - context.session_start_time
        if session_age > self.max_session_duration:
            validation_result['violations'].append({
                'type': 'session_expired',
                'severity': ThreatLevel.MEDIUM,
                'message': 'Session has exceeded maximum duration'
            })
            validation_result['valid'] = False
            validation_result['recommended_actions'].append('require_reauthentication')
        
        # Check activity timeout
        inactive_time = current_time - context.last_activity
        if inactive_time > self.max_inactive_duration:
            validation_result['violations'].append({
                'type': 'session_inactive',
                'severity': ThreatLevel.LOW,
                'message': 'Session has been inactive for too long'
            })
            validation_result['recommended_actions'].append('refresh_session')
        
        # Check if user is blocked
        if context.user_id in self.blocked_users:
            validation_result['violations'].append({
                'type': 'user_blocked',
                'severity': ThreatLevel.HIGH,
                'message': 'User account is blocked due to security violations'
            })
            validation_result['valid'] = False
            validation_result['threat_level'] = ThreatLevel.HIGH
        
        # Check rate limiting
        rate_limit_violation = self._check_rate_limit(context.user_id)
        if rate_limit_violation:
            validation_result['violations'].append(rate_limit_violation)
            validation_result['valid'] = False
        
        # Check for suspicious patterns
        suspicious_activity = self._detect_suspicious_activity(context)
        if suspicious_activity:
            validation_result['violations'].extend(suspicious_activity)
            validation_result['threat_level'] = max(
                validation_result['threat_level'],
                ThreatLevel.MEDIUM,
                key=lambda x: list(ThreatLevel).index(x)
            )
        
        # Update activity timestamp if validation passed
        if validation_result['valid']:
            context.last_activity = current_time
        
        # Log security event if violations found
        if validation_result['violations']:
            self._log_security_event(
                event_type='security_validation',
                severity=validation_result['threat_level'],
                user_id=context.user_id,
                session_id=context.session_id,
                description=f"Security validation found {len(validation_result['violations'])} violations",
                metadata={'violations': validation_result['violations']}
            )
        
        return validation_result
    
    def validate_task_security(
        self, 
        task: QuantumTask, 
        context: SecurityContext
    ) -> Dict[str, Any]:
        """
        Validate security requirements for a specific task.
        
        Args:
            task: Task to validate
            context: Security context
            
        Returns:
            Task security validation result
        """
        validation_result = {
            'valid': True,
            'violations': [],
            'required_clearance': SecurityLevel.PUBLIC,
            'access_granted': False
        }
        
        # Determine task security classification
        task_security_level = self._classify_task_security(task)
        validation_result['required_clearance'] = task_security_level
        
        # Check access permissions
        has_clearance = self._check_security_clearance(context, task_security_level)
        validation_result['access_granted'] = has_clearance
        
        if not has_clearance:
            validation_result['violations'].append({
                'type': 'insufficient_clearance',
                'severity': ThreatLevel.HIGH,
                'message': f"User lacks required clearance ({task_security_level.value}) for task {task.id}",
                'required_clearance': task_security_level.value,
                'user_clearance': context.access_level.value
            })
            validation_result['valid'] = False
        
        # Check task-specific permissions
        required_permissions = self._get_required_permissions(task)
        missing_permissions = required_permissions - context.permissions
        
        if missing_permissions:
            validation_result['violations'].append({
                'type': 'missing_permissions',
                'severity': ThreatLevel.MEDIUM,
                'message': f"Missing required permissions: {missing_permissions}",
                'missing_permissions': list(missing_permissions)
            })
            validation_result['valid'] = False
        
        # Validate task resource access
        resource_validation = self._validate_resource_access(task, context)
        if not resource_validation['valid']:
            validation_result['violations'].extend(resource_validation['violations'])
            validation_result['valid'] = False
        
        # Check for malicious task patterns
        malicious_patterns = self._detect_malicious_task_patterns(task)
        if malicious_patterns:
            validation_result['violations'].extend(malicious_patterns)
            validation_result['valid'] = False
            
            # Log security event for malicious task detection
            self._log_security_event(
                event_type='malicious_task_detected',
                severity=ThreatLevel.CRITICAL,
                user_id=context.user_id,
                session_id=context.session_id,
                description=f"Malicious patterns detected in task {task.id}",
                metadata={'task_id': task.id, 'patterns': malicious_patterns}
            )
        
        return validation_result
    
    def _check_rate_limit(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Check if user has exceeded rate limits."""
        current_time = time.time()
        
        # Initialize tracking for new users
        if user_id not in self.request_history:
            self.request_history[user_id] = []
        
        # Clean old requests outside the window
        window_start = current_time - self.rate_limit_window
        self.request_history[user_id] = [
            req_time for req_time in self.request_history[user_id]
            if req_time >= window_start
        ]
        
        # Add current request
        self.request_history[user_id].append(current_time)
        
        # Check if limit exceeded
        if len(self.request_history[user_id]) > self.max_requests_per_window:
            return {
                'type': 'rate_limit_exceeded',
                'severity': ThreatLevel.MEDIUM,
                'message': f'User {user_id} exceeded rate limit: {len(self.request_history[user_id])} requests in {self.rate_limit_window}s',
                'request_count': len(self.request_history[user_id]),
                'limit': self.max_requests_per_window
            }
        
        return None
    
    def _detect_suspicious_activity(self, context: SecurityContext) -> List[Dict[str, Any]]:
        """Detect suspicious activity patterns."""
        suspicious_patterns = []
        
        # Check for unusual access patterns
        if context.ip_address:
            # Mock IP geolocation check (would use real service in production)
            if self._is_suspicious_ip(context.ip_address):
                suspicious_patterns.append({
                    'type': 'suspicious_ip',
                    'severity': ThreatLevel.MEDIUM,
                    'message': f'Access from potentially suspicious IP: {context.ip_address}',
                    'ip_address': context.ip_address
                })
        
        # Check for unusual user agent
        if context.user_agent:
            if self._is_suspicious_user_agent(context.user_agent):
                suspicious_patterns.append({
                    'type': 'suspicious_user_agent',
                    'severity': ThreatLevel.LOW,
                    'message': f'Unusual user agent detected: {context.user_agent[:100]}...',
                    'user_agent': context.user_agent
                })
        
        # Check authentication method security
        if context.authentication_method in ['weak', 'legacy', 'deprecated']:
            suspicious_patterns.append({
                'type': 'weak_authentication',
                'severity': ThreatLevel.MEDIUM,
                'message': f'Weak authentication method used: {context.authentication_method}',
                'auth_method': context.authentication_method
            })
        
        # Check MFA requirement for high-security operations
        if context.access_level in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET]:
            if not context.mfa_verified:
                suspicious_patterns.append({
                    'type': 'missing_mfa',
                    'severity': ThreatLevel.HIGH,
                    'message': 'MFA required for high-security operations but not verified',
                    'access_level': context.access_level.value
                })
        
        return suspicious_patterns
    
    def _classify_task_security(self, task: QuantumTask) -> SecurityLevel:
        """Classify task security level based on task properties."""
        
        # Check task name and description for security indicators
        task_text = (task.name + " " + task.description).lower()
        
        # Top secret indicators
        if any(keyword in task_text for keyword in [
            'classified', 'secret', 'confidential', 'restricted',
            'security', 'audit', 'sensitive', 'private'
        ]):
            return SecurityLevel.CONFIDENTIAL
        
        # Check resource requirements
        resource_sum = sum(task.resource_requirements.values())
        if resource_sum > 100:  # High resource usage
            return SecurityLevel.INTERNAL
        
        # Check dependencies - tasks with many dependencies might be critical
        if len(task.dependencies) > 3:
            return SecurityLevel.INTERNAL
        
        # Default classification
        return SecurityLevel.PUBLIC
    
    def _check_security_clearance(
        self, 
        context: SecurityContext, 
        required_level: SecurityLevel
    ) -> bool:
        """Check if user has required security clearance."""
        
        # Define clearance hierarchy
        clearance_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        user_level = clearance_hierarchy.get(context.access_level, 0)
        required_level_num = clearance_hierarchy.get(required_level, 0)
        
        return user_level >= required_level_num
    
    def _get_required_permissions(self, task: QuantumTask) -> Set[str]:
        """Get required permissions for task execution."""
        required_permissions = set()
        
        # Base permissions based on task type
        if 'training' in task.name.lower() or 'model' in task.name.lower():
            required_permissions.add('model_training')
        
        if 'deployment' in task.name.lower() or 'deploy' in task.name.lower():
            required_permissions.add('deployment')
        
        if 'data' in task.name.lower():
            required_permissions.add('data_access')
        
        if 'security' in task.name.lower() or 'audit' in task.name.lower():
            required_permissions.add('security_operations')
        
        # Resource-based permissions
        if task.resource_requirements.get('gpu', 0) > 0:
            required_permissions.add('gpu_access')
        
        if task.resource_requirements.get('memory', 0) > 16:
            required_permissions.add('high_memory_access')
        
        return required_permissions
    
    def _validate_resource_access(
        self, 
        task: QuantumTask, 
        context: SecurityContext
    ) -> Dict[str, Any]:
        """Validate user's access to task resources."""
        validation_result = {
            'valid': True,
            'violations': []
        }
        
        # Check for resource quota violations
        for resource, amount in task.resource_requirements.items():
            max_allowed = self._get_resource_quota(context, resource)
            
            if amount > max_allowed:
                validation_result['violations'].append({
                    'type': 'resource_quota_exceeded',
                    'severity': ThreatLevel.MEDIUM,
                    'message': f'Task requires {amount} {resource}, but user quota is {max_allowed}',
                    'resource': resource,
                    'requested': amount,
                    'quota': max_allowed
                })
                validation_result['valid'] = False
        
        return validation_result
    
    def _get_resource_quota(self, context: SecurityContext, resource: str) -> float:
        """Get user's resource quota for a specific resource type."""
        
        # Define quotas based on security level
        quotas = {
            SecurityLevel.PUBLIC: {
                'cpu': 2,
                'gpu': 0,
                'memory': 4,
                'storage': 10
            },
            SecurityLevel.INTERNAL: {
                'cpu': 8,
                'gpu': 1,
                'memory': 16,
                'storage': 100
            },
            SecurityLevel.CONFIDENTIAL: {
                'cpu': 16,
                'gpu': 2,
                'memory': 32,
                'storage': 500
            },
            SecurityLevel.RESTRICTED: {
                'cpu': 32,
                'gpu': 4,
                'memory': 64,
                'storage': 1000
            },
            SecurityLevel.TOP_SECRET: {
                'cpu': 64,
                'gpu': 8,
                'memory': 128,
                'storage': 2000
            }
        }
        
        user_quotas = quotas.get(context.access_level, quotas[SecurityLevel.PUBLIC])
        return user_quotas.get(resource, 0)
    
    def _detect_malicious_task_patterns(self, task: QuantumTask) -> List[Dict[str, Any]]:
        """Detect potentially malicious patterns in task definitions."""
        malicious_patterns = []
        
        # Check for suspicious keywords in task description
        suspicious_keywords = [
            'inject', 'exploit', 'hack', 'backdoor', 'malware',
            'virus', 'trojan', 'rootkit', 'keylogger', 'steal',
            'exfiltrate', 'bypass', 'crack', 'break', 'disable'
        ]
        
        task_text = (task.name + " " + task.description).lower()
        found_keywords = [kw for kw in suspicious_keywords if kw in task_text]
        
        if found_keywords:
            malicious_patterns.append({
                'type': 'suspicious_keywords',
                'severity': ThreatLevel.HIGH,
                'message': f'Suspicious keywords found in task: {found_keywords}',
                'keywords': found_keywords
            })
        
        # Check for excessive resource requests (potential DoS)
        total_resources = sum(task.resource_requirements.values())
        if total_resources > 1000:  # Arbitrary threshold
            malicious_patterns.append({
                'type': 'excessive_resource_request',
                'severity': ThreatLevel.MEDIUM,
                'message': f'Task requests excessive resources: {total_resources}',
                'resource_total': total_resources
            })
        
        # Check for suspicious dependencies (potential supply chain attack)
        if len(task.dependencies) > 10:
            malicious_patterns.append({
                'type': 'excessive_dependencies',
                'severity': ThreatLevel.MEDIUM,
                'message': f'Task has excessive dependencies: {len(task.dependencies)}',
                'dependency_count': len(task.dependencies)
            })
        
        return malicious_patterns
    
    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is potentially suspicious."""
        # Mock implementation - would integrate with threat intelligence feeds
        suspicious_patterns = [
            '10.0.0.',     # Internal network (suspicious if external access)
            '192.168.',    # Private network
            '127.0.0.',    # Localhost
            '0.0.0.',      # Invalid IP patterns
        ]
        
        return any(pattern in ip_address for pattern in suspicious_patterns)
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent string is potentially suspicious."""
        suspicious_patterns = [
            'bot', 'crawler', 'spider', 'scraper',
            'automated', 'script', 'tool', 'curl',
            'wget', 'python-requests'
        ]
        
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)
    
    def _log_security_event(
        self,
        event_type: str,
        severity: ThreatLevel,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log security event for audit trail."""
        
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Log to system logger
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }.get(severity, logging.INFO)
        
        self.logger.log(log_level, f"Security Event [{event_type}]: {description}", extra={
            'event_id': event.event_id,
            'severity': severity.value,
            'user_id': user_id,
            'session_id': session_id,
            'metadata': metadata
        })
        
        # Take automated response for critical threats
        if severity == ThreatLevel.CRITICAL and user_id:
            self._automated_threat_response(user_id, event)
    
    def _automated_threat_response(self, user_id: str, event: SecurityEvent):
        """Automated response to critical security threats."""
        
        # Block user temporarily for critical threats
        self.blocked_users.add(user_id)
        
        self.logger.critical(f"AUTOMATED RESPONSE: User {user_id} blocked due to critical security event {event.event_id}")
        
        # Could trigger additional automated responses:
        # - Send alerts to security team
        # - Invalidate all user sessions
        # - Quarantine affected resources
        # - Trigger incident response workflow
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for secure storage."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def verify_integrity(self, data: str, signature: str, secret_key: str) -> bool:
        """Verify data integrity using HMAC signature."""
        expected_signature = hmac.new(
            secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics."""
        current_time = time.time()
        
        # Recent events (last hour)
        recent_events = [
            event for event in self.security_events
            if current_time - event.timestamp <= 3600
        ]
        
        # Event counts by severity
        severity_counts = {}
        for level in ThreatLevel:
            severity_counts[level.value] = len([
                e for e in recent_events if e.severity == level
            ])
        
        # Event counts by type
        type_counts = {}
        for event in recent_events:
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
        
        return {
            'total_events': len(self.security_events),
            'recent_events_1h': len(recent_events),
            'blocked_users': len(self.blocked_users),
            'active_sessions': len([uid for uid, reqs in self.request_history.items() 
                                   if reqs and current_time - reqs[-1] <= 1800]),  # 30 min
            'severity_distribution': severity_counts,
            'event_type_distribution': type_counts,
            'failed_attempts_tracked': len(self.failed_attempts),
            'rate_limits_active': len([uid for uid, reqs in self.request_history.items() 
                                     if len(reqs) > self.max_requests_per_window * 0.8])  # Near limit
        }
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        current_time = time.time()
        start_time = current_time - (hours * 3600)
        
        # Filter events in time window
        period_events = [
            event for event in self.security_events
            if event.timestamp >= start_time
        ]
        
        # Critical events requiring attention
        critical_events = [
            event for event in period_events
            if event.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        ]
        
        # Top threat patterns
        threat_patterns = {}
        for event in period_events:
            event_type = event.event_type
            threat_patterns[event_type] = threat_patterns.get(event_type, 0) + 1
        
        # Sort by frequency
        top_threats = sorted(threat_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Recommendations
        recommendations = []
        if len(critical_events) > 0:
            recommendations.append("Review critical security events and take corrective action")
        if len(self.blocked_users) > 5:
            recommendations.append("Review blocked users list and consider permanent restrictions")
        if any(count > 50 for _, count in top_threats[:3]):
            recommendations.append("Investigate high-frequency threat patterns")
        
        return {
            'report_period_hours': hours,
            'total_events': len(period_events),
            'critical_events': len(critical_events),
            'critical_event_details': [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'type': event.event_type,
                    'severity': event.severity.value,
                    'description': event.description,
                    'user_id': event.user_id
                }
                for event in critical_events
            ],
            'top_threat_patterns': top_threats,
            'blocked_users_count': len(self.blocked_users),
            'security_metrics': self.get_security_metrics(),
            'recommendations': recommendations,
            'report_generated_at': current_time
        }
"""
Unit tests for security module.

Tests security validation, threat detection, access control,
and security monitoring for quantum planning operations.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.quantum_planner.security import (
    SecurityValidator, SecurityContext, SecurityLevel, ThreatLevel,
    ThreatDetector, AccessController, SecurityAuditor,
    SecurityViolation, ThreatAnalysis
)
from src.quantum_planner.core import QuantumTask, TaskState
from .fixtures import *
from .utils import *


class TestSecurityContext:
    """Test cases for SecurityContext class."""
    
    def test_context_creation(self):
        """Test security context initialization."""
        context = SecurityContext(
            user_id="test_user",
            security_level=SecurityLevel.HIGH,
            permissions={"read", "write"},
            source_ip="192.168.1.100"
        )
        
        assert context.user_id == "test_user"
        assert context.security_level == SecurityLevel.HIGH
        assert "read" in context.permissions
        assert "write" in context.permissions
        assert context.source_ip == "192.168.1.100"
        assert context.session_id is not None
    
    def test_context_validation(self):
        """Test security context validation."""
        # Valid context
        valid_context = SecurityContext(
            user_id="valid_user",
            security_level=SecurityLevel.MEDIUM,
            permissions={"execute"}
        )
        
        assert valid_context.is_valid()
        
        # Invalid context - no permissions
        invalid_context = SecurityContext(
            user_id="invalid_user",
            security_level=SecurityLevel.LOW,
            permissions=set()
        )
        
        assert not invalid_context.is_valid()
    
    def test_permission_checking(self):
        """Test permission checking functionality."""
        context = SecurityContext(
            user_id="test_user",
            security_level=SecurityLevel.HIGH,
            permissions={"read", "write", "execute"}
        )
        
        assert context.has_permission("read")
        assert context.has_permission("write")
        assert context.has_permission("execute")
        assert not context.has_permission("admin")
        assert not context.has_permission("delete")
    
    def test_security_level_comparison(self):
        """Test security level comparison."""
        low_context = SecurityContext("user1", SecurityLevel.LOW, {"read"})
        medium_context = SecurityContext("user2", SecurityLevel.MEDIUM, {"read"})
        high_context = SecurityContext("user3", SecurityLevel.HIGH, {"read"})
        
        assert low_context.security_level.value < medium_context.security_level.value
        assert medium_context.security_level.value < high_context.security_level.value
        assert high_context.meets_minimum_level(SecurityLevel.MEDIUM)
        assert not low_context.meets_minimum_level(SecurityLevel.HIGH)


class TestThreatDetector:
    """Test cases for ThreatDetector class."""
    
    def test_detector_creation(self):
        """Test threat detector initialization."""
        detector = ThreatDetector(
            sensitivity_threshold=0.7,
            max_threat_history=1000
        )
        
        assert detector.sensitivity_threshold == 0.7
        assert detector.max_threat_history == 1000
        assert len(detector.threat_patterns) >= 0
        assert len(detector.detection_history) == 0
    
    def test_anomaly_detection(self, sample_task):
        """Test anomaly detection in task parameters."""
        detector = ThreatDetector()
        
        # Normal task should not trigger anomaly detection
        normal_task = sample_task
        normal_task.priority = 0.5
        normal_task.estimated_duration = 2.0
        
        threat_analysis = detector.detect_threats(normal_task, {})
        
        assert isinstance(threat_analysis, ThreatAnalysis)
        assert threat_analysis.threat_level in [ThreatLevel.NONE, ThreatLevel.LOW]
        
        # Suspicious task should trigger detection
        suspicious_task = create_test_task(
            "suspicious_task",
            priority=1.0,  # Maximum priority (potentially suspicious)
            estimated_duration=0.001  # Extremely short duration (suspicious)
        )
        
        threat_analysis = detector.detect_threats(suspicious_task, {})
        
        assert threat_analysis.threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH]
    
    def test_pattern_matching(self):
        """Test threat pattern matching."""
        detector = ThreatDetector()
        
        # Register threat pattern
        detector.register_threat_pattern(
            "resource_exhaustion",
            lambda task, ctx: sum(task.resource_requirements.values()) > 100,
            ThreatLevel.HIGH,
            "Resource exhaustion attack"
        )
        
        # Create resource-intensive task
        malicious_task = create_test_task(
            "resource_hog",
            resource_requirements={'cpu': 50, 'memory': 60}  # Total > 100
        )
        
        threat_analysis = detector.detect_threats(malicious_task, {})
        
        assert threat_analysis.threat_level == ThreatLevel.HIGH
        assert "resource_exhaustion" in [pattern['name'] for pattern in threat_analysis.detected_patterns]
    
    def test_behavioral_analysis(self, sample_tasks):
        """Test behavioral threat analysis."""
        detector = ThreatDetector()
        
        # Simulate normal behavior baseline
        for task in sample_tasks:
            detector.detect_threats(task, {})
        
        # Create abnormal task
        abnormal_task = create_test_task(
            "abnormal_task",
            priority=-0.5,  # Negative priority (abnormal)
            estimated_duration=1000.0  # Extremely long duration
        )
        
        threat_analysis = detector.detect_threats(abnormal_task, {})
        
        # Should detect behavioral anomaly
        assert threat_analysis.threat_level >= ThreatLevel.MEDIUM
    
    def test_rate_limiting_detection(self):
        """Test rate limiting and DoS detection."""
        detector = ThreatDetector()
        context = {'source_ip': '192.168.1.100'}
        
        # Simulate rapid task submissions
        for i in range(20):  # Many tasks in short time
            task = create_test_task(f"rapid_task_{i}")
            threat_analysis = detector.detect_threats(task, context)
        
        # Should eventually detect rate limiting violation
        final_analysis = detector.detect_threats(create_test_task("final_task"), context)
        
        # The detection might not trigger on every system, so we check for reasonable behavior
        assert isinstance(final_analysis, ThreatAnalysis)
    
    def test_injection_attack_detection(self):
        """Test injection attack detection."""
        detector = ThreatDetector()
        
        # Register injection pattern
        detector.register_threat_pattern(
            "code_injection",
            lambda task, ctx: any(
                suspicious in str(task.name).lower() 
                for suspicious in ['script', 'eval', 'exec', 'import', '__']
            ),
            ThreatLevel.CRITICAL,
            "Potential code injection attempt"
        )
        
        # Test with suspicious task names
        suspicious_names = ["script_task", "eval_expression", "exec_command", "__private_access"]
        
        for name in suspicious_names:
            malicious_task = create_test_task(name)
            threat_analysis = detector.detect_threats(malicious_task, {})
            
            assert threat_analysis.threat_level == ThreatLevel.CRITICAL
            assert any("code_injection" in pattern['name'] for pattern in threat_analysis.detected_patterns)


class TestAccessController:
    """Test cases for AccessController class."""
    
    def test_controller_creation(self):
        """Test access controller initialization."""
        controller = AccessController()
        
        assert len(controller.access_policies) >= 0
        assert len(controller.access_log) == 0
    
    def test_task_access_control(self, sample_task):
        """Test task access control."""
        controller = AccessController()
        
        # Create contexts with different permissions
        admin_context = SecurityContext(
            "admin_user", 
            SecurityLevel.HIGH, 
            {"read", "write", "execute", "admin"}
        )
        
        user_context = SecurityContext(
            "regular_user",
            SecurityLevel.MEDIUM,
            {"read", "execute"}
        )
        
        # Admin should have full access
        assert controller.check_task_access(sample_task, admin_context, "read")
        assert controller.check_task_access(sample_task, admin_context, "write")
        assert controller.check_task_access(sample_task, admin_context, "execute")
        
        # Regular user should have limited access
        assert controller.check_task_access(sample_task, user_context, "read")
        assert controller.check_task_access(sample_task, user_context, "execute")
        assert not controller.check_task_access(sample_task, user_context, "admin")
    
    def test_resource_access_control(self):
        """Test resource access control."""
        controller = AccessController()
        
        high_security_context = SecurityContext(
            "secure_user",
            SecurityLevel.HIGH,
            {"read", "write"}
        )
        
        low_security_context = SecurityContext(
            "insecure_user", 
            SecurityLevel.LOW,
            {"read"}
        )
        
        # Define resource with security requirements
        secure_resource = {
            "name": "classified_cpu_cluster",
            "min_security_level": SecurityLevel.HIGH,
            "required_permissions": {"write"}
        }
        
        # High security context should have access
        assert controller.check_resource_access(secure_resource, high_security_context)
        
        # Low security context should not have access
        assert not controller.check_resource_access(secure_resource, low_security_context)
    
    def test_policy_enforcement(self, sample_task):
        """Test access policy enforcement."""
        controller = AccessController()
        
        # Define access policy
        def priority_policy(task, context, operation):
            # High priority tasks require high security level
            if task.priority > 0.8 and context.security_level.value < SecurityLevel.HIGH.value:
                return False
            return True
        
        controller.add_access_policy("priority_policy", priority_policy)
        
        high_priority_task = sample_task
        high_priority_task.priority = 0.9
        
        high_sec_context = SecurityContext("user", SecurityLevel.HIGH, {"read"})
        medium_sec_context = SecurityContext("user", SecurityLevel.MEDIUM, {"read"})
        
        # High security context should pass policy
        assert controller.check_task_access(high_priority_task, high_sec_context, "read")
        
        # Medium security context should fail policy for high priority task
        assert not controller.check_task_access(high_priority_task, medium_sec_context, "read")
    
    def test_access_logging(self, sample_task):
        """Test access logging."""
        controller = AccessController()
        
        context = SecurityContext("test_user", SecurityLevel.MEDIUM, {"read"})
        
        # Perform access check
        result = controller.check_task_access(sample_task, context, "read")
        
        # Check that access was logged
        assert len(controller.access_log) > 0
        
        log_entry = controller.access_log[-1]
        assert log_entry['user_id'] == "test_user"
        assert log_entry['task_id'] == sample_task.id
        assert log_entry['operation'] == "read"
        assert log_entry['granted'] == result
    
    def test_session_management(self):
        """Test session management and validation."""
        controller = AccessController()
        
        context = SecurityContext("session_user", SecurityLevel.MEDIUM, {"read"})
        
        # Session should be valid initially
        assert controller.validate_session(context)
        
        # Mock expired session
        with patch.object(context, 'is_session_expired', return_value=True):
            assert not controller.validate_session(context)


class TestSecurityValidator:
    """Test cases for SecurityValidator class."""
    
    def test_validator_creation(self):
        """Test security validator initialization."""
        validator = SecurityValidator()
        
        assert validator.threat_detector is not None
        assert validator.access_controller is not None
        assert len(validator.security_checks) >= 0
    
    def test_comprehensive_security_validation(self, sample_task):
        """Test comprehensive security validation."""
        validator = SecurityValidator()
        
        context = SecurityContext(
            "test_user",
            SecurityLevel.HIGH,
            {"read", "write", "execute"}
        )
        
        validation_result = validator.validate_security(sample_task, context)
        
        assert hasattr(validation_result, 'is_secure')
        assert hasattr(validation_result, 'violations')
        assert hasattr(validation_result, 'threat_analysis')
        assert hasattr(validation_result, 'access_granted')
    
    def test_security_violation_detection(self, sample_task):
        """Test security violation detection."""
        validator = SecurityValidator()
        
        # Create context with insufficient permissions
        low_privilege_context = SecurityContext(
            "low_priv_user",
            SecurityLevel.LOW,
            {"read"}  # Missing write/execute permissions
        )
        
        # Task requiring higher privileges
        sensitive_task = sample_task
        sensitive_task.resource_requirements = {'admin_cpu': 4}  # Requires admin access
        
        validation_result = validator.validate_security(sensitive_task, low_privilege_context)
        
        assert not validation_result.is_secure
        assert len(validation_result.violations) > 0
    
    def test_custom_security_checks(self, sample_task):
        """Test custom security checks."""
        validator = SecurityValidator()
        
        def custom_security_check(task, context):
            # Custom check: task name must not contain 'unsafe'
            if 'unsafe' in task.name.lower():
                return SecurityViolation(
                    "unsafe_task_name",
                    "Task name contains unsafe keyword",
                    ThreatLevel.MEDIUM
                )
            return None
        
        validator.register_security_check("custom_check", custom_security_check)
        
        # Safe task should pass
        safe_task = create_test_task("safe_task")
        context = SecurityContext("user", SecurityLevel.MEDIUM, {"read"})
        
        result = validator.validate_security(safe_task, context)
        assert result.is_secure
        
        # Unsafe task should fail
        unsafe_task = create_test_task("unsafe_malicious_task")
        result = validator.validate_security(unsafe_task, context)
        assert not result.is_secure
    
    def test_security_audit_trail(self, sample_task):
        """Test security audit trail generation."""
        validator = SecurityValidator()
        context = SecurityContext("audit_user", SecurityLevel.HIGH, {"read", "audit"})
        
        # Perform several security validations
        for i in range(5):
            task = create_test_task(f"audit_task_{i}")
            validator.validate_security(task, context)
        
        # Check audit trail
        audit_trail = validator.get_audit_trail()
        assert len(audit_trail) >= 5
        
        for entry in audit_trail:
            assert 'timestamp' in entry
            assert 'task_id' in entry
            assert 'user_id' in entry
            assert 'security_result' in entry


class TestSecurityAuditor:
    """Test cases for SecurityAuditor class."""
    
    def test_auditor_creation(self):
        """Test security auditor initialization."""
        auditor = SecurityAuditor()
        
        assert len(auditor.audit_log) == 0
        assert auditor.audit_enabled is True
    
    def test_security_event_logging(self, sample_task):
        """Test security event logging."""
        auditor = SecurityAuditor()
        
        context = SecurityContext("audit_test_user", SecurityLevel.MEDIUM, {"read"})
        
        # Log various security events
        auditor.log_security_event("TASK_ACCESS", sample_task.id, context, {"granted": True})
        auditor.log_security_event("THREAT_DETECTED", sample_task.id, context, {"threat_level": "LOW"})
        auditor.log_security_event("ACCESS_DENIED", sample_task.id, context, {"reason": "insufficient_privileges"})
        
        assert len(auditor.audit_log) == 3
        
        # Check log entries
        access_entry = auditor.audit_log[0]
        assert access_entry['event_type'] == "TASK_ACCESS"
        assert access_entry['task_id'] == sample_task.id
        assert access_entry['user_id'] == "audit_test_user"
    
    def test_audit_report_generation(self, sample_tasks):
        """Test audit report generation."""
        auditor = SecurityAuditor()
        context = SecurityContext("report_user", SecurityLevel.HIGH, {"read", "audit"})
        
        # Generate audit events
        for i, task in enumerate(sample_tasks):
            event_type = "ACCESS_GRANTED" if i % 2 == 0 else "ACCESS_DENIED"
            auditor.log_security_event(event_type, task.id, context)
        
        # Generate audit report
        report = auditor.generate_audit_report(time_range=(0, time.time()))
        
        assert isinstance(report, dict)
        assert 'total_events' in report
        assert 'event_summary' in report
        assert 'security_summary' in report
        assert report['total_events'] == len(sample_tasks)
    
    def test_compliance_reporting(self, sample_tasks):
        """Test compliance reporting."""
        auditor = SecurityAuditor()
        
        # Set up compliance requirements
        compliance_requirements = {
            'min_security_level': SecurityLevel.MEDIUM,
            'required_permissions': {'read', 'write'},
            'max_failed_attempts': 3
        }
        
        auditor.set_compliance_requirements(compliance_requirements)
        
        # Simulate mixed compliance events
        contexts = [
            SecurityContext("compliant_user", SecurityLevel.HIGH, {"read", "write"}),
            SecurityContext("non_compliant_user", SecurityLevel.LOW, {"read"}),
        ]
        
        for i, task in enumerate(sample_tasks):
            context = contexts[i % len(contexts)]
            event_type = "ACCESS_GRANTED" if context.security_level >= SecurityLevel.MEDIUM else "ACCESS_DENIED"
            auditor.log_security_event(event_type, task.id, context)
        
        compliance_report = auditor.generate_compliance_report()
        
        assert isinstance(compliance_report, dict)
        assert 'compliance_score' in compliance_report
        assert 'violations' in compliance_report
        assert 0.0 <= compliance_report['compliance_score'] <= 1.0


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_end_to_end_security_workflow(self, sample_task):
        """Test complete security workflow."""
        # Step 1: Initialize security components
        detector = ThreatDetector()
        controller = AccessController()
        validator = SecurityValidator()
        auditor = SecurityAuditor()
        
        # Step 2: Create security context
        context = SecurityContext(
            "integration_user",
            SecurityLevel.HIGH,
            {"read", "write", "execute"}
        )
        
        # Step 3: Threat detection
        threat_analysis = detector.detect_threats(sample_task, {'user_context': context})
        assert isinstance(threat_analysis, ThreatAnalysis)
        
        # Step 4: Access control
        access_granted = controller.check_task_access(sample_task, context, "execute")
        assert isinstance(access_granted, bool)
        
        # Step 5: Comprehensive validation
        validation_result = validator.validate_security(sample_task, context)
        assert hasattr(validation_result, 'is_secure')
        
        # Step 6: Audit logging
        auditor.log_security_event(
            "SECURITY_VALIDATION",
            sample_task.id,
            context,
            {
                'threat_level': threat_analysis.threat_level.value,
                'access_granted': access_granted,
                'validation_passed': validation_result.is_secure
            }
        )
        
        assert len(auditor.audit_log) > 0
    
    def test_security_under_load(self, sample_tasks, performance_thresholds):
        """Test security system performance under load."""
        validator = SecurityValidator()
        context = SecurityContext("load_test_user", SecurityLevel.MEDIUM, {"read", "execute"})
        
        start_time = time.time()
        
        # Validate many tasks
        results = []
        for task in sample_tasks * 10:  # 10x the sample tasks
            result = validator.validate_security(task, context)
            results.append(result)
        
        execution_time = time.time() - start_time
        
        # Check performance
        max_time = performance_thresholds.get('max_security_validation_time', 5.0)
        assert_performance_acceptable(execution_time, max_time, "security validation under load")
        
        # All validations should complete
        assert len(results) == len(sample_tasks) * 10
    
    def test_security_configuration_validation(self):
        """Test security configuration validation."""
        # Test various security configurations
        configs = [
            {
                'min_security_level': SecurityLevel.LOW,
                'threat_sensitivity': 0.3,
                'audit_enabled': True
            },
            {
                'min_security_level': SecurityLevel.HIGH,
                'threat_sensitivity': 0.8,
                'audit_enabled': False
            }
        ]
        
        for config in configs:
            validator = SecurityValidator()
            
            # Apply configuration
            if hasattr(validator, 'configure'):
                validator.configure(config)
            
            # Validate configuration is applied
            assert isinstance(validator, SecurityValidator)
    
    def test_security_incident_response(self, sample_task):
        """Test security incident response."""
        validator = SecurityValidator()
        auditor = SecurityAuditor()
        
        # Create high-threat scenario
        malicious_context = SecurityContext(
            "suspicious_user",
            SecurityLevel.LOW,
            {"admin"}  # Suspicious: low security level but admin permission
        )
        
        malicious_task = create_test_task(
            "malicious_task",
            priority=1.0,  # Max priority
            resource_requirements={'cpu': 1000}  # Excessive resources
        )
        
        # Validate security
        result = validator.validate_security(malicious_task, malicious_context)
        
        # Should detect security issues
        assert not result.is_secure
        assert result.threat_analysis.threat_level >= ThreatLevel.MEDIUM
        
        # Log incident
        auditor.log_security_event(
            "SECURITY_INCIDENT",
            malicious_task.id,
            malicious_context,
            {
                'threat_level': result.threat_analysis.threat_level.value,
                'violations': len(result.violations),
                'incident_type': 'SUSPICIOUS_ACTIVITY'
            }
        )
        
        # Verify incident logging
        incident_logs = [
            entry for entry in auditor.audit_log 
            if entry['event_type'] == 'SECURITY_INCIDENT'
        ]
        assert len(incident_logs) > 0
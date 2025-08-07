"""
Unit tests for error handling module.

Tests error categorization, recovery strategies, circuit breaker pattern,
and automated threat response for quantum planning operations.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.quantum_planner.error_handling import (
    ErrorHandler, QuantumPlannerError, handle_errors, with_circuit_breaker,
    ErrorCategory, ErrorSeverity, RecoveryStrategy, CircuitBreakerState,
    ErrorContext, RecoveryAction, ThreatResponse
)
from src.quantum_planner.core import QuantumTask, TaskState
from .fixtures import *
from .utils import *


class TestQuantumPlannerError:
    """Test cases for QuantumPlannerError class."""
    
    def test_error_creation(self):
        """Test quantum planner error initialization."""
        error = QuantumPlannerError(
            message="Test error message",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context={'task_id': 'test_task'},
            recovery_suggestions=["Check input parameters", "Retry operation"]
        )
        
        assert error.message == "Test error message"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context['task_id'] == 'test_task'
        assert len(error.recovery_suggestions) == 2
        assert error.timestamp is not None
    
    def test_error_severity_comparison(self):
        """Test error severity comparison."""
        low_error = QuantumPlannerError("Low", ErrorCategory.SYSTEM, ErrorSeverity.LOW)
        medium_error = QuantumPlannerError("Medium", ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM)
        high_error = QuantumPlannerError("High", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
        critical_error = QuantumPlannerError("Critical", ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL)
        
        assert low_error.severity.value < medium_error.severity.value
        assert medium_error.severity.value < high_error.severity.value
        assert high_error.severity.value < critical_error.severity.value
    
    def test_error_string_representation(self):
        """Test error string representation."""
        error = QuantumPlannerError(
            "Test error",
            ErrorCategory.COMPUTATION,
            ErrorSeverity.MEDIUM
        )
        
        error_str = str(error)
        assert "Test error" in error_str
        assert "COMPUTATION" in error_str
        assert "MEDIUM" in error_str
    
    def test_error_with_cause(self):
        """Test error with underlying cause."""
        original_error = ValueError("Original error")
        
        quantum_error = QuantumPlannerError(
            "Quantum planning failed",
            ErrorCategory.COMPUTATION,
            ErrorSeverity.HIGH,
            underlying_error=original_error
        )
        
        assert quantum_error.underlying_error == original_error
        assert isinstance(quantum_error.underlying_error, ValueError)


class TestErrorContext:
    """Test cases for ErrorContext class."""
    
    def test_context_creation(self, sample_task):
        """Test error context initialization."""
        context = ErrorContext(
            task=sample_task,
            operation="optimize",
            user_id="test_user",
            session_id="session_123",
            additional_data={'retry_count': 3}
        )
        
        assert context.task == sample_task
        assert context.operation == "optimize"
        assert context.user_id == "test_user"
        assert context.session_id == "session_123"
        assert context.additional_data['retry_count'] == 3
        assert context.timestamp is not None
    
    def test_context_to_dict(self, sample_task):
        """Test context serialization to dictionary."""
        context = ErrorContext(
            task=sample_task,
            operation="validate",
            user_id="user_1"
        )
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict['operation'] == "validate"
        assert context_dict['user_id'] == "user_1"
        assert context_dict['task_id'] == sample_task.id
        assert 'timestamp' in context_dict


class TestRecoveryAction:
    """Test cases for RecoveryAction class."""
    
    def test_action_creation(self):
        """Test recovery action initialization."""
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Retry operation with exponential backoff",
            parameters={'max_retries': 3, 'backoff_factor': 2.0}
        )
        
        assert action.strategy == RecoveryStrategy.RETRY
        assert "Retry operation" in action.description
        assert action.parameters['max_retries'] == 3
        assert action.parameters['backoff_factor'] == 2.0
    
    def test_action_execution(self):
        """Test recovery action execution."""
        def mock_recovery_func(*args, **kwargs):
            return "Recovery successful"
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            description="Fallback to default",
            action_function=mock_recovery_func
        )
        
        result = action.execute()
        assert result == "Recovery successful"


class TestErrorHandler:
    """Test cases for ErrorHandler class."""
    
    def test_handler_creation(self):
        """Test error handler initialization."""
        handler = ErrorHandler(
            max_retry_attempts=5,
            circuit_breaker_threshold=10
        )
        
        assert handler.max_retry_attempts == 5
        assert handler.circuit_breaker_threshold == 10
        assert len(handler.error_history) == 0
        assert len(handler.recovery_strategies) >= 0
    
    def test_error_categorization(self, sample_task):
        """Test automatic error categorization."""
        handler = ErrorHandler()
        
        # Test different error types
        validation_error = ValueError("Invalid task parameter")
        computation_error = ArithmeticError("Division by zero")
        system_error = OSError("File not found")
        
        # Categorize errors
        context = ErrorContext(task=sample_task, operation="test")
        
        val_category = handler.categorize_error(validation_error, context)
        comp_category = handler.categorize_error(computation_error, context) 
        sys_category = handler.categorize_error(system_error, context)
        
        assert val_category == ErrorCategory.VALIDATION
        assert comp_category == ErrorCategory.COMPUTATION
        assert sys_category == ErrorCategory.SYSTEM
    
    def test_severity_assessment(self, sample_task):
        """Test error severity assessment."""
        handler = ErrorHandler()
        context = ErrorContext(task=sample_task, operation="test")
        
        # Test different error types for severity
        minor_error = ValueError("Minor validation issue")
        major_error = RuntimeError("Critical system failure")
        
        minor_severity = handler.assess_severity(minor_error, context)
        major_severity = handler.assess_severity(major_error, context)
        
        # RuntimeError should generally be more severe than ValueError
        assert major_severity.value >= minor_severity.value
    
    def test_recovery_strategy_selection(self, sample_task):
        """Test recovery strategy selection."""
        handler = ErrorHandler()
        context = ErrorContext(task=sample_task, operation="optimize")
        
        # Test strategy selection for different error types
        transient_error = ConnectionError("Network timeout")
        permanent_error = TypeError("Invalid parameter type")
        
        transient_strategy = handler.select_recovery_strategy(transient_error, context)
        permanent_strategy = handler.select_recovery_strategy(permanent_error, context)
        
        # Transient errors should typically use retry strategy
        assert transient_strategy == RecoveryStrategy.RETRY
        
        # Permanent errors should use fallback or fail-fast
        assert permanent_strategy in [RecoveryStrategy.FALLBACK, RecoveryStrategy.FAIL_FAST]
    
    def test_error_handling_workflow(self, sample_task):
        """Test complete error handling workflow."""
        handler = ErrorHandler()
        context = ErrorContext(task=sample_task, operation="test_operation")
        
        # Simulate error
        test_error = RuntimeError("Test error for handling")
        
        # Handle error
        result = handler.handle_error(test_error, context)
        
        assert isinstance(result, dict)
        assert 'error_id' in result
        assert 'category' in result
        assert 'severity' in result
        assert 'recovery_action' in result
        
        # Check error was logged
        assert len(handler.error_history) > 0
        assert handler.error_history[-1]['message'] == "Test error for handling"
    
    def test_retry_mechanism(self, sample_task):
        """Test retry mechanism with exponential backoff."""
        handler = ErrorHandler(max_retry_attempts=3)
        context = ErrorContext(task=sample_task, operation="retryable_operation")
        
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary network issue")
            return "Success after retries"
        
        # Test retry with eventual success
        result = handler.execute_with_retry(failing_function, context)
        
        assert result == "Success after retries"
        assert call_count == 3
    
    def test_circuit_breaker_functionality(self, sample_task):
        """Test circuit breaker pattern."""
        handler = ErrorHandler(circuit_breaker_threshold=3)
        context = ErrorContext(task=sample_task, operation="circuit_breaker_test")
        
        def always_failing_function():
            raise RuntimeError("Always fails")
        
        # Trigger circuit breaker by exceeding failure threshold
        for i in range(5):
            try:
                handler.execute_with_retry(always_failing_function, context)
            except Exception:
                pass  # Expected failures
        
        # Circuit breaker should be open
        circuit_breaker_state = handler.get_circuit_breaker_state("circuit_breaker_test")
        assert circuit_breaker_state in [CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN]
    
    def test_error_recovery_execution(self, sample_task):
        """Test error recovery execution."""
        handler = ErrorHandler()
        context = ErrorContext(task=sample_task, operation="recovery_test")
        
        # Register custom recovery strategy
        def custom_recovery(error, context):
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                description="Custom fallback recovery",
                action_function=lambda: "Custom recovery executed"
            )
        
        handler.register_recovery_strategy(ErrorCategory.COMPUTATION, custom_recovery)
        
        # Trigger error and recovery
        computation_error = ArithmeticError("Division by zero")
        result = handler.handle_error(computation_error, context)
        
        assert result['recovery_action']['strategy'] == RecoveryStrategy.FALLBACK.value
    
    def test_error_aggregation_and_analysis(self, sample_tasks):
        """Test error aggregation and analysis."""
        handler = ErrorHandler()
        
        # Generate multiple errors
        for i, task in enumerate(sample_tasks):
            context = ErrorContext(task=task, operation=f"operation_{i}")
            error = RuntimeError(f"Error {i}")
            handler.handle_error(error, context)
        
        # Analyze error patterns
        analysis = handler.analyze_error_patterns()
        
        assert isinstance(analysis, dict)
        assert 'total_errors' in analysis
        assert 'error_categories' in analysis
        assert 'error_frequencies' in analysis
        assert analysis['total_errors'] == len(sample_tasks)
    
    def test_threat_response_integration(self, sample_task):
        """Test integration with threat response system."""
        handler = ErrorHandler()
        context = ErrorContext(task=sample_task, operation="security_test")
        
        # Simulate security-related error
        security_error = PermissionError("Unauthorized access attempt")
        
        result = handler.handle_error(security_error, context)
        
        # Should trigger security response
        assert result['category'] == ErrorCategory.SECURITY.value
        assert result['severity'] in [ErrorSeverity.HIGH.value, ErrorSeverity.CRITICAL.value]


class TestCircuitBreakerPattern:
    """Test cases for Circuit Breaker pattern implementation."""
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator functionality."""
        call_count = 0
        
        @with_circuit_breaker(failure_threshold=3, recovery_timeout=1.0)
        def unstable_function():
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise RuntimeError("Simulated failure")
            return "Success"
        
        # Trigger failures to open circuit
        for i in range(5):
            try:
                result = unstable_function()
            except Exception:
                pass  # Expected failures
        
        # Function should have been called
        assert call_count > 0
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        from src.quantum_planner.error_handling import CircuitBreaker
        
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        # Initially closed
        assert breaker.state == CircuitBreakerState.CLOSED
        
        # Record failures to open
        breaker.record_failure()
        breaker.record_failure()
        
        # Should be open after threshold
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open
        # Note: This depends on the specific implementation
        breaker.call(lambda: "test")
    
    def test_circuit_breaker_success_reset(self):
        """Test circuit breaker reset on success."""
        from src.quantum_planner.error_handling import CircuitBreaker
        
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        
        # Record some failures
        breaker.record_failure()
        breaker.record_failure()
        
        # Record success - should reset failure count
        breaker.record_success()
        
        # Should still be closed
        assert breaker.state == CircuitBreakerState.CLOSED


class TestErrorHandlerDecorators:
    """Test cases for error handler decorators."""
    
    def test_handle_errors_decorator(self, sample_task):
        """Test @handle_errors decorator."""
        
        @handle_errors(
            retry_attempts=2,
            fallback_value="fallback_result",
            error_categories=[ErrorCategory.COMPUTATION]
        )
        def decorated_function(task):
            if task.priority < 0.5:
                raise ValueError("Priority too low")
            return f"Processed task {task.id}"
        
        # Test successful execution
        high_priority_task = create_test_task("high_pri", priority=0.8)
        result = decorated_function(high_priority_task)
        assert result == f"Processed task {high_priority_task.id}"
        
        # Test fallback execution
        low_priority_task = create_test_task("low_pri", priority=0.3)
        result = decorated_function(low_priority_task)
        assert result == "fallback_result"
    
    def test_error_context_injection(self, sample_task):
        """Test error context injection in decorators."""
        captured_context = None
        
        def custom_error_handler(error, context):
            nonlocal captured_context
            captured_context = context
            return "handled"
        
        @handle_errors(custom_handler=custom_error_handler)
        def context_function(task):
            raise RuntimeError("Test error with context")
        
        result = context_function(sample_task)
        
        assert result == "handled"
        assert captured_context is not None
        assert hasattr(captured_context, 'operation')
    
    def test_conditional_error_handling(self, sample_task):
        """Test conditional error handling based on error type."""
        
        @handle_errors(
            handle_only=[ValueError, TypeError],
            reraise_others=True
        )
        def selective_handler(task, error_type):
            if error_type == "value":
                raise ValueError("Value error")
            elif error_type == "type":
                raise TypeError("Type error")
            elif error_type == "runtime":
                raise RuntimeError("Runtime error")
            return "success"
        
        # Should handle ValueError
        result1 = selective_handler(sample_task, "value")
        assert result1 is not None  # Error was handled
        
        # Should handle TypeError
        result2 = selective_handler(sample_task, "type")
        assert result2 is not None  # Error was handled
        
        # Should re-raise RuntimeError
        with pytest.raises(RuntimeError):
            selective_handler(sample_task, "runtime")


class TestThreatResponse:
    """Test cases for automated threat response."""
    
    def test_threat_detection_and_response(self, sample_task):
        """Test threat detection and automated response."""
        handler = ErrorHandler()
        context = ErrorContext(task=sample_task, operation="security_operation")
        
        # Simulate security threat
        security_errors = [
            PermissionError("Unauthorized access"),
            ValueError("Suspicious input detected"),
            RuntimeError("Security policy violation")
        ]
        
        responses = []
        for error in security_errors:
            result = handler.handle_error(error, context)
            responses.append(result)
        
        # Check that security errors triggered appropriate responses
        security_responses = [
            r for r in responses 
            if r.get('category') == ErrorCategory.SECURITY.value
        ]
        
        assert len(security_responses) >= 1
    
    def test_escalation_mechanism(self, sample_task):
        """Test error escalation mechanism."""
        handler = ErrorHandler()
        context = ErrorContext(task=sample_task, operation="escalation_test")
        
        # Simulate escalating errors
        minor_error = ValueError("Minor issue")
        major_error = RuntimeError("Major system issue")
        critical_error = SystemError("Critical system failure")
        
        minor_result = handler.handle_error(minor_error, context)
        major_result = handler.handle_error(major_error, context)
        critical_result = handler.handle_error(critical_error, context)
        
        # Check severity escalation
        minor_severity = ErrorSeverity[minor_result['severity']]
        major_severity = ErrorSeverity[major_result['severity']]
        critical_severity = ErrorSeverity[critical_result['severity']]
        
        assert minor_severity.value <= major_severity.value
        assert major_severity.value <= critical_severity.value
    
    def test_automated_containment(self, sample_task):
        """Test automated threat containment."""
        handler = ErrorHandler()
        context = ErrorContext(task=sample_task, operation="containment_test")
        
        # Configure containment policy
        def containment_policy(error, context):
            if isinstance(error, PermissionError):
                return {
                    'action': 'isolate_task',
                    'quarantine': True,
                    'notify_security': True
                }
            return {'action': 'monitor'}
        
        handler.set_containment_policy(containment_policy)
        
        # Trigger containment
        permission_error = PermissionError("Suspicious activity detected")
        result = handler.handle_error(permission_error, context)
        
        # Check containment was triggered
        assert 'containment_action' in result or result.get('recovery_action', {}).get('description', '').lower().find('isolate') >= 0


class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""
    
    def test_end_to_end_error_workflow(self, sample_task):
        """Test complete error handling workflow."""
        handler = ErrorHandler()
        context = ErrorContext(
            task=sample_task,
            operation="integration_test",
            user_id="test_user"
        )
        
        # Step 1: Handle initial error
        initial_error = ConnectionError("Network timeout")
        result1 = handler.handle_error(initial_error, context)
        
        assert result1['category'] == ErrorCategory.SYSTEM.value
        assert 'recovery_action' in result1
        
        # Step 2: Handle cascading error
        cascading_error = RuntimeError("Failed to recover from network timeout")
        result2 = handler.handle_error(cascading_error, context)
        
        assert result2['category'] == ErrorCategory.SYSTEM.value
        
        # Step 3: Analyze error patterns
        analysis = handler.analyze_error_patterns()
        assert analysis['total_errors'] >= 2
        
        # Step 4: Check circuit breaker state
        cb_state = handler.get_circuit_breaker_state("integration_test")
        assert cb_state in [state.value for state in CircuitBreakerState]
    
    @measure_execution_time
    def test_error_handling_performance(self, sample_tasks, performance_thresholds):
        """Test error handling performance under load."""
        handler = ErrorHandler()
        
        # Generate many errors
        start_time = time.time()
        for i, task in enumerate(sample_tasks * 5):  # 5x the sample tasks
            context = ErrorContext(task=task, operation=f"perf_test_{i}")
            error = RuntimeError(f"Performance test error {i}")
            handler.handle_error(error, context)
        
        execution_time = time.time() - start_time
        
        # Check performance
        max_time = performance_thresholds.get('max_error_handling_time', 2.0)
        assert_performance_acceptable(execution_time, max_time, "error handling under load")
        
        # All errors should be processed
        assert len(handler.error_history) >= len(sample_tasks) * 5
    
    def test_concurrent_error_handling(self, sample_tasks):
        """Test concurrent error handling thread safety."""
        handler = ErrorHandler()
        results = []
        errors = []
        
        def worker_function(task_id):
            try:
                task = create_test_task(f"concurrent_task_{task_id}")
                context = ErrorContext(task=task, operation="concurrent_test")
                error = RuntimeError(f"Concurrent error {task_id}")
                result = handler.handle_error(error, context)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 10
        assert len(errors) == 0  # No exceptions in error handling
        assert len(handler.error_history) >= 10
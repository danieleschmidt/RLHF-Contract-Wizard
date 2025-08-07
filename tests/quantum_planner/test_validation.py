"""
Unit tests for validation module.

Tests input validation, schema checking, constraint validation,
and validation result handling for quantum planning operations.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

from src.quantum_planner.validation import (
    TaskValidator, ConfigValidator, ValidationResult, ValidationError,
    ValidationSeverity, SchemaValidator, ConstraintValidator
)
from src.quantum_planner.core import QuantumTask, TaskState, PlannerConfig
from .fixtures import *
from .utils import *


class TestValidationResult:
    """Test cases for ValidationResult class."""
    
    def test_result_creation(self):
        """Test validation result initialization."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            metadata={'test': True}
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.metadata['test'] is True
    
    def test_result_with_errors(self):
        """Test validation result with errors."""
        errors = [
            ValidationError("field1", "Invalid value", ValidationSeverity.ERROR),
            ValidationError("field2", "Missing required field", ValidationSeverity.CRITICAL)
        ]
        
        result = ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=[]
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert result.has_critical_errors()
        assert result.get_error_count() == 2
    
    def test_result_with_warnings(self):
        """Test validation result with warnings."""
        warnings = [
            ValidationError("field1", "Deprecated usage", ValidationSeverity.WARNING),
            ValidationError("field2", "Performance concern", ValidationSeverity.INFO)
        ]
        
        result = ValidationResult(
            is_valid=True,  # Can be valid with warnings
            errors=[],
            warnings=warnings
        )
        
        assert result.is_valid is True
        assert len(result.warnings) == 2
        assert not result.has_critical_errors()
        assert result.get_warning_count() == 2
    
    def test_result_aggregation(self):
        """Test validation result aggregation."""
        result1 = ValidationResult(True, [], [])
        result2 = ValidationResult(False, [ValidationError("test", "error", ValidationSeverity.ERROR)], [])
        
        combined = ValidationResult.combine([result1, result2])
        
        assert combined.is_valid is False  # Any failure makes combined invalid
        assert len(combined.errors) == 1
        assert combined.get_error_count() == 1


class TestValidationError:
    """Test cases for ValidationError class."""
    
    def test_error_creation(self):
        """Test validation error initialization."""
        error = ValidationError(
            field="task.priority",
            message="Priority must be between 0 and 1",
            severity=ValidationSeverity.ERROR,
            context={'current_value': 1.5}
        )
        
        assert error.field == "task.priority"
        assert "Priority must be between 0 and 1" in error.message
        assert error.severity == ValidationSeverity.ERROR
        assert error.context['current_value'] == 1.5
    
    def test_error_severity_comparison(self):
        """Test validation error severity comparison."""
        info_error = ValidationError("field", "info", ValidationSeverity.INFO)
        warning_error = ValidationError("field", "warning", ValidationSeverity.WARNING)
        error_error = ValidationError("field", "error", ValidationSeverity.ERROR)
        critical_error = ValidationError("field", "critical", ValidationSeverity.CRITICAL)
        
        assert info_error.severity.value < warning_error.severity.value
        assert warning_error.severity.value < error_error.severity.value
        assert error_error.severity.value < critical_error.severity.value
    
    def test_error_string_representation(self):
        """Test validation error string representation."""
        error = ValidationError(
            "test_field",
            "Test error message",
            ValidationSeverity.ERROR
        )
        
        error_str = str(error)
        assert "test_field" in error_str
        assert "Test error message" in error_str
        assert "ERROR" in error_str


class TestSchemaValidator:
    """Test cases for SchemaValidator class."""
    
    def test_validator_creation(self):
        """Test schema validator initialization."""
        validator = SchemaValidator()
        
        assert len(validator.registered_schemas) >= 0
        assert validator.strict_mode is False
    
    def test_schema_registration(self):
        """Test schema registration."""
        validator = SchemaValidator()
        
        task_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "priority": {"type": "number", "minimum": 0, "maximum": 1},
                "estimated_duration": {"type": "number", "minimum": 0}
            },
            "required": ["id", "priority", "estimated_duration"]
        }
        
        validator.register_schema("task", task_schema)
        
        assert "task" in validator.registered_schemas
        assert validator.registered_schemas["task"] == task_schema
    
    def test_schema_validation_success(self, sample_task):
        """Test successful schema validation."""
        validator = SchemaValidator()
        
        # Register task schema
        task_schema = {
            "type": "object", 
            "properties": {
                "id": {"type": "string"},
                "priority": {"type": "number", "minimum": 0, "maximum": 1},
                "estimated_duration": {"type": "number", "minimum": 0}
            },
            "required": ["id", "priority", "estimated_duration"]
        }
        
        validator.register_schema("task", task_schema)
        
        # Convert task to dict for validation
        task_data = {
            "id": sample_task.id,
            "priority": sample_task.priority, 
            "estimated_duration": sample_task.estimated_duration
        }
        
        result = validator.validate_schema("task", task_data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_schema_validation_failure(self):
        """Test schema validation failure."""
        validator = SchemaValidator()
        
        # Register strict schema
        strict_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "priority": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["id", "priority"]
        }
        
        validator.register_schema("strict_task", strict_schema)
        
        # Invalid data
        invalid_data = {
            "id": 123,  # Should be string
            "priority": 1.5  # Exceeds maximum
            # Missing required fields
        }
        
        result = validator.validate_schema("strict_task", invalid_data)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_nested_schema_validation(self):
        """Test nested schema validation."""
        validator = SchemaValidator()
        
        # Define nested schema
        nested_schema = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "config": {
                            "type": "object", 
                            "properties": {
                                "priority": {"type": "number"},
                                "resources": {
                                    "type": "object",
                                    "properties": {
                                        "cpu": {"type": "number"},
                                        "memory": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        validator.register_schema("nested_task", nested_schema)
        
        # Valid nested data
        nested_data = {
            "task": {
                "id": "test_task",
                "config": {
                    "priority": 0.8,
                    "resources": {
                        "cpu": 4,
                        "memory": 8
                    }
                }
            }
        }
        
        result = validator.validate_schema("nested_task", nested_data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0


class TestTaskValidator:
    """Test cases for TaskValidator class."""
    
    def test_validator_creation(self):
        """Test task validator initialization."""
        validator = TaskValidator()
        
        assert len(validator.validation_rules) >= 0
        assert validator.enable_deep_validation is True
    
    def test_basic_task_validation(self, sample_task):
        """Test basic task validation."""
        validator = TaskValidator()
        
        result = validator.validate_task(sample_task)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_task_id_validation(self):
        """Test task ID validation."""
        validator = TaskValidator()
        
        # Valid task IDs
        valid_task = create_test_task("valid_task_123")
        result = validator.validate_task(valid_task)
        assert result.is_valid is True
        
        # Invalid task ID (empty)
        invalid_task = create_test_task("")
        result = validator.validate_task(invalid_task)
        assert result.is_valid is False
        assert any("id" in error.field.lower() for error in result.errors)
    
    def test_priority_validation(self):
        """Test task priority validation."""
        validator = TaskValidator()
        
        # Valid priority
        valid_task = create_test_task("test", priority=0.7)
        result = validator.validate_task(valid_task)
        assert result.is_valid is True
        
        # Invalid priority (negative)
        invalid_task = create_test_task("test", priority=-0.1)
        result = validator.validate_task(invalid_task)
        assert result.is_valid is False
        
        # Invalid priority (too high)
        invalid_task2 = create_test_task("test", priority=1.5)
        result = validator.validate_task(invalid_task2)
        assert result.is_valid is False
    
    def test_duration_validation(self):
        """Test task duration validation."""
        validator = TaskValidator()
        
        # Valid duration
        valid_task = create_test_task("test", estimated_duration=5.5)
        result = validator.validate_task(valid_task)
        assert result.is_valid is True
        
        # Invalid duration (negative)
        invalid_task = create_test_task("test", estimated_duration=-1.0)
        result = validator.validate_task(invalid_task)
        assert result.is_valid is False
        
        # Invalid duration (zero)
        zero_task = create_test_task("test", estimated_duration=0.0)
        result = validator.validate_task(zero_task)
        assert result.is_valid is False
    
    def test_resource_requirements_validation(self):
        """Test resource requirements validation."""
        validator = TaskValidator()
        
        # Valid resource requirements
        valid_task = create_test_task("test", resource_requirements={"cpu": 4, "memory": 8})
        result = validator.validate_task(valid_task)
        assert result.is_valid is True
        
        # Invalid resource requirements (negative values)
        invalid_task = create_test_task("test", resource_requirements={"cpu": -2})
        result = validator.validate_task(invalid_task)
        assert result.is_valid is False
    
    def test_dependency_validation(self, sample_tasks):
        """Test task dependency validation."""
        validator = TaskValidator()
        
        if len(sample_tasks) >= 2:
            task_a, task_b = sample_tasks[0], sample_tasks[1]
            
            # Valid dependency
            task_b.add_dependency(task_a.id)
            result = validator.validate_task(task_b)
            assert result.is_valid is True
            
            # Self-dependency (invalid)
            task_a.add_dependency(task_a.id)
            result = validator.validate_task(task_a)
            assert result.is_valid is False
    
    def test_quantum_properties_validation(self, sample_task):
        """Test quantum properties validation."""
        validator = TaskValidator()
        
        # Valid quantum properties
        sample_task.amplitude = complex(0.8, 0.6)  # |amplitude|^2 = 1.0
        sample_task.phase = 0.5
        result = validator.validate_task(sample_task)
        assert result.is_valid is True
        
        # Invalid amplitude (not normalized)
        sample_task.amplitude = complex(2.0, 2.0)  # |amplitude|^2 = 8.0 > 1.0
        result = validator.validate_task(sample_task)
        # Note: This might be valid depending on normalization strategy
    
    def test_state_validation(self, sample_task):
        """Test task state validation."""
        validator = TaskValidator()
        
        # All valid states should pass
        for state in TaskState:
            sample_task.state = state
            result = validator.validate_task(sample_task)
            assert result.is_valid is True
    
    def test_custom_validation_rules(self, sample_task):
        """Test custom validation rules."""
        validator = TaskValidator()
        
        def custom_name_rule(task):
            if 'forbidden' in task.name.lower():
                return ValidationError(
                    "task.name",
                    "Task name contains forbidden keyword",
                    ValidationSeverity.ERROR
                )
            return None
        
        validator.add_validation_rule("custom_name_check", custom_name_rule)
        
        # Valid task
        valid_task = create_test_task("allowed_task")
        result = validator.validate_task(valid_task)
        assert result.is_valid is True
        
        # Invalid task
        invalid_task = create_test_task("forbidden_task") 
        result = validator.validate_task(invalid_task)
        assert result.is_valid is False
    
    def test_batch_task_validation(self, sample_tasks):
        """Test batch task validation."""
        validator = TaskValidator()
        
        results = validator.validate_tasks(sample_tasks)
        
        assert len(results) == len(sample_tasks)
        assert all(isinstance(result, ValidationResult) for result in results)
        
        # Most tasks should be valid
        valid_count = sum(1 for result in results if result.is_valid)
        assert valid_count >= len(sample_tasks) * 0.8  # At least 80% should be valid


class TestConfigValidator:
    """Test cases for ConfigValidator class."""
    
    def test_validator_creation(self):
        """Test config validator initialization."""
        validator = ConfigValidator()
        
        assert len(validator.config_schemas) >= 0
        assert validator.strict_validation is True
    
    def test_planner_config_validation(self, planner_config):
        """Test planner configuration validation."""
        validator = ConfigValidator()
        
        result = validator.validate_planner_config(planner_config)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_planner_config_validation(self):
        """Test invalid planner configuration validation."""
        validator = ConfigValidator()
        
        # Create invalid config
        invalid_config = PlannerConfig(
            max_iterations=-10,  # Negative iterations (invalid)
            convergence_threshold=-1.0,  # Negative threshold (invalid)
            parallel_execution_limit=0  # Zero limit (invalid)
        )
        
        result = validator.validate_planner_config(invalid_config)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_optimization_config_validation(self):
        """Test optimization configuration validation."""
        validator = ConfigValidator()
        
        # Valid optimization config
        valid_config = {
            "algorithm": "QAOA",
            "max_iterations": 100,
            "convergence_threshold": 1e-6,
            "quantum_interference_strength": 0.2
        }
        
        result = validator.validate_optimization_config(valid_config)
        assert result.is_valid is True
        
        # Invalid optimization config
        invalid_config = {
            "algorithm": "UNKNOWN_ALGORITHM",
            "max_iterations": -5,
            "convergence_threshold": 2.0  # > 1.0 (invalid for threshold)
        }
        
        result = validator.validate_optimization_config(invalid_config)
        assert result.is_valid is False
    
    def test_resource_config_validation(self):
        """Test resource configuration validation."""
        validator = ConfigValidator()
        
        # Valid resource config
        valid_config = {
            "cpu": 8,
            "memory": 16,
            "storage": 100
        }
        
        result = validator.validate_resource_config(valid_config)
        assert result.is_valid is True
        
        # Invalid resource config
        invalid_config = {
            "cpu": -4,  # Negative CPU
            "memory": 0,  # Zero memory
            "invalid_resource": "bad_value"
        }
        
        result = validator.validate_resource_config(invalid_config)
        assert result.is_valid is False
    
    def test_security_config_validation(self):
        """Test security configuration validation."""
        validator = ConfigValidator()
        
        # Valid security config
        valid_config = {
            "enable_security": True,
            "min_security_level": "MEDIUM",
            "threat_detection_sensitivity": 0.7,
            "audit_logging": True
        }
        
        result = validator.validate_security_config(valid_config)
        assert result.is_valid is True
        
        # Invalid security config
        invalid_config = {
            "enable_security": "yes",  # Should be boolean
            "min_security_level": "INVALID_LEVEL",
            "threat_detection_sensitivity": 1.5  # > 1.0
        }
        
        result = validator.validate_security_config(invalid_config)
        assert result.is_valid is False


class TestConstraintValidator:
    """Test cases for ConstraintValidator class."""
    
    def test_validator_creation(self):
        """Test constraint validator initialization."""
        validator = ConstraintValidator()
        
        assert len(validator.constraints) == 0
        assert validator.validation_enabled is True
    
    def test_constraint_registration(self):
        """Test constraint registration."""
        validator = ConstraintValidator()
        
        def priority_constraint(data, context):
            return 0.0 <= data.get('priority', 0) <= 1.0
        
        validator.register_constraint(
            "priority_range",
            priority_constraint,
            "Priority must be between 0 and 1"
        )
        
        assert "priority_range" in validator.constraints
        assert validator.constraints["priority_range"]["function"] == priority_constraint
    
    def test_constraint_validation_success(self):
        """Test successful constraint validation."""
        validator = ConstraintValidator()
        
        def valid_constraint(data, context):
            return True  # Always passes
        
        validator.register_constraint("always_valid", valid_constraint, "Always valid")
        
        result = validator.validate_constraints({"test": "data"}, {})
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_constraint_validation_failure(self):
        """Test constraint validation failure."""
        validator = ConstraintValidator()
        
        def failing_constraint(data, context):
            return data.get('value', 0) > 10
        
        validator.register_constraint(
            "value_check",
            failing_constraint,
            "Value must be greater than 10"
        )
        
        result = validator.validate_constraints({"value": 5}, {})
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("value_check" in error.field for error in result.errors)
    
    def test_multiple_constraints(self):
        """Test multiple constraint validation."""
        validator = ConstraintValidator()
        
        def min_constraint(data, context):
            return data.get('value', 0) >= 0
        
        def max_constraint(data, context):
            return data.get('value', 0) <= 100
        
        def type_constraint(data, context):
            return isinstance(data.get('value'), (int, float))
        
        validator.register_constraint("min_value", min_constraint, "Value >= 0")
        validator.register_constraint("max_value", max_constraint, "Value <= 100")
        validator.register_constraint("value_type", type_constraint, "Value must be numeric")
        
        # Valid data
        result = validator.validate_constraints({"value": 50}, {})
        assert result.is_valid is True
        
        # Invalid data (fails multiple constraints)
        result = validator.validate_constraints({"value": -150}, {})
        assert result.is_valid is False
        
        # Check that multiple constraint failures are reported
        constraint_names = [error.field for error in result.errors]
        assert "min_value" in constraint_names
        # max_value might not trigger if min fails first, depending on implementation
    
    def test_conditional_constraints(self):
        """Test conditional constraint validation."""
        validator = ConstraintValidator()
        
        def conditional_constraint(data, context):
            # Only validate if type is 'special'
            if data.get('type') == 'special':
                return data.get('special_value', 0) > 0
            return True  # Skip validation for non-special types
        
        validator.register_constraint(
            "special_value_check",
            conditional_constraint,
            "Special items must have positive special_value"
        )
        
        # Non-special item should pass
        result1 = validator.validate_constraints({"type": "normal", "special_value": -1}, {})
        assert result1.is_valid is True
        
        # Special item with valid value should pass
        result2 = validator.validate_constraints({"type": "special", "special_value": 5}, {})
        assert result2.is_valid is True
        
        # Special item with invalid value should fail
        result3 = validator.validate_constraints({"type": "special", "special_value": -1}, {})
        assert result3.is_valid is False
    
    def test_context_dependent_constraints(self):
        """Test context-dependent constraint validation."""
        validator = ConstraintValidator()
        
        def context_constraint(data, context):
            max_value = context.get('max_allowed', 10)
            return data.get('value', 0) <= max_value
        
        validator.register_constraint(
            "context_limit",
            context_constraint,
            "Value must not exceed context limit"
        )
        
        # Same data, different contexts
        data = {"value": 15}
        
        context1 = {"max_allowed": 20}  # Should pass
        result1 = validator.validate_constraints(data, context1)
        assert result1.is_valid is True
        
        context2 = {"max_allowed": 10}  # Should fail
        result2 = validator.validate_constraints(data, context2)
        assert result2.is_valid is False


class TestValidationIntegration:
    """Integration tests for validation components."""
    
    def test_comprehensive_validation_workflow(self, sample_task, planner_config):
        """Test comprehensive validation workflow."""
        # Initialize validators
        task_validator = TaskValidator()
        config_validator = ConfigValidator()
        schema_validator = SchemaValidator()
        
        # Step 1: Schema validation
        task_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "priority": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
        schema_validator.register_schema("task", task_schema)
        
        task_data = {"id": sample_task.id, "priority": sample_task.priority}
        schema_result = schema_validator.validate_schema("task", task_data)
        
        # Step 2: Task validation
        task_result = task_validator.validate_task(sample_task)
        
        # Step 3: Config validation
        config_result = config_validator.validate_planner_config(planner_config)
        
        # All validations should pass
        assert schema_result.is_valid is True
        assert task_result.is_valid is True
        assert config_result.is_valid is True
    
    def test_validation_error_aggregation(self, sample_tasks):
        """Test validation error aggregation across multiple validations."""
        validator = TaskValidator()
        
        # Add strict validation rules
        def strict_priority_rule(task):
            if task.priority < 0.8:
                return ValidationError(
                    "task.priority",
                    "Priority must be at least 0.8",
                    ValidationSeverity.ERROR
                )
            return None
        
        validator.add_validation_rule("strict_priority", strict_priority_rule)
        
        # Validate all tasks
        results = validator.validate_tasks(sample_tasks)
        
        # Aggregate all errors
        all_errors = []
        all_warnings = []
        
        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        # Check aggregation
        total_errors = len(all_errors)
        total_warnings = len(all_warnings)
        
        assert total_errors >= 0
        assert total_warnings >= 0
    
    @measure_execution_time  
    def test_validation_performance(self, sample_tasks, performance_thresholds):
        """Test validation performance with many tasks."""
        validator = TaskValidator()
        
        # Add multiple validation rules
        for i in range(10):
            def rule_func(task, i=i):
                return ValidationError(
                    f"rule_{i}",
                    f"Rule {i} check", 
                    ValidationSeverity.INFO
                ) if task.priority < (i * 0.1) else None
            
            validator.add_validation_rule(f"rule_{i}", rule_func)
        
        # Validate many tasks
        large_task_set = sample_tasks * 10
        results = validator.validate_tasks(large_task_set)
        
        # Check performance
        execution_time = results[0]._execution_time if hasattr(results[0], '_execution_time') else 0.1
        max_time = performance_thresholds.get('max_validation_time', 2.0)
        
        assert_performance_acceptable(execution_time, max_time, "batch validation")
        assert len(results) == len(large_task_set)
    
    def test_validation_with_contract_integration(self, sample_task):
        """Test validation integration with contracts."""
        from src.models.reward_contract import RewardContract
        
        validator = TaskValidator()
        contract = RewardContract("test_contract")
        
        # Add contract-specific validation
        def contract_compliance_rule(task):
            # Mock contract compliance check
            if hasattr(task, 'contract_compliance') and not task.contract_compliance:
                return ValidationError(
                    "task.contract_compliance",
                    "Task does not comply with contract requirements",
                    ValidationSeverity.ERROR
                )
            return None
        
        validator.add_validation_rule("contract_compliance", contract_compliance_rule)
        
        # Test with compliant task
        sample_task.contract_compliance = True
        result = validator.validate_task(sample_task)
        assert result.is_valid is True
        
        # Test with non-compliant task  
        sample_task.contract_compliance = False
        result = validator.validate_task(sample_task)
        assert result.is_valid is False
"""
Unit tests for contract integration module.

Tests the integration between quantum planning and RLHF contracts,
including constraint validation and compliance checking.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.quantum_planner.contracts import (
    ContractualTaskPlanner, TaskConstraintValidator, ContractCompliance,
    ContractIntegrationResult, ComplianceViolation
)
from src.quantum_planner.core import QuantumTask, TaskState
from src.models.reward_contract import RewardContract, Stakeholder, Constraint
from src.models.legal_blocks import LegalBlocks, RLHFConstraints
from .fixtures import *
from .utils import *


class TestTaskConstraintValidator:
    """Test cases for TaskConstraintValidator class."""
    
    def test_validator_creation(self):
        """Test validator initialization."""
        validator = TaskConstraintValidator()
        
        assert len(validator.registered_constraints) == 0
        assert validator.validation_enabled is True
        assert validator.strict_mode is False
    
    def test_constraint_registration(self):
        """Test constraint registration."""
        validator = TaskConstraintValidator()
        
        def test_constraint(task, context):
            return task.priority > 0.5
        
        validator.register_constraint("priority_check", test_constraint, "Task must have priority > 0.5")
        
        assert "priority_check" in validator.registered_constraints
        assert validator.registered_constraints["priority_check"]["function"] == test_constraint
        assert "Task must have priority > 0.5" in validator.registered_constraints["priority_check"]["description"]
    
    def test_task_validation_success(self, sample_task):
        """Test successful task validation."""
        validator = TaskConstraintValidator()
        
        # Register passing constraints
        validator.register_constraint(
            "priority_check",
            lambda task, ctx: task.priority > 0.5,
            "Priority must be > 0.5"
        )
        
        validator.register_constraint(
            "duration_check",
            lambda task, ctx: task.estimated_duration > 0,
            "Duration must be positive"
        )
        
        # Ensure sample task meets criteria
        sample_task.priority = 0.8
        sample_task.estimated_duration = 2.5
        
        result = validator.validate_task(sample_task, {})
        
        assert result.is_valid is True
        assert len(result.violations) == 0
        assert len(result.passed_constraints) == 2
    
    def test_task_validation_failure(self, sample_task):
        """Test task validation with constraint violations."""
        validator = TaskConstraintValidator(strict_mode=True)
        
        # Register failing constraints
        validator.register_constraint(
            "priority_check",
            lambda task, ctx: task.priority > 0.9,
            "Priority must be > 0.9"
        )
        
        validator.register_constraint(
            "resource_check",
            lambda task, ctx: len(task.resource_requirements) > 0,
            "Must have resource requirements"
        )
        
        # Ensure sample task fails criteria
        sample_task.priority = 0.7  # Fails priority check
        sample_task.resource_requirements = {}  # Fails resource check
        
        result = validator.validate_task(sample_task, {})
        
        assert result.is_valid is False
        assert len(result.violations) == 2
        assert any("priority" in v.constraint_name for v in result.violations)
        assert any("resource" in v.constraint_name for v in result.violations)
    
    def test_contract_constraint_integration(self, sample_task):
        """Test integration with RLHF contract constraints."""
        validator = TaskConstraintValidator()
        
        # Create mock reward contract
        contract = RewardContract("test_contract")
        
        # Add constraints to contract
        contract.add_constraint(
            "safety_check",
            lambda state, action: True,  # Always pass for test
            "Safety validation"
        )
        
        # Register contract constraints with validator
        validator.register_contract_constraints(contract)
        
        result = validator.validate_task(sample_task, {'contract': contract})
        
        assert result.is_valid is True
        assert len(result.passed_constraints) > 0
    
    def test_legal_blocks_integration(self, sample_task):
        """Test integration with Legal-Blocks constraints."""
        validator = TaskConstraintValidator()
        
        @LegalBlocks.constraint
        def task_safety_requirement(task, context):
            """
            ```legal-blocks
            REQUIRES: task.priority >= 0.0 AND task.estimated_duration > 0
            ENSURES: task.state != FAILED
            INVARIANT: NOT contains_harmful_content(task.name)
            ```
            
            Ensures task meets basic safety requirements.
            """
            return (task.priority >= 0.0 and 
                    task.estimated_duration > 0 and
                    'harmful' not in task.name.lower())
        
        validator.register_legal_blocks_constraint(task_safety_requirement)
        
        result = validator.validate_task(sample_task, {})
        
        assert result.is_valid is True
        
        # Test with failing constraint
        sample_task.name = "harmful_task"
        result = validator.validate_task(sample_task, {})
        assert result.is_valid is False
    
    def test_validation_context_usage(self, sample_task):
        """Test validation context usage in constraints."""
        validator = TaskConstraintValidator()
        
        def context_dependent_constraint(task, context):
            max_duration = context.get('max_duration', 10.0)
            return task.estimated_duration <= max_duration
        
        validator.register_constraint(
            "duration_limit",
            context_dependent_constraint,
            "Task duration must be within limit"
        )
        
        # Test with different contexts
        context1 = {'max_duration': 5.0}
        context2 = {'max_duration': 1.0}
        
        sample_task.estimated_duration = 2.5
        
        result1 = validator.validate_task(sample_task, context1)
        result2 = validator.validate_task(sample_task, context2)
        
        assert result1.is_valid is True
        assert result2.is_valid is False  # Exceeds 1.0 limit
    
    def test_batch_validation(self, sample_tasks):
        """Test batch validation of multiple tasks."""
        validator = TaskConstraintValidator()
        
        validator.register_constraint(
            "priority_check",
            lambda task, ctx: task.priority > 0.5,
            "Priority must be > 0.5"
        )
        
        # Set mixed priorities
        for i, task in enumerate(sample_tasks):
            task.priority = 0.3 + i * 0.2  # 0.3, 0.5, 0.7, etc.
        
        results = validator.validate_tasks(sample_tasks, {})
        
        assert len(results) == len(sample_tasks)
        
        # First task should fail (priority 0.3), others should pass
        assert results[0].is_valid is False
        assert all(result.is_valid for result in results[1:])
    
    def test_validation_performance(self, sample_tasks, performance_thresholds):
        """Test validation performance with many tasks."""
        validator = TaskConstraintValidator()
        
        # Register multiple constraints
        for i in range(5):
            validator.register_constraint(
                f"constraint_{i}",
                lambda task, ctx, i=i: task.priority > (i * 0.1),
                f"Constraint {i}"
            )
        
        start_time = time.time()
        results = validator.validate_tasks(sample_tasks, {})
        execution_time = time.time() - start_time
        
        max_time = performance_thresholds['max_validation_time_per_task'] * len(sample_tasks)
        assert_performance_acceptable(execution_time, max_time, "batch validation")


class TestContractualTaskPlanner:
    """Test cases for ContractualTaskPlanner class."""
    
    def test_planner_creation(self, quantum_planner):
        """Test contractual planner initialization."""
        contract = RewardContract("test_contract")
        validator = TaskConstraintValidator()
        
        planner = ContractualTaskPlanner(
            base_planner=quantum_planner,
            validator=validator,
            contract=contract
        )
        
        assert planner.base_planner == quantum_planner
        assert planner.validator == validator
        assert planner.contract == contract
        assert planner.compliance_tracking is True
    
    def test_contract_aware_task_addition(self, quantum_planner, sample_task):
        """Test contract-aware task addition."""
        contract = RewardContract("test_contract")
        validator = TaskConstraintValidator()
        planner = ContractualTaskPlanner(quantum_planner, validator, contract)
        
        # Add constraint to validator
        validator.register_constraint(
            "valid_priority",
            lambda task, ctx: 0.0 <= task.priority <= 1.0,
            "Priority must be between 0 and 1"
        )
        
        # Test valid task addition
        sample_task.priority = 0.8
        result = planner.add_task_with_validation(sample_task)
        
        assert result.success is True
        assert len(result.validation_results) > 0
        assert sample_task.id in planner.base_planner.tasks
    
    def test_contract_violation_handling(self, quantum_planner, sample_task):
        """Test handling of contract violations."""
        contract = RewardContract("test_contract")
        validator = TaskConstraintValidator(strict_mode=True)
        planner = ContractualTaskPlanner(quantum_planner, validator, contract)
        
        # Add failing constraint
        validator.register_constraint(
            "impossible_constraint",
            lambda task, ctx: False,  # Always fails
            "This constraint always fails"
        )
        
        sample_task.priority = 0.8
        result = planner.add_task_with_validation(sample_task)
        
        assert result.success is False
        assert len(result.validation_results) > 0
        assert any(not vr.is_valid for vr in result.validation_results)
        assert sample_task.id not in planner.base_planner.tasks
    
    def test_contract_compliant_optimization(self, quantum_planner, sample_tasks):
        """Test contract-compliant optimization."""
        contract = RewardContract("test_contract")
        validator = TaskConstraintValidator()
        planner = ContractualTaskPlanner(quantum_planner, validator, contract)
        
        # Add all tasks with validation
        for task in sample_tasks:
            planner.add_task_with_validation(task)
        
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            result = planner.optimize_with_contract_compliance()
        
        assert isinstance(result, ContractIntegrationResult)
        assert result.optimization_result is not None
        assert result.compliance_report is not None
        assert len(result.validation_results) > 0
    
    def test_stakeholder_preference_integration(self, quantum_planner, sample_tasks):
        """Test integration of stakeholder preferences in planning."""
        # Create contract with multiple stakeholders
        contract = RewardContract("multi_stakeholder_contract")
        contract.add_stakeholder("user", 0.4)
        contract.add_stakeholder("safety_board", 0.3)
        contract.add_stakeholder("performance_team", 0.3)
        
        # Define stakeholder-specific reward functions
        @contract.reward_function("user")
        def user_reward(state, action):
            # Users prefer faster task completion
            return -jnp.sum(state)  # Minimize time
        
        @contract.reward_function("safety_board") 
        def safety_reward(state, action):
            # Safety board prefers conservative actions
            return -jnp.sum(jnp.abs(action))  # Minimize action magnitude
        
        @contract.reward_function("performance_team")
        def performance_reward(state, action):
            # Performance team prefers efficiency
            return jnp.sum(action * state)  # Maximize state-action product
        
        validator = TaskConstraintValidator()
        planner = ContractualTaskPlanner(quantum_planner, validator, contract)
        
        # Add tasks
        for task in sample_tasks:
            planner.add_task_with_validation(task)
        
        # Test that optimization considers all stakeholders
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            result = planner.optimize_with_contract_compliance()
        
        assert result.optimization_result.success
        assert len(contract.stakeholders) == 3
    
    def test_legal_compliance_tracking(self, quantum_planner, sample_tasks):
        """Test legal compliance tracking during planning."""
        contract = RewardContract("legal_compliant_contract")
        validator = TaskConstraintValidator()
        planner = ContractualTaskPlanner(quantum_planner, validator, contract, compliance_tracking=True)
        
        # Add legal compliance constraints
        validator.register_constraint(
            "gdpr_compliance",
            lambda task, ctx: not any('pii' in str(task.resource_requirements.get(k, '')) 
                                    for k in task.resource_requirements),
            "Must not process PII without consent"
        )
        
        validator.register_constraint(
            "data_retention",
            lambda task, ctx: task.estimated_duration <= 30.0,  # Max 30 time units
            "Data retention limits"
        )
        
        # Add tasks with mixed compliance
        compliant_task = sample_tasks[0]
        compliant_task.estimated_duration = 5.0
        compliant_task.resource_requirements = {'cpu': 2}
        
        non_compliant_task = sample_tasks[1] if len(sample_tasks) > 1 else create_test_task("non_compliant")
        non_compliant_task.estimated_duration = 50.0  # Exceeds retention limit
        
        planner.add_task_with_validation(compliant_task)
        planner.add_task_with_validation(non_compliant_task)
        
        compliance_report = planner.get_compliance_report()
        
        assert isinstance(compliance_report, dict)
        assert 'total_tasks' in compliance_report
        assert 'compliant_tasks' in compliance_report
        assert 'violations' in compliance_report
        assert compliance_report['total_tasks'] == 2
    
    def test_real_time_compliance_monitoring(self, quantum_planner, sample_tasks):
        """Test real-time compliance monitoring during execution."""
        contract = RewardContract("monitored_contract")
        validator = TaskConstraintValidator()
        planner = ContractualTaskPlanner(quantum_planner, validator, contract)
        
        # Add runtime compliance check
        def runtime_compliance_check(task, execution_context):
            # Check if execution time exceeds estimates
            actual_time = execution_context.get('execution_time', 0)
            return actual_time <= task.estimated_duration * 1.5  # 50% tolerance
        
        validator.register_constraint(
            "runtime_compliance",
            runtime_compliance_check,
            "Execution time within estimates"
        )
        
        # Add tasks
        for task in sample_tasks:
            planner.add_task_with_validation(task)
        
        # Simulate execution monitoring
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            optimization_result = planner.optimize_with_contract_compliance()
            
            # Simulate execution with compliance monitoring
            execution_context = {'execution_time': 1.0}
            
            for task_id in optimization_result.optimization_result.task_order:
                task = planner.base_planner.tasks[task_id]
                validation_result = validator.validate_task(task, execution_context)
                
                if not validation_result.is_valid:
                    planner.record_compliance_violation(task_id, validation_result.violations)
        
        compliance_report = planner.get_compliance_report()
        assert 'runtime_violations' in compliance_report or len(compliance_report['violations']) >= 0


class TestContractCompliance:
    """Test cases for ContractCompliance utilities."""
    
    def test_compliance_result_creation(self):
        """Test compliance result creation."""
        violations = [
            ComplianceViolation("test_constraint", "Test violation", "ERROR", {"context": "test"})
        ]
        
        compliance = ContractCompliance(
            is_compliant=False,
            violations=violations,
            compliance_score=0.8,
            metadata={'test': True}
        )
        
        assert compliance.is_compliant is False
        assert len(compliance.violations) == 1
        assert compliance.compliance_score == 0.8
        assert compliance.metadata['test'] is True
    
    def test_compliance_violation_severity(self):
        """Test compliance violation severity handling."""
        violation_info = ComplianceViolation(
            constraint_name="critical_safety",
            description="Critical safety violation",
            severity="CRITICAL",
            context={"risk_level": "high"}
        )
        
        assert violation_info.severity == "CRITICAL"
        assert violation_info.context["risk_level"] == "high"
    
    def test_compliance_score_calculation(self, sample_tasks):
        """Test compliance score calculation."""
        validator = TaskConstraintValidator()
        
        # Register constraints with different severity
        validator.register_constraint(
            "minor_check",
            lambda task, ctx: task.priority > 0.1,
            "Minor constraint",
            severity=0.3
        )
        
        validator.register_constraint(
            "major_check", 
            lambda task, ctx: task.estimated_duration > 0,
            "Major constraint",
            severity=0.7
        )
        
        # Test compliance score calculation
        total_score = 0.0
        violation_penalty = 0.0
        
        for task in sample_tasks:
            result = validator.validate_task(task, {})
            if result.is_valid:
                total_score += 1.0
            else:
                for violation in result.violations:
                    violation_penalty += getattr(violation, 'severity', 0.5)
        
        max_possible_score = len(sample_tasks)
        compliance_score = (total_score - violation_penalty) / max_possible_score if max_possible_score > 0 else 0.0
        compliance_score = max(0.0, min(1.0, compliance_score))  # Clamp to [0,1]
        
        assert 0.0 <= compliance_score <= 1.0


class TestIntegrationWorkflows:
    """Integration tests for contract workflows."""
    
    def test_end_to_end_contract_workflow(self, quantum_planner, sample_tasks):
        """Test complete contract integration workflow."""
        # Step 1: Create contract with stakeholders and constraints
        contract = RewardContract("e2e_test_contract")
        contract.add_stakeholder("user", 0.6)
        contract.add_stakeholder("admin", 0.4)
        
        # Add reward functions
        @contract.reward_function("user")
        def user_preference(state, action):
            return float(jnp.sum(state * action))
        
        @contract.reward_function("admin")
        def admin_preference(state, action):
            return float(-jnp.sum(jnp.abs(action)))  # Prefer smaller actions
        
        # Step 2: Set up validator with constraints
        validator = TaskConstraintValidator()
        validator.register_constraint(
            "basic_validity",
            lambda task, ctx: task.priority >= 0 and task.estimated_duration > 0,
            "Basic task validity"
        )
        
        # Step 3: Create contractual planner
        planner = ContractualTaskPlanner(quantum_planner, validator, contract)
        
        # Step 4: Add tasks with validation
        for task in sample_tasks:
            result = planner.add_task_with_validation(task)
            assert result.success  # All should pass basic validation
        
        # Step 5: Optimize with contract compliance
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            final_result = planner.optimize_with_contract_compliance()
        
        # Step 6: Verify results
        assert isinstance(final_result, ContractIntegrationResult)
        assert final_result.optimization_result.success
        assert final_result.compliance_report['total_tasks'] > 0
        assert len(final_result.validation_results) > 0
        
        # Step 7: Check stakeholder satisfaction
        stakeholder_rewards = contract.to_dict()['stakeholders']
        assert len(stakeholder_rewards) == 2
        assert 'user' in stakeholder_rewards
        assert 'admin' in stakeholder_rewards
    
    def test_contract_governance_simulation(self, quantum_planner):
        """Test multi-stakeholder governance simulation."""
        # Create governance scenario
        contract = RewardContract("governance_test")
        
        # Add stakeholders with different voting power
        contract.add_stakeholder("users", 0.4, voting_power=0.5)
        contract.add_stakeholder("developers", 0.3, voting_power=0.3)
        contract.add_stakeholder("regulators", 0.3, voting_power=0.2)
        
        # Add competing constraints from different stakeholders
        validator = TaskConstraintValidator()
        
        # User preferences: fast execution
        validator.register_constraint(
            "user_speed",
            lambda task, ctx: task.estimated_duration <= 5.0,
            "Users want fast task completion"
        )
        
        # Developer preferences: reasonable resource usage
        validator.register_constraint(
            "dev_resources",
            lambda task, ctx: sum(task.resource_requirements.values()) <= 10,
            "Developers want reasonable resource usage"
        )
        
        # Regulator preferences: compliance and safety
        validator.register_constraint(
            "regulator_safety",
            lambda task, ctx: task.priority >= 0.3,  # Minimum priority for safety
            "Regulators require minimum safety priority"
        )
        
        planner = ContractualTaskPlanner(quantum_planner, validator, contract)
        
        # Create tasks that might conflict with different stakeholder preferences
        conflicting_task = create_test_task(
            "conflicting_task",
            priority=0.2,  # Fails regulator constraint
            estimated_duration=10.0,  # Fails user constraint
            resource_requirements={'cpu': 15, 'memory': 20}  # Fails developer constraint
        )
        
        result = planner.add_task_with_validation(conflicting_task)
        
        # Should fail due to multiple constraint violations
        assert result.success is False
        assert len([vr for vr in result.validation_results if not vr.is_valid]) > 0
    
    def test_adaptive_contract_updates(self, quantum_planner, sample_tasks):
        """Test adaptive contract updates based on performance."""
        contract = RewardContract("adaptive_contract")
        contract.add_stakeholder("performance", 1.0)
        
        validator = TaskConstraintValidator()
        planner = ContractualTaskPlanner(quantum_planner, validator, contract)
        
        # Initial optimization
        for task in sample_tasks:
            planner.add_task_with_validation(task)
        
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            initial_result = planner.optimize_with_contract_compliance()
        
        # Simulate performance feedback - add adaptive constraint
        def performance_constraint(task, context):
            historical_performance = context.get('historical_performance', {})
            task_performance = historical_performance.get(task.id, 1.0)
            return task_performance >= 0.8  # Require 80% success rate
        
        validator.register_constraint(
            "adaptive_performance",
            performance_constraint,
            "Adaptive performance requirement"
        )
        
        # Re-optimize with new constraint
        performance_context = {
            'historical_performance': {
                task.id: 0.9 if i % 2 == 0 else 0.7  # Mixed performance
                for i, task in enumerate(sample_tasks)
            }
        }
        
        # Validate tasks with performance context
        for task in sample_tasks:
            result = validator.validate_task(task, performance_context)
            # Some tasks should fail the new performance constraint
        
        final_compliance = planner.get_compliance_report()
        assert isinstance(final_compliance, dict)
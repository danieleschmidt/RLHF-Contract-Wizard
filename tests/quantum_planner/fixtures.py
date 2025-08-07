"""
Test fixtures for quantum task planning tests.

Provides reusable test data, mock objects, and configuration
for comprehensive test coverage.
"""

import pytest
import time
from typing import Dict, List, Set
from unittest.mock import Mock, MagicMock
import jax.numpy as jnp

from src.quantum_planner.core import QuantumTask, TaskState, QuantumTaskPlanner, PlannerConfig
from src.quantum_planner.contracts import ContractualTaskPlanner, TaskPlanningContext
from src.quantum_planner.security import SecurityContext, SecurityLevel, ThreatLevel
from src.models.reward_contract import RewardContract, AggregationStrategy
from src.models.legal_blocks import LegalBlocks


@pytest.fixture
def sample_task():
    """Create a sample QuantumTask for testing."""
    return QuantumTask(
        id="test_task_001",
        name="Sample Test Task",
        description="A task for testing quantum planning functionality",
        priority=0.7,
        estimated_duration=2.5,
        resource_requirements={"cpu": 2, "memory": 4, "storage": 10}
    )


@pytest.fixture
def sample_tasks():
    """Create a list of sample tasks with dependencies."""
    tasks = [
        QuantumTask(
            id="task_a",
            name="Task A - Data Processing", 
            description="Process input data",
            priority=0.8,
            estimated_duration=1.5,
            resource_requirements={"cpu": 2, "memory": 8}
        ),
        QuantumTask(
            id="task_b",
            name="Task B - Feature Engineering",
            description="Extract features from processed data", 
            priority=0.7,
            estimated_duration=2.0,
            resource_requirements={"cpu": 1, "memory": 4},
            dependencies={"task_a"}
        ),
        QuantumTask(
            id="task_c", 
            name="Task C - Model Training",
            description="Train ML model on engineered features",
            priority=0.9,
            estimated_duration=5.0,
            resource_requirements={"cpu": 4, "memory": 16, "gpu": 1},
            dependencies={"task_b"}
        ),
        QuantumTask(
            id="task_d",
            name="Task D - Model Validation",
            description="Validate trained model",
            priority=0.6,
            estimated_duration=1.0,
            resource_requirements={"cpu": 1, "memory": 2},
            dependencies={"task_c"}
        ),
        QuantumTask(
            id="task_e",
            name="Task E - Parallel Processing",
            description="Independent parallel task",
            priority=0.5,
            estimated_duration=3.0,
            resource_requirements={"cpu": 2, "memory": 8}
        )
    ]
    
    # Set up additional entanglements
    tasks[0].entangle_with("task_b")
    tasks[1].entangle_with("task_c") 
    tasks[2].entangle_with("task_d")
    
    return tasks


@pytest.fixture
def planner_config():
    """Create a test planner configuration."""
    return PlannerConfig(
        max_iterations=50,
        convergence_threshold=1e-3,
        quantum_interference_strength=0.1,
        entanglement_decay=0.95,
        superposition_collapse_threshold=0.8,
        resource_optimization_weight=0.3,
        time_optimization_weight=0.4,
        priority_weight=0.3,
        enable_quantum_speedup=True,
        parallel_execution_limit=3
    )


@pytest.fixture
def quantum_planner(planner_config):
    """Create a quantum task planner instance."""
    planner = QuantumTaskPlanner(planner_config)
    
    # Add sample resources
    planner.add_resource("cpu", 10)
    planner.add_resource("memory", 32)
    planner.add_resource("storage", 100)
    planner.add_resource("gpu", 2)
    
    return planner


@pytest.fixture  
def sample_contract():
    """Create a sample RLHF reward contract."""
    stakeholders = {
        "safety_team": 0.4,
        "product_team": 0.3,
        "legal_team": 0.2,
        "users": 0.1
    }
    
    contract = RewardContract(
        name="TestContract-v1.0",
        version="1.0.0",
        stakeholders=stakeholders,
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        jurisdiction="Test Jurisdiction"
    )
    
    # Add sample constraints
    @LegalBlocks.specification("""
        REQUIRES: safety_score >= 0.8
        ENSURES: NOT harmful_output(action)
        INVARIANT: compliance_monitoring_active
    """)
    def safety_constraint(state: jnp.ndarray, action: jnp.ndarray) -> bool:
        """Test safety constraint."""
        return jnp.mean(state) > 0.5 if len(state) > 0 else True
    
    contract.add_constraint(
        "test_safety_constraint",
        safety_constraint,
        description="Test safety constraint",
        severity=1.0,
        violation_penalty=-5.0
    )
    
    return contract


@pytest.fixture
def security_context():
    """Create a sample security context."""
    return SecurityContext(
        user_id="test_user_123",
        session_id="test_session_456", 
        access_level=SecurityLevel.INTERNAL,
        permissions={"model_training", "data_access", "gpu_access"},
        ip_address="192.168.1.100",
        user_agent="TestAgent/1.0",
        authentication_method="test_auth",
        mfa_verified=True
    )


@pytest.fixture
def planning_context():
    """Create a sample task planning context."""
    return TaskPlanningContext(
        user_id="test_user_123",
        session_id="test_session_456",
        resource_constraints={"max_cpu": 10, "max_memory": 32},
        time_constraints={"max_total_time": 3600},
        quality_requirements={"min_success_rate": 0.95},
        compliance_metadata={
            "monitoring_enabled": True,
            "rollback_plan": True,
            "user_demographics": {"age_group": "adult"},
            "audit_required": True
        }
    )


@pytest.fixture
def contractual_planner(sample_contract, planner_config):
    """Create a contractual task planner instance."""
    planner = ContractualTaskPlanner(sample_contract, planner_config)
    
    # Add sample resources
    planner.quantum_planner.add_resource("cpu", 10)
    planner.quantum_planner.add_resource("memory", 32)
    planner.quantum_planner.add_resource("storage", 100)
    planner.quantum_planner.add_resource("gpu", 2)
    
    return planner


@pytest.fixture
def mock_external_service():
    """Create a mock external service for testing integrations."""
    service = Mock()
    service.get_resource_availability.return_value = {"cpu": 100, "memory": 256, "gpu": 8}
    service.validate_security_context.return_value = {"valid": True, "permissions": ["all"]}
    service.log_audit_event.return_value = {"logged": True, "event_id": "test_event_123"}
    return service


@pytest.fixture
def performance_test_tasks():
    """Create a large set of tasks for performance testing."""
    tasks = []
    
    for i in range(100):  # 100 tasks for performance testing
        task = QuantumTask(
            id=f"perf_task_{i:03d}",
            name=f"Performance Test Task {i}",
            description=f"Task {i} for performance testing", 
            priority=0.1 + (i % 10) * 0.1,  # Varied priorities
            estimated_duration=0.5 + (i % 5) * 0.5,  # Varied durations
            resource_requirements={
                "cpu": 1 + (i % 4),
                "memory": 2 + (i % 8) * 2,
                "storage": 5 + (i % 3) * 5
            }
        )
        
        # Add some dependencies for complexity
        if i > 0 and i % 5 == 0:  # Every 5th task depends on previous
            task.dependencies.add(f"perf_task_{i-1:03d}")
        
        if i > 10 and i % 10 == 0:  # Every 10th task depends on task 10 back
            task.dependencies.add(f"perf_task_{i-10:03d}")
        
        # Add some entanglements
        if i > 0 and i % 3 == 0:  # Every 3rd task entangled with previous
            task.entangle_with(f"perf_task_{i-1:03d}")
        
        tasks.append(task)
    
    return tasks


@pytest.fixture
def mock_jax_functions():
    """Create mock JAX functions for testing without JAX dependency."""
    import unittest.mock
    
    # Mock JAX numpy functions
    mock_jnp = MagicMock()
    mock_jnp.array.side_effect = lambda x: x  # Pass through
    mock_jnp.mean.side_effect = lambda x: sum(x) / len(x) if x else 0
    mock_jnp.sum.side_effect = lambda x: sum(x) if hasattr(x, '__iter__') else x
    mock_jnp.exp.side_effect = lambda x: 2.718 ** x if isinstance(x, (int, float)) else [2.718 ** i for i in x]
    mock_jnp.sqrt.side_effect = lambda x: x ** 0.5
    mock_jnp.abs.side_effect = lambda x: abs(x) if isinstance(x, (int, float)) else [abs(i) for i in x]
    mock_jnp.conj.side_effect = lambda x: x  # Simplified
    mock_jnp.outer.side_effect = lambda x, y: [[i*j for j in y] for i in x]
    mock_jnp.diag.side_effect = lambda x: [x[i][i] for i in range(len(x))]
    mock_jnp.isfinite.side_effect = lambda x: not (x == float('inf') or x == float('-inf') or x != x)  # NaN check
    mock_jnp.size.side_effect = lambda x: len(x) if hasattr(x, '__len__') else 1
    
    with unittest.mock.patch('jax.numpy', mock_jnp):
        with unittest.mock.patch('jax.jit', lambda f: f):  # Pass-through for jit
            with unittest.mock.patch('jax.vmap', lambda f: f):  # Pass-through for vmap
                yield mock_jnp


@pytest.fixture
def error_inducing_task():
    """Create a task designed to trigger various error conditions."""
    return QuantumTask(
        id="error_task",
        name="Error Inducing Task",
        description="A task designed to test error handling",
        priority=1.5,  # Invalid priority > 1.0
        estimated_duration=-1.0,  # Invalid negative duration
        resource_requirements={"invalid_resource": 999999}  # Excessive resource requirement
    )


@pytest.fixture
def malicious_task():
    """Create a task with potentially malicious content for security testing."""
    return QuantumTask(
        id="<script>alert('xss')</script>",  # Malicious ID
        name="Malicious Task with <script>",
        description="This task contains suspicious keywords: hack, exploit, inject, bypass security",
        priority=0.5,
        estimated_duration=1.0,
        resource_requirements={"cpu": 999999}  # Excessive resource request
    )


@pytest.fixture
def contract_violation_task():
    """Create a task that violates contract constraints."""
    return QuantumTask(
        id="violation_task", 
        name="Contract Violation Task",
        description="This task is designed to violate safety constraints",
        priority=0.1,  # Low priority but requires high resources
        estimated_duration=100.0,  # Very long duration
        resource_requirements={
            "cpu": 50,    # Excessive CPU
            "memory": 100, # Excessive memory
            "gpu": 10     # Excessive GPU
        }
    )


@pytest.fixture(scope="session")
def test_database():
    """Create a test database connection for integration tests."""
    # Mock database connection
    db = Mock()
    db.execute.return_value = Mock()
    db.fetchall.return_value = []
    db.fetchone.return_value = None
    db.commit.return_value = None
    db.rollback.return_value = None
    db.close.return_value = None
    return db


@pytest.fixture
def mock_blockchain_service():
    """Create a mock blockchain service for testing contract deployments."""
    service = Mock()
    service.deploy_contract.return_value = {
        "transaction_hash": "0xtest123",
        "contract_address": "0xcontract456",
        "status": "success"
    }
    service.verify_contract.return_value = {"verified": True, "compliance_score": 0.95}
    service.get_contract_events.return_value = []
    return service


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file for testing."""
    log_file = tmp_path / "test_quantum_planner.log"
    return str(log_file)


@pytest.fixture
def mock_monitoring_system():
    """Create a mock monitoring system for testing."""
    monitoring = Mock()
    monitoring.metrics_collector = Mock()
    monitoring.health_checker = Mock()
    monitoring.alert_manager = Mock()
    
    monitoring.metrics_collector.record_metric.return_value = None
    monitoring.metrics_collector.get_metrics.return_value = []
    monitoring.metrics_collector.get_aggregates.return_value = {}
    
    monitoring.health_checker.run_all_checks.return_value = {
        "overall_status": "healthy",
        "checks": {}
    }
    
    monitoring.alert_manager.check_all_alerts.return_value = None
    monitoring.alert_manager.get_active_alerts.return_value = []
    
    return monitoring


# Property-based testing fixtures
@pytest.fixture
def property_test_config():
    """Configuration for property-based tests."""
    return {
        "max_examples": 100,
        "deadline": None,
        "suppress_health_check": [],
        "phases": None
    }


# Performance testing fixtures
@pytest.fixture
def performance_thresholds():
    """Performance test thresholds."""
    return {
        "max_task_creation_time": 0.01,      # 10ms
        "max_planning_time_per_task": 0.1,    # 100ms per task
        "max_execution_time_per_task": 0.05,  # 50ms per task
        "min_memory_efficiency": 0.8,         # 80% efficient memory usage
        "max_memory_usage_mb": 500,           # 500MB max memory usage
        "min_cache_hit_rate": 0.7             # 70% cache hit rate
    }


# Integration test fixtures
@pytest.fixture
def integration_test_environment():
    """Set up integration test environment."""
    return {
        "api_base_url": "http://localhost:8000/test",
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/15",  # Test DB
        "log_level": "DEBUG",
        "enable_monitoring": False,  # Disable for tests
        "enable_caching": True
    }
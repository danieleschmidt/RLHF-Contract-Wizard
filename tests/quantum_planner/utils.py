"""
Test utilities for quantum task planning tests.

Provides helper functions, assertions, and utilities
for comprehensive test coverage.
"""

import time
import functools
import contextlib
from typing import List, Dict, Any, Optional, Callable
import json
import tempfile
import os


def assert_task_valid(task):
    """Assert that a task meets basic validity requirements."""
    assert task.id is not None and isinstance(task.id, str)
    assert task.name is not None and isinstance(task.name, str)
    assert task.description is not None and isinstance(task.description, str)
    assert isinstance(task.priority, (int, float))
    assert 0.0 <= task.priority <= 1.0
    assert isinstance(task.estimated_duration, (int, float))
    assert task.estimated_duration > 0
    assert isinstance(task.resource_requirements, dict)
    assert isinstance(task.dependencies, set)
    assert isinstance(task.entangled_tasks, set)


def assert_planner_state_valid(planner):
    """Assert that a planner is in a valid state."""
    assert isinstance(planner.tasks, dict)
    assert isinstance(planner.resource_pool, dict)
    assert isinstance(planner.execution_history, list)
    assert isinstance(planner.entanglement_matrix, dict)
    assert all(isinstance(tid, str) for tid in planner.tasks.keys())
    assert all(amount >= 0 for amount in planner.resource_pool.values())


def assert_optimization_result_valid(result):
    """Assert that an optimization result is valid."""
    assert isinstance(result, dict)
    assert 'task_order' in result
    assert 'fitness_score' in result
    assert 'optimization_time' in result
    assert 'iterations' in result
    assert 'converged' in result
    
    assert isinstance(result['task_order'], list)
    assert isinstance(result['fitness_score'], (int, float))
    assert isinstance(result['optimization_time'], (int, float))
    assert isinstance(result['iterations'], int)
    assert isinstance(result['converged'], bool)
    
    assert result['fitness_score'] >= 0
    assert result['optimization_time'] >= 0
    assert result['iterations'] >= 0


def assert_execution_result_valid(result):
    """Assert that an execution result is valid."""
    assert isinstance(result, dict)
    assert 'execution_log' in result
    assert 'completed_tasks' in result
    assert 'failed_tasks' in result
    assert 'total_execution_time' in result
    assert 'success_rate' in result
    
    assert isinstance(result['execution_log'], list)
    assert isinstance(result['completed_tasks'], list)
    assert isinstance(result['failed_tasks'], list)
    assert isinstance(result['total_execution_time'], (int, float))
    assert isinstance(result['success_rate'], (int, float))
    
    assert 0.0 <= result['success_rate'] <= 1.0
    assert result['total_execution_time'] >= 0


def assert_validation_result_valid(result):
    """Assert that a validation result is valid."""
    from src.quantum_planner.validation import ValidationResult, ValidationSeverity
    
    assert isinstance(result, ValidationResult)
    assert isinstance(result.valid, bool)
    assert isinstance(result.severity, ValidationSeverity)
    assert isinstance(result.errors, list)
    assert isinstance(result.warnings, list)
    assert isinstance(result.info, list)
    assert isinstance(result.validation_time, (int, float))
    assert isinstance(result.metadata, dict)


def create_test_task(task_id: str = None, **kwargs) -> 'QuantumTask':
    """Create a test task with default or specified parameters."""
    from src.quantum_planner.core import QuantumTask
    
    defaults = {
        'id': task_id or f'test_task_{int(time.time() * 1000000)}',
        'name': 'Test Task',
        'description': 'A task created for testing',
        'priority': 0.5,
        'estimated_duration': 1.0,
        'resource_requirements': {'cpu': 1, 'memory': 2}
    }
    
    defaults.update(kwargs)
    return QuantumTask(**defaults)


def create_test_tasks_with_dependencies(count: int = 5) -> List['QuantumTask']:
    """Create a chain of tasks with dependencies."""
    tasks = []
    
    for i in range(count):
        task = create_test_task(
            task_id=f'chain_task_{i}',
            name=f'Chain Task {i}',
            priority=0.5 + (i * 0.1),
            estimated_duration=1.0 + (i * 0.5)
        )
        
        # Add dependency on previous task
        if i > 0:
            task.dependencies.add(f'chain_task_{i-1}')
        
        tasks.append(task)
    
    return tasks


def create_parallel_test_tasks(count: int = 5) -> List['QuantumTask']:
    """Create parallel tasks with no dependencies."""
    tasks = []
    
    for i in range(count):
        task = create_test_task(
            task_id=f'parallel_task_{i}',
            name=f'Parallel Task {i}',
            priority=0.2 + (i * 0.15),
            estimated_duration=0.5 + (i * 0.3)
        )
        tasks.append(task)
    
    return tasks


def measure_execution_time(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Add timing info to result if it's a dict
        if isinstance(result, dict):
            result['_execution_time'] = execution_time
        
        return result
    
    return wrapper


def assert_performance_acceptable(
    execution_time: float,
    max_time: float,
    operation_name: str = "operation"
):
    """Assert that execution time is within acceptable limits."""
    assert execution_time <= max_time, (
        f"{operation_name} took {execution_time:.3f}s, "
        f"which exceeds maximum allowed time of {max_time:.3f}s"
    )


def assert_memory_usage_acceptable(max_memory_mb: float = 500):
    """Assert that current memory usage is within acceptable limits."""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        assert memory_mb <= max_memory_mb, (
            f"Memory usage {memory_mb:.1f}MB exceeds maximum allowed {max_memory_mb:.1f}MB"
        )
        
        return memory_mb
    except ImportError:
        # psutil not available, skip check
        return 0.0


def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 10.0,
    interval: float = 0.1,
    error_message: str = "Condition not met within timeout"
):
    """Wait for a condition to become true within timeout."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition():
            return True
        time.sleep(interval)
    
    raise TimeoutError(error_message)


@contextlib.contextmanager
def temporary_file(content: str = "", suffix: str = ".tmp"):
    """Context manager for creating temporary files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    try:
        yield temp_path
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


@contextlib.contextmanager 
def capture_logs(logger_name: str = None, level: str = "DEBUG"):
    """Context manager to capture log messages."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    original_level = logger.level
    logger.setLevel(getattr(logging, level))
    logger.addHandler(handler)
    
    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def compare_dicts_approximately(
    dict1: Dict[str, Any], 
    dict2: Dict[str, Any], 
    tolerance: float = 1e-6
) -> bool:
    """Compare dictionaries with floating point tolerance."""
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    
    for key in dict1.keys():
        val1, val2 = dict1[key], dict2[key]
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > tolerance:
                return False
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not compare_dicts_approximately(val1, val2, tolerance):
                return False
        elif val1 != val2:
            return False
    
    return True


def generate_random_task_set(
    count: int = 10,
    seed: int = None,
    dependency_probability: float = 0.3
) -> List['QuantumTask']:
    """Generate a random set of tasks for testing."""
    import random
    
    if seed is not None:
        random.seed(seed)
    
    tasks = []
    
    for i in range(count):
        task = create_test_task(
            task_id=f'random_task_{i}',
            name=f'Random Task {i}',
            priority=random.uniform(0.1, 1.0),
            estimated_duration=random.uniform(0.5, 10.0),
            resource_requirements={
                'cpu': random.randint(1, 8),
                'memory': random.randint(1, 16),
                'storage': random.randint(1, 100)
            }
        )
        
        # Add random dependencies
        if i > 0:
            num_deps = random.randint(0, min(3, i))  # Max 3 dependencies
            available_deps = [f'random_task_{j}' for j in range(i)]
            dependencies = random.sample(available_deps, num_deps)
            task.dependencies.update(dependencies)
        
        tasks.append(task)
    
    return tasks


def verify_task_dependencies_satisfied(
    tasks: List['QuantumTask'], 
    execution_order: List[str]
) -> bool:
    """Verify that task execution order satisfies all dependencies."""
    completed = set()
    
    for task_id in execution_order:
        # Find the task
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            continue
        
        # Check if all dependencies are satisfied
        if not task.dependencies.issubset(completed):
            return False
        
        completed.add(task_id)
    
    return True


def calculate_resource_utilization(
    tasks: List['QuantumTask'],
    execution_order: List[str],
    available_resources: Dict[str, float]
) -> Dict[str, float]:
    """Calculate resource utilization for a task execution order."""
    max_usage = {resource: 0.0 for resource in available_resources.keys()}
    current_usage = {resource: 0.0 for resource in available_resources.keys()}
    
    for task_id in execution_order:
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            continue
        
        # Add task resource requirements
        for resource, amount in task.resource_requirements.items():
            if resource in current_usage:
                current_usage[resource] += amount
                max_usage[resource] = max(max_usage[resource], current_usage[resource])
        
        # Simulate task completion (resources released)
        # In reality, this would depend on task duration and overlap
        for resource, amount in task.resource_requirements.items():
            if resource in current_usage:
                current_usage[resource] = max(0, current_usage[resource] - amount)
    
    # Calculate utilization percentages
    utilization = {}
    for resource, available in available_resources.items():
        if available > 0:
            utilization[resource] = max_usage[resource] / available
        else:
            utilization[resource] = 0.0
    
    return utilization


def mock_quantum_computation(
    amplitudes: List[float],
    phases: List[float]
) -> List[float]:
    """Mock quantum computation for testing without JAX dependency."""
    import math
    
    # Simple interference simulation
    n = len(amplitudes)
    probabilities = []
    
    for i in range(n):
        # Simple probability calculation from amplitude
        prob = amplitudes[i] ** 2
        
        # Add phase-based interference effect
        interference = 1.0
        for j in range(n):
            if i != j:
                phase_diff = abs(phases[i] - phases[j])
                interference *= (1.0 + 0.1 * math.cos(phase_diff))
        
        prob *= interference
        probabilities.append(prob)
    
    # Normalize probabilities
    total_prob = sum(probabilities)
    if total_prob > 0:
        probabilities = [p / total_prob for p in probabilities]
    
    return probabilities


def create_stress_test_scenario(
    num_tasks: int = 1000,
    max_dependencies: int = 10,
    resource_types: List[str] = None
) -> List['QuantumTask']:
    """Create a large, complex task scenario for stress testing."""
    import random
    
    if resource_types is None:
        resource_types = ['cpu', 'memory', 'storage', 'gpu', 'network']
    
    tasks = []
    
    for i in range(num_tasks):
        # Create resource requirements
        num_resources = random.randint(1, len(resource_types))
        selected_resources = random.sample(resource_types, num_resources)
        resource_requirements = {
            res: random.randint(1, 20) for res in selected_resources
        }
        
        task = create_test_task(
            task_id=f'stress_task_{i:04d}',
            name=f'Stress Test Task {i}',
            priority=random.random(),
            estimated_duration=random.uniform(0.1, 30.0),
            resource_requirements=resource_requirements
        )
        
        # Add complex dependencies
        if i > 0:
            num_deps = random.randint(0, min(max_dependencies, i, 20))
            if num_deps > 0:
                available_tasks = list(range(max(0, i - 50), i))  # Limit dependency range
                dependencies = random.sample(available_tasks, num_deps)
                task.dependencies.update(f'stress_task_{dep:04d}' for dep in dependencies)
        
        # Add some entanglements
        if i > 10 and random.random() < 0.1:  # 10% chance
            entangle_with = f'stress_task_{random.randint(max(0, i-50), i-1):04d}'
            task.entangle_with(entangle_with)
        
        tasks.append(task)
    
    return tasks


def assert_contract_compliance(
    planning_result: Dict[str, Any],
    min_compliance_score: float = 0.8
):
    """Assert that planning result meets contract compliance requirements."""
    assert 'compliance_score' in planning_result
    assert 'validation_results' in planning_result
    
    compliance_score = planning_result['compliance_score']
    assert isinstance(compliance_score, (int, float))
    assert compliance_score >= min_compliance_score, (
        f"Compliance score {compliance_score:.3f} below minimum {min_compliance_score:.3f}"
    )
    
    validation_results = planning_result['validation_results']
    assert validation_results['valid'] == True, (
        f"Validation failed with violations: {validation_results.get('violations', [])}"
    )


def compare_performance_metrics(
    metrics1: Dict[str, Any],
    metrics2: Dict[str, Any],
    tolerance: float = 0.1
) -> Dict[str, str]:
    """Compare two sets of performance metrics and return differences."""
    differences = {}
    
    for metric_name in set(metrics1.keys()) | set(metrics2.keys()):
        val1 = metrics1.get(metric_name, 0)
        val2 = metrics2.get(metric_name, 0)
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if val1 != 0:
                relative_diff = abs(val1 - val2) / abs(val1)
                if relative_diff > tolerance:
                    if val1 > val2:
                        differences[metric_name] = f"decreased by {relative_diff:.2%}"
                    else:
                        differences[metric_name] = f"increased by {relative_diff:.2%}"
    
    return differences


def save_test_artifacts(
    test_name: str,
    artifacts: Dict[str, Any],
    output_dir: str = "/tmp/quantum_planner_test_artifacts"
):
    """Save test artifacts for debugging and analysis."""
    import os
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_file = os.path.join(output_dir, f"{test_name}_{timestamp}.json")
    
    try:
        with open(artifact_file, 'w') as f:
            json.dump(artifacts, f, indent=2, default=str)
        
        print(f"Test artifacts saved to: {artifact_file}")
        return artifact_file
    except Exception as e:
        print(f"Failed to save test artifacts: {e}")
        return None


def create_test_contract():
    """Create a test contract for testing."""
    # Mock contract object since the actual contract classes 
    # are more complex than needed for quality gate testing
    class MockContract:
        def __init__(self):
            self.stakeholders = {
                "user": {"weight": 0.4, "voting_power": 0.3},
                "system": {"weight": 0.3, "voting_power": 0.4}, 
                "auditor": {"weight": 0.3, "voting_power": 0.3}
            }
            self.constraints = {
                "max_execution_time": {"severity": "high", "enabled": True},
                "resource_limits": {"severity": "medium", "enabled": True}
            }
            self.metadata = {
                "name": "test_contract",
                "version": "1.0.0",
                "description": "Test contract for quality gates"
            }
    
    return MockContract()


def create_test_planning_context():
    """Create a test planning context."""
    from src.quantum_planner.contracts import TaskPlanningContext
    
    return TaskPlanningContext(
        user_id="test_user",
        session_id="test_session",
        resource_constraints={"cpu": 10, "memory": 32, "storage": 100},
        time_constraints={"deadline": time.time() + 3600},
        quality_requirements={"min_accuracy": 0.9},
        compliance_metadata={"requirements": ["data_privacy", "audit_trail"]}
    )
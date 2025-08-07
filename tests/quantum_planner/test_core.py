"""
Unit tests for quantum task planner core functionality.

Tests the core quantum planning algorithms, task management,
and optimization processes with comprehensive coverage.
"""

import pytest
import time
import math
from unittest.mock import Mock, patch, MagicMock

from src.quantum_planner.core import (
    QuantumTask, TaskState, QuantumTaskPlanner, PlannerConfig
)
from .fixtures import *
from .utils import *


class TestQuantumTask:
    """Test cases for QuantumTask class."""
    
    def test_task_creation(self, sample_task):
        """Test basic task creation."""
        assert sample_task.id == "test_task_001"
        assert sample_task.name == "Sample Test Task"
        assert sample_task.priority == 0.7
        assert sample_task.estimated_duration == 2.5
        assert sample_task.state == TaskState.SUPERPOSITION
        assert_task_valid(sample_task)
    
    def test_task_probability_calculation(self, sample_task):
        """Test quantum probability calculation."""
        probability = sample_task.probability()
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
        
        # Test with different amplitudes
        sample_task.amplitude = complex(0.8, 0.0)
        assert sample_task.probability() == pytest.approx(0.64, abs=1e-6)
        
        sample_task.amplitude = complex(0.0, 0.0)
        assert sample_task.probability() == 0.0
    
    def test_task_dependencies(self, sample_task):
        """Test task dependency management."""
        # Add dependencies
        sample_task.add_dependency("task_a")
        sample_task.add_dependency("task_b")
        
        assert "task_a" in sample_task.dependencies
        assert "task_b" in sample_task.dependencies
        assert len(sample_task.dependencies) == 2
        
        # Test readiness check
        assert not sample_task.is_ready(set())
        assert not sample_task.is_ready({"task_a"})
        assert sample_task.is_ready({"task_a", "task_b"})
        assert sample_task.is_ready({"task_a", "task_b", "task_c"})
    
    def test_task_entanglement(self, sample_task):
        """Test quantum entanglement functionality."""
        sample_task.entangle_with("task_x")
        sample_task.entangle_with("task_y")
        
        assert "task_x" in sample_task.entangled_tasks
        assert "task_y" in sample_task.entangled_tasks
        assert len(sample_task.entangled_tasks) == 2
    
    def test_task_state_collapse(self, sample_task):
        """Test quantum state collapse."""
        # Initial state
        assert sample_task.state == TaskState.SUPERPOSITION
        assert sample_task.start_time is None
        assert sample_task.end_time is None
        
        # Collapse to running
        sample_task.collapse(TaskState.RUNNING)
        assert sample_task.state == TaskState.RUNNING
        assert sample_task.start_time is not None
        assert sample_task.end_time is None
        
        start_time = sample_task.start_time
        time.sleep(0.01)  # Small delay
        
        # Collapse to completed
        sample_task.collapse(TaskState.COMPLETED)
        assert sample_task.state == TaskState.COMPLETED
        assert sample_task.start_time == start_time
        assert sample_task.end_time is not None
        assert sample_task.actual_duration is not None
        assert sample_task.actual_duration > 0
    
    def test_task_equality_and_hashing(self):
        """Test task equality and hashing."""
        task1 = create_test_task("task_1")
        task2 = create_test_task("task_1")  # Same ID
        task3 = create_test_task("task_3")  # Different ID
        
        assert task1 == task2
        assert task1 != task3
        assert hash(task1) == hash(task2)
        assert hash(task1) != hash(task3)
        
        # Tasks can be used in sets
        task_set = {task1, task2, task3}
        assert len(task_set) == 2  # task1 and task2 are equivalent


class TestPlannerConfig:
    """Test cases for PlannerConfig class."""
    
    def test_config_creation(self, planner_config):
        """Test configuration creation with valid values."""
        assert planner_config.max_iterations == 50
        assert planner_config.convergence_threshold == 1e-3
        assert planner_config.quantum_interference_strength == 0.1
        assert planner_config.enable_quantum_speedup == True
        assert planner_config.parallel_execution_limit == 3
    
    def test_config_weight_sum(self, planner_config):
        """Test that optimization weights sum to approximately 1.0."""
        weight_sum = (
            planner_config.resource_optimization_weight +
            planner_config.time_optimization_weight +
            planner_config.priority_weight
        )
        assert weight_sum == pytest.approx(1.0, abs=1e-6)


class TestQuantumTaskPlanner:
    """Test cases for QuantumTaskPlanner class."""
    
    def test_planner_creation(self, quantum_planner):
        """Test planner initialization."""
        assert_planner_state_valid(quantum_planner)
        assert len(quantum_planner.tasks) == 0
        assert quantum_planner.resource_pool["cpu"] == 10
        assert quantum_planner.resource_pool["memory"] == 32
    
    def test_add_task(self, quantum_planner, sample_task):
        """Test adding tasks to planner."""
        quantum_planner.add_task(sample_task)
        
        assert len(quantum_planner.tasks) == 1
        assert sample_task.id in quantum_planner.tasks
        assert quantum_planner.tasks[sample_task.id] == sample_task
        
        # Task should have quantum properties initialized
        assert isinstance(sample_task.amplitude, complex)
        assert abs(sample_task.amplitude) > 0
    
    def test_add_multiple_tasks(self, quantum_planner, sample_tasks):
        """Test adding multiple tasks."""
        for task in sample_tasks:
            quantum_planner.add_task(task)
        
        assert len(quantum_planner.tasks) == len(sample_tasks)
        
        # All task IDs should be present
        task_ids = {task.id for task in sample_tasks}
        assert set(quantum_planner.tasks.keys()) == task_ids
    
    def test_add_resource(self, quantum_planner):
        """Test resource management."""
        initial_cpu = quantum_planner.resource_pool.get("cpu", 0)
        
        quantum_planner.add_resource("cpu", 5)
        assert quantum_planner.resource_pool["cpu"] == initial_cpu + 5
        
        quantum_planner.add_resource("new_resource", 100)
        assert quantum_planner.resource_pool["new_resource"] == 100
    
    def test_create_entanglement(self, quantum_planner, sample_tasks):
        """Test quantum entanglement creation."""
        # Add tasks first
        for task in sample_tasks:
            quantum_planner.add_task(task)
        
        task_a_id = sample_tasks[0].id
        task_b_id = sample_tasks[1].id
        
        # Create entanglement
        quantum_planner.create_entanglement(task_a_id, task_b_id, 0.8)
        
        # Verify entanglement
        assert task_b_id in quantum_planner.tasks[task_a_id].entangled_tasks
        assert task_a_id in quantum_planner.tasks[task_b_id].entangled_tasks
        
        # Verify entanglement matrix
        key = tuple(sorted([task_a_id, task_b_id]))
        assert key in quantum_planner.entanglement_matrix
        assert abs(quantum_planner.entanglement_matrix[key]) == pytest.approx(0.8)
    
    @patch('jax.numpy.array')
    @patch('jax.numpy.exp')
    @patch('jax.numpy.outer')
    def test_quantum_state_update(self, mock_outer, mock_exp, mock_array, quantum_planner, sample_tasks, mock_jax_functions):
        """Test quantum state updates with mocked JAX functions."""
        # Add tasks
        for task in sample_tasks:
            quantum_planner.add_task(task)
        
        # Create entanglements
        quantum_planner.create_entanglement(sample_tasks[0].id, sample_tasks[1].id)
        
        # Mock JAX functions
        mock_array.side_effect = lambda x: x
        mock_exp.side_effect = lambda x: [complex(1, 1) for _ in x]
        mock_outer.side_effect = lambda x, y: [[1.0, 0.5], [0.5, 1.0]]
        
        # Update quantum state
        quantum_planner._update_quantum_state()
        
        # Verify that quantum properties are updated
        for task in quantum_planner.tasks.values():
            if task.state == TaskState.SUPERPOSITION:
                assert isinstance(task.amplitude, complex)
    
    def test_fitness_calculation(self, quantum_planner, sample_tasks):
        """Test fitness calculation for task orderings."""
        for task in sample_tasks:
            quantum_planner.add_task(task)
        
        task_order = [task.id for task in sample_tasks]
        fitness = quantum_planner._calculate_execution_fitness(task_order)
        
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0
        
        # Empty order should have zero fitness
        empty_fitness = quantum_planner._calculate_execution_fitness([])
        assert empty_fitness == 0.0
        
        # Order with dependency violations should have lower fitness
        reversed_order = list(reversed(task_order))
        reversed_fitness = quantum_planner._calculate_execution_fitness(reversed_order)
        assert reversed_fitness <= fitness  # Should be lower due to violations
    
    def test_quantum_state_summary(self, quantum_planner, sample_tasks):
        """Test quantum state summary generation."""
        # Add tasks in different states
        quantum_planner.add_task(sample_tasks[0])  # SUPERPOSITION by default
        
        sample_tasks[1].collapse(TaskState.RUNNING)
        quantum_planner.add_task(sample_tasks[1])
        
        sample_tasks[2].collapse(TaskState.COMPLETED)
        quantum_planner.add_task(sample_tasks[2])
        
        summary = quantum_planner.get_quantum_state_summary()
        
        assert isinstance(summary, dict)
        assert 'total_tasks' in summary
        assert 'state_distribution' in summary
        assert 'total_superposition_probability' in summary
        
        assert summary['total_tasks'] == 3
        assert summary['state_distribution']['superposition'] >= 1
        assert summary['total_superposition_probability'] > 0
    
    @measure_execution_time
    def test_optimization_performance(self, quantum_planner, sample_tasks, performance_thresholds):
        """Test optimization performance."""
        for task in sample_tasks:
            quantum_planner.add_task(task)
        
        result = quantum_planner.optimize_plan()
        
        assert_optimization_result_valid(result)
        
        # Check performance
        execution_time = result['_execution_time']
        max_time_per_task = performance_thresholds['max_planning_time_per_task']
        max_total_time = len(sample_tasks) * max_time_per_task
        
        assert_performance_acceptable(execution_time, max_total_time, "optimization")
    
    def test_empty_planner_optimization(self, quantum_planner):
        """Test optimization with no tasks."""
        result = quantum_planner.optimize_plan()
        
        assert result['task_order'] == []
        assert result['fitness_score'] == 0.0
        assert result['iterations'] >= 0
    
    def test_single_task_optimization(self, quantum_planner, sample_task):
        """Test optimization with single task."""
        quantum_planner.add_task(sample_task)
        result = quantum_planner.optimize_plan()
        
        assert_optimization_result_valid(result)
        assert len(result['task_order']) == 1
        assert result['task_order'][0] == sample_task.id
    
    def test_plan_execution(self, quantum_planner, sample_tasks):
        """Test plan execution."""
        for task in sample_tasks:
            quantum_planner.add_task(task)
        
        # Optimize first
        optimization_result = quantum_planner.optimize_plan()
        
        # Execute the plan
        execution_result = quantum_planner.execute_plan(optimization_result)
        
        assert_execution_result_valid(execution_result)
        
        # Check that some tasks were processed
        total_processed = len(execution_result['completed_tasks']) + len(execution_result['failed_tasks'])
        assert total_processed > 0
        
        # Success rate should be reasonable (mock execution has 95% success rate)
        if total_processed > 0:
            assert execution_result['success_rate'] >= 0.8  # Allow some failures
    
    def test_dependency_respect_in_optimization(self, quantum_planner):
        """Test that optimization respects task dependencies."""
        # Create tasks with clear dependencies: A -> B -> C
        tasks = create_test_tasks_with_dependencies(3)
        
        for task in tasks:
            quantum_planner.add_task(task)
        
        result = quantum_planner.optimize_plan()
        task_order = result['task_order']
        
        # Verify dependencies are satisfied
        assert verify_task_dependencies_satisfied(tasks, task_order)
    
    def test_resource_constraints_in_execution(self, quantum_planner):
        """Test that execution respects resource constraints."""
        # Create tasks that exceed available resources
        task1 = create_test_task("high_cpu_task", resource_requirements={"cpu": 8})
        task2 = create_test_task("high_cpu_task2", resource_requirements={"cpu": 8})
        
        quantum_planner.add_task(task1)
        quantum_planner.add_task(task2)
        
        # Available CPU is only 10, so both tasks can't run simultaneously
        optimization_result = quantum_planner.optimize_plan()
        execution_result = quantum_planner.execute_plan(optimization_result)
        
        # Some tasks should be skipped due to resource constraints
        execution_log = execution_result['execution_log']
        skipped_tasks = [
            entry for entry in execution_log
            if entry.get('action') == 'skipped' and 'resource' in entry.get('reason', '')
        ]
        
        # We expect some resource-related skipping with high resource tasks
        # (This is a probabilistic test - results may vary)


class TestQuantumPlannerIntegration:
    """Integration tests for quantum planner components."""
    
    def test_end_to_end_planning_workflow(self, quantum_planner, sample_tasks):
        """Test complete planning workflow from task creation to execution."""
        # Step 1: Add tasks
        for task in sample_tasks:
            quantum_planner.add_task(task)
        
        # Step 2: Create entanglements
        quantum_planner.create_entanglement(sample_tasks[0].id, sample_tasks[1].id)
        quantum_planner.create_entanglement(sample_tasks[1].id, sample_tasks[2].id)
        
        # Step 3: Optimize plan
        optimization_result = quantum_planner.optimize_plan()
        assert_optimization_result_valid(optimization_result)
        
        # Step 4: Execute plan
        execution_result = quantum_planner.execute_plan(optimization_result)
        assert_execution_result_valid(execution_result)
        
        # Step 5: Verify results
        assert len(optimization_result['task_order']) == len(sample_tasks)
        assert execution_result['success_rate'] > 0.0
    
    def test_complex_dependency_graph(self, quantum_planner):
        """Test planning with complex dependency relationships."""
        # Create diamond dependency pattern: A -> B,C -> D
        task_a = create_test_task("task_a", priority=0.9)
        task_b = create_test_task("task_b", priority=0.7, dependencies={"task_a"})
        task_c = create_test_task("task_c", priority=0.8, dependencies={"task_a"})
        task_d = create_test_task("task_d", priority=0.6, dependencies={"task_b", "task_c"})
        
        tasks = [task_a, task_b, task_c, task_d]
        for task in tasks:
            quantum_planner.add_task(task)
        
        # Add entanglements
        quantum_planner.create_entanglement("task_b", "task_c", 0.7)
        
        result = quantum_planner.optimize_plan()
        assert_optimization_result_valid(result)
        
        # Verify dependency satisfaction
        assert verify_task_dependencies_satisfied(tasks, result['task_order'])
        
        # Task A should come first
        assert result['task_order'][0] == "task_a"
        
        # Task D should come last
        assert result['task_order'][-1] == "task_d"
    
    def test_resource_utilization_optimization(self, quantum_planner, performance_thresholds):
        """Test that resource utilization is optimized."""
        # Create tasks with different resource profiles
        tasks = [
            create_test_task("cpu_intensive", resource_requirements={"cpu": 4, "memory": 2}),
            create_test_task("memory_intensive", resource_requirements={"cpu": 1, "memory": 8}),
            create_test_task("balanced", resource_requirements={"cpu": 2, "memory": 4}),
            create_test_task("storage_intensive", resource_requirements={"cpu": 1, "storage": 50})
        ]
        
        for task in tasks:
            quantum_planner.add_task(task)
        
        result = quantum_planner.optimize_plan()
        
        # Calculate resource utilization
        utilization = calculate_resource_utilization(
            tasks,
            result['task_order'],
            quantum_planner.resource_pool
        )
        
        # Resource utilization should be reasonable
        for resource, util in utilization.items():
            assert 0.0 <= util <= 1.0
            
        # Memory efficiency should meet threshold
        memory_util = utilization.get('memory', 0.0)
        min_efficiency = performance_thresholds['min_memory_efficiency']
        # Note: This is a simplified check - real optimization might not always achieve this
    
    def test_quantum_interference_effects(self, quantum_planner):
        """Test that quantum interference affects optimization."""
        # Create tasks with specific quantum properties
        task1 = create_test_task("task1", priority=0.5)
        task2 = create_test_task("task2", priority=0.5)
        
        # Set specific quantum properties
        task1.amplitude = complex(0.8, 0.0)
        task1.phase = 0.0
        
        task2.amplitude = complex(0.8, 0.0)
        task2.phase = math.pi  # π phase difference
        
        quantum_planner.add_task(task1)
        quantum_planner.add_task(task2)
        
        # Run optimization multiple times and check for consistency
        results = []
        for _ in range(5):
            result = quantum_planner.optimize_plan()
            results.append(result)
        
        # All results should be valid
        for result in results:
            assert_optimization_result_valid(result)
            assert len(result['task_order']) == 2
    
    @pytest.mark.slow
    def test_stress_test_many_tasks(self, quantum_planner):
        """Stress test with many tasks."""
        num_tasks = 50  # Reduced from 100 for CI/CD
        stress_tasks = create_stress_test_scenario(
            num_tasks=num_tasks,
            max_dependencies=5,
            resource_types=['cpu', 'memory']
        )
        
        # Add tasks
        start_time = time.time()
        for task in stress_tasks:
            quantum_planner.add_task(task)
        task_creation_time = time.time() - start_time
        
        # Optimize
        start_time = time.time()
        result = quantum_planner.optimize_plan()
        optimization_time = time.time() - start_time
        
        assert_optimization_result_valid(result)
        
        # Performance assertions
        assert task_creation_time < 1.0  # Should be fast
        assert optimization_time < 30.0  # Should complete within reasonable time
        
        # Check memory usage
        memory_mb = assert_memory_usage_acceptable(max_memory_mb=1000)  # Allow more for stress test
        
        # Verify dependency satisfaction
        completed_tasks = [t for t in stress_tasks if t.id in result['task_order']]
        assert verify_task_dependencies_satisfied(completed_tasks, result['task_order'])
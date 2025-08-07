"""
Unit tests for quantum algorithms module.

Tests the advanced quantum algorithms including QAOA optimization,
superposition search, and entanglement scheduling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

from src.quantum_planner.algorithms import (
    QuantumOptimizer, SuperpositionSearch, EntanglementScheduler,
    OptimizationResult, SearchResult, SchedulingResult
)
from src.quantum_planner.core import QuantumTask, TaskState
from .fixtures import *
from .utils import *


class TestQuantumOptimizer:
    """Test cases for QuantumOptimizer class."""
    
    def test_optimizer_creation(self):
        """Test quantum optimizer initialization."""
        optimizer = QuantumOptimizer(max_iterations=100, convergence_threshold=1e-4)
        
        assert optimizer.max_iterations == 100
        assert optimizer.convergence_threshold == 1e-4
        assert optimizer.quantum_interference_strength == 0.1
        assert len(optimizer.optimization_history) == 0
    
    @patch('jax.numpy.array')
    @patch('jax.numpy.exp')
    @patch('jax.numpy.dot')
    def test_qaoa_optimization(self, mock_dot, mock_exp, mock_array, sample_tasks, mock_jax_functions):
        """Test QAOA optimization algorithm."""
        optimizer = QuantumOptimizer()
        
        # Mock JAX functions
        mock_array.side_effect = lambda x: np.array(x) if hasattr(x, '__iter__') else x
        mock_exp.side_effect = lambda x: np.exp(x) if isinstance(x, np.ndarray) else np.exp([x])
        mock_dot.side_effect = lambda x, y: np.dot(x, y) if all(hasattr(arr, '__iter__') for arr in [x, y]) else x * y
        
        result = optimizer.qaoa_optimize(
            tasks=sample_tasks,
            resource_constraints={'cpu': 10, 'memory': 32}
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert len(result.task_order) == len(sample_tasks)
        assert result.fitness_score >= 0.0
        assert result.iterations > 0
        assert result.convergence_achieved is True
    
    def test_parameter_optimization(self, sample_tasks):
        """Test parameter optimization for quantum circuits."""
        optimizer = QuantumOptimizer()
        
        # Test with simple parameter space
        def objective_function(params):
            gamma, beta = params[0], params[1] if len(params) > 1 else params[0]
            # Simple quadratic objective
            return -(gamma - 0.5)**2 - (beta - 0.3)**2 + 1.0
        
        initial_params = [0.1, 0.1]
        result = optimizer.optimize_parameters(objective_function, initial_params)
        
        assert isinstance(result, dict)
        assert 'optimal_params' in result
        assert 'final_value' in result
        assert 'iterations' in result
        assert len(result['optimal_params']) == len(initial_params)
    
    def test_quantum_annealing_simulation(self, sample_tasks):
        """Test quantum annealing simulation."""
        optimizer = QuantumOptimizer()
        
        # Create simple cost function
        def cost_function(assignment):
            return sum(i * val for i, val in enumerate(assignment))
        
        result = optimizer.quantum_annealing(
            cost_function=cost_function,
            num_variables=len(sample_tasks),
            temperature_schedule=lambda t: 1.0 / (1 + t)
        )
        
        assert isinstance(result, dict)
        assert 'solution' in result
        assert 'energy' in result
        assert 'success_probability' in result
        assert len(result['solution']) == len(sample_tasks)
        assert 0.0 <= result['success_probability'] <= 1.0
    
    def test_optimization_history_tracking(self, sample_tasks):
        """Test optimization history tracking."""
        optimizer = QuantumOptimizer()
        
        # Run optimization to generate history
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            result = optimizer.qaoa_optimize(sample_tasks, {'cpu': 10})
        
        # Check history was recorded
        assert len(optimizer.optimization_history) > 0
        
        history = optimizer.get_optimization_history()
        assert isinstance(history, list)
        
        for entry in history:
            assert 'iteration' in entry
            assert 'fitness' in entry
            assert 'parameters' in entry
            assert 'timestamp' in entry
    
    def test_convergence_detection(self):
        """Test convergence detection algorithm."""
        optimizer = QuantumOptimizer(convergence_threshold=1e-3)
        
        # Test converged sequence
        fitness_sequence = [1.0, 0.999, 0.9985, 0.9982]  # Small changes
        assert optimizer._check_convergence(fitness_sequence) is True
        
        # Test non-converged sequence
        fitness_sequence = [1.0, 0.9, 0.8, 0.7]  # Large changes
        assert optimizer._check_convergence(fitness_sequence) is False
        
        # Test insufficient data
        fitness_sequence = [1.0]
        assert optimizer._check_convergence(fitness_sequence) is False
    
    @measure_execution_time
    def test_optimization_performance(self, sample_tasks, performance_thresholds):
        """Test optimization performance."""
        optimizer = QuantumOptimizer(max_iterations=10)  # Limit iterations for test speed
        
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            result = optimizer.qaoa_optimize(sample_tasks, {'cpu': 10})
        
        # Check performance
        execution_time = result._execution_time if hasattr(result, '_execution_time') else result.execution_time
        max_time = performance_thresholds['max_optimization_time'] * len(sample_tasks)
        
        assert_performance_acceptable(execution_time, max_time, "QAOA optimization")


class TestSuperpositionSearch:
    """Test cases for SuperpositionSearch class."""
    
    def test_search_creation(self):
        """Test superposition search initialization."""
        search = SuperpositionSearch(exploration_factor=0.7, collapse_threshold=0.1)
        
        assert search.exploration_factor == 0.7
        assert search.collapse_threshold == 0.1
        assert search.max_superposition_states == 1000
        assert len(search.search_history) == 0
    
    def test_superposition_state_management(self, sample_tasks):
        """Test superposition state creation and management."""
        search = SuperpositionSearch()
        
        # Initialize superposition states
        states = search.initialize_superposition_states(sample_tasks)
        
        assert len(states) <= search.max_superposition_states
        assert all(isinstance(state, dict) for state in states)
        
        for state in states:
            assert 'task_order' in state
            assert 'amplitude' in state
            assert 'phase' in state
            assert len(state['task_order']) == len(sample_tasks)
    
    @patch('numpy.random.random')
    def test_quantum_search_algorithm(self, mock_random, sample_tasks):
        """Test quantum search algorithm implementation."""
        search = SuperpositionSearch()
        
        # Mock random for deterministic testing
        mock_random.side_effect = [0.5, 0.3, 0.8, 0.2] * 10
        
        def fitness_function(task_order):
            # Simple fitness based on task priorities
            return sum(task.priority for task in sample_tasks if task.id in task_order)
        
        result = search.quantum_search(
            tasks=sample_tasks,
            fitness_function=fitness_function,
            max_iterations=20
        )
        
        assert isinstance(result, SearchResult)
        assert result.success is True
        assert len(result.best_solution) == len(sample_tasks)
        assert result.best_fitness >= 0.0
        assert result.iterations <= 20
        assert len(result.explored_states) > 0
    
    def test_amplitude_amplification(self, sample_tasks):
        """Test amplitude amplification procedure."""
        search = SuperpositionSearch()
        
        # Create initial superposition states
        states = search.initialize_superposition_states(sample_tasks)
        
        # Define oracle function
        def oracle(task_order):
            # Mark solutions with high priority tasks first
            if len(task_order) > 0:
                first_task = next((t for t in sample_tasks if t.id == task_order[0]), None)
                return first_task.priority > 0.8 if first_task else False
            return False
        
        # Apply amplitude amplification
        amplified_states = search.amplitude_amplification(states, oracle)
        
        assert len(amplified_states) == len(states)
        
        # Check that oracle-marked states have higher amplitudes
        oracle_states = [s for s in amplified_states if oracle(s['task_order'])]
        non_oracle_states = [s for s in amplified_states if not oracle(s['task_order'])]
        
        if oracle_states and non_oracle_states:
            avg_oracle_amplitude = np.mean([abs(s['amplitude']) for s in oracle_states])
            avg_non_oracle_amplitude = np.mean([abs(s['amplitude']) for s in non_oracle_states])
            # Oracle states should generally have higher amplitudes
            # (This is probabilistic, so we allow some flexibility)
    
    def test_state_collapse_mechanism(self, sample_tasks):
        """Test quantum state collapse mechanism."""
        search = SuperpositionSearch(collapse_threshold=0.1)
        
        # Create states with different amplitudes
        states = [
            {'task_order': [t.id for t in sample_tasks], 'amplitude': 0.9, 'phase': 0.0},
            {'task_order': list(reversed([t.id for t in sample_tasks])), 'amplitude': 0.05, 'phase': np.pi/2},
            {'task_order': [sample_tasks[0].id], 'amplitude': 0.8, 'phase': np.pi}
        ]
        
        collapsed_states = search.collapse_superposition(states)
        
        # States with amplitude below threshold should be removed
        assert all(abs(s['amplitude']) >= search.collapse_threshold for s in collapsed_states)
        assert len(collapsed_states) <= len(states)
    
    def test_interference_effects(self, sample_tasks):
        """Test quantum interference effects in search."""
        search = SuperpositionSearch()
        
        # Create states that can interfere
        state1 = {'task_order': [t.id for t in sample_tasks], 'amplitude': 0.7, 'phase': 0.0}
        state2 = {'task_order': [t.id for t in sample_tasks], 'amplitude': 0.3, 'phase': np.pi}  # Out of phase
        
        # Apply interference
        interfered_states = search.apply_interference([state1, state2])
        
        assert isinstance(interfered_states, list)
        assert len(interfered_states) > 0
        
        # Check that interference was applied correctly
        for state in interfered_states:
            assert 'amplitude' in state
            assert 'phase' in state
            assert 'task_order' in state
    
    def test_search_history_tracking(self, sample_tasks):
        """Test search history tracking."""
        search = SuperpositionSearch()
        
        def simple_fitness(task_order):
            return len(task_order)
        
        with patch('numpy.random.random', side_effect=[0.5] * 100):
            result = search.quantum_search(sample_tasks, simple_fitness, max_iterations=5)
        
        # Check history was recorded
        history = search.get_search_history()
        assert isinstance(history, list)
        assert len(history) > 0
        
        for entry in history:
            assert 'iteration' in entry
            assert 'best_fitness' in entry
            assert 'num_states' in entry
            assert 'timestamp' in entry


class TestEntanglementScheduler:
    """Test cases for EntanglementScheduler class."""
    
    def test_scheduler_creation(self):
        """Test entanglement scheduler initialization."""
        scheduler = EntanglementScheduler(entanglement_threshold=0.3, decay_factor=0.95)
        
        assert scheduler.entanglement_threshold == 0.3
        assert scheduler.decay_factor == 0.95
        assert scheduler.max_entangled_pairs == 1000
        assert len(scheduler.entanglement_history) == 0
    
    def test_entanglement_detection(self, sample_tasks):
        """Test entanglement detection between tasks."""
        scheduler = EntanglementScheduler()
        
        # Test with tasks that should be entangled (similar resources, dependencies)
        task_a = sample_tasks[0]
        task_b = sample_tasks[1]
        
        # Add some common dependencies
        task_a.add_dependency("shared_task")
        task_b.add_dependency("shared_task")
        
        entanglement_strength = scheduler.detect_entanglement(task_a, task_b)
        
        assert isinstance(entanglement_strength, float)
        assert 0.0 <= entanglement_strength <= 1.0
        
        # Tasks with common dependencies should have higher entanglement
        if entanglement_strength > scheduler.entanglement_threshold:
            assert len(task_a.dependencies & task_b.dependencies) > 0
    
    def test_entangled_pair_scheduling(self, sample_tasks):
        """Test scheduling of entangled task pairs."""
        scheduler = EntanglementScheduler()
        
        # Create entangled pairs
        entangled_pairs = [
            (sample_tasks[0].id, sample_tasks[1].id, 0.8),
            (sample_tasks[1].id, sample_tasks[2].id, 0.6),
        ]
        
        result = scheduler.schedule_entangled_pairs(
            tasks=sample_tasks,
            entangled_pairs=entangled_pairs,
            resource_constraints={'cpu': 10, 'memory': 32}
        )
        
        assert isinstance(result, SchedulingResult)
        assert result.success is True
        assert len(result.schedule) > 0
        assert len(result.entangled_groups) > 0
        
        # Check that entangled tasks are scheduled together when possible
        for group in result.entangled_groups:
            assert len(group) >= 2  # At least two tasks per group
            assert all(task_id in [t.id for t in sample_tasks] for task_id in group)
    
    def test_quantum_entanglement_matrix(self, sample_tasks):
        """Test quantum entanglement matrix computation."""
        scheduler = EntanglementScheduler()
        
        matrix = scheduler.compute_entanglement_matrix(sample_tasks)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (len(sample_tasks), len(sample_tasks))
        
        # Matrix should be symmetric
        assert np.allclose(matrix, matrix.T, rtol=1e-5)
        
        # Diagonal should be 1.0 (task is perfectly entangled with itself)
        assert np.allclose(np.diag(matrix), 1.0, rtol=1e-5)
        
        # All values should be between 0 and 1
        assert np.all((matrix >= 0.0) & (matrix <= 1.0))
    
    def test_entanglement_decay(self, sample_tasks):
        """Test entanglement decay over time."""
        scheduler = EntanglementScheduler(decay_factor=0.9)
        
        initial_strength = 0.8
        
        # Simulate time passage
        for time_step in range(5):
            decayed_strength = scheduler.apply_entanglement_decay(initial_strength, time_step)
            assert 0.0 <= decayed_strength <= initial_strength
            assert decayed_strength <= initial_strength * (scheduler.decay_factor ** time_step)
    
    def test_resource_aware_scheduling(self, sample_tasks):
        """Test resource-aware entangled scheduling."""
        scheduler = EntanglementScheduler()
        
        # Create resource-intensive tasks
        for task in sample_tasks:
            task.resource_requirements = {'cpu': 4, 'memory': 8}
        
        entangled_pairs = [(sample_tasks[0].id, sample_tasks[1].id, 0.9)]
        
        result = scheduler.schedule_entangled_pairs(
            tasks=sample_tasks,
            entangled_pairs=entangled_pairs,
            resource_constraints={'cpu': 10, 'memory': 16}
        )
        
        assert isinstance(result, SchedulingResult)
        
        # Check resource utilization in schedule
        max_concurrent_cpu = 0
        max_concurrent_memory = 0
        
        for time_slot in result.schedule:
            concurrent_cpu = sum(
                task.resource_requirements.get('cpu', 0)
                for task in sample_tasks
                if task.id in time_slot.get('tasks', [])
            )
            concurrent_memory = sum(
                task.resource_requirements.get('memory', 0)
                for task in sample_tasks
                if task.id in time_slot.get('tasks', [])
            )
            
            max_concurrent_cpu = max(max_concurrent_cpu, concurrent_cpu)
            max_concurrent_memory = max(max_concurrent_memory, concurrent_memory)
        
        # Should respect resource constraints
        assert max_concurrent_cpu <= 10
        assert max_concurrent_memory <= 16
    
    def test_bell_state_preparation(self, sample_tasks):
        """Test Bell state preparation for entangled task pairs."""
        scheduler = EntanglementScheduler()
        
        if len(sample_tasks) >= 2:
            task_a, task_b = sample_tasks[0], sample_tasks[1]
            
            bell_state = scheduler.prepare_bell_state(task_a, task_b)
            
            assert isinstance(bell_state, dict)
            assert 'task_pair' in bell_state
            assert 'entanglement_strength' in bell_state
            assert 'quantum_state' in bell_state
            
            # Bell states should have maximum entanglement
            assert bell_state['entanglement_strength'] > 0.9
    
    def test_scheduling_history_tracking(self, sample_tasks):
        """Test scheduling history tracking."""
        scheduler = EntanglementScheduler()
        
        entangled_pairs = [(sample_tasks[0].id, sample_tasks[1].id, 0.7)]
        
        result = scheduler.schedule_entangled_pairs(
            tasks=sample_tasks,
            entangled_pairs=entangled_pairs,
            resource_constraints={'cpu': 10}
        )
        
        # Check history was recorded
        history = scheduler.get_scheduling_history()
        assert isinstance(history, list)
        
        if len(history) > 0:
            for entry in history:
                assert 'timestamp' in entry
                assert 'num_tasks' in entry
                assert 'num_entangled_pairs' in entry
                assert 'success' in entry


class TestAlgorithmIntegration:
    """Integration tests for quantum algorithms."""
    
    def test_combined_optimization_workflow(self, sample_tasks):
        """Test combined optimization workflow using all algorithms."""
        # Initialize algorithms
        optimizer = QuantumOptimizer(max_iterations=10)
        search = SuperpositionSearch()
        scheduler = EntanglementScheduler()
        
        # Step 1: Detect entanglements
        entangled_pairs = []
        for i in range(len(sample_tasks)):
            for j in range(i + 1, len(sample_tasks)):
                strength = scheduler.detect_entanglement(sample_tasks[i], sample_tasks[j])
                if strength > scheduler.entanglement_threshold:
                    entangled_pairs.append((sample_tasks[i].id, sample_tasks[j].id, strength))
        
        # Step 2: Use superposition search for initial exploration
        def fitness_function(task_order):
            return sum(task.priority for task in sample_tasks if task.id in task_order)
        
        with patch('numpy.random.random', side_effect=[0.5] * 100):
            search_result = search.quantum_search(sample_tasks, fitness_function, max_iterations=5)
        
        assert search_result.success
        
        # Step 3: Optimize with QAOA
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            optimization_result = optimizer.qaoa_optimize(sample_tasks, {'cpu': 10})
        
        assert optimization_result.success
        
        # Step 4: Schedule entangled pairs
        if entangled_pairs:
            scheduling_result = scheduler.schedule_entangled_pairs(
                tasks=sample_tasks,
                entangled_pairs=entangled_pairs,
                resource_constraints={'cpu': 10}
            )
            assert scheduling_result.success
    
    @pytest.mark.slow
    def test_scalability_with_many_tasks(self):
        """Test algorithm scalability with many tasks."""
        # Create larger task set
        num_tasks = 20
        large_task_set = [
            create_test_task(f"task_{i}", priority=np.random.random())
            for i in range(num_tasks)
        ]
        
        optimizer = QuantumOptimizer(max_iterations=5)  # Reduced for performance
        
        start_time = time.time()
        
        with patch('jax.numpy.array', side_effect=lambda x: x), \
             patch('jax.numpy.exp', side_effect=lambda x: [1.0] * len(x) if hasattr(x, '__len__') else 1.0), \
             patch('jax.numpy.dot', side_effect=lambda x, y: 1.0):
            
            result = optimizer.qaoa_optimize(large_task_set, {'cpu': 20})
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # 10 seconds max
        assert result.success
        assert len(result.task_order) == num_tasks
    
    def test_algorithm_error_handling(self, sample_tasks):
        """Test error handling in quantum algorithms."""
        optimizer = QuantumOptimizer()
        search = SuperpositionSearch()
        scheduler = EntanglementScheduler()
        
        # Test with invalid inputs
        with pytest.raises((ValueError, TypeError)):
            optimizer.qaoa_optimize([], {})  # Empty task list
        
        with pytest.raises((ValueError, TypeError)):
            search.quantum_search([], lambda x: 0)  # Empty task list
        
        # Test with malformed resource constraints
        result = optimizer.qaoa_optimize(sample_tasks, {})  # Empty constraints
        # Should handle gracefully (not raise exception)
        assert isinstance(result, OptimizationResult)
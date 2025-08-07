"""
Advanced quantum-inspired algorithms for task planning optimization.

This module implements specialized quantum algorithms including:
- Quantum Approximate Optimization Algorithm (QAOA) for combinatorial problems
- Variational Quantum Eigensolver (VQE) for resource optimization
- Quantum superposition search with amplitude amplification
"""

import math
import cmath
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

from .core import QuantumTask, TaskState, PlannerConfig


@dataclass
class QuantumCircuitParams:
    """Parameters for quantum circuit simulation."""
    depth: int = 3
    mixing_angles: List[float] = None
    cost_angles: List[float] = None
    initial_state: Optional[jnp.ndarray] = None
    
    def __post_init__(self):
        if self.mixing_angles is None:
            self.mixing_angles = [0.1] * self.depth
        if self.cost_angles is None:
            self.cost_angles = [0.1] * self.depth


class QuantumOptimizer:
    """
    Quantum-inspired optimizer using QAOA-like algorithms for task scheduling.
    
    Implements a variational quantum approach to solve combinatorial optimization
    problems in task planning, including resource allocation and scheduling.
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.optimization_history: List[Dict[str, Any]] = []
        
    @jit
    def _apply_mixing_hamiltonian(
        self, 
        state_vector: jnp.ndarray, 
        angle: float
    ) -> jnp.ndarray:
        """Apply mixing Hamiltonian (X rotations) to state vector."""
        n_qubits = int(math.log2(len(state_vector)))
        
        # Apply X rotation to each qubit
        for qubit in range(n_qubits):
            # Create X gate for this qubit
            x_matrix = self._create_x_gate(qubit, n_qubits)
            rotation_matrix = jnp.cos(angle/2) * jnp.eye(2**n_qubits) - \
                             1j * jnp.sin(angle/2) * x_matrix
            state_vector = rotation_matrix @ state_vector
            
        return state_vector
    
    @jit
    def _apply_cost_hamiltonian(
        self, 
        state_vector: jnp.ndarray, 
        angle: float,
        cost_matrix: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply cost Hamiltonian (problem-specific) to state vector."""
        # Create rotation based on cost function
        rotation_matrix = jnp.cos(angle) * jnp.eye(len(state_vector)) - \
                         1j * jnp.sin(angle) * cost_matrix
        return rotation_matrix @ state_vector
    
    def _create_x_gate(self, target_qubit: int, n_qubits: int) -> jnp.ndarray:
        """Create Pauli-X gate matrix for target qubit in n-qubit system."""
        # Pauli-X matrix
        pauli_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
        identity = jnp.eye(2, dtype=jnp.complex64)
        
        # Build tensor product
        result = jnp.array([[1]], dtype=jnp.complex64)
        
        for qubit in range(n_qubits):
            if qubit == target_qubit:
                result = jnp.kron(result, pauli_x)
            else:
                result = jnp.kron(result, identity)
                
        return result
    
    def _encode_task_problem(self, tasks: Dict[str, QuantumTask]) -> jnp.ndarray:
        """Encode task scheduling problem as quantum cost matrix."""
        task_list = list(tasks.keys())
        n_tasks = len(task_list)
        
        if n_tasks == 0:
            return jnp.zeros((1, 1))
        
        # Create cost matrix based on task priorities and dependencies
        cost_matrix = jnp.zeros((2**n_tasks, 2**n_tasks), dtype=jnp.complex64)
        
        for i in range(2**n_tasks):
            # Binary representation gives task selection
            selected_tasks = []
            binary_repr = format(i, f'0{n_tasks}b')
            
            for j, bit in enumerate(binary_repr):
                if bit == '1':
                    selected_tasks.append(task_list[j])
            
            # Calculate cost for this task selection
            cost = self._calculate_selection_cost(selected_tasks, tasks)
            cost_matrix = cost_matrix.at[i, i].set(-cost)  # Negative for maximization
            
        return cost_matrix
    
    def _calculate_selection_cost(
        self, 
        selected_tasks: List[str], 
        tasks: Dict[str, QuantumTask]
    ) -> float:
        """Calculate cost/reward for a specific task selection."""
        if not selected_tasks:
            return 0.0
        
        total_priority = sum(tasks[tid].priority for tid in selected_tasks)
        total_duration = sum(tasks[tid].estimated_duration for tid in selected_tasks)
        
        # Check dependency satisfaction
        dependency_penalty = 0.0
        completed = set()
        
        for task_id in selected_tasks:
            task = tasks[task_id]
            if not task.dependencies.issubset(completed):
                dependency_penalty += 10.0  # Heavy penalty for dependency violations
            completed.add(task_id)
        
        # Reward high priority, penalize long duration and dependency violations
        return total_priority - 0.1 * total_duration - dependency_penalty
    
    def optimize_task_selection(
        self, 
        tasks: Dict[str, QuantumTask],
        circuit_params: Optional[QuantumCircuitParams] = None
    ) -> Tuple[List[str], float, Dict[str, Any]]:
        """
        Optimize task selection using quantum-inspired QAOA algorithm.
        
        Returns:
            Tuple of (selected_tasks, optimization_score, metrics)
        """
        if not tasks:
            return [], 0.0, {}
        
        if circuit_params is None:
            circuit_params = QuantumCircuitParams()
        
        task_list = list(tasks.keys())
        n_tasks = len(task_list)
        
        # Limit problem size for computational tractability
        if n_tasks > 10:
            # Select top priority tasks
            sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].priority, reverse=True)
            selected_dict = dict(sorted_tasks[:10])
            task_list = list(selected_dict.keys())
            n_tasks = len(task_list)
            tasks = selected_dict
        
        # Encode problem as quantum cost matrix
        cost_matrix = self._encode_task_problem(tasks)
        
        # Initialize uniform superposition
        n_states = 2**n_tasks
        initial_state = jnp.ones(n_states, dtype=jnp.complex64) / jnp.sqrt(n_states)
        
        if circuit_params.initial_state is not None:
            initial_state = circuit_params.initial_state
        
        # QAOA optimization loop
        best_params = (circuit_params.mixing_angles.copy(), circuit_params.cost_angles.copy())
        best_expectation = float('-inf')
        
        optimization_steps = []
        
        for step in range(50):  # Limit optimization steps
            # Apply QAOA circuit
            state = initial_state
            
            for layer in range(circuit_params.depth):
                # Apply cost Hamiltonian
                if layer < len(circuit_params.cost_angles):
                    state = self._apply_cost_hamiltonian(
                        state, 
                        circuit_params.cost_angles[layer], 
                        cost_matrix
                    )
                
                # Apply mixing Hamiltonian
                if layer < len(circuit_params.mixing_angles):
                    state = self._apply_mixing_hamiltonian(
                        state,
                        circuit_params.mixing_angles[layer]
                    )
            
            # Calculate expectation value
            expectation = jnp.real(jnp.conj(state).T @ cost_matrix @ state)
            
            optimization_steps.append({
                'step': step,
                'expectation': float(expectation),
                'mixing_angles': circuit_params.mixing_angles.copy(),
                'cost_angles': circuit_params.cost_angles.copy()
            })
            
            if expectation > best_expectation:
                best_expectation = expectation
                best_params = (circuit_params.mixing_angles.copy(), circuit_params.cost_angles.copy())
            
            # Simple gradient-free optimization (random search with cooling)
            temperature = 1.0 * (1.0 - step / 50)
            
            # Perturb parameters
            for i in range(len(circuit_params.mixing_angles)):
                circuit_params.mixing_angles[i] += np.random.normal(0, temperature * 0.1)
            
            for i in range(len(circuit_params.cost_angles)):
                circuit_params.cost_angles[i] += np.random.normal(0, temperature * 0.1)
        
        # Get final state with best parameters
        circuit_params.mixing_angles, circuit_params.cost_angles = best_params
        final_state = initial_state
        
        for layer in range(circuit_params.depth):
            if layer < len(circuit_params.cost_angles):
                final_state = self._apply_cost_hamiltonian(
                    final_state,
                    circuit_params.cost_angles[layer],
                    cost_matrix
                )
            
            if layer < len(circuit_params.mixing_angles):
                final_state = self._apply_mixing_hamiltonian(
                    final_state,
                    circuit_params.mixing_angles[layer]
                )
        
        # Sample from final state to get task selection
        probabilities = jnp.abs(final_state) ** 2
        best_state_idx = int(jnp.argmax(probabilities))
        
        # Decode selected tasks
        binary_repr = format(best_state_idx, f'0{n_tasks}b')
        selected_tasks = [
            task_list[i] for i, bit in enumerate(binary_repr) 
            if bit == '1'
        ]
        
        metrics = {
            'optimization_steps': optimization_steps,
            'final_expectation': float(best_expectation),
            'best_state_probability': float(probabilities[best_state_idx]),
            'quantum_circuit_depth': circuit_params.depth,
            'n_tasks_considered': n_tasks
        }
        
        return selected_tasks, float(best_expectation), metrics


class SuperpositionSearch:
    """
    Quantum superposition-based search algorithm for exploring task orderings.
    
    Uses quantum amplitude amplification to enhance good solutions while
    suppressing poor ones through iterative rotation operations.
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        
    def search(
        self, 
        tasks: Dict[str, QuantumTask],
        fitness_function: Callable[[List[str]], float],
        max_iterations: int = 100
    ) -> Tuple[List[str], float, Dict[str, Any]]:
        """
        Search for optimal task ordering using quantum superposition.
        
        Args:
            tasks: Dictionary of tasks to order
            fitness_function: Function to evaluate task ordering fitness
            max_iterations: Maximum search iterations
            
        Returns:
            Tuple of (best_ordering, best_fitness, search_metrics)
        """
        if not tasks:
            return [], 0.0, {}
        
        task_ids = list(tasks.keys())
        n_tasks = len(task_ids)
        
        # Initialize superposition state (all orderings equally likely)
        state_amplitudes: Dict[str, complex] = {}
        
        # For computational tractability, sample subset of orderings
        n_samples = min(1000, math.factorial(min(n_tasks, 7)))
        
        # Generate random permutations as basis states
        orderings = []
        for _ in range(n_samples):
            ordering = task_ids.copy()
            np.random.shuffle(ordering)
            orderings.append(tuple(ordering))
        
        # Remove duplicates
        orderings = list(set(orderings))
        
        # Initialize uniform superposition
        initial_amplitude = 1.0 / math.sqrt(len(orderings))
        for ordering in orderings:
            state_amplitudes[ordering] = complex(initial_amplitude, 0.0)
        
        search_history = []
        best_ordering = list(task_ids)
        best_fitness = fitness_function(best_ordering)
        
        for iteration in range(max_iterations):
            # Evaluate fitness for all orderings in superposition
            fitness_scores = {}
            for ordering in orderings:
                fitness_scores[ordering] = fitness_function(list(ordering))
                
                if fitness_scores[ordering] > best_fitness:
                    best_fitness = fitness_scores[ordering]
                    best_ordering = list(ordering)
            
            # Apply amplitude amplification based on fitness
            max_fitness = max(fitness_scores.values()) if fitness_scores else 1.0
            min_fitness = min(fitness_scores.values()) if fitness_scores else 0.0
            fitness_range = max_fitness - min_fitness
            
            if fitness_range > 1e-10:  # Avoid division by zero
                # Amplify good solutions, suppress bad ones
                for ordering in orderings:
                    normalized_fitness = (fitness_scores[ordering] - min_fitness) / fitness_range
                    
                    # Quantum rotation based on fitness
                    rotation_angle = normalized_fitness * math.pi / 4
                    current_amplitude = state_amplitudes[ordering]
                    
                    # Apply rotation (amplitude amplification)
                    new_amplitude = (
                        current_amplitude * math.cos(rotation_angle) +
                        1j * abs(current_amplitude) * math.sin(rotation_angle)
                    )
                    
                    state_amplitudes[ordering] = new_amplitude
            
            # Normalize amplitudes to maintain probability conservation
            total_probability = sum(abs(amp)**2 for amp in state_amplitudes.values())
            if total_probability > 1e-10:
                normalization = 1.0 / math.sqrt(total_probability)
                for ordering in state_amplitudes:
                    state_amplitudes[ordering] *= normalization
            
            # Track search progress
            current_probabilities = {
                ordering: abs(amp)**2 
                for ordering, amp in state_amplitudes.items()
            }
            
            entropy = -sum(
                p * math.log(p) for p in current_probabilities.values() 
                if p > 1e-10
            )
            
            search_history.append({
                'iteration': iteration,
                'best_fitness': best_fitness,
                'entropy': entropy,
                'top_probability': max(current_probabilities.values())
            })
            
            # Early stopping if converged
            if iteration > 10 and entropy < 0.1:
                break
        
        # Final sampling based on amplitudes
        final_probabilities = {
            ordering: abs(amp)**2 
            for ordering, amp in state_amplitudes.items()
        }
        
        # Select highest probability ordering as final result
        if final_probabilities:
            best_probability_ordering = max(
                final_probabilities.keys(), 
                key=lambda x: final_probabilities[x]
            )
            
            final_fitness = fitness_function(list(best_probability_ordering))
            if final_fitness > best_fitness:
                best_ordering = list(best_probability_ordering)
                best_fitness = final_fitness
        
        metrics = {
            'search_history': search_history,
            'final_entropy': entropy,
            'orderings_explored': len(orderings),
            'convergence_iteration': len(search_history),
            'amplitude_distribution': {
                str(ordering): abs(amp)**2 
                for ordering, amp in list(state_amplitudes.items())[:10]  # Top 10
            }
        }
        
        return best_ordering, best_fitness, metrics


class EntanglementScheduler:
    """
    Quantum entanglement-based scheduler for managing task dependencies.
    
    Uses entanglement correlations to optimize scheduling of interdependent
    tasks while maintaining dependency constraints.
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.entanglement_graph: Dict[Tuple[str, str], float] = {}
        
    def create_entanglement_network(
        self, 
        tasks: Dict[str, QuantumTask]
    ) -> Dict[str, Any]:
        """Create entanglement network based on task dependencies and properties."""
        entanglement_strength = {}
        dependency_graph = {}
        
        # Build dependency graph
        for task_id, task in tasks.items():
            dependency_graph[task_id] = {
                'dependencies': list(task.dependencies),
                'dependents': list(task.dependents),
                'priority': task.priority,
                'duration': task.estimated_duration
            }
        
        # Create entanglements
        task_ids = list(tasks.keys())
        
        for i, task_id1 in enumerate(task_ids):
            for j, task_id2 in enumerate(task_ids[i+1:], i+1):
                task1 = tasks[task_id1]
                task2 = tasks[task_id2]
                
                # Calculate entanglement strength based on various factors
                strength = 0.0
                
                # Direct dependency creates strong entanglement
                if task_id1 in task2.dependencies or task_id2 in task1.dependencies:
                    strength += 0.8
                
                # Shared resources create medium entanglement
                shared_resources = set(task1.resource_requirements.keys()) & \
                                 set(task2.resource_requirements.keys())
                if shared_resources:
                    strength += 0.3 * len(shared_resources)
                
                # Similar priorities create weak entanglement
                priority_similarity = 1.0 - abs(task1.priority - task2.priority)
                strength += 0.1 * priority_similarity
                
                # Duration compatibility
                duration_ratio = min(task1.estimated_duration, task2.estimated_duration) / \
                               max(task1.estimated_duration, task2.estimated_duration)
                strength += 0.05 * duration_ratio
                
                # Normalize strength
                strength = min(1.0, strength)
                
                if strength > 0.1:  # Only create significant entanglements
                    key = tuple(sorted([task_id1, task_id2]))
                    entanglement_strength[key] = strength
                    self.entanglement_graph[key] = strength
        
        return {
            'entanglements': entanglement_strength,
            'dependency_graph': dependency_graph,
            'total_entanglements': len(entanglement_strength),
            'average_strength': np.mean(list(entanglement_strength.values())) 
                              if entanglement_strength else 0.0
        }
    
    def schedule_with_entanglement(
        self, 
        tasks: Dict[str, QuantumTask]
    ) -> Tuple[List[List[str]], Dict[str, Any]]:
        """
        Schedule tasks in parallel batches considering entanglement constraints.
        
        Returns:
            Tuple of (batch_schedule, scheduling_metrics)
        """
        if not tasks:
            return [], {}
        
        # Create entanglement network
        network_info = self.create_entanglement_network(tasks)
        
        # Initialize scheduling state
        completed = set()
        scheduled_batches = []
        remaining_tasks = set(tasks.keys())
        
        scheduling_metrics = {
            'batches_created': 0,
            'parallel_efficiency': 0.0,
            'entanglement_violations': 0,
            'dependency_satisfaction': 1.0
        }
        
        while remaining_tasks:
            current_batch = []
            batch_entanglements = []
            
            # Find tasks ready for execution (dependencies satisfied)
            ready_tasks = [
                task_id for task_id in remaining_tasks
                if tasks[task_id].is_ready(completed)
            ]
            
            if not ready_tasks:
                # Handle circular dependencies or other issues
                # Take highest priority task
                if remaining_tasks:
                    highest_priority_task = max(
                        remaining_tasks, 
                        key=lambda tid: tasks[tid].priority
                    )
                    ready_tasks = [highest_priority_task]
                    scheduling_metrics['dependency_satisfaction'] -= 0.1
            
            # Select tasks for current batch using entanglement optimization
            resource_usage = {}
            
            for task_id in ready_tasks:
                task = tasks[task_id]
                
                # Check resource conflicts with current batch
                resource_conflict = False
                for existing_task_id in current_batch:
                    existing_task = tasks[existing_task_id]
                    
                    # Check for resource conflicts
                    for resource, amount in task.resource_requirements.items():
                        existing_amount = existing_task.resource_requirements.get(resource, 0)
                        total_usage = resource_usage.get(resource, 0) + amount
                        
                        # Simple conflict detection (assumes exclusive resource usage)
                        if existing_amount > 0 and amount > 0:
                            if resource not in resource.endswith('_shareable'):
                                resource_conflict = True
                                break
                    
                    if resource_conflict:
                        break
                
                if resource_conflict:
                    continue
                
                # Check entanglement compatibility
                entanglement_compatible = True
                for existing_task_id in current_batch:
                    key = tuple(sorted([task_id, existing_task_id]))
                    entanglement = self.entanglement_graph.get(key, 0.0)
                    
                    # Strong entanglements suggest tasks should not run in parallel
                    if entanglement > 0.7:
                        entanglement_compatible = False
                        scheduling_metrics['entanglement_violations'] += 1
                        break
                
                if not entanglement_compatible:
                    continue
                
                # Add task to current batch
                current_batch.append(task_id)
                
                # Update resource usage
                for resource, amount in task.resource_requirements.items():
                    resource_usage[resource] = resource_usage.get(resource, 0) + amount
                
                # Stop if batch is full
                if len(current_batch) >= self.config.parallel_execution_limit:
                    break
            
            # Finalize current batch
            if current_batch:
                scheduled_batches.append(current_batch)
                completed.update(current_batch)
                remaining_tasks -= set(current_batch)
                scheduling_metrics['batches_created'] += 1
            else:
                # Emergency exit to prevent infinite loop
                if remaining_tasks:
                    # Force schedule one task
                    task_id = next(iter(remaining_tasks))
                    scheduled_batches.append([task_id])
                    completed.add(task_id)
                    remaining_tasks.remove(task_id)
                    scheduling_metrics['batches_created'] += 1
        
        # Calculate parallel efficiency
        total_tasks = len(tasks)
        total_batches = len(scheduled_batches)
        if total_batches > 0:
            scheduling_metrics['parallel_efficiency'] = total_tasks / total_batches
        
        scheduling_metrics['network_info'] = network_info
        
        return scheduled_batches, scheduling_metrics
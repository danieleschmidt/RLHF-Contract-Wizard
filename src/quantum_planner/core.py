"""
Core quantum-inspired task planning implementation.

This module implements the fundamental quantum-inspired algorithms for task planning,
including superposition-based state exploration, quantum interference for optimization,
and entanglement-based dependency management.
"""

import time
import random
import math
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np


class TaskState(Enum):
    """Task execution states with quantum-inspired properties."""
    SUPERPOSITION = "superposition"  # Task exists in multiple potential states
    ENTANGLED = "entangled"         # Task dependent on other tasks
    COLLAPSED = "collapsed"         # Task state determined/executed
    PENDING = "pending"            # Task waiting for dependencies
    RUNNING = "running"            # Task currently executing
    COMPLETED = "completed"        # Task successfully finished
    FAILED = "failed"              # Task execution failed
    CANCELLED = "cancelled"        # Task cancelled before execution


@dataclass
class QuantumTask:
    """
    Represents a task with quantum-inspired properties.
    
    Tasks can exist in superposition (multiple potential execution paths),
    be entangled with other tasks (shared dependencies), and undergo
    quantum interference (optimization through probability amplification).
    """
    id: str
    name: str
    description: str
    priority: float = 0.5  # 0.0 = lowest, 1.0 = highest
    estimated_duration: float = 1.0  # in arbitrary time units
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)  # Task IDs this task depends on
    dependents: Set[str] = field(default_factory=set)    # Tasks that depend on this task
    
    # Quantum-inspired properties
    amplitude: complex = complex(1.0, 0.0)  # Quantum amplitude (probability amplitude)
    phase: float = 0.0                      # Phase angle for interference
    state: TaskState = TaskState.SUPERPOSITION
    entangled_tasks: Set[str] = field(default_factory=set)
    
    # Execution metadata
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    actual_duration: Optional[float] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    
    # Contract compliance
    contract_constraints: List[str] = field(default_factory=list)
    compliance_score: float = 1.0
    
    def probability(self) -> float:
        """Calculate probability from quantum amplitude (Born rule)."""
        return abs(self.amplitude) ** 2
    
    def add_dependency(self, task_id: str) -> 'QuantumTask':
        """Add a task dependency."""
        self.dependencies.add(task_id)
        return self
    
    def add_dependent(self, task_id: str) -> 'QuantumTask':
        """Add a task that depends on this one."""
        self.dependents.add(task_id)
        return self
    
    def entangle_with(self, task_id: str) -> 'QuantumTask':
        """Create quantum entanglement with another task."""
        self.entangled_tasks.add(task_id)
        return self
    
    def collapse(self, final_state: TaskState) -> 'QuantumTask':
        """Collapse the task from superposition to a definite state."""
        self.state = final_state
        if final_state == TaskState.RUNNING:
            self.start_time = time.time()
        elif final_state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
            self.end_time = time.time()
            if self.start_time:
                self.actual_duration = self.end_time - self.start_time
        return self
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (all dependencies completed)."""
        return self.dependencies.issubset(completed_tasks)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, QuantumTask):
            return False
        return self.id == other.id


@dataclass
class PlannerConfig:
    """Configuration for the quantum task planner."""
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    quantum_interference_strength: float = 0.1
    entanglement_decay: float = 0.95
    superposition_collapse_threshold: float = 0.8
    resource_optimization_weight: float = 0.3
    time_optimization_weight: float = 0.4
    priority_weight: float = 0.3
    enable_quantum_speedup: bool = True
    parallel_execution_limit: int = 4


class QuantumTaskPlanner:
    """
    Quantum-inspired task planner using superposition, entanglement, and interference.
    
    The planner explores multiple execution paths in superposition, uses entanglement
    to model task dependencies, and applies quantum interference to amplify optimal
    solutions while suppressing suboptimal ones.
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.tasks: Dict[str, QuantumTask] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.resource_pool: Dict[str, float] = {}
        self.current_time: float = 0.0
        
        # Quantum state tracking
        self.global_phase: float = 0.0
        self.entanglement_matrix: Dict[Tuple[str, str], complex] = {}
        self.interference_patterns: Dict[str, List[float]] = defaultdict(list)
    
    def add_task(self, task: QuantumTask) -> 'QuantumTaskPlanner':
        """Add a task to the planner."""
        self.tasks[task.id] = task
        
        # Initialize quantum properties
        if task.state == TaskState.SUPERPOSITION:
            # Normalize amplitude based on priority
            task.amplitude = complex(
                math.sqrt(task.priority + 0.1),  # Ensure non-zero amplitude
                0.0
            )
            task.phase = random.uniform(0, 2 * math.pi)
        
        return self
    
    def add_resource(self, resource_name: str, amount: float) -> 'QuantumTaskPlanner':
        """Add available resources."""
        self.resource_pool[resource_name] = amount
        return self
    
    def create_entanglement(self, task_id1: str, task_id2: str, strength: float = 1.0):
        """Create quantum entanglement between two tasks."""
        if task_id1 in self.tasks and task_id2 in self.tasks:
            self.tasks[task_id1].entangle_with(task_id2)
            self.tasks[task_id2].entangle_with(task_id1)
            
            # Store entanglement strength in matrix
            key = tuple(sorted([task_id1, task_id2]))
            self.entanglement_matrix[key] = complex(strength, 0.0)
    
    def _compute_interference_pattern(
        self,
        amplitudes: jnp.ndarray,
        phases: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute quantum interference pattern for amplitude optimization."""
        complex_amplitudes = amplitudes * jnp.exp(1j * phases)
        
        # Apply interference through amplitude combination
        interference_matrix = jnp.outer(complex_amplitudes, jnp.conj(complex_amplitudes))
        
        # Extract probability enhancements
        probabilities = jnp.abs(jnp.diag(interference_matrix))
        
        return probabilities / jnp.sum(probabilities)
    
    def _update_quantum_state(self):
        """Update quantum states of all tasks based on interference and entanglement."""
        task_ids = list(self.tasks.keys())
        n_tasks = len(task_ids)
        
        if n_tasks == 0:
            return
        
        # Extract current amplitudes and phases
        amplitudes = jnp.array([abs(self.tasks[tid].amplitude) for tid in task_ids])
        phases = jnp.array([self.tasks[tid].phase for tid in task_ids])
        
        # Apply quantum interference
        new_probabilities = self._compute_interference_pattern(amplitudes, phases)
        
        # Update task amplitudes based on interference
        for i, task_id in enumerate(task_ids):
            task = self.tasks[task_id]
            if task.state == TaskState.SUPERPOSITION:
                # Update amplitude while preserving phase relationships
                new_amplitude = math.sqrt(new_probabilities[i])
                task.amplitude = complex(
                    new_amplitude * math.cos(task.phase),
                    new_amplitude * math.sin(task.phase)
                )
                
                # Apply entanglement effects
                self._apply_entanglement_effects(task_id)
    
    def _apply_entanglement_effects(self, task_id: str):
        """Apply entanglement effects to a task's quantum state."""
        task = self.tasks[task_id]
        
        for entangled_id in task.entangled_tasks:
            if entangled_id in self.tasks:
                entangled_task = self.tasks[entangled_id]
                
                # Get entanglement strength
                key = tuple(sorted([task_id, entangled_id]))
                strength = self.entanglement_matrix.get(key, complex(0.5, 0.0))
                
                # Apply entanglement correlation
                if entangled_task.state == TaskState.SUPERPOSITION:
                    # Correlate phases
                    phase_diff = abs(task.phase - entangled_task.phase)
                    correlation = abs(strength) * math.cos(phase_diff)
                    
                    # Adjust amplitudes based on correlation
                    current_amplitude = abs(task.amplitude)
                    adjustment = self.config.quantum_interference_strength * correlation
                    
                    new_amplitude = min(1.0, current_amplitude + adjustment)
                    task.amplitude = complex(new_amplitude, task.amplitude.imag)
    
    def _calculate_execution_fitness(self, task_order: List[str]) -> float:
        """Calculate fitness score for a given task execution order."""
        if not task_order:
            return 0.0
        
        total_time = 0.0
        total_priority_weighted_delay = 0.0
        resource_violations = 0
        dependency_violations = 0
        
        completed = set()
        current_resources = dict(self.resource_pool)
        
        for i, task_id in enumerate(task_order):
            if task_id not in self.tasks:
                continue
                
            task = self.tasks[task_id]
            
            # Check dependency violations
            if not task.is_ready(completed):
                dependency_violations += 1
                continue
            
            # Check resource availability
            resources_ok = all(
                current_resources.get(res, 0) >= amount
                for res, amount in task.resource_requirements.items()
            )
            
            if not resources_ok:
                resource_violations += 1
                continue
            
            # Consume resources
            for res, amount in task.resource_requirements.items():
                current_resources[res] -= amount
            
            # Update timing and priority metrics
            delay = total_time * task.priority
            total_priority_weighted_delay += delay
            total_time += task.estimated_duration
            completed.add(task_id)
        
        # Calculate fitness (higher is better)
        fitness = (
            len(completed) / len(task_order) * 0.4 +  # Completion ratio
            (1.0 / (1.0 + total_time)) * self.config.time_optimization_weight +
            (1.0 / (1.0 + total_priority_weighted_delay)) * self.config.priority_weight +
            (1.0 / (1.0 + resource_violations + dependency_violations)) * 0.2
        )
        
        return max(0.0, min(1.0, fitness))
    
    def _generate_quantum_solution(self) -> List[str]:
        """Generate a solution using quantum-inspired superposition search."""
        superposition_tasks = [
            tid for tid, task in self.tasks.items()
            if task.state == TaskState.SUPERPOSITION
        ]
        
        if not superposition_tasks:
            return []
        
        # Create superposition of all possible orderings
        n_tasks = len(superposition_tasks)
        n_samples = min(50, math.factorial(min(n_tasks, 7)))  # Limit sampling for performance
        
        best_order = []
        best_fitness = -1.0
        
        # Sample from superposition space
        for _ in range(n_samples):
            # Create random ordering weighted by task probabilities
            order = superposition_tasks.copy()
            weights = [self.tasks[tid].probability() for tid in order]
            
            # Quantum-inspired weighted shuffle
            for i in range(len(order) - 1):
                # Select next task based on quantum probabilities
                remaining_weights = weights[i:]
                if sum(remaining_weights) > 0:
                    probs = [w / sum(remaining_weights) for w in remaining_weights]
                    choice_idx = np.random.choice(len(remaining_weights), p=probs)
                    
                    # Swap selected task to current position
                    actual_idx = i + choice_idx
                    order[i], order[actual_idx] = order[actual_idx], order[i]
                    weights[i], weights[actual_idx] = weights[actual_idx], weights[i]
            
            # Evaluate fitness
            fitness = self._calculate_execution_fitness(order)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_order = order.copy()
        
        return best_order
    
    def optimize_plan(self) -> Dict[str, Any]:
        """
        Optimize task execution plan using quantum-inspired algorithms.
        
        Returns:
            Dictionary containing the optimized plan and metadata
        """
        start_time = time.time()
        iteration_count = 0
        previous_fitness = 0.0
        
        optimization_history = []
        
        for iteration in range(self.config.max_iterations):
            iteration_count = iteration + 1
            
            # Update quantum states
            self._update_quantum_state()
            
            # Generate solution from current superposition
            proposed_order = self._generate_quantum_solution()
            current_fitness = self._calculate_execution_fitness(proposed_order)
            
            optimization_history.append({
                'iteration': iteration,
                'fitness': current_fitness,
                'order_length': len(proposed_order)
            })
            
            # Apply quantum interference to enhance good solutions
            if current_fitness > previous_fitness:
                # Amplify amplitudes of tasks in good solutions
                for task_id in proposed_order:
                    if task_id in self.tasks:
                        task = self.tasks[task_id]
                        if task.state == TaskState.SUPERPOSITION:
                            enhancement = 1.0 + self.config.quantum_interference_strength
                            current_amp = abs(task.amplitude)
                            new_amp = min(1.0, current_amp * enhancement)
                            task.amplitude = complex(new_amp, task.amplitude.imag)
            
            # Check convergence
            if abs(current_fitness - previous_fitness) < self.config.convergence_threshold:
                break
            
            previous_fitness = current_fitness
            
            # Apply entanglement decay
            for key in self.entanglement_matrix:
                self.entanglement_matrix[key] *= self.config.entanglement_decay
        
        optimization_time = time.time() - start_time
        
        # Final solution
        final_order = self._generate_quantum_solution()
        final_fitness = self._calculate_execution_fitness(final_order)
        
        return {
            'task_order': final_order,
            'fitness_score': final_fitness,
            'optimization_time': optimization_time,
            'iterations': iteration_count,
            'converged': iteration_count < self.config.max_iterations,
            'optimization_history': optimization_history,
            'quantum_metrics': {
                'entanglements': len(self.entanglement_matrix),
                'superposition_tasks': len([
                    t for t in self.tasks.values()
                    if t.state == TaskState.SUPERPOSITION
                ]),
                'average_probability': np.mean([
                    t.probability() for t in self.tasks.values()
                    if t.state == TaskState.SUPERPOSITION
                ]) if self.tasks else 0.0
            }
        }
    
    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the optimized task plan."""
        task_order = plan['task_order']
        execution_log = []
        completed_tasks = set()
        failed_tasks = set()
        
        start_time = time.time()
        current_resources = dict(self.resource_pool)
        
        for task_id in task_order:
            if task_id not in self.tasks:
                continue
                
            task = self.tasks[task_id]
            
            # Check if task is ready
            if not task.is_ready(completed_tasks):
                execution_log.append({
                    'task_id': task_id,
                    'action': 'skipped',
                    'reason': 'dependencies not met',
                    'time': time.time() - start_time
                })
                continue
            
            # Check resource availability
            resources_ok = all(
                current_resources.get(res, 0) >= amount
                for res, amount in task.resource_requirements.items()
            )
            
            if not resources_ok:
                execution_log.append({
                    'task_id': task_id,
                    'action': 'skipped',
                    'reason': 'insufficient resources',
                    'time': time.time() - start_time
                })
                continue
            
            # Collapse task from superposition to running state
            task.collapse(TaskState.RUNNING)
            
            # Consume resources
            for res, amount in task.resource_requirements.items():
                current_resources[res] -= amount
            
            execution_log.append({
                'task_id': task_id,
                'action': 'started',
                'time': time.time() - start_time,
                'resources_consumed': dict(task.resource_requirements)
            })
            
            # Simulate task execution (in real implementation, this would call actual task)
            execution_success = random.random() > 0.05  # 95% success rate
            execution_time = task.estimated_duration * random.uniform(0.8, 1.2)
            
            # Update current time
            self.current_time += execution_time
            
            if execution_success:
                task.collapse(TaskState.COMPLETED)
                completed_tasks.add(task_id)
                
                # Restore resources (if applicable)
                for res, amount in task.resource_requirements.items():
                    if res.endswith('_renewable'):
                        current_resources[res] = min(
                            self.resource_pool[res],
                            current_resources[res] + amount
                        )
                
                execution_log.append({
                    'task_id': task_id,
                    'action': 'completed',
                    'time': time.time() - start_time,
                    'execution_time': execution_time
                })
            else:
                task.collapse(TaskState.FAILED)
                failed_tasks.add(task_id)
                
                execution_log.append({
                    'task_id': task_id,
                    'action': 'failed',
                    'time': time.time() - start_time,
                    'execution_time': execution_time
                })
        
        total_execution_time = time.time() - start_time
        
        return {
            'execution_log': execution_log,
            'completed_tasks': list(completed_tasks),
            'failed_tasks': list(failed_tasks),
            'total_execution_time': total_execution_time,
            'resource_utilization': {
                res: (initial - current_resources.get(res, 0)) / initial
                for res, initial in self.resource_pool.items()
                if initial > 0
            },
            'success_rate': len(completed_tasks) / len(task_order) if task_order else 0.0
        }
    
    def get_quantum_state_summary(self) -> Dict[str, Any]:
        """Get summary of current quantum states."""
        state_counts = defaultdict(int)
        total_probability = 0.0
        entanglement_count = 0
        
        for task in self.tasks.values():
            state_counts[task.state.value] += 1
            if task.state == TaskState.SUPERPOSITION:
                total_probability += task.probability()
            entanglement_count += len(task.entangled_tasks)
        
        return {
            'total_tasks': len(self.tasks),
            'state_distribution': dict(state_counts),
            'total_superposition_probability': total_probability,
            'entanglement_connections': entanglement_count // 2,  # Each pair counted twice
            'global_phase': self.global_phase,
            'active_interferences': len(self.interference_patterns)
        }
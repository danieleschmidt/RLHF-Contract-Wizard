"""
Enhanced quantum-inspired task planning with advanced quantum algorithms.

This module extends the core quantum planning with novel quantum algorithms
including quantum tunneling for constraint handling, quantum interference
optimization, and entangled state management.
"""

import time
import random
import math
import cmath
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

# Note: JAX imports handled with try/except for graceful degradation
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, random as jax_random
    JAX_AVAILABLE = True
except ImportError:
    # Fallback to numpy when JAX unavailable
    import numpy as jnp
    JAX_AVAILABLE = False


class QuantumStateType(Enum):
    """Enhanced quantum state types for advanced planning."""
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    TUNNELING = "tunneling"        # Task bypassing constraints
    INTERFERENCE = "interference"   # State interference optimization
    DECOHERENCE = "decoherence"    # State decay due to environment


@dataclass
class QuantumAmplitude:
    """Complex amplitude representing quantum state probability."""
    real: float
    imag: float
    
    @property
    def complex_value(self) -> complex:
        """Convert to Python complex number."""
        return complex(self.real, self.imag)
    
    @property
    def probability(self) -> float:
        """Calculate probability from amplitude."""
        return self.real**2 + self.imag**2
    
    @property
    def phase(self) -> float:
        """Calculate phase angle."""
        return math.atan2(self.imag, self.real)


@dataclass
class EnhancedQuantumTask:
    """
    Enhanced quantum task with advanced quantum properties.
    
    Supports quantum tunneling, interference patterns, and decoherence.
    """
    id: str
    name: str
    description: str
    priority: float = 0.5
    estimated_duration: float = 1.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    # Enhanced quantum properties
    quantum_state: QuantumStateType = QuantumStateType.SUPERPOSITION
    amplitude: QuantumAmplitude = field(default_factory=lambda: QuantumAmplitude(1.0, 0.0))
    entanglement_partners: Set[str] = field(default_factory=set)
    tunneling_probability: float = 0.1
    interference_weight: float = 1.0
    decoherence_rate: float = 0.01
    
    # Advanced properties
    quantum_barriers: List[Dict[str, Any]] = field(default_factory=list)
    optimization_history: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def apply_quantum_evolution(self, time_step: float) -> None:
        """Apply quantum evolution over time."""
        # Decoherence
        decay = math.exp(-self.decoherence_rate * time_step)
        self.amplitude.real *= decay
        self.amplitude.imag *= decay
        
        # Phase evolution
        phase_shift = 2 * math.pi * time_step / self.estimated_duration
        old_real = self.amplitude.real
        self.amplitude.real = old_real * math.cos(phase_shift) - self.amplitude.imag * math.sin(phase_shift)
        self.amplitude.imag = old_real * math.sin(phase_shift) + self.amplitude.imag * math.cos(phase_shift)
    
    def calculate_tunneling_success(self, barrier_height: float) -> bool:
        """Calculate if task can tunnel through constraint barriers."""
        transmission_coefficient = math.exp(-2 * barrier_height * self.tunneling_probability)
        return random.random() < transmission_coefficient


class EnhancedQuantumTaskPlanner:
    """
    Enhanced quantum-inspired task planner with advanced algorithms.
    
    Implements quantum tunneling, interference optimization, and
    sophisticated entanglement management.
    """
    
    def __init__(self, name: str = "Enhanced Quantum Planner"):
        self.name = name
        self.tasks: Dict[str, EnhancedQuantumTask] = {}
        self.quantum_register = {}  # Quantum state register
        self.interference_matrix = np.array([])
        self.tunneling_paths: Dict[str, List[str]] = {}
        
        # Advanced planning parameters
        self.quantum_coherence_time = 10.0
        self.interference_strength = 0.8
        self.tunneling_energy = 1.0
        self.optimization_rounds = 100
        
        # Performance tracking
        self.planning_history = []
        self.quantum_efficiency_score = 0.0
        
    def add_enhanced_task(self, task: EnhancedQuantumTask) -> None:
        """Add task to quantum planning system."""
        self.tasks[task.id] = task
        self._initialize_quantum_state(task)
        self._update_interference_matrix()
        
    def _initialize_quantum_state(self, task: EnhancedQuantumTask) -> None:
        """Initialize quantum state in register."""
        self.quantum_register[task.id] = {
            'amplitude': task.amplitude,
            'phase': 0.0,
            'coherence': 1.0,
            'entanglement_strength': 0.0
        }
    
    def _update_interference_matrix(self) -> None:
        """Update quantum interference matrix."""
        n_tasks = len(self.tasks)
        if n_tasks == 0:
            return
            
        self.interference_matrix = np.zeros((n_tasks, n_tasks), dtype=complex)
        
        task_ids = list(self.tasks.keys())
        for i, task_id_1 in enumerate(task_ids):
            for j, task_id_2 in enumerate(task_ids):
                if i != j:
                    task_1 = self.tasks[task_id_1]
                    task_2 = self.tasks[task_id_2]
                    
                    # Calculate interference strength based on dependency overlap
                    interference = self._calculate_interference(task_1, task_2)
                    self.interference_matrix[i, j] = interference
    
    def _calculate_interference(self, task_1: EnhancedQuantumTask, task_2: EnhancedQuantumTask) -> complex:
        """Calculate quantum interference between two tasks."""
        # Dependency-based interference
        common_deps = len(task_1.dependencies.intersection(task_2.dependencies))
        total_deps = len(task_1.dependencies.union(task_2.dependencies))
        
        if total_deps == 0:
            overlap = 0.0
        else:
            overlap = common_deps / total_deps
        
        # Priority difference creates phase shift
        priority_diff = abs(task_1.priority - task_2.priority)
        phase = priority_diff * math.pi / 2
        
        # Interference amplitude based on overlap and inverse duration
        amplitude = overlap * self.interference_strength / (
            task_1.estimated_duration * task_2.estimated_duration
        )**0.5
        
        return amplitude * cmath.exp(1j * phase)
    
    def quantum_tunnel_optimization(self) -> List[str]:
        """
        Perform optimization using quantum tunneling to escape local optima.
        
        Returns optimized task execution order.
        """
        current_order = list(self.tasks.keys())
        current_energy = self._calculate_system_energy(current_order)
        
        best_order = current_order.copy()
        best_energy = current_energy
        
        for _ in range(self.optimization_rounds):
            # Create tunneling candidate
            candidate_order = self._quantum_tunnel_move(current_order)
            candidate_energy = self._calculate_system_energy(candidate_order)
            
            # Quantum tunneling acceptance
            energy_diff = candidate_energy - current_energy
            
            if energy_diff < 0:
                # Accept improvement
                current_order = candidate_order
                current_energy = candidate_energy
                
                if candidate_energy < best_energy:
                    best_order = candidate_order
                    best_energy = candidate_energy
            else:
                # Quantum tunneling through barrier
                tunneling_prob = math.exp(-energy_diff / self.tunneling_energy)
                if random.random() < tunneling_prob:
                    current_order = candidate_order
                    current_energy = candidate_energy
        
        self.quantum_efficiency_score = 1.0 / (1.0 + best_energy)
        return best_order
    
    def _quantum_tunnel_move(self, order: List[str]) -> List[str]:
        """Generate candidate move using quantum tunneling."""
        new_order = order.copy()
        
        # Quantum tunneling can make non-local moves
        if random.random() < 0.3:  # 30% chance of non-local move
            # Tunneling move: swap distant tasks
            i, j = random.sample(range(len(new_order)), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        else:
            # Local move: adjacent swap
            if len(new_order) > 1:
                i = random.randint(0, len(new_order) - 2)
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
        
        return new_order
    
    def _calculate_system_energy(self, order: List[str]) -> float:
        """Calculate total system energy for given task order."""
        energy = 0.0
        
        # Dependency violation penalties
        executed = set()
        for task_id in order:
            task = self.tasks[task_id]
            
            # Penalty for unmet dependencies
            unmet_deps = task.dependencies - executed
            energy += len(unmet_deps) * 10.0  # High penalty
            
            # Priority-based energy (higher priority = lower energy)
            energy += (1.0 - task.priority) * 5.0
            
            # Duration-based energy
            energy += task.estimated_duration
            
            executed.add(task_id)
        
        # Quantum interference contributions
        if len(order) > 1 and self.interference_matrix.size > 0:
            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    interference = abs(self.interference_matrix[i, j])
                    # Constructive interference reduces energy
                    energy -= interference * 2.0
        
        return energy
    
    def apply_quantum_interference_optimization(self) -> Dict[str, float]:
        """
        Apply quantum interference to optimize task priorities.
        
        Returns updated priority scores.
        """
        if self.interference_matrix.size == 0:
            return {task_id: task.priority for task_id, task in self.tasks.items()}
        
        task_ids = list(self.tasks.keys())
        n_tasks = len(task_ids)
        
        # Initialize quantum amplitudes
        amplitudes = np.array([
            self.quantum_register[task_id]['amplitude'].complex_value
            for task_id in task_ids
        ])
        
        # Apply interference evolution
        for evolution_step in range(10):  # Multiple evolution steps
            new_amplitudes = amplitudes.copy()
            
            for i in range(n_tasks):
                interference_sum = 0
                for j in range(n_tasks):
                    if i != j:
                        interference_sum += self.interference_matrix[i, j] * amplitudes[j]
                
                new_amplitudes[i] += 0.1 * interference_sum  # Small evolution step
            
            # Normalize to preserve total probability
            total_probability = np.sum(np.abs(new_amplitudes)**2)
            if total_probability > 0:
                amplitudes = new_amplitudes / np.sqrt(total_probability)
        
        # Convert amplitudes back to priorities
        priorities = {}
        for i, task_id in enumerate(task_ids):
            probability = abs(amplitudes[i])**2
            # Map probability to priority range [0, 1]
            priorities[task_id] = min(1.0, max(0.0, probability * len(task_ids)))
        
        return priorities
    
    def detect_quantum_entanglement(self) -> Dict[str, List[str]]:
        """
        Detect quantum entangled task groups.
        
        Returns groups of entangled tasks.
        """
        entanglement_groups = {}
        processed = set()
        
        for task_id, task in self.tasks.items():
            if task_id in processed:
                continue
            
            # Find entangled partners through dependency analysis
            entangled_group = self._find_entangled_group(task_id, set())
            
            if len(entangled_group) > 1:
                group_key = f"entangled_group_{len(entanglement_groups)}"
                entanglement_groups[group_key] = list(entangled_group)
                processed.update(entangled_group)
        
        return entanglement_groups
    
    def _find_entangled_group(self, task_id: str, visited: Set[str]) -> Set[str]:
        """Recursively find entangled task group."""
        if task_id in visited:
            return set()
        
        visited.add(task_id)
        group = {task_id}
        task = self.tasks[task_id]
        
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id in self.tasks and dep_id not in visited:
                group.update(self._find_entangled_group(dep_id, visited.copy()))
        
        # Check dependents
        for dep_id in task.dependents:
            if dep_id in self.tasks and dep_id not in visited:
                group.update(self._find_entangled_group(dep_id, visited.copy()))
        
        return group
    
    def run_enhanced_quantum_planning(self) -> Dict[str, Any]:
        """
        Execute complete enhanced quantum planning algorithm.
        
        Returns comprehensive planning results.
        """
        start_time = time.time()
        
        # Step 1: Quantum interference optimization
        optimized_priorities = self.apply_quantum_interference_optimization()
        
        # Step 2: Detect entanglement groups
        entanglement_groups = self.detect_quantum_entanglement()
        
        # Step 3: Quantum tunneling optimization
        optimal_order = self.quantum_tunnel_optimization()
        
        # Step 4: Calculate quantum metrics
        quantum_metrics = self._calculate_quantum_metrics()
        
        planning_time = time.time() - start_time
        
        results = {
            'optimal_execution_order': optimal_order,
            'optimized_priorities': optimized_priorities,
            'entanglement_groups': entanglement_groups,
            'quantum_metrics': quantum_metrics,
            'planning_time_seconds': planning_time,
            'quantum_efficiency_score': self.quantum_efficiency_score,
            'total_tasks': len(self.tasks),
            'algorithm_version': 'Enhanced Quantum v2.0'
        }
        
        self.planning_history.append(results)
        return results
    
    def _calculate_quantum_metrics(self) -> Dict[str, float]:
        """Calculate advanced quantum planning metrics."""
        if not self.tasks:
            return {}
        
        total_coherence = sum(
            self.quantum_register[task_id]['coherence']
            for task_id in self.tasks.keys()
        ) / len(self.tasks)
        
        average_entanglement = sum(
            len(task.entanglement_partners)
            for task in self.tasks.values()
        ) / len(self.tasks)
        
        total_tunneling_potential = sum(
            task.tunneling_probability
            for task in self.tasks.values()
        ) / len(self.tasks)
        
        return {
            'average_quantum_coherence': total_coherence,
            'average_entanglement_degree': average_entanglement,
            'total_tunneling_potential': total_tunneling_potential,
            'interference_matrix_rank': np.linalg.matrix_rank(self.interference_matrix) if self.interference_matrix.size > 0 else 0,
            'system_complexity_score': len(self.tasks) * average_entanglement
        }


# Example usage and demonstration
def demonstrate_enhanced_quantum_planning():
    """Demonstrate enhanced quantum planning capabilities."""
    planner = EnhancedQuantumTaskPlanner("Demo Enhanced Planner")
    
    # Create enhanced quantum tasks
    tasks = [
        EnhancedQuantumTask(
            id="quantum_init",
            name="Initialize Quantum System",
            description="Set up quantum computing environment",
            priority=0.9,
            estimated_duration=2.0,
            tunneling_probability=0.15,
            interference_weight=1.2
        ),
        EnhancedQuantumTask(
            id="data_prep",
            name="Quantum Data Preparation",
            description="Prepare data for quantum processing",
            priority=0.8,
            estimated_duration=1.5,
            dependencies={"quantum_init"},
            tunneling_probability=0.1,
            interference_weight=1.0
        ),
        EnhancedQuantumTask(
            id="quantum_compute",
            name="Quantum Computation",
            description="Execute quantum algorithms",
            priority=0.95,
            estimated_duration=3.0,
            dependencies={"data_prep"},
            tunneling_probability=0.2,
            interference_weight=1.5
        ),
        EnhancedQuantumTask(
            id="result_analysis",
            name="Analyze Quantum Results",
            description="Process and analyze quantum computation results",
            priority=0.7,
            estimated_duration=1.0,
            dependencies={"quantum_compute"},
            tunneling_probability=0.05,
            interference_weight=0.8
        ),
        EnhancedQuantumTask(
            id="optimization",
            name="Quantum Optimization",
            description="Optimize quantum parameters",
            priority=0.6,
            estimated_duration=2.5,
            dependencies={"result_analysis"},
            tunneling_probability=0.3,
            interference_weight=1.1
        )
    ]
    
    # Add tasks to planner
    for task in tasks:
        planner.add_enhanced_task(task)
    
    # Set up entanglements
    tasks[1].entanglement_partners = {"quantum_compute"}  # data_prep entangled with quantum_compute
    tasks[2].entanglement_partners = {"data_prep"}       # quantum_compute entangled with data_prep
    
    # Run enhanced quantum planning
    results = planner.run_enhanced_quantum_planning()
    
    return results


if __name__ == "__main__":
    # Demonstrate enhanced quantum planning
    demo_results = demonstrate_enhanced_quantum_planning()
    print("Enhanced Quantum Planning Results:")
    print(f"Optimal Order: {demo_results['optimal_execution_order']}")
    print(f"Quantum Efficiency: {demo_results['quantum_efficiency_score']:.4f}")
    print(f"Planning Time: {demo_results['planning_time_seconds']:.4f}s")
    print(f"Entanglement Groups: {demo_results['entanglement_groups']}")
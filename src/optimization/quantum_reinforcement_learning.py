"""
Quantum-Enhanced Reinforcement Learning for RLHF Optimization.

This module implements cutting-edge quantum-enhanced reinforcement learning
algorithms specifically designed for optimizing RLHF reward functions,
combining quantum computing principles with advanced RL techniques.
"""

import time
import random
import math
import cmath
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np

# Graceful JAX handling
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad, random as jax_random
    import optax
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False


class QuantumGate(Enum):
    """Quantum gates for quantum circuit simulation."""
    HADAMARD = "hadamard"
    PAULI_X = "pauli_x"
    PAULI_Y = "pauli_y"
    PAULI_Z = "pauli_z"
    ROTATION_X = "rotation_x"
    ROTATION_Y = "rotation_y"
    ROTATION_Z = "rotation_z"
    CNOT = "cnot"
    TOFFOLI = "toffoli"
    PHASE = "phase"


class QRLAlgorithm(Enum):
    """Quantum-enhanced RL algorithms."""
    VARIATIONAL_QUANTUM_POLICY = "vqp"
    QUANTUM_ACTOR_CRITIC = "qac"
    QUANTUM_ADVANTAGE_ACTOR_CRITIC = "qa2c"
    QUANTUM_DEEP_Q_NETWORK = "qdqn"
    QUANTUM_POLICY_GRADIENT = "qpg"
    HYBRID_QUANTUM_CLASSICAL = "hqc"


@dataclass
class QuantumState:
    """Represents a quantum state vector."""
    amplitudes: np.ndarray  # Complex amplitudes
    num_qubits: int
    
    def __post_init__(self):
        # Normalize the state
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
    
    @property
    def probabilities(self) -> np.ndarray:
        """Get measurement probabilities."""
        return np.abs(self.amplitudes) ** 2
    
    def measure(self) -> int:
        """Measure the quantum state, returning classical outcome."""
        probabilities = self.probabilities
        return np.random.choice(len(probabilities), p=probabilities)
    
    def apply_gate(self, gate: QuantumGate, target_qubit: int, 
                   control_qubit: Optional[int] = None, 
                   angle: Optional[float] = None) -> 'QuantumState':
        """Apply a quantum gate to the state."""
        gate_matrix = self._get_gate_matrix(gate, angle)
        
        if control_qubit is not None:
            # Two-qubit gate
            new_amplitudes = self._apply_two_qubit_gate(
                gate_matrix, target_qubit, control_qubit
            )
        else:
            # Single-qubit gate
            new_amplitudes = self._apply_single_qubit_gate(gate_matrix, target_qubit)
        
        return QuantumState(new_amplitudes, self.num_qubits)
    
    def _get_gate_matrix(self, gate: QuantumGate, angle: Optional[float] = None) -> np.ndarray:
        """Get the matrix representation of a quantum gate."""
        if gate == QuantumGate.HADAMARD:
            return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        elif gate == QuantumGate.PAULI_X:
            return np.array([[0, 1], [1, 0]])
        elif gate == QuantumGate.PAULI_Y:
            return np.array([[0, -1j], [1j, 0]])
        elif gate == QuantumGate.PAULI_Z:
            return np.array([[1, 0], [0, -1]])
        elif gate == QuantumGate.ROTATION_X and angle is not None:
            cos_half = np.cos(angle / 2)
            sin_half = np.sin(angle / 2)
            return np.array([[cos_half, -1j * sin_half], 
                           [-1j * sin_half, cos_half]])
        elif gate == QuantumGate.ROTATION_Y and angle is not None:
            cos_half = np.cos(angle / 2)
            sin_half = np.sin(angle / 2)
            return np.array([[cos_half, -sin_half], 
                           [sin_half, cos_half]])
        elif gate == QuantumGate.ROTATION_Z and angle is not None:
            return np.array([[np.exp(-1j * angle / 2), 0], 
                           [0, np.exp(1j * angle / 2)]])
        elif gate == QuantumGate.PHASE and angle is not None:
            return np.array([[1, 0], [0, np.exp(1j * angle)]])
        elif gate == QuantumGate.CNOT:
            return np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]])
        else:
            return np.eye(2)  # Identity gate
    
    def _apply_single_qubit_gate(self, gate_matrix: np.ndarray, target_qubit: int) -> np.ndarray:
        """Apply single-qubit gate to the quantum state."""
        # Simplified implementation for demonstration
        # In practice, would use tensor product operations
        new_amplitudes = self.amplitudes.copy()
        
        # Apply gate to target qubit (simplified)
        for i in range(0, len(new_amplitudes), 2**(target_qubit + 1)):
            for j in range(2**target_qubit):
                idx0 = i + j
                idx1 = i + j + 2**target_qubit
                
                if idx1 < len(new_amplitudes):
                    amp0, amp1 = new_amplitudes[idx0], new_amplitudes[idx1]
                    new_amplitudes[idx0] = gate_matrix[0, 0] * amp0 + gate_matrix[0, 1] * amp1
                    new_amplitudes[idx1] = gate_matrix[1, 0] * amp0 + gate_matrix[1, 1] * amp1
        
        return new_amplitudes
    
    def _apply_two_qubit_gate(self, gate_matrix: np.ndarray, 
                             target_qubit: int, control_qubit: int) -> np.ndarray:
        """Apply two-qubit gate to the quantum state."""
        # Simplified CNOT implementation
        new_amplitudes = self.amplitudes.copy()
        
        for i in range(len(new_amplitudes)):
            control_bit = (i >> control_qubit) & 1
            if control_bit == 1:  # Control qubit is |1⟩
                # Flip target qubit
                target_bit = (i >> target_qubit) & 1
                new_index = i ^ (1 << target_qubit)  # Flip target bit
                
                # Swap amplitudes
                new_amplitudes[i], new_amplitudes[new_index] = \
                    new_amplitudes[new_index], new_amplitudes[i]
        
        return new_amplitudes


@dataclass
class QuantumCircuit:
    """Represents a parameterized quantum circuit."""
    num_qubits: int
    gates: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    
    def add_gate(self, gate: QuantumGate, target_qubit: int, 
                 control_qubit: Optional[int] = None, 
                 parameter_name: Optional[str] = None) -> None:
        """Add a gate to the circuit."""
        gate_info = {
            'gate': gate,
            'target_qubit': target_qubit,
            'control_qubit': control_qubit,
            'parameter_name': parameter_name
        }
        self.gates.append(gate_info)
    
    def execute(self, initial_state: Optional[QuantumState] = None) -> QuantumState:
        """Execute the quantum circuit."""
        if initial_state is None:
            # Start with |0⟩^n state
            amplitudes = np.zeros(2**self.num_qubits, dtype=complex)
            amplitudes[0] = 1.0
            state = QuantumState(amplitudes, self.num_qubits)
        else:
            state = initial_state
        
        # Apply gates sequentially
        for gate_info in self.gates:
            gate = gate_info['gate']
            target = gate_info['target_qubit']
            control = gate_info['control_qubit']
            param_name = gate_info['parameter_name']
            
            angle = None
            if param_name and param_name in self.parameters:
                angle = self.parameters[param_name]
            
            state = state.apply_gate(gate, target, control, angle)
        
        return state
    
    def update_parameters(self, new_parameters: Dict[str, float]) -> None:
        """Update circuit parameters."""
        self.parameters.update(new_parameters)


class QuantumReinforcementLearner:
    """
    Quantum-enhanced reinforcement learning system.
    
    Implements variational quantum circuits for policy and value function
    approximation in reinforcement learning contexts.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 num_qubits: int = 4,
                 algorithm: QRLAlgorithm = QRLAlgorithm.VARIATIONAL_QUANTUM_POLICY,
                 learning_rate: float = 0.01):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_qubits = num_qubits
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        
        # Quantum circuits
        self.policy_circuit = self._create_policy_circuit()
        self.value_circuit = self._create_value_circuit()
        
        # Classical components
        self.optimizer = self._initialize_optimizer()
        self.experience_buffer = deque(maxlen=10000)
        
        # Training state
        self.episode_count = 0
        self.total_reward = 0.0
        self.training_history: List[Dict[str, float]] = []
        
        # Quantum-specific parameters
        self.entanglement_strength = 0.5
        self.measurement_shots = 1000
        self.circuit_depth = 3
        
        # Performance metrics
        self.quantum_advantage_score = 0.0
        self.circuit_fidelity = 1.0
        self.decoherence_rate = 0.01
    
    def _create_policy_circuit(self) -> QuantumCircuit:
        """Create parameterized quantum circuit for policy."""
        circuit = QuantumCircuit(self.num_qubits)
        
        # Input encoding layer
        for i in range(min(self.state_dim, self.num_qubits)):
            circuit.add_gate(QuantumGate.ROTATION_Y, i, parameter_name=f'input_{i}')
        
        # Variational layers
        for layer in range(self.circuit_depth):
            # Rotation gates
            for qubit in range(self.num_qubits):
                circuit.add_gate(QuantumGate.ROTATION_X, qubit, 
                               parameter_name=f'rx_{layer}_{qubit}')
                circuit.add_gate(QuantumGate.ROTATION_Z, qubit, 
                               parameter_name=f'rz_{layer}_{qubit}')
            
            # Entangling gates
            for qubit in range(self.num_qubits - 1):
                circuit.add_gate(QuantumGate.CNOT, (qubit + 1) % self.num_qubits, qubit)
        
        # Initialize parameters
        param_count = self.state_dim + 2 * self.circuit_depth * self.num_qubits
        for i, param_name in enumerate(circuit.parameters.keys()):
            if param_name not in circuit.parameters:
                circuit.parameters[param_name] = random.uniform(-np.pi, np.pi)
        
        return circuit
    
    def _create_value_circuit(self) -> QuantumCircuit:
        """Create parameterized quantum circuit for value estimation."""
        circuit = QuantumCircuit(self.num_qubits)
        
        # Similar structure to policy circuit but optimized for value estimation
        for i in range(min(self.state_dim, self.num_qubits)):
            circuit.add_gate(QuantumGate.ROTATION_Y, i, parameter_name=f'val_input_{i}')
        
        # Variational layers with different parameterization
        for layer in range(self.circuit_depth):
            for qubit in range(self.num_qubits):
                circuit.add_gate(QuantumGate.ROTATION_Y, qubit, 
                               parameter_name=f'val_ry_{layer}_{qubit}')
                circuit.add_gate(QuantumGate.ROTATION_Z, qubit, 
                               parameter_name=f'val_rz_{layer}_{qubit}')
            
            # Entangling pattern
            for qubit in range(self.num_qubits - 1):
                circuit.add_gate(QuantumGate.CNOT, qubit + 1, qubit)
        
        # Initialize value circuit parameters
        for param_name in [f'val_input_{i}' for i in range(min(self.state_dim, self.num_qubits))]:
            circuit.parameters[param_name] = random.uniform(-np.pi, np.pi)
        
        for layer in range(self.circuit_depth):
            for qubit in range(self.num_qubits):
                circuit.parameters[f'val_ry_{layer}_{qubit}'] = random.uniform(-np.pi, np.pi)
                circuit.parameters[f'val_rz_{layer}_{qubit}'] = random.uniform(-np.pi, np.pi)
        
        return circuit
    
    def _initialize_optimizer(self) -> Dict[str, Any]:
        """Initialize classical optimizer for quantum parameters."""
        return {
            'type': 'adam',
            'learning_rate': self.learning_rate,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'm': {},  # First moment estimates
            'v': {},  # Second moment estimates
            't': 0    # Time step
        }
    
    def encode_state(self, state: np.ndarray) -> Dict[str, float]:
        """Encode classical state into quantum circuit parameters."""
        encoded_params = {}
        
        # Normalize state to [-π, π] range
        normalized_state = np.arctan(state) if np.any(state) else np.zeros_like(state)
        
        # Encode into input parameters
        for i in range(min(len(normalized_state), self.num_qubits)):
            encoded_params[f'input_{i}'] = normalized_state[i]
            encoded_params[f'val_input_{i}'] = normalized_state[i]
        
        return encoded_params
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities from quantum policy."""
        # Encode state
        state_params = self.encode_state(state)
        
        # Update circuit parameters
        self.policy_circuit.update_parameters(state_params)
        
        # Execute quantum circuit
        quantum_state = self.policy_circuit.execute()
        
        # Multiple measurements to estimate probabilities
        measurements = []
        for _ in range(self.measurement_shots):
            measurement = quantum_state.measure()
            measurements.append(measurement)
        
        # Convert measurements to action probabilities
        action_probs = self._measurements_to_action_probs(measurements)
        
        return action_probs
    
    def _measurements_to_action_probs(self, measurements: List[int]) -> np.ndarray:
        """Convert quantum measurements to action probabilities."""
        # Count measurement outcomes
        outcome_counts = defaultdict(int)
        for measurement in measurements:
            outcome_counts[measurement] += 1
        
        # Map quantum states to actions
        action_probs = np.zeros(self.action_dim)
        
        total_measurements = len(measurements)
        for outcome, count in outcome_counts.items():
            action_idx = outcome % self.action_dim  # Map quantum state to action
            action_probs[action_idx] += count / total_measurements
        
        # Ensure probabilities sum to 1
        if np.sum(action_probs) > 0:
            action_probs = action_probs / np.sum(action_probs)
        else:
            action_probs = np.ones(self.action_dim) / self.action_dim
        
        return action_probs
    
    def select_action(self, state: np.ndarray, exploration_rate: float = 0.1) -> int:
        """Select action using quantum policy with exploration."""
        if random.random() < exploration_rate:
            return random.randint(0, self.action_dim - 1)
        
        action_probs = self.get_action_probabilities(state)
        return np.random.choice(self.action_dim, p=action_probs)
    
    def estimate_value(self, state: np.ndarray) -> float:
        """Estimate state value using quantum value circuit."""
        # Encode state
        state_params = self.encode_state(state)
        
        # Update value circuit parameters
        self.value_circuit.update_parameters(state_params)
        
        # Execute quantum circuit
        quantum_state = self.value_circuit.execute()
        
        # Estimate value from quantum state
        # Use expectation value of a Pauli-Z measurement on first qubit
        probabilities = quantum_state.probabilities
        
        # Calculate expectation value ⟨Z⟩ = P(0) - P(1)
        prob_0 = sum(probabilities[i] for i in range(len(probabilities)) if (i & 1) == 0)
        prob_1 = sum(probabilities[i] for i in range(len(probabilities)) if (i & 1) == 1)
        
        expectation_value = prob_0 - prob_1
        
        # Scale to appropriate value range
        value_estimate = expectation_value * 10.0  # Scale factor
        
        return value_estimate
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform one training step using quantum gradients."""
        if len(self.experience_buffer) < batch_size:
            return {'loss': 0.0, 'value_loss': 0.0, 'policy_loss': 0.0}
        
        # Sample batch from experience buffer
        batch = random.sample(list(self.experience_buffer), batch_size)
        
        # Calculate gradients using parameter shift rule
        policy_gradients = self._calculate_policy_gradients(batch)
        value_gradients = self._calculate_value_gradients(batch)
        
        # Update parameters using optimizer
        self._update_parameters(policy_gradients, value_gradients)
        
        # Calculate losses for monitoring
        policy_loss = np.mean([self._policy_loss(transition) for transition in batch])
        value_loss = np.mean([self._value_loss(transition) for transition in batch])
        total_loss = policy_loss + value_loss
        
        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'quantum_advantage': self.quantum_advantage_score
        }
    
    def _calculate_policy_gradients(self, batch: List[Tuple]) -> Dict[str, float]:
        """Calculate policy gradients using parameter shift rule."""
        gradients = {}
        
        for param_name in self.policy_circuit.parameters:
            if param_name.startswith('input_') or param_name.startswith('val_'):
                continue  # Skip input encoding parameters
            
            # Parameter shift rule: grad = (f(θ + π/2) - f(θ - π/2)) / 2
            shift = np.pi / 2
            
            # Forward pass with +shift
            original_value = self.policy_circuit.parameters[param_name]
            self.policy_circuit.parameters[param_name] = original_value + shift
            loss_plus = np.mean([self._policy_loss(transition) for transition in batch])
            
            # Forward pass with -shift
            self.policy_circuit.parameters[param_name] = original_value - shift
            loss_minus = np.mean([self._policy_loss(transition) for transition in batch])
            
            # Restore original value
            self.policy_circuit.parameters[param_name] = original_value
            
            # Calculate gradient
            gradients[param_name] = (loss_plus - loss_minus) / 2.0
        
        return gradients
    
    def _calculate_value_gradients(self, batch: List[Tuple]) -> Dict[str, float]:
        """Calculate value function gradients using parameter shift rule."""
        gradients = {}
        
        for param_name in self.value_circuit.parameters:
            if param_name.startswith('val_input_'):
                continue  # Skip input encoding parameters
            
            # Parameter shift rule
            shift = np.pi / 2
            
            # Forward pass with +shift
            original_value = self.value_circuit.parameters[param_name]
            self.value_circuit.parameters[param_name] = original_value + shift
            loss_plus = np.mean([self._value_loss(transition) for transition in batch])
            
            # Forward pass with -shift
            self.value_circuit.parameters[param_name] = original_value - shift
            loss_minus = np.mean([self._value_loss(transition) for transition in batch])
            
            # Restore original value
            self.value_circuit.parameters[param_name] = original_value
            
            # Calculate gradient
            gradients[param_name] = (loss_plus - loss_minus) / 2.0
        
        return gradients
    
    def _policy_loss(self, transition: Tuple) -> float:
        """Calculate policy loss for a single transition."""
        state, action, reward, next_state, done = transition
        
        # Get action probabilities
        action_probs = self.get_action_probabilities(state)
        
        # Calculate advantage (simplified)
        value = self.estimate_value(state)
        next_value = 0.0 if done else self.estimate_value(next_state)
        advantage = reward + 0.99 * next_value - value  # γ = 0.99
        
        # Policy gradient loss
        log_prob = np.log(action_probs[action] + 1e-8)
        loss = -log_prob * advantage
        
        return loss
    
    def _value_loss(self, transition: Tuple) -> float:
        """Calculate value function loss for a single transition."""
        state, action, reward, next_state, done = transition
        
        # Estimated value
        estimated_value = self.estimate_value(state)
        
        # Target value
        if done:
            target_value = reward
        else:
            target_value = reward + 0.99 * self.estimate_value(next_state)
        
        # Mean squared error
        loss = (estimated_value - target_value) ** 2
        
        return loss
    
    def _update_parameters(self, policy_gradients: Dict[str, float], 
                          value_gradients: Dict[str, float]) -> None:
        """Update parameters using Adam optimizer."""
        self.optimizer['t'] += 1
        t = self.optimizer['t']
        
        # Update policy parameters
        for param_name, gradient in policy_gradients.items():
            if param_name not in self.optimizer['m']:
                self.optimizer['m'][param_name] = 0.0
                self.optimizer['v'][param_name] = 0.0
            
            # Adam update
            self.optimizer['m'][param_name] = (
                self.optimizer['beta1'] * self.optimizer['m'][param_name] + 
                (1 - self.optimizer['beta1']) * gradient
            )
            self.optimizer['v'][param_name] = (
                self.optimizer['beta2'] * self.optimizer['v'][param_name] + 
                (1 - self.optimizer['beta2']) * gradient ** 2
            )
            
            # Bias correction
            m_corrected = self.optimizer['m'][param_name] / (1 - self.optimizer['beta1'] ** t)
            v_corrected = self.optimizer['v'][param_name] / (1 - self.optimizer['beta2'] ** t)
            
            # Parameter update
            self.policy_circuit.parameters[param_name] -= (
                self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.optimizer['epsilon'])
            )
        
        # Update value parameters
        for param_name, gradient in value_gradients.items():
            if param_name not in self.optimizer['m']:
                self.optimizer['m'][param_name] = 0.0
                self.optimizer['v'][param_name] = 0.0
            
            # Adam update (similar to above)
            self.optimizer['m'][param_name] = (
                self.optimizer['beta1'] * self.optimizer['m'][param_name] + 
                (1 - self.optimizer['beta1']) * gradient
            )
            self.optimizer['v'][param_name] = (
                self.optimizer['beta2'] * self.optimizer['v'][param_name] + 
                (1 - self.optimizer['beta2']) * gradient ** 2
            )
            
            # Bias correction
            m_corrected = self.optimizer['m'][param_name] / (1 - self.optimizer['beta1'] ** t)
            v_corrected = self.optimizer['v'][param_name] / (1 - self.optimizer['beta2'] ** t)
            
            # Parameter update
            self.value_circuit.parameters[param_name] -= (
                self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.optimizer['epsilon'])
            )
    
    def add_experience(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool) -> None:
        """Add experience to replay buffer."""
        experience = (state, action, reward, next_state, done)
        self.experience_buffer.append(experience)
    
    def train_episode(self, environment_step_function: Callable, 
                     max_steps: int = 1000) -> Dict[str, float]:
        """Train for one episode using the provided environment step function."""
        episode_start = time.time()
        
        # Initialize episode
        state = np.random.randn(self.state_dim)  # Random initial state
        episode_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state, exploration_rate=0.1)
            
            # Take step in environment (simulated)
            next_state, reward, done = environment_step_function(state, action)
            
            # Add experience
            self.add_experience(state, action, reward, next_state, done)
            
            episode_reward += reward
            steps += 1
            
            # Train if we have enough experience
            if len(self.experience_buffer) >= 32:
                train_metrics = self.train_step()
            
            if done:
                break
            
            state = next_state
        
        # Update quantum advantage score
        self._update_quantum_advantage_score(episode_reward)
        
        episode_time = time.time() - episode_start
        
        episode_metrics = {
            'episode_reward': episode_reward,
            'episode_steps': steps,
            'episode_time': episode_time,
            'quantum_advantage': self.quantum_advantage_score,
            'circuit_fidelity': self.circuit_fidelity,
            'experience_buffer_size': len(self.experience_buffer)
        }
        
        self.training_history.append(episode_metrics)
        self.episode_count += 1
        
        return episode_metrics
    
    def _update_quantum_advantage_score(self, episode_reward: float) -> None:
        """Update quantum advantage score based on performance."""
        # Simple heuristic for quantum advantage
        # In practice, would compare with classical baseline
        
        if len(self.training_history) > 0:
            recent_rewards = [ep['episode_reward'] for ep in self.training_history[-10:]]
            average_recent_reward = np.mean(recent_rewards)
            
            # Quantum advantage heuristic
            baseline_performance = 0.0  # Would be classical baseline
            quantum_performance = average_recent_reward
            
            if baseline_performance != 0:
                self.quantum_advantage_score = (quantum_performance - baseline_performance) / abs(baseline_performance)
            else:
                self.quantum_advantage_score = quantum_performance / 10.0  # Normalized
        
        # Decay circuit fidelity over time (simulate decoherence)
        self.circuit_fidelity = max(0.5, self.circuit_fidelity * (1 - self.decoherence_rate))
    
    def get_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        if not self.training_history:
            return {'status': 'No training data available'}
        
        recent_episodes = self.training_history[-10:]
        
        return {
            'total_episodes': self.episode_count,
            'average_recent_reward': np.mean([ep['episode_reward'] for ep in recent_episodes]),
            'max_episode_reward': max([ep['episode_reward'] for ep in self.training_history]),
            'average_episode_steps': np.mean([ep['episode_steps'] for ep in recent_episodes]),
            'quantum_advantage_score': self.quantum_advantage_score,
            'circuit_fidelity': self.circuit_fidelity,
            'policy_parameters': len(self.policy_circuit.parameters),
            'value_parameters': len(self.value_circuit.parameters),
            'experience_buffer_size': len(self.experience_buffer),
            'algorithm': self.algorithm.value,
            'quantum_circuit_depth': self.circuit_depth,
            'num_qubits': self.num_qubits
        }


# Demonstration function
def demonstrate_quantum_reinforcement_learning():
    """Demonstrate quantum reinforcement learning capabilities."""
    
    # Define a simple environment step function
    def simple_environment_step(state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
        """Simple reward environment for demonstration."""
        # Simple dynamics: next state is slightly modified current state
        next_state = state + np.random.normal(0, 0.1, size=state.shape)
        next_state = np.clip(next_state, -2, 2)  # Bounded state space
        
        # Simple reward: prefer actions that keep state near zero
        state_distance = np.linalg.norm(state)
        reward = 1.0 - state_distance / 2.0  # Reward decreases with distance from origin
        
        # Add action-based reward
        if action == 0:  # "good" action
            reward += 0.5
        else:  # other actions
            reward -= 0.1
        
        # Episode termination
        done = state_distance > 1.5 or random.random() < 0.05  # Random termination
        
        return next_state, reward, done
    
    # Initialize quantum RL agent
    qrl_agent = QuantumReinforcementLearner(
        state_dim=3,
        action_dim=2,
        num_qubits=4,
        algorithm=QRLAlgorithm.VARIATIONAL_QUANTUM_POLICY,
        learning_rate=0.01
    )
    
    print("🚀 QUANTUM REINFORCEMENT LEARNING DEMONSTRATION")
    print("=" * 60)
    
    # Train for several episodes
    training_results = []
    
    for episode in range(10):  # Smaller number for demo
        print(f"\n🧠 Training Episode {episode + 1}/10")
        
        episode_metrics = qrl_agent.train_episode(
            environment_step_function=simple_environment_step,
            max_steps=50  # Shorter episodes for demo
        )
        
        training_results.append(episode_metrics)
        
        print(f"   Episode Reward: {episode_metrics['episode_reward']:.3f}")
        print(f"   Steps: {episode_metrics['episode_steps']}")
        print(f"   Quantum Advantage: {episode_metrics['quantum_advantage']:.4f}")
    
    # Generate training report
    training_report = qrl_agent.get_training_report()
    
    return {
        'agent': qrl_agent,
        'training_results': training_results,
        'training_report': training_report
    }


if __name__ == "__main__":
    demo_results = demonstrate_quantum_reinforcement_learning()
    
    report = demo_results['training_report']
    results = demo_results['training_results']
    
    print("\n" + "=" * 60)
    print("🎯 QUANTUM RL TRAINING REPORT")
    print("=" * 60)
    
    print(f"Total Episodes: {report['total_episodes']}")
    print(f"Average Recent Reward: {report['average_recent_reward']:.3f}")
    print(f"Max Episode Reward: {report['max_episode_reward']:.3f}")
    print(f"Quantum Advantage Score: {report['quantum_advantage_score']:.4f}")
    print(f"Circuit Fidelity: {report['circuit_fidelity']:.4f}")
    
    print(f"\n🔬 QUANTUM SYSTEM DETAILS:")
    print(f"Algorithm: {report['algorithm']}")
    print(f"Number of Qubits: {report['num_qubits']}")
    print(f"Circuit Depth: {report['quantum_circuit_depth']}")
    print(f"Policy Parameters: {report['policy_parameters']}")
    print(f"Value Parameters: {report['value_parameters']}")
    print(f"Experience Buffer: {report['experience_buffer_size']}")
    
    print(f"\n📈 TRAINING PROGRESSION:")
    for i, result in enumerate(results[-5:]):  # Show last 5 episodes
        print(f"Episode {len(results)-4+i}: Reward {result['episode_reward']:.3f}, "
              f"Steps {result['episode_steps']}, QA {result['quantum_advantage']:.4f}")
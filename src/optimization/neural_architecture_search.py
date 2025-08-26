"""
Neural Architecture Search (NAS) for RLHF Reward Models.

This module implements advanced neural architecture search techniques
specifically optimized for reward modeling in RLHF systems, with support
for multi-objective optimization, efficiency constraints, and automated
hyperparameter optimization.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, grad
import flax.linen as nn
from flax.training import train_state
import optax

from ..models.reward_contract import RewardContract
from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


@dataclass
class ArchitectureSpec:
    """Specification for a neural architecture."""
    name: str
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]  # (from_layer, to_layer)
    input_dim: int
    output_dim: int
    parameters_count: int = 0
    flops_count: int = 0
    memory_usage_mb: float = 0.0
    
    # Performance metrics
    accuracy: float = 0.0
    convergence_speed: int = 0
    inference_latency_ms: float = 0.0
    training_time_hours: float = 0.0
    
    # Multi-objective scores
    pareto_rank: int = 0
    dominates_count: int = 0
    crowding_distance: float = 0.0


@dataclass
class SearchSpace:
    """Defines the neural architecture search space."""
    layer_types: List[str] = field(default_factory=lambda: [
        "linear", "attention", "mlp", "residual", "normalization", 
        "dropout", "activation", "embedding"
    ])
    
    activation_functions: List[str] = field(default_factory=lambda: [
        "relu", "gelu", "swish", "tanh", "leaky_relu"
    ])
    
    hidden_sizes: List[int] = field(default_factory=lambda: [
        64, 128, 256, 512, 768, 1024, 1536, 2048
    ])
    
    attention_heads: List[int] = field(default_factory=lambda: [4, 8, 12, 16])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])
    depth_range: Tuple[int, int] = (2, 12)
    width_multipliers: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5])


@dataclass
class OptimizationObjectives:
    """Multi-objective optimization targets."""
    maximize_accuracy: bool = True
    minimize_parameters: bool = True
    minimize_latency: bool = True
    minimize_training_time: bool = True
    maximize_convergence_speed: bool = True
    
    # Objective weights
    accuracy_weight: float = 0.4
    efficiency_weight: float = 0.3
    speed_weight: float = 0.2
    size_weight: float = 0.1
    
    # Constraints
    max_parameters_m: float = 100.0  # Million parameters
    max_inference_latency_ms: float = 100.0
    max_memory_usage_gb: float = 4.0


class RewardModelArchitecture(nn.Module):
    """Flexible reward model architecture from specification."""
    
    spec: ArchitectureSpec
    
    def setup(self):
        """Build the architecture from specification."""
        self.layers = []
        
        for i, layer_config in enumerate(self.spec.layers):
            layer_type = layer_config["type"]
            
            if layer_type == "linear":
                layer = nn.Dense(
                    features=layer_config["features"],
                    use_bias=layer_config.get("use_bias", True)
                )
            elif layer_type == "attention":
                layer = nn.MultiHeadDotProductAttention(
                    num_heads=layer_config["num_heads"],
                    qkv_features=layer_config.get("qkv_features"),
                    out_features=layer_config.get("out_features")
                )
            elif layer_type == "mlp":
                layer = MLP(
                    features=layer_config["features"],
                    activation=layer_config.get("activation", "relu")
                )
            elif layer_type == "residual":
                layer = ResidualBlock(
                    features=layer_config["features"],
                    activation=layer_config.get("activation", "relu")
                )
            elif layer_type == "normalization":
                norm_type = layer_config.get("norm_type", "layer_norm")
                if norm_type == "layer_norm":
                    layer = nn.LayerNorm()
                elif norm_type == "batch_norm":
                    layer = nn.BatchNorm(use_running_average=False)
                else:
                    layer = nn.LayerNorm()
            elif layer_type == "dropout":
                layer = nn.Dropout(rate=layer_config["rate"])
            elif layer_type == "activation":
                activation_fn = layer_config.get("function", "relu")
                layer = ActivationLayer(activation_fn)
            else:
                # Default to linear layer
                layer = nn.Dense(features=layer_config.get("features", 256))
            
            self.layers.append(layer)
    
    def __call__(self, x, training=False):
        """Forward pass through the architecture."""
        
        # Handle connections (for now, assume sequential)
        for i, layer in enumerate(self.layers):
            layer_config = self.spec.layers[i]
            layer_type = layer_config["type"]
            
            if layer_type == "dropout":
                x = layer(x, deterministic=not training)
            elif layer_type == "normalization":
                x = layer(x)
            elif layer_type == "activation":
                x = layer(x)
            else:
                x = layer(x)
        
        return x


class MLP(nn.Module):
    """Multi-layer perceptron block."""
    features: List[int]
    activation: str = "relu"
    
    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
        self.activation_fn = self._get_activation(self.activation)
    
    def _get_activation(self, activation: str):
        activations = {
            "relu": nn.relu,
            "gelu": nn.gelu,
            "swish": nn.swish,
            "tanh": nn.tanh,
            "leaky_relu": lambda x: nn.leaky_relu(x, negative_slope=0.01)
        }
        return activations.get(activation, nn.relu)
    
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on final layer
                x = self.activation_fn(x)
        return x


class ResidualBlock(nn.Module):
    """Residual connection block."""
    features: int
    activation: str = "relu"
    
    def setup(self):
        self.linear1 = nn.Dense(self.features)
        self.linear2 = nn.Dense(self.features)
        self.activation_fn = MLP._get_activation(None, self.activation)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
    
    def __call__(self, x):
        residual = x
        
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.linear1(x)
        
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        
        # Residual connection
        if x.shape == residual.shape:
            x = x + residual
        
        return x


class ActivationLayer(nn.Module):
    """Standalone activation layer."""
    function: str
    
    def __call__(self, x):
        activation_fn = MLP._get_activation(None, self.function)
        return activation_fn(x)


class NeuralArchitectureSearch:
    """
    Neural Architecture Search engine for reward models.
    
    Implements multiple search strategies:
    - Evolutionary search with Pareto optimization
    - Bayesian optimization for hyperparameters
    - Progressive search with early stopping
    - Hardware-aware optimization
    """
    
    def __init__(
        self,
        search_space: SearchSpace = None,
        objectives: OptimizationObjectives = None,
        population_size: int = 50,
        generations: int = 100,
        output_dir: Path = Path("nas_outputs")
    ):
        self.search_space = search_space or SearchSpace()
        self.objectives = objectives or OptimizationObjectives()
        self.population_size = population_size
        self.generations = generations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Search state
        self.population: List[ArchitectureSpec] = []
        self.generation = 0
        self.best_architectures: List[ArchitectureSpec] = []
        self.pareto_front: List[ArchitectureSpec] = []
        
        # Performance tracking
        self.search_history: List[Dict[str, Any]] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for NAS process."""
        log_file = self.output_dir / "nas_search.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def search_optimal_architecture(
        self,
        reward_contract: RewardContract,
        train_data: Dict[str, jnp.ndarray],
        val_data: Dict[str, jnp.ndarray],
        search_strategy: str = "evolutionary"
    ) -> List[ArchitectureSpec]:
        """
        Search for optimal neural architectures.
        
        Args:
            reward_contract: Target reward contract
            train_data: Training dataset
            val_data: Validation dataset  
            search_strategy: "evolutionary", "bayesian", or "progressive"
            
        Returns:
            List of optimal architectures (Pareto front)
        """
        self.logger.info(f"Starting NAS with strategy: {search_strategy}")
        
        if search_strategy == "evolutionary":
            return await self._evolutionary_search(reward_contract, train_data, val_data)
        elif search_strategy == "bayesian":
            return await self._bayesian_optimization(reward_contract, train_data, val_data)
        elif search_strategy == "progressive":
            return await self._progressive_search(reward_contract, train_data, val_data)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")
    
    async def _evolutionary_search(
        self,
        reward_contract: RewardContract,
        train_data: Dict[str, jnp.ndarray],
        val_data: Dict[str, jnp.ndarray]
    ) -> List[ArchitectureSpec]:
        """Evolutionary multi-objective architecture search."""
        
        # Initialize population
        self.logger.info("Initializing random population")
        self.population = self._generate_initial_population()
        
        # Evaluate initial population
        await self._evaluate_population(self.population, reward_contract, train_data, val_data)
        
        # Evolution loop
        for generation in range(self.generations):
            self.generation = generation
            self.logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Selection
            parents = self._selection(self.population)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._crossover(parents[i], parents[i + 1])
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                offspring.extend([child1, child2])
            
            # Evaluate offspring
            await self._evaluate_population(offspring, reward_contract, train_data, val_data)
            
            # Environmental selection (combine parents and offspring)
            combined = self.population + offspring
            self.population = self._environmental_selection(combined)
            
            # Update Pareto front
            self._update_pareto_front(self.population)
            
            # Track progress
            self._record_generation_stats()
            
            # Early stopping check
            if self._should_stop_search():
                self.logger.info(f"Early stopping at generation {generation + 1}")
                break
        
        self.logger.info(f"Search completed. Pareto front size: {len(self.pareto_front)}")
        
        # Save results
        self._save_search_results()
        
        return self.pareto_front
    
    def _generate_initial_population(self) -> List[ArchitectureSpec]:
        """Generate initial random population of architectures."""
        population = []
        
        for i in range(self.population_size):
            try:
                arch_spec = self._generate_random_architecture(f"arch_{i}")
                population.append(arch_spec)
            except Exception as e:
                self.logger.warning(f"Failed to generate architecture {i}: {e}")
                continue
        
        return population
    
    def _generate_random_architecture(self, name: str) -> ArchitectureSpec:
        """Generate a random architecture within search space."""
        rng = np.random.RandomState()
        
        # Random depth
        depth = rng.randint(*self.search_space.depth_range)
        
        # Random input/output dimensions
        input_dim = 512  # Standard for reward models
        output_dim = 1   # Single reward value
        
        # Generate layers
        layers = []
        current_dim = input_dim
        
        for layer_idx in range(depth):
            layer_type = rng.choice(self.search_space.layer_types)
            
            if layer_type == "linear":
                if layer_idx == depth - 1:  # Final layer
                    features = output_dim
                else:
                    features = rng.choice(self.search_space.hidden_sizes)
                
                layers.append({
                    "type": "linear",
                    "features": features,
                    "use_bias": rng.choice([True, False])
                })
                current_dim = features
                
            elif layer_type == "attention":
                num_heads = rng.choice(self.search_space.attention_heads)
                # Ensure dimension is divisible by heads
                features = ((current_dim // num_heads) * num_heads) or num_heads
                
                layers.append({
                    "type": "attention", 
                    "num_heads": num_heads,
                    "qkv_features": features,
                    "out_features": features
                })
                
            elif layer_type == "mlp":
                hidden_size = rng.choice(self.search_space.hidden_sizes)
                layers.append({
                    "type": "mlp",
                    "features": [current_dim, hidden_size, current_dim],
                    "activation": rng.choice(self.search_space.activation_functions)
                })
                
            elif layer_type == "residual":
                layers.append({
                    "type": "residual",
                    "features": current_dim,
                    "activation": rng.choice(self.search_space.activation_functions)
                })
                
            elif layer_type == "normalization":
                layers.append({
                    "type": "normalization",
                    "norm_type": rng.choice(["layer_norm", "batch_norm"])
                })
                
            elif layer_type == "dropout":
                if layer_idx < depth - 1:  # No dropout on final layer
                    layers.append({
                        "type": "dropout",
                        "rate": rng.choice(self.search_space.dropout_rates)
                    })
                    
            elif layer_type == "activation":
                if layer_idx < depth - 1:  # No activation on final layer
                    layers.append({
                        "type": "activation",
                        "function": rng.choice(self.search_space.activation_functions)
                    })
        
        # Ensure we have output layer
        if not layers or layers[-1]["type"] != "linear" or layers[-1].get("features") != output_dim:
            layers.append({
                "type": "linear",
                "features": output_dim,
                "use_bias": True
            })
        
        # Sequential connections for now
        connections = [(i, i + 1) for i in range(len(layers) - 1)]
        
        return ArchitectureSpec(
            name=name,
            layers=layers,
            connections=connections,
            input_dim=input_dim,
            output_dim=output_dim
        )
    
    async def _evaluate_population(
        self,
        population: List[ArchitectureSpec],
        reward_contract: RewardContract,
        train_data: Dict[str, jnp.ndarray],
        val_data: Dict[str, jnp.ndarray]
    ):
        """Evaluate a population of architectures in parallel."""
        self.logger.info(f"Evaluating population of {len(population)} architectures")
        
        # Use thread pool for parallel evaluation
        with ThreadPoolExecutor(max_workers=min(4, len(population))) as executor:
            futures = [
                executor.submit(
                    self._evaluate_architecture_sync,
                    arch, reward_contract, train_data, val_data
                )
                for arch in population
            ]
            
            # Collect results
            for future, arch in zip(futures, population):
                try:
                    metrics = future.result(timeout=300)  # 5 minute timeout
                    self._update_architecture_metrics(arch, metrics)
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {arch.name}: {e}")
                    # Assign poor performance for failed architectures
                    self._assign_failure_metrics(arch)
    
    def _evaluate_architecture_sync(
        self,
        arch_spec: ArchitectureSpec,
        reward_contract: RewardContract,
        train_data: Dict[str, jnp.ndarray],
        val_data: Dict[str, jnp.ndarray]
    ) -> Dict[str, float]:
        """Synchronous architecture evaluation."""
        return asyncio.run(
            self._evaluate_single_architecture(arch_spec, reward_contract, train_data, val_data)
        )
    
    async def _evaluate_single_architecture(
        self,
        arch_spec: ArchitectureSpec,
        reward_contract: RewardContract,
        train_data: Dict[str, jnp.ndarray],
        val_data: Dict[str, jnp.ndarray]
    ) -> Dict[str, float]:
        """Evaluate a single architecture's performance."""
        
        try:
            # Create model
            model = RewardModelArchitecture(spec=arch_spec)
            
            # Initialize parameters
            rng = random.PRNGKey(42)
            dummy_input = jnp.ones((1, arch_spec.input_dim))
            params = model.init(rng, dummy_input, training=False)
            
            # Count parameters
            param_count = sum(x.size for x in jax.tree_leaves(params))
            arch_spec.parameters_count = param_count
            
            # Estimate memory usage (rough approximation)
            arch_spec.memory_usage_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per param
            
            # Quick training evaluation (limited epochs for speed)
            metrics = await self._quick_training_evaluation(
                model, params, train_data, val_data, arch_spec
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating architecture {arch_spec.name}: {e}")
            return {
                "accuracy": 0.0,
                "convergence_speed": 1000,
                "inference_latency_ms": 1000.0,
                "training_time_hours": 24.0
            }
    
    async def _quick_training_evaluation(
        self,
        model: nn.Module,
        initial_params: Dict,
        train_data: Dict[str, jnp.ndarray],
        val_data: Dict[str, jnp.ndarray],
        arch_spec: ArchitectureSpec,
        max_epochs: int = 10
    ) -> Dict[str, float]:
        """Perform quick training evaluation for architecture scoring."""
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=3e-4)
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=initial_params,
            tx=optimizer
        )
        
        # Training metrics tracking
        train_losses = []
        val_accuracies = []
        training_start = time.time()
        
        # Training loop
        for epoch in range(max_epochs):
            
            # Training step
            batch_size = min(64, train_data["states"].shape[0])
            num_batches = train_data["states"].shape[0] // batch_size
            
            epoch_loss = 0.0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_states = train_data["states"][start_idx:end_idx]
                batch_rewards = train_data["rewards"][start_idx:end_idx]
                
                # Training step
                state, loss = self._training_step(state, batch_states, batch_rewards)
                epoch_loss += loss
            
            avg_loss = epoch_loss / max(1, num_batches)
            train_losses.append(avg_loss)
            
            # Validation step
            val_accuracy = self._validation_step(state, val_data)
            val_accuracies.append(val_accuracy)
            
            # Early convergence check
            if len(val_accuracies) >= 3:
                recent_improvement = val_accuracies[-1] - val_accuracies[-3]
                if recent_improvement < 0.001:  # Minimal improvement
                    break
        
        training_time_hours = (time.time() - training_start) / 3600.0
        
        # Calculate metrics
        final_accuracy = val_accuracies[-1] if val_accuracies else 0.0
        convergence_speed = len(val_accuracies)  # Epochs to convergence
        
        # Estimate inference latency (simplified)
        inference_latency_ms = arch_spec.parameters_count / 1e6 * 10  # Rough estimation
        
        return {
            "accuracy": final_accuracy,
            "convergence_speed": convergence_speed, 
            "inference_latency_ms": inference_latency_ms,
            "training_time_hours": training_time_hours
        }
    
    @jit
    def _training_step(self, state, batch_states, batch_rewards):
        """JIT-compiled training step."""
        
        def loss_fn(params):
            predictions = state.apply_fn(params, batch_states, training=True)
            predictions = predictions.squeeze(-1)  # Remove last dimension
            loss = jnp.mean((predictions - batch_rewards) ** 2)  # MSE loss
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    def _validation_step(self, state, val_data):
        """Compute validation accuracy."""
        val_states = val_data["states"]
        val_rewards = val_data["rewards"]
        
        predictions = state.apply_fn(state.params, val_states, training=False)
        predictions = predictions.squeeze(-1)
        
        # Calculate correlation as proxy for accuracy
        correlation = jnp.corrcoef(predictions, val_rewards)[0, 1]
        
        # Convert to accuracy-like metric [0, 1]
        accuracy = (correlation + 1) / 2  # Map [-1, 1] to [0, 1]
        
        return float(accuracy)
    
    def _update_architecture_metrics(self, arch_spec: ArchitectureSpec, metrics: Dict[str, float]):
        """Update architecture specification with evaluation metrics."""
        arch_spec.accuracy = metrics["accuracy"]
        arch_spec.convergence_speed = int(metrics["convergence_speed"])
        arch_spec.inference_latency_ms = metrics["inference_latency_ms"] 
        arch_spec.training_time_hours = metrics["training_time_hours"]
    
    def _assign_failure_metrics(self, arch_spec: ArchitectureSpec):
        """Assign poor performance metrics to failed architectures."""
        arch_spec.accuracy = 0.0
        arch_spec.convergence_speed = 1000
        arch_spec.inference_latency_ms = 1000.0
        arch_spec.training_time_hours = 24.0
        arch_spec.parameters_count = 1000000  # Assume large
        arch_spec.memory_usage_mb = 1000.0
    
    def _selection(self, population: List[ArchitectureSpec]) -> List[ArchitectureSpec]:
        """Select parents for next generation using tournament selection."""
        tournament_size = 3
        parents = []
        
        for _ in range(len(population)):
            # Tournament selection
            tournament = np.random.choice(population, size=tournament_size, replace=False)
            
            # Select based on Pareto dominance and crowding distance
            best = min(tournament, key=lambda x: (x.pareto_rank, -x.crowding_distance))
            parents.append(best)
        
        return parents
    
    def _crossover(self, parent1: ArchitectureSpec, parent2: ArchitectureSpec) -> Tuple[ArchitectureSpec, ArchitectureSpec]:
        """Create two children through crossover of parent architectures."""
        
        # Simple layer-wise crossover
        child1_layers = []
        child2_layers = []
        
        max_layers = max(len(parent1.layers), len(parent2.layers))
        
        for i in range(max_layers):
            if np.random.random() < 0.5:
                # Take from parent1
                if i < len(parent1.layers):
                    child1_layers.append(parent1.layers[i].copy())
                if i < len(parent2.layers):
                    child2_layers.append(parent2.layers[i].copy())
            else:
                # Take from parent2  
                if i < len(parent2.layers):
                    child1_layers.append(parent2.layers[i].copy())
                if i < len(parent1.layers):
                    child2_layers.append(parent1.layers[i].copy())
        
        # Ensure valid architectures
        child1_layers = self._ensure_valid_architecture(child1_layers)
        child2_layers = self._ensure_valid_architecture(child2_layers)
        
        child1 = ArchitectureSpec(
            name=f"child_{len(self.population)}",
            layers=child1_layers,
            connections=[(i, i + 1) for i in range(len(child1_layers) - 1)],
            input_dim=parent1.input_dim,
            output_dim=parent1.output_dim
        )
        
        child2 = ArchitectureSpec(
            name=f"child_{len(self.population) + 1}",
            layers=child2_layers,
            connections=[(i, i + 1) for i in range(len(child2_layers) - 1)], 
            input_dim=parent2.input_dim,
            output_dim=parent2.output_dim
        )
        
        return child1, child2
    
    def _mutate(self, architecture: ArchitectureSpec) -> ArchitectureSpec:
        """Mutate an architecture with small random changes."""
        mutation_rate = 0.1
        
        mutated_layers = []
        
        for layer in architecture.layers:
            mutated_layer = layer.copy()
            
            if np.random.random() < mutation_rate:
                # Mutate this layer
                layer_type = layer["type"]
                
                if layer_type == "linear":
                    # Mutate features
                    current_features = layer["features"]
                    if current_features > 1:  # Don't mutate output layer
                        multiplier = np.random.choice([0.5, 0.75, 1.25, 1.5])
                        new_features = int(current_features * multiplier)
                        new_features = max(16, min(2048, new_features))
                        mutated_layer["features"] = new_features
                        
                elif layer_type == "dropout":
                    # Mutate dropout rate
                    mutated_layer["rate"] = np.random.choice(self.search_space.dropout_rates)
                    
                elif layer_type == "activation":
                    # Mutate activation function
                    mutated_layer["function"] = np.random.choice(self.search_space.activation_functions)
            
            mutated_layers.append(mutated_layer)
        
        # Structural mutations (add/remove layers)
        if np.random.random() < 0.05:  # 5% chance of structural mutation
            if np.random.random() < 0.5 and len(mutated_layers) > 2:
                # Remove a layer (not first or last)
                remove_idx = np.random.randint(1, len(mutated_layers) - 1)
                mutated_layers.pop(remove_idx)
            else:
                # Add a layer
                insert_idx = np.random.randint(1, len(mutated_layers))
                new_layer = {
                    "type": np.random.choice(["linear", "activation", "normalization"]),
                    "features": np.random.choice(self.search_space.hidden_sizes)
                }
                mutated_layers.insert(insert_idx, new_layer)
        
        # Ensure valid architecture
        mutated_layers = self._ensure_valid_architecture(mutated_layers)
        
        mutated_architecture = ArchitectureSpec(
            name=f"mutated_{architecture.name}",
            layers=mutated_layers,
            connections=[(i, i + 1) for i in range(len(mutated_layers) - 1)],
            input_dim=architecture.input_dim,
            output_dim=architecture.output_dim
        )
        
        return mutated_architecture
    
    def _ensure_valid_architecture(self, layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure architecture is valid (has proper input/output)."""
        if not layers:
            # Minimal valid architecture
            return [
                {"type": "linear", "features": 256, "use_bias": True},
                {"type": "activation", "function": "relu"},
                {"type": "linear", "features": 1, "use_bias": True}
            ]
        
        # Ensure final layer outputs single reward value
        if layers[-1]["type"] != "linear" or layers[-1].get("features") != 1:
            layers[-1] = {"type": "linear", "features": 1, "use_bias": True}
        
        return layers
    
    def _environmental_selection(self, population: List[ArchitectureSpec]) -> List[ArchitectureSpec]:
        """Select next generation using non-dominated sorting and crowding distance."""
        
        # Non-dominated sorting
        fronts = self._non_dominated_sort(population)
        
        new_population = []
        front_idx = 0
        
        while len(new_population) < self.population_size and front_idx < len(fronts):
            front = fronts[front_idx]
            
            if len(new_population) + len(front) <= self.population_size:
                # Add entire front
                new_population.extend(front)
            else:
                # Add part of front based on crowding distance
                remaining_slots = self.population_size - len(new_population)
                front_with_distance = self._calculate_crowding_distance(front)
                
                # Sort by crowding distance (descending)
                front_with_distance.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front_with_distance[:remaining_slots])
            
            front_idx += 1
        
        return new_population[:self.population_size]
    
    def _non_dominated_sort(self, population: List[ArchitectureSpec]) -> List[List[ArchitectureSpec]]:
        """Perform non-dominated sorting on population."""
        
        # Initialize domination counts and dominated sets
        for arch in population:
            arch.dominates_count = 0
            arch.dominated_by = []
        
        # Calculate domination relationships
        for i, arch1 in enumerate(population):
            for j, arch2 in enumerate(population):
                if i != j:
                    if self._dominates(arch1, arch2):
                        arch1.dominates_count += 1
                    elif self._dominates(arch2, arch1):
                        arch1.dominated_by.append(arch2)
        
        # Create fronts
        fronts = []
        current_front = []
        
        # First front (non-dominated solutions)
        for arch in population:
            if len(arch.dominated_by) == 0:
                arch.pareto_rank = 0
                current_front.append(arch)
        
        if current_front:
            fronts.append(current_front)
        
        # Subsequent fronts
        front_idx = 0
        while front_idx < len(fronts):
            next_front = []
            
            for arch1 in fronts[front_idx]:
                for arch2 in population:
                    if arch1 != arch2 and arch1 in arch2.dominated_by:
                        arch2.dominated_by.remove(arch1)
                        if len(arch2.dominated_by) == 0:
                            arch2.pareto_rank = front_idx + 1
                            next_front.append(arch2)
            
            if next_front:
                fronts.append(next_front)
            
            front_idx += 1
        
        return fronts
    
    def _dominates(self, arch1: ArchitectureSpec, arch2: ArchitectureSpec) -> bool:
        """Check if arch1 dominates arch2 in multi-objective sense."""
        
        # Convert to objective values (lower is better for all objectives here)
        obj1 = self._get_objective_values(arch1)
        obj2 = self._get_objective_values(arch2)
        
        # Check domination: arch1 dominates arch2 if arch1 is better in at least one 
        # objective and not worse in any objective
        better_in_at_least_one = False
        worse_in_any = False
        
        for o1, o2 in zip(obj1, obj2):
            if o1 < o2:  # arch1 is better
                better_in_at_least_one = True
            elif o1 > o2:  # arch1 is worse
                worse_in_any = True
        
        return better_in_at_least_one and not worse_in_any
    
    def _get_objective_values(self, arch: ArchitectureSpec) -> List[float]:
        """Get objective values for an architecture (lower is better)."""
        
        # Convert metrics to minimization objectives
        objectives = []
        
        # Accuracy objective (minimize negative accuracy)
        if self.objectives.maximize_accuracy:
            objectives.append(-arch.accuracy * self.objectives.accuracy_weight)
        
        # Parameter count objective (minimize)
        if self.objectives.minimize_parameters:
            normalized_params = arch.parameters_count / 1e6  # Normalize to millions
            objectives.append(normalized_params * self.objectives.size_weight)
        
        # Latency objective (minimize)
        if self.objectives.minimize_latency:
            normalized_latency = arch.inference_latency_ms / 100.0  # Normalize to 100ms
            objectives.append(normalized_latency * self.objectives.speed_weight)
        
        # Training time objective (minimize)
        if self.objectives.minimize_training_time:
            normalized_time = arch.training_time_hours / 10.0  # Normalize to 10 hours
            objectives.append(normalized_time * self.objectives.efficiency_weight)
        
        # Convergence speed objective (minimize)
        if self.objectives.maximize_convergence_speed:
            normalized_convergence = arch.convergence_speed / 100.0  # Normalize to 100 epochs
            objectives.append(normalized_convergence * self.objectives.speed_weight)
        
        return objectives
    
    def _calculate_crowding_distance(self, front: List[ArchitectureSpec]) -> List[ArchitectureSpec]:
        """Calculate crowding distance for a front."""
        
        if len(front) <= 2:
            for arch in front:
                arch.crowding_distance = float('inf')
            return front
        
        # Initialize crowding distances
        for arch in front:
            arch.crowding_distance = 0.0
        
        # Get number of objectives
        num_objectives = len(self._get_objective_values(front[0]))
        
        # Calculate crowding distance for each objective
        for obj_idx in range(num_objectives):
            # Sort front by this objective
            front.sort(key=lambda x: self._get_objective_values(x)[obj_idx])
            
            # Boundary points have infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_values = [self._get_objective_values(arch)[obj_idx] for arch in front]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range > 0:
                # Calculate distance for intermediate points
                for i in range(1, len(front) - 1):
                    if front[i].crowding_distance != float('inf'):
                        distance = (obj_values[i + 1] - obj_values[i - 1]) / obj_range
                        front[i].crowding_distance += distance
        
        return front
    
    def _update_pareto_front(self, population: List[ArchitectureSpec]):
        """Update the global Pareto front."""
        
        fronts = self._non_dominated_sort(population)
        if fronts:
            self.pareto_front = fronts[0]  # First front is Pareto optimal
        else:
            self.pareto_front = []
    
    def _record_generation_stats(self):
        """Record statistics for current generation."""
        
        best_arch = min(self.population, key=lambda x: -x.accuracy)  # Best accuracy
        avg_accuracy = np.mean([arch.accuracy for arch in self.population])
        
        stats = {
            "generation": self.generation,
            "best_accuracy": best_arch.accuracy,
            "avg_accuracy": avg_accuracy,
            "pareto_front_size": len(self.pareto_front),
            "population_diversity": self._calculate_population_diversity(),
            "timestamp": time.time()
        }
        
        self.search_history.append(stats)
        
        self.logger.info(
            f"Gen {self.generation}: Best Accuracy={best_arch.accuracy:.3f}, "
            f"Avg Accuracy={avg_accuracy:.3f}, Pareto Front Size={len(self.pareto_front)}"
        )
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity metric."""
        if len(self.population) < 2:
            return 0.0
        
        # Simple diversity based on parameter count variance
        param_counts = [arch.parameters_count for arch in self.population]
        return float(np.std(param_counts) / (np.mean(param_counts) + 1e-8))
    
    def _should_stop_search(self) -> bool:
        """Check if search should stop early."""
        if len(self.search_history) < 10:
            return False
        
        # Check if improvement has stagnated
        recent_best = [stats["best_accuracy"] for stats in self.search_history[-5:]]
        improvement = max(recent_best) - min(recent_best)
        
        return improvement < 0.001  # Minimal improvement threshold
    
    def _save_search_results(self):
        """Save search results to files."""
        
        # Save Pareto front
        pareto_data = []
        for arch in self.pareto_front:
            pareto_data.append({
                "name": arch.name,
                "layers": arch.layers,
                "parameters_count": arch.parameters_count,
                "accuracy": arch.accuracy,
                "convergence_speed": arch.convergence_speed,
                "inference_latency_ms": arch.inference_latency_ms,
                "training_time_hours": arch.training_time_hours,
                "memory_usage_mb": arch.memory_usage_mb
            })
        
        pareto_file = self.output_dir / "pareto_front.json"
        with open(pareto_file, 'w') as f:
            json.dump(pareto_data, f, indent=2)
        
        # Save search history
        history_file = self.output_dir / "search_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.search_history, f, indent=2)
        
        # Save best architecture
        if self.pareto_front:
            best_arch = max(self.pareto_front, key=lambda x: x.accuracy)
            best_file = self.output_dir / "best_architecture.json"
            with open(best_file, 'w') as f:
                json.dump({
                    "name": best_arch.name,
                    "layers": best_arch.layers,
                    "connections": best_arch.connections,
                    "metrics": {
                        "accuracy": best_arch.accuracy,
                        "parameters_count": best_arch.parameters_count,
                        "inference_latency_ms": best_arch.inference_latency_ms,
                        "convergence_speed": best_arch.convergence_speed
                    }
                }, f, indent=2)
        
        self.logger.info(f"Search results saved to {self.output_dir}")
    
    async def _bayesian_optimization(self, reward_contract, train_data, val_data):
        """Bayesian optimization for architecture search."""
        # Placeholder for Bayesian optimization implementation
        self.logger.info("Bayesian optimization not yet implemented")
        return []
    
    async def _progressive_search(self, reward_contract, train_data, val_data):
        """Progressive search with complexity gradually increasing."""
        # Placeholder for progressive search implementation
        self.logger.info("Progressive search not yet implemented")
        return []


# Example usage
if __name__ == "__main__":
    
    async def main():
        # Create NAS engine
        nas = NeuralArchitectureSearch(
            population_size=20,
            generations=10,
            output_dir=Path("nas_results")
        )
        
        # Create dummy reward contract and data
        contract = RewardContract(name="test_contract", stakeholders={"user": 1.0})
        
        # Dummy training data
        train_data = {
            "states": jnp.ones((1000, 512)),
            "rewards": jnp.ones((1000,)) * 0.5
        }
        
        val_data = {
            "states": jnp.ones((200, 512)),
            "rewards": jnp.ones((200,)) * 0.5
        }
        
        # Run NAS
        pareto_front = await nas.search_optimal_architecture(
            contract, train_data, val_data, search_strategy="evolutionary"
        )
        
        print(f"Found {len(pareto_front)} optimal architectures")
        for arch in pareto_front:
            print(f"  {arch.name}: accuracy={arch.accuracy:.3f}, params={arch.parameters_count}")
    
    asyncio.run(main())
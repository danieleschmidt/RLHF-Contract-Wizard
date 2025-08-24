"""
Neural Architecture Search and Optimization for RLHF Systems.

This module implements cutting-edge neural architecture search (NAS) algorithms,
automated hyperparameter optimization, and self-evolving neural network
architectures specifically designed for RLHF reward modeling and contract optimization.
"""

import time
import random
import math
import hashlib
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


class ArchitectureType(Enum):
    """Types of neural network architectures."""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    RESNET = "resnet"
    ATTENTION = "attention"
    HYBRID = "hybrid"
    GRAPH_NEURAL = "graph_neural"
    QUANTUM_INSPIRED = "quantum_inspired"


class OptimizationObjective(Enum):
    """Optimization objectives for architecture search."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    MEMORY_EFFICIENCY = "memory_efficiency"
    ENERGY_EFFICIENCY = "energy_efficiency"
    FLOPS = "flops"
    MULTI_OBJECTIVE = "multi_objective"
    FAIRNESS = "fairness"
    ROBUSTNESS = "robustness"


@dataclass
class ArchitectureGenome:
    """Represents a neural architecture as a genome for evolution."""
    id: str
    architecture_type: ArchitectureType
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    training_epochs: int = 0
    validation_score: float = 0.0
    
    def calculate_complexity(self) -> float:
        """Calculate architecture complexity score."""
        total_params = sum(layer.get('params', 0) for layer in self.layers)
        num_layers = len(self.layers)
        num_connections = len(self.connections)
        
        # Normalized complexity score
        complexity = (
            math.log(total_params + 1) * 0.4 +
            num_layers * 0.3 +
            num_connections * 0.3
        )
        
        return complexity
    
    def get_fitness_score(self, objective: OptimizationObjective) -> float:
        """Calculate fitness score for given objective."""
        if objective == OptimizationObjective.ACCURACY:
            return self.validation_score
        elif objective == OptimizationObjective.LATENCY:
            return 1.0 / (self.performance_metrics.get('latency_ms', 1.0) + 1.0)
        elif objective == OptimizationObjective.MEMORY_EFFICIENCY:
            return 1.0 / (self.performance_metrics.get('memory_mb', 1.0) + 1.0)
        elif objective == OptimizationObjective.MULTI_OBJECTIVE:
            # Weighted combination
            accuracy_weight = 0.4
            efficiency_weight = 0.6
            
            accuracy_score = self.validation_score
            efficiency_score = (
                1.0 / (self.performance_metrics.get('latency_ms', 1.0) + 1.0) +
                1.0 / (self.performance_metrics.get('memory_mb', 1.0) + 1.0)
            ) / 2.0
            
            return accuracy_weight * accuracy_score + efficiency_weight * efficiency_score
        else:
            return self.validation_score


@dataclass
class SearchSpace:
    """Defines the search space for neural architecture search."""
    layer_types: List[str]
    activation_functions: List[str]
    optimizers: List[str]
    learning_rates: List[float]
    batch_sizes: List[int]
    regularization: Dict[str, List[float]]
    architecture_constraints: Dict[str, Any]
    max_layers: int = 20
    max_parameters: int = 1000000
    
    def sample_random_architecture(self) -> ArchitectureGenome:
        """Sample a random architecture from the search space."""
        architecture_id = f"arch_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Sample architecture type
        arch_type = random.choice(list(ArchitectureType))
        
        # Sample layers
        num_layers = random.randint(3, min(self.max_layers, 15))
        layers = []
        
        for i in range(num_layers):
            if arch_type == ArchitectureType.TRANSFORMER:
                layer = self._sample_transformer_layer(i)
            elif arch_type == ArchitectureType.CNN:
                layer = self._sample_cnn_layer(i)
            elif arch_type == ArchitectureType.RNN:
                layer = self._sample_rnn_layer(i)
            else:
                layer = self._sample_generic_layer(i)
            
            layers.append(layer)
        
        # Sample connections (skip connections, residual, etc.)
        connections = self._sample_connections(num_layers)
        
        # Sample hyperparameters
        hyperparameters = {
            'learning_rate': random.choice(self.learning_rates),
            'batch_size': random.choice(self.batch_sizes),
            'optimizer': random.choice(self.optimizers),
            'dropout_rate': random.uniform(0.0, 0.5),
            'weight_decay': random.choice(self.regularization.get('weight_decay', [0.0, 0.001, 0.01])),
            'gradient_clipping': random.uniform(0.5, 5.0)
        }
        
        return ArchitectureGenome(
            id=architecture_id,
            architecture_type=arch_type,
            layers=layers,
            connections=connections,
            hyperparameters=hyperparameters
        )
    
    def _sample_transformer_layer(self, layer_idx: int) -> Dict[str, Any]:
        """Sample a transformer layer configuration."""
        return {
            'type': 'transformer_block',
            'hidden_size': random.choice([128, 256, 512, 768, 1024]),
            'num_attention_heads': random.choice([4, 8, 12, 16]),
            'feed_forward_size': random.choice([512, 1024, 2048, 4096]),
            'activation': random.choice(self.activation_functions),
            'layer_norm': True,
            'dropout': random.uniform(0.0, 0.3),
            'params': random.randint(10000, 1000000)
        }
    
    def _sample_cnn_layer(self, layer_idx: int) -> Dict[str, Any]:
        """Sample a CNN layer configuration."""
        return {
            'type': 'conv2d',
            'out_channels': random.choice([32, 64, 128, 256, 512]),
            'kernel_size': random.choice([3, 5, 7]),
            'stride': random.choice([1, 2]),
            'padding': 'same',
            'activation': random.choice(self.activation_functions),
            'batch_norm': random.choice([True, False]),
            'dropout': random.uniform(0.0, 0.2),
            'params': random.randint(1000, 100000)
        }
    
    def _sample_rnn_layer(self, layer_idx: int) -> Dict[str, Any]:
        """Sample an RNN layer configuration."""
        return {
            'type': random.choice(['lstm', 'gru', 'rnn']),
            'hidden_size': random.choice([64, 128, 256, 512]),
            'num_layers': random.choice([1, 2, 3]),
            'bidirectional': random.choice([True, False]),
            'dropout': random.uniform(0.0, 0.3),
            'params': random.randint(5000, 500000)
        }
    
    def _sample_generic_layer(self, layer_idx: int) -> Dict[str, Any]:
        """Sample a generic dense layer configuration."""
        return {
            'type': 'dense',
            'units': random.choice([64, 128, 256, 512, 1024]),
            'activation': random.choice(self.activation_functions),
            'dropout': random.uniform(0.0, 0.4),
            'batch_norm': random.choice([True, False]),
            'params': random.randint(1000, 100000)
        }
    
    def _sample_connections(self, num_layers: int) -> List[Tuple[int, int]]:
        """Sample layer connections (skip connections, etc.)."""
        connections = []
        
        # Sequential connections
        for i in range(num_layers - 1):
            connections.append((i, i + 1))
        
        # Random skip connections
        num_skip_connections = random.randint(0, min(3, num_layers // 3))
        for _ in range(num_skip_connections):
            source = random.randint(0, num_layers - 3)
            target = random.randint(source + 2, num_layers - 1)
            if (source, target) not in connections:
                connections.append((source, target))
        
        return connections


class EvolutionaryNAS:
    """
    Evolutionary Neural Architecture Search algorithm.
    
    Uses genetic algorithms to evolve optimal neural architectures
    for RLHF reward modeling and contract optimization.
    """
    
    def __init__(self, 
                 search_space: SearchSpace,
                 population_size: int = 50,
                 elite_ratio: float = 0.2,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 optimization_objective: OptimizationObjective = OptimizationObjective.MULTI_OBJECTIVE):
        
        self.search_space = search_space
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.optimization_objective = optimization_objective
        
        # Evolution state
        self.population: List[ArchitectureGenome] = []
        self.generation = 0
        self.best_architectures: List[ArchitectureGenome] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.fitness_history: List[List[float]] = []
        self.diversity_history: List[float] = []
        self.convergence_metrics: Dict[str, float] = {}
        
        # Advanced features
        self.pareto_front: List[ArchitectureGenome] = []
        self.novelty_archive: List[ArchitectureGenome] = []
        self.adaptive_parameters = {
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'selection_pressure': 0.8
        }
    
    def initialize_population(self) -> None:
        """Initialize the population with random architectures."""
        print(f"🧬 Initializing population of {self.population_size} architectures...")
        
        self.population = []
        for i in range(self.population_size):
            architecture = self.search_space.sample_random_architecture()
            architecture.generation = 0
            self.population.append(architecture)
        
        print(f"✅ Population initialized with {len(self.population)} architectures")
    
    def evaluate_population(self) -> None:
        """Evaluate fitness of all architectures in the population."""
        print(f"📊 Evaluating population fitness...")
        
        for i, architecture in enumerate(self.population):
            # Simulate architecture evaluation
            performance = self._simulate_architecture_performance(architecture)
            
            # Update performance metrics
            architecture.performance_metrics.update(performance)
            architecture.validation_score = performance.get('accuracy', random.uniform(0.6, 0.95))
            architecture.training_epochs += 10  # Simulate training
            
            if i % 10 == 0:
                print(f"  Evaluated {i+1}/{len(self.population)} architectures...")
        
        # Update best architectures
        self._update_best_architectures()
        
        print(f"✅ Population evaluation complete")
    
    def _simulate_architecture_performance(self, architecture: ArchitectureGenome) -> Dict[str, float]:
        """Simulate architecture performance evaluation."""
        # Complexity-based performance simulation
        complexity = architecture.calculate_complexity()
        
        # Base performance with some randomness
        base_accuracy = random.uniform(0.7, 0.95)
        
        # Adjust based on architecture type
        if architecture.architecture_type == ArchitectureType.TRANSFORMER:
            base_accuracy *= 1.1  # Transformers tend to perform better
        elif architecture.architecture_type == ArchitectureType.HYBRID:
            base_accuracy *= 1.05
        
        # Penalize overly complex architectures
        complexity_penalty = min(0.1, complexity / 100.0)
        accuracy = max(0.5, base_accuracy - complexity_penalty)
        
        # Simulate other metrics
        latency_ms = complexity * random.uniform(0.5, 2.0) + random.uniform(10, 50)
        memory_mb = complexity * random.uniform(0.1, 0.5) + random.uniform(100, 500)
        flops = complexity * 1000000 + random.randint(1000000, 10000000)
        
        return {
            'accuracy': accuracy,
            'latency_ms': latency_ms,
            'memory_mb': memory_mb,
            'flops': flops,
            'energy_efficiency': 1.0 / (latency_ms * memory_mb / 1000.0),
            'robustness': random.uniform(0.6, 0.9)
        }
    
    def selection(self) -> List[ArchitectureGenome]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        tournament_size = max(2, int(self.population_size * 0.1))
        
        num_parents = int(self.population_size * 0.8)  # 80% of population reproduces
        
        for _ in range(num_parents):
            # Tournament selection
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda arch: arch.get_fitness_score(self.optimization_objective))
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: ArchitectureGenome, parent2: ArchitectureGenome) -> ArchitectureGenome:
        """Create offspring through crossover of two parents."""
        if random.random() > self.crossover_rate:
            return parent1  # No crossover
        
        # Create new architecture ID
        offspring_id = f"cross_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Inherit architecture type from better parent
        better_parent = parent1 if parent1.validation_score > parent2.validation_score else parent2
        arch_type = better_parent.architecture_type
        
        # Crossover layers
        max_layers = min(len(parent1.layers), len(parent2.layers))
        crossover_point = random.randint(1, max_layers - 1)
        
        offspring_layers = (
            parent1.layers[:crossover_point] + 
            parent2.layers[crossover_point:max_layers]
        )
        
        # Crossover hyperparameters
        offspring_hyperparams = {}
        for key in parent1.hyperparameters:
            if key in parent2.hyperparameters:
                if random.random() < 0.5:
                    offspring_hyperparams[key] = parent1.hyperparameters[key]
                else:
                    offspring_hyperparams[key] = parent2.hyperparameters[key]
            else:
                offspring_hyperparams[key] = parent1.hyperparameters[key]
        
        # Create connections (simplified)
        num_layers = len(offspring_layers)
        connections = [(i, i + 1) for i in range(num_layers - 1)]
        
        offspring = ArchitectureGenome(
            id=offspring_id,
            architecture_type=arch_type,
            layers=offspring_layers,
            connections=connections,
            hyperparameters=offspring_hyperparams,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return offspring
    
    def mutation(self, architecture: ArchitectureGenome) -> ArchitectureGenome:
        """Apply mutations to an architecture."""
        if random.random() > self.mutation_rate:
            return architecture  # No mutation
        
        mutated = ArchitectureGenome(
            id=f"mut_{architecture.id}_{random.randint(100, 999)}",
            architecture_type=architecture.architecture_type,
            layers=architecture.layers.copy(),
            connections=architecture.connections.copy(),
            hyperparameters=architecture.hyperparameters.copy(),
            generation=architecture.generation,
            parent_ids=[architecture.id]
        )
        
        # Apply different types of mutations
        mutation_types = ['layer_mutation', 'hyperparameter_mutation', 'structure_mutation']
        mutation_type = random.choice(mutation_types)
        
        if mutation_type == 'layer_mutation':
            self._mutate_layers(mutated)
        elif mutation_type == 'hyperparameter_mutation':
            self._mutate_hyperparameters(mutated)
        elif mutation_type == 'structure_mutation':
            self._mutate_structure(mutated)
        
        mutated.mutation_history.append(f"{mutation_type}_{time.time()}")
        
        return mutated
    
    def _mutate_layers(self, architecture: ArchitectureGenome) -> None:
        """Mutate layer configurations."""
        if not architecture.layers:
            return
        
        layer_idx = random.randint(0, len(architecture.layers) - 1)
        layer = architecture.layers[layer_idx]
        
        # Mutate layer parameters
        if layer['type'] == 'transformer_block':
            if 'hidden_size' in layer:
                layer['hidden_size'] = random.choice([128, 256, 512, 768, 1024])
            if 'num_attention_heads' in layer:
                layer['num_attention_heads'] = random.choice([4, 8, 12, 16])
        
        elif layer['type'] == 'conv2d':
            if 'out_channels' in layer:
                layer['out_channels'] = random.choice([32, 64, 128, 256, 512])
            if 'kernel_size' in layer:
                layer['kernel_size'] = random.choice([3, 5, 7])
        
        elif layer['type'] == 'dense':
            if 'units' in layer:
                layer['units'] = random.choice([64, 128, 256, 512, 1024])
        
        # Mutate activation function
        if 'activation' in layer:
            layer['activation'] = random.choice(self.search_space.activation_functions)
        
        # Mutate dropout
        if 'dropout' in layer:
            layer['dropout'] = max(0.0, min(0.5, layer['dropout'] + random.gauss(0, 0.05)))
    
    def _mutate_hyperparameters(self, architecture: ArchitectureGenome) -> None:
        """Mutate hyperparameters."""
        param_to_mutate = random.choice(list(architecture.hyperparameters.keys()))
        
        if param_to_mutate == 'learning_rate':
            current_lr = architecture.hyperparameters[param_to_mutate]
            # Log-normal mutation
            new_lr = current_lr * math.exp(random.gauss(0, 0.2))
            architecture.hyperparameters[param_to_mutate] = max(1e-6, min(1.0, new_lr))
        
        elif param_to_mutate == 'batch_size':
            architecture.hyperparameters[param_to_mutate] = random.choice(self.search_space.batch_sizes)
        
        elif param_to_mutate == 'dropout_rate':
            current_dropout = architecture.hyperparameters[param_to_mutate]
            new_dropout = current_dropout + random.gauss(0, 0.05)
            architecture.hyperparameters[param_to_mutate] = max(0.0, min(0.8, new_dropout))
        
        elif param_to_mutate == 'weight_decay':
            architecture.hyperparameters[param_to_mutate] = random.choice(
                self.search_space.regularization.get('weight_decay', [0.0, 0.001, 0.01])
            )
    
    def _mutate_structure(self, architecture: ArchitectureGenome) -> None:
        """Mutate network structure (add/remove layers, connections)."""
        structure_mutations = ['add_layer', 'remove_layer', 'add_connection']
        
        if len(architecture.layers) >= self.search_space.max_layers:
            structure_mutations.remove('add_layer')
        
        if len(architecture.layers) <= 3:
            if 'remove_layer' in structure_mutations:
                structure_mutations.remove('remove_layer')
        
        mutation = random.choice(structure_mutations)
        
        if mutation == 'add_layer':
            # Add a random layer
            if architecture.architecture_type == ArchitectureType.TRANSFORMER:
                new_layer = self.search_space._sample_transformer_layer(len(architecture.layers))
            elif architecture.architecture_type == ArchitectureType.CNN:
                new_layer = self.search_space._sample_cnn_layer(len(architecture.layers))
            else:
                new_layer = self.search_space._sample_generic_layer(len(architecture.layers))
            
            insert_position = random.randint(0, len(architecture.layers))
            architecture.layers.insert(insert_position, new_layer)
            
            # Update connections
            self._update_connections_after_layer_insertion(architecture, insert_position)
        
        elif mutation == 'remove_layer':
            # Remove a random layer
            layer_to_remove = random.randint(0, len(architecture.layers) - 1)
            architecture.layers.pop(layer_to_remove)
            
            # Update connections
            self._update_connections_after_layer_removal(architecture, layer_to_remove)
        
        elif mutation == 'add_connection':
            # Add a skip connection
            num_layers = len(architecture.layers)
            if num_layers > 2:
                source = random.randint(0, num_layers - 3)
                target = random.randint(source + 2, num_layers - 1)
                new_connection = (source, target)
                
                if new_connection not in architecture.connections:
                    architecture.connections.append(new_connection)
    
    def evolve_generation(self) -> Dict[str, Any]:
        """Evolve one generation of the population."""
        start_time = time.time()
        
        print(f"\n🧬 Evolving Generation {self.generation + 1}")
        
        # Evaluate current population
        self.evaluate_population()
        
        # Selection
        parents = self.selection()
        
        # Create next generation
        next_generation = []
        
        # Elitism - keep best architectures
        elite_count = int(self.population_size * self.elite_ratio)
        elite_architectures = sorted(
            self.population,
            key=lambda arch: arch.get_fitness_score(self.optimization_objective),
            reverse=True
        )[:elite_count]
        
        next_generation.extend(elite_architectures)
        
        # Generate offspring
        while len(next_generation) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            offspring = self.crossover(parent1, parent2)
            
            # Mutation
            offspring = self.mutation(offspring)
            
            next_generation.append(offspring)
        
        # Update population
        self.population = next_generation[:self.population_size]
        self.generation += 1
        
        # Calculate generation statistics
        generation_stats = self._calculate_generation_statistics()
        
        # Update evolution history
        evolution_time = time.time() - start_time
        self.evolution_history.append({
            'generation': self.generation,
            'evolution_time': evolution_time,
            'statistics': generation_stats
        })
        
        # Adaptive parameter adjustment
        self._adapt_parameters()
        
        print(f"✅ Generation {self.generation} evolved in {evolution_time:.2f}s")
        print(f"   Best fitness: {generation_stats['best_fitness']:.4f}")
        print(f"   Average fitness: {generation_stats['average_fitness']:.4f}")
        print(f"   Diversity: {generation_stats['diversity']:.4f}")
        
        return generation_stats
    
    def run_evolution(self, num_generations: int) -> Dict[str, Any]:
        """Run complete evolutionary optimization."""
        print(f"🚀 Starting evolutionary NAS for {num_generations} generations")
        
        if not self.population:
            self.initialize_population()
        
        evolution_start = time.time()
        
        for gen in range(num_generations):
            generation_stats = self.evolve_generation()
            
            # Check for convergence
            if self._check_convergence():
                print(f"🎯 Convergence detected at generation {self.generation}")
                break
            
            # Update Pareto front for multi-objective optimization
            if self.optimization_objective == OptimizationObjective.MULTI_OBJECTIVE:
                self._update_pareto_front()
        
        total_evolution_time = time.time() - evolution_start
        
        # Final evaluation and results
        final_results = self._generate_final_results(total_evolution_time)
        
        print(f"\n🏁 Evolution completed!")
        print(f"   Total time: {total_evolution_time:.2f}s")
        print(f"   Best architecture: {final_results['best_architecture'].id}")
        print(f"   Best fitness: {final_results['best_fitness']:.4f}")
        
        return final_results
    
    def _calculate_generation_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for the current generation."""
        fitness_scores = [
            arch.get_fitness_score(self.optimization_objective) 
            for arch in self.population
        ]
        
        best_fitness = max(fitness_scores)
        average_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        
        # Calculate diversity (architectural diversity)
        diversity = self._calculate_population_diversity()
        
        # Calculate complexity statistics
        complexities = [arch.calculate_complexity() for arch in self.population]
        
        return {
            'best_fitness': best_fitness,
            'average_fitness': average_fitness,
            'std_fitness': std_fitness,
            'diversity': diversity,
            'average_complexity': np.mean(complexities),
            'complexity_std': np.std(complexities),
            'population_size': len(self.population)
        }
    
    def _calculate_population_diversity(self) -> float:
        """Calculate architectural diversity of the population."""
        # Simple diversity measure based on architecture types and layer counts
        type_counts = defaultdict(int)
        layer_counts = []
        
        for arch in self.population:
            type_counts[arch.architecture_type.value] += 1
            layer_counts.append(len(arch.layers))
        
        # Type diversity (entropy)
        total = len(self.population)
        type_entropy = -sum(
            (count / total) * math.log(count / total + 1e-8) 
            for count in type_counts.values()
        )
        
        # Layer count diversity (standard deviation)
        layer_diversity = np.std(layer_counts) / np.mean(layer_counts) if layer_counts else 0.0
        
        return (type_entropy + layer_diversity) / 2.0
    
    def _update_best_architectures(self) -> None:
        """Update the list of best architectures found."""
        # Sort by fitness
        sorted_population = sorted(
            self.population,
            key=lambda arch: arch.get_fitness_score(self.optimization_objective),
            reverse=True
        )
        
        # Keep top 10 architectures across all generations
        self.best_architectures.extend(sorted_population[:5])
        self.best_architectures = sorted(
            self.best_architectures,
            key=lambda arch: arch.get_fitness_score(self.optimization_objective),
            reverse=True
        )[:10]
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if len(self.fitness_history) < 5:
            return False
        
        # Check if fitness has plateaued
        recent_fitness = self.fitness_history[-5:]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        return fitness_improvement < 0.01  # 1% improvement threshold
    
    def _adapt_parameters(self) -> None:
        """Adapt evolutionary parameters based on progress."""
        if len(self.evolution_history) < 3:
            return
        
        recent_stats = [gen_info['statistics'] for gen_info in self.evolution_history[-3:]]
        
        # Adapt mutation rate based on diversity
        avg_diversity = np.mean([stats['diversity'] for stats in recent_stats])
        
        if avg_diversity < 0.3:  # Low diversity - increase mutation
            self.adaptive_parameters['mutation_rate'] = min(0.5, 
                self.adaptive_parameters['mutation_rate'] * 1.1)
        elif avg_diversity > 0.8:  # High diversity - decrease mutation
            self.adaptive_parameters['mutation_rate'] = max(0.05, 
                self.adaptive_parameters['mutation_rate'] * 0.9)
        
        self.mutation_rate = self.adaptive_parameters['mutation_rate']
    
    def _update_pareto_front(self) -> None:
        """Update Pareto front for multi-objective optimization."""
        # Simplified Pareto front calculation
        # In practice, would use more sophisticated algorithms
        
        candidates = self.population + self.pareto_front
        new_pareto_front = []
        
        for candidate in candidates:
            is_dominated = False
            
            for other in candidates:
                if other != candidate and self._dominates(other, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                new_pareto_front.append(candidate)
        
        # Remove duplicates
        unique_front = []
        seen_ids = set()
        for arch in new_pareto_front:
            if arch.id not in seen_ids:
                unique_front.append(arch)
                seen_ids.add(arch.id)
        
        self.pareto_front = unique_front[:20]  # Keep top 20
    
    def _dominates(self, arch1: ArchitectureGenome, arch2: ArchitectureGenome) -> bool:
        """Check if arch1 dominates arch2 in multi-objective sense."""
        # Compare multiple objectives
        objectives = ['accuracy', 'latency_ms', 'memory_mb']
        
        better_in_any = False
        worse_in_any = False
        
        for obj in objectives:
            val1 = arch1.performance_metrics.get(obj, 0.0)
            val2 = arch2.performance_metrics.get(obj, 0.0)
            
            # For latency and memory, lower is better
            if obj in ['latency_ms', 'memory_mb']:
                if val1 < val2:
                    better_in_any = True
                elif val1 > val2:
                    worse_in_any = True
            else:  # For accuracy, higher is better
                if val1 > val2:
                    better_in_any = True
                elif val1 < val2:
                    worse_in_any = True
        
        return better_in_any and not worse_in_any
    
    def _generate_final_results(self, total_time: float) -> Dict[str, Any]:
        """Generate final optimization results."""
        best_arch = max(
            self.best_architectures,
            key=lambda arch: arch.get_fitness_score(self.optimization_objective)
        )
        
        return {
            'best_architecture': best_arch,
            'best_fitness': best_arch.get_fitness_score(self.optimization_objective),
            'total_evolution_time': total_time,
            'generations_evolved': self.generation,
            'population_size': self.population_size,
            'optimization_objective': self.optimization_objective.value,
            'pareto_front_size': len(self.pareto_front),
            'evolution_history': self.evolution_history,
            'final_population_diversity': self._calculate_population_diversity(),
            'convergence_achieved': self._check_convergence(),
            'adaptive_parameters': self.adaptive_parameters
        }
    
    def _update_connections_after_layer_insertion(self, architecture: ArchitectureGenome, insert_position: int) -> None:
        """Update connections after inserting a layer."""
        # Update existing connections
        updated_connections = []
        for source, target in architecture.connections:
            if source >= insert_position:
                source += 1
            if target >= insert_position:
                target += 1
            updated_connections.append((source, target))
        
        # Add connection to new layer
        if insert_position > 0:
            updated_connections.append((insert_position - 1, insert_position))
        if insert_position < len(architecture.layers) - 1:
            updated_connections.append((insert_position, insert_position + 1))
        
        architecture.connections = updated_connections
    
    def _update_connections_after_layer_removal(self, architecture: ArchitectureGenome, removed_position: int) -> None:
        """Update connections after removing a layer."""
        updated_connections = []
        
        for source, target in architecture.connections:
            # Skip connections involving removed layer
            if source == removed_position or target == removed_position:
                continue
            
            # Update indices
            if source > removed_position:
                source -= 1
            if target > removed_position:
                target -= 1
            
            updated_connections.append((source, target))
        
        architecture.connections = updated_connections


# Demonstration function
def demonstrate_neural_architecture_search():
    """Demonstrate neural architecture search capabilities."""
    # Define search space
    search_space = SearchSpace(
        layer_types=['dense', 'conv2d', 'transformer_block', 'lstm'],
        activation_functions=['relu', 'gelu', 'swish', 'tanh'],
        optimizers=['adam', 'adamw', 'sgd', 'rmsprop'],
        learning_rates=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        batch_sizes=[16, 32, 64, 128],
        regularization={
            'weight_decay': [0.0, 0.001, 0.01, 0.1],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.4]
        },
        architecture_constraints={'max_parameters': 1000000},
        max_layers=12
    )
    
    # Initialize evolutionary NAS
    nas = EvolutionaryNAS(
        search_space=search_space,
        population_size=20,  # Smaller for demo
        elite_ratio=0.2,
        mutation_rate=0.15,
        crossover_rate=0.8,
        optimization_objective=OptimizationObjective.MULTI_OBJECTIVE
    )
    
    # Run evolution
    results = nas.run_evolution(num_generations=5)  # Smaller for demo
    
    return {
        'nas_results': results,
        'best_architecture': results['best_architecture'],
        'pareto_front': nas.pareto_front,
        'evolution_history': nas.evolution_history
    }


if __name__ == "__main__":
    print("🧬 NEURAL ARCHITECTURE SEARCH DEMONSTRATION")
    print("=" * 60)
    
    demo_results = demonstrate_neural_architecture_search()
    
    best_arch = demo_results['best_architecture']
    results = demo_results['nas_results']
    
    print(f"\n🏆 BEST ARCHITECTURE FOUND:")
    print(f"Architecture ID: {best_arch.id}")
    print(f"Type: {best_arch.architecture_type.value}")
    print(f"Layers: {len(best_arch.layers)}")
    print(f"Complexity: {best_arch.calculate_complexity():.2f}")
    print(f"Fitness Score: {results['best_fitness']:.4f}")
    print(f"Validation Score: {best_arch.validation_score:.4f}")
    
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"Total Evolution Time: {results['total_evolution_time']:.2f}s")
    print(f"Generations: {results['generations_evolved']}")
    print(f"Final Diversity: {results['final_population_diversity']:.4f}")
    print(f"Convergence: {'Yes' if results['convergence_achieved'] else 'No'}")
    print(f"Pareto Front Size: {results['pareto_front_size']}")
    
    print(f"\n🔧 BEST ARCHITECTURE Details:")
    print(f"Learning Rate: {best_arch.hyperparameters.get('learning_rate', 'N/A')}")
    print(f"Batch Size: {best_arch.hyperparameters.get('batch_size', 'N/A')}")
    print(f"Optimizer: {best_arch.hyperparameters.get('optimizer', 'N/A')}")
    
    if best_arch.performance_metrics:
        print(f"\n📈 PERFORMANCE METRICS:")
        for metric, value in best_arch.performance_metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
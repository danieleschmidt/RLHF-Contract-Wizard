#!/usr/bin/env python3
"""
Advanced Optimization Module for RLHF Contract Wizard

Implements cutting-edge optimization algorithms for reward function learning,
contract verification, and quantum-inspired task planning with adaptive
performance tuning and multi-objective optimization.
"""

import time
import math
import random
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from collections import defaultdict, deque


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"
    QUANTUM_ANNEALING = "quantum_annealing"
    HYBRID = "hybrid"


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    CONSTRAINED = "constrained"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    strategy: OptimizationStrategy = OptimizationStrategy.ADAM
    objective_type: ObjectiveType = ObjectiveType.MAXIMIZE
    max_iterations: int = 1000
    learning_rate: float = 0.001
    tolerance: float = 1e-6
    population_size: int = 100  # For evolutionary algorithms
    elite_ratio: float = 0.2
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    parallel_workers: int = 4
    adaptive_learning: bool = True
    early_stopping: bool = True
    patience: int = 50
    gradient_clipping: float = 1.0
    regularization_strength: float = 0.01


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    optimal_parameters: jnp.ndarray
    optimal_value: float
    iterations: int
    convergence_time: float
    converged: bool
    optimization_history: List[float]
    gradient_norms: List[float]
    constraint_violations: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveOptimizer:
    """
    Advanced adaptive optimizer with multiple strategies and auto-tuning.
    
    Features:
    - Multiple optimization algorithms with automatic selection
    - Adaptive hyperparameter tuning during optimization
    - Multi-objective optimization with Pareto frontiers
    - Constraint handling and penalty methods
    - Parallel evaluation for population-based methods
    - Real-time convergence monitoring and early stopping
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.history: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        
        # Adaptive parameters
        self.current_lr = config.learning_rate
        self.momentum = 0.9
        self.beta1 = 0.9  # Adam parameter
        self.beta2 = 0.999  # Adam parameter
        self.epsilon = 1e-8
        
        # State variables
        self.iteration = 0
        self.stagnation_counter = 0
        self.best_value = float('-inf') if config.objective_type == ObjectiveType.MAXIMIZE else float('inf')
        
        # Population-based algorithm state
        self.population: Optional[jnp.ndarray] = None
        self.population_fitness: Optional[jnp.ndarray] = None
        
        # Gradient-based algorithm state
        self.velocity: Optional[jnp.ndarray] = None
        self.m_t: Optional[jnp.ndarray] = None  # Adam first moment
        self.v_t: Optional[jnp.ndarray] = None  # Adam second moment
    
    def optimize(
        self,
        objective_fn: Callable[[jnp.ndarray], float],
        initial_params: jnp.ndarray,
        bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        constraints: Optional[List[Callable]] = None
    ) -> OptimizationResult:
        """
        Run optimization with the configured strategy.
        
        Args:
            objective_fn: Function to optimize
            initial_params: Starting parameter values
            bounds: Optional parameter bounds (min, max)
            constraints: Optional constraint functions
            
        Returns:
            Optimization result with optimal parameters and metadata
        """
        start_time = time.time()
        
        # Reset state
        self._reset_state(initial_params)
        
        # Choose optimization strategy
        if self.config.strategy == OptimizationStrategy.ADAM:
            result = self._optimize_adam(objective_fn, initial_params, bounds, constraints)
        elif self.config.strategy == OptimizationStrategy.EVOLUTIONARY:
            result = self._optimize_evolutionary(objective_fn, initial_params, bounds, constraints)
        elif self.config.strategy == OptimizationStrategy.QUANTUM_ANNEALING:
            result = self._optimize_quantum_annealing(objective_fn, initial_params, bounds, constraints)
        elif self.config.strategy == OptimizationStrategy.HYBRID:
            result = self._optimize_hybrid(objective_fn, initial_params, bounds, constraints)
        else:
            result = self._optimize_gradient_descent(objective_fn, initial_params, bounds, constraints)
        
        result.convergence_time = time.time() - start_time
        
        # Update best result
        if self.best_result is None or self._is_better(result.optimal_value, self.best_result.optimal_value):
            self.best_result = result
        
        self.history.append(result)
        return result
    
    def _reset_state(self, initial_params: jnp.ndarray):
        """Reset optimizer state for new optimization run."""
        self.iteration = 0
        self.stagnation_counter = 0
        self.current_lr = self.config.learning_rate
        
        param_shape = initial_params.shape
        self.velocity = jnp.zeros(param_shape)
        self.m_t = jnp.zeros(param_shape)
        self.v_t = jnp.zeros(param_shape)
        
        if self.config.objective_type == ObjectiveType.MAXIMIZE:
            self.best_value = float('-inf')
        else:
            self.best_value = float('inf')
    
    def _is_better(self, new_value: float, current_best: float) -> bool:
        """Check if new value is better than current best."""
        if self.config.objective_type == ObjectiveType.MAXIMIZE:
            return new_value > current_best
        else:
            return new_value < current_best
    
    def _optimize_adam(
        self,
        objective_fn: Callable,
        initial_params: jnp.ndarray,
        bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
        constraints: Optional[List[Callable]]
    ) -> OptimizationResult:
        """Optimize using Adam algorithm with adaptive learning rate."""
        
        params = initial_params.copy()
        history = []
        gradient_norms = []
        constraint_violations = []
        
        # JIT compile gradient computation
        grad_fn = jit(grad(objective_fn))
        
        for iteration in range(self.config.max_iterations):
            self.iteration = iteration
            
            # Compute objective and gradient
            current_value = objective_fn(params)
            current_grad = grad_fn(params)
            
            # Apply gradient clipping
            grad_norm = jnp.linalg.norm(current_grad)
            if grad_norm > self.config.gradient_clipping:
                current_grad = current_grad * (self.config.gradient_clipping / grad_norm)
            
            # Adam updates
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * current_grad
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (current_grad ** 2)
            
            # Bias correction
            m_hat = self.m_t / (1 - self.beta1 ** (iteration + 1))
            v_hat = self.v_t / (1 - self.beta2 ** (iteration + 1))
            
            # Parameter update
            update = self.current_lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
            
            if self.config.objective_type == ObjectiveType.MAXIMIZE:
                params = params + update
            else:
                params = params - update
            
            # Apply bounds if specified
            if bounds is not None:
                params = jnp.clip(params, bounds[0], bounds[1])
            
            # Check constraints
            constraint_violation = 0.0
            if constraints:
                for constraint in constraints:
                    violation = max(0, -constraint(params))  # Assume constraint(x) >= 0
                    constraint_violation += violation
                    if violation > 0:
                        # Penalty method: adjust objective
                        current_value -= violation * 1000
            
            # Record progress
            history.append(float(current_value))
            gradient_norms.append(float(grad_norm))
            constraint_violations.append(float(constraint_violation))
            
            # Adaptive learning rate
            if self.config.adaptive_learning:
                if iteration > 0 and self._is_better(current_value, history[-2]):
                    self.current_lr *= 1.05  # Increase if improving
                else:
                    self.current_lr *= 0.95  # Decrease if not improving
                
                self.current_lr = jnp.clip(self.current_lr, 1e-6, 1.0)
            
            # Early stopping check
            if self._is_better(current_value, self.best_value):
                self.best_value = current_value
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            if (self.config.early_stopping and 
                self.stagnation_counter >= self.config.patience):
                break
            
            # Convergence check
            if (iteration > 0 and 
                abs(history[-1] - history[-2]) < self.config.tolerance):
                break
        
        return OptimizationResult(
            optimal_parameters=params,
            optimal_value=float(current_value),
            iterations=iteration + 1,
            convergence_time=0.0,  # Will be set by caller
            converged=self.stagnation_counter < self.config.patience,
            optimization_history=history,
            gradient_norms=gradient_norms,
            constraint_violations=constraint_violations,
            metadata={
                "algorithm": "adam",
                "final_learning_rate": float(self.current_lr),
                "stagnation_counter": self.stagnation_counter
            }
        )
    
    def _optimize_evolutionary(
        self,
        objective_fn: Callable,
        initial_params: jnp.ndarray,
        bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
        constraints: Optional[List[Callable]]
    ) -> OptimizationResult:
        """Optimize using evolutionary algorithm with parallel evaluation."""
        
        param_dim = initial_params.size
        pop_size = self.config.population_size
        elite_size = max(1, int(pop_size * self.config.elite_ratio))
        
        # Initialize population
        if bounds is not None:
            lower, upper = bounds
            population = np.random.uniform(
                lower, upper, size=(pop_size, param_dim)
            )
        else:
            population = np.random.normal(
                initial_params, 0.1, size=(pop_size, param_dim)
            )
        
        history = []
        best_individual = None
        best_fitness = float('-inf') if self.config.objective_type == ObjectiveType.MAXIMIZE else float('inf')
        
        # Parallel evaluation function
        def evaluate_individual(individual):
            fitness = objective_fn(jnp.array(individual))
            
            # Apply constraint penalties
            if constraints:
                penalty = 0.0
                for constraint in constraints:
                    violation = max(0, -constraint(jnp.array(individual)))
                    penalty += violation * 1000
                fitness -= penalty
            
            return fitness
        
        for generation in range(self.config.max_iterations):
            # Parallel fitness evaluation
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                fitness_values = list(executor.map(evaluate_individual, population))
            
            fitness_array = np.array(fitness_values)
            
            # Update best solution
            if self.config.objective_type == ObjectiveType.MAXIMIZE:
                gen_best_idx = np.argmax(fitness_array)
                gen_best_fitness = fitness_array[gen_best_idx]
                if gen_best_fitness > best_fitness:
                    best_fitness = gen_best_fitness
                    best_individual = population[gen_best_idx].copy()
            else:
                gen_best_idx = np.argmin(fitness_array)
                gen_best_fitness = fitness_array[gen_best_idx]
                if gen_best_fitness < best_fitness:
                    best_fitness = gen_best_fitness
                    best_individual = population[gen_best_idx].copy()
            
            history.append(float(gen_best_fitness))
            
            # Selection (tournament selection)
            if self.config.objective_type == ObjectiveType.MAXIMIZE:
                elite_indices = np.argsort(fitness_array)[-elite_size:]
            else:
                elite_indices = np.argsort(fitness_array)[:elite_size]
            
            # Create next generation
            new_population = []
            
            # Keep elite individuals
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Select parents (tournament selection)
                parent1 = self._tournament_selection(population, fitness_array, 3)
                parent2 = self._tournament_selection(population, fitness_array, 3)
                
                # Crossover
                if random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutate(child1, bounds)
                child2 = self._mutate(child2, bounds)
                
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:pop_size])
            
            # Early stopping check
            if (generation > self.config.patience and 
                len(set(history[-self.config.patience:])) == 1):
                break
        
        return OptimizationResult(
            optimal_parameters=jnp.array(best_individual),
            optimal_value=best_fitness,
            iterations=generation + 1,
            convergence_time=0.0,
            converged=True,
            optimization_history=history,
            gradient_norms=[],
            constraint_violations=[],
            metadata={
                "algorithm": "evolutionary",
                "population_size": pop_size,
                "elite_size": elite_size,
                "final_generation": generation + 1
            }
        )
    
    def _tournament_selection(self, population, fitness, tournament_size):
        """Select individual via tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        
        if self.config.objective_type == ObjectiveType.MAXIMIZE:
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        else:
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        
        return population[winner_idx]
    
    def _crossover(self, parent1, parent2):
        """Perform single-point crossover."""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def _mutate(self, individual, bounds):
        """Apply Gaussian mutation."""
        if random.random() < self.config.mutation_rate:
            mutation = np.random.normal(0, 0.1, size=individual.shape)
            individual = individual + mutation
            
            # Apply bounds
            if bounds is not None:
                individual = np.clip(individual, bounds[0], bounds[1])
        
        return individual
    
    def _optimize_quantum_annealing(
        self,
        objective_fn: Callable,
        initial_params: jnp.ndarray,
        bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
        constraints: Optional[List[Callable]]
    ) -> OptimizationResult:
        """Simulate quantum annealing optimization."""
        
        current_params = initial_params.copy()
        current_value = objective_fn(current_params)
        best_params = current_params.copy()
        best_value = current_value
        
        history = []
        temperature = 1.0
        cooling_rate = 0.995
        
        for iteration in range(self.config.max_iterations):
            # Generate neighbor solution
            if bounds is not None:
                perturbation = np.random.uniform(-0.1, 0.1, size=current_params.shape)
            else:
                perturbation = np.random.normal(0, 0.1, size=current_params.shape)
            
            new_params = current_params + perturbation
            
            # Apply bounds
            if bounds is not None:
                new_params = jnp.clip(new_params, bounds[0], bounds[1])
            
            new_value = objective_fn(new_params)
            
            # Apply constraint penalties
            if constraints:
                penalty = 0.0
                for constraint in constraints:
                    violation = max(0, -constraint(new_params))
                    penalty += violation * 1000
                new_value -= penalty
            
            # Acceptance criterion (simulated annealing)
            if self.config.objective_type == ObjectiveType.MAXIMIZE:
                delta = new_value - current_value
            else:
                delta = current_value - new_value
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_params = new_params
                current_value = new_value
                
                # Update best solution
                if self._is_better(current_value, best_value):
                    best_params = current_params.copy()
                    best_value = current_value
            
            history.append(float(best_value))
            temperature *= cooling_rate
            
            # Early stopping if temperature is too low
            if temperature < 1e-6:
                break
        
        return OptimizationResult(
            optimal_parameters=best_params,
            optimal_value=best_value,
            iterations=iteration + 1,
            convergence_time=0.0,
            converged=True,
            optimization_history=history,
            gradient_norms=[],
            constraint_violations=[],
            metadata={
                "algorithm": "quantum_annealing",
                "final_temperature": temperature,
                "cooling_rate": cooling_rate
            }
        )
    
    def _optimize_hybrid(
        self,
        objective_fn: Callable,
        initial_params: jnp.ndarray,
        bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
        constraints: Optional[List[Callable]]
    ) -> OptimizationResult:
        """Hybrid optimization combining multiple strategies."""
        
        results = []
        
        # Stage 1: Evolutionary search for global exploration
        self.config.strategy = OptimizationStrategy.EVOLUTIONARY
        self.config.max_iterations = self.config.max_iterations // 3
        evo_result = self._optimize_evolutionary(objective_fn, initial_params, bounds, constraints)
        results.append(evo_result)
        
        # Stage 2: Quantum annealing for local search
        self.config.strategy = OptimizationStrategy.QUANTUM_ANNEALING
        qa_result = self._optimize_quantum_annealing(
            objective_fn, evo_result.optimal_parameters, bounds, constraints
        )
        results.append(qa_result)
        
        # Stage 3: Adam for fine-tuning
        self.config.strategy = OptimizationStrategy.ADAM
        self.config.learning_rate = 0.0001  # Smaller learning rate for fine-tuning
        adam_result = self._optimize_adam(
            objective_fn, qa_result.optimal_parameters, bounds, constraints
        )
        results.append(adam_result)
        
        # Select best result
        best_result = min(results, key=lambda r: r.optimal_value) if self.config.objective_type == ObjectiveType.MINIMIZE else max(results, key=lambda r: r.optimal_value)
        
        # Combine histories
        combined_history = []
        for result in results:
            combined_history.extend(result.optimization_history)
        
        return OptimizationResult(
            optimal_parameters=best_result.optimal_parameters,
            optimal_value=best_result.optimal_value,
            iterations=sum(r.iterations for r in results),
            convergence_time=0.0,
            converged=best_result.converged,
            optimization_history=combined_history,
            gradient_norms=adam_result.gradient_norms,
            constraint_violations=adam_result.constraint_violations,
            metadata={
                "algorithm": "hybrid",
                "stages": ["evolutionary", "quantum_annealing", "adam"],
                "stage_results": [r.metadata for r in results]
            }
        )
    
    def _optimize_gradient_descent(
        self,
        objective_fn: Callable,
        initial_params: jnp.ndarray,
        bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]],
        constraints: Optional[List[Callable]]
    ) -> OptimizationResult:
        """Basic gradient descent with momentum."""
        
        params = initial_params.copy()
        history = []
        gradient_norms = []
        
        grad_fn = jit(grad(objective_fn))
        
        for iteration in range(self.config.max_iterations):
            current_value = objective_fn(params)
            current_grad = grad_fn(params)
            
            grad_norm = jnp.linalg.norm(current_grad)
            gradient_norms.append(float(grad_norm))
            
            # Momentum update
            self.velocity = self.momentum * self.velocity + self.current_lr * current_grad
            
            if self.config.objective_type == ObjectiveType.MAXIMIZE:
                params = params + self.velocity
            else:
                params = params - self.velocity
            
            # Apply bounds
            if bounds is not None:
                params = jnp.clip(params, bounds[0], bounds[1])
            
            history.append(float(current_value))
            
            # Convergence check
            if grad_norm < self.config.tolerance:
                break
        
        return OptimizationResult(
            optimal_parameters=params,
            optimal_value=float(current_value),
            iterations=iteration + 1,
            convergence_time=0.0,
            converged=grad_norm < self.config.tolerance,
            optimization_history=history,
            gradient_norms=gradient_norms,
            constraint_violations=[],
            metadata={"algorithm": "gradient_descent"}
        )


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization with Pareto frontier computation.
    
    Handles multiple competing objectives and finds the Pareto-optimal
    solutions that represent the best trade-offs between objectives.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.pareto_front: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
    
    def optimize(
        self,
        objective_fns: List[Callable[[jnp.ndarray], float]],
        initial_params: jnp.ndarray,
        bounds: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Optimize multiple objectives simultaneously.
        
        Args:
            objective_fns: List of objective functions
            initial_params: Starting parameter values
            bounds: Optional parameter bounds
            weights: Optional weights for scalarization
            
        Returns:
            Dictionary containing Pareto front and optimization results
        """
        
        if weights is None:
            weights = [1.0 / len(objective_fns)] * len(objective_fns)
        
        # Generate multiple solutions with different weight combinations
        solutions = []
        
        # Systematic weight variation
        n_points = 20
        for i in range(n_points):
            # Generate random weights
            w = np.random.dirichlet([1] * len(objective_fns))
            
            # Create scalarized objective
            def scalarized_objective(params):
                values = [fn(params) for fn in objective_fns]
                return sum(wi * vi for wi, vi in zip(w, values))
            
            # Optimize scalarized objective
            optimizer = AdaptiveOptimizer(self.config)
            result = optimizer.optimize(scalarized_objective, initial_params, bounds)
            
            # Evaluate all objectives for this solution
            objective_values = jnp.array([fn(result.optimal_parameters) for fn in objective_fns])
            solutions.append((result.optimal_parameters, objective_values))
        
        # Compute Pareto front
        self.pareto_front = self._compute_pareto_front(solutions)
        
        return {
            "pareto_front": self.pareto_front,
            "n_solutions": len(solutions),
            "n_pareto_optimal": len(self.pareto_front),
            "hypervolume": self._compute_hypervolume(),
            "spread": self._compute_spread()
        }
    
    def _compute_pareto_front(self, solutions: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Compute Pareto front from solutions."""
        pareto_front = []
        
        for i, (params_i, obj_i) in enumerate(solutions):
            is_dominated = False
            
            for j, (params_j, obj_j) in enumerate(solutions):
                if i != j and self._dominates(obj_j, obj_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append((params_i, obj_i))
        
        return pareto_front
    
    def _dominates(self, obj1: jnp.ndarray, obj2: jnp.ndarray) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)."""
        return jnp.all(obj1 >= obj2) and jnp.any(obj1 > obj2)
    
    def _compute_hypervolume(self) -> float:
        """Compute hypervolume indicator for Pareto front quality."""
        if not self.pareto_front:
            return 0.0
        
        # Simple hypervolume calculation (for 2D case)
        if len(self.pareto_front[0][1]) == 2:
            # Sort by first objective
            sorted_front = sorted(self.pareto_front, key=lambda x: x[1][0])
            
            hypervolume = 0.0
            for i in range(len(sorted_front)):
                if i == 0:
                    width = sorted_front[i][1][0]
                else:
                    width = sorted_front[i][1][0] - sorted_front[i-1][1][0]
                height = sorted_front[i][1][1]
                hypervolume += width * height
            
            return hypervolume
        
        return 0.0  # Not implemented for higher dimensions
    
    def _compute_spread(self) -> float:
        """Compute spread indicator for solution diversity."""
        if len(self.pareto_front) < 2:
            return 0.0
        
        objectives = jnp.array([obj for _, obj in self.pareto_front])
        distances = []
        
        for i in range(len(objectives)):
            min_dist = float('inf')
            for j in range(len(objectives)):
                if i != j:
                    dist = jnp.linalg.norm(objectives[i] - objectives[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        return float(jnp.std(jnp.array(distances)))


# Utility functions for common optimization tasks

def optimize_reward_function(
    contract_data: Dict[str, Any],
    training_data: List[Tuple[jnp.ndarray, jnp.ndarray, float]],
    config: Optional[OptimizationConfig] = None
) -> OptimizationResult:
    """
    Optimize reward function parameters for a given contract.
    
    Args:
        contract_data: Contract specification
        training_data: List of (state, action, reward) tuples
        config: Optimization configuration
        
    Returns:
        Optimization result with learned parameters
    """
    if config is None:
        config = OptimizationConfig(
            strategy=OptimizationStrategy.ADAM,
            max_iterations=1000,
            learning_rate=0.001
        )
    
    # Initialize reward function parameters
    n_params = 10  # Example: 10 parameters
    initial_params = jnp.array(np.random.normal(0, 0.1, n_params))
    
    def objective(params):
        """Mean squared error between predicted and true rewards."""
        total_error = 0.0
        
        for state, action, true_reward in training_data:
            # Simple linear reward model
            features = jnp.concatenate([state, action])
            if len(features) < len(params):
                features = jnp.pad(features, (0, len(params) - len(features)))
            else:
                features = features[:len(params)]
            
            predicted_reward = jnp.dot(params, features)
            error = (predicted_reward - true_reward) ** 2
            total_error += error
        
        return -total_error / len(training_data)  # Negative because we minimize
    
    optimizer = AdaptiveOptimizer(config)
    return optimizer.optimize(objective, initial_params)


async def optimize_contract_verification(
    contract_spec: Dict[str, Any],
    verification_properties: List[str],
    config: Optional[OptimizationConfig] = None
) -> OptimizationResult:
    """
    Optimize verification parameters for faster contract verification.
    
    Args:
        contract_spec: Contract specification
        verification_properties: Properties to verify
        config: Optimization configuration
        
    Returns:
        Optimization result with optimal verification parameters
    """
    if config is None:
        config = OptimizationConfig(
            strategy=OptimizationStrategy.HYBRID,
            max_iterations=500
        )
    
    # Simulation of verification optimization
    initial_params = jnp.array([0.5, 0.3, 0.8, 0.2])  # Example parameters
    
    def verification_objective(params):
        """Simulate verification efficiency metric."""
        # Higher values mean faster verification
        efficiency = jnp.sum(params ** 2) - jnp.sum(jnp.abs(params - 0.5))
        return efficiency
    
    # Constraints: parameters must be in [0, 1]
    bounds = (jnp.zeros(4), jnp.ones(4))
    
    optimizer = AdaptiveOptimizer(config)
    return optimizer.optimize(verification_objective, initial_params, bounds)


def benchmark_optimization_strategies(
    test_functions: List[Callable],
    strategies: List[OptimizationStrategy],
    dimensions: List[int]
) -> Dict[str, Any]:
    """
    Benchmark different optimization strategies on test functions.
    
    Args:
        test_functions: List of test functions to optimize
        strategies: Optimization strategies to compare
        dimensions: Problem dimensions to test
        
    Returns:
        Comprehensive benchmark results
    """
    results = defaultdict(list)
    
    for func in test_functions:
        for strategy in strategies:
            for dim in dimensions:
                config = OptimizationConfig(
                    strategy=strategy,
                    max_iterations=100,
                    population_size=20
                )
                
                initial_params = jnp.array(np.random.normal(0, 1, dim))
                optimizer = AdaptiveOptimizer(config)
                
                start_time = time.time()
                result = optimizer.optimize(func, initial_params)
                total_time = time.time() - start_time
                
                results[f"{func.__name__}_{strategy.value}_{dim}d"].append({
                    "optimal_value": result.optimal_value,
                    "iterations": result.iterations,
                    "convergence_time": total_time,
                    "converged": result.converged
                })
    
    return dict(results)


# Example test functions for benchmarking
def sphere_function(x: jnp.ndarray) -> float:
    """Sphere function: f(x) = sum(x^2)"""
    return -jnp.sum(x ** 2)  # Negative for maximization


def rosenbrock_function(x: jnp.ndarray) -> float:
    """Rosenbrock function: classic optimization test"""
    return -jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin_function(x: jnp.ndarray) -> float:
    """Rastrigin function: multimodal test function"""
    A = 10
    n = len(x)
    return -(A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x)))


if __name__ == "__main__":
    # Example usage and benchmarking
    print("🔧 Running Advanced Optimization Demo")
    
    # Test single-objective optimization
    config = OptimizationConfig(
        strategy=OptimizationStrategy.HYBRID,
        max_iterations=100
    )
    
    optimizer = AdaptiveOptimizer(config)
    result = optimizer.optimize(sphere_function, jnp.array([1.0, 2.0, -1.0]))
    
    print(f"✅ Optimization completed:")
    print(f"   Optimal value: {result.optimal_value:.6f}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Converged: {result.converged}")
    
    # Test multi-objective optimization
    def obj1(x): return jnp.sum(x**2)
    def obj2(x): return jnp.sum((x - 1)**2)
    
    mo_optimizer = MultiObjectiveOptimizer(config)
    mo_result = mo_optimizer.optimize([obj1, obj2], jnp.array([0.5, 0.5]))
    
    print(f"✅ Multi-objective optimization:")
    print(f"   Pareto solutions: {mo_result['n_pareto_optimal']}")
    print(f"   Hypervolume: {mo_result['hypervolume']:.6f}")
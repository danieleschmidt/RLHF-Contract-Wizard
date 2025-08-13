"""
Core reward contract model implementation.

This module defines the primary RewardContract class that encapsulates
RLHF reward functions with legal constraints and multi-stakeholder governance.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import jax
import jax.numpy as jnp
from jax import jit, vmap
from ..optimization.contract_cache import reward_cache


class AggregationStrategy(Enum):
    """Stakeholder preference aggregation strategies."""
    WEIGHTED_AVERAGE = "weighted_average"
    NASH_BARGAINING = "nash_bargaining"
    UTILITARIAN = "utilitarian"
    LEXICOGRAPHIC = "lexicographic"


@dataclass
class Stakeholder:
    """Represents a stakeholder in the reward contract."""
    name: str
    weight: float
    address: Optional[str] = None
    voting_power: float = 1.0
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Constraint:
    """Represents a legal/safety constraint in the contract."""
    name: str
    description: str
    constraint_fn: Callable
    severity: float = 1.0
    enabled: bool = True
    violation_penalty: float = -1.0


@dataclass
class ContractMetadata:
    """Metadata for the reward contract."""
    name: str
    version: str
    created_at: float
    updated_at: float
    creator: str
    jurisdiction: str = "Global"
    regulatory_framework: Optional[str] = None


class RewardContract:
    """
    Core reward contract implementation.
    
    Encodes RLHF reward functions with legal constraints, multi-stakeholder
    governance, and formal verification support.
    """
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        stakeholders: Optional[Dict[str, float]] = None,
        aggregation: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE,
        creator: str = "system",
        jurisdiction: str = "Global"
    ):
        """
        Initialize a new reward contract.
        
        Args:
            name: Contract identifier
            version: Contract version
            stakeholders: Dict mapping stakeholder names to weights
            aggregation: Strategy for aggregating stakeholder preferences
            creator: Contract creator identifier
            jurisdiction: Legal jurisdiction
        """
        self.metadata = ContractMetadata(
            name=name,
            version=version,
            created_at=time.time(),
            updated_at=time.time(),
            creator=creator,
            jurisdiction=jurisdiction
        )
        
        # Initialize stakeholders
        self.stakeholders: Dict[str, Stakeholder] = {}
        if stakeholders:
            total_weight = sum(stakeholders.values())
            for name, weight in stakeholders.items():
                self.stakeholders[name] = Stakeholder(
                    name=name,
                    weight=weight / total_weight  # Normalize weights
                )
        
        self.aggregation_strategy = aggregation
        self.constraints: Dict[str, Constraint] = {}
        self.reward_functions: Dict[str, Callable] = {}
        self._compiled_reward_fn: Optional[Callable] = None
        self._contract_hash: Optional[str] = None
    
    def add_stakeholder(self, name: str, weight: float, **kwargs) -> 'RewardContract':
        """Add a new stakeholder to the contract."""
        self.stakeholders[name] = Stakeholder(name=name, weight=weight, **kwargs)
        self._normalize_stakeholder_weights()
        self._invalidate_cache()
        return self
    
    def add_constraint(
        self, 
        name: str, 
        constraint_fn: Callable,
        description: str = "",
        severity: float = 1.0,
        violation_penalty: float = -1.0
    ) -> 'RewardContract':
        """Add a legal/safety constraint to the contract."""
        self.constraints[name] = Constraint(
            name=name,
            description=description,
            constraint_fn=constraint_fn,
            severity=severity,
            violation_penalty=violation_penalty
        )
        self._invalidate_cache()
        return self
    
    def reward_function(self, stakeholder: Optional[str] = None):
        """Decorator for registering reward functions."""
        def decorator(func: Callable):
            if stakeholder:
                if stakeholder not in self.stakeholders:
                    raise ValueError(f"Unknown stakeholder: {stakeholder}")
                self.reward_functions[stakeholder] = func
            else:
                self.reward_functions["default"] = func
            self._invalidate_cache()
            return func
        return decorator
    
    def compute_reward(self, state: jnp.ndarray, action: jnp.ndarray, use_cache: bool = True) -> float:
        """
        Compute aggregated reward with constraint enforcement.
        
        Args:
            state: Current environment state
            action: Proposed action
            
        Returns:
            Final reward value considering all stakeholders and constraints
        """
        try:
            # Input validation
            if state is None or action is None:
                raise ValueError("State and action cannot be None")
            
            if not isinstance(state, jnp.ndarray) or not isinstance(action, jnp.ndarray):
                raise ValueError("State and action must be JAX arrays")
            
            if state.size == 0 or action.size == 0:
                raise ValueError("State and action cannot be empty")
            
            # Check cache first (if enabled)
            if use_cache:
                cache_key = self._generate_cache_key(state, action)
                cached_reward = reward_cache.get(cache_key)
                if cached_reward is not None:
                    return cached_reward
            
            if self._compiled_reward_fn is None:
                self._compile_reward_function()
            
            reward = self._compiled_reward_fn(state, action)
            
            # Ensure reward is finite
            if not jnp.isfinite(reward):
                raise ValueError(f"Computed reward is not finite: {reward}")
            
            reward_float = float(reward)
            
            # Cache result (if enabled)
            if use_cache:
                reward_cache.set(
                    cache_key, 
                    reward_float,
                    tags={'contract', self.metadata.name}
                )
            
            return reward_float
            
        except Exception as e:
            # Import error handling utilities
            from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity
            
            handle_error(
                error=e,
                operation="compute_reward",
                category=ErrorCategory.COMPUTATION,
                severity=ErrorSeverity.HIGH,
                additional_info={
                    "contract_name": self.metadata.name,
                    "state_shape": state.shape if hasattr(state, 'shape') else None,
                    "action_shape": action.shape if hasattr(action, 'shape') else None
                }
            )
            raise
    
    def check_violations(self, state: jnp.ndarray, action: jnp.ndarray) -> Dict[str, bool]:
        """Check for constraint violations with comprehensive error handling."""
        violations = {}
        
        try:
            # Input validation
            if state is None or action is None:
                raise ValueError("State and action cannot be None")
            
            for name, constraint in self.constraints.items():
                if not constraint.enabled:
                    violations[name] = False
                    continue
                
                try:
                    # Execute constraint with timeout protection
                    result = constraint.constraint_fn(state, action)
                    violations[name] = not bool(result)
                    
                except Exception as e:
                    # Import error handling utilities
                    from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity
                    
                    handle_error(
                        error=e,
                        operation=f"constraint_check:{name}",
                        category=ErrorCategory.CONTRACT,
                        severity=ErrorSeverity.MEDIUM,
                        additional_info={
                            "contract_name": self.metadata.name,
                            "constraint_name": name,
                            "constraint_description": constraint.description
                        },
                        attempt_recovery=False
                    )
                    
                    # Treat constraint check failure as violation for safety
                    violations[name] = True
                    
        except Exception as e:
            from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity
            
            handle_error(
                error=e,
                operation="check_violations",
                category=ErrorCategory.CONTRACT,
                severity=ErrorSeverity.HIGH,
                additional_info={"contract_name": self.metadata.name}
            )
            
            # Return all constraints as violated for safety
            violations = {name: True for name in self.constraints.keys()}
            
        return violations
    
    def get_violation_penalty(self, violations: Dict[str, bool]) -> float:
        """Calculate total penalty for constraint violations."""
        total_penalty = 0.0
        for name, violated in violations.items():
            if violated and name in self.constraints:
                constraint = self.constraints[name]
                total_penalty += constraint.violation_penalty * constraint.severity
        return total_penalty
    
    def _compile_reward_function(self):
        """Compile stakeholder preferences into single JAX function with error handling."""
        try:
            if not self.reward_functions:
                raise ValueError("No reward functions defined")
            
            if not self.stakeholders:
                raise ValueError("No stakeholders defined")
            
            # Validate stakeholder weights sum to approximately 1.0
            total_weight = sum(s.weight for s in self.stakeholders.values())
            if abs(total_weight - 1.0) > 1e-6:
                from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity
                handle_error(
                    error=ValueError(f"Stakeholder weights sum to {total_weight}, expected 1.0"),
                    operation="compile_reward_function",
                    category=ErrorCategory.CONTRACT,
                    severity=ErrorSeverity.MEDIUM,
                    additional_info={"total_weight": total_weight},
                    attempt_recovery=False
                )
            
            # Pre-compute stakeholder info for optimization
            stakeholder_list = list(self.stakeholders.values())
            stakeholder_names = [s.name for s in stakeholder_list]
            stakeholder_weights = jnp.array([s.weight for s in stakeholder_list])
            
            # Validate weights array
            if not jnp.all(jnp.isfinite(stakeholder_weights)):
                raise ValueError("Stakeholder weights contain invalid values")
            
            def compiled_fn(state: jnp.ndarray, action: jnp.ndarray) -> float:
                # Vectorized reward computation
                rewards = []
                
                for i, stakeholder_name in enumerate(stakeholder_names):
                    if stakeholder_name in self.reward_functions:
                        reward_fn = self.reward_functions[stakeholder_name]
                        reward = reward_fn(state, action)
                    elif "default" in self.reward_functions:
                        reward_fn = self.reward_functions["default"]
                        reward = reward_fn(state, action)
                    else:
                        reward = 0.0
                    rewards.append(reward)
                
                rewards_array = jnp.array(rewards)
                
                # Optimized aggregation
                if self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
                    aggregated_reward = jnp.dot(stakeholder_weights, rewards_array)
                elif self.aggregation_strategy == AggregationStrategy.UTILITARIAN:
                    aggregated_reward = jnp.sum(rewards_array)
                else:
                    aggregated_reward = jnp.dot(stakeholder_weights, rewards_array)
                
                # Optimized constraint checking
                total_penalty = 0.0
                for name, constraint in self.constraints.items():
                    if constraint.enabled:
                        try:
                            if not constraint.constraint_fn(state, action):
                                total_penalty += constraint.violation_penalty * constraint.severity
                        except:
                            total_penalty += constraint.violation_penalty * constraint.severity
                
                return aggregated_reward + total_penalty
            
            self._compiled_reward_fn = jit(compiled_fn)
            
        except Exception as e:
            from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity
            
            handle_error(
                error=e,
                operation="compile_reward_function",
                category=ErrorCategory.CONTRACT,
                severity=ErrorSeverity.HIGH,
                additional_info={"contract_name": self.metadata.name}
            )
            raise
    
    def _generate_cache_key(self, state: jnp.ndarray, action: jnp.ndarray) -> str:
        """Generate cache key for reward computation."""
        # Create deterministic key from contract hash and input arrays
        contract_hash = self.compute_hash()[:16]  # First 16 chars
        state_hash = hashlib.md5(state.tobytes()).hexdigest()[:8]
        action_hash = hashlib.md5(action.tobytes()).hexdigest()[:8]
        return f"reward:{contract_hash}:{state_hash}:{action_hash}"
    
    def _normalize_stakeholder_weights(self):
        """Normalize stakeholder weights to sum to 1.0."""
        if not self.stakeholders:
            return
        
        total_weight = sum(s.weight for s in self.stakeholders.values())
        if total_weight > 0:
            for stakeholder in self.stakeholders.values():
                stakeholder.weight /= total_weight
    
    def _invalidate_cache(self):
        """Invalidate cached compiled functions when contract changes."""
        self._compiled_reward_fn = None
        self._contract_hash = None
        self.metadata.updated_at = time.time()
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of contract specification with caching."""
        if self._contract_hash is not None:
            return self._contract_hash
        
        # Create deterministic representation
        contract_data = {
            "metadata": {
                "name": self.metadata.name,
                "version": self.metadata.version,
                "jurisdiction": self.metadata.jurisdiction
            },
            "stakeholders": {
                name: {
                    "weight": stakeholder.weight,
                    "voting_power": stakeholder.voting_power
                }
                for name, stakeholder in sorted(self.stakeholders.items())
            },
            "constraints": {
                name: {
                    "description": constraint.description,
                    "severity": constraint.severity,
                    "violation_penalty": constraint.violation_penalty
                }
                for name, constraint in sorted(self.constraints.items())
            },
            "aggregation_strategy": self.aggregation_strategy.value
        }
        
        contract_json = json.dumps(contract_data, sort_keys=True)
        self._contract_hash = hashlib.sha256(contract_json.encode()).hexdigest()
        return self._contract_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Export contract to dictionary format."""
        return {
            "metadata": {
                "name": self.metadata.name,
                "version": self.metadata.version,
                "created_at": self.metadata.created_at,
                "updated_at": self.metadata.updated_at,
                "creator": self.metadata.creator,
                "jurisdiction": self.metadata.jurisdiction,
                "regulatory_framework": self.metadata.regulatory_framework
            },
            "stakeholders": {
                name: {
                    "weight": stakeholder.weight,
                    "voting_power": stakeholder.voting_power,
                    "address": stakeholder.address,
                    "preferences": stakeholder.preferences
                }
                for name, stakeholder in self.stakeholders.items()
            },
            "constraints": {
                name: {
                    "description": constraint.description,
                    "severity": constraint.severity,
                    "enabled": constraint.enabled,
                    "violation_penalty": constraint.violation_penalty
                }
                for name, constraint in self.constraints.items()
            },
            "aggregation_strategy": self.aggregation_strategy.value,
            "contract_hash": self.compute_hash()
        }
    
    def __repr__(self) -> str:
        return (
            f"RewardContract(name='{self.metadata.name}', "
            f"version='{self.metadata.version}', "
            f"stakeholders={len(self.stakeholders)}, "
            f"constraints={len(self.constraints)})"
        )
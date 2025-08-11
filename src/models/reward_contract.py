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
    
    def compute_reward(self, state: jnp.ndarray, action: jnp.ndarray) -> float:
        """
        Compute aggregated reward with constraint enforcement.
        
        Args:
            state: Current environment state
            action: Proposed action
            
        Returns:
            Final reward value considering all stakeholders and constraints
        """
        if self._compiled_reward_fn is None:
            self._compile_reward_function()
        
        return self._compiled_reward_fn(state, action)
    
    def check_violations(self, state: jnp.ndarray, action: jnp.ndarray) -> Dict[str, bool]:
        """Check for constraint violations."""
        violations = {}
        for name, constraint in self.constraints.items():
            if constraint.enabled:
                try:
                    violations[name] = not constraint.constraint_fn(state, action)
                except Exception as e:
                    # Log error and treat as violation
                    violations[name] = True
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
        """Compile stakeholder preferences into single JAX function."""
        if not self.reward_functions:
            raise ValueError("No reward functions defined")
        
        # Pre-compute stakeholder info for optimization
        stakeholder_list = list(self.stakeholders.values())
        stakeholder_names = [s.name for s in stakeholder_list]
        stakeholder_weights = jnp.array([s.weight for s in stakeholder_list])
        
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
        """Compute deterministic hash of contract specification."""
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
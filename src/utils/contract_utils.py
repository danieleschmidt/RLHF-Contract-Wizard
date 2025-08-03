"""
Utility functions for contract operations and validation.

Provides helper functions for contract creation, validation,
serialization, and common operations.
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import logging
from datetime import datetime, timedelta


class ContractError(Exception):
    """Base exception for contract-related errors."""
    pass


class ValidationError(ContractError):
    """Raised when contract validation fails."""
    pass


class SerializationError(ContractError):
    """Raised when contract serialization fails."""
    pass


def generate_contract_id(name: str, version: str, creator: str) -> str:
    """
    Generate a unique contract identifier.
    
    Args:
        name: Contract name
        version: Contract version
        creator: Contract creator
        
    Returns:
        Unique contract ID
    """
    content = f"{name}:{version}:{creator}:{time.time()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def validate_stakeholder_weights(weights: Dict[str, float]) -> bool:
    """
    Validate stakeholder weight distribution.
    
    Args:
        weights: Dictionary of stakeholder weights
        
    Returns:
        True if weights are valid
        
    Raises:
        ValidationError: If weights are invalid
    """
    if not weights:
        raise ValidationError("At least one stakeholder required")
    
    # Check positive weights
    for name, weight in weights.items():
        if weight <= 0:
            raise ValidationError(f"Stakeholder '{name}' has non-positive weight: {weight}")
    
    # Check weight sum
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        logging.warning(f"Stakeholder weights sum to {total_weight}, expected 1.0")
    
    return True


def create_dummy_data(state_dim: int = 10, action_dim: int = 5) -> tuple:
    """
    Create dummy state and action data for testing.
    
    Args:
        state_dim: Dimension of state vector
        action_dim: Dimension of action vector
        
    Returns:
        Tuple of (state, action, context)
    """
    state = jnp.zeros(state_dim)
    action = jnp.zeros(action_dim)
    context = {
        'timestamp': time.time(),
        'user_id': 'test_user',
        'session_id': 'test_session'
    }
    
    return state, action, context


def compute_constraint_complexity(constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the complexity of contract constraints.
    
    Args:
        constraints: Dictionary of constraint definitions
        
    Returns:
        Dictionary with complexity metrics
    """
    metrics = {
        'total_constraints': len(constraints),
        'enabled_constraints': 0,
        'severity_distribution': {},
        'avg_severity': 0.0,
        'max_severity': 0.0,
        'constraint_types': set()
    }
    
    severities = []
    
    for name, constraint in constraints.items():
        if hasattr(constraint, 'enabled') and constraint.enabled:
            metrics['enabled_constraints'] += 1
        
        if hasattr(constraint, 'severity'):
            severity = constraint.severity
            severities.append(severity)
            metrics['max_severity'] = max(metrics['max_severity'], severity)
            
            # Categorize severity
            if severity < 0.3:
                category = 'low'
            elif severity < 0.7:
                category = 'medium'
            else:
                category = 'high'
            
            metrics['severity_distribution'][category] = \
                metrics['severity_distribution'].get(category, 0) + 1
        
        # Analyze constraint type from function metadata
        if hasattr(constraint, 'constraint_fn'):
            func_name = getattr(constraint.constraint_fn, '__name__', 'unknown')
            metrics['constraint_types'].add(func_name)
    
    if severities:
        metrics['avg_severity'] = sum(severities) / len(severities)
    
    metrics['constraint_types'] = list(metrics['constraint_types'])
    
    return metrics


def estimate_computational_overhead(
    num_stakeholders: int,
    num_constraints: int,
    state_dim: int,
    action_dim: int
) -> Dict[str, float]:
    """
    Estimate computational overhead for contract evaluation.
    
    Args:
        num_stakeholders: Number of stakeholders
        num_constraints: Number of constraints
        state_dim: State vector dimension
        action_dim: Action vector dimension
        
    Returns:
        Dictionary with overhead estimates
    """
    # Simple heuristic model for computational cost
    base_cost = 1.0  # Baseline cost
    
    # Cost increases with number of stakeholders (reward aggregation)
    stakeholder_cost = num_stakeholders * 0.1
    
    # Cost increases with constraints (validation overhead)
    constraint_cost = num_constraints * 0.05
    
    # Cost increases with input dimensionality
    dimension_cost = (state_dim + action_dim) * 0.001
    
    total_overhead = base_cost + stakeholder_cost + constraint_cost + dimension_cost
    
    return {
        'total_overhead_factor': total_overhead,
        'stakeholder_overhead': stakeholder_cost,
        'constraint_overhead': constraint_cost,
        'dimension_overhead': dimension_cost,
        'estimated_latency_ms': total_overhead * 5.0,  # Rough latency estimate
        'memory_overhead_mb': total_overhead * 10.0    # Rough memory estimate
    }


def serialize_contract_metadata(contract) -> Dict[str, Any]:
    """
    Serialize contract metadata for storage or transmission.
    
    Args:
        contract: RewardContract instance
        
    Returns:
        Serializable metadata dictionary
    """
    try:
        return {
            'metadata': {
                'name': contract.metadata.name,
                'version': contract.metadata.version,
                'created_at': contract.metadata.created_at,
                'updated_at': contract.metadata.updated_at,
                'creator': contract.metadata.creator,
                'jurisdiction': contract.metadata.jurisdiction,
                'regulatory_framework': contract.metadata.regulatory_framework
            },
            'stakeholders': {
                name: {
                    'weight': stakeholder.weight,
                    'voting_power': stakeholder.voting_power,
                    'address': stakeholder.address
                }
                for name, stakeholder in contract.stakeholders.items()
            },
            'constraints': {
                name: {
                    'description': constraint.description,
                    'severity': constraint.severity,
                    'enabled': constraint.enabled,
                    'violation_penalty': constraint.violation_penalty
                }
                for name, constraint in contract.constraints.items()
            },
            'aggregation_strategy': contract.aggregation_strategy.value,
            'contract_hash': contract.compute_hash(),
            'serialization_timestamp': time.time()
        }
    except Exception as e:
        raise SerializationError(f"Failed to serialize contract metadata: {e}")


def validate_contract_structure(contract_data: Dict[str, Any]) -> List[str]:
    """
    Validate the structure of a contract data dictionary.
    
    Args:
        contract_data: Contract data to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required top-level fields
    required_fields = ['metadata', 'stakeholders', 'constraints']
    for field in required_fields:
        if field not in contract_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate metadata
    if 'metadata' in contract_data:
        metadata = contract_data['metadata']
        required_metadata = ['name', 'version', 'created_at', 'creator']
        for field in required_metadata:
            if field not in metadata:
                errors.append(f"Missing metadata field: {field}")
    
    # Validate stakeholders
    if 'stakeholders' in contract_data:
        stakeholders = contract_data['stakeholders']
        if not stakeholders:
            errors.append("At least one stakeholder required")
        else:
            for name, stakeholder in stakeholders.items():
                if 'weight' not in stakeholder:
                    errors.append(f"Stakeholder '{name}' missing weight")
                elif stakeholder['weight'] <= 0:
                    errors.append(f"Stakeholder '{name}' has non-positive weight")
    
    # Validate constraints
    if 'constraints' in contract_data:
        constraints = contract_data['constraints']
        for name, constraint in constraints.items():
            required_constraint_fields = ['description', 'severity']
            for field in required_constraint_fields:
                if field not in constraint:
                    errors.append(f"Constraint '{name}' missing field: {field}")
    
    return errors


def compute_contract_fingerprint(contract_data: Dict[str, Any]) -> str:
    """
    Compute a unique fingerprint for contract data.
    
    Args:
        contract_data: Contract data
        
    Returns:
        Hex-encoded fingerprint
    """
    # Create normalized representation for consistent hashing
    normalized = {
        'name': contract_data.get('metadata', {}).get('name', ''),
        'version': contract_data.get('metadata', {}).get('version', ''),
        'stakeholders': sorted(contract_data.get('stakeholders', {}).items()),
        'constraints': sorted([
            (name, constraint.get('description', ''), constraint.get('severity', 0))
            for name, constraint in contract_data.get('constraints', {}).items()
        ])
    }
    
    fingerprint_data = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(fingerprint_data.encode()).hexdigest()


def format_contract_summary(contract) -> str:
    """
    Format a human-readable summary of a contract.
    
    Args:
        contract: RewardContract instance
        
    Returns:
        Formatted summary string
    """
    summary_lines = [
        f"Contract: {contract.metadata.name} v{contract.metadata.version}",
        f"Created: {datetime.fromtimestamp(contract.metadata.created_at).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Creator: {contract.metadata.creator}",
        f"Jurisdiction: {contract.metadata.jurisdiction}",
        "",
        f"Stakeholders ({len(contract.stakeholders)}):",
    ]
    
    for name, stakeholder in contract.stakeholders.items():
        summary_lines.append(f"  - {name}: {stakeholder.weight:.1%} weight")
    
    summary_lines.extend([
        "",
        f"Constraints ({len(contract.constraints)}):",
    ])
    
    for name, constraint in contract.constraints.items():
        status = "✓" if constraint.enabled else "✗"
        summary_lines.append(f"  {status} {name}: {constraint.description[:50]}...")
    
    summary_lines.extend([
        "",
        f"Aggregation: {contract.aggregation_strategy.value}",
        f"Hash: {contract.compute_hash()[:16]}..."
    ])
    
    return "\n".join(summary_lines)


def benchmark_contract_performance(
    contract,
    num_iterations: int = 1000,
    state_dim: int = 10,
    action_dim: int = 5
) -> Dict[str, float]:
    """
    Benchmark contract performance.
    
    Args:
        contract: RewardContract instance
        num_iterations: Number of benchmark iterations
        state_dim: State vector dimension
        action_dim: Action vector dimension
        
    Returns:
        Performance metrics dictionary
    """
    import time
    
    # Prepare test data
    states = [jnp.random.normal(size=(state_dim,)) for _ in range(num_iterations)]
    actions = [jnp.random.normal(size=(action_dim,)) for _ in range(num_iterations)]
    contexts = [{'iteration': i} for i in range(num_iterations)]
    
    # Benchmark reward computation
    start_time = time.time()
    rewards = []
    
    for i in range(num_iterations):
        reward = contract.compute_reward(states[i], actions[i], contexts[i])
        rewards.append(reward)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_call = total_time / num_iterations * 1000  # ms
    
    # Benchmark constraint checking
    start_time = time.time()
    violations_list = []
    
    for i in range(num_iterations):
        violations = contract.check_violations(states[i], actions[i], contexts[i])
        violations_list.append(violations)
    
    constraint_time = time.time() - start_time
    avg_constraint_time = constraint_time / num_iterations * 1000  # ms
    
    # Calculate statistics
    rewards_array = np.array(rewards)
    
    return {
        'num_iterations': num_iterations,
        'total_time_ms': total_time * 1000,
        'avg_reward_computation_ms': avg_time_per_call,
        'avg_constraint_check_ms': avg_constraint_time,
        'rewards_mean': float(rewards_array.mean()),
        'rewards_std': float(rewards_array.std()),
        'rewards_min': float(rewards_array.min()),
        'rewards_max': float(rewards_array.max()),
        'total_violations': sum(sum(v.values()) for v in violations_list),
        'throughput_calls_per_second': num_iterations / total_time
    }


def migrate_contract_version(
    old_contract_data: Dict[str, Any],
    target_version: str
) -> Dict[str, Any]:
    """
    Migrate contract data to a newer version format.
    
    Args:
        old_contract_data: Contract data in old format
        target_version: Target version to migrate to
        
    Returns:
        Migrated contract data
    """
    # This is a simplified migration example
    # In practice, this would handle various version transitions
    
    migrated = old_contract_data.copy()
    
    # Update version metadata
    if 'metadata' in migrated:
        migrated['metadata']['version'] = target_version
        migrated['metadata']['migrated_at'] = time.time()
        migrated['metadata']['migration_from'] = old_contract_data.get('metadata', {}).get('version', 'unknown')
    
    # Add any new required fields for the target version
    if target_version >= "2.0.0":
        # Example: Add regulatory compliance fields
        if 'metadata' in migrated:
            migrated['metadata'].setdefault('regulatory_framework', None)
            migrated['metadata'].setdefault('compliance_status', 'pending')
    
    logging.info(f"Migrated contract to version {target_version}")
    return migrated
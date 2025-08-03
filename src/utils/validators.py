"""
Input validation utilities for RLHF contracts.

Provides validation functions for contract inputs, constraints,
and safety checks.
"""

import re
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
import jax.numpy as jnp


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_contract_name(name: str) -> bool:
    """
    Validate contract name format.
    
    Args:
        name: Contract name to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValidationError("Contract name must be a non-empty string")
    
    if len(name) < 3 or len(name) > 64:
        raise ValidationError("Contract name must be between 3 and 64 characters")
    
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
        raise ValidationError(
            "Contract name must start with a letter and contain only "
            "letters, numbers, underscores, and hyphens"
        )
    
    return True


def validate_version(version: str) -> bool:
    """
    Validate semantic version format.
    
    Args:
        version: Version string to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If version is invalid
    """
    if not version or not isinstance(version, str):
        raise ValidationError("Version must be a non-empty string")
    
    # Semantic versioning pattern: MAJOR.MINOR.PATCH
    pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*))?(?:\+([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*))?$'
    
    if not re.match(pattern, version):
        raise ValidationError(
            "Version must follow semantic versioning format (e.g., '1.0.0', '2.1.3-alpha')"
        )
    
    return True


def validate_stakeholder_weights(weights: Dict[str, float]) -> bool:
    """
    Validate stakeholder weight distribution.
    
    Args:
        weights: Dictionary of stakeholder weights
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If weights are invalid
    """
    if not weights or not isinstance(weights, dict):
        raise ValidationError("Stakeholder weights must be a non-empty dictionary")
    
    if len(weights) > 50:
        raise ValidationError("Maximum 50 stakeholders allowed")
    
    # Validate individual weights
    for name, weight in weights.items():
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Stakeholder names must be non-empty strings")
        
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValidationError(f"Stakeholder '{name}' weight must be positive")
        
        if weight > 1.0:
            raise ValidationError(f"Stakeholder '{name}' weight cannot exceed 1.0")
    
    # Check total weight
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise ValidationError(
            f"Total stakeholder weights must sum to 1.0, got {total_weight}"
        )
    
    return True


def validate_ethereum_address(address: str) -> bool:
    """
    Validate Ethereum address format.
    
    Args:
        address: Ethereum address to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If address is invalid
    """
    if not address or not isinstance(address, str):
        raise ValidationError("Address must be a non-empty string")
    
    # Check format: 0x followed by 40 hex characters
    if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
        raise ValidationError("Invalid Ethereum address format")
    
    return True


def validate_jax_array(array: jnp.ndarray, name: str = "array") -> bool:
    """
    Validate JAX array for contract computation.
    
    Args:
        array: JAX array to validate
        name: Name for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If array is invalid
    """
    if not isinstance(array, jnp.ndarray):
        raise ValidationError(f"{name} must be a JAX array")
    
    # Check for NaN or infinity
    if jnp.any(jnp.isnan(array)):
        raise ValidationError(f"{name} contains NaN values")
    
    if jnp.any(jnp.isinf(array)):
        raise ValidationError(f"{name} contains infinite values")
    
    # Check reasonable size (prevent memory issues)
    if array.size > 1000000:  # 1M elements
        raise ValidationError(f"{name} is too large (max 1M elements)")
    
    return True


def validate_reward_value(reward: float) -> bool:
    """
    Validate reward value is in acceptable range.
    
    Args:
        reward: Reward value to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If reward is invalid
    """
    if not isinstance(reward, (int, float)):
        raise ValidationError("Reward must be a number")
    
    if jnp.isnan(reward):
        raise ValidationError("Reward cannot be NaN")
    
    if jnp.isinf(reward):
        raise ValidationError("Reward cannot be infinite")
    
    # Typical RLHF reward range
    if reward < -10.0 or reward > 10.0:
        raise ValidationError("Reward should be in range [-10.0, 10.0]")
    
    return True


def validate_legal_blocks_spec(specification: str) -> bool:
    """
    Validate Legal-Blocks specification syntax.
    
    Args:
        specification: Legal-Blocks specification string
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If specification is invalid
    """
    if not specification or not isinstance(specification, str):
        raise ValidationError("Legal-Blocks specification must be a non-empty string")
    
    # Check for required keywords
    keywords = ['REQUIRES', 'ENSURES', 'INVARIANT', 'FORALL', 'EXISTS']
    if not any(keyword in specification.upper() for keyword in keywords):
        raise ValidationError(
            "Legal-Blocks specification must contain at least one constraint keyword: "
            f"{', '.join(keywords)}"
        )
    
    # Basic syntax checks
    lines = specification.strip().split('\n')
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Check for malformed lines
        if ':' not in line and any(kw in line.upper() for kw in keywords):
            raise ValidationError(f"Line {i}: Missing colon after constraint keyword")
    
    return True


def sanitize_user_input(user_input: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        user_input: Raw user input
        max_length: Maximum allowed length
        
    Returns:
        Sanitized input string
        
    Raises:
        ValidationError: If input is malicious
    """
    if not isinstance(user_input, str):
        raise ValidationError("Input must be a string")
    
    if len(user_input) > max_length:
        raise ValidationError(f"Input too long (max {max_length} characters)")
    
    # Remove potentially dangerous characters
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',               # JavaScript URLs
        r'on\w+\s*=',                # Event handlers
        r'eval\s*\(',                # eval() calls
        r'exec\s*\(',                # exec() calls
    ]
    
    sanitized = user_input
    for pattern in dangerous_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            raise ValidationError("Input contains potentially dangerous content")
    
    # Additional sanitization
    sanitized = re.sub(r'[<>"\']', '', sanitized)  # Remove HTML chars
    sanitized = sanitized.strip()
    
    return sanitized


def validate_gas_limit(gas_limit: int) -> bool:
    """
    Validate blockchain gas limit.
    
    Args:
        gas_limit: Gas limit to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If gas limit is invalid
    """
    if not isinstance(gas_limit, int):
        raise ValidationError("Gas limit must be an integer")
    
    if gas_limit <= 0:
        raise ValidationError("Gas limit must be positive")
    
    # Reasonable bounds
    min_gas = 21000    # Minimum for a transaction
    max_gas = 30000000 # Block gas limit (approximate)
    
    if gas_limit < min_gas:
        raise ValidationError(f"Gas limit too low (minimum {min_gas})")
    
    if gas_limit > max_gas:
        raise ValidationError(f"Gas limit too high (maximum {max_gas})")
    
    return True


def validate_constraint_function(constraint_fn: Any) -> bool:
    """
    Validate constraint function signature and behavior.
    
    Args:
        constraint_fn: Function to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If function is invalid
    """
    if not callable(constraint_fn):
        raise ValidationError("Constraint must be callable")
    
    # Check function signature
    import inspect
    sig = inspect.signature(constraint_fn)
    
    if len(sig.parameters) < 2:
        raise ValidationError(
            "Constraint function must accept at least 2 parameters (state, action)"
        )
    
    # Test with dummy data
    try:
        dummy_state = jnp.zeros(10)
        dummy_action = jnp.zeros(5)
        result = constraint_fn(dummy_state, dummy_action)
        
        if not isinstance(result, (bool, jnp.bool_, int, float)):
            raise ValidationError(
                "Constraint function must return a boolean or numeric value"
            )
    
    except Exception as e:
        raise ValidationError(f"Constraint function test failed: {str(e)}")
    
    return True


def validate_file_path(filepath: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate file path for security.
    
    Args:
        filepath: File path to validate
        allowed_extensions: List of allowed file extensions
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If path is invalid
    """
    if not filepath or not isinstance(filepath, str):
        raise ValidationError("File path must be a non-empty string")
    
    # Check for path traversal attempts
    if '..' in filepath or filepath.startswith('/'):
        raise ValidationError("Invalid file path (path traversal detected)")
    
    # Check extension if specified
    if allowed_extensions:
        extension = filepath.split('.')[-1].lower()
        if extension not in allowed_extensions:
            raise ValidationError(
                f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
            )
    
    return True


def compute_content_hash(content: Union[str, bytes]) -> str:
    """
    Compute secure hash of content.
    
    Args:
        content: Content to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    return hashlib.sha256(content).hexdigest()


def validate_hash(hash_string: str, expected_length: int = 64) -> bool:
    """
    Validate hash string format.
    
    Args:
        hash_string: Hash to validate
        expected_length: Expected hash length
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If hash is invalid
    """
    if not hash_string or not isinstance(hash_string, str):
        raise ValidationError("Hash must be a non-empty string")
    
    if len(hash_string) != expected_length:
        raise ValidationError(f"Hash must be {expected_length} characters long")
    
    if not re.match(r'^[a-fA-F0-9]+$', hash_string):
        raise ValidationError("Hash must contain only hexadecimal characters")
    
    return True
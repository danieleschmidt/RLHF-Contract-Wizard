"""
Utilities package for RLHF-Contract-Wizard.

Contains validation functions, helpers, and common utilities
used throughout the application.
"""

from .validators import (
    ValidationError,
    validate_contract_name,
    validate_version,
    validate_stakeholder_weights,
    validate_ethereum_address,
    validate_jax_array,
    validate_reward_value,
    validate_legal_blocks_spec,
    sanitize_user_input,
    validate_gas_limit,
    validate_constraint_function,
    validate_file_path,
    compute_content_hash,
    validate_hash
)

from .helpers import (
    setup_logging,
    create_timestamp,
    format_timestamp,
    ensure_directory,
    load_json_file,
    save_json_file,
    serialize_jax_array,
    deserialize_jax_array,
    batch_process,
    retry_with_backoff,
    compute_similarity,
    generate_contract_id,
    validate_network_connection,
    format_bytes,
    measure_execution_time,
    deep_merge_dicts,
    safe_divide,
    clamp,
    normalize_weights,
    create_progress_bar,
    get_system_info
)

__all__ = [
    # Validators
    'ValidationError',
    'validate_contract_name',
    'validate_version',
    'validate_stakeholder_weights',
    'validate_ethereum_address',
    'validate_jax_array',
    'validate_reward_value',
    'validate_legal_blocks_spec',
    'sanitize_user_input',
    'validate_gas_limit',
    'validate_constraint_function',
    'validate_file_path',
    'compute_content_hash',
    'validate_hash',
    
    # Helpers
    'setup_logging',
    'create_timestamp',
    'format_timestamp',
    'ensure_directory',
    'load_json_file',
    'save_json_file',
    'serialize_jax_array',
    'deserialize_jax_array',
    'batch_process',
    'retry_with_backoff',
    'compute_similarity',
    'generate_contract_id',
    'validate_network_connection',
    'format_bytes',
    'measure_execution_time',
    'deep_merge_dicts',
    'safe_divide',
    'clamp',
    'normalize_weights',
    'create_progress_bar',
    'get_system_info'
]
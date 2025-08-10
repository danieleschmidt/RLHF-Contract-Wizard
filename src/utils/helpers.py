"""
Helper utilities for RLHF contract operations.

Common utility functions for contract management, data processing,
and system operations.
"""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timezone
import jax
import jax.numpy as jnp
from dataclasses import asdict, is_dataclass


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup structured logging for the application.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("rlhf_contract")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            import json
            return json.load(f)
        else:
            return yaml.safe_load(f)


def create_timestamp() -> float:
    """Create UTC timestamp."""
    return time.time()


def format_timestamp(timestamp: float, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Unix timestamp
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime(format_str)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file safely.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json_file(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    # Convert dataclasses to dictionaries
    if is_dataclass(data):
        data = asdict(data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def serialize_jax_array(array: jnp.ndarray) -> List[float]:
    """
    Serialize JAX array to JSON-compatible list.
    
    Args:
        array: JAX array to serialize
        
    Returns:
        Python list
    """
    return array.tolist()


def deserialize_jax_array(data: List[float]) -> jnp.ndarray:
    """
    Deserialize list to JAX array.
    
    Args:
        data: List of numbers
        
    Returns:
        JAX array
    """
    return jnp.array(data)


def batch_process(
    items: List[Any],
    process_fn: Callable[[Any], Any],
    batch_size: int = 100,
    show_progress: bool = True
) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: Items to process
        process_fn: Processing function
        batch_size: Size of each batch
        show_progress: Whether to show progress
        
    Returns:
        List of processed results
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = [process_fn(item) for item in batch]
        results.extend(batch_results)
        
        if show_progress:
            batch_num = i // batch_size + 1
            print(f"Processed batch {batch_num}/{total_batches}")
    
    return results


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying function calls with exponential backoff.
    
    Args:
        func: Function to wrap
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exceptions to catch and retry
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        delay = base_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt == max_retries:
                    break
                
                time.sleep(min(delay, max_delay))
                delay *= 2  # Exponential backoff
        
        raise last_exception
    
    return wrapper


def compute_similarity(vec1: jnp.ndarray, vec2: jnp.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    # Normalize vectors
    norm1 = jnp.linalg.norm(vec1)
    norm2 = jnp.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(jnp.dot(vec1, vec2) / (norm1 * norm2))


def generate_contract_id(name: str, version: str, timestamp: float) -> str:
    """
    Generate unique contract identifier.
    
    Args:
        name: Contract name
        version: Contract version
        timestamp: Creation timestamp
        
    Returns:
        Unique contract ID
    """
    import hashlib
    content = f"{name}:{version}:{timestamp}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def validate_network_connection(url: str, timeout: float = 5.0) -> bool:
    """
    Validate network connection to URL.
    
    Args:
        url: URL to test
        timeout: Timeout in seconds
        
    Returns:
        True if connection successful
    """
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except:
        return False


def format_bytes(size_bytes: int) -> str:
    """
    Format byte size for human reading.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger = logging.getLogger("rlhf_contract")
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge in
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to range.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0.
    
    Args:
        weights: Dictionary of weights
        
    Returns:
        Normalized weights
    """
    total = sum(weights.values())
    if total == 0:
        return weights
    
    return {key: value / total for key, value in weights.items()}


def create_progress_bar(total: int, prefix: str = "Progress") -> Callable[[int], None]:
    """
    Create a simple progress bar function.
    
    Args:
        total: Total number of items
        prefix: Progress bar prefix
        
    Returns:
        Function to update progress
    """
    def update_progress(current: int):
        percentage = (current / total) * 100
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = '█' * filled + '-' * (bar_length - filled)
        print(f'\r{prefix}: |{bar}| {percentage:.1f}% ({current}/{total})', end='')
        if current == total:
            print()  # New line when complete
    
    return update_progress


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()]
    }
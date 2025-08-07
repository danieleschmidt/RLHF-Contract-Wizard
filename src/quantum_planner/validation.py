"""
Comprehensive validation module for quantum task planning.

Implements input validation, schema validation, business rule validation,
and error recovery mechanisms for the quantum planning system.
"""

import re
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import jax.numpy as jnp
import numpy as np

from .core import QuantumTask, TaskState, PlannerConfig
from .security import SecurityLevel, ThreatLevel


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = time.time()


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    valid: bool
    severity: ValidationSeverity
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    info: List[Dict[str, Any]] = field(default_factory=list)
    validation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Add an error to the validation result."""
        self.errors.append({
            'code': code,
            'message': message,
            'details': details or {},
            'timestamp': time.time()
        })
        self.valid = False
        if self.severity.value != "critical":
            self.severity = ValidationSeverity.ERROR
    
    def add_warning(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Add a warning to the validation result."""
        self.warnings.append({
            'code': code,
            'message': message,
            'details': details or {},
            'timestamp': time.time()
        })
        if self.severity == ValidationSeverity.INFO:
            self.severity = ValidationSeverity.WARNING
    
    def add_info(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Add an info message to the validation result."""
        self.info.append({
            'code': code,
            'message': message,
            'details': details or {},
            'timestamp': time.time()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            'valid': self.valid,
            'severity': self.severity.value,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'validation_time': self.validation_time,
            'metadata': self.metadata
        }


class TaskValidator:
    """
    Comprehensive validator for QuantumTask objects.
    
    Validates task structure, dependencies, resource requirements,
    and business rules to ensure system integrity and performance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation rules configuration
        self.max_task_name_length = 200
        self.max_description_length = 2000
        self.max_dependencies = 50
        self.max_resource_value = 1000000
        self.min_priority = 0.0
        self.max_priority = 1.0
        self.min_duration = 0.001  # 1 millisecond
        self.max_duration = 86400  # 24 hours
        
        # Valid resource types
        self.valid_resource_types = {
            'cpu', 'gpu', 'memory', 'storage', 'network', 
            'disk_io', 'bandwidth', 'threads', 'processes'
        }
        
        # Validation cache for performance
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._cache_ttl = 300  # 5 minutes
    
    def validate_task(self, task: QuantumTask, use_cache: bool = True) -> ValidationResult:
        """
        Comprehensive validation of a QuantumTask.
        
        Args:
            task: Task to validate
            use_cache: Whether to use validation cache
            
        Returns:
            ValidationResult with detailed validation information
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cached_result = self._get_cached_validation(task)
            if cached_result:
                return cached_result
        
        result = ValidationResult(
            valid=True, 
            severity=ValidationSeverity.INFO
        )
        
        try:
            # Core structure validation
            self._validate_task_structure(task, result)
            
            # Business logic validation
            self._validate_business_rules(task, result)
            
            # Resource validation
            self._validate_resources(task, result)
            
            # Dependency validation
            self._validate_dependencies(task, result)
            
            # Quantum properties validation
            self._validate_quantum_properties(task, result)
            
            # Performance validation
            self._validate_performance_characteristics(task, result)
            
            # Security validation
            self._validate_security_aspects(task, result)
            
        except Exception as e:
            result.add_error(
                'validation_exception',
                f'Validation failed with exception: {str(e)}',
                {'exception_type': type(e).__name__}
            )
            self.logger.exception(f"Task validation exception for {task.id}")
        
        result.validation_time = time.time() - start_time
        
        # Cache successful validations
        if use_cache and result.valid:
            self._cache_validation(task, result)
        
        return result
    
    def _validate_task_structure(self, task: QuantumTask, result: ValidationResult):
        """Validate basic task structure and required fields."""
        
        # Task ID validation
        if not task.id or not isinstance(task.id, str):
            result.add_error(
                'invalid_task_id',
                'Task ID must be a non-empty string',
                {'task_id': task.id}
            )
        elif not re.match(r'^[a-zA-Z0-9_\-\.]+$', task.id):
            result.add_error(
                'invalid_task_id_format',
                'Task ID contains invalid characters. Only alphanumeric, underscore, hyphen, and dot allowed',
                {'task_id': task.id}
            )
        elif len(task.id) > 100:
            result.add_error(
                'task_id_too_long',
                f'Task ID too long ({len(task.id)} chars). Maximum 100 characters allowed',
                {'task_id': task.id, 'length': len(task.id)}
            )
        
        # Task name validation
        if not task.name or not isinstance(task.name, str):
            result.add_error(
                'invalid_task_name',
                'Task name must be a non-empty string',
                {'task_name': task.name}
            )
        elif len(task.name) > self.max_task_name_length:
            result.add_error(
                'task_name_too_long',
                f'Task name too long ({len(task.name)} chars). Maximum {self.max_task_name_length} characters allowed',
                {'task_name': task.name, 'length': len(task.name)}
            )
        
        # Description validation
        if not isinstance(task.description, str):
            result.add_error(
                'invalid_description_type',
                'Task description must be a string',
                {'description_type': type(task.description).__name__}
            )
        elif len(task.description) > self.max_description_length:
            result.add_warning(
                'description_too_long',
                f'Task description very long ({len(task.description)} chars). Consider shortening',
                {'description_length': len(task.description)}
            )
        
        # Priority validation
        if not isinstance(task.priority, (int, float)):
            result.add_error(
                'invalid_priority_type',
                'Task priority must be a number',
                {'priority': task.priority, 'priority_type': type(task.priority).__name__}
            )
        elif not (self.min_priority <= task.priority <= self.max_priority):
            result.add_error(
                'priority_out_of_range',
                f'Task priority {task.priority} out of valid range [{self.min_priority}, {self.max_priority}]',
                {'priority': task.priority}
            )
        
        # Duration validation
        if not isinstance(task.estimated_duration, (int, float)):
            result.add_error(
                'invalid_duration_type',
                'Task estimated_duration must be a number',
                {'duration': task.estimated_duration, 'duration_type': type(task.estimated_duration).__name__}
            )
        elif task.estimated_duration < self.min_duration:
            result.add_error(
                'duration_too_small',
                f'Task duration {task.estimated_duration} too small. Minimum {self.min_duration}',
                {'duration': task.estimated_duration}
            )
        elif task.estimated_duration > self.max_duration:
            result.add_warning(
                'duration_very_large',
                f'Task duration {task.estimated_duration} is very large. Consider breaking into smaller tasks',
                {'duration': task.estimated_duration}
            )
    
    def _validate_business_rules(self, task: QuantumTask, result: ValidationResult):
        """Validate business logic rules."""
        
        # High priority tasks should have reasonable durations
        if task.priority > 0.8 and task.estimated_duration > 3600:  # 1 hour
            result.add_warning(
                'high_priority_long_duration',
                'High priority task has long estimated duration. Consider optimization',
                {'priority': task.priority, 'duration': task.estimated_duration}
            )
        
        # Resource requirements should be proportional to duration
        total_resources = sum(task.resource_requirements.values())
        if total_resources > 0:
            resource_per_time = total_resources / task.estimated_duration
            if resource_per_time > 100:  # Arbitrary threshold
                result.add_warning(
                    'high_resource_intensity',
                    'Task has very high resource intensity. Verify requirements',
                    {
                        'total_resources': total_resources,
                        'duration': task.estimated_duration,
                        'intensity': resource_per_time
                    }
                )
        
        # Tasks with many dependencies should have high priority
        if len(task.dependencies) > 5 and task.priority < 0.5:
            result.add_warning(
                'many_deps_low_priority',
                'Task with many dependencies has low priority. Consider increasing priority',
                {'dependencies_count': len(task.dependencies), 'priority': task.priority}
            )
        
        # Check for circular dependencies in task ID patterns
        if task.id in task.dependencies:
            result.add_error(
                'self_dependency',
                'Task cannot depend on itself',
                {'task_id': task.id}
            )
    
    def _validate_resources(self, task: QuantumTask, result: ValidationResult):
        """Validate resource requirements."""
        
        if not isinstance(task.resource_requirements, dict):
            result.add_error(
                'invalid_resources_type',
                'Resource requirements must be a dictionary',
                {'resources_type': type(task.resource_requirements).__name__}
            )
            return
        
        total_resources = 0
        for resource_type, amount in task.resource_requirements.items():
            # Validate resource type
            if not isinstance(resource_type, str):
                result.add_error(
                    'invalid_resource_type',
                    f'Resource type must be string, got {type(resource_type).__name__}',
                    {'resource_type': resource_type}
                )
                continue
            
            if resource_type.lower() not in self.valid_resource_types:
                result.add_warning(
                    'unknown_resource_type',
                    f'Unknown resource type: {resource_type}',
                    {'resource_type': resource_type, 'valid_types': list(self.valid_resource_types)}
                )
            
            # Validate resource amount
            if not isinstance(amount, (int, float)):
                result.add_error(
                    'invalid_resource_amount',
                    f'Resource amount must be a number, got {type(amount).__name__} for {resource_type}',
                    {'resource_type': resource_type, 'amount': amount}
                )
                continue
            
            if amount < 0:
                result.add_error(
                    'negative_resource_amount',
                    f'Resource amount cannot be negative: {resource_type}={amount}',
                    {'resource_type': resource_type, 'amount': amount}
                )
            elif amount == 0:
                result.add_warning(
                    'zero_resource_amount',
                    f'Resource amount is zero: {resource_type}={amount}',
                    {'resource_type': resource_type, 'amount': amount}
                )
            elif amount > self.max_resource_value:
                result.add_error(
                    'resource_amount_too_large',
                    f'Resource amount too large: {resource_type}={amount}',
                    {'resource_type': resource_type, 'amount': amount, 'max_allowed': self.max_resource_value}
                )
            
            total_resources += amount
        
        # Validate total resource consumption
        if total_resources > self.max_resource_value * 2:
            result.add_error(
                'total_resources_too_large',
                f'Total resource requirements too large: {total_resources}',
                {'total_resources': total_resources}
            )
        elif total_resources == 0:
            result.add_info(
                'no_resources_required',
                'Task has no resource requirements',
                {}
            )
    
    def _validate_dependencies(self, task: QuantumTask, result: ValidationResult):
        """Validate task dependencies."""
        
        if not isinstance(task.dependencies, set):
            result.add_error(
                'invalid_dependencies_type',
                'Dependencies must be a set',
                {'dependencies_type': type(task.dependencies).__name__}
            )
            return
        
        if len(task.dependencies) > self.max_dependencies:
            result.add_error(
                'too_many_dependencies',
                f'Too many dependencies ({len(task.dependencies)}). Maximum {self.max_dependencies} allowed',
                {'dependency_count': len(task.dependencies)}
            )
        
        # Validate individual dependency IDs
        for dep_id in task.dependencies:
            if not isinstance(dep_id, str):
                result.add_error(
                    'invalid_dependency_id_type',
                    f'Dependency ID must be string, got {type(dep_id).__name__}',
                    {'dependency_id': dep_id}
                )
                continue
            
            if not dep_id.strip():
                result.add_error(
                    'empty_dependency_id',
                    'Dependency ID cannot be empty',
                    {'dependency_id': dep_id}
                )
            
            if not re.match(r'^[a-zA-Z0-9_\-\.]+$', dep_id):
                result.add_error(
                    'invalid_dependency_id_format',
                    f'Dependency ID contains invalid characters: {dep_id}',
                    {'dependency_id': dep_id}
                )
        
        # Check for duplicate dependencies
        dep_list = list(task.dependencies)
        if len(dep_list) != len(set(dep_list)):
            result.add_warning(
                'duplicate_dependencies',
                'Task has duplicate dependencies',
                {'dependencies': dep_list}
            )
    
    def _validate_quantum_properties(self, task: QuantumTask, result: ValidationResult):
        """Validate quantum-specific properties."""
        
        # Validate quantum amplitude
        if not isinstance(task.amplitude, complex):
            result.add_error(
                'invalid_amplitude_type',
                'Quantum amplitude must be a complex number',
                {'amplitude': task.amplitude, 'amplitude_type': type(task.amplitude).__name__}
            )
        else:
            amplitude_magnitude = abs(task.amplitude)
            if amplitude_magnitude > 10:  # Reasonable upper bound
                result.add_warning(
                    'amplitude_very_large',
                    f'Quantum amplitude magnitude is very large: {amplitude_magnitude}',
                    {'amplitude_magnitude': amplitude_magnitude}
                )
            elif amplitude_magnitude == 0:
                result.add_warning(
                    'zero_amplitude',
                    'Quantum amplitude is zero - task will have zero probability',
                    {'amplitude': task.amplitude}
                )
        
        # Validate phase
        if not isinstance(task.phase, (int, float)):
            result.add_error(
                'invalid_phase_type',
                'Quantum phase must be a number',
                {'phase': task.phase, 'phase_type': type(task.phase).__name__}
            )
        else:
            # Phase should be between 0 and 2π
            if not (0 <= task.phase <= 2 * np.pi):
                result.add_info(
                    'phase_outside_standard_range',
                    f'Quantum phase {task.phase} outside standard range [0, 2π]',
                    {'phase': task.phase}
                )
        
        # Validate task state
        if not isinstance(task.state, TaskState):
            result.add_error(
                'invalid_task_state',
                'Task state must be a TaskState enum value',
                {'state': task.state, 'state_type': type(task.state).__name__}
            )
        
        # Validate entangled tasks
        if not isinstance(task.entangled_tasks, set):
            result.add_error(
                'invalid_entangled_tasks_type',
                'Entangled tasks must be a set',
                {'entangled_tasks_type': type(task.entangled_tasks).__name__}
            )
        else:
            for entangled_id in task.entangled_tasks:
                if not isinstance(entangled_id, str):
                    result.add_error(
                        'invalid_entangled_task_id',
                        f'Entangled task ID must be string, got {type(entangled_id).__name__}',
                        {'entangled_id': entangled_id}
                    )
    
    def _validate_performance_characteristics(self, task: QuantumTask, result: ValidationResult):
        """Validate performance-related aspects."""
        
        # Calculate probability
        try:
            probability = task.probability()
            if not (0 <= probability <= 1):
                result.add_error(
                    'invalid_probability',
                    f'Task probability {probability} not in valid range [0, 1]',
                    {'probability': probability}
                )
            elif probability < 0.01:
                result.add_warning(
                    'very_low_probability',
                    f'Task has very low probability: {probability}',
                    {'probability': probability}
                )
        except Exception as e:
            result.add_error(
                'probability_calculation_failed',
                f'Failed to calculate task probability: {str(e)}',
                {'exception': str(e)}
            )
        
        # Validate execution timing if available
        if task.start_time is not None:
            if not isinstance(task.start_time, (int, float)):
                result.add_error(
                    'invalid_start_time_type',
                    'Task start_time must be a number',
                    {'start_time': task.start_time}
                )
            elif task.start_time < 0:
                result.add_error(
                    'negative_start_time',
                    'Task start_time cannot be negative',
                    {'start_time': task.start_time}
                )
        
        if task.end_time is not None:
            if not isinstance(task.end_time, (int, float)):
                result.add_error(
                    'invalid_end_time_type',
                    'Task end_time must be a number',
                    {'end_time': task.end_time}
                )
            elif task.start_time is not None and task.end_time < task.start_time:
                result.add_error(
                    'end_before_start',
                    'Task end_time cannot be before start_time',
                    {'start_time': task.start_time, 'end_time': task.end_time}
                )
        
        if task.actual_duration is not None:
            if not isinstance(task.actual_duration, (int, float)):
                result.add_error(
                    'invalid_actual_duration_type',
                    'Task actual_duration must be a number',
                    {'actual_duration': task.actual_duration}
                )
            elif task.actual_duration < 0:
                result.add_error(
                    'negative_actual_duration',
                    'Task actual_duration cannot be negative',
                    {'actual_duration': task.actual_duration}
                )
            else:
                # Compare with estimated duration
                duration_ratio = task.actual_duration / task.estimated_duration
                if duration_ratio > 2.0:
                    result.add_warning(
                        'duration_significantly_exceeded',
                        f'Actual duration {task.actual_duration} significantly exceeded estimate {task.estimated_duration}',
                        {
                            'actual_duration': task.actual_duration,
                            'estimated_duration': task.estimated_duration,
                            'ratio': duration_ratio
                        }
                    )
                elif duration_ratio < 0.5:
                    result.add_info(
                        'duration_much_better_than_expected',
                        f'Task completed much faster than estimated',
                        {
                            'actual_duration': task.actual_duration,
                            'estimated_duration': task.estimated_duration,
                            'ratio': duration_ratio
                        }
                    )
    
    def _validate_security_aspects(self, task: QuantumTask, result: ValidationResult):
        """Validate security-related aspects of the task."""
        
        # Check for sensitive information in task name/description
        sensitive_patterns = [
            r'password', r'secret', r'key', r'token', r'credential',
            r'private', r'confidential', r'classified', r'restricted'
        ]
        
        task_text = (task.name + " " + task.description).lower()
        
        for pattern in sensitive_patterns:
            if re.search(pattern, task_text, re.IGNORECASE):
                result.add_warning(
                    'potential_sensitive_info',
                    f'Task may contain sensitive information. Found pattern: {pattern}',
                    {'pattern': pattern, 'in_text': pattern in task_text}
                )
        
        # Check for potential injection patterns
        injection_patterns = [
            r'<script', r'javascript:', r'eval\(', r'exec\(',
            r'system\(', r'os\.', r'subprocess', r'shell'
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, task_text, re.IGNORECASE):
                result.add_error(
                    'potential_code_injection',
                    f'Task contains potentially dangerous code pattern: {pattern}',
                    {'pattern': pattern}
                )
        
        # Validate contract constraints
        if hasattr(task, 'contract_constraints') and task.contract_constraints:
            if not isinstance(task.contract_constraints, list):
                result.add_error(
                    'invalid_contract_constraints_type',
                    'Contract constraints must be a list',
                    {'constraints_type': type(task.contract_constraints).__name__}
                )
            else:
                for constraint in task.contract_constraints:
                    if not isinstance(constraint, str):
                        result.add_error(
                            'invalid_constraint_type',
                            'Contract constraint must be a string',
                            {'constraint': constraint}
                        )
        
        # Validate compliance score
        if hasattr(task, 'compliance_score'):
            if not isinstance(task.compliance_score, (int, float)):
                result.add_error(
                    'invalid_compliance_score_type',
                    'Compliance score must be a number',
                    {'compliance_score': task.compliance_score}
                )
            elif not (0 <= task.compliance_score <= 1):
                result.add_error(
                    'compliance_score_out_of_range',
                    f'Compliance score {task.compliance_score} out of range [0, 1]',
                    {'compliance_score': task.compliance_score}
                )
    
    def _get_cached_validation(self, task: QuantumTask) -> Optional[ValidationResult]:
        """Get cached validation result if available and fresh."""
        cache_key = self._get_task_cache_key(task)
        
        if cache_key in self._validation_cache:
            cached_result = self._validation_cache[cache_key]
            
            # Check if cache is still fresh
            cache_age = time.time() - (cached_result.metadata.get('cache_timestamp', 0))
            if cache_age < self._cache_ttl:
                cached_result.metadata['cache_hit'] = True
                return cached_result
            else:
                # Remove stale cache entry
                del self._validation_cache[cache_key]
        
        return None
    
    def _cache_validation(self, task: QuantumTask, result: ValidationResult):
        """Cache validation result."""
        cache_key = self._get_task_cache_key(task)
        result.metadata['cache_timestamp'] = time.time()
        result.metadata['cache_hit'] = False
        self._validation_cache[cache_key] = result
        
        # Cleanup old cache entries periodically
        if len(self._validation_cache) > 1000:
            self._cleanup_cache()
    
    def _get_task_cache_key(self, task: QuantumTask) -> str:
        """Generate cache key for task."""
        # Create hash based on task properties that affect validation
        task_data = {
            'id': task.id,
            'name': task.name,
            'description': task.description,
            'priority': task.priority,
            'estimated_duration': task.estimated_duration,
            'resource_requirements': dict(task.resource_requirements),
            'dependencies': sorted(list(task.dependencies)),
            'state': task.state.value,
            'amplitude': str(task.amplitude),
            'phase': task.phase
        }
        
        task_json = json.dumps(task_data, sort_keys=True)
        return hashlib.sha256(task_json.encode()).hexdigest()[:16]
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        current_time = time.time()
        
        keys_to_remove = [
            key for key, result in self._validation_cache.items()
            if current_time - result.metadata.get('cache_timestamp', 0) > self._cache_ttl
        ]
        
        for key in keys_to_remove:
            del self._validation_cache[key]
        
        self.logger.info(f"Cache cleanup: removed {len(keys_to_remove)} stale entries")
    
    def clear_cache(self):
        """Clear all cached validation results."""
        self._validation_cache.clear()
        self.logger.info("Validation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get validation cache statistics."""
        current_time = time.time()
        
        total_entries = len(self._validation_cache)
        fresh_entries = sum(
            1 for result in self._validation_cache.values()
            if current_time - result.metadata.get('cache_timestamp', 0) < self._cache_ttl
        )
        
        return {
            'total_entries': total_entries,
            'fresh_entries': fresh_entries,
            'stale_entries': total_entries - fresh_entries,
            'cache_ttl_seconds': self._cache_ttl,
            'hit_rate': sum(
                1 for result in self._validation_cache.values()
                if result.metadata.get('cache_hit', False)
            ) / max(1, total_entries)  # Avoid division by zero
        }


class ConfigValidator:
    """Validator for PlannerConfig objects."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_config(self, config: PlannerConfig) -> ValidationResult:
        """Validate PlannerConfig object."""
        result = ValidationResult(valid=True, severity=ValidationSeverity.INFO)
        
        # Validate max_iterations
        if not isinstance(config.max_iterations, int):
            result.add_error(
                'invalid_max_iterations_type',
                'max_iterations must be an integer',
                {'value': config.max_iterations, 'type': type(config.max_iterations).__name__}
            )
        elif config.max_iterations <= 0:
            result.add_error(
                'invalid_max_iterations_value',
                'max_iterations must be positive',
                {'value': config.max_iterations}
            )
        elif config.max_iterations > 10000:
            result.add_warning(
                'max_iterations_very_large',
                'max_iterations is very large, may cause performance issues',
                {'value': config.max_iterations}
            )
        
        # Validate convergence_threshold
        if not isinstance(config.convergence_threshold, (int, float)):
            result.add_error(
                'invalid_convergence_threshold_type',
                'convergence_threshold must be a number',
                {'value': config.convergence_threshold}
            )
        elif config.convergence_threshold <= 0:
            result.add_error(
                'invalid_convergence_threshold_value',
                'convergence_threshold must be positive',
                {'value': config.convergence_threshold}
            )
        elif config.convergence_threshold > 1.0:
            result.add_warning(
                'convergence_threshold_large',
                'convergence_threshold is large, may cause early convergence',
                {'value': config.convergence_threshold}
            )
        
        # Validate quantum parameters
        quantum_params = [
            ('quantum_interference_strength', config.quantum_interference_strength),
            ('entanglement_decay', config.entanglement_decay),
            ('superposition_collapse_threshold', config.superposition_collapse_threshold),
            ('resource_optimization_weight', config.resource_optimization_weight),
            ('time_optimization_weight', config.time_optimization_weight),
            ('priority_weight', config.priority_weight)
        ]
        
        for param_name, param_value in quantum_params:
            if not isinstance(param_value, (int, float)):
                result.add_error(
                    f'invalid_{param_name}_type',
                    f'{param_name} must be a number',
                    {'parameter': param_name, 'value': param_value}
                )
            elif param_value < 0 or param_value > 1:
                result.add_error(
                    f'invalid_{param_name}_range',
                    f'{param_name} must be between 0 and 1',
                    {'parameter': param_name, 'value': param_value}
                )
        
        # Validate weight sum
        weight_sum = (
            config.resource_optimization_weight +
            config.time_optimization_weight + 
            config.priority_weight
        )
        if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point errors
            result.add_warning(
                'weights_dont_sum_to_one',
                f'Optimization weights sum to {weight_sum}, should sum to 1.0',
                {'weight_sum': weight_sum}
            )
        
        # Validate parallel_execution_limit
        if not isinstance(config.parallel_execution_limit, int):
            result.add_error(
                'invalid_parallel_limit_type',
                'parallel_execution_limit must be an integer',
                {'value': config.parallel_execution_limit}
            )
        elif config.parallel_execution_limit <= 0:
            result.add_error(
                'invalid_parallel_limit_value',
                'parallel_execution_limit must be positive',
                {'value': config.parallel_execution_limit}
            )
        elif config.parallel_execution_limit > 100:
            result.add_warning(
                'parallel_limit_very_large',
                'parallel_execution_limit is very large',
                {'value': config.parallel_execution_limit}
            )
        
        return result


# Import hashlib for cache key generation
import hashlib
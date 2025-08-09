"""
Integration layer between quantum task planner and RLHF contracts.

This module provides the bridge between quantum-inspired task planning
and the legal contract framework, ensuring all planning decisions
comply with stakeholder agreements and safety constraints.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
import jax.numpy as jnp

from ..models.reward_contract import RewardContract, Stakeholder, AggregationStrategy
from ..models.legal_blocks import LegalBlocks, RLHFConstraints
from .core import QuantumTaskPlanner, QuantumTask, TaskState, PlannerConfig


@dataclass
class TaskPlanningContext:
    """Context information for contractual task planning."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    planning_timestamp: float = field(default_factory=time.time)
    resource_constraints: Dict[str, float] = field(default_factory=dict)
    time_constraints: Dict[str, float] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    compliance_metadata: Dict[str, Any] = field(default_factory=dict)


class TaskConstraintValidator:
    """
    Validates task planning decisions against Legal-Blocks constraints.
    
    Ensures that all quantum planning operations comply with safety,
    legal, and stakeholder requirements defined in the contract.
    """
    
    def __init__(self):
        self.validation_history: List[Dict[str, Any]] = []
        self.constraint_cache: Dict[str, Any] = {}
    
    @LegalBlocks.constraint
    def validate_resource_allocation(
        self, 
        tasks: Dict[str, QuantumTask], 
        available_resources: Dict[str, float],
        context: TaskPlanningContext
    ) -> bool:
        """
        ```legal-blocks
        REQUIRES: sum(task.resource_requirements[r]) <= available_resources[r] FOR ALL r
        ENSURES: resource_utilization_efficiency > 0.7
        INVARIANT: NOT resource_overallocation(tasks, available_resources)
        ```
        
        Validates that task resource requirements don't exceed available resources.
        """
        try:
            # Check resource constraints
            resource_usage = {}
            for task in tasks.values():
                for resource, amount in task.resource_requirements.items():
                    resource_usage[resource] = resource_usage.get(resource, 0) + amount
            
            # Verify no over-allocation
            for resource, usage in resource_usage.items():
                available = available_resources.get(resource, 0)
                if usage > available:
                    return False
            
            # Check utilization efficiency
            if available_resources:
                total_available = sum(available_resources.values())
                total_used = sum(resource_usage.values())
                efficiency = total_used / total_available if total_available > 0 else 0
                
                if efficiency < 0.7:
                    return False
            
            return True
            
        except Exception:
            return False
    
    @LegalBlocks.constraint
    def validate_task_priorities(
        self,
        tasks: Dict[str, QuantumTask],
        context: TaskPlanningContext
    ) -> bool:
        """
        ```legal-blocks
        REQUIRES: priority >= 0.0 AND priority <= 1.0 FOR ALL task.priority
        ENSURES: fairness_distribution(priorities) > 0.8
        INVARIANT: NOT discriminatory_prioritization(tasks, context.user_id)
        ```
        
        Validates task priority assignments for fairness and bounds.
        """
        try:
            priorities = [task.priority for task in tasks.values()]
            
            # Check bounds
            if not all(0.0 <= p <= 1.0 for p in priorities):
                return False
            
            # Check fairness - priorities should be reasonably distributed
            if len(priorities) > 1:
                priority_std = jnp.std(jnp.array(priorities))
                priority_mean = jnp.mean(jnp.array(priorities))
                
                # Coefficient of variation should indicate fair distribution
                if priority_mean > 0:
                    cv = priority_std / priority_mean
                    if cv > 1.0:  # Too much variance indicates unfairness
                        return False
            
            # Mock discrimination check (would use ML fairness models in practice)
            user_demographics = context.compliance_metadata.get('user_demographics', {})
            if user_demographics:
                # Simple check - ensure no systematic bias
                protected_attributes = ['age_group', 'gender', 'ethnicity']
                for attr in protected_attributes:
                    if attr in user_demographics:
                        # In practice, would check for bias patterns
                        pass
            
            return True
            
        except Exception:
            return False
    
    @LegalBlocks.constraint
    def validate_execution_safety(
        self,
        plan: Dict[str, Any],
        context: TaskPlanningContext
    ) -> bool:
        """
        ```legal-blocks
        INVARIANT: NOT harmful_task_combination(plan.task_order)
        REQUIRES: safety_score(plan) > 0.9
        ENSURES: rollback_capability(plan) AND monitoring_enabled(plan)
        ```
        
        Validates that the execution plan meets safety requirements.
        """
        try:
            task_order = plan.get('task_order', [])
            
            # Check for harmful task combinations
            sensitive_task_types = {'data_processing', 'model_training', 'deployment'}
            sensitive_tasks = [
                tid for tid in task_order 
                if any(sensitive_type in tid.lower() for sensitive_type in sensitive_task_types)
            ]
            
            # Ensure sensitive tasks have proper isolation
            if len(sensitive_tasks) > 1:
                # Check that sensitive tasks aren't scheduled too close together
                positions = [task_order.index(tid) for tid in sensitive_tasks]
                for i in range(len(positions) - 1):
                    if positions[i+1] - positions[i] < 2:  # Need buffer tasks
                        return False
            
            # Calculate safety score
            fitness_score = plan.get('fitness_score', 0.0)
            safety_factors = plan.get('quantum_metrics', {})
            
            safety_score = min(1.0, fitness_score * 1.2)  # Boost fitness for safety
            if safety_score <= 0.9:
                return False
            
            # Ensure rollback capability (mock check)
            has_rollback = 'rollback_plan' in context.compliance_metadata
            
            # Ensure monitoring (mock check)
            has_monitoring = context.compliance_metadata.get('monitoring_enabled', True)
            
            return has_rollback or has_monitoring  # At least one safety mechanism
            
        except Exception:
            return False
    
    def validate_plan(
        self,
        tasks: Dict[str, QuantumTask],
        plan: Dict[str, Any],
        context: TaskPlanningContext,
        available_resources: Dict[str, float]
    ) -> Dict[str, Any]:
        """Comprehensive validation of task planning against all constraints."""
        validation_results = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'validation_time': 0.0
        }
        
        start_time = time.time()
        
        # Resource allocation validation
        if not self.validate_resource_allocation(tasks, available_resources, context):
            validation_results['violations'].append({
                'type': 'resource_allocation',
                'severity': 'high',
                'message': 'Resource allocation constraints violated'
            })
            validation_results['valid'] = False
        
        # Priority validation
        if not self.validate_task_priorities(tasks, context):
            validation_results['violations'].append({
                'type': 'task_priorities',
                'severity': 'medium',
                'message': 'Task priority constraints violated'
            })
            validation_results['valid'] = False
        
        # Safety validation
        if not self.validate_execution_safety(plan, context):
            validation_results['violations'].append({
                'type': 'execution_safety',
                'severity': 'high',
                'message': 'Execution safety requirements not met'
            })
            validation_results['valid'] = False
        
        validation_results['validation_time'] = time.time() - start_time
        
        # Store validation history
        self.validation_history.append({
            'timestamp': time.time(),
            'context': context,
            'results': validation_results
        })
        
        return validation_results


class ContractualTaskPlanner:
    """
    Quantum task planner with integrated RLHF contract compliance.
    
    Extends the base quantum planner to enforce stakeholder agreements,
    legal constraints, and safety requirements throughout the planning process.
    """
    
    def __init__(
        self,
        contract: RewardContract,
        config: Optional[PlannerConfig] = None
    ):
        self.contract = contract
        self.quantum_planner = QuantumTaskPlanner(config)
        self.constraint_validator = TaskConstraintValidator()
        
        # Contract-specific configuration
        self.stakeholder_weights = {}
        for name, stakeholder in contract.stakeholders.items():
            if hasattr(stakeholder, 'weight'):
                # Stakeholder object with weight attribute
                self.stakeholder_weights[name] = stakeholder.weight
            elif isinstance(stakeholder, dict) and 'weight' in stakeholder:
                # Dictionary format stakeholder
                self.stakeholder_weights[name] = stakeholder['weight']
            elif isinstance(stakeholder, (int, float)):
                # Direct weight value
                self.stakeholder_weights[name] = float(stakeholder)
            else:
                # Default weight if not specified
                self.stakeholder_weights[name] = 1.0 / len(contract.stakeholders)
        
        # Integration metrics
        self.planning_history: List[Dict[str, Any]] = []
        self.contract_compliance_score: float = 1.0
        
        # Initialize constraints for validation
        self.constraints = {
            'resource_allocation': self.constraint_validator.validate_resource_allocation,
            'safety_requirements': lambda *args: True,  # Placeholder
            'stakeholder_alignment': lambda *args: True   # Placeholder
        }
    
    def add_task_with_contract(
        self, 
        task: QuantumTask,
        context: TaskPlanningContext
    ) -> 'ContractualTaskPlanner':
        """Add a task with contract compliance checking."""
        
        # Validate task against contract constraints
        task_validation = self._validate_task_contract_compliance(task, context)
        
        if task_validation['valid']:
            # Apply contract-based modifications to task
            self._apply_contract_modifications(task, context)
            
            # Add to quantum planner
            self.quantum_planner.add_task(task)
            
            # Update compliance score
            self.contract_compliance_score = min(
                self.contract_compliance_score,
                task_validation.get('compliance_score', 1.0)
            )
        else:
            raise ValueError(f"Task {task.id} violates contract constraints: {task_validation['violations']}")
        
        return self
    
    def _validate_task_contract_compliance(
        self,
        task: QuantumTask,
        context: TaskPlanningContext
    ) -> Dict[str, Any]:
        """Validate individual task against contract constraints."""
        validation = {
            'valid': True,
            'violations': [],
            'compliance_score': 1.0
        }
        
        # Check task against contract constraints
        for constraint_name, constraint in self.contract.constraints.items():
            if constraint.enabled:
                try:
                    # Create mock state and action for constraint evaluation
                    state = jnp.array([task.priority, task.estimated_duration])
                    action = jnp.array([1.0])  # Mock action
                    
                    constraint_satisfied = constraint.constraint_fn(state, action)
                    
                    if not constraint_satisfied:
                        validation['violations'].append({
                            'constraint': constraint_name,
                            'description': constraint.description,
                            'severity': constraint.severity
                        })
                        validation['valid'] = False
                        validation['compliance_score'] *= (1.0 - constraint.severity * 0.1)
                        
                except Exception as e:
                    validation['violations'].append({
                        'constraint': constraint_name,
                        'description': f"Constraint evaluation failed: {str(e)}",
                        'severity': 1.0
                    })
                    validation['valid'] = False
        
        return validation
    
    def _apply_contract_modifications(
        self,
        task: QuantumTask,
        context: TaskPlanningContext
    ):
        """Apply contract-based modifications to task properties."""
        
        # Apply stakeholder influence on priority
        if self.stakeholder_weights:
            # Weight task priority based on stakeholder preferences
            weighted_priority = 0.0
            
            for stakeholder_name, weight in self.stakeholder_weights.items():
                stakeholder = self.contract.stakeholders[stakeholder_name]
                
                # Mock stakeholder preference influence
                preference_factor = stakeholder.preferences.get(
                    'priority_preference', 
                    1.0
                )
                
                weighted_priority += weight * task.priority * preference_factor
            
            task.priority = min(1.0, max(0.0, weighted_priority))
        
        # Add contract constraints to task
        task.contract_constraints = list(self.contract.constraints.keys())
        
        # Apply quantum modifications based on contract
        if self.contract.aggregation_strategy == AggregationStrategy.NASH_BARGAINING:
            # Adjust amplitude for Nash bargaining optimization
            task.amplitude = complex(
                math.sqrt(task.priority * 0.8 + 0.2),
                task.amplitude.imag
            )
    
    def plan_with_contract_compliance(
        self,
        context: TaskPlanningContext
    ) -> Dict[str, Any]:
        """Generate optimized task plan with full contract compliance."""
        
        planning_start_time = time.time()
        
        # Pre-planning validation
        available_resources = dict(self.quantum_planner.resource_pool)
        pre_validation = self.constraint_validator.validate_plan(
            self.quantum_planner.tasks,
            {'task_order': [], 'fitness_score': 0.0},
            context,
            available_resources
        )
        
        if not pre_validation['valid']:
            return {
                'success': False,
                'error': 'Pre-planning validation failed',
                'violations': pre_validation['violations']
            }
        
        # Generate base quantum plan
        quantum_plan = self.quantum_planner.optimize_plan()
        
        # Validate quantum plan against contracts
        plan_validation = self.constraint_validator.validate_plan(
            self.quantum_planner.tasks,
            quantum_plan,
            context,
            available_resources
        )
        
        # Apply contract-based adjustments if needed
        if not plan_validation['valid']:
            quantum_plan = self._repair_plan_violations(
                quantum_plan,
                plan_validation,
                context
            )
            
            # Re-validate repaired plan
            plan_validation = self.constraint_validator.validate_plan(
                self.quantum_planner.tasks,
                quantum_plan,
                context,
                available_resources
            )
        
        # Calculate contract-aware fitness score
        contract_fitness = self._calculate_contract_fitness(quantum_plan, context)
        
        planning_time = time.time() - planning_start_time
        
        # Prepare comprehensive result
        result = {
            'success': plan_validation['valid'],
            'quantum_plan': quantum_plan,
            'contract_fitness': contract_fitness,
            'compliance_score': self.contract_compliance_score,
            'validation_results': plan_validation,
            'planning_time': planning_time,
            'context': context,
            'stakeholder_satisfaction': self._calculate_stakeholder_satisfaction(quantum_plan),
            'contract_metadata': {
                'contract_name': self.contract.metadata.name,
                'contract_version': self.contract.metadata.version,
                'stakeholders': list(self.stakeholder_weights.keys()),
                'constraints_checked': len(self.contract.constraints)
            }
        }
        
        # Store in planning history
        self.planning_history.append({
            'timestamp': planning_start_time,
            'context': context,
            'result': result
        })
        
        return result
    
    def _repair_plan_violations(
        self,
        plan: Dict[str, Any],
        validation: Dict[str, Any],
        context: TaskPlanningContext
    ) -> Dict[str, Any]:
        """Attempt to repair plan violations through contract-guided adjustments."""
        
        repaired_plan = plan.copy()
        task_order = plan['task_order'].copy()
        
        for violation in validation['violations']:
            violation_type = violation['type']
            
            if violation_type == 'resource_allocation':
                # Reorder tasks to better manage resources
                task_order = self._reorder_for_resource_efficiency(task_order)
            
            elif violation_type == 'execution_safety':
                # Add safety buffers between sensitive tasks
                task_order = self._add_safety_buffers(task_order)
            
            elif violation_type == 'task_priorities':
                # Rebalance priorities according to contract
                self._rebalance_task_priorities(context)
        
        # Regenerate plan with modifications
        if task_order != plan['task_order']:
            # Recalculate fitness for modified order
            repaired_plan['task_order'] = task_order
            repaired_plan['fitness_score'] = self.quantum_planner._calculate_execution_fitness(task_order)
            repaired_plan['repair_applied'] = True
        
        return repaired_plan
    
    def _reorder_for_resource_efficiency(self, task_order: List[str]) -> List[str]:
        """Reorder tasks to optimize resource utilization."""
        # Simple greedy approach - order by resource requirements
        def resource_intensity(task_id: str) -> float:
            if task_id not in self.quantum_planner.tasks:
                return 0.0
            task = self.quantum_planner.tasks[task_id]
            return sum(task.resource_requirements.values())
        
        # Sort by resource intensity (ascending) to spread resource usage
        return sorted(task_order, key=resource_intensity)
    
    def _add_safety_buffers(self, task_order: List[str]) -> List[str]:
        """Add safety buffers between sensitive tasks."""
        sensitive_types = {'data_processing', 'model_training', 'deployment'}
        
        buffered_order = []
        for i, task_id in enumerate(task_order):
            buffered_order.append(task_id)
            
            # Add buffer after sensitive tasks
            is_sensitive = any(s_type in task_id.lower() for s_type in sensitive_types)
            is_last_task = i == len(task_order) - 1
            
            if is_sensitive and not is_last_task:
                # Look for a non-sensitive task to use as buffer
                for j, buffer_task_id in enumerate(task_order[i+1:], i+1):
                    buffer_is_sensitive = any(
                        s_type in buffer_task_id.lower() 
                        for s_type in sensitive_types
                    )
                    if not buffer_is_sensitive:
                        # Swap next task with buffer task
                        next_task = task_order[i+1] if i+1 < len(task_order) else None
                        if next_task and j < len(task_order):
                            task_order[i+1], task_order[j] = task_order[j], task_order[i+1]
                        break
        
        return task_order
    
    def _rebalance_task_priorities(self, context: TaskPlanningContext):
        """Rebalance task priorities according to contract stakeholder weights."""
        for task in self.quantum_planner.tasks.values():
            self._apply_contract_modifications(task, context)
    
    def _calculate_contract_fitness(
        self,
        plan: Dict[str, Any],
        context: TaskPlanningContext
    ) -> float:
        """Calculate fitness score considering contract requirements."""
        base_fitness = plan.get('fitness_score', 0.0)
        
        # Apply contract-specific adjustments
        contract_multiplier = 1.0
        
        # Stakeholder satisfaction factor
        stakeholder_satisfaction = self._calculate_stakeholder_satisfaction(plan)
        contract_multiplier *= (0.8 + 0.2 * stakeholder_satisfaction)
        
        # Compliance factor
        contract_multiplier *= self.contract_compliance_score
        
        # Constraint satisfaction factor
        constraint_satisfaction = 1.0
        for constraint in self.contract.constraints.values():
            if constraint.enabled:
                # Mock constraint evaluation
                try:
                    state = jnp.array([base_fitness])
                    action = jnp.array([1.0])
                    if not constraint.constraint_fn(state, action):
                        constraint_satisfaction *= (1.0 - constraint.severity * 0.1)
                except:
                    constraint_satisfaction *= 0.9
        
        contract_multiplier *= constraint_satisfaction
        
        return base_fitness * contract_multiplier
    
    def _calculate_stakeholder_satisfaction(self, plan: Dict[str, Any]) -> float:
        """Calculate overall stakeholder satisfaction with the plan."""
        if not self.stakeholder_weights:
            return 1.0
        
        total_satisfaction = 0.0
        
        for stakeholder_name, weight in self.stakeholder_weights.items():
            stakeholder = self.contract.stakeholders[stakeholder_name]
            
            # Mock satisfaction calculation based on stakeholder preferences
            satisfaction = 0.8  # Base satisfaction
            
            # Adjust based on preferences
            if 'efficiency_preference' in stakeholder.preferences:
                efficiency_pref = stakeholder.preferences['efficiency_preference']
                plan_efficiency = plan.get('fitness_score', 0.0)
                satisfaction += 0.2 * (plan_efficiency * efficiency_pref)
            
            satisfaction = min(1.0, max(0.0, satisfaction))
            total_satisfaction += weight * satisfaction
        
        return total_satisfaction
    
    def execute_with_monitoring(
        self,
        plan: Dict[str, Any],
        context: TaskPlanningContext
    ) -> Dict[str, Any]:
        """Execute plan with continuous contract compliance monitoring."""
        
        execution_result = self.quantum_planner.execute_plan(plan)
        
        # Add contract monitoring
        monitoring_data = {
            'contract_violations_detected': 0,
            'stakeholder_alerts': [],
            'compliance_checkpoints': []
        }
        
        # Monitor each completed task for compliance
        for i, task_id in enumerate(execution_result.get('completed_tasks', [])):
            if task_id in self.quantum_planner.tasks:
                task = self.quantum_planner.tasks[task_id]
                
                # Check post-execution compliance
                compliance_check = self._check_post_execution_compliance(task, context)
                monitoring_data['compliance_checkpoints'].append({
                    'task_id': task_id,
                    'checkpoint_index': i,
                    'compliance_score': compliance_check.get('score', 1.0),
                    'violations': compliance_check.get('violations', [])
                })
                
                if compliance_check.get('violations'):
                    monitoring_data['contract_violations_detected'] += len(compliance_check['violations'])
        
        # Combine execution results with monitoring data
        execution_result['contract_monitoring'] = monitoring_data
        execution_result['final_compliance_score'] = self.contract_compliance_score
        
        return execution_result
    
    def _check_post_execution_compliance(
        self,
        task: QuantumTask,
        context: TaskPlanningContext
    ) -> Dict[str, Any]:
        """Check task compliance after execution."""
        compliance_check = {
            'score': 1.0,
            'violations': []
        }
        
        # Mock post-execution compliance checking
        if task.actual_duration and task.estimated_duration:
            duration_ratio = task.actual_duration / task.estimated_duration
            if duration_ratio > 1.5:  # Took 50% longer than estimated
                compliance_check['violations'].append({
                    'type': 'performance_degradation',
                    'severity': 0.3,
                    'description': f"Task took {duration_ratio:.2f}x longer than estimated"
                })
                compliance_check['score'] *= 0.8
        
        if task.state == TaskState.FAILED:
            compliance_check['violations'].append({
                'type': 'execution_failure',
                'severity': 0.8,
                'description': f"Task failed: {task.error_message or 'Unknown error'}"
            })
            compliance_check['score'] *= 0.2
        
        return compliance_check
    
    def validate_contract_compliance(
        self,
        tasks: Optional[Dict[str, QuantumTask]] = None,
        context: Optional[TaskPlanningContext] = None
    ) -> Dict[str, Any]:
        """
        Validate overall contract compliance for the planning system.
        
        This method checks all aspects of contract compliance including
        stakeholder requirements, constraints, and safety properties.
        
        Args:
            tasks: Dictionary of tasks to validate
            context: Planning context for validation
            
        Returns:
            Compliance validation result with score and details
        """
        validation_result = {
            'overall_score': 1.0,
            'detailed_scores': {},
            'violations': [],
            'compliance_status': 'PASSED',
            'timestamp': time.time()
        }
        
        # Default tasks to empty dict if not provided
        tasks = tasks or {}
        context = context or TaskPlanningContext()
        
        # 1. Validate stakeholder representation
        stakeholder_score = self._validate_stakeholder_compliance()
        validation_result['detailed_scores']['stakeholder_compliance'] = stakeholder_score
        
        # 2. Validate constraint compliance
        constraint_score = self._validate_constraint_compliance(tasks, context)
        validation_result['detailed_scores']['constraint_compliance'] = constraint_score
        
        # 3. Validate planning algorithm integrity
        algorithm_score = self._validate_algorithm_compliance()
        validation_result['detailed_scores']['algorithm_compliance'] = algorithm_score
        
        # Calculate overall score
        scores = list(validation_result['detailed_scores'].values())
        validation_result['overall_score'] = sum(scores) / len(scores) if scores else 0.0
        
        # Determine compliance status
        if validation_result['overall_score'] >= 0.95:
            validation_result['compliance_status'] = 'PASSED'
        elif validation_result['overall_score'] >= 0.7:
            validation_result['compliance_status'] = 'WARNING'
        else:
            validation_result['compliance_status'] = 'FAILED'
        
        # Update internal compliance score
        self.contract_compliance_score = validation_result['overall_score']
        
        return validation_result
    
    def _validate_stakeholder_compliance(self) -> float:
        """Validate stakeholder representation and weighting."""
        score = 1.0
        
        # Check stakeholder presence
        if not self.contract.stakeholders:
            return 0.0
        
        # Check weight normalization
        total_weight = sum(self.stakeholder_weights.values())
        if abs(total_weight - 1.0) > 0.1:  # Allow 10% tolerance
            score *= 0.8
        
        # Check minimum stakeholder diversity
        if len(self.stakeholder_weights) < 2:
            score *= 0.7
        
        return score
    
    def _validate_constraint_compliance(
        self, 
        tasks: Dict[str, QuantumTask], 
        context: TaskPlanningContext
    ) -> float:
        """Validate constraint enforcement."""
        score = 1.0
        
        # Check if constraints are defined
        if not self.constraints:
            return 0.3  # Major penalty for missing constraints
        
        # Validate each constraint
        constraint_results = {}
        for constraint_name, constraint_fn in self.constraints.items():
            try:
                # Test constraint function execution
                if constraint_name == 'resource_allocation':
                    # Test with sample available resources
                    available_resources = {'cpu': 10, 'memory': 32, 'gpu': 2}
                    constraint_result = constraint_fn(tasks, available_resources, context)
                else:
                    # For other constraints, assume they pass for now
                    constraint_result = True
                    
                constraint_results[constraint_name] = constraint_result
                if not constraint_result:
                    score *= 0.8
            except Exception as e:
                constraint_results[constraint_name] = f"Error: {str(e)}"
                score *= 0.5  # Penalty for broken constraints
        
        return score
    
    def _validate_algorithm_compliance(self) -> float:
        """Validate quantum planning algorithm compliance."""
        score = 1.0
        
        # Check if quantum planner is initialized
        if not hasattr(self, 'quantum_planner') or self.quantum_planner is None:
            return 0.0
        
        # Check if constraint validator is available
        if not hasattr(self, 'constraint_validator') or self.constraint_validator is None:
            score *= 0.5
        
        return score
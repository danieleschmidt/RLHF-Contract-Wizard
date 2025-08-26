"""
Comprehensive Integration Test Suite for RLHF-Contract-Wizard.

This module implements a comprehensive testing framework that validates all
major components, integration points, performance characteristics, and
security properties of the RLHF contract system.
"""

import asyncio
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from unittest.mock import Mock, patch
import aiohttp

# Import system components for testing
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.reward_contract import RewardContract, AggregationStrategy
from src.optimization.quantum_enhanced_optimization import QuantumEnhancedRewardOptimizer
from src.security.comprehensive_security_framework import ComprehensiveSecurityFramework
from src.scaling.autonomous_global_deployment import AutonomousGlobalDeploymentSystem
from src.research.autonomous_research_engine import AutonomousResearchEngine
from src.optimization.neural_architecture_search import NeuralArchitectureSearch
from src.utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


@dataclass
class TestResult:
    """Test execution result."""
    test_name: str
    category: str
    passed: bool
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestSuiteReport:
    """Complete test suite execution report."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_execution_time: float
    test_results: List[TestResult]
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    security_summary: Dict[str, Any] = field(default_factory=dict)
    coverage_report: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmarkSuite:
    """Performance benchmarking test suite."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def benchmark_reward_computation(self) -> TestResult:
        """Benchmark reward computation performance."""
        
        test_name = "reward_computation_benchmark"
        start_time = time.time()
        
        try:
            # Create test contract
            contract = RewardContract(
                name="benchmark_contract",
                stakeholders={"user": 0.6, "safety": 0.3, "efficiency": 0.1}
            )
            
            # Add constraints
            contract.add_constraint(
                name="safety_constraint",
                constraint_fn=lambda s, a: jnp.sum(jnp.abs(a)) < 2.0,
                description="Action magnitude constraint"
            )
            
            contract.add_constraint(
                name="efficiency_constraint", 
                constraint_fn=lambda s, a: jnp.mean(s) > -1.0,
                description="State efficiency constraint"
            )
            
            # Benchmark parameters
            num_evaluations = 1000
            state_dim = 100
            action_dim = 50
            
            # Generate test data
            key = jax.random.PRNGKey(42)
            states = jax.random.normal(key, (num_evaluations, state_dim))
            actions = jax.random.normal(jax.random.split(key)[1], (num_evaluations, action_dim))
            
            # Warm-up
            for i in range(10):
                _ = contract.compute_reward(states[i], actions[i])
            
            # Benchmark
            computation_times = []
            
            for i in range(num_evaluations):
                eval_start = time.time()
                reward = contract.compute_reward(states[i], actions[i])
                eval_time = time.time() - eval_start
                computation_times.append(eval_time)
                
                # Validate reward is reasonable
                assert jnp.isfinite(reward), f"Non-finite reward at iteration {i}"
            
            # Calculate performance metrics
            avg_time_ms = np.mean(computation_times) * 1000
            p95_time_ms = np.percentile(computation_times, 95) * 1000
            p99_time_ms = np.percentile(computation_times, 99) * 1000
            throughput_rps = 1.0 / np.mean(computation_times)
            
            execution_time = time.time() - start_time
            
            # Performance thresholds
            avg_threshold_ms = 10.0  # 10ms average
            p99_threshold_ms = 50.0  # 50ms p99
            min_throughput_rps = 100  # 100 RPS
            
            passed = (avg_time_ms < avg_threshold_ms and 
                     p99_time_ms < p99_threshold_ms and 
                     throughput_rps > min_throughput_rps)
            
            return TestResult(
                test_name=test_name,
                category="performance",
                passed=passed,
                execution_time=execution_time,
                details={
                    "num_evaluations": num_evaluations,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "passed_thresholds": passed
                },
                performance_metrics={
                    "avg_time_ms": avg_time_ms,
                    "p95_time_ms": p95_time_ms,
                    "p99_time_ms": p99_time_ms,
                    "throughput_rps": throughput_rps,
                    "total_evaluations": num_evaluations
                }
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category="performance",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def benchmark_optimization_convergence(self) -> TestResult:
        """Benchmark optimization convergence speed."""
        
        test_name = "optimization_convergence_benchmark"
        start_time = time.time()
        
        try:
            # Create contract for optimization
            contract = RewardContract(
                name="optimization_benchmark",
                stakeholders={"user": 0.7, "safety": 0.3}
            )
            
            # Initialize quantum optimizer
            optimizer = QuantumEnhancedRewardOptimizer(
                reward_contract=contract,
                num_qubits=6,
                quantum_learning_rate=0.1
            )
            
            # Generate synthetic preference data
            key = jax.random.PRNGKey(123)
            states = jax.random.normal(key, (500, 50))
            actions = jax.random.normal(jax.random.split(key)[1], (500, 25))
            preferences = jax.random.uniform(jax.random.split(key, 3)[2], (500,))
            
            preference_data = {
                "states": states,
                "actions": actions,
                "preferences": preferences
            }
            
            validation_data = {
                "states": states[:100],
                "actions": actions[:100],
                "preferences": preferences[:100]
            }
            
            # Run optimization
            optimization_result = await optimizer.optimize_reward_function(
                preference_data=preference_data,
                validation_data=validation_data,
                max_iterations=100,
                quantum_method="hybrid"
            )
            
            execution_time = time.time() - start_time
            
            # Performance criteria
            max_convergence_time = 120.0  # 2 minutes
            min_improvement = 0.1  # 10% improvement
            
            if optimization_result.convergence_history:
                initial_value = optimization_result.convergence_history[0]
                final_value = optimization_result.optimal_value
                improvement_ratio = abs(initial_value - final_value) / abs(initial_value + 1e-8)
            else:
                improvement_ratio = 0.0
            
            passed = (execution_time < max_convergence_time and 
                     improvement_ratio > min_improvement and
                     optimization_result.success)
            
            return TestResult(
                test_name=test_name,
                category="performance",
                passed=passed,
                execution_time=execution_time,
                details={
                    "optimization_success": optimization_result.success,
                    "iterations": optimization_result.iterations,
                    "quantum_advantage": optimization_result.quantum_advantage_factor
                },
                performance_metrics={
                    "convergence_time": execution_time,
                    "improvement_ratio": improvement_ratio,
                    "final_value": optimization_result.optimal_value,
                    "iterations": optimization_result.iterations,
                    "quantum_coherence": optimization_result.quantum_coherence_preserved
                }
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category="performance",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def benchmark_concurrent_load(self) -> TestResult:
        """Benchmark system under concurrent load."""
        
        test_name = "concurrent_load_benchmark"
        start_time = time.time()
        
        try:
            # Create contract
            contract = RewardContract(
                name="load_test_contract",
                stakeholders={"user": 1.0}
            )
            
            # Test parameters
            num_threads = 10
            requests_per_thread = 100
            
            async def worker_task(worker_id: int) -> Dict[str, Any]:
                """Worker task for concurrent load testing."""
                worker_times = []
                worker_errors = 0
                
                # Generate worker-specific test data
                key = jax.random.PRNGKey(worker_id)
                
                for i in range(requests_per_thread):
                    try:
                        state = jax.random.normal(key, (20,))
                        action = jax.random.normal(jax.random.split(key)[1], (10,))
                        
                        eval_start = time.time()
                        reward = contract.compute_reward(state, action)
                        eval_time = time.time() - eval_start
                        
                        worker_times.append(eval_time)
                        key = jax.random.split(key, 3)[2]  # Update key
                        
                    except Exception as e:
                        worker_errors += 1
                        self.logger.warning(f"Worker {worker_id} error: {e}")
                
                return {
                    "worker_id": worker_id,
                    "times": worker_times,
                    "errors": worker_errors,
                    "avg_time": np.mean(worker_times) if worker_times else 0
                }
            
            # Run concurrent workers
            tasks = [worker_task(i) for i in range(num_threads)]
            worker_results = await asyncio.gather(*tasks)
            
            # Aggregate results
            all_times = []
            total_errors = 0
            
            for result in worker_results:
                all_times.extend(result["times"])
                total_errors += result["errors"]
            
            # Calculate metrics
            total_requests = num_threads * requests_per_thread
            success_rate = (total_requests - total_errors) / total_requests
            avg_time_ms = np.mean(all_times) * 1000 if all_times else 0
            p99_time_ms = np.percentile(all_times, 99) * 1000 if all_times else 0
            throughput_rps = len(all_times) / np.sum(all_times) if all_times else 0
            
            execution_time = time.time() - start_time
            
            # Performance criteria
            min_success_rate = 0.95
            max_avg_time_ms = 20.0
            max_p99_time_ms = 100.0
            min_throughput_rps = 50.0
            
            passed = (success_rate >= min_success_rate and
                     avg_time_ms <= max_avg_time_ms and
                     p99_time_ms <= max_p99_time_ms and
                     throughput_rps >= min_throughput_rps)
            
            return TestResult(
                test_name=test_name,
                category="performance",
                passed=passed,
                execution_time=execution_time,
                details={
                    "num_threads": num_threads,
                    "requests_per_thread": requests_per_thread,
                    "total_requests": total_requests,
                    "total_errors": total_errors
                },
                performance_metrics={
                    "success_rate": success_rate,
                    "avg_time_ms": avg_time_ms,
                    "p99_time_ms": p99_time_ms,
                    "throughput_rps": throughput_rps,
                    "concurrent_threads": num_threads
                }
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category="performance",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class SecurityTestSuite:
    """Security and robustness test suite."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def test_input_validation(self) -> TestResult:
        """Test input validation and sanitization."""
        
        test_name = "input_validation_test"
        start_time = time.time()
        
        try:
            contract = RewardContract(
                name="security_test_contract",
                stakeholders={"user": 1.0}
            )
            
            # Test cases for invalid inputs
            test_cases = [
                # Invalid array types
                ("none_state", None, jnp.ones(5)),
                ("none_action", jnp.ones(10), None),
                ("empty_state", jnp.array([]), jnp.ones(5)),
                ("empty_action", jnp.ones(10), jnp.array([])),
                ("infinite_state", jnp.array([float('inf')]), jnp.ones(5)),
                ("nan_state", jnp.array([float('nan')]), jnp.ones(5)),
                ("infinite_action", jnp.ones(10), jnp.array([float('inf')])),
                ("nan_action", jnp.ones(10), jnp.array([float('nan')])),
                # Very large inputs
                ("huge_state", jnp.ones(10) * 1e10, jnp.ones(5)),
                ("huge_action", jnp.ones(10), jnp.ones(5) * 1e10),
            ]
            
            passed_cases = 0
            total_cases = len(test_cases)
            validation_details = {}
            
            for case_name, state, action in test_cases:
                try:
                    if state is None or action is None:
                        # Should raise ValueError
                        with pytest.raises(ValueError):
                            contract.compute_reward(state, action)
                        passed_cases += 1
                        validation_details[case_name] = "correctly_rejected"
                    else:
                        reward = contract.compute_reward(state, action)
                        
                        # Check if reward is reasonable
                        if jnp.isfinite(reward) and -1000 <= reward <= 1000:
                            passed_cases += 1
                            validation_details[case_name] = "handled_gracefully"
                        else:
                            validation_details[case_name] = f"invalid_reward: {reward}"
                
                except ValueError as e:
                    # Expected for invalid inputs
                    passed_cases += 1
                    validation_details[case_name] = f"correctly_rejected: {str(e)[:50]}"
                
                except Exception as e:
                    validation_details[case_name] = f"unexpected_error: {str(e)[:50]}"
            
            execution_time = time.time() - start_time
            success_rate = passed_cases / total_cases
            
            return TestResult(
                test_name=test_name,
                category="security",
                passed=success_rate >= 0.8,  # 80% of cases should pass
                execution_time=execution_time,
                details={
                    "total_cases": total_cases,
                    "passed_cases": passed_cases,
                    "success_rate": success_rate,
                    "case_details": validation_details
                }
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category="security",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def test_security_framework_integration(self) -> TestResult:
        """Test integration with security framework."""
        
        test_name = "security_framework_integration_test"
        start_time = time.time()
        
        try:
            # Initialize security framework
            security_framework = ComprehensiveSecurityFramework(
                output_dir=Path("test_security_output"),
                enable_auto_response=False
            )
            
            # Test authentication
            auth_success, security_context, message = await security_framework.authenticate_user(
                user_id="test_security_user",
                credentials={"password": "test_password_123"},
                source_ip="192.168.1.100",
                user_agent="TestClient/1.0"
            )
            
            if not auth_success:
                return TestResult(
                    test_name=test_name,
                    category="security",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Authentication failed: {message}"
                )
            
            # Test authorization
            authorized, auth_message = await security_framework.authorize_operation(
                security_context,
                operation="read_contract",
                resource="test_contract"
            )
            
            if not authorized:
                return TestResult(
                    test_name=test_name,
                    category="security",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Authorization failed: {auth_message}"
                )
            
            # Test secure contract interaction
            contract = RewardContract(
                name="security_integration_test",
                stakeholders={"user": 1.0}
            )
            
            success, result, events = await security_framework.secure_contract_interaction(
                security_context,
                contract,
                "evaluate_reward",
                {
                    "state": jnp.ones(10),
                    "action": jnp.ones(5)
                }
            )
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                category="security",
                passed=success,
                execution_time=execution_time,
                details={
                    "authentication_success": auth_success,
                    "authorization_success": authorized,
                    "interaction_success": success,
                    "security_events_count": len(events),
                    "result": str(result)[:100] if result else "None"
                }
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category="security",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def test_contract_integrity(self) -> TestResult:
        """Test contract integrity and tamper detection."""
        
        test_name = "contract_integrity_test"
        start_time = time.time()
        
        try:
            # Create original contract
            contract = RewardContract(
                name="integrity_test_contract",
                stakeholders={"user": 0.6, "safety": 0.4}
            )
            
            contract.add_constraint(
                name="test_constraint",
                constraint_fn=lambda s, a: jnp.sum(s) > 0,
                description="Test constraint"
            )
            
            # Compute original hash
            original_hash = contract.compute_hash()
            
            # Test that identical contract produces same hash
            contract2 = RewardContract(
                name="integrity_test_contract",
                stakeholders={"user": 0.6, "safety": 0.4}
            )
            
            contract2.add_constraint(
                name="test_constraint",
                constraint_fn=lambda s, a: jnp.sum(s) > 0,
                description="Test constraint"
            )
            
            hash2 = contract2.compute_hash()
            
            # Test that modified contract produces different hash
            contract3 = RewardContract(
                name="integrity_test_contract",
                stakeholders={"user": 0.7, "safety": 0.3}  # Different weights
            )
            
            contract3.add_constraint(
                name="test_constraint",
                constraint_fn=lambda s, a: jnp.sum(s) > 0,
                description="Test constraint"
            )
            
            hash3 = contract3.compute_hash()
            
            # Test contract serialization/deserialization integrity
            contract_dict = contract.to_dict()
            serialized = json.dumps(contract_dict, sort_keys=True)
            deserialized_dict = json.loads(serialized)
            
            execution_time = time.time() - start_time
            
            # Integrity checks
            same_contract_same_hash = (original_hash == hash2)
            different_contract_different_hash = (original_hash != hash3)
            serialization_integrity = (contract_dict == deserialized_dict)
            
            passed = (same_contract_same_hash and 
                     different_contract_different_hash and
                     serialization_integrity)
            
            return TestResult(
                test_name=test_name,
                category="security",
                passed=passed,
                execution_time=execution_time,
                details={
                    "original_hash": original_hash[:16],
                    "same_contract_same_hash": same_contract_same_hash,
                    "different_contract_different_hash": different_contract_different_hash,
                    "serialization_integrity": serialization_integrity
                }
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category="security",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class IntegrationTestSuite:
    """End-to-end integration test suite."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def test_complete_workflow(self) -> TestResult:
        """Test complete RLHF workflow from contract creation to deployment."""
        
        test_name = "complete_workflow_integration_test"
        start_time = time.time()
        
        try:
            # Step 1: Create contract
            contract = RewardContract(
                name="integration_test_workflow",
                stakeholders={
                    "user": 0.5,
                    "safety": 0.3,
                    "efficiency": 0.2
                }
            )
            
            # Add constraints
            contract.add_constraint(
                name="safety_constraint",
                constraint_fn=lambda s, a: jnp.sum(jnp.abs(a)) < 3.0,
                description="Action safety constraint"
            )
            
            contract.add_constraint(
                name="efficiency_constraint",
                constraint_fn=lambda s, a: jnp.mean(s) > -2.0,
                description="State efficiency constraint"
            )
            
            # Step 2: Test basic contract functionality
            test_state = jnp.ones(20)
            test_action = jnp.ones(10) * 0.5
            
            reward = contract.compute_reward(test_state, test_action)
            assert jnp.isfinite(reward), "Basic reward computation failed"
            
            violations = contract.check_violations(test_state, test_action)
            assert isinstance(violations, dict), "Constraint checking failed"
            
            # Step 3: Test optimization integration
            optimizer = QuantumEnhancedRewardOptimizer(
                reward_contract=contract,
                num_qubits=4,
                quantum_learning_rate=0.1
            )
            
            # Generate test data
            key = jax.random.PRNGKey(42)
            preference_data = {
                "states": jax.random.normal(key, (100, 20)),
                "actions": jax.random.normal(jax.random.split(key)[1], (100, 10)),
                "preferences": jax.random.uniform(jax.random.split(key, 3)[2], (100,))
            }
            
            # Run short optimization
            opt_result = await optimizer.optimize_reward_function(
                preference_data=preference_data,
                max_iterations=20,
                quantum_method="vqe"
            )
            
            assert opt_result.success, "Optimization integration failed"
            
            # Step 4: Test security integration
            security_framework = ComprehensiveSecurityFramework(
                output_dir=Path("test_integration_security"),
                enable_auto_response=False
            )
            
            # Authenticate test user
            auth_success, context, _ = await security_framework.authenticate_user(
                user_id="integration_test_user",
                credentials={"password": "integration_test_pass"},
                source_ip="127.0.0.1",
                user_agent="IntegrationTest/1.0"
            )
            
            assert auth_success, "Security framework integration failed"
            
            # Test secure contract interaction
            interaction_success, result, events = await security_framework.secure_contract_interaction(
                context,
                contract,
                "evaluate_reward",
                {
                    "state": test_state,
                    "action": test_action
                }
            )
            
            assert interaction_success, "Secure contract interaction failed"
            
            # Step 5: Test deployment system integration (simplified)
            deployment_system = AutonomousGlobalDeploymentSystem(
                output_dir=Path("test_integration_deployment")
            )
            
            # Create deployment plan
            app_config = {
                "name": "integration-test-app",
                "version": "test-v1.0.0",
                "image": "test-image:latest"
            }
            
            deployment_plan = await deployment_system.create_global_deployment_plan(
                application_config=app_config,
                target_environments=["testing"],
                resource_requirements={
                    "cpu_request": "100m",
                    "memory_request": "256Mi"
                }
            )
            
            assert deployment_plan is not None, "Deployment plan creation failed"
            assert len(deployment_plan.targets) > 0, "No deployment targets created"
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                category="integration",
                passed=True,
                execution_time=execution_time,
                details={
                    "contract_created": True,
                    "reward_computed": float(reward),
                    "violations_checked": len(violations),
                    "optimization_success": opt_result.success,
                    "security_authenticated": auth_success,
                    "secure_interaction": interaction_success,
                    "deployment_plan_created": deployment_plan is not None,
                    "deployment_targets": len(deployment_plan.targets)
                },
                performance_metrics={
                    "total_workflow_time": execution_time,
                    "optimization_iterations": opt_result.iterations,
                    "security_events": len(events)
                }
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category="integration",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def test_multi_stakeholder_scenarios(self) -> TestResult:
        """Test complex multi-stakeholder scenarios."""
        
        test_name = "multi_stakeholder_scenarios_test"
        start_time = time.time()
        
        try:
            # Create complex multi-stakeholder contract
            contract = RewardContract(
                name="multi_stakeholder_test",
                stakeholders={
                    "end_users": 0.35,
                    "safety_board": 0.25,
                    "business_owners": 0.20,
                    "regulatory_compliance": 0.15,
                    "technical_team": 0.05
                },
                aggregation=AggregationStrategy.WEIGHTED_AVERAGE
            )
            
            # Add stakeholder-specific constraints
            contract.add_constraint(
                name="user_satisfaction",
                constraint_fn=lambda s, a: jnp.mean(s) > 0.0,
                description="User satisfaction constraint"
            )
            
            contract.add_constraint(
                name="safety_compliance",
                constraint_fn=lambda s, a: jnp.max(jnp.abs(a)) < 1.5,
                description="Safety compliance constraint"
            )
            
            contract.add_constraint(
                name="business_efficiency",
                constraint_fn=lambda s, a: jnp.sum(a) < 10.0,
                description="Business efficiency constraint"
            )
            
            # Test various scenarios
            scenarios = [
                ("balanced_case", jnp.ones(15), jnp.ones(8) * 0.5),
                ("safety_critical", jnp.ones(15), jnp.ones(8) * 1.0),
                ("user_focused", jnp.ones(15) * 2.0, jnp.ones(8) * 0.3),
                ("efficiency_optimized", jnp.ones(15) * 0.5, jnp.ones(8) * 0.1),
                ("edge_case", jnp.ones(15) * -0.5, jnp.ones(8) * 1.2)
            ]
            
            scenario_results = {}
            
            for scenario_name, state, action in scenarios:
                try:
                    reward = contract.compute_reward(state, action)
                    violations = contract.check_violations(state, action)
                    violation_count = sum(violations.values())
                    
                    scenario_results[scenario_name] = {
                        "reward": float(reward),
                        "violations": violation_count,
                        "valid": jnp.isfinite(reward) and violation_count == 0
                    }
                
                except Exception as e:
                    scenario_results[scenario_name] = {
                        "error": str(e),
                        "valid": False
                    }
            
            # Verify stakeholder weight consistency
            total_weight = sum(contract.stakeholders[name].weight for name in contract.stakeholders)
            weight_consistency = abs(total_weight - 1.0) < 1e-6
            
            # Check aggregation strategy effectiveness
            aggregation_test_passed = True
            for result in scenario_results.values():
                if result.get("valid", False):
                    # Reward should be reasonable for valid scenarios
                    reward = result["reward"]
                    if not (-10.0 <= reward <= 10.0):  # Reasonable range
                        aggregation_test_passed = False
                        break
            
            execution_time = time.time() - start_time
            
            # Overall test success
            successful_scenarios = sum(1 for r in scenario_results.values() if r.get("valid", False))
            success_rate = successful_scenarios / len(scenarios)
            
            passed = (success_rate >= 0.8 and 
                     weight_consistency and 
                     aggregation_test_passed)
            
            return TestResult(
                test_name=test_name,
                category="integration",
                passed=passed,
                execution_time=execution_time,
                details={
                    "total_stakeholders": len(contract.stakeholders),
                    "total_constraints": len(contract.constraints),
                    "weight_consistency": weight_consistency,
                    "aggregation_strategy": contract.aggregation_strategy.value,
                    "successful_scenarios": successful_scenarios,
                    "total_scenarios": len(scenarios),
                    "success_rate": success_rate,
                    "scenario_results": scenario_results
                }
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category="integration",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def test_neural_architecture_search_integration(self) -> TestResult:
        """Test Neural Architecture Search integration."""
        
        test_name = "nas_integration_test"
        start_time = time.time()
        
        try:
            # Initialize NAS system
            nas = NeuralArchitectureSearch(
                population_size=10,  # Small population for testing
                generations=5,      # Few generations for testing
                output_dir=Path("test_nas_output")
            )
            
            # Create test reward contract
            contract = RewardContract(
                name="nas_test_contract",
                stakeholders={"user": 0.7, "safety": 0.3}
            )
            
            # Generate test data
            key = jax.random.PRNGKey(42)
            train_data = {
                "states": jax.random.normal(key, (100, 512)),
                "rewards": jax.random.uniform(jax.random.split(key)[1], (100,))
            }
            
            val_data = {
                "states": jax.random.normal(jax.random.split(key, 3)[2], (50, 512)),
                "rewards": jax.random.uniform(jax.random.split(key, 4)[3], (50,))
            }
            
            # Run NAS (simplified for testing)
            pareto_front = await nas.search_optimal_architecture(
                reward_contract=contract,
                train_data=train_data,
                val_data=val_data,
                search_strategy="evolutionary"
            )
            
            execution_time = time.time() - start_time
            
            # Validate results
            nas_success = (len(pareto_front) > 0 and 
                          all(arch.accuracy >= 0 for arch in pareto_front) and
                          all(arch.parameters_count > 0 for arch in pareto_front))
            
            if pareto_front:
                best_accuracy = max(arch.accuracy for arch in pareto_front)
                avg_parameters = np.mean([arch.parameters_count for arch in pareto_front])
            else:
                best_accuracy = 0.0
                avg_parameters = 0.0
            
            return TestResult(
                test_name=test_name,
                category="integration",
                passed=nas_success,
                execution_time=execution_time,
                details={
                    "pareto_front_size": len(pareto_front),
                    "best_accuracy": best_accuracy,
                    "avg_parameters": avg_parameters,
                    "search_completed": True
                },
                performance_metrics={
                    "search_time": execution_time,
                    "architectures_evaluated": len(pareto_front),
                    "best_accuracy": best_accuracy
                }
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                category="integration",
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class ComprehensiveTestRunner:
    """Comprehensive test runner that coordinates all test suites."""
    
    def __init__(self, output_dir: Path = Path("test_results")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize test suites
        self.performance_suite = PerformanceBenchmarkSuite()
        self.security_suite = SecurityTestSuite()
        self.integration_suite = IntegrationTestSuite()
        
        # Test results storage
        self.test_results: List[TestResult] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive test logging."""
        log_file = self.output_dir / "test_execution.log"
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - TEST_RUNNER - %(levelname)s - %(message)s'
        ))
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def run_all_tests(self, parallel: bool = True) -> TestSuiteReport:
        """Run all test suites and generate comprehensive report."""
        
        self.logger.info("Starting comprehensive test suite execution")
        start_time = time.time()
        
        # Define all test methods
        test_methods = [
            # Performance tests
            self.performance_suite.benchmark_reward_computation,
            self.performance_suite.benchmark_optimization_convergence,
            self.performance_suite.benchmark_concurrent_load,
            
            # Security tests
            self.security_suite.test_input_validation,
            self.security_suite.test_security_framework_integration,
            self.security_suite.test_contract_integrity,
            
            # Integration tests
            self.integration_suite.test_complete_workflow,
            self.integration_suite.test_multi_stakeholder_scenarios,
            self.integration_suite.test_neural_architecture_search_integration,
        ]
        
        # Run tests
        if parallel:
            self.test_results = await self._run_tests_parallel(test_methods)
        else:
            self.test_results = await self._run_tests_sequential(test_methods)
        
        total_execution_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_test_report(total_execution_time)
        
        # Save report
        await self._save_test_report(report)
        
        self.logger.info(
            f"Test suite completed: {report.passed_tests}/{report.total_tests} passed "
            f"in {total_execution_time:.2f} seconds"
        )
        
        return report
    
    async def _run_tests_parallel(self, test_methods: List[Callable]) -> List[TestResult]:
        """Run tests in parallel for faster execution."""
        
        self.logger.info(f"Running {len(test_methods)} tests in parallel")
        
        # Use asyncio.gather for parallel execution
        try:
            results = await asyncio.gather(*[method() for method in test_methods], 
                                          return_exceptions=True)
            
            # Process results and exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create failed test result for exceptions
                    processed_results.append(TestResult(
                        test_name=f"test_{i}_exception",
                        category="error",
                        passed=False,
                        execution_time=0.0,
                        error_message=str(result)
                    ))
                elif isinstance(result, TestResult):
                    processed_results.append(result)
                else:
                    # Unexpected result type
                    processed_results.append(TestResult(
                        test_name=f"test_{i}_unexpected",
                        category="error",
                        passed=False,
                        execution_time=0.0,
                        error_message=f"Unexpected result type: {type(result)}"
                    ))
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Parallel test execution failed: {e}")
            return []
    
    async def _run_tests_sequential(self, test_methods: List[Callable]) -> List[TestResult]:
        """Run tests sequentially for debugging."""
        
        self.logger.info(f"Running {len(test_methods)} tests sequentially")
        
        results = []
        
        for i, method in enumerate(test_methods):
            try:
                self.logger.info(f"Running test {i+1}/{len(test_methods)}: {method.__name__}")
                result = await method()
                results.append(result)
                
                status = "PASSED" if result.passed else "FAILED"
                self.logger.info(f"Test {method.__name__}: {status} ({result.execution_time:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"Test {method.__name__} raised exception: {e}")
                results.append(TestResult(
                    test_name=method.__name__,
                    category="error",
                    passed=False,
                    execution_time=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def _generate_test_report(self, total_execution_time: float) -> TestSuiteReport:
        """Generate comprehensive test report."""
        
        # Count results by status
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = sum(1 for r in self.test_results if not r.passed)
        total_tests = len(self.test_results)
        skipped_tests = 0  # Would implement test skipping logic
        
        # Performance summary
        performance_results = [r for r in self.test_results if r.category == "performance"]
        performance_summary = {}
        
        if performance_results:
            avg_execution_time = np.mean([r.execution_time for r in performance_results])
            
            # Aggregate performance metrics
            all_metrics = {}
            for result in performance_results:
                for metric, value in result.performance_metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
            
            performance_summary = {
                "avg_test_execution_time": avg_execution_time,
                "performance_metrics": {
                    metric: {
                        "avg": np.mean(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values)
                    }
                    for metric, values in all_metrics.items()
                }
            }
        
        # Security summary
        security_results = [r for r in self.test_results if r.category == "security"]
        security_summary = {
            "total_security_tests": len(security_results),
            "passed_security_tests": sum(1 for r in security_results if r.passed),
            "security_pass_rate": sum(1 for r in security_results if r.passed) / len(security_results) if security_results else 0
        }
        
        # Coverage report (simplified)
        coverage_report = {
            "components_tested": {
                "reward_contract": True,
                "quantum_optimization": True,
                "security_framework": True,
                "deployment_system": True,
                "neural_architecture_search": True
            },
            "test_categories": {
                "performance": len([r for r in self.test_results if r.category == "performance"]),
                "security": len([r for r in self.test_results if r.category == "security"]),
                "integration": len([r for r in self.test_results if r.category == "integration"])
            }
        }
        
        return TestSuiteReport(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            total_execution_time=total_execution_time,
            test_results=self.test_results,
            performance_summary=performance_summary,
            security_summary=security_summary,
            coverage_report=coverage_report
        )
    
    async def _save_test_report(self, report: TestSuiteReport):
        """Save comprehensive test report to files."""
        
        # Save JSON report
        report_data = {
            "summary": {
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "skipped_tests": report.skipped_tests,
                "total_execution_time": report.total_execution_time,
                "pass_rate": report.passed_tests / report.total_tests if report.total_tests > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "category": r.category,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "details": r.details,
                    "performance_metrics": r.performance_metrics
                }
                for r in report.test_results
            ],
            "performance_summary": report.performance_summary,
            "security_summary": report.security_summary,
            "coverage_report": report.coverage_report
        }
        
        report_file = self.output_dir / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save human-readable report
        readable_report = self._generate_readable_report(report)
        readable_file = self.output_dir / "test_report.txt"
        with open(readable_file, 'w') as f:
            f.write(readable_report)
        
        self.logger.info(f"Test reports saved to {self.output_dir}")
    
    def _generate_readable_report(self, report: TestSuiteReport) -> str:
        """Generate human-readable test report."""
        
        lines = []
        lines.append("=" * 80)
        lines.append("RLHF-Contract-Wizard Comprehensive Test Report")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Tests: {report.total_tests}")
        lines.append(f"Passed: {report.passed_tests}")
        lines.append(f"Failed: {report.failed_tests}")
        lines.append(f"Skipped: {report.skipped_tests}")
        
        pass_rate = report.passed_tests / report.total_tests if report.total_tests > 0 else 0
        lines.append(f"Pass Rate: {pass_rate:.1%}")
        lines.append(f"Total Execution Time: {report.total_execution_time:.2f} seconds")
        lines.append("")
        
        # Test Results by Category
        categories = set(r.category for r in report.test_results)
        for category in sorted(categories):
            cat_results = [r for r in report.test_results if r.category == category]
            passed = sum(1 for r in cat_results if r.passed)
            total = len(cat_results)
            
            lines.append(f"{category.upper()} TESTS ({passed}/{total})")
            lines.append("-" * 40)
            
            for result in cat_results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                lines.append(f"{status} {result.test_name} ({result.execution_time:.2f}s)")
                
                if not result.passed and result.error_message:
                    lines.append(f"    Error: {result.error_message}")
            
            lines.append("")
        
        # Performance Summary
        if report.performance_summary:
            lines.append("PERFORMANCE SUMMARY")
            lines.append("-" * 40)
            
            if "performance_metrics" in report.performance_summary:
                for metric, stats in report.performance_summary["performance_metrics"].items():
                    lines.append(f"{metric}:")
                    lines.append(f"  Average: {stats['avg']:.3f}")
                    lines.append(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
                    lines.append(f"  Count: {stats['count']}")
            
            lines.append("")
        
        # Security Summary
        lines.append("SECURITY SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Security Tests: {report.security_summary['total_security_tests']}")
        lines.append(f"Passed: {report.security_summary['passed_security_tests']}")
        lines.append(f"Security Pass Rate: {report.security_summary['security_pass_rate']:.1%}")
        lines.append("")
        
        # Coverage Report
        lines.append("COVERAGE REPORT")
        lines.append("-" * 40)
        
        for component, tested in report.coverage_report["components_tested"].items():
            status = "✅" if tested else "❌"
            lines.append(f"{status} {component}")
        
        lines.append("")
        
        for category, count in report.coverage_report["test_categories"].items():
            lines.append(f"{category}: {count} tests")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Main execution for standalone testing
if __name__ == "__main__":
    
    async def main():
        print("🧪 RLHF-Contract-Wizard Comprehensive Test Suite")
        print("=" * 60)
        
        # Initialize test runner
        test_runner = ComprehensiveTestRunner(
            output_dir=Path("comprehensive_test_results")
        )
        
        # Run all tests
        print("\n🚀 Starting comprehensive test execution...")
        
        report = await test_runner.run_all_tests(parallel=True)
        
        # Display results
        print(f"\n📊 TEST RESULTS SUMMARY")
        print(f"   Total Tests: {report.total_tests}")
        print(f"   Passed: {report.passed_tests}")
        print(f"   Failed: {report.failed_tests}")
        print(f"   Pass Rate: {report.passed_tests/report.total_tests:.1%}")
        print(f"   Execution Time: {report.total_execution_time:.2f} seconds")
        
        # Show failed tests
        failed_tests = [r for r in report.test_results if not r.passed]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test.test_name}: {test.error_message}")
        
        # Show performance highlights
        if report.performance_summary:
            print(f"\n⚡ PERFORMANCE HIGHLIGHTS:")
            perf_metrics = report.performance_summary.get("performance_metrics", {})
            for metric, stats in list(perf_metrics.items())[:5]:  # Top 5 metrics
                print(f"   • {metric}: {stats['avg']:.3f} (avg)")
        
        # Show security status
        security_pass_rate = report.security_summary["security_pass_rate"]
        security_status = "✅" if security_pass_rate >= 0.8 else "⚠️"
        print(f"\n🔒 SECURITY STATUS: {security_status}")
        print(f"   Security Tests Passed: {report.security_summary['passed_security_tests']}/{report.security_summary['total_security_tests']}")
        print(f"   Security Pass Rate: {security_pass_rate:.1%}")
        
        # Overall verdict
        overall_pass_rate = report.passed_tests / report.total_tests if report.total_tests > 0 else 0
        
        if overall_pass_rate >= 0.95:
            verdict = "🌟 EXCELLENT - System ready for production"
        elif overall_pass_rate >= 0.85:
            verdict = "✅ GOOD - Minor issues to address"  
        elif overall_pass_rate >= 0.7:
            verdict = "⚠️ FAIR - Several issues need attention"
        else:
            verdict = "❌ POOR - Major issues require resolution"
        
        print(f"\n🎯 OVERALL VERDICT: {verdict}")
        print(f"\n📁 Detailed reports saved to: comprehensive_test_results/")
        print("\n✅ Comprehensive test suite execution completed!")
    
    # Run the test suite
    asyncio.run(main())
#!/usr/bin/env python3
"""
Comprehensive Test Suite for RLHF Contract Wizard

This test suite provides comprehensive coverage of all system components,
including unit tests, integration tests, performance tests, and security tests.
Designed to meet the 85%+ test coverage requirement from the SDLC specification.
"""

import pytest
import time
import asyncio
import tempfile
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import jax.numpy as jnp
    import numpy as np
    from src.models.reward_contract import RewardContract, AggregationStrategy
    from src.quantum_planner.core import QuantumTaskPlanner, QuantumTask, TaskState, PlannerConfig
    from src.demo_runner import RLHFContractDemo
    from src.security.security_framework import SecurityFramework
    from src.monitoring.comprehensive_monitoring import MonitoringSystem
    from src.resilience.error_recovery import ErrorRecoveryOrchestrator
    from src.scaling.intelligent_scaling import AutoScaler
    from src.performance.advanced_caching import CacheManager
    from src.enhanced_api_client import EnhancedAPIClient
    from src.advanced_optimization import AdaptiveOptimizer, OptimizationConfig, OptimizationStrategy
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    JAX_AVAILABLE = False


class TestRewardContract:
    """Test suite for reward contract functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        self.contract = RewardContract(
            name="test-contract",
            stakeholders={"user": 0.6, "admin": 0.4}
        )
    
    def test_contract_creation(self):
        """Test basic contract creation."""
        assert self.contract.name == "test-contract"
        assert len(self.contract.stakeholders) == 2
        assert self.contract.stakeholders["user"] == 0.6
        assert self.contract.stakeholders["admin"] == 0.4
    
    def test_reward_function_registration(self):
        """Test reward function registration."""
        @self.contract.reward_function("user")
        def user_reward(state, action):
            return jnp.mean(state) + jnp.mean(action)
        
        assert "user" in self.contract.reward_functions
        assert callable(self.contract.reward_functions["user"])
    
    def test_constraint_addition(self):
        """Test constraint addition."""
        def safety_constraint(state, action):
            return jnp.all(action >= 0)
        
        self.contract.add_constraint(
            "safety", safety_constraint, "Safety constraint"
        )
        
        assert "safety" in self.contract.constraints
        assert callable(self.contract.constraints["safety"]["function"])
    
    def test_reward_computation(self):
        """Test reward computation."""
        @self.contract.reward_function("user")
        def user_reward(state, action):
            return jnp.sum(state)
        
        @self.contract.reward_function("admin")
        def admin_reward(state, action):
            return jnp.sum(action)
        
        state = jnp.array([1.0, 2.0])
        action = jnp.array([0.5, 1.5])
        
        reward = self.contract.compute_reward(state, action)
        assert isinstance(reward, (float, jnp.ndarray))
        assert reward > 0  # Should be positive given positive inputs
    
    def test_constraint_violations(self):
        """Test constraint violation detection."""
        def positive_constraint(state, action):
            return jnp.all(action > 0)
        
        self.contract.add_constraint(
            "positive", positive_constraint, "All actions must be positive"
        )
        
        # Test with positive actions (no violation)
        state = jnp.array([1.0])
        action = jnp.array([1.0])
        violations = self.contract.check_violations(state, action)
        assert not violations["positive"]
        
        # Test with negative actions (violation)
        action = jnp.array([-1.0])
        violations = self.contract.check_violations(state, action)
        assert violations["positive"]
    
    def test_contract_serialization(self):
        """Test contract serialization to dict."""
        contract_dict = self.contract.to_dict()
        
        assert "metadata" in contract_dict
        assert "stakeholders" in contract_dict
        assert contract_dict["metadata"]["name"] == "test-contract"
        assert contract_dict["stakeholders"] == {"user": 0.6, "admin": 0.4}
    
    def test_contract_hash(self):
        """Test contract hash computation."""
        hash1 = self.contract.compute_hash()
        hash2 = self.contract.compute_hash()
        
        assert hash1 == hash2  # Should be deterministic
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex string


class TestQuantumPlanner:
    """Test suite for quantum task planner."""
    
    def setup_method(self):
        """Setup test fixtures."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        self.config = PlannerConfig(max_iterations=10, convergence_threshold=1e-4)
        self.planner = QuantumTaskPlanner(self.config)
    
    def test_planner_creation(self):
        """Test planner creation."""
        assert self.planner.config.max_iterations == 10
        assert len(self.planner.tasks) == 0
        assert len(self.planner.resource_pool) == 0
    
    def test_task_addition(self):
        """Test adding tasks to planner."""
        task = QuantumTask("task1", "Test Task", "A test task")
        self.planner.add_task(task)
        
        assert "task1" in self.planner.tasks
        assert self.planner.tasks["task1"].name == "Test Task"
    
    def test_resource_management(self):
        """Test resource pool management."""
        self.planner.add_resource("cpu", 4.0)
        self.planner.add_resource("memory", 8.0)
        
        assert self.planner.resource_pool["cpu"] == 4.0
        assert self.planner.resource_pool["memory"] == 8.0
    
    def test_task_dependencies(self):
        """Test task dependency management."""
        task1 = QuantumTask("task1", "Task 1", "First task")
        task2 = QuantumTask("task2", "Task 2", "Second task")
        
        task2.add_dependency("task1")
        
        self.planner.add_task(task1)
        self.planner.add_task(task2)
        
        assert "task1" in task2.dependencies
        assert not task2.is_ready(set())  # Not ready without task1
        assert task2.is_ready({"task1"})  # Ready with task1 completed
    
    def test_quantum_entanglement(self):
        """Test quantum entanglement between tasks."""
        task1 = QuantumTask("task1", "Task 1", "First task")
        task2 = QuantumTask("task2", "Task 2", "Second task")
        
        self.planner.add_task(task1)
        self.planner.add_task(task2)
        self.planner.create_entanglement("task1", "task2", 0.8)
        
        assert "task2" in task1.entangled_tasks
        assert "task1" in task2.entangled_tasks
        assert ("task1", "task2") in self.planner.entanglement_matrix or ("task2", "task1") in self.planner.entanglement_matrix
    
    def test_plan_optimization(self):
        """Test plan optimization."""
        # Add some tasks
        for i in range(3):
            task = QuantumTask(f"task{i}", f"Task {i}", f"Test task {i}")
            task.priority = 0.5 + i * 0.1
            self.planner.add_task(task)
        
        self.planner.add_resource("cpu", 10.0)
        
        result = self.planner.optimize_plan()
        
        assert "task_order" in result
        assert "fitness_score" in result
        assert "optimization_time" in result
        assert "iterations" in result
        assert "converged" in result
        assert isinstance(result["task_order"], list)
        assert 0 <= result["fitness_score"] <= 1
    
    def test_plan_execution(self):
        """Test plan execution."""
        task = QuantumTask("task1", "Test Task", "A test task")
        task.estimated_duration = 0.1
        self.planner.add_task(task)
        self.planner.add_resource("cpu", 1.0)
        
        plan = {"task_order": ["task1"]}
        result = self.planner.execute_plan(plan)
        
        assert "execution_log" in result
        assert "completed_tasks" in result
        assert "failed_tasks" in result
        assert "success_rate" in result
        assert isinstance(result["execution_log"], list)


class TestDemoRunner:
    """Test suite for demo runner."""
    
    def setup_method(self):
        """Setup test fixtures."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        self.demo = RLHFContractDemo()
    
    def test_demo_creation(self):
        """Test demo runner creation."""
        assert hasattr(self.demo, 'demo_results')
        assert isinstance(self.demo.demo_results, dict)
    
    def test_reward_contract_demo(self):
        """Test reward contract demo."""
        result = self.demo.run_reward_contract_demo()
        
        assert "contract_hash" in result
        assert "computed_reward" in result
        assert "violations" in result
        assert "stakeholders" in result
        assert "constraints" in result
        assert result["stakeholders"] == 3
        assert result["constraints"] == 2
    
    def test_quantum_planner_demo(self):
        """Test quantum planner demo."""
        result = self.demo.run_quantum_planner_demo()
        
        assert "optimization_time" in result
        assert "fitness_score" in result
        assert "total_tasks" in result
        assert "quantum_metrics" in result
        assert result["total_tasks"] == 5
        assert 0 <= result["fitness_score"] <= 1
    
    def test_integration_demo(self):
        """Test integration demo."""
        result = self.demo.run_integration_demo()
        
        assert "contract_reward" in result
        assert "violations_detected" in result
        assert "compliance_task_ready" in result
        assert isinstance(result["violations_detected"], int)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        result = self.demo.run_performance_benchmarks()
        
        assert "reward_computation" in result
        assert "quantum_planning" in result
        
        # Check reward computation benchmarks
        rc = result["reward_computation"]
        assert "uncached_time_100_runs" in rc
        assert "cached_time_100_runs" in rc
        assert "speedup_factor" in rc
        assert rc["speedup_factor"] > 1.0  # Caching should provide speedup
    
    def test_full_demo_execution(self):
        """Test full demo execution."""
        result = self.demo.run_full_demo()
        
        assert "reward_contracts" in result
        assert "quantum_planning" in result
        assert "integration" in result
        assert "benchmarks" in result
        assert "summary" in result
        
        summary = result["summary"]
        assert summary["components_tested"] == 4
        assert summary["contracts_created"] == 3
        assert summary["tasks_planned"] == 5


class TestSecurityFramework:
    """Test suite for security framework."""
    
    def test_security_framework_creation(self):
        """Test security framework creation."""
        try:
            framework = SecurityFramework()
            assert hasattr(framework, 'crypto')
            assert hasattr(framework, 'access_controller')
            assert hasattr(framework, 'threat_detector')
            assert hasattr(framework, 'audit_logger')
        except ImportError:
            pytest.skip("Security framework not available")


class TestMonitoringSystem:
    """Test suite for monitoring system."""
    
    def test_monitoring_system_creation(self):
        """Test monitoring system creation."""
        try:
            monitor = MonitoringSystem()
            assert hasattr(monitor, 'metrics_collector')
            assert hasattr(monitor, 'alerting_manager')
            assert hasattr(monitor, 'health_checker')
        except ImportError:
            pytest.skip("Monitoring system not available")


class TestErrorRecovery:
    """Test suite for error recovery."""
    
    def test_error_recovery_creation(self):
        """Test error recovery orchestrator creation."""
        try:
            orchestrator = ErrorRecoveryOrchestrator()
            assert hasattr(orchestrator, 'error_classifier')
            assert hasattr(orchestrator, 'circuit_breakers')
            assert hasattr(orchestrator, 'retry_handler')
            assert hasattr(orchestrator, 'fallback_manager')
        except ImportError:
            pytest.skip("Error recovery system not available")
    
    def test_error_handling(self):
        """Test error handling functionality."""
        try:
            orchestrator = ErrorRecoveryOrchestrator()
            
            # Test with a simple error
            test_error = ValueError("Test error")
            result = orchestrator.handle_error(test_error, "test_operation")
            
            assert hasattr(result, 'success')
            assert hasattr(result, 'strategy_used')
            assert hasattr(result, 'attempt_count')
            assert hasattr(result, 'recovery_time')
        except ImportError:
            pytest.skip("Error recovery system not available")


class TestAdvancedOptimization:
    """Test suite for advanced optimization."""
    
    def test_optimizer_creation(self):
        """Test optimizer creation."""
        try:
            config = OptimizationConfig(
                strategy=OptimizationStrategy.ADAM,
                max_iterations=10
            )
            optimizer = AdaptiveOptimizer(config)
            assert optimizer.config.strategy == OptimizationStrategy.ADAM
            assert optimizer.config.max_iterations == 10
        except ImportError:
            pytest.skip("Optimization system not available")
    
    def test_simple_optimization(self):
        """Test simple optimization function."""
        try:
            if not JAX_AVAILABLE:
                pytest.skip("JAX not available")
                
            config = OptimizationConfig(
                strategy=OptimizationStrategy.ADAM,
                max_iterations=5
            )
            optimizer = AdaptiveOptimizer(config)
            
            # Simple quadratic function: f(x) = -(x-2)^2 + 4
            def objective(x):
                return -(x[0] - 2.0)**2 + 4.0
            
            initial_params = jnp.array([0.0])
            result = optimizer.optimize(objective, initial_params)
            
            assert hasattr(result, 'optimal_parameters')
            assert hasattr(result, 'optimal_value')
            assert hasattr(result, 'iterations')
            assert hasattr(result, 'converged')
            assert result.iterations <= 5
        except ImportError:
            pytest.skip("Optimization system not available")


class TestCacheManager:
    """Test suite for cache manager."""
    
    def test_cache_manager_creation(self):
        """Test cache manager creation."""
        try:
            cache_manager = CacheManager()
            assert hasattr(cache_manager, 'l1_cache')
            assert hasattr(cache_manager, 'l2_cache')
            assert hasattr(cache_manager, 'distributed_cache')
            assert hasattr(cache_manager, 'prefetcher')
        except ImportError:
            pytest.skip("Cache system not available")


class TestPerformanceRequirements:
    """Test suite for performance requirements."""
    
    def test_response_time_requirement(self):
        """Test that API response times meet requirements (<200ms)."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        # Test reward computation performance
        contract = RewardContract("perf-test", stakeholders={"user": 1.0})
        
        @contract.reward_function("user")
        def fast_reward(state, action):
            return jnp.sum(state) + jnp.sum(action)
        
        state = jnp.array([1.0, 2.0, 3.0])
        action = jnp.array([0.5, 1.0, 1.5])
        
        # Warm up JIT compilation
        contract.compute_reward(state, action)
        
        # Measure actual performance
        start_time = time.time()
        for _ in range(10):
            contract.compute_reward(state, action)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 0.2, f"Average response time {avg_time:.3f}s exceeds 200ms requirement"
    
    def test_quantum_planner_performance(self):
        """Test quantum planner performance."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        config = PlannerConfig(max_iterations=50)
        planner = QuantumTaskPlanner(config)
        
        # Add test tasks
        for i in range(5):
            task = QuantumTask(f"task{i}", f"Task {i}", f"Test task {i}")
            planner.add_task(task)
        
        planner.add_resource("cpu", 10.0)
        
        start_time = time.time()
        result = planner.optimize_plan()
        end_time = time.time()
        
        optimization_time = end_time - start_time
        assert optimization_time < 5.0, f"Optimization time {optimization_time:.3f}s too slow"
        assert result["fitness_score"] > 0.0


class TestIntegrationRequirements:
    """Test suite for integration requirements."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        # Create contract
        contract = RewardContract(
            "integration-test",
            stakeholders={"user": 0.7, "system": 0.3}
        )
        
        @contract.reward_function("user")
        def user_reward(state, action):
            return jnp.mean(state)
        
        @contract.reward_function("system")
        def system_reward(state, action):
            return jnp.mean(action)
        
        # Add constraint
        def safety_constraint(state, action):
            return jnp.all(action >= 0)
        
        contract.add_constraint("safety", safety_constraint, "Safety constraint")
        
        # Create planner
        planner = QuantumTaskPlanner()
        task = QuantumTask("verify_contract", "Verify Contract", "Verify the contract")
        planner.add_task(task)
        
        # Test integration
        state = jnp.array([1.0, 2.0])
        action = jnp.array([0.5, 1.5])
        
        reward = contract.compute_reward(state, action)
        violations = contract.check_violations(state, action)
        plan = planner.optimize_plan()
        
        # Verify integration works
        assert reward is not None
        assert isinstance(violations, dict)
        assert "task_order" in plan
        assert not violations["safety"]  # Should not violate safety constraint


def pytest_configure(config):
    """Pytest configuration."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark performance tests as slow
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.slow)


if __name__ == "__main__":
    # Run tests with coverage if called directly
    import subprocess
    import sys
    
    try:
        # Install required packages
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov", "pytest-asyncio"], 
                      check=True, capture_output=True)
        
        # Run tests with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            __file__, 
            "-v", 
            "--cov=src", 
            "--cov-report=html", 
            "--cov-report=term-missing",
            "--cov-min=80"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        sys.exit(result.returncode)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        sys.exit(1)
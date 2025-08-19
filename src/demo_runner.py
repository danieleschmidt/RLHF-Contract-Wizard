#!/usr/bin/env python3
"""
RLHF Contract Wizard Demo Runner

Demonstrates the key features of the system with comprehensive examples
showing the quantum planner, reward contracts, and API functionality.
"""

import asyncio
import time
import json
from typing import Dict, List, Any
import jax.numpy as jnp
import numpy as np

from src.models.reward_contract import RewardContract, AggregationStrategy, Stakeholder
from src.quantum_planner.core import QuantumTaskPlanner, QuantumTask, TaskState, PlannerConfig


class RLHFContractDemo:
    """Comprehensive demonstration of RLHF Contract Wizard capabilities."""
    
    def __init__(self):
        self.demo_results: Dict[str, Any] = {}
    
    def run_reward_contract_demo(self) -> Dict[str, Any]:
        """Demonstrate reward contract functionality."""
        print("🔗 Running Reward Contract Demo...")
        
        # Create multi-stakeholder contract
        contract = RewardContract(
            name="SafeAssistant-v1",
            version="1.0.0",
            stakeholders={
                "safety_board": 0.4,
                "operators": 0.35, 
                "users": 0.25
            },
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE
        )
        
        # Add safety constraints
        def safety_constraint(state, action):
            # Simulate safety check
            return not self._contains_harmful_content(action)
        
        contract.add_constraint(
            "no_harmful_content",
            safety_constraint,
            description="Prevent generation of harmful content",
            severity=1.0,
            violation_penalty=-10.0
        )
        
        def privacy_constraint(state, action):
            # Simulate privacy check
            return not self._contains_pii(action)
        
        contract.add_constraint(
            "privacy_protection",
            privacy_constraint,
            description="Protect personally identifiable information",
            severity=0.8,
            violation_penalty=-5.0
        )
        
        # Register stakeholder reward functions
        @contract.reward_function("safety_board")
        def safety_reward(state, action):
            # Safety-focused reward prioritizing harmlessness
            safety_score = jnp.where(
                self._is_safe(action),
                1.0,
                0.0
            )
            return safety_score * 0.9 + 0.05  # Fixed noise instead of random
        
        @contract.reward_function("operators")
        def operator_reward(state, action):
            # Efficiency and performance focused
            efficiency = self._compute_efficiency(action)
            user_satisfaction = self._compute_satisfaction(action)
            return 0.6 * efficiency + 0.4 * user_satisfaction
        
        @contract.reward_function("users")
        def user_reward(state, action):
            # User experience focused
            helpfulness = self._compute_helpfulness(action)
            responsiveness = self._compute_responsiveness(action)
            return 0.7 * helpfulness + 0.3 * responsiveness
        
        # Test contract with sample data
        sample_state = jnp.array([0.5, 0.3, 0.8, 0.2])
        sample_action = jnp.array([0.7, 0.4, 0.6])
        
        # Compute rewards
        reward = contract.compute_reward(sample_state, sample_action)
        violations = contract.check_violations(sample_state, sample_action)
        
        # Export contract
        contract_dict = contract.to_dict()
        
        return {
            "contract_hash": contract.compute_hash(),
            "computed_reward": float(reward),
            "violations": violations,
            "stakeholders": len(contract.stakeholders),
            "constraints": len(contract.constraints),
            "contract_metadata": contract_dict["metadata"]
        }
    
    def run_quantum_planner_demo(self) -> Dict[str, Any]:
        """Demonstrate quantum task planning capabilities.""" 
        print("⚛️  Running Quantum Planner Demo...")
        
        # Create optimized planner configuration
        config = PlannerConfig(
            max_iterations=100,
            convergence_threshold=1e-4,
            quantum_interference_strength=0.15,
            enable_quantum_speedup=True,
            parallel_execution_limit=6
        )
        
        planner = QuantumTaskPlanner(config)
        
        # Add computational resources
        planner.add_resource("cpu_cores", 8.0)
        planner.add_resource("memory_gb", 16.0)
        planner.add_resource("gpu_memory_gb", 12.0)
        
        # Create interdependent tasks
        tasks = [
            QuantumTask(
                id="data_preprocessing",
                name="Preprocess Training Data",
                description="Clean and prepare RLHF training datasets",
                priority=0.9,
                estimated_duration=2.0,
                resource_requirements={"cpu_cores": 2.0, "memory_gb": 4.0}
            ),
            QuantumTask(
                id="model_training",
                name="Train Base Model",
                description="Train the foundational language model",
                priority=1.0,
                estimated_duration=8.0,
                resource_requirements={"cpu_cores": 4.0, "memory_gb": 8.0, "gpu_memory_gb": 8.0}
            ),
            QuantumTask(
                id="reward_model_training",
                name="Train Reward Model",
                description="Train the contractual reward model",
                priority=0.8,
                estimated_duration=4.0,
                resource_requirements={"cpu_cores": 3.0, "memory_gb": 6.0, "gpu_memory_gb": 4.0}
            ),
            QuantumTask(
                id="contract_verification",
                name="Verify Contract Compliance",
                description="Formally verify contract constraints",
                priority=0.7,
                estimated_duration=1.5,
                resource_requirements={"cpu_cores": 2.0, "memory_gb": 3.0}
            ),
            QuantumTask(
                id="deployment_preparation",
                name="Prepare Production Deployment",
                description="Package and prepare for deployment",
                priority=0.6,
                estimated_duration=1.0,
                resource_requirements={"cpu_cores": 1.0, "memory_gb": 2.0}
            )
        ]
        
        # Add tasks and dependencies
        for task in tasks:
            planner.add_task(task)
        
        # Create dependency relationships
        planner.tasks["model_training"].add_dependency("data_preprocessing")
        planner.tasks["reward_model_training"].add_dependency("data_preprocessing")
        planner.tasks["contract_verification"].add_dependency("reward_model_training")
        planner.tasks["deployment_preparation"].add_dependency("model_training")
        planner.tasks["deployment_preparation"].add_dependency("contract_verification")
        
        # Create quantum entanglements for parallel optimization
        planner.create_entanglement("model_training", "reward_model_training", strength=0.8)
        planner.create_entanglement("contract_verification", "deployment_preparation", strength=0.6)
        
        # Optimize execution plan
        optimization_result = planner.optimize_plan()
        
        # Execute the plan
        execution_result = planner.execute_plan(optimization_result)
        
        # Get quantum state summary
        quantum_summary = planner.get_quantum_state_summary()
        
        return {
            "optimization_time": optimization_result["optimization_time"],
            "fitness_score": optimization_result["fitness_score"],
            "total_tasks": len(tasks),
            "completed_tasks": len(execution_result["completed_tasks"]),
            "failed_tasks": len(execution_result["failed_tasks"]),
            "success_rate": execution_result["success_rate"],
            "resource_utilization": execution_result["resource_utilization"],
            "quantum_metrics": optimization_result["quantum_metrics"],
            "converged": optimization_result["converged"],
            "quantum_state": quantum_summary
        }
    
    def run_integration_demo(self) -> Dict[str, Any]:
        """Demonstrate integration between contracts and quantum planning."""
        print("🔄 Running Integration Demo...")
        
        # Create contract-constrained planning scenario
        contract = RewardContract(
            name="ProductionML-v1",
            stakeholders={"ml_team": 0.4, "security": 0.3, "business": 0.3}
        )
        
        # Add performance constraints
        def performance_constraint(state, action):
            return jnp.sum(action) <= 10.0  # Resource budget limit
        
        contract.add_constraint("performance_budget", performance_constraint, "Stay within computational budget")
        
        def security_constraint(state, action):
            return jnp.all(action >= 0.0)  # No negative resource usage
            
        contract.add_constraint("security_compliance", security_constraint, "Meet security requirements")
        
        # Add a simple reward function
        @contract.reward_function()  # Use default, no stakeholder specified
        def simple_reward(state, action):
            return jnp.mean(state) + jnp.mean(action)
        
        # Create contract-aware planner
        planner = QuantumTaskPlanner()
        
        # Add contract compliance checking to tasks
        compliance_task = QuantumTask(
            id="contract_compliance",
            name="Verify Contract Compliance",
            description="Ensure all tasks meet contractual obligations",
            priority=1.0,
            contract_constraints=["performance_budget", "security_compliance"]
        )
        
        planner.add_task(compliance_task)
        
        # Test contract compliance
        test_state = jnp.array([1.0, 2.0, 3.0])
        test_action = jnp.array([2.0, 3.0, 4.0])  # Sum = 9.0, within budget
        
        reward = contract.compute_reward(test_state, test_action)
        violations = contract.check_violations(test_state, test_action)
        
        return {
            "contract_reward": float(reward),
            "violations_detected": sum(violations.values()),
            "compliance_task_ready": compliance_task.is_ready(set()),
            "integration_successful": len(violations) > 0 and not any(violations.values())
        }
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks for key operations."""
        print("⚡ Running Performance Benchmarks...")
        
        results = {}
        
        # Benchmark reward computation
        contract = RewardContract(
            name="benchmark-contract",
            stakeholders={"benchmarker": 1.0}  # Add stakeholder to fix error
        )
        
        # Add a benchmark reward function
        @contract.reward_function("benchmarker")
        def benchmark_reward(state, action):
            return jnp.mean(state) * jnp.mean(action)
        
        state = jnp.array(np.random.rand(100))
        action = jnp.array(np.random.rand(50))
        
        start_time = time.time()
        for _ in range(100):
            contract.compute_reward(state, action, use_cache=False)
        uncached_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(100):
            contract.compute_reward(state, action, use_cache=True)
        cached_time = time.time() - start_time
        
        results["reward_computation"] = {
            "uncached_time_100_runs": uncached_time,
            "cached_time_100_runs": cached_time,
            "speedup_factor": uncached_time / cached_time if cached_time > 0 else float('inf')
        }
        
        # Benchmark quantum planning
        planner = QuantumTaskPlanner()
        for i in range(10):
            task = QuantumTask(f"task_{i}", f"Task {i}", "Benchmark task", priority=0.5)
            planner.add_task(task)
        
        start_time = time.time()
        optimization_result = planner.optimize_plan()
        planning_time = time.time() - start_time
        
        results["quantum_planning"] = {
            "planning_time_10_tasks": planning_time,
            "fitness_achieved": optimization_result["fitness_score"],
            "convergence_iterations": optimization_result["iterations"]
        }
        
        return results
    
    def run_full_demo(self) -> Dict[str, Any]:
        """Run comprehensive demonstration of all features."""
        print("🚀 Starting RLHF Contract Wizard Full Demo")
        print("=" * 60)
        
        demo_start = time.time()
        
        # Run all demo components
        self.demo_results["reward_contracts"] = self.run_reward_contract_demo()
        print("✅ Reward Contract Demo completed")
        
        self.demo_results["quantum_planning"] = self.run_quantum_planner_demo()
        print("✅ Quantum Planning Demo completed")
        
        self.demo_results["integration"] = self.run_integration_demo()
        print("✅ Integration Demo completed")
        
        self.demo_results["benchmarks"] = self.run_performance_benchmarks()
        print("✅ Performance Benchmarks completed")
        
        total_time = time.time() - demo_start
        
        # Summary statistics
        self.demo_results["summary"] = {
            "total_demo_time": total_time,
            "components_tested": 4,
            "contracts_created": 3,
            "tasks_planned": 5,
            "all_tests_passed": self._verify_demo_success()
        }
        
        print("\n🎉 Full Demo Completed Successfully!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print("📊 Results:", json.dumps(self.demo_results["summary"], indent=2))
        
        return self.demo_results
    
    # Helper methods for simulated computations (JAX-compatible)
    def _contains_harmful_content(self, action: jnp.ndarray) -> jnp.ndarray:
        """Simulate harmful content detection."""
        return jnp.max(action) > 0.9
    
    def _contains_pii(self, action: jnp.ndarray) -> jnp.ndarray:
        """Simulate PII detection."""
        return jnp.mean(action) > 0.8
    
    def _is_safe(self, action: jnp.ndarray) -> jnp.ndarray:
        """Simulate safety evaluation."""
        return jnp.max(action) < 0.8
    
    def _compute_efficiency(self, action: jnp.ndarray) -> jnp.ndarray:
        """Simulate efficiency computation."""
        return 1.0 / (1.0 + jnp.var(action))
    
    def _compute_satisfaction(self, action: jnp.ndarray) -> jnp.ndarray:
        """Simulate user satisfaction computation."""
        return jnp.tanh(jnp.mean(action))
    
    def _compute_helpfulness(self, action: jnp.ndarray) -> jnp.ndarray:
        """Simulate helpfulness computation."""
        # Use manual sigmoid implementation since jnp.sigmoid doesn't exist
        x = jnp.sum(action) - 1.0
        return 1.0 / (1.0 + jnp.exp(-x))
    
    def _compute_responsiveness(self, action: jnp.ndarray) -> jnp.ndarray:
        """Simulate responsiveness computation."""
        return 1.0 - jnp.exp(-jnp.mean(action))
    
    def _verify_demo_success(self) -> bool:
        """Verify that all demo components completed successfully."""
        try:
            # Check reward contracts
            rc = self.demo_results["reward_contracts"]
            if rc["computed_reward"] is None or rc["stakeholders"] != 3:
                return False
            
            # Check quantum planning
            qp = self.demo_results["quantum_planning"] 
            if qp["total_tasks"] != 5 or qp["fitness_score"] < 0.4:
                return False
            
            # Check integration
            integration = self.demo_results["integration"]
            if integration["violations_detected"] > 0:
                return False
            
            # Check benchmarks
            benchmarks = self.demo_results["benchmarks"]
            if benchmarks["reward_computation"]["speedup_factor"] < 1.0:
                return False
            
            return True
            
        except (KeyError, TypeError):
            return False


def main():
    """Main demo runner function."""
    demo = RLHFContractDemo()
    results = demo.run_full_demo()
    
    # Save results to file
    with open("demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to demo_results.json")
    return results


if __name__ == "__main__":
    results = main()
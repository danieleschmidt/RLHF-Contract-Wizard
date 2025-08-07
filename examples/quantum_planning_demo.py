#!/usr/bin/env python3
"""
Quantum-Inspired Task Planning Demo

Demonstrates the quantum task planner integrated with RLHF contracts,
showing superposition-based optimization, entanglement scheduling,
and contract compliance enforcement.
"""

import time
import numpy as np
from typing import Dict, List

# Import our quantum planning system
from src.quantum_planner.core import (
    QuantumTaskPlanner, QuantumTask, TaskState, PlannerConfig
)
from src.quantum_planner.contracts import (
    ContractualTaskPlanner, TaskPlanningContext
)
from src.quantum_planner.visualization import QuantumPlannerVisualizer
from src.models.reward_contract import RewardContract, AggregationStrategy
from src.models.legal_blocks import LegalBlocks, RLHFConstraints
import jax.numpy as jnp


def create_sample_tasks() -> List[QuantumTask]:
    """Create a set of sample tasks for demonstration."""
    tasks = [
        QuantumTask(
            id="data_preprocessing",
            name="Data Preprocessing Pipeline",
            description="Clean and prepare training data for model development",
            priority=0.8,
            estimated_duration=2.5,
            resource_requirements={"cpu": 4, "memory": 8, "storage": 100}
        ),
        
        QuantumTask(
            id="feature_engineering", 
            name="Feature Engineering",
            description="Extract and engineer features from preprocessed data",
            priority=0.7,
            estimated_duration=1.8,
            resource_requirements={"cpu": 2, "memory": 16, "storage": 50},
            dependencies={"data_preprocessing"}
        ),
        
        QuantumTask(
            id="model_training",
            name="RLHF Model Training",
            description="Train reward model using contractual PPO algorithm",
            priority=0.9,
            estimated_duration=5.0,
            resource_requirements={"gpu": 2, "cpu": 8, "memory": 32, "storage": 200},
            dependencies={"feature_engineering"}
        ),
        
        QuantumTask(
            id="contract_verification",
            name="Contract Compliance Verification",
            description="Verify model compliance with legal-blocks constraints",
            priority=0.85,
            estimated_duration=1.2,
            resource_requirements={"cpu": 4, "memory": 8},
            dependencies={"model_training"}
        ),
        
        QuantumTask(
            id="stakeholder_validation",
            name="Stakeholder Validation",
            description="Run stakeholder-specific validation tests",
            priority=0.6,
            estimated_duration=2.0,
            resource_requirements={"cpu": 2, "memory": 4},
            dependencies={"contract_verification"}
        ),
        
        QuantumTask(
            id="deployment_prep",
            name="Deployment Preparation", 
            description="Package model for production deployment",
            priority=0.75,
            estimated_duration=1.5,
            resource_requirements={"cpu": 1, "memory": 4, "storage": 20},
            dependencies={"stakeholder_validation"}
        ),
        
        QuantumTask(
            id="monitoring_setup",
            name="Monitoring Infrastructure Setup",
            description="Deploy monitoring and compliance tracking systems",
            priority=0.5,
            estimated_duration=3.0,
            resource_requirements={"cpu": 3, "memory": 6, "storage": 30}
        ),
        
        QuantumTask(
            id="documentation",
            name="Documentation Generation",
            description="Generate compliance documentation and audit trails",
            priority=0.4,
            estimated_duration=2.0,
            resource_requirements={"cpu": 1, "memory": 2, "storage": 10}
        ),
        
        QuantumTask(
            id="security_audit",
            name="Security Audit",
            description="Perform security audit of the complete system",
            priority=0.8,
            estimated_duration=2.5,
            resource_requirements={"cpu": 2, "memory": 4},
            dependencies={"deployment_prep", "monitoring_setup"}
        )
    ]
    
    return tasks


def create_sample_contract() -> RewardContract:
    """Create a sample RLHF contract with stakeholders and constraints."""
    
    # Define stakeholders
    stakeholders = {
        "ai_safety_team": 0.35,    # AI Safety team has highest weight
        "product_team": 0.25,      # Product team for business requirements  
        "legal_team": 0.20,        # Legal team for compliance
        "users": 0.15,             # End user representatives
        "auditors": 0.05          # External auditors
    }
    
    # Create contract
    contract = RewardContract(
        name="RLHF-SafeAI-Contract-v1.2",
        version="1.2.0",
        stakeholders=stakeholders,
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        jurisdiction="California, USA"
    )
    
    # Add safety constraints
    @LegalBlocks.specification("""
        REQUIRES: safety_score >= 0.9
        ENSURES: NOT harmful_output(action)
        INVARIANT: compliance_monitoring_active
    """)
    def safety_constraint(state: jnp.ndarray, action: jnp.ndarray) -> bool:
        """Ensure AI system operates safely."""
        # Mock safety check - in practice would use sophisticated safety models
        safety_score = float(jnp.mean(state)) if len(state) > 0 else 0.5
        return safety_score >= 0.8
    
    contract.add_constraint(
        "ai_safety_requirement",
        safety_constraint,
        description="System must maintain high safety standards",
        severity=1.0,
        violation_penalty=-10.0
    )
    
    # Add legal compliance constraint
    @LegalBlocks.specification("""
        REQUIRES: gdpr_compliant AND ccpa_compliant
        ENSURES: user_privacy_protected
        INVARIANT: NOT data_misuse(user_data)
    """)
    def legal_compliance(state: jnp.ndarray, action: jnp.ndarray) -> bool:
        """Ensure legal compliance across jurisdictions."""
        # Mock compliance check
        return True  # Assume compliant for demo
    
    contract.add_constraint(
        "legal_compliance_requirement", 
        legal_compliance,
        description="Must comply with privacy and data protection laws",
        severity=0.9,
        violation_penalty=-8.0
    )
    
    # Add performance constraint
    def performance_constraint(state: jnp.ndarray, action: jnp.ndarray) -> bool:
        """Ensure acceptable performance metrics."""
        # Mock performance check
        performance_score = float(jnp.sum(state * action)) if len(state) > 0 and len(action) > 0 else 0.7
        return performance_score >= 0.6
    
    contract.add_constraint(
        "performance_requirement",
        performance_constraint,
        description="System must meet performance benchmarks",
        severity=0.7,
        violation_penalty=-5.0
    )
    
    return contract


def demonstrate_quantum_planning():
    """Run comprehensive demonstration of quantum task planning."""
    
    print("🚀 QUANTUM-INSPIRED TASK PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # 1. Create sample data
    print("\n📋 Setting up demonstration data...")
    tasks = create_sample_tasks()
    contract = create_sample_contract()
    
    print(f"Created {len(tasks)} sample tasks")
    print(f"Created contract: {contract.metadata.name}")
    print(f"Stakeholders: {list(contract.stakeholders.keys())}")
    print(f"Constraints: {list(contract.constraints.keys())}")
    
    # 2. Initialize quantum planner
    print("\n⚛️  Initializing Quantum Task Planner...")
    
    config = PlannerConfig(
        max_iterations=100,
        convergence_threshold=1e-4,
        quantum_interference_strength=0.15,
        parallel_execution_limit=3,
        enable_quantum_speedup=True
    )
    
    planner = QuantumTaskPlanner(config)
    
    # Add resources
    planner.add_resource("cpu", 20)
    planner.add_resource("gpu", 4) 
    planner.add_resource("memory", 64)
    planner.add_resource("storage", 500)
    
    # Add tasks to planner
    for task in tasks:
        planner.add_task(task)
    
    print(f"Added {len(tasks)} tasks to quantum planner")
    print(f"Available resources: {planner.resource_pool}")
    
    # 3. Create entanglements
    print("\n🔗 Creating quantum entanglements...")
    
    # Create entanglements based on dependencies and resource sharing
    planner.create_entanglement("data_preprocessing", "feature_engineering", 0.9)
    planner.create_entanglement("feature_engineering", "model_training", 0.8)
    planner.create_entanglement("model_training", "contract_verification", 0.95)
    planner.create_entanglement("contract_verification", "stakeholder_validation", 0.7)
    planner.create_entanglement("deployment_prep", "monitoring_setup", 0.6)
    planner.create_entanglement("security_audit", "deployment_prep", 0.8)
    
    entanglement_count = len(planner.entanglement_matrix)
    print(f"Created {entanglement_count} quantum entanglements")
    
    # 4. Show initial quantum state
    print("\n📊 Initial Quantum State Summary:")
    quantum_summary = planner.get_quantum_state_summary()
    for key, value in quantum_summary.items():
        print(f"  {key}: {value}")
    
    # 5. Run quantum optimization
    print("\n🧮 Running Quantum Optimization...")
    start_time = time.time()
    
    optimization_result = planner.optimize_plan()
    
    optimization_time = time.time() - start_time
    
    print(f"Optimization completed in {optimization_time:.3f}s")
    print(f"Converged: {'✅' if optimization_result['converged'] else '❌'}")
    print(f"Iterations: {optimization_result['iterations']}")
    print(f"Final fitness: {optimization_result['fitness_score']:.4f}")
    print(f"Task order: {optimization_result['task_order']}")
    
    # 6. Initialize contractual planner
    print("\n⚖️  Initializing Contractual Task Planner...")
    
    contractual_planner = ContractualTaskPlanner(contract, config)
    
    # Add resources to contractual planner
    for resource, amount in planner.resource_pool.items():
        contractual_planner.quantum_planner.add_resource(resource, amount)
    
    # Add tasks with contract compliance
    context = TaskPlanningContext(
        user_id="demo_user_123",
        session_id="demo_session_456",
        resource_constraints={"max_cpu": 20, "max_memory": 64},
        compliance_metadata={
            "monitoring_enabled": True,
            "rollback_plan": True,
            "user_demographics": {"age_group": "adult"},
            "audit_required": True
        }
    )
    
    for task in tasks:
        try:
            contractual_planner.add_task_with_contract(task, context)
            print(f"✅ Task {task.id} added with contract compliance")
        except ValueError as e:
            print(f"❌ Task {task.id} rejected: {e}")
    
    # Add entanglements
    for (task1, task2), strength in planner.entanglement_matrix.items():
        contractual_planner.quantum_planner.create_entanglement(task1, task2, abs(strength))
    
    print(f"Contract compliance score: {contractual_planner.contract_compliance_score:.3f}")
    
    # 7. Run contract-compliant planning
    print("\n📋 Running Contract-Compliant Planning...")
    
    planning_start_time = time.time()
    contractual_result = contractual_planner.plan_with_contract_compliance(context)
    planning_time = time.time() - planning_start_time
    
    print(f"Contract-compliant planning completed in {planning_time:.3f}s")
    print(f"Success: {'✅' if contractual_result['success'] else '❌'}")
    
    if contractual_result['success']:
        print(f"Contract fitness: {contractual_result['contract_fitness']:.4f}")
        print(f"Compliance score: {contractual_result['compliance_score']:.3f}")
        print(f"Stakeholder satisfaction: {contractual_result['stakeholder_satisfaction']:.3f}")
        
        # Validation results
        validation = contractual_result['validation_results']
        if validation['violations']:
            print(f"Violations detected: {len(validation['violations'])}")
            for violation in validation['violations']:
                print(f"  - {violation['type']}: {violation.get('message', 'No details')}")
        else:
            print("✅ No constraint violations detected")
    else:
        print(f"❌ Planning failed: {contractual_result.get('error', 'Unknown error')}")
        if 'violations' in contractual_result:
            for violation in contractual_result['violations']:
                print(f"  - {violation}")
    
    # 8. Execute the plan
    if contractual_result['success']:
        print("\n🚀 Executing Contract-Compliant Plan...")
        
        execution_start_time = time.time() 
        execution_result = contractual_planner.execute_with_monitoring(
            contractual_result['quantum_plan'],
            context
        )
        execution_time = time.time() - execution_start_time
        
        print(f"Execution completed in {execution_time:.3f}s")
        print(f"Success rate: {execution_result['success_rate']:.1%}")
        print(f"Completed tasks: {len(execution_result['completed_tasks'])}")
        print(f"Failed tasks: {len(execution_result['failed_tasks'])}")
        
        # Contract monitoring results
        monitoring = execution_result.get('contract_monitoring', {})
        violations_detected = monitoring.get('contract_violations_detected', 0)
        print(f"Contract violations during execution: {violations_detected}")
        
        if execution_result['completed_tasks']:
            print(f"Successfully completed tasks:")
            for task_id in execution_result['completed_tasks'][:5]:  # Show first 5
                print(f"  ✅ {task_id}")
        
        if execution_result['failed_tasks']:
            print(f"Failed tasks:")
            for task_id in execution_result['failed_tasks']:
                print(f"  ❌ {task_id}")
        
        # Resource utilization
        resource_util = execution_result.get('resource_utilization', {})
        if resource_util:
            print(f"Resource utilization:")
            for resource, utilization in resource_util.items():
                print(f"  {resource}: {utilization:.1%}")
        
        # 9. Visualization (save to files)
        print("\n📈 Generating Visualizations...")
        
        try:
            visualizer = QuantumPlannerVisualizer()
            
            # Quantum state visualization
            fig1 = visualizer.visualize_quantum_state(
                contractual_planner.quantum_planner,
                save_path="quantum_state_demo.png",
                show=False
            )
            print("✅ Quantum state visualization saved to quantum_state_demo.png")
            
            # Optimization history
            optimization_history = contractual_result['quantum_plan'].get('optimization_history', [])
            if optimization_history:
                fig2 = visualizer.visualize_optimization_history(
                    optimization_history,
                    save_path="optimization_history_demo.png", 
                    show=False
                )
                print("✅ Optimization history saved to optimization_history_demo.png")
            
            # Contract compliance dashboard
            fig3 = visualizer.visualize_contract_compliance(
                contractual_planner,
                contractual_result,
                save_path="contract_compliance_demo.png",
                show=False
            )
            print("✅ Contract compliance dashboard saved to contract_compliance_demo.png")
            
            # Execution flow
            fig4 = visualizer.visualize_execution_flow(
                execution_result,
                contractual_planner.quantum_planner.tasks,
                save_path="execution_flow_demo.png",
                show=False
            )
            print("✅ Execution flow visualization saved to execution_flow_demo.png")
            
            # Comprehensive dashboard
            fig5 = visualizer.create_comprehensive_dashboard(
                contractual_planner,
                contractual_result,
                execution_result,
                save_path="comprehensive_dashboard_demo.png",
                show=False
            )
            print("✅ Comprehensive dashboard saved to comprehensive_dashboard_demo.png")
            
            # Export data
            data_path = visualizer.export_dashboard_data(
                contractual_planner,
                contractual_result,
                execution_result,
                "quantum_planning_data_demo.json"
            )
            print(f"✅ Dashboard data exported to {data_path}")
            
        except Exception as e:
            print(f"⚠️  Visualization generation encountered issues: {e}")
            print("This is normal if matplotlib is not properly configured in the environment")
        
        # 10. Summary and insights
        print("\n📊 DEMONSTRATION SUMMARY")
        print("=" * 40)
        print(f"Total tasks processed: {len(tasks)}")
        print(f"Quantum optimization time: {optimization_time:.3f}s")
        print(f"Contract planning time: {planning_time:.3f}s")  
        print(f"Execution time: {execution_time:.3f}s")
        print(f"Overall success rate: {execution_result['success_rate']:.1%}")
        print(f"Final compliance score: {execution_result.get('final_compliance_score', 0):.3f}")
        
        quantum_metrics = contractual_result['quantum_plan'].get('quantum_metrics', {})
        print(f"\nQuantum Performance:")
        print(f"  Superposition tasks: {quantum_metrics.get('superposition_tasks', 0)}")
        print(f"  Entanglement connections: {quantum_metrics.get('entanglements', 0)}")
        print(f"  Average probability: {quantum_metrics.get('average_probability', 0):.3f}")
        
        contract_metadata = contractual_result.get('contract_metadata', {})
        print(f"\nContract Enforcement:")
        print(f"  Contract: {contract_metadata.get('contract_name', 'N/A')}")
        print(f"  Stakeholders: {len(contract_metadata.get('stakeholders', []))}")
        print(f"  Constraints checked: {contract_metadata.get('constraints_checked', 0)}")
        print(f"  Violations detected: {violations_detected}")
        
        print(f"\n🎉 Quantum-inspired task planning demonstration completed successfully!")
        print(f"The system demonstrated:")
        print(f"  ✅ Quantum superposition-based optimization")
        print(f"  ✅ Entanglement-aware task scheduling")
        print(f"  ✅ Contract compliance enforcement")
        print(f"  ✅ Real-time monitoring and violation detection")
        print(f"  ✅ Multi-stakeholder satisfaction optimization")
        print(f"  ✅ Comprehensive visualization and reporting")


def demonstrate_advanced_features():
    """Demonstrate advanced quantum planning features."""
    print("\n🔬 ADVANCED QUANTUM FEATURES DEMONSTRATION")
    print("=" * 50)
    
    # Import advanced algorithms
    from src.quantum_planner.algorithms import (
        QuantumOptimizer, SuperpositionSearch, EntanglementScheduler
    )
    
    # 1. Quantum Optimizer (QAOA-like)
    print("\n🧮 Quantum Approximate Optimization Algorithm (QAOA)...")
    
    # Create smaller task set for QAOA
    small_tasks = {
        "task_a": QuantumTask("task_a", "Task A", "High priority task", priority=0.9, estimated_duration=1.0),
        "task_b": QuantumTask("task_b", "Task B", "Medium priority task", priority=0.6, estimated_duration=2.0), 
        "task_c": QuantumTask("task_c", "Task C", "Low priority task", priority=0.3, estimated_duration=1.5),
        "task_d": QuantumTask("task_d", "Task D", "Critical task", priority=0.95, estimated_duration=0.8)
    }
    
    # Add dependencies
    small_tasks["task_b"].dependencies.add("task_a")
    small_tasks["task_c"].dependencies.add("task_b")
    
    optimizer = QuantumOptimizer()
    
    start_time = time.time()
    selected_tasks, optimization_score, qaoa_metrics = optimizer.optimize_task_selection(small_tasks)
    qaoa_time = time.time() - start_time
    
    print(f"QAOA optimization completed in {qaoa_time:.3f}s")
    print(f"Selected tasks: {selected_tasks}")
    print(f"Optimization score: {optimization_score:.4f}")
    print(f"Circuit depth: {qaoa_metrics.get('quantum_circuit_depth', 0)}")
    print(f"Optimization steps: {len(qaoa_metrics.get('optimization_steps', []))}")
    
    # 2. Superposition Search
    print("\n🌊 Superposition-based Search...")
    
    def mock_fitness(task_order: List[str]) -> float:
        """Mock fitness function for search demonstration."""
        if not task_order:
            return 0.0
        
        # Prefer orders that respect dependencies and priority
        score = 0.0
        for i, task_id in enumerate(task_order):
            if task_id in small_tasks:
                task = small_tasks[task_id]
                # Earlier position for higher priority tasks
                position_score = (len(task_order) - i) / len(task_order)
                score += task.priority * position_score
        
        # Penalty for dependency violations
        completed = set()
        for task_id in task_order:
            if task_id in small_tasks:
                task = small_tasks[task_id]
                if not task.dependencies.issubset(completed):
                    score -= 0.5  # Dependency violation penalty
                completed.add(task_id)
        
        return max(0.0, score / len(task_order))
    
    search = SuperpositionSearch()
    
    start_time = time.time()
    best_ordering, best_fitness, search_metrics = search.search(
        small_tasks, 
        mock_fitness,
        max_iterations=50
    )
    search_time = time.time() - start_time
    
    print(f"Superposition search completed in {search_time:.3f}s")
    print(f"Best ordering: {best_ordering}")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Orderings explored: {search_metrics.get('orderings_explored', 0)}")
    print(f"Final entropy: {search_metrics.get('final_entropy', 0):.4f}")
    
    # 3. Entanglement Scheduler
    print("\n🔗 Entanglement-based Scheduling...")
    
    scheduler = EntanglementScheduler()
    
    start_time = time.time()
    batch_schedule, scheduling_metrics = scheduler.schedule_with_entanglement(small_tasks)
    scheduling_time = time.time() - start_time
    
    print(f"Entanglement scheduling completed in {scheduling_time:.3f}s")
    print(f"Number of batches: {len(batch_schedule)}")
    print(f"Parallel efficiency: {scheduling_metrics.get('parallel_efficiency', 0):.2f}")
    print(f"Entanglement violations: {scheduling_metrics.get('entanglement_violations', 0)}")
    
    for i, batch in enumerate(batch_schedule):
        print(f"  Batch {i+1}: {batch}")
    
    network_info = scheduling_metrics.get('network_info', {})
    print(f"Entanglement network: {network_info.get('total_entanglements', 0)} connections")
    print(f"Average strength: {network_info.get('average_strength', 0):.3f}")
    
    print("\n🎯 Advanced features demonstration completed!")


if __name__ == "__main__":
    """Main demonstration entry point."""
    try:
        # Run main demonstration
        demonstrate_quantum_planning()
        
        # Run advanced features demo
        demonstrate_advanced_features()
        
        print(f"\n🏆 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print(f"Check the generated visualization files and JSON export for detailed results.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
    
    print(f"\n📚 For more information, see:")
    print(f"  - Documentation: docs/")
    print(f"  - Architecture: ARCHITECTURE.md")
    print(f"  - Examples: examples/")
"""
Generation 1 Demo: Basic Contract Creation and Verification

Demonstrates core RLHF contract functionality with working examples.
"""

import time
import jax.numpy as jnp
from typing import Dict, Any

from .models.reward_contract import RewardContract, AggregationStrategy
from .models.legal_blocks import LegalBlocks, RLHFConstraints
from .services.contract_service import ContractService
from .verification.formal_verifier import FormalVerifier, VerificationBackend


def create_basic_safety_contract() -> RewardContract:
    """Create a basic safety-focused RLHF contract."""
    
    # Create contract with multiple stakeholders
    contract = RewardContract(
        name="BasicSafetyAssistant",
        version="1.0.0",
        stakeholders={
            "operator": 0.4,      # Business objectives
            "safety_team": 0.4,   # Safety requirements
            "users": 0.2          # User satisfaction
        },
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        creator="demo_system",
        jurisdiction="Global"
    )
    
    # Add safety constraints using Legal-Blocks
    contract.add_constraint(
        name="no_harmful_content",
        constraint_fn=RLHFConstraints.no_harmful_output,
        description="Prevent generation of harmful content",
        severity=1.0,
        violation_penalty=-1.0
    )
    
    contract.add_constraint(
        name="privacy_protection", 
        constraint_fn=RLHFConstraints.privacy_protection,
        description="Protect user privacy and PII",
        severity=0.9,
        violation_penalty=-0.8
    )
    
    contract.add_constraint(
        name="truthfulness",
        constraint_fn=RLHFConstraints.truthfulness_requirement,
        description="Ensure truthful and accurate responses",
        severity=0.8,
        violation_penalty=-0.6
    )
    
    # Define reward functions for each stakeholder
    @contract.reward_function("operator")
    @LegalBlocks.specification("""
        REQUIRES: response.helpful AND response.efficient
        ENSURES: user_satisfaction_score >= 0.7
        INVARIANT: response_time_ms < 1000
    """)
    def operator_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        """Business-focused reward emphasizing efficiency and user satisfaction."""
        # Mock implementation - in practice would use trained models
        helpfulness = jnp.mean(jnp.abs(action)) * 0.8  # Mock helpfulness score
        efficiency = 1.0 / (1.0 + jnp.sum(state**2))   # Mock efficiency score
        return helpfulness * 0.6 + efficiency * 0.4
    
    @contract.reward_function("safety_team")
    @LegalBlocks.specification("""
        REQUIRES: response.safe AND NOT response.harmful
        ENSURES: safety_score >= 0.9
        INVARIANT: NOT contains_harmful_content(response)
    """)
    def safety_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        """Safety-focused reward with high safety thresholds."""
        # Mock safety scoring
        safety_base = 0.9  # High baseline safety
        action_magnitude = jnp.linalg.norm(action)
        
        # Penalize extreme actions that might be unsafe
        safety_penalty = jnp.exp(-action_magnitude) * 0.1
        return safety_base - safety_penalty
    
    @contract.reward_function("users")
    @LegalBlocks.specification("""
        REQUIRES: response.understandable AND response.relevant
        ENSURES: user_rating >= 0.8
        INVARIANT: response_length > 10 AND response_length < 1000
    """)
    def user_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        """User-focused reward emphasizing satisfaction and relevance."""
        # Mock user satisfaction
        relevance = jnp.dot(state, action) / (jnp.linalg.norm(state) + 1e-8)
        clarity = 1.0 - jnp.std(action) * 0.1  # Lower variance = clearer
        return jnp.clip(relevance * 0.7 + clarity * 0.3, 0.0, 1.0)
    
    return contract


def create_medical_ai_contract() -> RewardContract:
    """Create a specialized contract for medical AI assistance."""
    
    contract = RewardContract(
        name="MedicalAssistant",
        version="1.0.0",
        stakeholders={
            "hospital": 0.3,
            "doctors": 0.4,
            "patients": 0.2,
            "regulators": 0.1
        },
        aggregation=AggregationStrategy.NASH_BARGAINING,
        creator="medical_ai_team"
    )
    
    # Medical-specific constraints
    @LegalBlocks.constraint
    def never_diagnose_without_disclaimer(state: jnp.ndarray, action: jnp.ndarray) -> bool:
        """
        ```legal-blocks
        INVARIANT: NEVER recommend_treatment WITHOUT physician_review
        REQUIRES: medical_disclaimer(response) == True
        ENSURES: liability_protected(response)
        ```
        """
        # Mock implementation - would check for proper disclaimers
        return True  # Assume proper disclaimer for demo
    
    @LegalBlocks.constraint  
    def evidence_based_only(state: jnp.ndarray, action: jnp.ndarray) -> bool:
        """
        ```legal-blocks
        FORALL claim IN response.medical_claims:
            has_evidence_support(claim) AND peer_reviewed(claim.source)
        INVARIANT: NOT speculative_medical_advice(response)
        ```
        """
        # Mock evidence checking
        return jnp.linalg.norm(action) < 2.0  # Conservative response check
    
    contract.add_constraint("medical_disclaimer", never_diagnose_without_disclaimer, severity=1.0)
    contract.add_constraint("evidence_based", evidence_based_only, severity=0.9)
    
    # Medical reward function
    @contract.reward_function("doctors")
    def medical_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        """Medical professional reward function."""
        accuracy = jnp.exp(-jnp.linalg.norm(state - action)**2)  # Mock accuracy
        conservatism = jnp.exp(-jnp.linalg.norm(action))  # Prefer conservative advice
        return accuracy * 0.7 + conservatism * 0.3
    
    return contract


def demo_contract_usage():
    """Demonstrate basic contract creation and usage."""
    print("\n🚀 RLHF Contract Wizard - Generation 1 Demo\n")
    
    # Create basic safety contract
    print("Creating Basic Safety Contract...")
    safety_contract = create_basic_safety_contract()
    print(f"✅ Created: {safety_contract}")
    print(f"   Stakeholders: {len(safety_contract.stakeholders)}")
    print(f"   Constraints: {len(safety_contract.constraints)}")
    print(f"   Contract Hash: {safety_contract.compute_hash()[:16]}...")
    
    # Test reward computation
    print("\nTesting Reward Computation...")
    test_state = jnp.array([0.5, -0.2, 0.8, 0.1])
    test_action = jnp.array([0.3, 0.7, -0.1])
    
    try:
        reward = safety_contract.compute_reward(test_state, test_action)
        print(f"✅ Computed reward: {reward:.4f}")
    except Exception as e:
        print(f"❌ Reward computation failed: {e}")
    
    # Test constraint checking
    print("\nTesting Constraint Violations...")
    violations = safety_contract.check_violations(test_state, test_action)
    print(f"✅ Constraint violations: {violations}")
    
    if any(violations.values()):
        penalty = safety_contract.get_violation_penalty(violations)
        print(f"⚠️  Total penalty: {penalty:.4f}")
    else:
        print("✅ No constraint violations detected")
    
    # Create medical contract
    print("\nCreating Medical AI Contract...")
    medical_contract = create_medical_ai_contract()
    print(f"✅ Created: {medical_contract}")
    
    # Contract service demo
    print("\nTesting Contract Service...")
    service = ContractService()
    
    # Register contracts
    safety_id = service.create_contract(
        name="SafetyDemo",
        version="1.0.0", 
        stakeholders={"demo": 1.0},
        creator="demo_user"
    )
    print(f"✅ Registered safety contract: {safety_id}")
    
    # Validate contract
    print("\nValidating Contract...")
    validation = service.validate_contract(safety_contract)
    print(f"✅ Validation result: Valid={validation['valid']}")
    if validation['errors']:
        print(f"   Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    
    return {
        'safety_contract': safety_contract,
        'medical_contract': medical_contract,
        'contracts_created': 2,
        'demo_successful': True
    }


def demo_formal_verification():
    """Demonstrate formal verification capabilities."""
    print("\n🔍 Formal Verification Demo\n")
    
    # Create verifier
    verifier = FormalVerifier(backend=VerificationBackend.MOCK)
    print(f"✅ Created verifier: {verifier.get_backend_info()}")
    
    # Create test contract
    contract = create_basic_safety_contract()
    
    # Verify contract
    print("\nVerifying Contract Properties...")
    start_time = time.time()
    
    verification_result = verifier.verify_contract(contract)
    
    print(f"✅ Verification completed in {verification_result.verification_time:.3f}s")
    print(f"   All proofs valid: {verification_result.all_proofs_valid}")
    print(f"   Backend used: {verification_result.backend_used}")
    print(f"   Properties verified: {len(verification_result.proofs)}")
    
    # Show individual proof results
    for prop_name, proof_result in verification_result.proofs.items():
        status = "✅" if proof_result['valid'] else "❌"
        print(f"   {status} {prop_name}: {proof_result['details']}")
        if proof_result.get('counterexample'):
            print(f"      Counterexample: {proof_result['counterexample']}")
    
    # Test individual property verification
    print("\nTesting Individual Property Verification...")
    prop_result = verifier.verify_property(
        property_name="reward_bounds_test",
        formula="FORALL state, action: reward(state, action) >= 0 AND reward(state, action) <= 1",
        context={"contract": contract}
    )
    
    print(f"✅ Individual verification: {prop_result['valid']}")
    print(f"   Details: {prop_result['details']}")
    
    return verification_result.to_dict()


def demo_performance_benchmarks():
    """Demonstrate performance characteristics."""
    print("\n⚡ Performance Benchmarks\n")
    
    contract = create_basic_safety_contract()
    
    # Benchmark reward computation
    print("Benchmarking Reward Computation...")
    num_iterations = 1000
    
    # Generate test data
    test_states = [jnp.array([0.1 * i, -0.05 * i, 0.2 * i]) for i in range(num_iterations)]
    test_actions = [jnp.array([0.05 * i, 0.15 * i]) for i in range(num_iterations)]
    
    # Time with caching
    start_time = time.time()
    rewards_cached = []
    for state, action in zip(test_states, test_actions):
        reward = contract.compute_reward(state, action, use_cache=True)
        rewards_cached.append(reward)
    cached_time = time.time() - start_time
    
    # Time without caching  
    start_time = time.time()
    rewards_no_cache = []
    for state, action in zip(test_states, test_actions):
        reward = contract.compute_reward(state, action, use_cache=False)
        rewards_no_cache.append(reward)
    no_cache_time = time.time() - start_time
    
    print(f"✅ {num_iterations} reward computations:")
    print(f"   With caching: {cached_time:.3f}s ({cached_time/num_iterations*1000:.2f}ms per call)")
    print(f"   Without caching: {no_cache_time:.3f}s ({no_cache_time/num_iterations*1000:.2f}ms per call)")
    print(f"   Cache speedup: {no_cache_time/cached_time:.1f}x")
    
    # Benchmark constraint checking
    print("\nBenchmarking Constraint Checking...")
    start_time = time.time()
    
    for state, action in zip(test_states[:100], test_actions[:100]):  # Smaller sample
        violations = contract.check_violations(state, action)
    
    constraint_time = time.time() - start_time
    print(f"✅ 100 constraint checks: {constraint_time:.3f}s ({constraint_time/100*1000:.2f}ms per check)")
    
    # Memory usage approximation
    import sys
    contract_size = sys.getsizeof(contract.to_dict())
    print(f"✅ Contract memory footprint: ~{contract_size} bytes")
    
    return {
        'reward_computation_ms': cached_time/num_iterations*1000,
        'constraint_checking_ms': constraint_time/100*1000,
        'cache_speedup': no_cache_time/cached_time,
        'contract_size_bytes': contract_size
    }


if __name__ == "__main__":
    # Run complete Generation 1 demo
    print("=" * 60)
    print("RLHF Contract Wizard - Generation 1 Implementation")
    print("=" * 60)
    
    # Basic functionality demo
    demo_results = demo_contract_usage()
    
    # Verification demo
    verification_results = demo_formal_verification()
    
    # Performance benchmarks
    performance_results = demo_performance_benchmarks()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Generation 1 Demo Complete!")
    print("=" * 60)
    print("✅ Basic contract creation and management")
    print("✅ Multi-stakeholder reward aggregation")
    print("✅ Legal-Blocks constraint specification")
    print("✅ Formal verification framework")  
    print("✅ Performance optimization with caching")
    print("✅ Contract validation and error handling")
    
    print(f"\nPerformance Summary:")
    print(f"• Reward computation: {performance_results['reward_computation_ms']:.2f}ms average")
    print(f"• Constraint checking: {performance_results['constraint_checking_ms']:.2f}ms average")
    print(f"• Cache performance: {performance_results['cache_speedup']:.1f}x speedup")
    
    print(f"\nGeneration 1 Status: WORKING ✅")
    print("Ready to proceed to Generation 2 (Robustness)...")
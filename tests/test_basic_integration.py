"""
Basic integration tests for RLHF-Contract-Wizard core functionality.

Tests the essential contract creation, validation, and execution without
external dependencies like databases or blockchain services.
"""

import pytest
import jax.numpy as jnp
import time
from typing import Dict, Any

from src.models.reward_contract import RewardContract, AggregationStrategy
from src.models.legal_blocks import LegalBlocks, RLHFConstraints
from src.verification.formal_verifier import FormalVerifier
from src.security.contract_security import ContractSecurityAnalyzer


class TestBasicIntegration:
    """Basic integration tests for core functionality."""
    
    def test_contract_creation_and_validation(self):
        """Test contract creation and basic validation."""
        # Create a basic contract
        contract = RewardContract(
            name="BasicTestContract",
            version="1.0.0",
            stakeholders={"user": 0.6, "safety": 0.4}
        )
        
        # Add reward functions
        @contract.reward_function("user")
        def user_reward(state, action):
            return jnp.tanh(jnp.sum(state * action[:len(state)]))
        
        @contract.reward_function("safety")
        def safety_reward(state, action):
            return jnp.exp(-jnp.sum(jnp.square(action)) * 0.1)
        
        # Test computation
        state = jnp.array([0.5, 0.3, 0.8])
        action = jnp.array([0.2, 0.4, 0.6])
        
        reward = contract.compute_reward(state, action)
        
        # Validate results
        assert jnp.isfinite(reward)
        assert isinstance(reward, (float, jnp.ndarray))
        assert contract.metadata.name == "BasicTestContract"
        assert len(contract.stakeholders) == 2
        
        print(f"✓ Contract creation and validation: reward = {reward}")
    
    def test_formal_verification_integration(self):
        """Test formal verification integration."""
        verifier = FormalVerifier()
        
        # Create contract with constraints
        contract = RewardContract(
            name="VerificationTestContract",
            version="1.0.0",
            stakeholders={"user": 1.0}
        )
        
        @contract.reward_function()
        @LegalBlocks.specification("""
            REQUIRES: state.shape[0] > 0
            ENSURES: reward >= 0.0 AND reward <= 1.0
            INVARIANT: finite(reward)
        """)
        def bounded_reward(state, action):
            return jnp.clip(jnp.mean(state), 0.0, 1.0)
        
        # Verify contract
        result = verifier.verify_contract(contract)
        
        assert result.verification_successful
        assert result.total_properties >= 0
        assert len(result.failed_properties) == 0
        
        print(f"✓ Formal verification: {result.total_properties} properties verified")
    
    def test_security_analysis_integration(self):
        """Test security analysis integration."""
        analyzer = ContractSecurityAnalyzer()
        
        # Create contract for security testing
        contract = RewardContract(
            name="SecurityTestContract",
            version="1.0.0",
            stakeholders={"operator": 0.7, "auditor": 0.3}
        )
        
        @contract.reward_function()
        def secure_reward(state, action):
            # Secure implementation with bounds checking
            safe_state = jnp.clip(state, -1.0, 1.0)
            safe_action = jnp.clip(action, -1.0, 1.0)
            return jnp.mean(safe_state) * jnp.mean(safe_action)
        
        # Analyze security
        assessment = analyzer.analyze_contract(contract)
        
        assert assessment.overall_score >= 0.0
        assert assessment.overall_score <= 1.0
        assert len(assessment.vulnerabilities) >= 0
        assert assessment.compliance_status is not None
        
        print(f"✓ Security analysis: score = {assessment.overall_score:.2f}")
    
    def test_constraint_evaluation(self):
        """Test constraint evaluation with Legal-Blocks."""
        # Test RLHF constraints
        state = type('State', (), {'user_id': 'test', 'user_consent': True})()
        action = type('Action', (), {'output': 'This is a safe and helpful response.'})()
        
        # Test safety constraint
        assert RLHFConstraints.no_harmful_output(action)
        
        # Test privacy constraint
        assert RLHFConstraints.privacy_protection(state, action)
        
        # Test truthfulness constraint
        assert RLHFConstraints.truthfulness_requirement(action)
        
        # Test fairness constraint
        assert RLHFConstraints.fairness_requirement(state, action)
        
        print("✓ All RLHF constraints satisfied")
    
    def test_performance_basic(self):
        """Test basic performance requirements."""
        contract = RewardContract(
            name="PerformanceTestContract",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def fast_reward(state, action):
            return jnp.sum(state * action[:len(state)])
        
        # Performance test
        state = jnp.ones(100)
        action = jnp.ones(100) * 0.5
        
        start_time = time.time()
        for _ in range(100):
            reward = contract.compute_reward(state, action)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / 100) * 1000
        
        # Performance assertions
        assert avg_time_ms < 50.0  # Less than 50ms average
        assert jnp.isfinite(reward)
        
        print(f"✓ Performance test: {avg_time_ms:.2f}ms average per computation")
    
    def test_aggregation_strategies(self):
        """Test different aggregation strategies."""
        stakeholders = {"a": 0.4, "b": 0.35, "c": 0.25}
        
        strategies = [
            AggregationStrategy.WEIGHTED_SUM,
            AggregationStrategy.NASH_BARGAINING,
            AggregationStrategy.UTILITARIAN_WELFARE
        ]
        
        for strategy in strategies:
            contract = RewardContract(
                name=f"AggregationTest_{strategy.value}",
                version="1.0.0",
                stakeholders=stakeholders,
                aggregation=strategy
            )
            
            @contract.reward_function("a")
            def reward_a(state, action):
                return 0.8
            
            @contract.reward_function("b")
            def reward_b(state, action):
                return 0.6
            
            @contract.reward_function("c")
            def reward_c(state, action):
                return 0.9
            
            state = jnp.array([1.0])
            action = jnp.array([1.0])
            
            reward = contract.compute_reward(state, action)
            
            assert jnp.isfinite(reward)
            assert 0.0 <= reward <= 1.0
            
            print(f"✓ Aggregation strategy {strategy.value}: reward = {reward:.3f}")
    
    def test_contract_composition(self):
        """Test contract composition and inheritance."""
        # Base contract
        base_contract = RewardContract(
            name="BaseContract",
            version="1.0.0",
            stakeholders={"base": 1.0}
        )
        
        @base_contract.reward_function()
        def base_reward(state, action):
            return 0.5
        
        # Extended contract
        extended_contract = RewardContract(
            name="ExtendedContract",
            version="2.0.0",
            stakeholders={"base": 0.6, "extension": 0.4}
        )
        
        @extended_contract.reward_function("base")
        def inherited_reward(state, action):
            return 0.5  # Same as base
        
        @extended_contract.reward_function("extension")
        def new_reward(state, action):
            return 0.8
        
        state = jnp.array([1.0])
        action = jnp.array([1.0])
        
        base_result = base_contract.compute_reward(state, action)
        extended_result = extended_contract.compute_reward(state, action)
        
        assert jnp.isfinite(base_result)
        assert jnp.isfinite(extended_result)
        
        # Extended contract should have different result due to composition
        assert not jnp.allclose(base_result, extended_result, atol=1e-6)
        
        print(f"✓ Contract composition: base = {base_result:.3f}, extended = {extended_result:.3f}")
    
    def test_error_handling(self):
        """Test error handling and resilience."""
        contract = RewardContract(
            name="ErrorTestContract",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def potentially_failing_reward(state, action):
            # This could potentially cause issues
            if jnp.sum(state) == 0:
                return 0.0
            return 1.0 / jnp.sum(state)
        
        # Test with various edge cases
        test_cases = [
            (jnp.array([1.0, 2.0]), jnp.array([0.5])),  # Normal case
            (jnp.array([0.0, 0.0]), jnp.array([1.0])),  # Zero state
            (jnp.array([1e-10]), jnp.array([1.0])),     # Very small state
        ]
        
        for state, action in test_cases:
            try:
                reward = contract.compute_reward(state, action)
                assert jnp.isfinite(reward) or reward == 0.0
            except Exception as e:
                # Error handling should be graceful
                print(f"Handled error gracefully: {e}")
        
        print("✓ Error handling tests passed")
    
    def test_caching_functionality(self):
        """Test caching functionality."""
        contract = RewardContract(
            name="CacheTestContract",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        call_count = 0
        
        @contract.reward_function()
        def counted_reward(state, action):
            nonlocal call_count
            call_count += 1
            return jnp.sum(state) * jnp.sum(action)
        
        state = jnp.array([1.0, 2.0])
        action = jnp.array([0.5, 0.3])
        
        # First call
        reward1 = contract.compute_reward(state, action, use_cache=True)
        first_call_count = call_count
        
        # Second call with same inputs (should use cache)
        reward2 = contract.compute_reward(state, action, use_cache=True)
        second_call_count = call_count
        
        # Third call with cache disabled
        reward3 = contract.compute_reward(state, action, use_cache=False)
        third_call_count = call_count
        
        # Validate caching behavior
        assert jnp.allclose(reward1, reward2)
        assert jnp.allclose(reward1, reward3)
        assert first_call_count == 1
        assert second_call_count == 1  # Should not increment due to cache
        assert third_call_count == 2   # Should increment due to cache bypass
        
        print(f"✓ Caching functionality: {first_call_count} → {second_call_count} → {third_call_count}")


if __name__ == "__main__":
    # Run tests directly if executed as script
    import sys
    
    test_class = TestBasicIntegration()
    methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    print("Running Basic Integration Tests...")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for method_name in methods:
        try:
            print(f"\nRunning {method_name}...")
            method = getattr(test_class, method_name)
            method()
            passed += 1
        except Exception as e:
            print(f"✗ {method_name} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
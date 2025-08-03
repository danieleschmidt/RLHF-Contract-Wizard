"""
Unit tests for RewardContract model.

Tests the core reward contract functionality including stakeholder management,
constraint handling, and reward computation.
"""

import pytest
import jax.numpy as jnp
from unittest.mock import Mock

from src.models.reward_contract import (
    RewardContract,
    Stakeholder,
    Constraint,
    AggregationStrategy
)
from src.models.legal_blocks import LegalBlocks


class TestRewardContract:
    """Test cases for RewardContract class."""
    
    def test_contract_creation(self, sample_stakeholders):
        """Test basic contract creation."""
        contract = RewardContract(
            name="TestContract",
            version="1.0.0",
            stakeholders=sample_stakeholders,
            creator="test_user"
        )
        
        assert contract.metadata.name == "TestContract"
        assert contract.metadata.version == "1.0.0"
        assert contract.metadata.creator == "test_user"
        assert len(contract.stakeholders) == 4
        
        # Check weight normalization
        total_weight = sum(s.weight for s in contract.stakeholders.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_contract_creation_empty_stakeholders(self):
        """Test contract creation without stakeholders."""
        contract = RewardContract(
            name="EmptyContract",
            version="1.0.0",
            creator="test_user"
        )
        
        assert len(contract.stakeholders) == 0
        assert contract.metadata.name == "EmptyContract"
    
    def test_add_stakeholder(self, basic_contract):
        """Test adding stakeholders to contract."""
        initial_count = len(basic_contract.stakeholders)
        
        basic_contract.add_stakeholder("new_stakeholder", 0.5)
        
        assert len(basic_contract.stakeholders) == initial_count + 1
        assert "new_stakeholder" in basic_contract.stakeholders
        
        # Check weight normalization after addition
        total_weight = sum(s.weight for s in basic_contract.stakeholders.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_add_constraint(self, basic_contract):
        """Test adding constraints to contract."""
        initial_count = len(basic_contract.constraints)
        
        def test_constraint(state, action):
            return jnp.sum(action) > 0
        
        basic_contract.add_constraint(
            name="test_constraint",
            constraint_fn=test_constraint,
            description="Test constraint",
            severity=5.0,
            violation_penalty=-5.0
        )
        
        assert len(basic_contract.constraints) == initial_count + 1
        assert "test_constraint" in basic_contract.constraints
        
        constraint = basic_contract.constraints["test_constraint"]
        assert constraint.severity == 5.0
        assert constraint.violation_penalty == -5.0
    
    def test_reward_function_decorator(self):
        """Test reward function decorator."""
        contract = RewardContract(
            name="TestContract",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def test_reward(state, action):
            return 0.5
        
        assert "default" in contract.reward_functions
        assert contract.reward_functions["default"] == test_reward
    
    def test_stakeholder_specific_reward_function(self):
        """Test stakeholder-specific reward function."""
        contract = RewardContract(
            name="TestContract",
            version="1.0.0",
            stakeholders={"operator": 0.6, "users": 0.4}
        )
        
        @contract.reward_function("operator")
        def operator_reward(state, action):
            return 0.8
        
        @contract.reward_function("users") 
        def user_reward(state, action):
            return 0.6
        
        assert "operator" in contract.reward_functions
        assert "users" in contract.reward_functions
    
    def test_compute_reward_basic(self, basic_contract, sample_state, sample_action):
        """Test basic reward computation."""
        reward = basic_contract.compute_reward(sample_state, sample_action)
        
        assert isinstance(reward, (float, jnp.float32, jnp.float64))
        assert not jnp.isnan(reward)
        assert not jnp.isinf(reward)
    
    def test_compute_reward_with_violations(self, sample_state, sample_action):
        """Test reward computation with constraint violations."""
        contract = RewardContract(
            name="TestContract",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def base_reward(state, action):
            return 1.0
        
        # Add constraint that always fails
        def failing_constraint(state, action):
            return False
        
        contract.add_constraint(
            name="always_fail",
            constraint_fn=failing_constraint,
            violation_penalty=-0.5
        )
        
        reward = contract.compute_reward(sample_state, sample_action)
        
        # Should be base reward (1.0) + violation penalty (-0.5) = 0.5
        assert reward == 0.5
    
    def test_check_violations(self, sample_state, sample_action):
        """Test constraint violation checking."""
        contract = RewardContract(
            name="TestContract", 
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        def good_constraint(state, action):
            return True
        
        def bad_constraint(state, action):
            return False
        
        contract.add_constraint("good", good_constraint)
        contract.add_constraint("bad", bad_constraint)
        
        violations = contract.check_violations(sample_state, sample_action)
        
        assert violations["good"] is False  # No violation
        assert violations["bad"] is True    # Violation
    
    def test_aggregation_strategies(self, sample_state, sample_action):
        """Test different aggregation strategies."""
        stakeholders = {"a": 0.3, "b": 0.7}
        
        # Test weighted average
        contract_wa = RewardContract(
            name="TestWA",
            version="1.0.0",
            stakeholders=stakeholders,
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE
        )
        
        @contract_wa.reward_function("a")
        def reward_a(state, action):
            return 1.0
        
        @contract_wa.reward_function("b")
        def reward_b(state, action):
            return 0.0
        
        reward_wa = contract_wa.compute_reward(sample_state, sample_action)
        expected_wa = 0.3 * 1.0 + 0.7 * 0.0  # 0.3
        assert abs(reward_wa - expected_wa) < 1e-6
        
        # Test utilitarian
        contract_util = RewardContract(
            name="TestUtil",
            version="1.0.0",
            stakeholders=stakeholders,
            aggregation=AggregationStrategy.UTILITARIAN
        )
        
        @contract_util.reward_function("a")
        def reward_a_util(state, action):
            return 1.0
        
        @contract_util.reward_function("b")
        def reward_b_util(state, action):
            return 0.5
        
        reward_util = contract_util.compute_reward(sample_state, sample_action)
        expected_util = 1.0 + 0.5  # 1.5
        assert abs(reward_util - expected_util) < 1e-6
    
    def test_contract_hash_computation(self, basic_contract):
        """Test contract hash computation."""
        hash1 = basic_contract.compute_hash()
        hash2 = basic_contract.compute_hash()
        
        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 64  # SHA256 hex length
        assert isinstance(hash1, str)
    
    def test_contract_hash_changes_with_modifications(self, basic_contract):
        """Test that contract hash changes when contract is modified."""
        original_hash = basic_contract.compute_hash()
        
        # Modify contract
        basic_contract.add_stakeholder("new_stakeholder", 0.1)
        
        new_hash = basic_contract.compute_hash()
        assert original_hash != new_hash
    
    def test_contract_to_dict(self, basic_contract):
        """Test contract serialization to dictionary."""
        contract_dict = basic_contract.to_dict()
        
        assert isinstance(contract_dict, dict)
        assert "metadata" in contract_dict
        assert "stakeholders" in contract_dict
        assert "constraints" in contract_dict
        assert "aggregation_strategy" in contract_dict
        assert "contract_hash" in contract_dict
        
        # Check metadata
        metadata = contract_dict["metadata"]
        assert metadata["name"] == "TestContract"
        assert metadata["version"] == "1.0.0"
        assert metadata["creator"] == "test_user"
    
    def test_contract_repr(self, basic_contract):
        """Test contract string representation."""
        repr_str = repr(basic_contract)
        
        assert "RewardContract" in repr_str
        assert "TestContract" in repr_str
        assert "1.0.0" in repr_str
    
    def test_legal_blocks_integration(self):
        """Test integration with Legal-Blocks specifications."""
        contract = RewardContract(
            name="LegalTest",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        @LegalBlocks.specification("""
            REQUIRES: action_valid(action)
            ENSURES: reward >= 0.0 AND reward <= 1.0
            INVARIANT: NOT harmful(action)
        """)
        def legal_reward(state, action):
            return 0.7
        
        # Check that Legal-Blocks metadata is attached
        assert hasattr(legal_reward, '__legal_blocks__')
        blocks_info = legal_reward.__legal_blocks__
        assert 'specification' in blocks_info
        assert 'blocks' in blocks_info
        assert len(blocks_info['blocks']) > 0


class TestStakeholder:
    """Test cases for Stakeholder class."""
    
    def test_stakeholder_creation(self):
        """Test stakeholder creation."""
        stakeholder = Stakeholder(
            name="test_stakeholder",
            weight=0.5,
            voting_power=2.0,
            address="0x123..."
        )
        
        assert stakeholder.name == "test_stakeholder"
        assert stakeholder.weight == 0.5
        assert stakeholder.voting_power == 2.0
        assert stakeholder.address == "0x123..."


class TestConstraint:
    """Test cases for Constraint class."""
    
    def test_constraint_creation(self):
        """Test constraint creation."""
        def test_fn(state, action):
            return True
        
        constraint = Constraint(
            name="test_constraint",
            description="Test description",
            constraint_fn=test_fn,
            severity=5.0,
            violation_penalty=-2.0
        )
        
        assert constraint.name == "test_constraint"
        assert constraint.description == "Test description"
        assert constraint.constraint_fn == test_fn
        assert constraint.severity == 5.0
        assert constraint.violation_penalty == -2.0
        assert constraint.enabled is True


class TestPerformance:
    """Performance tests for reward contracts."""
    
    @pytest.mark.slow
    def test_reward_computation_performance(self, performance_contract, large_state, large_action):
        """Test reward computation performance with large inputs."""
        import time
        
        start_time = time.time()
        reward = performance_contract.compute_reward(large_state, large_action)
        computation_time = time.time() - start_time
        
        assert computation_time < 0.1  # Should compute in under 100ms
        assert not jnp.isnan(reward)
    
    @pytest.mark.slow
    def test_many_stakeholders_performance(self, large_state, large_action):
        """Test performance with many stakeholders."""
        # Create contract with many stakeholders
        stakeholders = {f"stakeholder_{i}": 1.0/50 for i in range(50)}
        
        contract = RewardContract(
            name="ManyStakeholders",
            version="1.0.0",
            stakeholders=stakeholders
        )
        
        @contract.reward_function()
        def simple_reward(state, action):
            return jnp.mean(state)
        
        import time
        start_time = time.time()
        reward = contract.compute_reward(large_state, large_action)
        computation_time = time.time() - start_time
        
        assert computation_time < 0.5  # Should handle 50 stakeholders efficiently
        assert not jnp.isnan(reward)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_stakeholders_reward_computation(self, sample_state, sample_action):
        """Test reward computation with no stakeholders."""
        contract = RewardContract(
            name="Empty",
            version="1.0.0"
        )
        
        # Should handle empty stakeholders gracefully
        with pytest.raises(ValueError):
            contract.compute_reward(sample_state, sample_action)
    
    def test_nan_action_handling(self, basic_contract, sample_state):
        """Test handling of NaN values in action."""
        nan_action = jnp.array([float('nan'), 1.0, 2.0])
        
        # Should handle NaN gracefully without crashing
        try:
            reward = basic_contract.compute_reward(sample_state, nan_action)
            # If it doesn't raise, the result should be NaN or handled appropriately
            assert jnp.isnan(reward) or isinstance(reward, (float, int))
        except (ValueError, RuntimeError):
            # It's also acceptable to raise an error for NaN inputs
            pass
    
    def test_zero_weight_stakeholder(self):
        """Test stakeholder with zero weight."""
        with pytest.raises(ValueError):
            RewardContract(
                name="ZeroWeight",
                version="1.0.0",
                stakeholders={"zero": 0.0, "normal": 1.0}
            )
    
    def test_negative_weight_stakeholder(self):
        """Test stakeholder with negative weight.""" 
        with pytest.raises(ValueError):
            RewardContract(
                name="NegativeWeight",
                version="1.0.0",
                stakeholders={"negative": -0.5, "positive": 1.5}
            )


class TestConstraintViolations:
    """Test constraint violation scenarios."""
    
    def test_constraint_exception_handling(self, sample_state, sample_action):
        """Test handling of exceptions in constraint functions."""
        contract = RewardContract(
            name="ExceptionTest",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        def failing_constraint(state, action):
            raise RuntimeError("Constraint failed")
        
        contract.add_constraint("failing", failing_constraint)
        
        violations = contract.check_violations(sample_state, sample_action)
        
        # Should treat exception as violation
        assert violations["failing"] is True
    
    def test_multiple_violations(self, sample_state, sample_action):
        """Test multiple constraint violations."""
        contract = RewardContract(
            name="MultiViolation",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def base_reward(state, action):
            return 2.0
        
        # Add multiple failing constraints with different penalties
        contract.add_constraint("fail1", lambda s, a: False, violation_penalty=-0.5)
        contract.add_constraint("fail2", lambda s, a: False, violation_penalty=-0.3)
        contract.add_constraint("pass", lambda s, a: True, violation_penalty=-1.0)
        
        reward = contract.compute_reward(sample_state, sample_action)
        
        # Should be base (2.0) + fail1 (-0.5) + fail2 (-0.3) = 1.2
        expected = 2.0 - 0.5 - 0.3
        assert abs(reward - expected) < 1e-6
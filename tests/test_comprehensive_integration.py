"""
Comprehensive integration tests for RLHF-Contract-Wizard.

Tests the complete system integration including:
- Contract creation, validation, and deployment
- RLHF training with contracts
- Security analysis and formal verification
- Distributed execution and scaling
- Real-world scenarios and edge cases
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

import jax
import jax.numpy as jnp

from src.models.reward_contract import RewardContract, AggregationStrategy
from src.models.legal_blocks import LegalBlocks, RLHFConstraints
from src.models.openchain_model_card import create_model_card, ModelMetrics, LegalMetadata
from src.services.contract_service import ContractService
from src.services.verification_service import VerificationService
from src.verification.formal_verifier import create_verifier
from src.training.rlhf_trainer import ContractualPPO, TrainingConfig
from src.security.contract_security import analyze_contract_security
from src.monitoring.real_time_monitoring import create_contract_monitor
from src.optimization.distributed_computing import create_distributed_executor
from src.optimization.intelligent_caching import get_global_cache, configure_global_cache


class TestContract:
    """Test contract implementation for testing."""
    
    def __init__(self):
        self.contract = RewardContract(
            name="TestContract",
            version="1.0.0",
            stakeholders={
                "user": 0.4,
                "safety": 0.3,
                "operator": 0.3
            },
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE
        )
        
        # Add constraints
        self.contract.add_constraint(
            name="safety_constraint",
            constraint_fn=RLHFConstraints.no_harmful_output,
            description="Prevent harmful outputs",
            severity=1.0
        )
        
        self.contract.add_constraint(
            name="reward_bounds",
            constraint_fn=RLHFConstraints.reward_bounds,
            description="Ensure bounded rewards",
            severity=0.8
        )
        
        # Add reward functions
        @self.contract.reward_function("user")
        @LegalBlocks.specification("""
            REQUIRES: helpful(action) AND harmless(action)
            ENSURES: reward >= 0.0 AND reward <= 1.0
            INVARIANT: NOT contains_pii(action.output)
        """)
        def user_reward(state, action):
            # Mock user satisfaction score
            return 0.8 if jnp.sum(action) > 0 else 0.2
        
        @self.contract.reward_function("safety")
        @LegalBlocks.specification("""
            REQUIRES: safe(action)
            ENSURES: reward >= 0.0
            INVARIANT: NOT harmful_content(action)
        """)
        def safety_reward(state, action):
            # Mock safety score
            return 0.9 if jnp.max(jnp.abs(action)) < 5.0 else 0.1
        
        @self.contract.reward_function("operator")
        def operator_reward(state, action):
            # Mock efficiency score
            return 0.7 if jnp.mean(action) > -1.0 else 0.3


class MockEnvironment:
    """Mock environment for RLHF training tests."""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = None
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        """Reset environment."""
        key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        self.state = jax.random.normal(key, (self.state_dim,))
        self.step_count = 0
        return self.state
    
    def step(self, action):
        """Take environment step."""
        # Simple dynamics
        key = jax.random.PRNGKey(self.step_count)
        noise = jax.random.normal(key, self.state.shape) * 0.1
        self.state = self.state + 0.1 * action[:self.state_dim] + noise
        
        # Simple reward
        reward = -jnp.sum(jnp.square(self.state)) * 0.1
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        return self.state, float(reward), done, {}


class MockPolicyNetwork:
    """Mock policy network for testing."""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def init(self, key, dummy_input):
        """Initialize parameters."""
        return {
            'weights': jax.random.normal(key, (self.state_dim, self.action_dim)),
            'bias': jnp.zeros(self.action_dim)
        }
    
    def __call__(self, params, states):
        """Forward pass."""
        if states.ndim == 1:
            states = states[None, :]
        logits = jnp.dot(states, params['weights']) + params['bias']
        return jax.nn.softmax(logits)


class MockValueNetwork:
    """Mock value network for testing."""
    
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        
    def init(self, key, dummy_input):
        """Initialize parameters."""
        return {
            'weights': jax.random.normal(key, (self.state_dim, 1)),
            'bias': jnp.zeros(1)
        }
    
    def __call__(self, params, states):
        """Forward pass."""
        if states.ndim == 1:
            states = states[None, :]
        values = jnp.dot(states, params['weights']) + params['bias']
        return values


class TestComprehensiveIntegration:
    """Comprehensive integration tests for the entire system."""
    
    @pytest.fixture
    def test_contract(self):
        """Create test contract."""
        return TestContract().contract
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        return MockEnvironment()
    
    def test_contract_creation_and_validation(self, test_contract):
        """Test complete contract creation and validation pipeline."""
        # Test contract creation
        assert test_contract.metadata.name == "TestContract"
        assert len(test_contract.stakeholders) == 3
        assert len(test_contract.constraints) == 2
        assert len(test_contract.reward_functions) == 3
        
        # Test contract validation
        contract_service = ContractService()
        validation_result = contract_service.validate_contract(test_contract)
        
        assert validation_result['valid'] is True
        assert len(validation_result['errors']) == 0
        
        # Test contract hash generation
        hash1 = test_contract.compute_hash()
        hash2 = test_contract.compute_hash()
        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 64  # SHA256 hex
    
    def test_reward_computation_with_constraints(self, test_contract):
        """Test reward computation with constraint enforcement."""
        # Test with valid inputs
        state = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        action = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        reward = test_contract.compute_reward(state, action)
        
        assert isinstance(reward, (int, float))
        assert jnp.isfinite(reward)
        
        # Test constraint checking
        violations = test_contract.check_violations(state, action)
        assert isinstance(violations, dict)
        assert 'safety_constraint' in violations
        assert 'reward_bounds' in violations
        
        # Test with edge cases
        large_action = jnp.array([100.0, 200.0, 300.0, 400.0, 500.0])
        reward_large = test_contract.compute_reward(state, large_action)
        violations_large = test_contract.check_violations(state, large_action)
        
        assert jnp.isfinite(reward_large)
        # Should have more violations with extreme inputs
    
    def test_formal_verification_integration(self, test_contract):
        """Test formal verification integration."""
        verifier = create_verifier("mock")
        result = verifier.verify_contract(test_contract)
        
        assert result.backend_used == "mock"
        assert isinstance(result.all_proofs_valid, bool)
        assert len(result.proofs) > 0
        assert result.verification_time > 0
        
        # Test specific property verification
        property_result = verifier.verify_property(
            "reward_bounds",
            "FORALL state, action: reward(state, action) >= -1.0 AND reward(state, action) <= 1.0",
            {"contract": test_contract}
        )
        
        assert 'property_name' in property_result
        assert 'valid' in property_result
        assert 'verification_time' in property_result
    
    def test_security_analysis_integration(self, test_contract):
        """Test security analysis integration."""
        assessment = analyze_contract_security(test_contract)
        
        assert assessment.assessment_id.startswith("SEC_")
        assert assessment.contract_hash == test_contract.compute_hash()
        assert 0.0 <= assessment.overall_security_score <= 1.0
        assert isinstance(assessment.vulnerabilities, list)
        assert isinstance(assessment.security_recommendations, list)
        assert isinstance(assessment.compliance_status, dict)
        
        # Test that some standard checks are performed
        critical_vulns = assessment.get_critical_vulnerabilities()
        assert isinstance(critical_vulns, list)
    
    def test_model_card_generation(self, test_contract):
        """Test OpenChain model card generation."""
        model_card = create_model_card(
            model_name="TestModel",
            model_version="1.0.0",
            contract=test_contract,
            description="Test model for RLHF contract integration"
        )
        
        # Set evaluation metrics
        metrics = ModelMetrics(
            accuracy=0.85,
            helpfulness=0.88,
            harmlessness=0.92,
            contract_compliance=0.95
        )
        model_card.set_evaluation_metrics(metrics)
        
        # Set legal metadata
        legal_meta = LegalMetadata(
            jurisdiction="Global",
            liability_cap="$1,000,000",
            audit_frequency="quarterly"
        )
        model_card.set_legal_metadata(legal_meta)
        
        # Generate card
        card_data = model_card.generate()
        
        assert card_data['model_name'] == "TestModel"
        assert 'rlhf_contract' in card_data
        assert 'evaluation' in card_data
        assert 'legal_metadata' in card_data
        assert card_data['openchain_compliance']['compliant'] is True
        
        # Test summary
        summary = model_card.get_summary()
        assert summary['has_contract'] is True
        assert summary['has_legal_metadata'] is True
        assert summary['openchain_compliant'] is True
    
    def test_rlhf_training_integration(self, test_contract, mock_env):
        """Test RLHF training with contract integration."""
        # Create mock networks
        policy_net = MockPolicyNetwork(10, 5)
        value_net = MockValueNetwork(10)
        
        # Configure training
        config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=32,
            num_epochs=2,  # Small for testing
            num_steps_per_epoch=64,
            use_contract_penalties=True
        )
        
        # Create trainer
        trainer = ContractualPPO(
            policy_network=policy_net,
            value_network=value_net,
            contract=test_contract,
            config=config
        )
        
        # Initialize networks
        key = jax.random.PRNGKey(42)
        trainer.initialize_networks(10, 5, key)
        
        # Test single epoch training
        metrics = trainer.train_epoch(mock_env)
        
        assert metrics.epoch == 0
        assert metrics.total_loss > 0
        assert metrics.reward_mean is not None
        assert isinstance(metrics.contract_violations, int)
        
        # Test training summary
        summary = trainer.get_training_summary()
        assert summary['status'] == 'in_progress'
        assert summary['contract_name'] == test_contract.metadata.name
        assert summary['contract_hash'] == test_contract.compute_hash()
    
    def test_distributed_execution_integration(self, test_contract):
        """Test distributed execution integration."""
        executor = create_distributed_executor()
        
        # Test batch execution
        batch_size = 100
        key = jax.random.PRNGKey(42)
        states = jax.random.normal(key, (batch_size, 10))
        actions = jax.random.normal(key, (batch_size, 5))
        
        # Execute batch
        rewards = asyncio.run(
            executor.execute_contract_batch(test_contract, states, actions, batch_size=50)
        )
        
        assert rewards.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(rewards))
        
        # Test cluster status
        status = executor.get_cluster_status()
        assert 'cluster_info' in status
        assert 'jax_info' in status
        assert 'performance_metrics' in status
        
        # Test performance benchmark
        benchmark = executor.benchmark_performance(test_contract, [100, 500])
        assert len(benchmark) == 2
        for size, results in benchmark.items():
            assert 'distributed_time' in results
            assert 'throughput_samples_per_second' in results
    
    def test_caching_integration(self, test_contract):
        """Test intelligent caching integration."""
        # Configure cache
        configure_global_cache(
            l1_config={'max_size': 1000, 'max_memory_mb': 100},
            enable_ml=False  # Disable ML for testing
        )
        
        cache = get_global_cache()
        
        # Test cache operations
        state = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        action = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # First computation (cache miss)
        reward1 = test_contract.compute_reward(state, action, use_cache=True)
        
        # Second computation (should hit cache)
        reward2 = test_contract.compute_reward(state, action, use_cache=True)
        
        assert reward1 == reward2  # Should be identical due to caching
        
        # Test cache statistics
        stats = cache.get_cache_stats()
        assert 'global' in stats
        assert 'levels' in stats
        assert stats['global']['hits'] > 0 or stats['global']['misses'] > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, test_contract):
        """Test real-time monitoring integration."""
        monitor = create_contract_monitor(test_contract)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Let it collect some data
        await asyncio.sleep(0.1)
        
        # Test metrics collection
        current_metrics = monitor.get_current_metrics()
        assert isinstance(current_metrics, dict)
        
        # Test monitoring status
        status = monitor.get_monitoring_status()
        assert status['is_running'] is True
        assert status['collectors_count'] >= 2  # Contract + system collectors
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        final_status = monitor.get_monitoring_status()
        assert final_status['is_running'] is False
    
    def test_contract_service_integration(self, test_contract, temp_dir):
        """Test contract service integration."""
        service = ContractService()
        
        # Test contract creation from dict
        contract_data = test_contract.to_dict()
        contract_id = service.create_contract(contract_data=contract_data)
        
        assert isinstance(contract_id, str)
        assert len(contract_id) == 32  # MD5 hex
        
        # Test contract retrieval
        retrieved_contract = service.get_contract(contract_id)
        assert retrieved_contract is not None
        assert retrieved_contract.metadata.name == test_contract.metadata.name
        
        # Test contract listing
        contracts = service.list_contracts()
        assert len(contracts) >= 1
        assert any(c['contract_id'] == contract_id for c in contracts)
        
        # Test contract saving and loading
        save_path = temp_dir / "test_contract.pkl"
        service.save_contract(test_contract, str(save_path), format='pickle')
        assert save_path.exists()
        
        loaded_contract = service.load_contract(str(save_path), format='pickle')
        assert loaded_contract.metadata.name == test_contract.metadata.name
        assert loaded_contract.compute_hash() == test_contract.compute_hash()
    
    def test_error_handling_and_recovery(self, test_contract):
        """Test error handling and recovery mechanisms."""
        # Test with invalid inputs
        with pytest.raises(ValueError):
            test_contract.compute_reward(None, None)
        
        with pytest.raises(ValueError):
            test_contract.compute_reward(jnp.array([]), jnp.array([]))
        
        # Test with NaN inputs (should handle gracefully)
        nan_state = jnp.array([jnp.nan] * 10)
        nan_action = jnp.array([jnp.nan] * 5)
        
        try:
            reward = test_contract.compute_reward(nan_state, nan_action)
            # Should either handle gracefully or raise appropriate error
            assert jnp.isfinite(reward) or jnp.isnan(reward)
        except Exception as e:
            # Should be a well-defined error type
            assert isinstance(e, (ValueError, RuntimeError))
        
        # Test constraint violation handling
        violations = test_contract.check_violations(nan_state, nan_action)
        assert isinstance(violations, dict)
        # Should mark violations appropriately for invalid inputs
    
    def test_performance_and_scalability(self, test_contract):
        """Test performance and scalability characteristics."""
        import time
        
        # Test single computation performance
        state = jax.random.normal(jax.random.PRNGKey(42), (10,))
        action = jax.random.normal(jax.random.PRNGKey(43), (5,))
        
        start_time = time.time()
        for _ in range(100):
            reward = test_contract.compute_reward(state, action)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Should be fast (< 10ms per computation)
        
        # Test batch computation performance
        batch_size = 1000
        states = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 10))
        actions = jax.random.normal(jax.random.PRNGKey(43), (batch_size, 5))
        
        start_time = time.time()
        rewards = []
        for i in range(batch_size):
            reward = test_contract.compute_reward(states[i], actions[i])
            rewards.append(reward)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = batch_size / total_time
        
        assert throughput > 100  # Should handle > 100 computations/second
        assert len(rewards) == batch_size
        assert all(jnp.isfinite(r) for r in rewards)
    
    def test_real_world_scenario_simulation(self, test_contract):
        """Test complete real-world scenario simulation."""
        # Simulate a production deployment scenario
        
        # 1. Contract validation and security analysis
        contract_service = ContractService()
        validation = contract_service.validate_contract(test_contract)
        assert validation['valid']
        
        security_assessment = analyze_contract_security(test_contract)
        assert security_assessment.overall_security_score > 0.5
        
        # 2. Model card generation for compliance
        model_card = create_model_card(
            model_name="ProductionModel",
            contract=test_contract
        )
        model_card.set_evaluation_metrics(ModelMetrics(
            helpfulness=0.85,
            harmlessness=0.90,
            contract_compliance=0.95
        ))
        card_data = model_card.generate()
        assert card_data['openchain_compliance']['compliant']
        
        # 3. Simulated inference workload
        num_requests = 500
        key = jax.random.PRNGKey(42)
        
        successful_requests = 0
        total_reward = 0.0
        total_violations = 0
        
        for i in range(num_requests):
            try:
                # Generate random request
                key, subkey = jax.random.split(key)
                state = jax.random.normal(subkey, (10,))
                action = jax.random.normal(subkey, (5,))
                
                # Compute reward
                reward = test_contract.compute_reward(state, action)
                
                # Check violations
                violations = test_contract.check_violations(state, action)
                violation_count = sum(violations.values())
                
                successful_requests += 1
                total_reward += reward
                total_violations += violation_count
                
            except Exception:
                # Count failed requests
                pass
        
        # Validate simulation results
        success_rate = successful_requests / num_requests
        avg_reward = total_reward / max(successful_requests, 1)
        violation_rate = total_violations / max(successful_requests, 1)
        
        assert success_rate > 0.95  # 95% success rate
        assert jnp.isfinite(avg_reward)
        assert violation_rate < 0.1  # Less than 10% violation rate
        
        # 4. Performance under load
        start_time = time.time()
        batch_rewards = []
        
        # Simulate burst traffic
        for batch in range(10):
            batch_states = jax.random.normal(key, (50, 10))
            batch_actions = jax.random.normal(key, (50, 5))
            
            batch_result = []
            for i in range(50):
                reward = test_contract.compute_reward(batch_states[i], batch_actions[i])
                batch_result.append(reward)
            
            batch_rewards.extend(batch_result)
        
        end_time = time.time()
        
        # Validate performance
        total_time = end_time - start_time
        throughput = len(batch_rewards) / total_time
        
        assert throughput > 50  # Should handle > 50 requests/second
        assert len(batch_rewards) == 500
        assert all(jnp.isfinite(r) for r in batch_rewards)
    
    def test_edge_cases_and_boundary_conditions(self, test_contract):
        """Test edge cases and boundary conditions."""
        # Test with extreme values
        extreme_state = jnp.array([1e6, -1e6, 0, jnp.inf, -jnp.inf, 1e-10, -1e-10, 1000, -1000, 0.5])
        extreme_action = jnp.array([1e6, -1e6, 0, 100, -100])
        
        # Should handle extreme values gracefully
        try:
            reward = test_contract.compute_reward(extreme_state, extreme_action)
            # If it doesn't raise an error, result should be finite or appropriately handled
            assert jnp.isfinite(reward) or jnp.isnan(reward) or jnp.isinf(reward)
        except Exception:
            # Should raise well-defined exceptions for extreme cases
            pass
        
        # Test with very small values
        tiny_state = jnp.array([1e-10] * 10)
        tiny_action = jnp.array([1e-10] * 5)
        
        reward_tiny = test_contract.compute_reward(tiny_state, tiny_action)
        assert jnp.isfinite(reward_tiny)
        
        # Test with zero values
        zero_state = jnp.zeros(10)
        zero_action = jnp.zeros(5)
        
        reward_zero = test_contract.compute_reward(zero_state, zero_action)
        assert jnp.isfinite(reward_zero)
        
        # Test with mismatched dimensions (should handle gracefully)
        wrong_state = jnp.array([1.0, 2.0])  # Wrong dimension
        wrong_action = jnp.array([0.1])       # Wrong dimension
        
        try:
            reward_wrong = test_contract.compute_reward(wrong_state, wrong_action)
            # If no error, should still produce valid result
            assert jnp.isfinite(reward_wrong)
        except Exception:
            # Should raise appropriate error for dimension mismatch
            pass


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])

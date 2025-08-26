"""
Comprehensive test suite for experimental reward architectures.

Tests all advanced research components including quantum-hybrid models,
neural architecture search, meta-adaptive learning, and performance
breakthrough analysis.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import asyncio
import time
from unittest.mock import patch, MagicMock

from src.research.experimental_reward_architectures import (
    ExperimentalRewardFramework,
    ExperimentalConfig,
    RewardArchitecture,
    QuantumHybridReward,
    NeuralArchitectureSearchReward,
    MetaAdaptiveReward,
    CausalRewardModel,
    run_experimental_protocol
)
from src.research.performance_breakthrough_analyzer import (
    PerformanceProfiler,
    BreakthroughAnalyzer,
    PerformanceMetric,
    BottleneckType,
    run_breakthrough_analysis
)
from src.research.adaptive_security_framework import (
    AdaptiveSecurityFramework,
    QuantumResistantCrypto,
    BehavioralAnalyzer,
    ThreatLevel,
    AttackVector,
    run_security_assessment
)
from src.models.reward_contract import RewardContract


class TestExperimentalArchitectures:
    """Test experimental reward architectures."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)
        self.test_state = random.normal(self.key, (10,))
        self.test_action = random.normal(self.key, (5,))
        
        self.config = ExperimentalConfig(
            architecture=RewardArchitecture.QUANTUM_HYBRID,
            quantum_depth=2,  # Reduced for testing
            meta_learning_rate=1e-3,
            regularization_strength=1e-4
        )
    
    def test_quantum_hybrid_reward_model(self):
        """Test quantum-hybrid reward model functionality."""
        model = QuantumHybridReward(quantum_depth=2, classical_hidden=64, quantum_params=8)
        
        # Initialize model
        params = model.init(self.key, self.test_state, self.test_action)
        
        # Test forward pass
        reward = model.apply(params, self.test_state, self.test_action)
        
        assert reward.shape == ()  # Scalar output
        assert jnp.isfinite(reward)
        assert -1.0 <= reward <= 1.0  # Bounded by tanh
    
    def test_neural_architecture_search_model(self):
        """Test neural architecture search reward model."""
        model = NeuralArchitectureSearchReward(search_space_size=100, max_layers=5)
        
        arch_key = random.normal(self.key, (20,))
        params = model.init(self.key, self.test_state, self.test_action, arch_key)
        
        reward = model.apply(params, self.test_state, self.test_action, arch_key)
        
        assert reward.shape == ()
        assert jnp.isfinite(reward)
        assert -1.0 <= reward <= 1.0
    
    def test_meta_adaptive_reward_model(self):
        """Test meta-adaptive reward model."""
        model = MetaAdaptiveReward(meta_hidden=64, adaptation_steps=3)
        
        context = random.normal(self.key, (8,))
        params = model.init(self.key, self.test_state, self.test_action, context)
        
        reward = model.apply(params, self.test_state, self.test_action, context)
        
        assert reward.shape == ()
        assert jnp.isfinite(reward)
        assert -1.0 <= reward <= 1.0
    
    def test_causal_reward_model(self):
        """Test causal inference reward model."""
        model = CausalRewardModel(causal_layers=2, intervention_dim=16)
        
        causal_graph = random.normal(self.key, (20,))
        params = model.init(self.key, self.test_state, self.test_action, causal_graph)
        
        reward = model.apply(params, self.test_state, self.test_action, causal_graph)
        
        assert reward.shape == ()
        assert jnp.isfinite(reward)
        assert -1.0 <= reward <= 1.0
    
    def test_experimental_framework_initialization(self):
        """Test experimental framework initialization."""
        framework = ExperimentalRewardFramework(self.config)
        
        assert len(framework.models) > 0
        assert len(framework.train_states) > 0
        assert self.config.architecture.value in framework.models
    
    def test_experimental_framework_training(self):
        """Test training of experimental architectures."""
        framework = ExperimentalRewardFramework(self.config)
        
        # Create synthetic training data
        n_samples = 50
        training_data = {
            'states': random.normal(self.key, (n_samples, 10)),
            'actions': random.normal(self.key, (n_samples, 5)),
            'rewards': random.uniform(self.key, (n_samples,))
        }
        
        # Train architecture (reduced epochs for testing)
        metrics = framework.train_architecture(
            self.config.architecture.value, 
            training_data, 
            num_epochs=5
        )
        
        assert 'final_loss' in metrics
        assert 'convergence_rate' in metrics
        assert metrics['final_loss'] >= 0
    
    def test_comparative_benchmark(self):
        """Test comparative benchmarking across architectures."""
        # Test multiple architectures
        architectures = [
            RewardArchitecture.QUANTUM_HYBRID,
            RewardArchitecture.META_ADAPTIVE
        ]
        
        results = {}
        for arch in architectures:
            config = ExperimentalConfig(architecture=arch)
            framework = ExperimentalRewardFramework(config)
            
            # Quick training
            training_data = {
                'states': random.normal(self.key, (30, 10)),
                'actions': random.normal(self.key, (30, 5)),
                'rewards': random.uniform(self.key, (30,))
            }
            framework.train_architecture(arch.value, training_data, num_epochs=3)
            
            results[arch.value] = framework
        
        # Test data for benchmarking
        test_data = {
            'states': random.normal(self.key, (20, 10)),
            'actions': random.normal(self.key, (20, 5)),
            'rewards': random.uniform(self.key, (20,))
        }
        
        # Run benchmark on one framework
        benchmark_results = list(results.values())[0].comparative_benchmark(test_data)
        
        assert isinstance(benchmark_results, dict)
        assert len(benchmark_results) > 0
        
        # Check benchmark metrics
        for arch_name, result in benchmark_results.items():
            if 'error' not in result:
                assert 'mse' in result
                assert 'r2_score' in result
                assert 'avg_inference_time' in result
    
    def test_research_report_generation(self):
        """Test research report generation."""
        framework = ExperimentalRewardFramework(self.config)
        
        # Quick training
        training_data = {
            'states': random.normal(self.key, (30, 10)),
            'actions': random.normal(self.key, (30, 5)),
            'rewards': random.uniform(self.key, (30,))
        }
        framework.train_architecture(self.config.architecture.value, training_data, num_epochs=3)
        
        # Test data
        test_data = {
            'states': random.normal(self.key, (20, 10)),
            'actions': random.normal(self.key, (20, 5)),
            'rewards': random.uniform(self.key, (20,))
        }
        
        benchmark_results = framework.comparative_benchmark(test_data)
        report = framework.generate_research_report(benchmark_results)
        
        assert 'experiment_config' in report
        assert 'performance_comparison' in report
        assert 'statistical_analysis' in report
        assert 'research_insights' in report
        assert 'publication_ready_metrics' in report
    
    @pytest.mark.asyncio
    async def test_experimental_protocol(self):
        """Test full experimental protocol execution."""
        # This test might take longer, so we use a simplified version
        with patch('src.research.experimental_reward_architectures.random.normal') as mock_normal:
            mock_normal.return_value = jnp.ones((10,))  # Simplified data
            
            # This should complete quickly with mocked data
            results = run_experimental_protocol()
            
            assert 'experimental_results' in results
            assert 'benchmark_results' in results  
            assert 'research_report' in results


class TestPerformanceBreakthroughAnalyzer:
    """Test performance breakthrough analysis components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.profiler = PerformanceProfiler(sampling_interval=0.01)  # Fast sampling for tests
        self.analyzer = BreakthroughAnalyzer(self.profiler)
    
    def test_profiler_initialization(self):
        """Test performance profiler initialization."""
        assert self.profiler.sampling_interval == 0.01
        assert not self.profiler.is_profiling
        assert len(self.profiler.snapshots) == 0
    
    def test_profiler_start_stop(self):
        """Test profiler start and stop functionality."""
        self.profiler.start_profiling()
        assert self.profiler.is_profiling
        
        time.sleep(0.1)  # Let it collect some data
        
        self.profiler.stop_profiling()
        assert not self.profiler.is_profiling
        assert len(self.profiler.snapshots) > 0
    
    def test_performance_snapshot_collection(self):
        """Test performance snapshot collection."""
        snapshot = self.profiler._capture_performance_snapshot()
        
        assert hasattr(snapshot, 'timestamp')
        assert hasattr(snapshot, 'metrics')
        assert hasattr(snapshot, 'system_info')
        
        # Check required metrics are present
        assert PerformanceMetric.CPU_UTILIZATION in snapshot.metrics
        assert PerformanceMetric.MEMORY_USAGE in snapshot.metrics
        assert PerformanceMetric.LATENCY in snapshot.metrics
    
    def test_bottleneck_prediction(self):
        """Test bottleneck prediction functionality."""
        # Generate some fake snapshots
        for _ in range(20):
            snapshot = self.profiler._capture_performance_snapshot()
            self.profiler.snapshots.append(snapshot)
            time.sleep(0.01)
        
        predictions = self.profiler.predict_bottlenecks(lookahead_seconds=60)
        
        assert isinstance(predictions, list)
        # Predictions may be empty if no bottlenecks detected
        for prediction in predictions:
            assert hasattr(prediction, 'bottleneck_type')
            assert hasattr(prediction, 'confidence')
            assert hasattr(prediction, 'suggested_optimizations')
            assert 0 <= prediction.confidence <= 1
    
    def test_baseline_establishment(self):
        """Test baseline establishment."""
        # Start profiling briefly
        self.profiler.start_profiling()
        time.sleep(0.1)
        
        # Establish baseline
        self.analyzer.establish_baseline(duration_seconds=1)
        
        assert self.analyzer.baseline_established
        assert len(self.profiler.baseline_metrics) > 0
        
        self.profiler.stop_profiling()
    
    def test_optimization_recommendations(self):
        """Test optimization recommendation generation."""
        # Mock baseline establishment
        self.analyzer.baseline_established = True
        self.profiler.baseline_metrics = {
            PerformanceMetric.LATENCY: {'mean': 50.0, 'std': 10.0}
        }
        
        # Add some fake snapshots with high latency
        for i in range(10):
            fake_snapshot = MagicMock()
            fake_snapshot.metrics = {
                PerformanceMetric.LATENCY: 100.0,  # High latency
                PerformanceMetric.CPU_UTILIZATION: 80.0,  # High CPU
                PerformanceMetric.CACHE_HIT_RATE: 0.5  # Low cache hit rate
            }
            self.profiler.snapshots.append(fake_snapshot)
        
        recommendations = self.analyzer.analyze_breakthrough_opportunities()
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert hasattr(rec, 'category')
            assert hasattr(rec, 'description')
            assert hasattr(rec, 'expected_improvement')
            assert hasattr(rec, 'implementation_difficulty')
            assert 0 <= rec.expected_improvement <= 1
            assert 1 <= rec.implementation_difficulty <= 5
    
    def test_optimization_plan_generation(self):
        """Test optimization plan generation."""
        # Create some mock recommendations
        from src.research.performance_breakthrough_analyzer import OptimizationRecommendation
        
        recommendations = [
            OptimizationRecommendation(
                category="Algorithmic",
                description="Test optimization",
                expected_improvement=0.3,
                implementation_difficulty=2,
                resource_requirements={'development_hours': 10, 'testing_hours': 5},
                code_changes_required=["Add JIT compilation"],
                testing_strategy=["Benchmark performance"]
            )
        ]
        
        plan = self.analyzer.generate_optimization_plan(recommendations)
        
        assert 'executive_summary' in plan
        assert 'implementation_phases' in plan
        assert 'detailed_recommendations' in plan
        assert 'risk_assessment' in plan
        
        assert plan['executive_summary']['total_recommendations'] == 1
    
    @pytest.mark.asyncio
    async def test_breakthrough_analysis_integration(self):
        """Test complete breakthrough analysis workflow."""
        # Create a test contract
        contract = RewardContract("test_contract", version="1.0.0")
        
        @contract.reward_function()
        def simple_reward(state, action):
            return jnp.mean(state * action[:len(state)])
        
        # Mock the analysis to avoid long execution
        with patch('src.research.performance_breakthrough_analyzer._generate_synthetic_load') as mock_load:
            mock_load.return_value = None  # Mock async function
            
            with patch.object(PerformanceProfiler, 'establish_baseline') as mock_baseline:
                mock_baseline.return_value = None
                
                results = await run_breakthrough_analysis(contract)
        
        # Should return results even with mocked components
        assert isinstance(results, dict)
        assert 'performance_trends' in results or 'baseline_metrics' in results


class TestAdaptiveSecurityFramework:
    """Test adaptive security framework components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.security_framework = AdaptiveSecurityFramework()
        self.crypto = QuantumResistantCrypto()
        self.behavioral_analyzer = BehavioralAnalyzer()
    
    def test_quantum_resistant_crypto_init(self):
        """Test quantum-resistant crypto initialization."""
        assert self.crypto.key_size == 3072
        assert len(self.crypto.hash_algorithms) > 0
    
    def test_quantum_safe_keypair_generation(self):
        """Test quantum-safe key pair generation."""
        private_key, public_key = self.crypto.generate_quantum_safe_keypair()
        
        assert isinstance(private_key, bytes)
        assert isinstance(public_key, bytes)
        assert len(private_key) > 0
        assert len(public_key) > 0
        
        # Keys should be different
        assert private_key != public_key
    
    def test_quantum_safe_signing(self):
        """Test quantum-safe digital signing."""
        private_key, public_key = self.crypto.generate_quantum_safe_keypair()
        message = b"Test message for signing"
        
        signature = self.crypto.quantum_safe_sign(message, private_key)
        
        assert isinstance(signature, bytes)
        assert len(signature) > 0
        
        # Verify signature
        is_valid = self.crypto.quantum_safe_verify(message, signature, public_key)
        assert is_valid
        
        # Invalid signature should fail
        invalid_signature = signature[:-1] + b'x'  # Corrupt signature
        is_valid = self.crypto.quantum_safe_verify(message, invalid_signature, public_key)
        assert not is_valid
    
    def test_behavioral_analyzer_initialization(self):
        """Test behavioral analyzer initialization."""
        assert self.behavioral_analyzer.anomaly_threshold == 2.5
        assert len(self.behavioral_analyzer.baseline_profiles) == 0
        assert len(self.behavioral_analyzer.feature_extractors) > 0
    
    def test_behavioral_baseline_learning(self):
        """Test behavioral baseline learning."""
        entity_id = "test_entity"
        
        # Generate fake observations
        observations = []
        for i in range(20):
            obs = {
                'request_count': 10 + i,
                'time_window': 60,
                'payload_size': 1000 + i * 10,
                'accessed_endpoints': ['endpoint1', 'endpoint2'],
                'timestamp': time.time() + i
            }
            observations.append(obs)
        
        self.behavioral_analyzer.learn_baseline_behavior(entity_id, observations)
        
        assert entity_id in self.behavioral_analyzer.baseline_profiles
        profile = self.behavioral_analyzer.baseline_profiles[entity_id]
        
        assert 'request_frequency' in profile
        assert 'payload_size' in profile
        assert profile['request_frequency']['mean'] > 0
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        entity_id = "test_entity"
        
        # Establish baseline first
        normal_observations = [
            {
                'request_count': 10,
                'time_window': 60,
                'payload_size': 1000,
                'accessed_endpoints': ['endpoint1'],
                'timestamp': time.time()
            } for _ in range(10)
        ]
        
        self.behavioral_analyzer.learn_baseline_behavior(entity_id, normal_observations)
        
        # Test normal observation
        normal_obs = {
            'request_count': 12,
            'time_window': 60,
            'payload_size': 1100,
            'accessed_endpoints': ['endpoint1'],
            'timestamp': time.time()
        }
        
        is_anomaly, score, anomalies = self.behavioral_analyzer.detect_anomaly(entity_id, normal_obs)
        assert not is_anomaly or score < 3.0  # Should be normal or low score
        
        # Test anomalous observation
        anomalous_obs = {
            'request_count': 1000,  # Much higher than baseline
            'time_window': 60,
            'payload_size': 100000,  # Much larger than baseline
            'accessed_endpoints': ['endpoint1', 'endpoint2', 'endpoint3', 'endpoint4'],
            'timestamp': time.time()
        }
        
        is_anomaly, score, anomalies = self.behavioral_analyzer.detect_anomaly(entity_id, anomalous_obs)
        # Should detect anomaly or high score
        assert is_anomaly or score > 1.0
    
    def test_security_framework_initialization(self):
        """Test security framework initialization."""
        assert len(self.security_framework.threat_signatures) > 0
        assert len(self.security_framework.security_policies) > 0
        assert not self.security_framework.is_monitoring
    
    def test_request_analysis(self):
        """Test security request analysis."""
        # Normal request
        normal_request = {
            'request_id': 'req_001',
            'client_id': 'client_001',
            'payload': 'normal data',
            'timestamp': time.time(),
            'request_rate': 5,
            'baseline_rate': 10
        }
        
        analysis = self.security_framework.analyze_request(normal_request)
        
        assert 'threats_detected' in analysis
        assert 'risk_score' in analysis
        assert 'recommended_action' in analysis
        assert 0 <= analysis['risk_score'] <= 1
        assert analysis['recommended_action'] in ['allow', 'monitor', 'challenge', 'block']
        
        # Suspicious request
        suspicious_request = {
            'request_id': 'req_002',
            'client_id': 'attacker_001',
            'payload': "'; DROP TABLE users; --",  # SQL injection attempt
            'timestamp': time.time(),
            'request_rate': 100,  # High rate
            'baseline_rate': 10
        }
        
        analysis = self.security_framework.analyze_request(suspicious_request)
        
        # Should detect threats
        assert len(analysis['threats_detected']) > 0 or analysis['risk_score'] > 0.5
    
    def test_secure_contract_execution(self):
        """Test secure contract execution."""
        # Create test contract
        contract = RewardContract("secure_test", version="1.0.0")
        
        @contract.reward_function()
        def test_reward(state, action):
            return jnp.sum(state * action[:len(state)])
        
        # Test execution
        state = jnp.array([1.0, 2.0, 3.0])
        action = jnp.array([0.5, -0.5, 0.2])
        
        execution_context = {
            'client_id': 'test_client',
            'expected_contract_hash': contract.compute_hash()
        }
        
        result = self.security_framework.secure_contract_execution(
            contract, state, action, execution_context
        )
        
        assert 'execution_id' in result
        assert 'security_analysis' in result
        assert 'execution_allowed' in result
        
        if result['execution_allowed']:
            assert 'reward' in result
            assert 'security_attestation' in result
    
    def test_security_monitoring(self):
        """Test security monitoring functionality."""
        # Start monitoring briefly
        self.security_framework.start_monitoring()
        assert self.security_framework.is_monitoring
        
        time.sleep(0.1)  # Brief monitoring
        
        self.security_framework.stop_monitoring()
        assert not self.security_framework.is_monitoring
    
    def test_security_report_generation(self):
        """Test security report generation."""
        # Add some fake incidents for testing
        from src.research.adaptive_security_framework import SecurityIncident
        
        fake_incident = SecurityIncident(
            incident_id="test_inc_001",
            timestamp=time.time(),
            threat_level=ThreatLevel.MEDIUM,
            attack_vector=AttackVector.INJECTION_ATTACK,
            affected_components=['test_component'],
            evidence={'test': 'data'},
            response_actions=['log_incident']
        )
        
        self.security_framework.active_incidents[fake_incident.incident_id] = fake_incident
        
        report = self.security_framework.generate_security_report()
        
        assert 'report_timestamp' in report
        assert 'executive_summary' in report
        assert 'threat_landscape' in report
        assert 'security_controls' in report
        assert 'performance_metrics' in report
        assert 'recommendations' in report
        
        assert report['executive_summary']['total_incidents'] >= 1
    
    @pytest.mark.asyncio
    async def test_security_assessment_integration(self):
        """Test complete security assessment workflow."""
        # Create test contract
        contract = RewardContract("security_test", version="1.0.0")
        
        @contract.reward_function()
        def test_reward(state, action):
            return jnp.mean(state + action[:len(state)])
        
        # Mock the assessment to avoid long execution
        with patch('time.sleep') as mock_sleep:
            mock_sleep.return_value = None
            
            results = await run_security_assessment(contract)
        
        assert isinstance(results, dict)
        assert 'attack_scenario_results' in results
        assert 'secure_execution_test' in results
        assert 'security_report' in results
        assert 'recommendations' in results


class TestIntegrationScenarios:
    """Test integration between research components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)
    
    def test_quantum_hybrid_with_security(self):
        """Test integration of quantum hybrid model with security framework."""
        # Create experimental framework
        config = ExperimentalConfig(architecture=RewardArchitecture.QUANTUM_HYBRID)
        exp_framework = ExperimentalRewardFramework(config)
        
        # Create security framework
        security_framework = AdaptiveSecurityFramework()
        
        # Create base contract
        contract = RewardContract("hybrid_secure_test", version="1.0.0")
        
        # Should integrate without errors
        assert len(exp_framework.models) > 0
        assert len(security_framework.threat_signatures) > 0
    
    def test_performance_analysis_with_scaling(self):
        """Test integration of performance analysis with scaling decisions."""
        # Create performance profiler
        profiler = PerformanceProfiler(sampling_interval=0.01)
        analyzer = BreakthroughAnalyzer(profiler)
        
        # Mock some performance data
        profiler.baseline_established = True
        profiler.baseline_metrics = {
            PerformanceMetric.CPU_UTILIZATION: {'mean': 0.5, 'std': 0.1}
        }
        
        # Generate optimization recommendations
        recommendations = analyzer.analyze_breakthrough_opportunities()
        
        # Should generate actionable recommendations
        assert isinstance(recommendations, list)
        
        if recommendations:
            plan = analyzer.generate_optimization_plan(recommendations)
            assert 'implementation_phases' in plan
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_pipeline(self):
        """Test complete end-to-end research pipeline."""
        # Create base contract
        contract = RewardContract("e2e_research_test", version="1.0.0")
        
        @contract.reward_function()
        def research_reward(state, action):
            return jnp.tanh(jnp.sum(state * action[:len(state)]))
        
        # Test experimental architectures (simplified)
        config = ExperimentalConfig(
            architecture=RewardArchitecture.QUANTUM_HYBRID,
            quantum_depth=2
        )
        exp_framework = ExperimentalRewardFramework(config)
        
        # Quick training with minimal data
        training_data = {
            'states': random.normal(self.key, (20, 10)),
            'actions': random.normal(self.key, (20, 5)),
            'rewards': random.uniform(self.key, (20,))
        }
        
        metrics = exp_framework.train_architecture(
            config.architecture.value,
            training_data,
            num_epochs=2
        )
        
        assert 'final_loss' in metrics
        
        # Test security analysis (simplified)
        security_framework = AdaptiveSecurityFramework()
        
        test_request = {
            'request_id': 'e2e_test',
            'client_id': 'research_client',
            'payload': 'test data',
            'timestamp': time.time()
        }
        
        security_analysis = security_framework.analyze_request(test_request)
        assert 'risk_score' in security_analysis
        
        # Test performance profiling (brief)
        profiler = PerformanceProfiler(sampling_interval=0.01)
        profiler.start_profiling()
        
        # Simulate some work
        time.sleep(0.05)
        
        profiler.stop_profiling()
        
        assert len(profiler.snapshots) > 0
        
        print("End-to-end research pipeline test completed successfully")


# Performance and stress tests
class TestPerformanceScaling:
    """Test performance characteristics and scaling behavior."""
    
    def test_large_scale_quantum_hybrid_inference(self):
        """Test quantum hybrid model with larger inputs."""
        model = QuantumHybridReward(quantum_depth=4, classical_hidden=128)
        key = random.PRNGKey(42)
        
        # Larger state and action spaces
        large_state = random.normal(key, (50,))
        large_action = random.normal(key, (20,))
        
        params = model.init(key, large_state, large_action)
        
        start_time = time.time()
        reward = model.apply(params, large_state, large_action)
        inference_time = time.time() - start_time
        
        assert jnp.isfinite(reward)
        assert inference_time < 1.0  # Should be reasonably fast
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        model = QuantumHybridReward(quantum_depth=2, classical_hidden=64)
        key = random.PRNGKey(42)
        
        batch_size = 32
        states = random.normal(key, (batch_size, 10))
        actions = random.normal(key, (batch_size, 5))
        
        # Initialize with single example
        params = model.init(key, states[0], actions[0])
        
        # Vectorize model application
        batch_apply = jax.vmap(lambda s, a: model.apply(params, s, a))
        
        start_time = time.time()
        rewards = batch_apply(states, actions)
        batch_time = time.time() - start_time
        
        assert rewards.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(rewards))
        print(f"Batch processing time for {batch_size} examples: {batch_time:.4f}s")
    
    def test_security_framework_stress(self):
        """Test security framework under high load."""
        security_framework = AdaptiveSecurityFramework()
        
        # Generate many requests
        num_requests = 100
        requests = []
        
        for i in range(num_requests):
            request = {
                'request_id': f'stress_test_{i}',
                'client_id': f'client_{i % 10}',  # 10 different clients
                'payload': f'test payload {i}',
                'timestamp': time.time() + i * 0.001,
                'request_rate': 10 + (i % 20),
                'baseline_rate': 10
            }
            requests.append(request)
        
        start_time = time.time()
        results = []
        
        for request in requests:
            analysis = security_framework.analyze_request(request)
            results.append(analysis)
        
        processing_time = time.time() - start_time
        
        assert len(results) == num_requests
        print(f"Processed {num_requests} security requests in {processing_time:.4f}s")
        print(f"Average processing time: {processing_time/num_requests*1000:.2f}ms per request")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
#!/usr/bin/env python3
"""
Generation 2 Demo: MAKE IT ROBUST (Reliable)

Demonstrates comprehensive error handling, security validation, health monitoring,
and reliability features added in Generation 2 of RLHF-Contract-Wizard.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List
import jax.numpy as jnp
import numpy as np

from .models.reward_contract import RewardContract, AggregationStrategy
from .reliability.advanced_error_recovery import (
    AdvancedErrorRecovery, get_error_recovery, robust_execute, 
    ErrorCategory, CircuitBreakerConfig
)
from .security.comprehensive_security_validation import (
    ComprehensiveSecurityValidator, get_security_validator,
    SecurityLevel, ThreatType
)
from .monitoring.advanced_health_monitoring import (
    AdvancedHealthMonitor, get_health_monitor, quick_health_check
)
from .enhanced_contract_runtime import EnhancedContractRuntime, RuntimeConfig


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_error_recovery():
    """Demonstrate advanced error recovery capabilities."""
    logger.info("🛡️ Starting Error Recovery Demo")
    
    recovery = get_error_recovery()
    
    # Register circuit breakers for different operations
    contract_breaker = recovery.register_circuit_breaker(
        "contract_execution",
        CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0)
    )
    
    computation_breaker = recovery.register_circuit_breaker(
        "computation",
        CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60.0)
    )
    
    logger.info(f"📊 Circuit breakers registered: {len(recovery.circuit_breakers)}")
    
    # Test 1: Successful operation with recovery
    async def successful_operation():
        await asyncio.sleep(0.1)  # Simulate work
        return "success"
    
    result = await recovery.execute_with_recovery(
        operation=successful_operation,
        operation_name="test_successful_operation",
        error_category=ErrorCategory.COMPUTATION
    )
    
    logger.info(f"✅ Successful operation result: {result}")
    
    # Test 2: Operation that fails initially but recovers
    attempt_count = 0
    
    async def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= 2:
            raise RuntimeError(f"Simulated failure (attempt {attempt_count})")
        return f"recovered_after_{attempt_count}_attempts"
    
    try:
        result = await recovery.execute_with_recovery(
            operation=flaky_operation,
            operation_name="test_flaky_operation",
            error_category=ErrorCategory.NETWORK,
            circuit_breaker_name="computation"
        )
        logger.info(f"🔄 Flaky operation recovered: {result}")
    except Exception as e:
        logger.error(f"❌ Flaky operation failed: {e}")
    
    # Test 3: Operation that fails and triggers circuit breaker
    async def always_failing_operation():
        raise ValueError("This operation always fails")
    
    # Try multiple times to trigger circuit breaker
    for i in range(8):
        try:
            await recovery.execute_with_recovery(
                operation=always_failing_operation,
                operation_name=f"always_failing_operation_{i}",
                error_category=ErrorCategory.COMPUTATION,
                circuit_breaker_name="computation"
            )
        except Exception as e:
            if i < 3:
                logger.info(f"🔄 Expected failure {i+1}: {e}")
            elif i == 6:
                logger.info("⚡ Circuit breaker should be OPEN now")
    
    # Test 4: Fallback mechanism
    async def main_operation():
        raise ConnectionError("Primary service unavailable")
    
    async def fallback_operation():
        await asyncio.sleep(0.1)
        return "fallback_result"
    
    result = await recovery.execute_with_recovery(
        operation=main_operation,
        operation_name="test_fallback_operation",
        error_category=ErrorCategory.NETWORK,
        fallback=fallback_operation
    )
    
    logger.info(f"🔀 Fallback operation result: {result}")
    
    # Get error analytics
    analytics = recovery.get_error_analytics()
    logger.info("📈 Error Analytics:")
    logger.info(f"   Total errors: {analytics.get('total_errors', 0)}")
    logger.info(f"   System health: {analytics.get('system_health', 0):.2%}")
    logger.info(f"   Top error patterns: {analytics.get('top_error_patterns', [])[:3]}")
    
    return recovery


async def demo_security_validation():
    """Demonstrate comprehensive security validation."""
    logger.info("🔒 Starting Security Validation Demo")
    
    validator = get_security_validator(SecurityLevel.HIGH)
    
    # Test 1: Safe string input
    safe_input = "Hello, world! This is a normal string."
    result = await validator.validate_input(
        input_data=safe_input,
        operation="test_safe_string",
        source_ip="192.168.1.100",
        user_id="test_user"
    )
    
    logger.info(f"✅ Safe string validation: {'PASSED' if result.passed else 'FAILED'}")
    logger.info(f"   Threat score: {result.threat_score:.2f}")
    
    # Test 2: Potentially malicious string input
    malicious_input = "'; DROP TABLE users; --"
    result = await validator.validate_input(
        input_data=malicious_input,
        operation="test_malicious_string",
        source_ip="192.168.1.200",
        user_id="suspicious_user"
    )
    
    logger.info(f"🚨 Malicious string validation: {'PASSED' if result.passed else 'BLOCKED'}")
    logger.info(f"   Violations: {result.violations}")
    logger.info(f"   Threat score: {result.threat_score:.2f}")
    
    # Test 3: Array validation
    safe_array = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = await validator.validate_input(
        input_data=safe_array,
        operation="test_safe_array",
        source_ip="192.168.1.100"
    )
    
    logger.info(f"✅ Safe array validation: {'PASSED' if result.passed else 'FAILED'}")
    
    # Test 4: Problematic array with NaN values
    problematic_array = jnp.array([1.0, float('nan'), 3.0, float('inf')])
    result = await validator.validate_input(
        input_data=problematic_array,
        operation="test_problematic_array",
        source_ip="192.168.1.200"
    )
    
    logger.info(f"🚨 Problematic array validation: {'PASSED' if result.passed else 'BLOCKED'}")
    logger.info(f"   Violations: {result.violations}")
    
    # Test 5: Dictionary validation
    complex_dict = {
        "safe_key": "safe_value",
        "nested": {
            "level2": {
                "level3": "deep_value"
            }
        },
        "array_data": [1, 2, 3, 4, 5]
    }
    
    result = await validator.validate_input(
        input_data=complex_dict,
        operation="test_complex_dict",
        source_ip="192.168.1.100"
    )
    
    logger.info(f"✅ Complex dict validation: {'PASSED' if result.passed else 'FAILED'}")
    
    # Test 6: Data protection (encryption/signing)
    sensitive_data = {
        "user_id": "user123",
        "reward_score": 0.85,
        "preferences": ["privacy", "security"]
    }
    
    protected_data = validator.protect_sensitive_data(sensitive_data)
    logger.info("🔐 Data protected with encryption and signing")
    
    # Test unprotection
    unprotected_data = validator.unprotect_sensitive_data(protected_data)
    logger.info(f"🔓 Data unprotected successfully: {unprotected_data == sensitive_data}")
    
    # Generate security dashboard
    dashboard = validator.get_security_dashboard()
    logger.info("🏠 Security Dashboard:")
    logger.info(f"   Security level: {dashboard['security_level']}")
    logger.info(f"   Recent incidents: {dashboard['threat_monitoring']['recent_events']}")
    logger.info(f"   Blocked IPs: {dashboard['system_health']['blocked_ips']}")
    logger.info(f"   Recommendations: {dashboard['recommendations'][:2]}")
    
    return validator


async def demo_health_monitoring():
    """Demonstrate advanced health monitoring."""
    logger.info("❤️ Starting Health Monitoring Demo")
    
    monitor = get_health_monitor()
    
    # Add custom alert callback
    async def custom_alert_handler(alert):
        logger.warning(f"🚨 CUSTOM ALERT: {alert.message} (Severity: {alert.severity.value})")
    
    monitor.add_alert_callback(custom_alert_handler)
    
    # Register custom health checks
    def custom_contract_health():
        """Custom health check for contract system."""
        # Simulate checking contract system health
        return {
            "status": "healthy",
            "active_contracts": 15,
            "average_execution_time": 0.05,
            "cache_hit_rate": 0.89
        }
    
    async def custom_quantum_health():
        """Custom health check for quantum systems."""
        # Simulate quantum system health check
        await asyncio.sleep(0.1)  # Simulate async check
        return {
            "status": "healthy",
            "quantum_coherence": 0.95,
            "entanglement_stability": 0.87,
            "superposition_integrity": 0.92
        }
    
    monitor.register_custom_check("contract_system", custom_contract_health)
    monitor.register_custom_check("quantum_systems", custom_quantum_health)
    
    logger.info(f"📋 Custom health checks registered: {len(monitor.custom_checks)}")
    
    # Perform initial health check
    health_data = await monitor.perform_health_check()
    logger.info("🔍 Initial Health Check Results:")
    logger.info(f"   Overall status: {health_data['overall_status']}")
    logger.info(f"   Components monitored: {len(health_data['components'])}")
    
    # Simulate some application activity for monitoring
    for i in range(10):
        response_time = 0.05 + (i * 0.01)  # Gradually increasing response time
        success = i < 8  # Simulate 2 failures out of 10
        monitor.record_request(response_time, success)
    
    # Perform another health check to see the impact
    health_data = await monitor.perform_health_check()
    app_metrics = health_data['components']['application']['metrics']
    
    logger.info("📊 Application Metrics After Activity:")
    logger.info(f"   Error rate: {app_metrics['error_rate']['value']:.1f}%")
    logger.info(f"   Average response time: {app_metrics['avg_response_time']['value']:.1f}ms")
    
    # Start continuous monitoring for a short period
    logger.info("🔄 Starting continuous monitoring...")
    await monitor.start_monitoring()
    await asyncio.sleep(5)  # Monitor for 5 seconds
    await monitor.stop_monitoring()
    logger.info("⏹️ Stopped continuous monitoring")
    
    # Run comprehensive diagnostic
    logger.info("🧪 Running system diagnostic...")
    diagnostic_results = await monitor.run_diagnostic()
    
    logger.info("🧪 Diagnostic Results:")
    logger.info(f"   Total tests: {diagnostic_results['summary']['total_tests']}")
    logger.info(f"   Passed: {diagnostic_results['summary']['passed']}")
    logger.info(f"   Failed: {diagnostic_results['summary']['failed']}")
    
    # Generate health dashboard
    dashboard = monitor.get_health_dashboard()
    logger.info("🏠 Health Dashboard Summary:")
    logger.info(f"   Overall status: {dashboard['overall_status']}")
    logger.info(f"   Active alerts: {dashboard['alerts']['summary']['active']}")
    logger.info(f"   Components: {len(dashboard['components'])}")
    
    return monitor


async def demo_integrated_robust_contract():
    """Demonstrate robust contract execution with all Generation 2 features."""
    logger.info("🚀 Starting Integrated Robust Contract Demo")
    
    # Initialize all systems
    recovery = get_error_recovery()
    validator = get_security_validator(SecurityLevel.HIGH)
    monitor = get_health_monitor()
    
    # Configure enhanced runtime
    config = RuntimeConfig(
        enable_caching=True,
        max_concurrent_contracts=5,
        timeout_seconds=15.0,
        enable_global_compliance=True,
        performance_monitoring=True,
        auto_recovery=True
    )
    
    runtime = EnhancedContractRuntime(config)
    
    # Create a robust contract
    contract = RewardContract(
        name="RobustAssistant-v2",
        version="2.0.0",
        stakeholders={
            "operator": 0.35,
            "safety_board": 0.35,
            "users": 0.30
        },
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE
    )
    
    # Add robust reward functions with error handling
    @contract.reward_function("operator")
    def robust_operator_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        try:
            # Validate inputs
            if not jnp.all(jnp.isfinite(state)) or not jnp.all(jnp.isfinite(action)):
                return 0.0
            
            efficiency = jnp.clip(jnp.mean(action) * 0.8, 0.0, 1.0)
            return float(efficiency)
        except Exception:
            return 0.0  # Safe fallback
    
    @contract.reward_function("safety_board")
    def robust_safety_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        try:
            # Enhanced safety checks
            if jnp.any(jnp.abs(action) > 1.0):
                return -0.5  # Penalty for extreme actions
            
            safety_score = 1.0 - jnp.max(jnp.abs(action - 0.5)) * 2.0
            return float(jnp.clip(safety_score, 0.0, 1.0))
        except Exception:
            return 0.0
    
    @contract.reward_function("users")
    def robust_user_reward(state: jnp.ndarray, action: jnp.ndarray) -> float:
        try:
            satisfaction = jnp.sum(state * action) / (jnp.sum(jnp.abs(state)) + 1e-8)
            return float(jnp.clip(satisfaction, 0.0, 1.0))
        except Exception:
            return 0.0
    
    # Add comprehensive constraints
    contract.add_constraint(
        name="input_validation",
        constraint_fn=lambda state, action: bool(
            jnp.all(jnp.isfinite(state)) and 
            jnp.all(jnp.isfinite(action)) and
            len(state) > 0 and len(action) > 0
        ),
        description="All inputs must be finite and non-empty",
        violation_penalty=-1.0
    )
    
    contract.add_constraint(
        name="action_bounds",
        constraint_fn=lambda state, action: bool(jnp.all(jnp.abs(action) <= 1.0)),
        description="Actions must be within [-1, 1] range",
        violation_penalty=-0.5
    )
    
    contract.add_constraint(
        name="state_consistency",
        constraint_fn=lambda state, action: bool(
            jnp.all(state >= 0.0) and jnp.all(state <= 1.0)
        ),
        description="State values must be normalized",
        violation_penalty=-0.3
    )
    
    # Register contract
    contract_id = runtime.register_contract(contract)
    logger.info(f"📋 Robust contract registered: {contract.metadata.name}")
    
    # Test data with various security and robustness challenges
    test_cases = [
        {
            "name": "Normal Case",
            "state": jnp.array([0.3, 0.7, 0.5, 0.2, 0.8]),
            "action": jnp.array([0.4, -0.2, 0.6]),
            "should_pass": True
        },
        {
            "name": "Edge Case - Zero State",
            "state": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "action": jnp.array([0.1, 0.1, 0.1]),
            "should_pass": True
        },
        {
            "name": "Boundary Case - Max Values",
            "state": jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "action": jnp.array([1.0, -1.0, 0.0]),
            "should_pass": True
        },
        {
            "name": "Invalid Case - NaN Values",
            "state": jnp.array([0.5, float('nan'), 0.3, 0.7, 0.1]),
            "action": jnp.array([0.2, 0.3, 0.4]),
            "should_pass": False
        },
        {
            "name": "Invalid Case - Out of Bounds",
            "state": jnp.array([0.5, 0.3, 0.7, 0.2, 0.8]),
            "action": jnp.array([2.0, -3.0, 1.5]),  # Out of bounds
            "should_pass": False
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        logger.info(f"🧪 Testing: {test_case['name']}")
        
        try:
            # Security validation first
            state_validation = await validator.validate_input(
                input_data=test_case['state'],
                operation=f"test_state_{test_case['name']}",
                source_ip="127.0.0.1",
                user_id="test_user"
            )
            
            action_validation = await validator.validate_input(
                input_data=test_case['action'],
                operation=f"test_action_{test_case['name']}",
                source_ip="127.0.0.1",
                user_id="test_user"
            )
            
            # Execute with robust error handling
            if state_validation.passed and action_validation.passed:
                execution_result = await robust_execute(
                    operation=lambda: runtime.execute_contract(
                        contract_id=contract_id,
                        state=test_case['state'],
                        action=test_case['action'],
                        context={
                            "test_case": test_case['name'],
                            "jurisdiction": "global",
                            "security_level": "high"
                        }
                    ),
                    operation_name=f"contract_execution_{test_case['name']}",
                    error_category=ErrorCategory.CONTRACT
                )
                
                result = {
                    "test_case": test_case['name'],
                    "status": "success",
                    "reward": execution_result.reward,
                    "execution_time": execution_result.execution_time,
                    "compliance_score": execution_result.compliance_score,
                    "violations": execution_result.violations
                }
                
                logger.info(f"   ✅ Reward: {result['reward']:.4f}")
                logger.info(f"   ⏱️ Time: {result['execution_time']:.4f}s")
                logger.info(f"   📊 Compliance: {result['compliance_score']:.4f}")
                
            else:
                result = {
                    "test_case": test_case['name'],
                    "status": "blocked_by_security",
                    "state_violations": state_validation.violations,
                    "action_violations": action_validation.violations
                }
                
                logger.info(f"   🚨 Blocked by security validation")
                logger.info(f"   State violations: {len(state_validation.violations)}")
                logger.info(f"   Action violations: {len(action_validation.violations)}")
            
        except Exception as e:
            result = {
                "test_case": test_case['name'],
                "status": "error",
                "error": str(e)
            }
            
            logger.error(f"   ❌ Error: {e}")
        
        results.append(result)
        
        # Record request for monitoring
        response_time = result.get('execution_time', 0.0)
        success = result['status'] == 'success'
        monitor.record_request(response_time, success)
    
    # Final health check
    logger.info("🏥 Final Health Check...")
    final_health = await monitor.perform_health_check()
    
    # Performance summary
    performance = runtime.get_performance_summary()
    logger.info("📊 Final Performance Summary:")
    logger.info(f"   Total executions: {performance.get('total_executions', 0)}")
    logger.info(f"   Average execution time: {performance.get('average_execution_time', 0):.4f}s")
    logger.info(f"   Average compliance score: {performance.get('average_compliance_score', 0):.4f}")
    
    # Error recovery summary
    error_analytics = recovery.get_error_analytics()
    logger.info("🛡️ Error Recovery Summary:")
    logger.info(f"   System health: {error_analytics.get('system_health', 0):.2%}")
    logger.info(f"   Total errors handled: {error_analytics.get('total_errors', 0)}")
    
    # Security summary
    security_dashboard = validator.get_security_dashboard()
    logger.info("🔒 Security Summary:")
    logger.info(f"   Recent incidents: {security_dashboard['threat_monitoring']['recent_events']}")
    logger.info(f"   System health: Critical threats: {security_dashboard['system_health']['critical_threats']}")
    
    return {
        "test_results": results,
        "performance": performance,
        "health": final_health,
        "security": security_dashboard,
        "error_recovery": error_analytics
    }


def demo_generation2_features():
    """Demonstrate all Generation 2 features together."""
    logger.info("🏆 Starting Generation 2 Complete Demo")
    logger.info("=" * 70)
    
    async def run_all_demos():
        # Feature 1: Advanced Error Recovery
        logger.info("FEATURE 1: Advanced Error Recovery")
        await demo_error_recovery()
        logger.info("")
        
        # Feature 2: Comprehensive Security Validation
        logger.info("FEATURE 2: Comprehensive Security Validation")
        await demo_security_validation()
        logger.info("")
        
        # Feature 3: Advanced Health Monitoring
        logger.info("FEATURE 3: Advanced Health Monitoring")
        await demo_health_monitoring()
        logger.info("")
        
        # Feature 4: Integrated Robust Contract System
        logger.info("FEATURE 4: Integrated Robust Contract System")
        await demo_integrated_robust_contract()
        logger.info("")
        
        logger.info("🏆 Generation 2 Demo Completed Successfully!")
        logger.info("🚀 System is now ROBUST and ready for scaling!")
        logger.info("=" * 70)
    
    # Run all demonstrations
    asyncio.run(run_all_demos())


if __name__ == "__main__":
    demo_generation2_features()
"""
Enhanced contract runtime with Generation 1 improvements.

Provides immediate functionality improvements for reward contract execution,
including better error handling, performance optimization, and global compliance.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

from .models.reward_contract import RewardContract, AggregationStrategy
from .optimization.contract_cache import reward_cache
from .global_compliance.regulatory_compliance import ComplianceChecker
from .utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


@dataclass
class RuntimeConfig:
    """Configuration for enhanced contract runtime."""
    enable_caching: bool = True
    max_concurrent_contracts: int = 10
    timeout_seconds: float = 30.0
    enable_global_compliance: bool = True
    performance_monitoring: bool = True
    auto_recovery: bool = True


@dataclass
class ExecutionResult:
    """Result of contract execution."""
    reward: float
    execution_time: float
    violations: Dict[str, bool]
    compliance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class EnhancedContractRuntime:
    """
    Enhanced runtime for reward contract execution with Generation 1 improvements.
    
    Features:
    - Async execution with concurrent processing
    - Advanced error handling and recovery
    - Performance monitoring and optimization
    - Global compliance checking
    - Intelligent caching strategies
    """
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.active_contracts: Dict[str, RewardContract] = {}
        self.execution_history: List[ExecutionResult] = []
        self.compliance_checker = ComplianceChecker() if self.config.enable_global_compliance else None
        self.performance_metrics: Dict[str, List[float]] = {
            "execution_times": [],
            "reward_values": [],
            "compliance_scores": []
        }
        self.logger = logging.getLogger(__name__)
    
    def register_contract(self, contract: RewardContract) -> str:
        """Register a contract for execution."""
        contract_id = contract.compute_hash()
        self.active_contracts[contract_id] = contract
        
        self.logger.info(f"Contract registered: {contract.metadata.name} ({contract_id[:8]})")
        
        # Warm up cache for better performance
        if self.config.enable_caching:
            self._warm_contract_cache(contract)
        
        return contract_id
    
    async def execute_contract(
        self,
        contract_id: str,
        state: jnp.ndarray,
        action: jnp.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a registered contract asynchronously."""
        start_time = time.time()
        
        try:
            if contract_id not in self.active_contracts:
                raise ValueError(f"Contract {contract_id} not registered")
            
            contract = self.active_contracts[contract_id]
            
            # Global compliance check
            compliance_score = 1.0
            if self.compliance_checker:
                compliance_score = await self._check_compliance(contract, state, action, context)
            
            # Execute with timeout protection
            reward = await asyncio.wait_for(
                self._compute_reward_async(contract, state, action, context),
                timeout=self.config.timeout_seconds
            )
            
            # Check violations
            violations = contract.check_violations(state, action)
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            
            result = ExecutionResult(
                reward=reward,
                execution_time=execution_time,
                violations=violations,
                compliance_score=compliance_score,
                metadata={
                    "contract_name": contract.metadata.name,
                    "contract_version": contract.metadata.version,
                    "stakeholders": len(contract.stakeholders),
                    "constraints": len(contract.constraints),
                    "cached": self.config.enable_caching
                }
            )
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Store execution history
            self.execution_history.append(result)
            if len(self.execution_history) > 1000:  # Keep last 1000 executions
                self.execution_history = self.execution_history[-1000:]
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Contract execution timeout after {self.config.timeout_seconds}s"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                reward=0.0,
                execution_time=self.config.timeout_seconds,
                violations={},
                compliance_score=0.0,
                error=error_msg
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Contract execution failed: {str(e)}"
            
            handle_error(
                error=e,
                operation="execute_contract",
                category=ErrorCategory.CONTRACT,
                severity=ErrorSeverity.HIGH,
                additional_info={
                    "contract_id": contract_id,
                    "execution_time": execution_time
                }
            )
            
            return ExecutionResult(
                reward=0.0,
                execution_time=execution_time,
                violations={},
                compliance_score=0.0,
                error=error_msg
            )
    
    async def batch_execute(
        self,
        contract_requests: List[Dict[str, Any]]
    ) -> List[ExecutionResult]:
        """Execute multiple contracts concurrently."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_contracts)
        
        async def execute_single(request):
            async with semaphore:
                return await self.execute_contract(
                    contract_id=request["contract_id"],
                    state=request["state"],
                    action=request["action"],
                    context=request.get("context")
                )
        
        tasks = [execute_single(req) for req in contract_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ExecutionResult(
                    reward=0.0,
                    execution_time=0.0,
                    violations={},
                    compliance_score=0.0,
                    error=f"Batch execution error: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _compute_reward_async(
        self,
        contract: RewardContract,
        state: jnp.ndarray,
        action: jnp.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Async wrapper for reward computation."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool for CPU-intensive computation
        reward = await loop.run_in_executor(
            None,
            lambda: contract.compute_reward(state, action, self.config.enable_caching, context)
        )
        
        return reward
    
    async def _check_compliance(
        self,
        contract: RewardContract,
        state: jnp.ndarray,
        action: jnp.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Check global compliance asynchronously."""
        if not self.compliance_checker:
            return 1.0
        
        try:
            # Extract jurisdiction from contract or context
            jurisdiction = contract.metadata.jurisdiction
            if context and "jurisdiction" in context:
                jurisdiction = context["jurisdiction"]
            
            compliance_result = await self.compliance_checker.check_compliance_async(
                contract=contract,
                state=state,
                action=action,
                jurisdiction=jurisdiction,
                context=context
            )
            
            return compliance_result.score
            
        except Exception as e:
            self.logger.warning(f"Compliance check failed: {e}")
            return 0.5  # Partial compliance on error
    
    def _warm_contract_cache(self, contract: RewardContract):
        """Pre-warm cache with common patterns."""
        try:
            # Generate common state/action patterns for caching
            dummy_states = [
                jnp.ones((10,)),
                jnp.zeros((10,)),
                jnp.array([0.5] * 10)
            ]
            
            dummy_actions = [
                jnp.ones((5,)),
                jnp.zeros((5,)),
                jnp.array([0.5] * 5)
            ]
            
            for state in dummy_states:
                for action in dummy_actions:
                    try:
                        contract.compute_reward(state, action, use_cache=True)
                    except:
                        continue  # Skip invalid combinations
                        
        except Exception as e:
            self.logger.warning(f"Cache warming failed: {e}")
    
    def _update_performance_metrics(self, result: ExecutionResult):
        """Update performance metrics for monitoring."""
        if not self.config.performance_monitoring:
            return
        
        self.performance_metrics["execution_times"].append(result.execution_time)
        self.performance_metrics["reward_values"].append(result.reward)
        self.performance_metrics["compliance_scores"].append(result.compliance_score)
        
        # Keep only last 1000 metrics
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 1000:
                self.performance_metrics[key] = self.performance_metrics[key][-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.performance_metrics["execution_times"]:
            return {"status": "no_data"}
        
        exec_times = self.performance_metrics["execution_times"]
        rewards = self.performance_metrics["reward_values"]
        compliance = self.performance_metrics["compliance_scores"]
        
        return {
            "total_executions": len(exec_times),
            "average_execution_time": sum(exec_times) / len(exec_times),
            "p95_execution_time": sorted(exec_times)[int(len(exec_times) * 0.95)] if exec_times else 0,
            "average_reward": sum(rewards) / len(rewards) if rewards else 0,
            "average_compliance_score": sum(compliance) / len(compliance) if compliance else 0,
            "active_contracts": len(self.active_contracts),
            "cache_hit_rate": reward_cache.hit_rate() if hasattr(reward_cache, 'hit_rate') else 0.0
        }
    
    def optimize_runtime(self) -> Dict[str, Any]:
        """Optimize runtime performance based on metrics."""
        performance = self.get_performance_summary()
        optimizations_applied = []
        
        # Adjust cache size based on hit rate
        if hasattr(reward_cache, 'hit_rate') and reward_cache.hit_rate() < 0.7:
            reward_cache.resize(min(20000, reward_cache.max_size * 2))
            optimizations_applied.append("increased_cache_size")
        
        # Adjust concurrent execution limit based on performance
        avg_time = performance.get("average_execution_time", 0)
        if avg_time > 1.0 and self.config.max_concurrent_contracts > 5:
            self.config.max_concurrent_contracts = max(5, self.config.max_concurrent_contracts - 2)
            optimizations_applied.append("reduced_concurrency")
        elif avg_time < 0.1 and self.config.max_concurrent_contracts < 20:
            self.config.max_concurrent_contracts = min(20, self.config.max_concurrent_contracts + 2)
            optimizations_applied.append("increased_concurrency")
        
        return {
            "optimizations_applied": optimizations_applied,
            "new_performance_target": {
                "max_concurrent_contracts": self.config.max_concurrent_contracts,
                "cache_size": getattr(reward_cache, 'max_size', 10000)
            }
        }
    
    @asynccontextmanager
    async def execution_context(self, contract_id: str):
        """Context manager for contract execution with cleanup."""
        start_time = time.time()
        try:
            if contract_id not in self.active_contracts:
                raise ValueError(f"Contract {contract_id} not found")
            
            contract = self.active_contracts[contract_id]
            self.logger.info(f"Starting execution context for {contract.metadata.name}")
            
            yield contract
            
        except Exception as e:
            self.logger.error(f"Execution context error: {e}")
            raise
        finally:
            execution_time = time.time() - start_time
            self.logger.info(f"Execution context completed in {execution_time:.3f}s")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "active_contracts": len(self.active_contracts),
            "total_executions": len(self.execution_history),
            "cache_status": "active" if self.config.enable_caching else "disabled",
            "compliance_checker": "active" if self.compliance_checker else "disabled",
            "performance_monitoring": "active" if self.config.performance_monitoring else "disabled"
        }
        
        # Check recent error rate
        recent_executions = self.execution_history[-100:] if len(self.execution_history) >= 100 else self.execution_history
        error_rate = sum(1 for r in recent_executions if r.error) / max(1, len(recent_executions))
        
        if error_rate > 0.1:
            health_status["status"] = "degraded"
            health_status["error_rate"] = error_rate
        
        # Check average execution time
        if recent_executions:
            avg_time = sum(r.execution_time for r in recent_executions) / len(recent_executions)
            health_status["average_execution_time"] = avg_time
            
            if avg_time > 5.0:
                health_status["status"] = "degraded"
                health_status["performance_issue"] = "high_execution_time"
        
        return health_status


# Global runtime instance
_runtime_instance: Optional[EnhancedContractRuntime] = None


def get_runtime(config: Optional[RuntimeConfig] = None) -> EnhancedContractRuntime:
    """Get or create the global runtime instance."""
    global _runtime_instance
    
    if _runtime_instance is None:
        _runtime_instance = EnhancedContractRuntime(config)
    
    return _runtime_instance


async def execute_simple_contract(
    contract: RewardContract,
    state: jnp.ndarray,
    action: jnp.ndarray,
    context: Optional[Dict[str, Any]] = None
) -> ExecutionResult:
    """Simple interface for single contract execution."""
    runtime = get_runtime()
    contract_id = runtime.register_contract(contract)
    result = await runtime.execute_contract(contract_id, state, action, context)
    return result
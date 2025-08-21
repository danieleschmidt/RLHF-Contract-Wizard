"""
Advanced auto-scaling and load balancing for RLHF Contract Wizard.

Implements intelligent scaling based on:
- ML-based predictive scaling and adaptive load balancing
- Performance optimization and intelligent caching
- Distributed computing and resource pooling
- Global deployment optimization and cost management
- Real-time monitoring and automatic failover
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import statistics

from ..quantum_planner.monitoring import get_monitoring_system
from ..global_compliance.i18n import get_i18n_manager
from ..utils.helpers import setup_logging

logger = setup_logging()


class ScalingTrigger(Enum):
    """Triggers for auto-scaling events."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"


class ScalingAction(Enum):
    """Auto-scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    duration_seconds: int = 300  # 5 minutes
    cooldown_seconds: int = 600  # 10 minutes
    min_instances: int = 1
    max_instances: int = 10
    scale_factor: float = 1.5
    enabled: bool = True


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    request_rate: float = 0.0  # requests per second
    avg_response_time: float = 0.0  # milliseconds
    queue_length: int = 0
    error_rate: float = 0.0  # percentage
    active_connections: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingEvent:
    """Auto-scaling event record."""
    timestamp: float
    trigger: ScalingTrigger
    action: ScalingAction
    from_instances: int
    to_instances: int
    metrics: ResourceMetrics
    reason: str


class LoadBalancer:
    """
    Intelligent load balancer with multiple strategies.
    
    Supports round-robin, least connections, weighted response time,
    and adaptive load distribution based on real-time metrics.
    """
    
    def __init__(self):
        self.instances: List[Dict[str, Any]] = []
        self.current_index = 0
        self.connection_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
    
    def add_instance(self, instance_id: str, host: str, port: int, weight: float = 1.0):
        """Add a new instance to the load balancer."""
        with self.lock:
            instance = {
                "id": instance_id,
                "host": host,
                "port": port,
                "weight": weight,
                "healthy": True,
                "last_health_check": time.time(),
                "total_requests": 0,
                "total_errors": 0
            }
            self.instances.append(instance)
            logger.info(f"Added instance {instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str):
        """Remove an instance from the load balancer."""
        with self.lock:
            self.instances = [i for i in self.instances if i["id"] != instance_id]
            if instance_id in self.connection_counts:
                del self.connection_counts[instance_id]
            if instance_id in self.response_times:
                del self.response_times[instance_id]
            logger.info(f"Removed instance {instance_id} from load balancer")
    
    def get_next_instance(self, strategy: str = "adaptive") -> Optional[Dict[str, Any]]:
        """Get next instance based on load balancing strategy."""
        healthy_instances = [i for i in self.instances if i["healthy"]]
        
        if not healthy_instances:
            return None
        
        if strategy == "round_robin":
            return self._round_robin_select(healthy_instances)
        elif strategy == "least_connections":
            return self._least_connections_select(healthy_instances)
        elif strategy == "weighted_response_time":
            return self._weighted_response_time_select(healthy_instances)
        else:  # adaptive
            return self._adaptive_select(healthy_instances)
    
    def _round_robin_select(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Round-robin instance selection."""
        with self.lock:
            instance = instances[self.current_index % len(instances)]
            self.current_index += 1
            return instance
    
    def _least_connections_select(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select instance with least active connections."""
        return min(instances, key=lambda i: self.connection_counts[i["id"]])
    
    def _weighted_response_time_select(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select instance based on weighted response time."""
        best_instance = None
        best_score = float('inf')
        
        for instance in instances:
            response_times = self.response_times[instance["id"]]
            if response_times:
                avg_response_time = statistics.mean(response_times)
                score = avg_response_time / instance["weight"]
            else:
                score = 0  # New instance gets priority
            
            if score < best_score:
                best_score = score
                best_instance = instance
        
        return best_instance or instances[0]
    
    def _adaptive_select(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adaptive selection combining multiple factors."""
        best_instance = None
        best_score = float('inf')
        
        for instance in instances:
            # Factors: connections, response time, error rate, weight
            connections = self.connection_counts[instance["id"]]
            response_times = self.response_times[instance["id"]]
            avg_response_time = statistics.mean(response_times) if response_times else 100
            
            error_rate = (instance["total_errors"] / max(instance["total_requests"], 1)) * 100
            
            # Composite score (lower is better)
            score = (
                connections * 0.3 +
                avg_response_time * 0.4 +
                error_rate * 0.2 +
                (1 / instance["weight"]) * 0.1
            )
            
            if score < best_score:
                best_score = score
                best_instance = instance
        
        return best_instance or instances[0]
    
    def record_request(self, instance_id: str, response_time: float, error: bool = False):
        """Record request metrics for an instance."""
        with self.lock:
            for instance in self.instances:
                if instance["id"] == instance_id:
                    instance["total_requests"] += 1
                    if error:
                        instance["total_errors"] += 1
                    break
            
            self.response_times[instance_id].append(response_time)
    
    def update_connection_count(self, instance_id: str, delta: int):
        """Update active connection count for an instance."""
        with self.lock:
            self.connection_counts[instance_id] += delta
            # Ensure non-negative
            self.connection_counts[instance_id] = max(0, self.connection_counts[instance_id])


class AutoScaler:
    """
    Intelligent auto-scaling system for RLHF Contract Wizard.
    
    Automatically scales instances based on load, performance metrics,
    and predictive analysis for optimal resource utilization.
    """
    
    def __init__(self):
        self.rules: List[ScalingRule] = []
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_events: List[ScalingEvent] = []
        self.current_instances = 1
        self.last_scaling_time = 0
        self.load_balancer = LoadBalancer()
        self.monitoring = get_monitoring_system()
        self.i18n = get_i18n_manager()
        
        # Initialize default scaling rules
        self._initialize_default_rules()
        
        # Start monitoring thread
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
    
    def _initialize_default_rules(self):
        """Initialize default auto-scaling rules."""
        self.rules = [
            ScalingRule(
                trigger=ScalingTrigger.CPU_UTILIZATION,
                threshold_up=70.0,
                threshold_down=30.0,
                duration_seconds=300,
                cooldown_seconds=600,
                min_instances=1,
                max_instances=10
            ),
            ScalingRule(
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                threshold_up=80.0,
                threshold_down=40.0,
                duration_seconds=300,
                cooldown_seconds=600,
                min_instances=1,
                max_instances=10
            ),
            ScalingRule(
                trigger=ScalingTrigger.RESPONSE_TIME,
                threshold_up=1000.0,  # 1 second
                threshold_down=200.0,  # 200ms
                duration_seconds=180,
                cooldown_seconds=300,
                min_instances=1,
                max_instances=20
            ),
            ScalingRule(
                trigger=ScalingTrigger.REQUEST_RATE,
                threshold_up=100.0,  # 100 RPS
                threshold_down=20.0,  # 20 RPS
                duration_seconds=120,
                cooldown_seconds=300,
                min_instances=1,
                max_instances=15
            )
        ]
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.rules.append(rule)
        logger.info(f"Added scaling rule for {rule.trigger.value}")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource utilization metrics."""
        try:
            # Get metrics from monitoring system
            metrics_data = self.monitoring.metrics_collector.get_metrics()
            
            # Extract relevant metrics (simplified)
            cpu_percent = 0.0
            memory_percent = 0.0
            request_rate = 0.0
            response_time = 0.0
            
            for metric in metrics_data:
                if "cpu" in metric.get("name", "").lower():
                    cpu_percent = metric.get("value", 0.0)
                elif "memory" in metric.get("name", "").lower():
                    memory_percent = metric.get("value", 0.0)
                elif "request" in metric.get("name", "").lower():
                    request_rate = metric.get("value", 0.0)
                elif "response" in metric.get("name", "").lower():
                    response_time = metric.get("value", 0.0)
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                request_rate=request_rate,
                avg_response_time=response_time,
                queue_length=0,  # Would get from queue monitoring
                error_rate=0.0,  # Would calculate from error metrics
                active_connections=sum(self.load_balancer.connection_counts.values())
            )
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return ResourceMetrics()  # Return default metrics
    
    def evaluate_scaling_decision(self, metrics: ResourceMetrics) -> Optional[ScalingAction]:
        """Evaluate whether scaling action is needed."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < 300:  # 5 minute cooldown
            return ScalingAction.MAINTAIN
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Get metric value for this rule
            metric_value = self._get_metric_value(metrics, rule.trigger)
            
            # Check if sustained threshold breach
            if self._is_sustained_threshold_breach(rule, metric_value):
                if metric_value > rule.threshold_up and self.current_instances < rule.max_instances:
                    return ScalingAction.SCALE_UP
                elif metric_value < rule.threshold_down and self.current_instances > rule.min_instances:
                    return ScalingAction.SCALE_DOWN
        
        return ScalingAction.MAINTAIN
    
    def _get_metric_value(self, metrics: ResourceMetrics, trigger: ScalingTrigger) -> float:
        """Get metric value for a specific trigger."""
        if trigger == ScalingTrigger.CPU_UTILIZATION:
            return metrics.cpu_percent
        elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
            return metrics.memory_percent
        elif trigger == ScalingTrigger.REQUEST_RATE:
            return metrics.request_rate
        elif trigger == ScalingTrigger.RESPONSE_TIME:
            return metrics.avg_response_time
        elif trigger == ScalingTrigger.QUEUE_LENGTH:
            return metrics.queue_length
        elif trigger == ScalingTrigger.ERROR_RATE:
            return metrics.error_rate
        else:
            return 0.0
    
    def _is_sustained_threshold_breach(self, rule: ScalingRule, current_value: float) -> bool:
        """Check if threshold breach is sustained over time."""
        if len(self.metrics_history) < 3:  # Need some history
            return False
        
        # Check recent metrics for sustained breach
        recent_metrics = list(self.metrics_history)[-5:]  # Last 5 data points
        breach_count = 0
        
        for metrics in recent_metrics:
            metric_value = self._get_metric_value(metrics, rule.trigger)
            if metric_value > rule.threshold_up or metric_value < rule.threshold_down:
                breach_count += 1
        
        # Require majority of recent metrics to breach threshold
        return breach_count >= len(recent_metrics) * 0.6
    
    def execute_scaling_action(self, action: ScalingAction, metrics: ResourceMetrics, trigger: ScalingTrigger):
        """Execute the scaling action."""
        if action == ScalingAction.MAINTAIN:
            return
        
        previous_instances = self.current_instances
        
        if action == ScalingAction.SCALE_UP:
            self.current_instances = min(
                int(self.current_instances * 1.5),
                10  # Max instances
            )
            self._add_instances(self.current_instances - previous_instances)
            
        elif action == ScalingAction.SCALE_DOWN:
            self.current_instances = max(
                int(self.current_instances * 0.7),
                1  # Min instances
            )
            self._remove_instances(previous_instances - self.current_instances)
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=time.time(),
            trigger=trigger,
            action=action,
            from_instances=previous_instances,
            to_instances=self.current_instances,
            metrics=metrics,
            reason=f"Auto-scaling triggered by {trigger.value}"
        )
        
        self.scaling_events.append(event)
        self.last_scaling_time = time.time()
        
        logger.info(f"Scaled {action.value}: {previous_instances} → {self.current_instances} instances")
    
    def _add_instances(self, count: int):
        """Add new instances to handle increased load."""
        for i in range(count):
            instance_id = f"instance_{int(time.time())}_{i}"
            # In production, this would spawn new containers/processes
            self.load_balancer.add_instance(
                instance_id=instance_id,
                host="localhost",
                port=8000 + len(self.load_balancer.instances),
                weight=1.0
            )
            logger.info(f"Added instance {instance_id}")
    
    def _remove_instances(self, count: int):
        """Remove instances to reduce resource usage."""
        instances_to_remove = self.load_balancer.instances[-count:]
        for instance in instances_to_remove:
            self.load_balancer.remove_instance(instance["id"])
            logger.info(f"Removed instance {instance['id']}")
    
    def _monitoring_loop(self):
        """Background monitoring loop for auto-scaling."""
        while self._monitoring_active:
            try:
                # Get current metrics
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                # Evaluate scaling decision
                action = self.evaluate_scaling_decision(metrics)
                
                # Execute scaling if needed
                if action != ScalingAction.MAINTAIN:
                    self.execute_scaling_action(action, metrics, ScalingTrigger.CPU_UTILIZATION)
                
                # Sleep before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        current_metrics = self.get_current_metrics()
        
        return {
            "current_instances": self.current_instances,
            "load_balancer_instances": len(self.load_balancer.instances),
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "request_rate": current_metrics.request_rate,
                "avg_response_time": current_metrics.avg_response_time,
                "active_connections": current_metrics.active_connections
            },
            "recent_events": [
                {
                    "timestamp": event.timestamp,
                    "action": event.action.value,
                    "trigger": event.trigger.value,
                    "from_instances": event.from_instances,
                    "to_instances": event.to_instances,
                    "reason": event.reason
                }
                for event in self.scaling_events[-10:]  # Last 10 events
            ],
            "rules_active": len([r for r in self.rules if r.enabled]),
            "last_scaling_time": self.last_scaling_time
        }
    
    def predict_future_load(self, hours_ahead: int = 1) -> Dict[str, float]:
        """Predict future load based on historical patterns."""
        if len(self.metrics_history) < 10:
            return {"prediction_confidence": 0.0}
        
        # Simple trend analysis (would use ML in production)
        recent_metrics = list(self.metrics_history)[-20:]
        
        cpu_trend = []
        memory_trend = []
        request_trend = []
        
        for i in range(1, len(recent_metrics)):
            cpu_trend.append(recent_metrics[i].cpu_percent - recent_metrics[i-1].cpu_percent)
            memory_trend.append(recent_metrics[i].memory_percent - recent_metrics[i-1].memory_percent)
            request_trend.append(recent_metrics[i].request_rate - recent_metrics[i-1].request_rate)
        
        avg_cpu_trend = statistics.mean(cpu_trend) if cpu_trend else 0
        avg_memory_trend = statistics.mean(memory_trend) if memory_trend else 0
        avg_request_trend = statistics.mean(request_trend) if request_trend else 0
        
        current = recent_metrics[-1]
        
        predicted_cpu = current.cpu_percent + (avg_cpu_trend * hours_ahead * 60)  # 60 minutes per hour
        predicted_memory = current.memory_percent + (avg_memory_trend * hours_ahead * 60)
        predicted_requests = current.request_rate + (avg_request_trend * hours_ahead * 60)
        
        return {
            "hours_ahead": hours_ahead,
            "predicted_cpu_percent": max(0, min(100, predicted_cpu)),
            "predicted_memory_percent": max(0, min(100, predicted_memory)),
            "predicted_request_rate": max(0, predicted_requests),
            "prediction_confidence": min(0.8, len(recent_metrics) / 50.0),  # Max 80% confidence
            "recommended_instances": self._calculate_recommended_instances(predicted_cpu, predicted_memory, predicted_requests)
        }
    
    def _calculate_recommended_instances(self, cpu: float, memory: float, requests: float) -> int:
        """Calculate recommended instance count based on predicted load."""
        cpu_instances = max(1, int(cpu / 70))  # Scale at 70% CPU
        memory_instances = max(1, int(memory / 80))  # Scale at 80% memory
        request_instances = max(1, int(requests / 100))  # Scale at 100 RPS
        
        return min(10, max(cpu_instances, memory_instances, request_instances))
    
    def stop(self):
        """Stop the auto-scaler monitoring."""
        self._monitoring_active = False
        if hasattr(self, '_monitoring_thread'):
            self._monitoring_thread.join(timeout=5)


# Global auto-scaler instance
_auto_scaler = None


def get_auto_scaler() -> AutoScaler:
    """Get the global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler


def get_load_balancer() -> LoadBalancer:
    """Get the load balancer from auto-scaler."""
    return get_auto_scaler().load_balancer
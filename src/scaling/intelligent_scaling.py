#!/usr/bin/env python3
"""
Intelligent Auto-Scaling System for RLHF Contract Wizard

Implements advanced auto-scaling with predictive analytics, load forecasting,
resource optimization, and intelligent workload distribution for cloud-native
production deployments.
"""

import time
import asyncio
import json
import logging
import threading
import math
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import statistics


class ScalingDirection(Enum):
    """Direction of scaling operation."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    INSTANCES = "instances"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    REACTIVE = "reactive"          # Based on current metrics
    PREDICTIVE = "predictive"      # Based on forecasts
    SCHEDULED = "scheduled"        # Time-based scaling
    EVENT_DRIVEN = "event_driven"  # External events
    COST_OPTIMIZED = "cost_optimized"  # Cost-aware scaling


class WorkloadPattern(Enum):
    """Identified workload patterns."""
    STEADY = "steady"
    BURSTY = "bursty"
    PERIODIC = "periodic"
    TRENDING = "trending"
    SEASONAL = "seasonal"
    CHAOTIC = "chaotic"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    gpu_usage: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingPolicy:
    """Scaling policy configuration."""
    resource_type: ResourceType
    min_instances: int
    max_instances: int
    target_utilization: float
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_cooldown: timedelta
    scale_down_cooldown: timedelta
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    enabled: bool = True


@dataclass
class ScalingAction:
    """Scaling action to be executed."""
    action_id: str
    resource_type: ResourceType
    direction: ScalingDirection
    current_capacity: int
    target_capacity: int
    trigger: ScalingTrigger
    reason: str
    timestamp: datetime
    estimated_cost_impact: float = 0.0
    priority: int = 1


@dataclass
class LoadForecast:
    """Load forecast prediction."""
    timestamp: datetime
    predicted_load: float
    confidence_interval: Tuple[float, float]
    forecast_horizon_minutes: int
    model_accuracy: float


class MetricsCollector:
    """Collects and aggregates system metrics for scaling decisions."""
    
    def __init__(self, collection_interval: int = 10):
        self.collection_interval = collection_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.custom_metric_collectors: Dict[str, Callable[[], float]] = {}
        
        # Background collection
        self.collection_thread: Optional[threading.Thread] = None
        self.collecting = False
        
    def start_collection(self):
        """Start background metrics collection."""
        if self.collecting:
            return
            
        self.collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        
    def stop_collection(self):
        """Stop background metrics collection."""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
            
    def _collection_loop(self):
        """Background collection loop."""
        while self.collecting:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logging.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
                
    def _collect_current_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Network I/O (simplified)
        network_io = 0.0  # Would implement actual network monitoring
        
        # Custom metrics
        custom_metrics = {}
        for name, collector in self.custom_metric_collectors.items():
            try:
                custom_metrics[name] = collector()
            except Exception as e:
                logging.warning(f"Failed to collect custom metric {name}: {e}")
                custom_metrics[name] = 0.0
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            custom_metrics=custom_metrics
        )
        
    def register_custom_metric(self, name: str, collector: Callable[[], float]):
        """Register custom metric collector."""
        self.custom_metric_collectors[name] = collector
        
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
        
    def get_metrics_window(
        self,
        window_minutes: int = 5
    ) -> List[ResourceMetrics]:
        """Get metrics within specified time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        return [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
    def get_average_metrics(self, window_minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get averaged metrics over time window."""
        window_metrics = self.get_metrics_window(window_minutes)
        
        if not window_metrics:
            return None
            
        avg_metrics = ResourceMetrics(
            cpu_usage=statistics.mean(m.cpu_usage for m in window_metrics),
            memory_usage=statistics.mean(m.memory_usage for m in window_metrics),
            disk_usage=statistics.mean(m.disk_usage for m in window_metrics),
            network_io=statistics.mean(m.network_io for m in window_metrics),
            gpu_usage=statistics.mean(m.gpu_usage for m in window_metrics),
        )
        
        # Average custom metrics
        if window_metrics[0].custom_metrics:
            for metric_name in window_metrics[0].custom_metrics.keys():
                values = [m.custom_metrics.get(metric_name, 0.0) for m in window_metrics]
                avg_metrics.custom_metrics[metric_name] = statistics.mean(values)
                
        return avg_metrics


class LoadPredictor:
    """Predicts future load patterns using machine learning techniques."""
    
    def __init__(self):
        self.historical_data: deque = deque(maxlen=10000)
        self.seasonal_patterns: Dict[str, List[float]] = {}
        self.trend_coefficient: float = 0.0
        self.model_accuracy: float = 0.75  # Simulated accuracy
        
    def add_data_point(self, timestamp: datetime, load: float):
        """Add data point to historical data."""
        self.historical_data.append((timestamp, load))
        self._update_patterns()
        
    def predict_load(
        self,
        forecast_horizon_minutes: int = 30
    ) -> LoadForecast:
        """Predict load for specified horizon."""
        if len(self.historical_data) < 10:
            # Not enough data for prediction
            current_load = self.historical_data[-1][1] if self.historical_data else 50.0
            return LoadForecast(
                timestamp=datetime.now() + timedelta(minutes=forecast_horizon_minutes),
                predicted_load=current_load,
                confidence_interval=(current_load * 0.8, current_load * 1.2),
                forecast_horizon_minutes=forecast_horizon_minutes,
                model_accuracy=0.5
            )
        
        # Simple prediction model (in production, would use sophisticated ML)
        recent_loads = [load for _, load in list(self.historical_data)[-50:]]
        
        # Trend analysis
        if len(recent_loads) >= 10:
            x = np.arange(len(recent_loads))
            y = np.array(recent_loads)
            trend_coefficient = np.polyfit(x, y, 1)[0]
        else:
            trend_coefficient = 0.0
            
        # Base prediction on recent average + trend
        base_load = statistics.mean(recent_loads)
        trend_adjustment = trend_coefficient * forecast_horizon_minutes
        predicted_load = max(0, base_load + trend_adjustment)
        
        # Add seasonal adjustment
        now = datetime.now()
        hour_of_day = now.hour
        if "hourly" in self.seasonal_patterns:
            seasonal_factor = self.seasonal_patterns["hourly"][hour_of_day % 24]
            predicted_load *= seasonal_factor
            
        # Calculate confidence interval
        recent_variance = statistics.variance(recent_loads) if len(recent_loads) > 1 else predicted_load * 0.1
        std_dev = math.sqrt(recent_variance)
        confidence_interval = (
            max(0, predicted_load - 2 * std_dev),
            predicted_load + 2 * std_dev
        )
        
        return LoadForecast(
            timestamp=datetime.now() + timedelta(minutes=forecast_horizon_minutes),
            predicted_load=predicted_load,
            confidence_interval=confidence_interval,
            forecast_horizon_minutes=forecast_horizon_minutes,
            model_accuracy=self.model_accuracy
        )
        
    def _update_patterns(self):
        """Update seasonal patterns from historical data."""
        if len(self.historical_data) < 24:
            return
            
        # Analyze hourly patterns
        hourly_loads = defaultdict(list)
        for timestamp, load in self.historical_data:
            hour = timestamp.hour
            hourly_loads[hour].append(load)
            
        # Calculate average load for each hour
        hourly_averages = {}
        overall_average = statistics.mean(load for _, load in self.historical_data)
        
        for hour in range(24):
            if hour in hourly_loads and hourly_loads[hour]:
                hourly_averages[hour] = statistics.mean(hourly_loads[hour])
            else:
                hourly_averages[hour] = overall_average
                
        # Convert to seasonal factors (relative to overall average)
        self.seasonal_patterns["hourly"] = [
            hourly_averages[hour] / overall_average
            for hour in range(24)
        ]
        
    def detect_workload_pattern(self) -> WorkloadPattern:
        """Detect the current workload pattern."""
        if len(self.historical_data) < 50:
            return WorkloadPattern.STEADY
            
        recent_loads = [load for _, load in list(self.historical_data)[-100:]]
        
        # Calculate metrics
        mean_load = statistics.mean(recent_loads)
        variance = statistics.variance(recent_loads)
        coefficient_of_variation = math.sqrt(variance) / mean_load if mean_load > 0 else 0
        
        # Trend analysis
        x = np.arange(len(recent_loads))
        y = np.array(recent_loads)
        slope = np.polyfit(x, y, 1)[0] if len(recent_loads) >= 2 else 0
        
        # Pattern classification
        if coefficient_of_variation > 0.5:
            if abs(slope) > mean_load * 0.01:
                return WorkloadPattern.TRENDING
            else:
                return WorkloadPattern.BURSTY
        elif abs(slope) > mean_load * 0.005:
            return WorkloadPattern.TRENDING
        elif self._has_periodic_pattern(recent_loads):
            return WorkloadPattern.PERIODIC
        else:
            return WorkloadPattern.STEADY
            
    def _has_periodic_pattern(self, loads: List[float]) -> bool:
        """Check if loads show periodic pattern."""
        if len(loads) < 20:
            return False
            
        # Simple autocorrelation check
        try:
            loads_array = np.array(loads)
            autocorr = np.correlate(loads_array, loads_array, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for peaks in autocorrelation
            normalized_autocorr = autocorr / autocorr[0]
            for i in range(5, min(len(normalized_autocorr), 50)):
                if normalized_autocorr[i] > 0.6:  # Strong correlation
                    return True
                    
        except Exception:
            pass
            
        return False


class ScalingDecisionEngine:
    """Makes intelligent scaling decisions based on metrics and predictions."""
    
    def __init__(self):
        self.policies: Dict[ResourceType, ScalingPolicy] = {}
        self.last_scaling_actions: Dict[ResourceType, datetime] = {}
        self.current_capacity: Dict[ResourceType, int] = {
            ResourceType.INSTANCES: 1,
            ResourceType.CPU: 2,
            ResourceType.MEMORY: 4,
        }
        
        # Cost optimization
        self.cost_per_unit: Dict[ResourceType, float] = {
            ResourceType.INSTANCES: 0.10,  # $ per hour per instance
            ResourceType.CPU: 0.05,        # $ per hour per CPU core
            ResourceType.MEMORY: 0.02,     # $ per hour per GB
        }
        
        # Setup default policies
        self._setup_default_policies()
        
    def _setup_default_policies(self):
        """Setup default scaling policies."""
        # Instance scaling policy
        self.policies[ResourceType.INSTANCES] = ScalingPolicy(
            resource_type=ResourceType.INSTANCES,
            min_instances=1,
            max_instances=10,
            target_utilization=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=50.0,
            scale_up_cooldown=timedelta(minutes=5),
            scale_down_cooldown=timedelta(minutes=15),
            scale_up_factor=1.5,
            scale_down_factor=0.8
        )
        
        # CPU scaling policy
        self.policies[ResourceType.CPU] = ScalingPolicy(
            resource_type=ResourceType.CPU,
            min_instances=1,
            max_instances=16,
            target_utilization=65.0,
            scale_up_threshold=75.0,
            scale_down_threshold=40.0,
            scale_up_cooldown=timedelta(minutes=3),
            scale_down_cooldown=timedelta(minutes=10),
            scale_up_factor=1.3,
            scale_down_factor=0.8
        )
        
        # Memory scaling policy
        self.policies[ResourceType.MEMORY] = ScalingPolicy(
            resource_type=ResourceType.MEMORY,
            min_instances=2,
            max_instances=32,
            target_utilization=70.0,
            scale_up_threshold=85.0,
            scale_down_threshold=45.0,
            scale_up_cooldown=timedelta(minutes=2),
            scale_down_cooldown=timedelta(minutes=8),
            scale_up_factor=1.4,
            scale_down_factor=0.7
        )
        
    def make_scaling_decision(
        self,
        current_metrics: ResourceMetrics,
        load_forecast: Optional[LoadForecast] = None,
        workload_pattern: Optional[WorkloadPattern] = None
    ) -> List[ScalingAction]:
        """Make scaling decisions based on current state and predictions."""
        
        scaling_actions = []
        current_time = datetime.now()
        
        for resource_type, policy in self.policies.items():
            if not policy.enabled:
                continue
                
            # Get current utilization for this resource
            if resource_type == ResourceType.INSTANCES:
                current_utilization = self._calculate_overall_utilization(current_metrics)
            elif resource_type == ResourceType.CPU:
                current_utilization = current_metrics.cpu_usage
            elif resource_type == ResourceType.MEMORY:
                current_utilization = current_metrics.memory_usage
            else:
                continue  # Skip unsupported resource types
                
            # Check cooldown periods
            last_action_time = self.last_scaling_actions.get(resource_type)
            if last_action_time:
                time_since_last = current_time - last_action_time
                if time_since_last < policy.scale_up_cooldown:
                    continue  # Still in cooldown
                    
            # Determine scaling direction
            scaling_direction = self._determine_scaling_direction(
                current_utilization,
                policy,
                load_forecast,
                workload_pattern
            )
            
            if scaling_direction == ScalingDirection.STABLE:
                continue
                
            # Calculate target capacity
            current_capacity = self.current_capacity.get(resource_type, 1)
            target_capacity = self._calculate_target_capacity(
                current_capacity,
                scaling_direction,
                policy,
                current_utilization,
                load_forecast
            )
            
            # Validate capacity bounds
            target_capacity = max(policy.min_instances, min(policy.max_instances, target_capacity))
            
            if target_capacity == current_capacity:
                continue  # No change needed
                
            # Create scaling action
            action = ScalingAction(
                action_id=f"{resource_type.value}_{int(time.time())}",
                resource_type=resource_type,
                direction=scaling_direction,
                current_capacity=current_capacity,
                target_capacity=target_capacity,
                trigger=self._determine_trigger_type(load_forecast),
                reason=self._generate_scaling_reason(
                    resource_type, current_utilization, policy, scaling_direction
                ),
                timestamp=current_time,
                estimated_cost_impact=self._calculate_cost_impact(
                    resource_type, current_capacity, target_capacity
                )
            )
            
            scaling_actions.append(action)
            
        # Prioritize and optimize actions
        return self._optimize_scaling_actions(scaling_actions)
        
    def _calculate_overall_utilization(self, metrics: ResourceMetrics) -> float:
        """Calculate overall system utilization."""
        return max(metrics.cpu_usage, metrics.memory_usage)
        
    def _determine_scaling_direction(
        self,
        current_utilization: float,
        policy: ScalingPolicy,
        load_forecast: Optional[LoadForecast],
        workload_pattern: Optional[WorkloadPattern]
    ) -> ScalingDirection:
        """Determine if scaling up, down, or stable."""
        
        # Reactive scaling based on current metrics
        if current_utilization >= policy.scale_up_threshold:
            return ScalingDirection.UP
        elif current_utilization <= policy.scale_down_threshold:
            return ScalingDirection.DOWN
            
        # Predictive scaling based on forecast
        if load_forecast and load_forecast.forecast_horizon_minutes <= 30:
            predicted_utilization = (
                current_utilization * 
                (load_forecast.predicted_load / 100.0)
            )
            
            # Be more aggressive with scaling up for predictions
            if predicted_utilization >= policy.scale_up_threshold * 0.9:
                return ScalingDirection.UP
                
        # Pattern-based adjustments
        if workload_pattern == WorkloadPattern.BURSTY:
            # Be more conservative with scaling down for bursty workloads
            if current_utilization <= policy.scale_down_threshold * 0.7:
                return ScalingDirection.DOWN
        elif workload_pattern == WorkloadPattern.TRENDING:
            # Be more aggressive with predictive scaling for trending workloads
            if load_forecast:
                trend_factor = 1.2 if load_forecast.predicted_load > 100 else 0.8
                adjusted_threshold = policy.scale_up_threshold * trend_factor
                if current_utilization >= adjusted_threshold:
                    return ScalingDirection.UP
                    
        return ScalingDirection.STABLE
        
    def _calculate_target_capacity(
        self,
        current_capacity: int,
        direction: ScalingDirection,
        policy: ScalingPolicy,
        current_utilization: float,
        load_forecast: Optional[LoadForecast]
    ) -> int:
        """Calculate target capacity for scaling."""
        
        if direction == ScalingDirection.UP:
            # Scale up calculation
            if load_forecast and load_forecast.predicted_load > 100:
                # Predictive scaling
                scale_factor = load_forecast.predicted_load / 100.0
                target_capacity = int(current_capacity * scale_factor * 1.1)  # 10% buffer
            else:
                # Reactive scaling
                utilization_factor = current_utilization / policy.target_utilization
                target_capacity = int(current_capacity * utilization_factor * 1.1)
                
            # Apply policy scale-up factor
            target_capacity = max(target_capacity, int(current_capacity * policy.scale_up_factor))
            
        else:  # ScalingDirection.DOWN
            # Scale down calculation
            target_capacity = int(current_capacity * policy.scale_down_factor)
            
            # Don't scale down too aggressively
            if current_utilization > 0:
                min_required = int((current_capacity * current_utilization) / policy.target_utilization)
                target_capacity = max(target_capacity, min_required)
                
        return target_capacity
        
    def _determine_trigger_type(self, load_forecast: Optional[LoadForecast]) -> ScalingTrigger:
        """Determine what triggered the scaling decision."""
        if load_forecast and load_forecast.forecast_horizon_minutes > 0:
            return ScalingTrigger.PREDICTIVE
        else:
            return ScalingTrigger.REACTIVE
            
    def _generate_scaling_reason(
        self,
        resource_type: ResourceType,
        utilization: float,
        policy: ScalingPolicy,
        direction: ScalingDirection
    ) -> str:
        """Generate human-readable scaling reason."""
        
        if direction == ScalingDirection.UP:
            return (
                f"{resource_type.value.upper()} utilization ({utilization:.1f}%) "
                f"exceeded threshold ({policy.scale_up_threshold:.1f}%)"
            )
        else:
            return (
                f"{resource_type.value.upper()} utilization ({utilization:.1f}%) "
                f"below threshold ({policy.scale_down_threshold:.1f}%)"
            )
            
    def _calculate_cost_impact(
        self,
        resource_type: ResourceType,
        current_capacity: int,
        target_capacity: int
    ) -> float:
        """Calculate estimated cost impact of scaling action."""
        
        capacity_delta = target_capacity - current_capacity
        cost_per_unit = self.cost_per_unit.get(resource_type, 0.0)
        
        # Estimate hourly cost impact
        return capacity_delta * cost_per_unit
        
    def _optimize_scaling_actions(self, actions: List[ScalingAction]) -> List[ScalingAction]:
        """Optimize scaling actions to minimize cost and conflicts."""
        
        if not actions:
            return actions
            
        # Sort by priority (high priority first)
        actions.sort(key=lambda a: a.priority, reverse=True)
        
        # Remove conflicting actions (keep highest priority)
        seen_resources = set()
        optimized_actions = []
        
        for action in actions:
            if action.resource_type not in seen_resources:
                optimized_actions.append(action)
                seen_resources.add(action.resource_type)
                
        return optimized_actions
        
    def update_capacity(self, resource_type: ResourceType, new_capacity: int):
        """Update current capacity after scaling action."""
        self.current_capacity[resource_type] = new_capacity
        self.last_scaling_actions[resource_type] = datetime.now()
        
    def get_scaling_recommendations(
        self,
        time_horizon_hours: int = 24
    ) -> Dict[str, Any]:
        """Get scaling recommendations for specified time horizon."""
        
        recommendations = {
            "time_horizon_hours": time_horizon_hours,
            "current_capacity": dict(self.current_capacity),
            "policies": {
                rt.value: {
                    "min": policy.min_instances,
                    "max": policy.max_instances,
                    "target_utilization": policy.target_utilization
                }
                for rt, policy in self.policies.items()
            },
            "cost_optimization": {
                "current_hourly_cost": sum(
                    capacity * self.cost_per_unit.get(rt, 0.0)
                    for rt, capacity in self.current_capacity.items()
                ),
                "recommendations": [
                    "Consider scheduling scale-down during low-usage hours",
                    "Enable cost-optimized scaling for non-critical workloads",
                    "Review resource allocation efficiency"
                ]
            }
        }
        
        return recommendations


class AutoScaler:
    """
    Main auto-scaling orchestrator combining all components.
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.load_predictor = LoadPredictor()
        self.decision_engine = ScalingDecisionEngine()
        
        # Scaling execution
        self.scaling_executor = ScalingExecutor()
        
        # Background processing
        self.scaling_thread: Optional[threading.Thread] = None
        self.scaling_active = False
        
        # Scaling history
        self.scaling_history: deque = deque(maxlen=1000)
        
    def start_auto_scaling(self, check_interval_seconds: int = 30):
        """Start auto-scaling system."""
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start scaling loop
        if not self.scaling_active:
            self.scaling_active = True
            self.scaling_thread = threading.Thread(
                target=self._scaling_loop,
                args=(check_interval_seconds,),
                daemon=True
            )
            self.scaling_thread.start()
            
        logging.info("Auto-scaling system started")
        
    def stop_auto_scaling(self):
        """Stop auto-scaling system."""
        self.scaling_active = False
        self.metrics_collector.stop_collection()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10)
            
        logging.info("Auto-scaling system stopped")
        
    def _scaling_loop(self, check_interval: int):
        """Main scaling decision loop."""
        while self.scaling_active:
            try:
                self._process_scaling_cycle()
                time.sleep(check_interval)
            except Exception as e:
                logging.error(f"Error in scaling loop: {e}")
                time.sleep(check_interval)
                
    def _process_scaling_cycle(self):
        """Process one scaling decision cycle."""
        
        # Get current metrics
        current_metrics = self.metrics_collector.get_current_metrics()
        if not current_metrics:
            return
            
        # Update load predictor
        current_load = max(current_metrics.cpu_usage, current_metrics.memory_usage)
        self.load_predictor.add_data_point(current_metrics.timestamp, current_load)
        
        # Get predictions
        load_forecast = self.load_predictor.predict_load(forecast_horizon_minutes=15)
        workload_pattern = self.load_predictor.detect_workload_pattern()
        
        # Make scaling decisions
        scaling_actions = self.decision_engine.make_scaling_decision(
            current_metrics,
            load_forecast,
            workload_pattern
        )
        
        # Execute scaling actions
        for action in scaling_actions:
            try:
                self._execute_scaling_action(action)
                self.scaling_history.append(action)
                logging.info(
                    f"Executed scaling action: {action.resource_type.value} "
                    f"{action.direction.value} from {action.current_capacity} "
                    f"to {action.target_capacity}"
                )
            except Exception as e:
                logging.error(f"Failed to execute scaling action: {e}")
                
    def _execute_scaling_action(self, action: ScalingAction):
        """Execute a scaling action."""
        
        # Update decision engine capacity
        self.decision_engine.update_capacity(
            action.resource_type,
            action.target_capacity
        )
        
        # Execute through scaling executor
        self.scaling_executor.execute_scaling(action)
        
    def register_custom_metric(self, name: str, collector: Callable[[], float]):
        """Register custom metric for scaling decisions."""
        self.metrics_collector.register_custom_metric(name, collector)
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status."""
        current_metrics = self.metrics_collector.get_current_metrics()
        recent_actions = list(self.scaling_history)[-10:]  # Last 10 actions
        
        return {
            "active": self.scaling_active,
            "current_metrics": asdict(current_metrics) if current_metrics else None,
            "current_capacity": dict(self.decision_engine.current_capacity),
            "recent_actions": [asdict(action) for action in recent_actions],
            "workload_pattern": self.load_predictor.detect_workload_pattern().value,
            "scaling_policies": {
                rt.value: {
                    "enabled": policy.enabled,
                    "target_utilization": policy.target_utilization,
                    "min_capacity": policy.min_instances,
                    "max_capacity": policy.max_instances
                }
                for rt, policy in self.decision_engine.policies.items()
            }
        }


class ScalingExecutor:
    """Executes scaling actions on the actual infrastructure."""
    
    def __init__(self):
        self.execution_history: List[ScalingAction] = []
        
    def execute_scaling(self, action: ScalingAction):
        """Execute scaling action on infrastructure."""
        
        # In a real implementation, this would interact with:
        # - Kubernetes HPA/VPA
        # - Cloud provider APIs (AWS, GCP, Azure)
        # - Container orchestrators
        # - Load balancers
        
        logging.info(f"Executing scaling action: {action.reason}")
        
        if action.resource_type == ResourceType.INSTANCES:
            self._scale_instances(action)
        elif action.resource_type == ResourceType.CPU:
            self._scale_cpu(action)
        elif action.resource_type == ResourceType.MEMORY:
            self._scale_memory(action)
        else:
            logging.warning(f"Unsupported resource type: {action.resource_type}")
            
        self.execution_history.append(action)
        
    def _scale_instances(self, action: ScalingAction):
        """Scale number of instances."""
        # Simulate instance scaling
        logging.info(
            f"Scaling instances from {action.current_capacity} "
            f"to {action.target_capacity}"
        )
        
    def _scale_cpu(self, action: ScalingAction):
        """Scale CPU resources."""
        # Simulate CPU scaling
        logging.info(
            f"Scaling CPU from {action.current_capacity} cores "
            f"to {action.target_capacity} cores"
        )
        
    def _scale_memory(self, action: ScalingAction):
        """Scale memory resources."""
        # Simulate memory scaling
        logging.info(
            f"Scaling memory from {action.current_capacity} GB "
            f"to {action.target_capacity} GB"
        )


# Example usage and demonstration
async def demonstrate_auto_scaling():
    """Demonstrate the auto-scaling system."""
    
    print("🚀 Starting Auto-Scaling Demonstration")
    print("=" * 50)
    
    # Initialize auto-scaler
    auto_scaler = AutoScaler()
    
    # Register custom metrics
    def contract_processing_rate():
        return random.uniform(10, 100)  # Contracts per minute
        
    def quantum_planning_queue_size():
        return random.uniform(0, 50)  # Queue size
        
    auto_scaler.register_custom_metric("contract_processing_rate", contract_processing_rate)
    auto_scaler.register_custom_metric("quantum_planning_queue_size", quantum_planning_queue_size)
    
    # Start auto-scaling
    auto_scaler.start_auto_scaling(check_interval_seconds=10)
    
    # Run for demonstration
    for i in range(6):  # Run for 60 seconds
        await asyncio.sleep(10)
        
        # Get status
        status = auto_scaler.get_scaling_status()
        
        print(f"\\nCycle {i+1}:")
        if status["current_metrics"]:
            metrics = status["current_metrics"]
            print(f"  CPU: {metrics['cpu_usage']:.1f}%")
            print(f"  Memory: {metrics['memory_usage']:.1f}%")
            print(f"  Custom metrics: {metrics.get('custom_metrics', {})}")
            
        print(f"  Current capacity: {status['current_capacity']}")
        print(f"  Workload pattern: {status['workload_pattern']}")
        
        if status["recent_actions"]:
            latest_action = status["recent_actions"][-1]
            print(f"  Latest action: {latest_action['reason']}")
            
    # Stop auto-scaling
    auto_scaler.stop_auto_scaling()
    
    # Get final status
    final_status = auto_scaler.get_scaling_status()
    print(f"\\n📊 Final Status:")
    print(json.dumps(final_status, indent=2, default=str))


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    import random
    asyncio.run(demonstrate_auto_scaling())
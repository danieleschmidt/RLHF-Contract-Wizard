"""
Auto-scaling system for dynamic resource management.

Provides horizontal and vertical scaling based on real-time metrics.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    WORKERS = "workers"
    CONNECTIONS = "connections"


@dataclass
class ScalingMetric:
    """Metric for scaling decisions."""
    name: str
    current_value: float
    target_value: float
    scale_up_threshold: float
    scale_down_threshold: float
    weight: float = 1.0
    
    @property
    def utilization_ratio(self) -> float:
        """Current utilization as ratio of target."""
        return self.current_value / max(self.target_value, 1)
    
    @property
    def suggested_action(self) -> ScalingAction:
        """Suggest scaling action based on thresholds."""
        if self.current_value > self.scale_up_threshold:
            return ScalingAction.SCALE_UP
        elif self.current_value < self.scale_down_threshold:
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.NO_ACTION


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    resource_type: ResourceType
    action: ScalingAction
    reason: str
    before_value: float
    after_value: float
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourcePool:
    """Managed resource pool."""
    name: str
    resource_type: ResourceType
    min_instances: int
    max_instances: int
    current_instances: int
    target_instances: int
    instance_startup_time: float = 30.0  # seconds
    instance_shutdown_time: float = 10.0  # seconds
    
    @property
    def can_scale_up(self) -> bool:
        """Check if pool can scale up."""
        return self.current_instances < self.max_instances
    
    @property
    def can_scale_down(self) -> bool:
        """Check if pool can scale down."""
        return self.current_instances > self.min_instances
    
    @property
    def scaling_in_progress(self) -> bool:
        """Check if scaling is in progress."""
        return self.current_instances != self.target_instances


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, history_window: int = 1440):  # 24 hours at 1-minute intervals
        """
        Initialize predictive scaler.
        
        Args:
            history_window: Number of historical data points to keep
        """
        self.history_window = history_window
        self.metric_history: Dict[str, deque] = {}
        self.pattern_cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def add_metric_data(self, metric_name: str, value: float, timestamp: float) -> None:
        """Add metric data point for prediction."""
        with self._lock:
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = deque(maxlen=self.history_window)
            
            self.metric_history[metric_name].append((timestamp, value))
    
    def predict_next_values(self, metric_name: str, minutes_ahead: int = 30) -> List[float]:
        """
        Predict future metric values.
        
        Args:
            metric_name: Name of metric to predict
            minutes_ahead: How many minutes to predict ahead
            
        Returns:
            List of predicted values
        """
        with self._lock:
            if metric_name not in self.metric_history:
                return []
            
            history = list(self.metric_history[metric_name])
            if len(history) < 60:  # Need at least 1 hour of data
                return []
            
            # Simple moving average prediction
            recent_values = [value for _, value in history[-60:]]  # Last hour
            moving_avg = sum(recent_values) / len(recent_values)
            
            # Detect trend
            early_avg = sum(recent_values[:30]) / 30
            late_avg = sum(recent_values[-30:]) / 30
            trend = late_avg - early_avg
            
            # Predict with trend
            predictions = []
            for i in range(minutes_ahead):
                predicted_value = moving_avg + (trend * (i + 1) / 30)
                predictions.append(max(0, predicted_value))
            
            return predictions
    
    def detect_patterns(self, metric_name: str) -> Dict[str, Any]:
        """Detect recurring patterns in metric data."""
        with self._lock:
            if metric_name not in self.metric_history:
                return {}
            
            history = list(self.metric_history[metric_name])
            if len(history) < 1440:  # Need at least 24 hours
                return {}
            
            values = [value for _, value in history]
            
            # Detect daily patterns (24-hour cycles)
            daily_pattern = self._detect_daily_pattern(values)
            
            # Detect hourly patterns
            hourly_pattern = self._detect_hourly_pattern(values)
            
            return {
                "daily_pattern": daily_pattern,
                "hourly_pattern": hourly_pattern,
                "peak_hours": self._find_peak_hours(values),
                "low_hours": self._find_low_hours(values)
            }
    
    def _detect_daily_pattern(self, values: List[float]) -> List[float]:
        """Detect daily recurring patterns."""
        # Group by hour of day and average
        hourly_averages = [0.0] * 24
        hourly_counts = [0] * 24
        
        for i, value in enumerate(values):
            hour = (i // 60) % 24  # Assuming 1-minute intervals
            hourly_averages[hour] += value
            hourly_counts[hour] += 1
        
        # Calculate averages
        for i in range(24):
            if hourly_counts[i] > 0:
                hourly_averages[i] /= hourly_counts[i]
        
        return hourly_averages
    
    def _detect_hourly_pattern(self, values: List[float]) -> List[float]:
        """Detect hourly patterns within the hour."""
        # Group by minute within hour
        minute_averages = [0.0] * 60
        minute_counts = [0] * 60
        
        for i, value in enumerate(values):
            minute = i % 60
            minute_averages[minute] += value
            minute_counts[minute] += 1
        
        # Calculate averages
        for i in range(60):
            if minute_counts[i] > 0:
                minute_averages[i] /= minute_counts[i]
        
        return minute_averages
    
    def _find_peak_hours(self, values: List[float]) -> List[int]:
        """Find peak usage hours."""
        hourly_pattern = self._detect_daily_pattern(values)
        avg_value = sum(hourly_pattern) / len(hourly_pattern)
        
        peak_hours = []
        for hour, value in enumerate(hourly_pattern):
            if value > avg_value * 1.3:  # 30% above average
                peak_hours.append(hour)
        
        return peak_hours
    
    def _find_low_hours(self, values: List[float]) -> List[int]:
        """Find low usage hours."""
        hourly_pattern = self._detect_daily_pattern(values)
        avg_value = sum(hourly_pattern) / len(hourly_pattern)
        
        low_hours = []
        for hour, value in enumerate(hourly_pattern):
            if value < avg_value * 0.7:  # 30% below average
                low_hours.append(hour)
        
        return low_hours


class AutoScaler:
    """Comprehensive auto-scaling system."""
    
    def __init__(
        self,
        cooldown_period: float = 300.0,  # 5 minutes
        evaluation_interval: float = 60.0,  # 1 minute
        enable_predictive_scaling: bool = True
    ):
        """
        Initialize auto-scaler.
        
        Args:
            cooldown_period: Minimum time between scaling actions
            evaluation_interval: How often to evaluate scaling decisions
            enable_predictive_scaling: Whether to use predictive scaling
        """
        self.cooldown_period = cooldown_period
        self.evaluation_interval = evaluation_interval
        self.enable_predictive_scaling = enable_predictive_scaling
        
        # Resource pools
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        # Scaling metrics
        self.scaling_metrics: Dict[str, ScalingMetric] = {}
        
        # Event history
        self.scaling_events: List[ScalingEvent] = []
        self.max_event_history = 1000
        
        # Predictive scaler
        self.predictive_scaler = PredictiveScaler() if enable_predictive_scaling else None
        
        # State
        self.last_scaling_action: Dict[str, float] = {}
        self.scaling_enabled = True
        self._evaluation_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Scaling callbacks
        self.scale_up_callbacks: Dict[str, Callable] = {}
        self.scale_down_callbacks: Dict[str, Callable] = {}
    
    def register_resource_pool(
        self,
        name: str,
        resource_type: ResourceType,
        min_instances: int,
        max_instances: int,
        current_instances: int,
        scale_up_callback: Callable,
        scale_down_callback: Callable
    ) -> None:
        """Register a resource pool for auto-scaling."""
        pool = ResourcePool(
            name=name,
            resource_type=resource_type,
            min_instances=min_instances,
            max_instances=max_instances,
            current_instances=current_instances,
            target_instances=current_instances
        )
        
        self.resource_pools[name] = pool
        self.scale_up_callbacks[name] = scale_up_callback
        self.scale_down_callbacks[name] = scale_down_callback
        self.last_scaling_action[name] = 0.0
        
        logger.info(f"Registered resource pool: {name} ({resource_type.value})")
    
    def register_scaling_metric(
        self,
        name: str,
        target_value: float,
        scale_up_threshold: float,
        scale_down_threshold: float,
        weight: float = 1.0
    ) -> None:
        """Register a metric for scaling decisions."""
        metric = ScalingMetric(
            name=name,
            current_value=0.0,
            target_value=target_value,
            scale_up_threshold=scale_up_threshold,
            scale_down_threshold=scale_down_threshold,
            weight=weight
        )
        
        self.scaling_metrics[name] = metric
        logger.info(f"Registered scaling metric: {name}")
    
    def update_metric(self, name: str, value: float) -> None:
        """Update metric value."""
        if name in self.scaling_metrics:
            self.scaling_metrics[name].current_value = value
            
            # Add to predictive scaler
            if self.predictive_scaler:
                self.predictive_scaler.add_metric_data(name, value, time.time())
    
    async def start_auto_scaling(self) -> None:
        """Start automatic scaling evaluation."""
        if self._evaluation_task and not self._evaluation_task.done():
            logger.warning("Auto-scaling already running")
            return
        
        self.scaling_enabled = True
        self._evaluation_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto-scaling started")
    
    async def stop_auto_scaling(self) -> None:
        """Stop automatic scaling."""
        self.scaling_enabled = False
        
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaling stopped")
    
    async def _scaling_loop(self) -> None:
        """Main scaling evaluation loop."""
        while self.scaling_enabled:
            try:
                await self._evaluate_scaling_decisions()
                await asyncio.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_scaling_decisions(self) -> None:
        """Evaluate and execute scaling decisions."""
        async with self._lock:
            current_time = time.time()
            
            for pool_name, pool in self.resource_pools.items():
                # Check cooldown period
                if current_time - self.last_scaling_action[pool_name] < self.cooldown_period:
                    continue
                
                # Get scaling decision
                decision = await self._make_scaling_decision(pool_name)
                
                if decision != ScalingAction.NO_ACTION:
                    await self._execute_scaling_action(pool_name, decision)
                    self.last_scaling_action[pool_name] = current_time
    
    async def _make_scaling_decision(self, pool_name: str) -> ScalingAction:
        """Make scaling decision for a resource pool."""
        pool = self.resource_pools[pool_name]
        
        # Collect relevant metrics
        relevant_metrics = []
        for metric_name, metric in self.scaling_metrics.items():
            if pool.resource_type.value in metric_name.lower():
                relevant_metrics.append(metric)
        
        if not relevant_metrics:
            return ScalingAction.NO_ACTION
        
        # Calculate weighted decision
        scale_up_score = 0.0
        scale_down_score = 0.0
        total_weight = 0.0
        
        for metric in relevant_metrics:
            if metric.suggested_action == ScalingAction.SCALE_UP:
                scale_up_score += metric.weight * metric.utilization_ratio
            elif metric.suggested_action == ScalingAction.SCALE_DOWN:
                scale_down_score += metric.weight * (2.0 - metric.utilization_ratio)
            
            total_weight += metric.weight
        
        # Normalize scores
        if total_weight > 0:
            scale_up_score /= total_weight
            scale_down_score /= total_weight
        
        # Predictive scaling adjustment
        if self.enable_predictive_scaling and self.predictive_scaler:
            prediction_adjustment = await self._get_predictive_adjustment(pool_name)
            scale_up_score += prediction_adjustment
            scale_down_score -= prediction_adjustment
        
        # Make decision with hysteresis
        if scale_up_score > 1.2 and pool.can_scale_up:  # 20% threshold
            return ScalingAction.SCALE_UP
        elif scale_down_score > 1.2 and pool.can_scale_down:  # 20% threshold
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.NO_ACTION
    
    async def _get_predictive_adjustment(self, pool_name: str) -> float:
        """Get predictive scaling adjustment."""
        # Find relevant metric for prediction
        pool = self.resource_pools[pool_name]
        relevant_metric = None
        
        for metric_name, metric in self.scaling_metrics.items():
            if pool.resource_type.value in metric_name.lower():
                relevant_metric = metric
                break
        
        if not relevant_metric:
            return 0.0
        
        # Get predictions
        predictions = self.predictive_scaler.predict_next_values(
            relevant_metric.name, 
            minutes_ahead=15
        )
        
        if not predictions:
            return 0.0
        
        # Calculate if we expect to hit thresholds
        max_predicted = max(predictions)
        min_predicted = min(predictions)
        
        adjustment = 0.0
        
        if max_predicted > relevant_metric.scale_up_threshold:
            adjustment += 0.3  # Proactive scale up
        
        if min_predicted < relevant_metric.scale_down_threshold:
            adjustment -= 0.3  # Proactive scale down
        
        return adjustment
    
    async def _execute_scaling_action(
        self, 
        pool_name: str, 
        action: ScalingAction
    ) -> None:
        """Execute scaling action."""
        pool = self.resource_pools[pool_name]
        
        before_value = pool.current_instances
        
        if action == ScalingAction.SCALE_UP:
            new_instances = min(pool.current_instances + 1, pool.max_instances)
            pool.target_instances = new_instances
            
            # Execute scale up callback
            try:
                await asyncio.to_thread(self.scale_up_callbacks[pool_name])
                pool.current_instances = new_instances
                logger.info(f"Scaled up {pool_name}: {before_value} -> {new_instances}")
                
            except Exception as e:
                logger.error(f"Failed to scale up {pool_name}: {e}")
                return
        
        elif action == ScalingAction.SCALE_DOWN:
            new_instances = max(pool.current_instances - 1, pool.min_instances)
            pool.target_instances = new_instances
            
            # Execute scale down callback
            try:
                await asyncio.to_thread(self.scale_down_callbacks[pool_name])
                pool.current_instances = new_instances
                logger.info(f"Scaled down {pool_name}: {before_value} -> {new_instances}")
                
            except Exception as e:
                logger.error(f"Failed to scale down {pool_name}: {e}")
                return
        
        # Record scaling event
        metrics_snapshot = {
            name: metric.current_value 
            for name, metric in self.scaling_metrics.items()
        }
        
        event = ScalingEvent(
            timestamp=time.time(),
            resource_type=pool.resource_type,
            action=action,
            reason=f"Metric-based scaling for {pool_name}",
            before_value=before_value,
            after_value=pool.current_instances,
            metrics_snapshot=metrics_snapshot
        )
        
        self.scaling_events.append(event)
        
        # Maintain event history size
        if len(self.scaling_events) > self.max_event_history:
            self.scaling_events.pop(0)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "scaling_enabled": self.scaling_enabled,
            "resource_pools": {
                name: {
                    "current_instances": pool.current_instances,
                    "target_instances": pool.target_instances,
                    "min_instances": pool.min_instances,
                    "max_instances": pool.max_instances,
                    "resource_type": pool.resource_type.value,
                    "scaling_in_progress": pool.scaling_in_progress
                }
                for name, pool in self.resource_pools.items()
            },
            "recent_events": [
                {
                    "timestamp": event.timestamp,
                    "action": event.action.value,
                    "resource_type": event.resource_type.value,
                    "reason": event.reason,
                    "before": event.before_value,
                    "after": event.after_value
                }
                for event in self.scaling_events[-10:]  # Last 10 events
            ],
            "current_metrics": {
                name: {
                    "current_value": metric.current_value,
                    "target_value": metric.target_value,
                    "utilization_ratio": metric.utilization_ratio,
                    "suggested_action": metric.suggested_action.value
                }
                for name, metric in self.scaling_metrics.items()
            }
        }
    
    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Get scaling recommendations based on patterns."""
        recommendations = []
        
        if not self.predictive_scaler:
            return recommendations
        
        # Analyze patterns for each metric
        for metric_name, metric in self.scaling_metrics.items():
            patterns = self.predictive_scaler.detect_patterns(metric_name)
            
            if "peak_hours" in patterns and patterns["peak_hours"]:
                recommendations.append({
                    "type": "schedule_scale_up",
                    "metric": metric_name,
                    "peak_hours": patterns["peak_hours"],
                    "description": f"Consider pre-scaling before peak hours: {patterns['peak_hours']}"
                })
            
            if "low_hours" in patterns and patterns["low_hours"]:
                recommendations.append({
                    "type": "schedule_scale_down",
                    "metric": metric_name,
                    "low_hours": patterns["low_hours"],
                    "description": f"Consider scaling down during low hours: {patterns['low_hours']}"
                })
        
        return recommendations
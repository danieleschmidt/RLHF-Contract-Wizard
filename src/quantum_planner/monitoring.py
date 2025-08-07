"""
Comprehensive monitoring and observability for quantum task planning.

Implements real-time metrics collection, health checks, alerting,
distributed tracing, and integration with monitoring systems.
"""

import time
import threading
import json
import queue
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import functools
import weakref

from .core import QuantumTask, TaskState, QuantumTaskPlanner
from .contracts import ContractualTaskPlanner
from .logging_config import get_logger, EventType


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"      # Monotonically increasing value
    GAUGE = "gauge"         # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    TIMING = "timing"       # Timing measurements


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp,
            'labels': self.labels
        }


@dataclass
class HealthCheck:
    """Health check definition and result."""
    name: str
    check_function: Callable[[], bool]
    description: str
    timeout: float = 30.0
    critical: bool = False
    last_check_time: float = 0.0
    last_result: bool = True
    last_error: Optional[str] = None
    check_count: int = 0
    failure_count: int = 0
    
    def run_check(self) -> bool:
        """Run the health check."""
        start_time = time.time()
        
        try:
            result = self.check_function()
            self.last_result = result
            self.last_error = None
            
            if not result:
                self.failure_count += 1
            
        except Exception as e:
            self.last_result = False
            self.last_error = str(e)
            self.failure_count += 1
            result = False
        
        self.last_check_time = time.time()
        self.check_count += 1
        
        return result
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.check_count == 0:
            return 0.0
        return self.failure_count / self.check_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'critical': self.critical,
            'last_check_time': self.last_check_time,
            'last_result': self.last_result,
            'last_error': self.last_error,
            'check_count': self.check_count,
            'failure_count': self.failure_count,
            'failure_rate': self.failure_rate
        }


@dataclass
class Alert:
    """Alert definition and state."""
    name: str
    condition: Callable[['MonitoringSystem'], bool]
    severity: AlertSeverity
    description: str
    cooldown_seconds: float = 300.0  # 5 minutes
    last_triggered: float = 0.0
    trigger_count: int = 0
    active: bool = False
    
    def should_trigger(self, monitoring_system: 'MonitoringSystem') -> bool:
        """Check if alert should trigger."""
        current_time = time.time()
        
        # Check cooldown
        if self.active and (current_time - self.last_triggered) < self.cooldown_seconds:
            return False
        
        try:
            return self.condition(monitoring_system)
        except Exception:
            return False
    
    def trigger(self) -> Dict[str, Any]:
        """Trigger the alert."""
        self.last_triggered = time.time()
        self.trigger_count += 1
        self.active = True
        
        return {
            'alert_name': self.name,
            'severity': self.severity.value,
            'description': self.description,
            'timestamp': self.last_triggered,
            'trigger_count': self.trigger_count
        }
    
    def resolve(self):
        """Resolve the alert."""
        self.active = False


class MetricsCollector:
    """
    Metrics collection system for quantum planning operations.
    
    Collects, aggregates, and exports metrics for monitoring systems.
    """
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.metric_aggregates: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background aggregation
        self._aggregation_interval = 60.0  # 1 minute
        self._last_aggregation = time.time()
        
        self.logger = get_logger()
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a single metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            labels=labels or {}
        )
        
        with self._lock:
            self.metrics_buffer.append(metric)
            self._update_aggregates(metric)
        
        # Periodic aggregation
        if time.time() - self._last_aggregation > self._aggregation_interval:
            self._aggregate_metrics()
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        self.record_metric(name, duration, MetricType.TIMING, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self.record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def _update_aggregates(self, metric: Metric):
        """Update metric aggregates."""
        key = f"{metric.name}_{json.dumps(metric.labels, sort_keys=True)}"
        
        if key not in self.metric_aggregates:
            self.metric_aggregates[key] = {
                'name': metric.name,
                'type': metric.metric_type.value,
                'labels': metric.labels,
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'last_value': 0.0,
                'last_timestamp': 0.0
            }
        
        agg = self.metric_aggregates[key]
        
        if metric.metric_type == MetricType.COUNTER:
            agg['sum'] += metric.value
        elif metric.metric_type == MetricType.GAUGE:
            agg['last_value'] = metric.value
        elif metric.metric_type in [MetricType.TIMING, MetricType.HISTOGRAM]:
            agg['sum'] += metric.value
            agg['min'] = min(agg['min'], metric.value)
            agg['max'] = max(agg['max'], metric.value)
        
        agg['count'] += 1
        agg['last_timestamp'] = metric.timestamp
    
    def _aggregate_metrics(self):
        """Perform periodic metric aggregation."""
        with self._lock:
            self._last_aggregation = time.time()
            
            # Calculate derived metrics
            for key, agg in self.metric_aggregates.items():
                if agg['count'] > 0:
                    if agg['type'] in ['timing', 'histogram']:
                        agg['average'] = agg['sum'] / agg['count']
                        
                        # Calculate percentiles (simplified)
                        # In production, would use more sophisticated percentile calculation
                        agg['p95'] = agg['max'] * 0.95  # Rough approximation
                        agg['p99'] = agg['max'] * 0.99
        
        self.logger.debug(f"Metrics aggregation completed: {len(self.metric_aggregates)} metric keys")
    
    def get_metrics(self, since: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get metrics since specified timestamp."""
        with self._lock:
            if since is None:
                return [metric.to_dict() for metric in self.metrics_buffer]
            else:
                return [
                    metric.to_dict() for metric in self.metrics_buffer
                    if metric.timestamp >= since
                ]
    
    def get_aggregates(self) -> Dict[str, Dict[str, Any]]:
        """Get metric aggregates."""
        with self._lock:
            return dict(self.metric_aggregates)
    
    def clear_metrics(self, older_than: Optional[float] = None):
        """Clear old metrics."""
        with self._lock:
            if older_than is None:
                self.metrics_buffer.clear()
                self.metric_aggregates.clear()
            else:
                # Remove old metrics
                self.metrics_buffer = deque(
                    (m for m in self.metrics_buffer if m.timestamp >= older_than),
                    maxlen=self.buffer_size
                )
                
                # Clean aggregates
                keys_to_remove = [
                    key for key, agg in self.metric_aggregates.items()
                    if agg['last_timestamp'] < older_than
                ]
                for key in keys_to_remove:
                    del self.metric_aggregates[key]


class HealthChecker:
    """
    Health checking system for quantum planning components.
    
    Monitors system health and provides status information.
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_status = HealthStatus.HEALTHY
        self.last_check_time = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background checking
        self._check_interval = 30.0  # 30 seconds
        self._check_thread = None
        self._stop_checking = threading.Event()
        
        self.logger = get_logger()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        # Memory usage check
        def check_memory_usage() -> bool:
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent < 90  # Less than 90% memory usage
            except ImportError:
                return True  # Can't check without psutil
        
        self.register_health_check(
            "memory_usage",
            check_memory_usage,
            "System memory usage below 90%",
            critical=True
        )
        
        # CPU usage check
        def check_cpu_usage() -> bool:
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                return cpu_percent < 95  # Less than 95% CPU usage
            except ImportError:
                return True  # Can't check without psutil
        
        self.register_health_check(
            "cpu_usage",
            check_cpu_usage,
            "System CPU usage below 95%"
        )
        
        # Thread availability check
        def check_thread_availability() -> bool:
            try:
                import threading
                return threading.active_count() < 100  # Less than 100 active threads
            except:
                return True
        
        self.register_health_check(
            "thread_availability",
            check_thread_availability,
            "Thread count below safety limit"
        )
    
    def register_health_check(
        self,
        name: str,
        check_function: Callable[[], bool],
        description: str,
        timeout: float = 30.0,
        critical: bool = False
    ):
        """Register a health check."""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            description=description,
            timeout=timeout,
            critical=critical
        )
        
        with self._lock:
            self.health_checks[name] = health_check
        
        self.logger.info(f"Registered health check: {name} (critical: {critical})")
    
    def start_background_checking(self):
        """Start background health checking."""
        if self._check_thread is None or not self._check_thread.is_alive():
            self._stop_checking.clear()
            self._check_thread = threading.Thread(target=self._background_check_loop, daemon=True)
            self._check_thread.start()
            self.logger.info("Background health checking started")
    
    def stop_background_checking(self):
        """Stop background health checking."""
        self._stop_checking.set()
        if self._check_thread and self._check_thread.is_alive():
            self._check_thread.join(timeout=5.0)
        self.logger.info("Background health checking stopped")
    
    def _background_check_loop(self):
        """Background health check loop."""
        while not self._stop_checking.is_set():
            try:
                self.run_all_checks()
                self._stop_checking.wait(self._check_interval)
            except Exception as e:
                self.logger.error(f"Health check loop error: {str(e)}")
                self._stop_checking.wait(10.0)  # Wait 10 seconds on error
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        start_time = time.time()
        results = {}
        critical_failures = []
        
        with self._lock:
            for name, health_check in self.health_checks.items():
                try:
                    result = health_check.run_check()
                    results[name] = health_check.to_dict()
                    
                    if not result and health_check.critical:
                        critical_failures.append(name)
                        
                except Exception as e:
                    self.logger.error(f"Health check {name} failed with exception: {str(e)}")
                    results[name] = {
                        'name': name,
                        'last_result': False,
                        'last_error': str(e),
                        'critical': health_check.critical
                    }
                    
                    if health_check.critical:
                        critical_failures.append(name)
        
        # Determine overall status
        if critical_failures:
            self.overall_status = HealthStatus.CRITICAL
        elif any(not r.get('last_result', True) for r in results.values()):
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.HEALTHY
        
        self.last_check_time = time.time()
        
        summary = {
            'overall_status': self.overall_status.value,
            'last_check_time': self.last_check_time,
            'check_duration': time.time() - start_time,
            'total_checks': len(results),
            'passed_checks': sum(1 for r in results.values() if r.get('last_result', False)),
            'failed_checks': sum(1 for r in results.values() if not r.get('last_result', True)),
            'critical_failures': critical_failures,
            'checks': results
        }
        
        # Log significant status changes
        if critical_failures:
            self.logger.error(
                f"Critical health check failures: {critical_failures}",
                event_type=EventType.SYSTEM_START,  # Could add HEALTH_CHECK_FAILED
                critical_failures=critical_failures
            )
        
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            return {
                'overall_status': self.overall_status.value,
                'last_check_time': self.last_check_time,
                'registered_checks': len(self.health_checks),
                'checks': {
                    name: check.to_dict() 
                    for name, check in self.health_checks.items()
                }
            }


class AlertManager:
    """
    Alert management system for quantum planning operations.
    
    Manages alert rules, triggers, and notifications.
    """
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background processing
        self._check_interval = 30.0  # 30 seconds
        self._check_thread = None
        self._stop_checking = threading.Event()
        
        self.logger = get_logger()
        
        # Register default alerts
        self._register_default_alerts()
    
    def _register_default_alerts(self):
        """Register default system alerts."""
        
        # High error rate alert
        def high_error_rate_condition(monitoring_system: 'MonitoringSystem') -> bool:
            metrics = monitoring_system.metrics_collector.get_aggregates()
            
            # Check error rate in last 5 minutes
            error_metrics = [
                agg for agg in metrics.values()
                if 'error' in agg['name'].lower() and 
                time.time() - agg['last_timestamp'] < 300
            ]
            
            if not error_metrics:
                return False
            
            # Calculate error rate
            total_errors = sum(agg.get('sum', 0) for agg in error_metrics)
            return total_errors > 10  # More than 10 errors in 5 minutes
        
        self.register_alert(
            "high_error_rate",
            high_error_rate_condition,
            AlertSeverity.ERROR,
            "High error rate detected (>10 errors/5min)",
            cooldown_seconds=300
        )
        
        # Memory usage alert
        def high_memory_usage_condition(monitoring_system: 'MonitoringSystem') -> bool:
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent > 85  # More than 85% memory usage
            except ImportError:
                return False
        
        self.register_alert(
            "high_memory_usage",
            high_memory_usage_condition,
            AlertSeverity.WARNING,
            "High memory usage detected (>85%)",
            cooldown_seconds=600
        )
        
        # Task failure rate alert
        def high_task_failure_rate_condition(monitoring_system: 'MonitoringSystem') -> bool:
            metrics = monitoring_system.metrics_collector.get_aggregates()
            
            # Find task completion and failure metrics
            completion_metrics = [
                agg for agg in metrics.values()
                if 'task_completed' in agg['name']
            ]
            failure_metrics = [
                agg for agg in metrics.values()
                if 'task_failed' in agg['name']
            ]
            
            if not completion_metrics or not failure_metrics:
                return False
            
            total_completions = sum(agg.get('sum', 0) for agg in completion_metrics)
            total_failures = sum(agg.get('sum', 0) for agg in failure_metrics)
            
            if total_completions + total_failures < 10:  # Not enough data
                return False
            
            failure_rate = total_failures / (total_completions + total_failures)
            return failure_rate > 0.2  # More than 20% failure rate
        
        self.register_alert(
            "high_task_failure_rate",
            high_task_failure_rate_condition,
            AlertSeverity.WARNING,
            "High task failure rate detected (>20%)",
            cooldown_seconds=600
        )
    
    def register_alert(
        self,
        name: str,
        condition: Callable[['MonitoringSystem'], bool],
        severity: AlertSeverity,
        description: str,
        cooldown_seconds: float = 300.0
    ):
        """Register an alert rule."""
        alert = Alert(
            name=name,
            condition=condition,
            severity=severity,
            description=description,
            cooldown_seconds=cooldown_seconds
        )
        
        with self._lock:
            self.alerts[name] = alert
        
        self.logger.info(f"Registered alert: {name} (severity: {severity.value})")
    
    def start_background_checking(self, monitoring_system: 'MonitoringSystem'):
        """Start background alert checking."""
        if self._check_thread is None or not self._check_thread.is_alive():
            self._stop_checking.clear()
            self._check_thread = threading.Thread(
                target=self._background_check_loop,
                args=(monitoring_system,),
                daemon=True
            )
            self._check_thread.start()
            self.logger.info("Background alert checking started")
    
    def stop_background_checking(self):
        """Stop background alert checking."""
        self._stop_checking.set()
        if self._check_thread and self._check_thread.is_alive():
            self._check_thread.join(timeout=5.0)
        self.logger.info("Background alert checking stopped")
    
    def _background_check_loop(self, monitoring_system: 'MonitoringSystem'):
        """Background alert check loop."""
        while not self._stop_checking.is_set():
            try:
                self.check_all_alerts(monitoring_system)
                self._stop_checking.wait(self._check_interval)
            except Exception as e:
                self.logger.error(f"Alert check loop error: {str(e)}")
                self._stop_checking.wait(10.0)  # Wait 10 seconds on error
    
    def check_all_alerts(self, monitoring_system: 'MonitoringSystem'):
        """Check all registered alerts."""
        with self._lock:
            for name, alert in self.alerts.items():
                try:
                    if alert.should_trigger(monitoring_system):
                        alert_data = alert.trigger()
                        self.alert_history.append(alert_data)
                        
                        # Log the alert
                        self.logger.log(
                            {
                                AlertSeverity.INFO: 20,    # INFO
                                AlertSeverity.WARNING: 30, # WARNING
                                AlertSeverity.ERROR: 40,   # ERROR
                                AlertSeverity.CRITICAL: 50 # CRITICAL
                            }.get(alert.severity, 30),
                            f"Alert triggered: {alert.name} - {alert.description}",
                            extra={
                                'alert_name': alert.name,
                                'severity': alert.severity.value,
                                'description': alert.description,
                                'trigger_count': alert.trigger_count
                            }
                        )
                    
                    elif alert.active and not alert.should_trigger(monitoring_system):
                        # Alert condition no longer met, resolve it
                        alert.resolve()
                        self.logger.info(f"Alert resolved: {alert.name}")
                
                except Exception as e:
                    self.logger.error(f"Alert check failed for {name}: {str(e)}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        with self._lock:
            return [
                {
                    'name': alert.name,
                    'severity': alert.severity.value,
                    'description': alert.description,
                    'last_triggered': alert.last_triggered,
                    'trigger_count': alert.trigger_count
                }
                for alert in self.alerts.values()
                if alert.active
            ]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        with self._lock:
            return list(self.alert_history)[-limit:]


class MonitoringSystem:
    """
    Comprehensive monitoring system for quantum task planning.
    
    Integrates metrics collection, health checking, and alerting
    for complete observability of the quantum planning system.
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        
        # Monitoring state
        self.start_time = time.time()
        self.monitoring_active = False
        
        self.logger = get_logger()
    
    def start_monitoring(self):
        """Start all monitoring subsystems."""
        if not self.monitoring_active:
            self.health_checker.start_background_checking()
            self.alert_manager.start_background_checking(self)
            self.monitoring_active = True
            
            self.logger.info("Monitoring system started")
            
            # Record system start metric
            self.metrics_collector.increment_counter("system_starts")
            self.metrics_collector.set_gauge("system_uptime", 0)
    
    def stop_monitoring(self):
        """Stop all monitoring subsystems."""
        if self.monitoring_active:
            self.health_checker.stop_background_checking()
            self.alert_manager.stop_background_checking()
            self.monitoring_active = False
            
            uptime = time.time() - self.start_time
            self.metrics_collector.set_gauge("system_uptime", uptime)
            
            self.logger.info(f"Monitoring system stopped (uptime: {uptime:.1f}s)")
    
    def monitor_task_operation(
        self,
        operation_name: str,
        task: Optional[QuantumTask] = None,
        **labels
    ):
        """Context manager for monitoring task operations."""
        return self._TaskOperationMonitor(self, operation_name, task, labels)
    
    class _TaskOperationMonitor:
        """Context manager for monitoring task operations."""
        
        def __init__(self, monitoring_system, operation_name, task, labels):
            self.monitoring_system = monitoring_system
            self.operation_name = operation_name
            self.task = task
            self.labels = labels
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            
            # Record operation start
            self.monitoring_system.metrics_collector.increment_counter(
                f"{self.operation_name}_started",
                labels=self.labels
            )
            
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.time() - self.start_time
                
                # Record operation completion
                if exc_type is None:
                    # Success
                    self.monitoring_system.metrics_collector.increment_counter(
                        f"{self.operation_name}_completed",
                        labels=self.labels
                    )
                else:
                    # Failure
                    self.monitoring_system.metrics_collector.increment_counter(
                        f"{self.operation_name}_failed",
                        labels={**self.labels, 'error_type': exc_type.__name__}
                    )
                
                # Record timing
                self.monitoring_system.metrics_collector.record_timing(
                    f"{self.operation_name}_duration",
                    duration,
                    labels=self.labels
                )
                
                # Task-specific metrics
                if self.task:
                    self.monitoring_system.metrics_collector.set_gauge(
                        "task_priority",
                        self.task.priority,
                        labels={**self.labels, 'task_id': self.task.id}
                    )
                    
                    self.monitoring_system.metrics_collector.set_gauge(
                        "task_duration_estimate",
                        self.task.estimated_duration,
                        labels={**self.labels, 'task_id': self.task.id}
                    )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            'timestamp': current_time,
            'uptime_seconds': uptime,
            'monitoring_active': self.monitoring_active,
            'health': self.health_checker.get_health_status(),
            'active_alerts': self.alert_manager.get_active_alerts(),
            'metrics_summary': self._get_metrics_summary(),
            'system_resources': self._get_system_resources()
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        aggregates = self.metrics_collector.get_aggregates()
        
        return {
            'total_metric_types': len(aggregates),
            'total_data_points': sum(agg.get('count', 0) for agg in aggregates.values()),
            'last_metric_time': max(
                (agg.get('last_timestamp', 0) for agg in aggregates.values()),
                default=0
            )
        }
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource information."""
        try:
            import psutil
            
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory information
            memory = psutil.virtual_memory()
            
            # Disk information
            disk = psutil.disk_usage('/')
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'total_gb': memory.total / 1024 / 1024 / 1024,
                    'available_gb': memory.available / 1024 / 1024 / 1024,
                    'percent': memory.percent
                },
                'disk': {
                    'total_gb': disk.total / 1024 / 1024 / 1024,
                    'free_gb': disk.free / 1024 / 1024 / 1024,
                    'percent': (disk.used / disk.total) * 100
                }
            }
        except ImportError:
            return {'error': 'psutil not available for system monitoring'}
    
    def export_metrics(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export metrics in specified format."""
        metrics = self.metrics_collector.get_metrics()
        aggregates = self.metrics_collector.get_aggregates()
        
        export_data = {
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time,
            'raw_metrics': metrics,
            'aggregated_metrics': aggregates,
            'health_status': self.health_checker.get_health_status(),
            'active_alerts': self.alert_manager.get_active_alerts()
        }
        
        if format.lower() == 'json':
            return json.dumps(export_data, indent=2, default=str)
        else:
            return export_data


# Global monitoring system instance
_global_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> MonitoringSystem:
    """Get global monitoring system instance."""
    global _global_monitoring_system
    
    if _global_monitoring_system is None:
        _global_monitoring_system = MonitoringSystem()
    
    return _global_monitoring_system


def monitor_operation(operation_name: str, task: Optional[QuantumTask] = None, **labels):
    """Decorator for monitoring operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitoring_system = get_monitoring_system()
            
            with monitoring_system.monitor_task_operation(operation_name, task, **labels):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator
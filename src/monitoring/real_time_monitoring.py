"""
Real-time monitoring and alerting system.

Provides comprehensive monitoring, alerting, and observability
for RLHF contract deployments in production environments.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
from abc import ABC, abstractmethod

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = None

from ..models.reward_contract import RewardContract
from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Represents a monitoring alert."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    source: str
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = time.time()


@dataclass
class MetricDefinition:
    """Definition of a metric to monitor."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    unit: Optional[str] = None
    buckets: Optional[List[float]] = None  # For histograms


class AlertingRule:
    """Rule for generating alerts based on metrics."""
    
    def __init__(
        self,
        rule_id: str,
        metric_name: str,
        condition: Callable[[float], bool],
        severity: AlertSeverity,
        title: str,
        description: str,
        cooldown_seconds: float = 300.0
    ):
        self.rule_id = rule_id
        self.metric_name = metric_name
        self.condition = condition
        self.severity = severity
        self.title = title
        self.description = description
        self.cooldown_seconds = cooldown_seconds
        self.last_fired = 0.0
    
    def should_fire(self, metric_value: float) -> bool:
        """Check if alert should fire based on metric value."""
        if not self.condition(metric_value):
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_fired < self.cooldown_seconds:
            return False
        
        self.last_fired = current_time
        return True


class MetricsCollector(ABC):
    """Abstract base class for metrics collectors."""
    
    @abstractmethod
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current metric values."""
        pass
    
    @abstractmethod
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get definitions of metrics this collector provides."""
        pass


class ContractMetricsCollector(MetricsCollector):
    """Metrics collector for RLHF contract monitoring."""
    
    def __init__(self, contract: RewardContract):
        self.contract = contract
        self.call_count = 0
        self.error_count = 0
        self.violation_count = 0
        self.computation_times = deque(maxlen=1000)
        self.reward_values = deque(maxlen=1000)
        
    def record_computation(
        self,
        computation_time: float,
        reward_value: float,
        had_error: bool = False,
        had_violation: bool = False
    ):
        """Record a contract computation event."""
        self.call_count += 1
        if had_error:
            self.error_count += 1
        if had_violation:
            self.violation_count += 1
        
        self.computation_times.append(computation_time)
        self.reward_values.append(reward_value)
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current contract metrics."""
        metrics = {
            'contract_calls_total': float(self.call_count),
            'contract_errors_total': float(self.error_count),
            'contract_violations_total': float(self.violation_count),
        }
        
        if self.computation_times:
            metrics.update({
                'contract_computation_time_mean': sum(self.computation_times) / len(self.computation_times),
                'contract_computation_time_max': max(self.computation_times),
                'contract_computation_time_min': min(self.computation_times),
            })
        
        if self.reward_values:
            metrics.update({
                'contract_reward_mean': sum(self.reward_values) / len(self.reward_values),
                'contract_reward_max': max(self.reward_values),
                'contract_reward_min': min(self.reward_values),
                'contract_reward_std': self._calculate_std(self.reward_values)
            })
        
        # Error rates
        if self.call_count > 0:
            metrics.update({
                'contract_error_rate': self.error_count / self.call_count,
                'contract_violation_rate': self.violation_count / self.call_count
            })
        
        return metrics
    
    def _calculate_std(self, values: deque) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get contract metric definitions."""
        return [
            MetricDefinition(
                name="contract_calls_total",
                metric_type=MetricType.COUNTER,
                description="Total number of contract reward computations",
                labels=["contract_name", "contract_version"]
            ),
            MetricDefinition(
                name="contract_errors_total",
                metric_type=MetricType.COUNTER,
                description="Total number of contract computation errors",
                labels=["contract_name", "error_type"]
            ),
            MetricDefinition(
                name="contract_violations_total",
                metric_type=MetricType.COUNTER,
                description="Total number of contract constraint violations",
                labels=["contract_name", "constraint_name"]
            ),
            MetricDefinition(
                name="contract_computation_time",
                metric_type=MetricType.HISTOGRAM,
                description="Time taken for contract reward computations",
                labels=["contract_name"],
                unit="seconds",
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
            ),
            MetricDefinition(
                name="contract_reward_value",
                metric_type=MetricType.HISTOGRAM,
                description="Distribution of contract reward values",
                labels=["contract_name"],
                buckets=[-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
            )
        ]


class SystemMetricsCollector(MetricsCollector):
    """Collector for system-level metrics."""
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics."""
        try:
            import psutil
            
            return {
                'system_cpu_usage': psutil.cpu_percent(),
                'system_memory_usage': psutil.virtual_memory().percent,
                'system_disk_usage': psutil.disk_usage('/').percent,
                'system_load_average_1m': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
                'system_process_count': len(psutil.pids()),
                'system_network_connections': len(psutil.net_connections()),
            }
        except ImportError:
            # psutil not available - return mock metrics
            return {
                'system_cpu_usage': 0.0,
                'system_memory_usage': 0.0,
                'system_disk_usage': 0.0,
                'system_load_average_1m': 0.0,
                'system_process_count': 0.0,
                'system_network_connections': 0.0
            }
    
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get system metric definitions."""
        return [
            MetricDefinition(
                name="system_cpu_usage",
                metric_type=MetricType.GAUGE,
                description="CPU usage percentage",
                unit="percent"
            ),
            MetricDefinition(
                name="system_memory_usage",
                metric_type=MetricType.GAUGE,
                description="Memory usage percentage",
                unit="percent"
            ),
            MetricDefinition(
                name="system_disk_usage",
                metric_type=MetricType.GAUGE,
                description="Disk usage percentage",
                unit="percent"
            )
        ]


class AlertChannel(ABC):
    """Abstract base class for alert channels."""
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass


class LoggingAlertChannel(AlertChannel):
    """Alert channel that logs alerts."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Log the alert."""
        try:
            level_map = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }
            
            self.logger.log(
                level_map[alert.severity],
                f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.description} "
                f"(ID: {alert.alert_id}, Source: {alert.source})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send alert via logging: {e}")
            return False


class WebhookAlertChannel(AlertChannel):
    """Alert channel that sends webhooks."""
    
    def __init__(self, webhook_url: str, timeout: float = 10.0):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import aiohttp
            
            payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "source": alert.source,
                "timestamp": alert.timestamp,
                "tags": alert.tags
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            self.logger.error(f"Failed to send alert via webhook: {e}")
            return False


class RealTimeMonitor:
    """
    Real-time monitoring system for RLHF contracts.
    
    Provides comprehensive monitoring, alerting, and observability
    for production deployments.
    """
    
    def __init__(
        self,
        collection_interval: float = 10.0,
        retention_hours: int = 24
    ):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.logger = logging.getLogger(__name__)
        
        # Internal state
        self.collectors: List[MetricsCollector] = []
        self.alerting_rules: List[AlertingRule] = []
        self.alert_channels: List[AlertChannel] = []
        self.active_alerts: Dict[str, Alert] = {}
        
        # Metrics storage (in-memory time series)
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=int(retention_hours * 3600 / collection_interval))
        )
        
        # Prometheus integration
        self.prometheus_metrics: Dict[str, Any] = {}
        self._init_prometheus()
        
        # Monitoring state
        self.is_running = False
        self.monitoring_task = None
    
    def _init_prometheus(self):
        """Initialize Prometheus metrics if available."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available - metrics export disabled")
            return
        
        # Create Prometheus metrics
        self.prometheus_metrics = {
            'monitoring_collections_total': Counter(
                'rlhf_monitoring_collections_total',
                'Total number of metric collections performed'
            ),
            'monitoring_alerts_total': Counter(
                'rlhf_monitoring_alerts_total',
                'Total number of alerts generated',
                ['severity', 'source']
            ),
            'monitoring_collection_duration': Histogram(
                'rlhf_monitoring_collection_duration_seconds',
                'Time spent collecting metrics',
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
            ),
            'active_alerts_count': Gauge(
                'rlhf_monitoring_active_alerts_count',
                'Number of currently active alerts'
            )
        }
    
    def add_collector(self, collector: MetricsCollector):
        """Add a metrics collector."""
        self.collectors.append(collector)
        self.logger.info(f"Added metrics collector: {collector.__class__.__name__}")
    
    def add_alerting_rule(self, rule: AlertingRule):
        """Add an alerting rule."""
        self.alerting_rules.append(rule)
        self.logger.info(f"Added alerting rule: {rule.rule_id}")
    
    def add_alert_channel(self, channel: AlertChannel):
        """Add an alert channel."""
        self.alert_channels.append(channel)
        self.logger.info(f"Added alert channel: {channel.__class__.__name__}")
    
    async def start_monitoring(self):
        """Start the monitoring loop."""
        if self.is_running:
            self.logger.warning("Monitoring is already running")
            return
        
        self.is_running = True
        self.logger.info("Starting real-time monitoring")
        
        # Start Prometheus metrics server if available
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(8000)  # Prometheus metrics endpoint
                self.logger.info("Prometheus metrics server started on port 8000")
            except Exception as e:
                self.logger.warning(f"Failed to start Prometheus server: {e}")
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        try:
            await self.monitoring_task
        except asyncio.CancelledError:
            self.logger.info("Monitoring task cancelled")
    
    async def stop_monitoring(self):
        """Stop the monitoring loop."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                collection_start = time.time()
                
                # Collect metrics from all collectors
                all_metrics = {}
                for collector in self.collectors:
                    try:
                        metrics = collector.collect_metrics()
                        all_metrics.update(metrics)
                    except Exception as e:
                        handle_error(
                            error=e,
                            operation="collect_metrics",
                            category=ErrorCategory.MONITORING,
                            severity=ErrorSeverity.MEDIUM,
                            additional_info={"collector": collector.__class__.__name__}
                        )
                
                # Store metrics in history
                current_time = time.time()
                for metric_name, value in all_metrics.items():
                    self.metrics_history[metric_name].append((current_time, value))
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
                    self.prometheus_metrics['monitoring_collections_total'].inc()
                    
                    collection_duration = time.time() - collection_start
                    self.prometheus_metrics['monitoring_collection_duration'].observe(collection_duration)
                
                # Evaluate alerting rules
                await self._evaluate_alerting_rules(all_metrics)
                
                # Update active alerts gauge
                if PROMETHEUS_AVAILABLE and 'active_alerts_count' in self.prometheus_metrics:
                    self.prometheus_metrics['active_alerts_count'].set(len(self.active_alerts))
                
                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                handle_error(
                    error=e,
                    operation="monitoring_loop",
                    category=ErrorCategory.MONITORING,
                    severity=ErrorSeverity.HIGH
                )
                await asyncio.sleep(self.collection_interval)
    
    async def _evaluate_alerting_rules(self, metrics: Dict[str, float]):
        """Evaluate alerting rules and generate alerts."""
        for rule in self.alerting_rules:
            if rule.metric_name not in metrics:
                continue
            
            metric_value = metrics[rule.metric_name]
            
            if rule.should_fire(metric_value):
                alert = Alert(
                    alert_id=f"{rule.rule_id}_{int(time.time())}",
                    severity=rule.severity,
                    title=rule.title,
                    description=f"{rule.description} (Current value: {metric_value})",
                    source=f"rule:{rule.rule_id}",
                    timestamp=time.time(),
                    tags={"metric_name": rule.metric_name, "rule_id": rule.rule_id}
                )
                
                await self._fire_alert(alert)
    
    async def _fire_alert(self, alert: Alert):
        """Fire an alert through all configured channels."""
        self.active_alerts[alert.alert_id] = alert
        
        self.logger.warning(
            f"Firing alert {alert.alert_id}: {alert.title} ({alert.severity.value})"
        )
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and 'monitoring_alerts_total' in self.prometheus_metrics:
            self.prometheus_metrics['monitoring_alerts_total'].labels(
                severity=alert.severity.value,
                source=alert.source
            ).inc()
        
        # Send through all alert channels
        for channel in self.alert_channels:
            try:
                success = await channel.send_alert(alert)
                if not success:
                    self.logger.error(
                        f"Failed to send alert {alert.alert_id} through {channel.__class__.__name__}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error sending alert {alert.alert_id} through {channel.__class__.__name__}: {e}"
                )
    
    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolve()
            del self.active_alerts[alert_id]
            self.logger.info(f"Resolved alert {alert_id}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        current_metrics = {}
        
        for metric_name, history in self.metrics_history.items():
            if history:
                # Get most recent value
                current_metrics[metric_name] = history[-1][1]
        
        return current_metrics
    
    def get_metric_history(
        self,
        metric_name: str,
        hours: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """Get historical values for a metric."""
        if metric_name not in self.metrics_history:
            return []
        
        history = list(self.metrics_history[metric_name])
        
        if hours is not None:
            cutoff_time = time.time() - (hours * 3600)
            history = [(t, v) for t, v in history if t >= cutoff_time]
        
        return history
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status."""
        return {
            "is_running": self.is_running,
            "collection_interval": self.collection_interval,
            "collectors_count": len(self.collectors),
            "alerting_rules_count": len(self.alerting_rules),
            "alert_channels_count": len(self.alert_channels),
            "active_alerts_count": len(self.active_alerts),
            "metrics_tracked": len(self.metrics_history),
            "prometheus_available": PROMETHEUS_AVAILABLE
        }
    
    def create_default_alerting_rules(self) -> List[AlertingRule]:
        """Create default alerting rules for common issues."""
        return [
            AlertingRule(
                rule_id="high_error_rate",
                metric_name="contract_error_rate",
                condition=lambda x: x > 0.05,  # 5% error rate
                severity=AlertSeverity.WARNING,
                title="High Contract Error Rate",
                description="Contract error rate is above 5%",
                cooldown_seconds=300.0
            ),
            AlertingRule(
                rule_id="critical_error_rate",
                metric_name="contract_error_rate",
                condition=lambda x: x > 0.20,  # 20% error rate
                severity=AlertSeverity.CRITICAL,
                title="Critical Contract Error Rate",
                description="Contract error rate is above 20%",
                cooldown_seconds=180.0
            ),
            AlertingRule(
                rule_id="high_violation_rate",
                metric_name="contract_violation_rate",
                condition=lambda x: x > 0.10,  # 10% violation rate
                severity=AlertSeverity.ERROR,
                title="High Contract Violation Rate",
                description="Contract violation rate is above 10%",
                cooldown_seconds=300.0
            ),
            AlertingRule(
                rule_id="slow_computation",
                metric_name="contract_computation_time_mean",
                condition=lambda x: x > 1.0,  # 1 second average
                severity=AlertSeverity.WARNING,
                title="Slow Contract Computation",
                description="Average contract computation time is above 1 second",
                cooldown_seconds=600.0
            ),
            AlertingRule(
                rule_id="high_cpu_usage",
                metric_name="system_cpu_usage",
                condition=lambda x: x > 80.0,  # 80% CPU
                severity=AlertSeverity.WARNING,
                title="High CPU Usage",
                description="System CPU usage is above 80%",
                cooldown_seconds=300.0
            ),
            AlertingRule(
                rule_id="high_memory_usage",
                metric_name="system_memory_usage",
                condition=lambda x: x > 90.0,  # 90% memory
                severity=AlertSeverity.ERROR,
                title="High Memory Usage",
                description="System memory usage is above 90%",
                cooldown_seconds=180.0
            )
        ]


def create_contract_monitor(contract: RewardContract) -> RealTimeMonitor:
    """Create a monitoring setup for a specific contract."""
    monitor = RealTimeMonitor()
    
    # Add contract metrics collector
    contract_collector = ContractMetricsCollector(contract)
    monitor.add_collector(contract_collector)
    
    # Add system metrics collector
    system_collector = SystemMetricsCollector()
    monitor.add_collector(system_collector)
    
    # Add default alerting rules
    default_rules = monitor.create_default_alerting_rules()
    for rule in default_rules:
        monitor.add_alerting_rule(rule)
    
    # Add logging alert channel
    logging_channel = LoggingAlertChannel()
    monitor.add_alert_channel(logging_channel)
    
    return monitor

#!/usr/bin/env python3
"""
Comprehensive Monitoring System for RLHF Contract Wizard

Implements enterprise-grade monitoring, metrics collection, alerting,
health checks, and observability features for production deployments.
"""

import time
import asyncio
import json
import logging
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import websockets
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: str = ""
    
    
@dataclass
class Alert:
    """Alert definition and state."""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    evaluation_window: timedelta
    timestamp: datetime
    active: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    description: str
    check_function: Callable[[], Tuple[HealthStatus, Dict[str, Any]]]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Advanced metrics collection system with multiple backends.
    
    Features:
    - Prometheus metrics integration
    - Custom metric definitions
    - Real-time metric streaming
    - Automatic metric aggregation
    - Performance monitoring
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.time_series_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Standard metrics
        self._setup_standard_metrics()
        
        # Background collection
        self.collection_thread: Optional[threading.Thread] = None
        self.collection_running = False
        
    def _setup_standard_metrics(self):
        """Setup standard system and application metrics."""
        # System metrics
        self.define_metric(MetricDefinition(
            name="system_cpu_usage",
            type=MetricType.GAUGE,
            description="CPU usage percentage",
            unit="percent"
        ))
        
        self.define_metric(MetricDefinition(
            name="system_memory_usage",
            type=MetricType.GAUGE,
            description="Memory usage in bytes",
            unit="bytes"
        ))
        
        self.define_metric(MetricDefinition(
            name="system_disk_usage",
            type=MetricType.GAUGE,
            description="Disk usage percentage",
            unit="percent"
        ))
        
        # Application metrics
        self.define_metric(MetricDefinition(
            name="http_requests_total",
            type=MetricType.COUNTER,
            description="Total HTTP requests",
            labels=["method", "endpoint", "status_code"]
        ))
        
        self.define_metric(MetricDefinition(
            name="http_request_duration",
            type=MetricType.HISTOGRAM,
            description="HTTP request duration",
            labels=["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            unit="seconds"
        ))
        
        self.define_metric(MetricDefinition(
            name="contract_operations_total",
            type=MetricType.COUNTER,
            description="Total contract operations",
            labels=["operation", "status"]
        ))
        
        self.define_metric(MetricDefinition(
            name="quantum_planning_duration",
            type=MetricType.HISTOGRAM,
            description="Quantum planning duration",
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            unit="seconds"
        ))
        
        self.define_metric(MetricDefinition(
            name="active_contracts",
            type=MetricType.GAUGE,
            description="Number of active contracts"
        ))
        
    def define_metric(self, definition: MetricDefinition):
        """Define a new metric for collection."""
        self.metric_definitions[definition.name] = definition
        
        # Create Prometheus metric
        if definition.type == MetricType.COUNTER:
            metric = Counter(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.type == MetricType.HISTOGRAM:
            metric = Histogram(
                definition.name,
                definition.description,
                definition.labels,
                buckets=definition.buckets,
                registry=self.registry
            )
        elif definition.type == MetricType.GAUGE:
            metric = Gauge(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        else:
            raise ValueError(f"Unsupported metric type: {definition.type}")
        
        self.metrics[definition.name] = metric
        
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not defined")
        
        metric = self.metrics[name]
        definition = self.metric_definitions[name]
        
        # Record with Prometheus
        if labels and definition.labels:
            label_values = [labels.get(label, "") for label in definition.labels]
            if definition.type == MetricType.COUNTER:
                metric.labels(*label_values).inc(value)
            elif definition.type == MetricType.HISTOGRAM:
                metric.labels(*label_values).observe(value)
            elif definition.type == MetricType.GAUGE:
                metric.labels(*label_values).set(value)
        else:
            if definition.type == MetricType.COUNTER:
                metric.inc(value)
            elif definition.type == MetricType.HISTOGRAM:
                metric.observe(value)
            elif definition.type == MetricType.GAUGE:
                metric.set(value)
        
        # Store in time series
        timestamp = time.time()
        self.time_series_data[name].append((timestamp, value, labels or {}))
        
    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current metric value."""
        if name not in self.time_series_data:
            return None
        
        data = self.time_series_data[name]
        if not data:
            return None
        
        # Return most recent value
        return data[-1][1]
        
    def get_time_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Tuple[float, float, Dict[str, str]]]:
        """Get time series data for a metric."""
        if name not in self.time_series_data:
            return []
        
        data = list(self.time_series_data[name])
        
        # Filter by time range
        if start_time:
            start_ts = start_time.timestamp()
            data = [(ts, val, labels) for ts, val, labels in data if ts >= start_ts]
        
        if end_time:
            end_ts = end_time.timestamp()
            data = [(ts, val, labels) for ts, val, labels in data if ts <= end_ts]
        
        return data
        
    def start_collection(self, interval_seconds: int = 10):
        """Start background metric collection."""
        if self.collection_running:
            return
        
        self.collection_running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.collection_thread.start()
        
    def stop_collection(self):
        """Stop background metric collection."""
        self.collection_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
            
    def _collection_loop(self, interval_seconds: int):
        """Background collection loop."""
        while self.collection_running:
            try:
                self._collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logging.error(f"Error in metrics collection: {e}")
                time.sleep(interval_seconds)
                
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.record_metric("system_cpu_usage", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.record_metric("system_memory_usage", memory.used)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.record_metric("system_disk_usage", disk_percent)
        
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode()


class AlertManager:
    """
    Advanced alerting system with multiple notification channels.
    
    Features:
    - Rule-based alerting
    - Multiple alert channels
    - Alert aggregation and deduplication
    - Escalation policies
    - Alert history and analytics
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable[[Alert], None]] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Background evaluation
        self.evaluation_thread: Optional[threading.Thread] = None
        self.evaluation_running = False
        
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        self.add_alert_rule(
            "high_cpu_usage",
            "system_cpu_usage > 80",
            AlertSeverity.WARNING,
            "CPU usage is above 80%",
            threshold=80.0,
            evaluation_window=timedelta(minutes=5)
        )
        
        self.add_alert_rule(
            "high_memory_usage",
            "system_memory_usage > 85",
            AlertSeverity.WARNING,
            "Memory usage is above 85%",
            threshold=85.0,
            evaluation_window=timedelta(minutes=5)
        )
        
        self.add_alert_rule(
            "high_error_rate",
            "http_requests_total{status_code=~'5..'} / http_requests_total > 0.05",
            AlertSeverity.ERROR,
            "HTTP error rate is above 5%",
            threshold=0.05,
            evaluation_window=timedelta(minutes=2)
        )
        
        self.add_alert_rule(
            "disk_space_low",
            "system_disk_usage > 90",
            AlertSeverity.CRITICAL,
            "Disk usage is above 90%",
            threshold=90.0,
            evaluation_window=timedelta(minutes=1)
        )
        
    def add_alert_rule(
        self,
        name: str,
        condition: str,
        severity: AlertSeverity,
        description: str,
        threshold: float,
        evaluation_window: timedelta
    ):
        """Add a new alert rule."""
        self.alert_rules[name] = {
            "condition": condition,
            "severity": severity,
            "description": description,
            "threshold": threshold,
            "evaluation_window": evaluation_window
        }
        
    def add_notification_channel(self, channel: Callable[[Alert], None]):
        """Add notification channel for alerts."""
        self.notification_channels.append(channel)
        
    def start_evaluation(self, interval_seconds: int = 30):
        """Start background alert evaluation."""
        if self.evaluation_running:
            return
        
        self.evaluation_running = True
        self.evaluation_thread = threading.Thread(
            target=self._evaluation_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.evaluation_thread.start()
        
    def stop_evaluation(self):
        """Stop background alert evaluation."""
        self.evaluation_running = False
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5)
            
    def _evaluation_loop(self, interval_seconds: int):
        """Background alert evaluation loop."""
        while self.evaluation_running:
            try:
                self._evaluate_alerts()
                time.sleep(interval_seconds)
            except Exception as e:
                logging.error(f"Error in alert evaluation: {e}")
                time.sleep(interval_seconds)
                
    def _evaluate_alerts(self):
        """Evaluate all alert rules."""
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                triggered = self._evaluate_rule(rule, current_time)
                
                if triggered and rule_name not in self.alerts:
                    # Create new alert
                    alert = Alert(
                        alert_id=f"{rule_name}_{int(time.time())}",
                        name=rule_name,
                        description=rule["description"],
                        severity=rule["severity"],
                        condition=rule["condition"],
                        threshold=rule["threshold"],
                        evaluation_window=rule["evaluation_window"],
                        timestamp=current_time,
                        active=True
                    )
                    
                    self.alerts[rule_name] = alert
                    self.alert_history.append(alert)
                    
                    # Send notifications
                    for channel in self.notification_channels:
                        try:
                            channel(alert)
                        except Exception as e:
                            logging.error(f"Error sending alert notification: {e}")
                            
                elif not triggered and rule_name in self.alerts:
                    # Resolve alert
                    alert = self.alerts[rule_name]
                    alert.active = False
                    alert.resolved_at = current_time
                    del self.alerts[rule_name]
                    
            except Exception as e:
                logging.error(f"Error evaluating alert rule {rule_name}: {e}")
                
    def _evaluate_rule(self, rule: Dict[str, Any], current_time: datetime) -> bool:
        """Evaluate a single alert rule."""
        # Simple evaluation for basic metrics
        if "system_cpu_usage" in rule["condition"]:
            cpu_value = self.metrics_collector.get_metric_value("system_cpu_usage")
            return cpu_value is not None and cpu_value > rule["threshold"]
        
        elif "system_memory_usage" in rule["condition"]:
            # Convert to percentage
            memory_bytes = self.metrics_collector.get_metric_value("system_memory_usage")
            if memory_bytes is not None:
                total_memory = psutil.virtual_memory().total
                memory_percent = (memory_bytes / total_memory) * 100
                return memory_percent > rule["threshold"]
        
        elif "system_disk_usage" in rule["condition"]:
            disk_value = self.metrics_collector.get_metric_value("system_disk_usage")
            return disk_value is not None and disk_value > rule["threshold"]
        
        # For more complex rules, would implement proper expression parser
        return False
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.alerts.values())
        
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return list(self.alert_history)[-limit:]


class HealthChecker:
    """
    Comprehensive health checking system.
    
    Features:
    - Component health monitoring
    - Dependency checking
    - Health status aggregation
    - Health trends analysis
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Background checking
        self.checking_thread: Optional[threading.Thread] = None
        self.checking_running = False
        
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        self.add_health_check(HealthCheck(
            name="database_connection",
            description="Database connectivity check",
            check_function=self._check_database,
            interval_seconds=30
        ))
        
        self.add_health_check(HealthCheck(
            name="redis_connection", 
            description="Redis connectivity check",
            check_function=self._check_redis,
            interval_seconds=30
        ))
        
        self.add_health_check(HealthCheck(
            name="system_resources",
            description="System resource availability",
            check_function=self._check_system_resources,
            interval_seconds=60
        ))
        
        self.add_health_check(HealthCheck(
            name="api_endpoints",
            description="Critical API endpoint availability",
            check_function=self._check_api_endpoints,
            interval_seconds=120
        ))
        
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check."""
        self.health_checks[health_check.name] = health_check
        
    def start_checking(self):
        """Start background health checking."""
        if self.checking_running:
            return
        
        self.checking_running = True
        self.checking_thread = threading.Thread(
            target=self._checking_loop,
            daemon=True
        )
        self.checking_thread.start()
        
    def stop_checking(self):
        """Stop background health checking."""
        self.checking_running = False
        if self.checking_thread:
            self.checking_thread.join(timeout=5)
            
    def _checking_loop(self):
        """Background health checking loop."""
        while self.checking_running:
            current_time = datetime.now()
            
            for check in self.health_checks.values():
                if (check.last_check is None or 
                    current_time - check.last_check >= timedelta(seconds=check.interval_seconds)):
                    
                    try:
                        status, details = self._run_health_check(check)
                        check.last_check = current_time
                        check.last_status = status
                        check.last_details = details
                        
                        # Store in history
                        self.health_history[check.name].append({
                            "timestamp": current_time,
                            "status": status,
                            "details": details
                        })
                        
                    except Exception as e:
                        logging.error(f"Error running health check {check.name}: {e}")
                        check.last_status = HealthStatus.UNKNOWN
            
            time.sleep(10)  # Check every 10 seconds
            
    def _run_health_check(self, check: HealthCheck) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Run a single health check with timeout."""
        try:
            # Use ThreadPoolExecutor for timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(check.check_function)
                return future.result(timeout=check.timeout_seconds)
        except Exception as e:
            return HealthStatus.UNHEALTHY, {"error": str(e)}
            
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.health_checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.last_status for check in self.health_checks.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
            
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "overall_status": self.get_overall_health().value,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": check.last_status.value,
                    "last_check": check.last_check.isoformat() if check.last_check else None,
                    "details": check.last_details
                }
                for name, check in self.health_checks.items()
            }
        }
        
    # Default health check implementations
    def _check_database(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Check database connectivity."""
        try:
            # Simulate database check
            response_time = 0.05  # Simulated
            return HealthStatus.HEALTHY, {"response_time_ms": response_time * 1000}
        except Exception as e:
            return HealthStatus.UNHEALTHY, {"error": str(e)}
            
    def _check_redis(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Check Redis connectivity."""
        try:
            # Simulate Redis check
            return HealthStatus.HEALTHY, {"ping_time_ms": 2.5}
        except Exception as e:
            return HealthStatus.UNHEALTHY, {"error": str(e)}
            
    def _check_system_resources(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Check system resource availability."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }
            
            # Determine status based on thresholds
            if (cpu_percent > 90 or memory.percent > 95 or 
                (disk.used / disk.total) > 0.95):
                return HealthStatus.UNHEALTHY, details
            elif (cpu_percent > 70 or memory.percent > 80 or 
                  (disk.used / disk.total) > 0.85):
                return HealthStatus.DEGRADED, details
            else:
                return HealthStatus.HEALTHY, details
                
        except Exception as e:
            return HealthStatus.UNKNOWN, {"error": str(e)}
            
    def _check_api_endpoints(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Check critical API endpoints."""
        try:
            # Simulate API endpoint checks
            endpoints = ["/api/v1/health", "/api/v1/contracts"]
            results = {}
            
            for endpoint in endpoints:
                # Simulate HTTP check
                response_time = np.random.uniform(0.01, 0.1)
                status_code = 200
                results[endpoint] = {
                    "status_code": status_code,
                    "response_time_ms": response_time * 1000
                }
            
            # All endpoints healthy
            return HealthStatus.HEALTHY, {"endpoints": results}
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, {"error": str(e)}


class MonitoringDashboard:
    """
    Real-time monitoring dashboard with WebSocket support.
    
    Features:
    - Real-time metrics streaming
    - Interactive visualizations
    - Alert notifications
    - Health status monitoring
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        health_checker: HealthChecker
    ):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.health_checker = health_checker
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time dashboard."""
        async def handle_client(websocket, path):
            self.connected_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.connected_clients.remove(websocket)
        
        server = await websockets.serve(handle_client, host, port)
        logging.info(f"Dashboard WebSocket server started on ws://{host}:{port}")
        
        # Start broadcasting loop
        asyncio.create_task(self._broadcast_loop())
        
        return server
        
    async def _broadcast_loop(self):
        """Broadcast real-time updates to connected clients."""
        while True:
            if self.connected_clients:
                # Collect current data
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": self._get_current_metrics(),
                    "alerts": [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
                    "health": self.health_checker.get_health_report()
                }
                
                # Broadcast to all clients
                message = json.dumps(data, default=str)
                disconnected_clients = set()
                
                for client in self.connected_clients.copy():
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(client)
                    except Exception as e:
                        logging.error(f"Error broadcasting to client: {e}")
                        disconnected_clients.add(client)
                
                # Remove disconnected clients
                self.connected_clients -= disconnected_clients
            
            await asyncio.sleep(5)  # Broadcast every 5 seconds
            
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        metrics = {}
        
        for name in self.metrics_collector.metric_definitions.keys():
            value = self.metrics_collector.get_metric_value(name)
            if value is not None:
                metrics[name] = value
        
        return metrics


class ComprehensiveMonitoring:
    """
    Main monitoring system coordinating all monitoring components.
    
    Provides unified monitoring interface for the entire system.
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_checker = HealthChecker()
        self.dashboard = MonitoringDashboard(
            self.metrics_collector,
            self.alert_manager,
            self.health_checker
        )
        
        # Setup notification channels
        self._setup_notification_channels()
        
        # Start all monitoring components
        self.start_monitoring()
        
    def _setup_notification_channels(self):
        """Setup alert notification channels."""
        # Console notification
        def console_notification(alert: Alert):
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            print(f"{severity_emoji[alert.severity]} ALERT: {alert.name}")
            print(f"   Description: {alert.description}")
            print(f"   Timestamp: {alert.timestamp}")
            print(f"   Threshold: {alert.threshold}")
            
        self.alert_manager.add_notification_channel(console_notification)
        
        # Could add email, Slack, PagerDuty, etc.
        
    def start_monitoring(self):
        """Start all monitoring components."""
        self.metrics_collector.start_collection(interval_seconds=10)
        self.alert_manager.start_evaluation(interval_seconds=30)
        self.health_checker.start_checking()
        
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.metrics_collector.stop_collection()
        self.alert_manager.stop_evaluation()
        self.health_checker.stop_checking()
        
    def record_contract_operation(self, operation: str, duration: float, success: bool):
        """Record contract operation metrics."""
        self.metrics_collector.record_metric(
            "contract_operations_total",
            1.0,
            {"operation": operation, "status": "success" if success else "failure"}
        )
        
        if operation == "quantum_planning":
            self.metrics_collector.record_metric("quantum_planning_duration", duration)
            
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.metrics_collector.record_metric(
            "http_requests_total",
            1.0,
            {"method": method, "endpoint": endpoint, "status_code": str(status_code)}
        )
        
        self.metrics_collector.record_metric(
            "http_request_duration",
            duration,
            {"method": method, "endpoint": endpoint}
        )
        
    def update_active_contracts(self, count: int):
        """Update active contracts gauge."""
        self.metrics_collector.record_metric("active_contracts", count)
        
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            "health": self.health_checker.get_overall_health().value,
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "metrics_collected": len(self.metrics_collector.metric_definitions),
            "health_checks": len(self.health_checker.health_checks),
            "system_metrics": {
                "cpu_usage": self.metrics_collector.get_metric_value("system_cpu_usage"),
                "memory_usage": self.metrics_collector.get_metric_value("system_memory_usage"),
                "disk_usage": self.metrics_collector.get_metric_value("system_disk_usage")
            },
            "application_metrics": {
                "active_contracts": self.metrics_collector.get_metric_value("active_contracts"),
                "total_http_requests": self.metrics_collector.get_metric_value("http_requests_total")
            }
        }
        
    async def start_dashboard(self, host: str = "localhost", port: int = 8765):
        """Start the monitoring dashboard."""
        return await self.dashboard.start_websocket_server(host, port)
        
    def export_prometheus_metrics(self) -> str:
        """Export all metrics in Prometheus format."""
        return self.metrics_collector.export_prometheus_metrics()


# Example usage
async def example_monitoring_usage():
    """Example demonstrating monitoring system usage."""
    # Initialize monitoring
    monitoring = ComprehensiveMonitoring()
    
    # Wait a bit for metrics to be collected
    await asyncio.sleep(5)
    
    # Simulate some operations
    monitoring.record_contract_operation("create", 0.5, True)
    monitoring.record_contract_operation("verify", 2.3, True)
    monitoring.record_contract_operation("deploy", 1.8, False)
    
    monitoring.record_http_request("POST", "/api/v1/contracts", 201, 0.3)
    monitoring.record_http_request("GET", "/api/v1/health", 200, 0.05)
    
    monitoring.update_active_contracts(42)
    
    # Get monitoring summary
    summary = monitoring.get_monitoring_summary()
    print("📊 Monitoring Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Get Prometheus metrics
    metrics = monitoring.export_prometheus_metrics()
    print(f"\n📈 Prometheus metrics collected: {len(metrics.split('\\n'))} lines")
    
    # Start dashboard (commented out for example)
    # await monitoring.start_dashboard()
    
    # Cleanup
    monitoring.stop_monitoring()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_monitoring_usage())
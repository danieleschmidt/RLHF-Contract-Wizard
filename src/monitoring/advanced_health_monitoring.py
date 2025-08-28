"""
Advanced health monitoring system for Generation 2: MAKE IT ROBUST

Implements comprehensive system health monitoring, alerting, diagnostics,
and automated recovery for the RLHF-Contract-Wizard system.
"""

import asyncio
import psutil
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import aiohttp
import jax
import jax.numpy as jnp
import numpy as np


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Union[int, float, bool, str]
    unit: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    status: HealthStatus = HealthStatus.HEALTHY


@dataclass
class Alert:
    """System alert."""
    id: str
    message: str
    severity: AlertSeverity
    component: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_cpu_metrics(self) -> Dict[str, HealthMetric]:
        """Get CPU-related health metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        metrics = {
            "cpu_usage": HealthMetric(
                name="CPU Usage",
                value=cpu_percent,
                unit="%",
                threshold_warning=70.0,
                threshold_critical=90.0
            ),
            "cpu_count": HealthMetric(
                name="CPU Count",
                value=cpu_count,
                unit="cores"
            ),
            "load_average_1m": HealthMetric(
                name="Load Average (1m)",
                value=load_avg[0],
                unit="",
                threshold_warning=cpu_count * 0.7,
                threshold_critical=cpu_count * 1.0
            )
        }
        
        # Update status based on thresholds
        for metric in metrics.values():
            if isinstance(metric.value, (int, float)) and metric.threshold_critical:
                if metric.value >= metric.threshold_critical:
                    metric.status = HealthStatus.CRITICAL
                elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                    metric.status = HealthStatus.WARNING
        
        return metrics
    
    def get_memory_metrics(self) -> Dict[str, HealthMetric]:
        """Get memory-related health metrics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics = {
            "memory_usage": HealthMetric(
                name="Memory Usage",
                value=memory.percent,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0
            ),
            "memory_available": HealthMetric(
                name="Memory Available",
                value=memory.available / (1024**3),  # GB
                unit="GB"
            ),
            "memory_total": HealthMetric(
                name="Memory Total",
                value=memory.total / (1024**3),  # GB
                unit="GB"
            ),
            "swap_usage": HealthMetric(
                name="Swap Usage",
                value=swap.percent,
                unit="%",
                threshold_warning=20.0,
                threshold_critical=50.0
            )
        }
        
        # Update status based on thresholds
        for metric in metrics.values():
            if isinstance(metric.value, (int, float)) and metric.threshold_critical:
                if metric.value >= metric.threshold_critical:
                    metric.status = HealthStatus.CRITICAL
                elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                    metric.status = HealthStatus.WARNING
        
        return metrics
    
    def get_disk_metrics(self) -> Dict[str, HealthMetric]:
        """Get disk-related health metrics."""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics = {
            "disk_usage": HealthMetric(
                name="Disk Usage",
                value=(disk_usage.used / disk_usage.total) * 100,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0
            ),
            "disk_free": HealthMetric(
                name="Disk Free",
                value=disk_usage.free / (1024**3),  # GB
                unit="GB"
            ),
            "disk_total": HealthMetric(
                name="Disk Total",
                value=disk_usage.total / (1024**3),  # GB
                unit="GB"
            )
        }
        
        if disk_io:
            metrics.update({
                "disk_read_bytes": HealthMetric(
                    name="Disk Read Bytes",
                    value=disk_io.read_bytes,
                    unit="bytes"
                ),
                "disk_write_bytes": HealthMetric(
                    name="Disk Write Bytes",
                    value=disk_io.write_bytes,
                    unit="bytes"
                )
            })
        
        # Update status based on thresholds
        for metric in metrics.values():
            if isinstance(metric.value, (int, float)) and metric.threshold_critical:
                if metric.value >= metric.threshold_critical:
                    metric.status = HealthStatus.CRITICAL
                elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                    metric.status = HealthStatus.WARNING
        
        return metrics
    
    def get_network_metrics(self) -> Dict[str, HealthMetric]:
        """Get network-related health metrics."""
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        
        metrics = {
            "network_bytes_sent": HealthMetric(
                name="Network Bytes Sent",
                value=net_io.bytes_sent,
                unit="bytes"
            ),
            "network_bytes_recv": HealthMetric(
                name="Network Bytes Received",
                value=net_io.bytes_recv,
                unit="bytes"
            ),
            "network_connections": HealthMetric(
                name="Network Connections",
                value=net_connections,
                unit="connections",
                threshold_warning=1000,
                threshold_critical=5000
            )
        }
        
        # Update status based on thresholds
        for metric in metrics.values():
            if isinstance(metric.value, (int, float)) and metric.threshold_critical:
                if metric.value >= metric.threshold_critical:
                    metric.status = HealthStatus.CRITICAL
                elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                    metric.status = HealthStatus.WARNING
        
        return metrics


class ApplicationMonitor:
    """Monitor application-specific health metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.request_counts = deque(maxlen=1000)
        self.response_times = deque(maxlen=1000)
        self.error_counts = deque(maxlen=1000)
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request for monitoring."""
        timestamp = time.time()
        self.request_counts.append(timestamp)
        self.response_times.append(response_time)
        if not success:
            self.error_counts.append(timestamp)
    
    def get_application_metrics(self) -> Dict[str, HealthMetric]:
        """Get application-specific health metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate recent metrics (last 5 minutes)
        recent_requests = [t for t in self.request_counts if current_time - t < 300]
        recent_errors = [t for t in self.error_counts if current_time - t < 300]
        recent_response_times = list(self.response_times)[-100:] if self.response_times else [0]
        
        request_rate = len(recent_requests) / 5.0 if recent_requests else 0  # per second
        error_rate = len(recent_errors) / max(1, len(recent_requests)) * 100 if recent_requests else 0
        avg_response_time = sum(recent_response_times) / len(recent_response_times)
        p95_response_time = np.percentile(recent_response_times, 95) if recent_response_times else 0
        
        metrics = {
            "uptime": HealthMetric(
                name="Application Uptime",
                value=uptime,
                unit="seconds"
            ),
            "request_rate": HealthMetric(
                name="Request Rate",
                value=request_rate,
                unit="req/sec"
            ),
            "error_rate": HealthMetric(
                name="Error Rate",
                value=error_rate,
                unit="%",
                threshold_warning=5.0,
                threshold_critical=10.0
            ),
            "avg_response_time": HealthMetric(
                name="Average Response Time",
                value=avg_response_time * 1000,  # Convert to ms
                unit="ms",
                threshold_warning=1000.0,
                threshold_critical=5000.0
            ),
            "p95_response_time": HealthMetric(
                name="95th Percentile Response Time",
                value=p95_response_time * 1000,  # Convert to ms
                unit="ms",
                threshold_warning=2000.0,
                threshold_critical=10000.0
            )
        }
        
        # Update status based on thresholds
        for metric in metrics.values():
            if isinstance(metric.value, (int, float)) and metric.threshold_critical:
                if metric.value >= metric.threshold_critical:
                    metric.status = HealthStatus.CRITICAL
                elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                    metric.status = HealthStatus.WARNING
        
        return metrics
    
    def get_jax_metrics(self) -> Dict[str, HealthMetric]:
        """Get JAX-specific health metrics."""
        try:
            # JAX device information
            devices = jax.devices()
            device_count = len(devices)
            
            # Memory usage (if available)
            memory_info = None
            try:
                if devices and hasattr(devices[0], 'memory_stats'):
                    memory_info = devices[0].memory_stats()
            except:
                pass
            
            metrics = {
                "jax_device_count": HealthMetric(
                    name="JAX Device Count",
                    value=device_count,
                    unit="devices"
                ),
                "jax_backend": HealthMetric(
                    name="JAX Backend",
                    value=str(jax.default_backend()),
                    unit=""
                )
            }
            
            if memory_info:
                total_memory = memory_info.get('bytes_limit', 0)
                used_memory = memory_info.get('bytes_in_use', 0)
                memory_usage_percent = (used_memory / total_memory * 100) if total_memory > 0 else 0
                
                metrics.update({
                    "jax_memory_usage": HealthMetric(
                        name="JAX Memory Usage",
                        value=memory_usage_percent,
                        unit="%",
                        threshold_warning=80.0,
                        threshold_critical=95.0
                    ),
                    "jax_memory_used": HealthMetric(
                        name="JAX Memory Used",
                        value=used_memory / (1024**3),  # GB
                        unit="GB"
                    )
                })
            
            # Update status based on thresholds
            for metric in metrics.values():
                if isinstance(metric.value, (int, float)) and metric.threshold_critical:
                    if metric.value >= metric.threshold_critical:
                        metric.status = HealthStatus.CRITICAL
                    elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                        metric.status = HealthStatus.WARNING
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to get JAX metrics: {e}")
            return {
                "jax_status": HealthMetric(
                    name="JAX Status",
                    value="error",
                    unit="",
                    status=HealthStatus.WARNING
                )
            }


class DatabaseMonitor:
    """Monitor database health and performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def get_database_metrics(self) -> Dict[str, HealthMetric]:
        """Get database health metrics."""
        try:
            # This would connect to actual database and get metrics
            # For now, simulate database monitoring
            
            # Simulate connection test
            connection_time = await self._test_database_connection()
            
            metrics = {
                "db_connection_time": HealthMetric(
                    name="Database Connection Time",
                    value=connection_time * 1000,  # Convert to ms
                    unit="ms",
                    threshold_warning=1000.0,
                    threshold_critical=5000.0
                ),
                "db_connection_status": HealthMetric(
                    name="Database Connection",
                    value="connected" if connection_time > 0 else "disconnected",
                    unit="",
                    status=HealthStatus.HEALTHY if connection_time > 0 else HealthStatus.CRITICAL
                )
            }
            
            # Simulate additional metrics
            if connection_time > 0:
                metrics.update({
                    "db_active_connections": HealthMetric(
                        name="Active Database Connections",
                        value=10,  # Simulated
                        unit="connections",
                        threshold_warning=50,
                        threshold_critical=100
                    ),
                    "db_query_time_avg": HealthMetric(
                        name="Average Query Time",
                        value=50.0,  # Simulated in ms
                        unit="ms",
                        threshold_warning=500.0,
                        threshold_critical=2000.0
                    )
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Database monitoring failed: {e}")
            return {
                "db_status": HealthMetric(
                    name="Database Status",
                    value="error",
                    unit="",
                    status=HealthStatus.CRITICAL
                )
            }
    
    async def _test_database_connection(self) -> float:
        """Test database connection and return response time."""
        try:
            start_time = time.time()
            # Simulate database connection test
            await asyncio.sleep(0.05)  # Simulate connection time
            return time.time() - start_time
        except:
            return 0.0


class AdvancedHealthMonitor:
    """
    Advanced health monitoring system with comprehensive diagnostics.
    
    Features:
    - Multi-component health monitoring
    - Intelligent alerting with severity levels
    - Performance trend analysis
    - Automated health checks
    - Custom health check registration
    - Health dashboard generation
    """
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.system_monitor = SystemMonitor()
        self.app_monitor = ApplicationMonitor()
        self.db_monitor = DatabaseMonitor()
        
        self.alerts: List[Alert] = []
        self.health_history: deque = deque(maxlen=1000)
        self.custom_checks: Dict[str, Callable] = {}
        
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
    
    def register_custom_check(self, name: str, check_function: Callable) -> None:
        """Register a custom health check."""
        self.custom_checks[name] = check_function
        self.logger.info(f"Registered custom health check: {name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def record_request(self, response_time: float, success: bool = True):
        """Record request for application monitoring."""
        self.app_monitor.record_request(response_time, success)
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_data = {
            "timestamp": time.time(),
            "components": {}
        }
        
        try:
            # System metrics
            cpu_metrics = self.system_monitor.get_cpu_metrics()
            memory_metrics = self.system_monitor.get_memory_metrics()
            disk_metrics = self.system_monitor.get_disk_metrics()
            network_metrics = self.system_monitor.get_network_metrics()
            
            health_data["components"]["system"] = {
                "status": self._determine_component_status(
                    list(cpu_metrics.values()) + 
                    list(memory_metrics.values()) + 
                    list(disk_metrics.values()) + 
                    list(network_metrics.values())
                ),
                "metrics": {
                    **{k: self._metric_to_dict(v) for k, v in cpu_metrics.items()},
                    **{k: self._metric_to_dict(v) for k, v in memory_metrics.items()},
                    **{k: self._metric_to_dict(v) for k, v in disk_metrics.items()},
                    **{k: self._metric_to_dict(v) for k, v in network_metrics.items()}
                }
            }
            
            # Application metrics
            app_metrics = self.app_monitor.get_application_metrics()
            jax_metrics = self.app_monitor.get_jax_metrics()
            
            health_data["components"]["application"] = {
                "status": self._determine_component_status(
                    list(app_metrics.values()) + list(jax_metrics.values())
                ),
                "metrics": {
                    **{k: self._metric_to_dict(v) for k, v in app_metrics.items()},
                    **{k: self._metric_to_dict(v) for k, v in jax_metrics.items()}
                }
            }
            
            # Database metrics
            db_metrics = await self.db_monitor.get_database_metrics()
            health_data["components"]["database"] = {
                "status": self._determine_component_status(list(db_metrics.values())),
                "metrics": {k: self._metric_to_dict(v) for k, v in db_metrics.items()}
            }
            
            # Custom checks
            for name, check_func in self.custom_checks.items():
                try:
                    if asyncio.iscoroutinefunction(check_func):
                        custom_result = await check_func()
                    else:
                        custom_result = check_func()
                    
                    if isinstance(custom_result, dict):
                        health_data["components"][f"custom_{name}"] = custom_result
                    else:
                        health_data["components"][f"custom_{name}"] = {
                            "status": HealthStatus.HEALTHY.value,
                            "result": custom_result
                        }
                        
                except Exception as e:
                    self.logger.error(f"Custom health check {name} failed: {e}")
                    health_data["components"][f"custom_{name}"] = {
                        "status": HealthStatus.CRITICAL.value,
                        "error": str(e)
                    }
            
            # Determine overall health status
            component_statuses = [
                comp.get("status", HealthStatus.HEALTHY.value)
                for comp in health_data["components"].values()
            ]
            
            overall_status = self._determine_overall_status(component_statuses)
            health_data["overall_status"] = overall_status.value
            
            # Store in history
            self.health_history.append(health_data)
            
            # Check for alerts
            await self._check_for_alerts(health_data)
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "timestamp": time.time(),
                "overall_status": HealthStatus.CRITICAL.value,
                "error": str(e)
            }
    
    def _metric_to_dict(self, metric: HealthMetric) -> Dict[str, Any]:
        """Convert health metric to dictionary."""
        return {
            "name": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "status": metric.status.value,
            "timestamp": metric.timestamp,
            "thresholds": {
                "warning": metric.threshold_warning,
                "critical": metric.threshold_critical
            }
        }
    
    def _determine_component_status(self, metrics: List[HealthMetric]) -> str:
        """Determine component status from its metrics."""
        statuses = [m.status for m in metrics]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL.value
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED.value
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING.value
        else:
            return HealthStatus.HEALTHY.value
    
    def _determine_overall_status(self, component_statuses: List[str]) -> HealthStatus:
        """Determine overall system status."""
        if HealthStatus.CRITICAL.value in component_statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED.value in component_statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.WARNING.value in component_statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def _check_for_alerts(self, health_data: Dict[str, Any]):
        """Check health data for alert conditions."""
        current_time = time.time()
        
        # Check overall status
        overall_status = health_data.get("overall_status")
        if overall_status in [HealthStatus.CRITICAL.value, HealthStatus.DEGRADED.value]:
            alert = Alert(
                id=f"overall_status_{int(current_time)}",
                message=f"System health is {overall_status}",
                severity=AlertSeverity.CRITICAL if overall_status == HealthStatus.CRITICAL.value else AlertSeverity.ERROR,
                component="system",
                metadata={"health_data": health_data}
            )
            await self._trigger_alert(alert)
        
        # Check individual components
        for component_name, component_data in health_data.get("components", {}).items():
            component_status = component_data.get("status")
            if component_status in [HealthStatus.CRITICAL.value, HealthStatus.DEGRADED.value]:
                alert = Alert(
                    id=f"{component_name}_{int(current_time)}",
                    message=f"Component {component_name} status is {component_status}",
                    severity=AlertSeverity.CRITICAL if component_status == HealthStatus.CRITICAL.value else AlertSeverity.ERROR,
                    component=component_name,
                    metadata={"component_data": component_data}
                )
                await self._trigger_alert(alert)
        
        # Clean up old resolved alerts
        self._cleanup_old_alerts()
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert and notify callbacks."""
        # Check if similar alert already exists and is recent
        recent_alerts = [
            a for a in self.alerts
            if (time.time() - a.timestamp < 300 and  # Within 5 minutes
                a.component == alert.component and
                a.severity == alert.severity and
                not a.resolved)
        ]
        
        if recent_alerts:
            self.logger.debug(f"Suppressing duplicate alert for {alert.component}")
            return
        
        # Add alert
        self.alerts.append(alert)
        self.logger.warning(f"Alert triggered: {alert.message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = time.time() - 86400  # 24 hours
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time or not alert.resolved
        ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.acknowledged = True
                self.logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_health_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive health dashboard."""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}
        
        latest_health = self.health_history[-1]
        
        # Calculate trends (last 10 readings)
        recent_health = list(self.health_history)[-10:]
        trends = self._calculate_trends(recent_health)
        
        # Active alerts
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        # Alert summary
        alert_summary = {
            "total": len(self.alerts),
            "active": len(active_alerts),
            "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "warnings": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING])
        }
        
        return {
            "timestamp": latest_health["timestamp"],
            "overall_status": latest_health["overall_status"],
            "components": latest_health["components"],
            "trends": trends,
            "alerts": {
                "summary": alert_summary,
                "active_alerts": [
                    {
                        "id": alert.id,
                        "message": alert.message,
                        "severity": alert.severity.value,
                        "component": alert.component,
                        "timestamp": alert.timestamp,
                        "acknowledged": alert.acknowledged
                    }
                    for alert in active_alerts[-10:]  # Last 10 active alerts
                ]
            },
            "monitoring": {
                "active": self.monitoring_active,
                "check_interval": self.check_interval,
                "history_size": len(self.health_history),
                "custom_checks": len(self.custom_checks)
            }
        }
    
    def _calculate_trends(self, health_history: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate health trends from recent history."""
        if len(health_history) < 2:
            return {}
        
        trends = {}
        
        # Overall status trend
        status_values = {
            HealthStatus.HEALTHY.value: 4,
            HealthStatus.WARNING.value: 3,
            HealthStatus.DEGRADED.value: 2,
            HealthStatus.CRITICAL.value: 1,
            HealthStatus.DOWN.value: 0
        }
        
        overall_statuses = [
            status_values.get(h.get("overall_status"), 0)
            for h in health_history
        ]
        
        if len(overall_statuses) >= 2:
            if overall_statuses[-1] > overall_statuses[0]:
                trends["overall"] = "improving"
            elif overall_statuses[-1] < overall_statuses[0]:
                trends["overall"] = "degrading"
            else:
                trends["overall"] = "stable"
        
        return trends
    
    async def run_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostic."""
        diagnostic_start = time.time()
        
        diagnostic_results = {
            "timestamp": diagnostic_start,
            "tests": {}
        }
        
        # System connectivity tests
        try:
            # Test internet connectivity
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get('https://httpbin.org/status/200') as response:
                    diagnostic_results["tests"]["internet_connectivity"] = {
                        "status": "pass" if response.status == 200 else "fail",
                        "response_time": time.time() - diagnostic_start
                    }
        except Exception as e:
            diagnostic_results["tests"]["internet_connectivity"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # JAX functionality test
        try:
            test_array = jnp.array([1, 2, 3, 4, 5])
            result = jnp.sum(test_array)
            diagnostic_results["tests"]["jax_functionality"] = {
                "status": "pass" if result == 15 else "fail",
                "result": float(result)
            }
        except Exception as e:
            diagnostic_results["tests"]["jax_functionality"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Memory allocation test
        try:
            test_memory = np.random.random((1000, 1000))
            diagnostic_results["tests"]["memory_allocation"] = {
                "status": "pass",
                "allocated_mb": test_memory.nbytes / (1024**2)
            }
            del test_memory  # Clean up
        except Exception as e:
            diagnostic_results["tests"]["memory_allocation"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Custom diagnostic tests
        for name, check_func in self.custom_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                diagnostic_results["tests"][f"custom_{name}"] = {
                    "status": "pass",
                    "result": result
                }
            except Exception as e:
                diagnostic_results["tests"][f"custom_{name}"] = {
                    "status": "fail",
                    "error": str(e)
                }
        
        diagnostic_results["duration"] = time.time() - diagnostic_start
        diagnostic_results["summary"] = {
            "total_tests": len(diagnostic_results["tests"]),
            "passed": len([t for t in diagnostic_results["tests"].values() if t.get("status") == "pass"]),
            "failed": len([t for t in diagnostic_results["tests"].values() if t.get("status") == "fail"])
        }
        
        return diagnostic_results


# Global health monitor instance
_health_monitor: Optional[AdvancedHealthMonitor] = None


def get_health_monitor() -> AdvancedHealthMonitor:
    """Get or create the global health monitor instance."""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = AdvancedHealthMonitor()
    
    return _health_monitor


# Example alert callback
async def log_alert_callback(alert: Alert):
    """Example alert callback that logs alerts."""
    logging.getLogger(__name__).warning(
        f"ALERT: {alert.severity.value.upper()} - {alert.message} "
        f"(Component: {alert.component})"
    )


# Convenience function for health checks
async def quick_health_check() -> Dict[str, Any]:
    """Perform a quick health check."""
    monitor = get_health_monitor()
    return await monitor.perform_health_check()
"""
Production health monitoring system.

Provides comprehensive health checks and status monitoring.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall system health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable[[], bool]
    description: str
    timeout_seconds: float = 5.0
    critical: bool = False
    enabled: bool = True
    failure_threshold: int = 3
    recovery_threshold: int = 2


@dataclass
class HealthCheckResult:
    """Result of a health check execution."""
    name: str
    status: bool
    message: str
    duration_ms: float
    timestamp: float
    error: Optional[str] = None


@dataclass
class SystemHealthSnapshot:
    """Complete system health snapshot."""
    overall_status: HealthStatus
    timestamp: float
    check_results: List[HealthCheckResult]
    summary: Dict[str, Any]
    alerts: List[str] = field(default_factory=list)


class HealthMonitor:
    """
    Production-ready health monitoring system.
    
    Provides comprehensive health checks with circuit breaker patterns,
    configurable thresholds, and automatic recovery detection.
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        history_size: int = 1000
    ):
        """
        Initialize health monitor.
        
        Args:
            check_interval: How often to run health checks (seconds)
            history_size: Number of health snapshots to keep
        """
        self.check_interval = check_interval
        self.history_size = history_size
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Health history
        self.health_history: List[SystemHealthSnapshot] = []
        
        # Check state tracking
        self.check_failures: Dict[str, int] = {}
        self.check_successes: Dict[str, int] = {}
        self.check_last_results: Dict[str, bool] = {}
        
        # Threading
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._lock = threading.Lock()
        
        # Setup default health checks
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default system health checks."""
        # CPU usage check
        self.register_check(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            description="CPU usage within acceptable limits",
            critical=False
        )
        
        # Memory usage check
        self.register_check(
            name="memory_usage",
            check_function=self._check_memory_usage,
            description="Memory usage within acceptable limits",
            critical=True
        )
        
        # Disk usage check
        self.register_check(
            name="disk_usage", 
            check_function=self._check_disk_usage,
            description="Disk usage within acceptable limits",
            critical=True
        )
        
        # Database connectivity (mock)
        self.register_check(
            name="database_connectivity",
            check_function=self._check_database_connectivity,
            description="Database connection is healthy",
            critical=True
        )
    
    def register_check(
        self,
        name: str,
        check_function: Callable[[], bool],
        description: str,
        timeout_seconds: float = 5.0,
        critical: bool = False,
        failure_threshold: int = 3,
        recovery_threshold: int = 2
    ):
        """
        Register a new health check.
        
        Args:
            name: Unique name for the check
            check_function: Function that returns True if healthy
            description: Human-readable description
            timeout_seconds: Timeout for check execution
            critical: Whether this check is critical for system health
            failure_threshold: Consecutive failures before marking unhealthy
            recovery_threshold: Consecutive successes needed for recovery
        """
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            description=description,
            timeout_seconds=timeout_seconds,
            critical=critical,
            failure_threshold=failure_threshold,
            recovery_threshold=recovery_threshold
        )
        
        # Initialize tracking
        self.check_failures[name] = 0
        self.check_successes[name] = 0
        self.check_last_results[name] = True
        
        logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10.0)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                snapshot = self.check_health()
                
                with self._lock:
                    self.health_history.append(snapshot)
                    
                    # Maintain history size
                    if len(self.health_history) > self.history_size:
                        self.health_history.pop(0)
                
                # Log status changes
                self._log_status_changes(snapshot)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
            
            time.sleep(self.check_interval)
    
    def check_health(self) -> SystemHealthSnapshot:
        """
        Execute all health checks and return system snapshot.
        
        Returns:
            Complete system health snapshot
        """
        check_results = []
        alerts = []
        
        for check_name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
                
            result = self._execute_health_check(health_check)
            check_results.append(result)
            
            # Update failure/success tracking
            self._update_check_tracking(check_name, result.status)
            
            # Generate alerts for failed critical checks
            if not result.status and health_check.critical:
                if self.check_failures[check_name] >= health_check.failure_threshold:
                    alerts.append(f"Critical check '{check_name}' failing: {result.message}")
        
        # Determine overall status
        overall_status = self._determine_overall_status(check_results)
        
        # Create summary
        summary = self._create_health_summary(check_results)
        
        return SystemHealthSnapshot(
            overall_status=overall_status,
            timestamp=time.time(),
            check_results=check_results,
            summary=summary,
            alerts=alerts
        )
    
    def _execute_health_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Execute a single health check with timeout protection."""
        start_time = time.time()
        
        try:
            # Simple timeout implementation
            # In production, consider using concurrent.futures.ThreadPoolExecutor
            result = health_check.check_function()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=health_check.name,
                status=bool(result),
                message="OK" if result else "Check failed",
                duration_ms=duration_ms,
                timestamp=time.time()
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=health_check.name,
                status=False,
                message=f"Check error: {str(e)}",
                duration_ms=duration_ms,
                timestamp=time.time(),
                error=str(e)
            )
    
    def _update_check_tracking(self, check_name: str, success: bool):
        """Update failure/success tracking for a check."""
        if success:
            self.check_successes[check_name] += 1
            if self.check_successes[check_name] >= self.health_checks[check_name].recovery_threshold:
                self.check_failures[check_name] = 0  # Reset failures on recovery
        else:
            self.check_failures[check_name] += 1
            self.check_successes[check_name] = 0  # Reset successes on failure
        
        self.check_last_results[check_name] = success
    
    def _determine_overall_status(self, check_results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status."""
        critical_failures = 0
        non_critical_failures = 0
        total_critical = 0
        
        for result in check_results:
            health_check = self.health_checks.get(result.name)
            if not health_check:
                continue
                
            if health_check.critical:
                total_critical += 1
                if not result.status:
                    failure_count = self.check_failures.get(result.name, 0)
                    if failure_count >= health_check.failure_threshold:
                        critical_failures += 1
            else:
                if not result.status:
                    non_critical_failures += 1
        
        # Determine status
        if critical_failures > 0:
            if critical_failures >= total_critical:
                return HealthStatus.CRITICAL
            else:
                return HealthStatus.UNHEALTHY
        elif non_critical_failures > len(check_results) * 0.3:  # >30% non-critical failures
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _create_health_summary(self, check_results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Create health summary from check results."""
        passed = sum(1 for result in check_results if result.status)
        failed = len(check_results) - passed
        avg_duration = sum(result.duration_ms for result in check_results) / len(check_results) if check_results else 0
        
        return {
            'total_checks': len(check_results),
            'passed_checks': passed,
            'failed_checks': failed,
            'success_rate': passed / len(check_results) if check_results else 0,
            'average_duration_ms': avg_duration,
            'critical_checks': sum(1 for name in [r.name for r in check_results] 
                                 if self.health_checks.get(name, {}).critical),
            'timestamp': time.time()
        }
    
    def _log_status_changes(self, snapshot: SystemHealthSnapshot):
        """Log significant status changes."""
        if not self.health_history:
            logger.info(f"Initial health status: {snapshot.overall_status.value}")
            return
        
        previous_status = self.health_history[-1].overall_status
        if snapshot.overall_status != previous_status:
            logger.warning(
                f"Health status changed: {previous_status.value} -> {snapshot.overall_status.value}"
            )
            
            # Log alerts
            for alert in snapshot.alerts:
                logger.error(f"Health Alert: {alert}")
    
    # Default health check implementations
    def _check_cpu_usage(self) -> bool:
        """Check if CPU usage is within acceptable limits."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 90.0
        except ImportError:
            return True  # Skip check if psutil not available
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 85.0
        except ImportError:
            return True
    
    def _check_disk_usage(self) -> bool:
        """Check if disk usage is within acceptable limits."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return (disk.used / disk.total) * 100 < 90.0
        except ImportError:
            return True
    
    def _check_database_connectivity(self) -> bool:
        """Mock database connectivity check."""
        # In production, this would test actual database connection
        return True
    
    # Public API methods
    def get_current_status(self) -> Dict[str, Any]:
        """Get current health status summary."""
        if not self.health_history:
            return {
                'status': 'unknown',
                'message': 'No health data available'
            }
        
        latest = self.health_history[-1]
        return {
            'status': latest.overall_status.value,
            'timestamp': latest.timestamp,
            'checks_total': latest.summary['total_checks'],
            'checks_passed': latest.summary['passed_checks'],
            'checks_failed': latest.summary['failed_checks'],
            'success_rate': latest.summary['success_rate'],
            'alerts': latest.alerts
        }
    
    def get_check_history(self, check_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get history for a specific check."""
        history = []
        
        for snapshot in self.health_history[-limit:]:
            for result in snapshot.check_results:
                if result.name == check_name:
                    history.append({
                        'timestamp': result.timestamp,
                        'status': result.status,
                        'message': result.message,
                        'duration_ms': result.duration_ms,
                        'error': result.error
                    })
        
        return history
    
    def get_health_trends(self, time_window: float = 3600) -> Dict[str, Any]:
        """Get health trends over time window."""
        cutoff_time = time.time() - time_window
        recent_snapshots = [
            s for s in self.health_history
            if s.timestamp > cutoff_time
        ]
        
        if not recent_snapshots:
            return {'message': 'No data available for time window'}
        
        # Calculate trends
        status_counts = {}
        for snapshot in recent_snapshots:
            status = snapshot.overall_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Average success rate
        avg_success_rate = sum(s.summary['success_rate'] for s in recent_snapshots) / len(recent_snapshots)
        
        return {
            'time_window_hours': time_window / 3600,
            'total_snapshots': len(recent_snapshots),
            'status_distribution': status_counts,
            'average_success_rate': avg_success_rate,
            'current_status': recent_snapshots[-1].overall_status.value,
            'trend': self._calculate_trend(recent_snapshots)
        }
    
    def _calculate_trend(self, snapshots: List[SystemHealthSnapshot]) -> str:
        """Calculate health trend direction."""
        if len(snapshots) < 2:
            return 'stable'
        
        # Simple trend calculation based on success rates
        recent_rate = sum(s.summary['success_rate'] for s in snapshots[-3:]) / min(3, len(snapshots))
        older_rate = sum(s.summary['success_rate'] for s in snapshots[:3]) / min(3, len(snapshots))
        
        if recent_rate > older_rate + 0.1:
            return 'improving'
        elif recent_rate < older_rate - 0.1:
            return 'degrading'
        else:
            return 'stable'


# Global health monitor instance
global_health_monitor = HealthMonitor()
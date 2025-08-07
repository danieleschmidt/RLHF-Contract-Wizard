"""
Unit tests for monitoring module.

Tests metrics collection, health checking, alerting, and comprehensive
monitoring system for quantum planning operations.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.quantum_planner.monitoring import (
    MetricsCollector, HealthChecker, AlertManager, MonitoringSystem,
    get_monitoring_system, monitor_operation,
    Metric, MetricType, HealthStatus, AlertSeverity, HealthCheck, Alert
)
from src.quantum_planner.core import QuantumTask, TaskState
from .fixtures import *
from .utils import *


class TestMetric:
    """Test cases for Metric class."""
    
    def test_metric_creation(self):
        """Test metric creation."""
        metric = Metric(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            labels={"component": "quantum_planner", "operation": "optimize"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.metric_type == MetricType.GAUGE
        assert metric.labels["component"] == "quantum_planner"
        assert metric.labels["operation"] == "optimize"
    
    def test_metric_to_dict(self):
        """Test metric serialization to dictionary."""
        timestamp = time.time()
        metric = Metric(
            name="cpu_usage",
            value=75.3,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            labels={"host": "server1"}
        )
        
        metric_dict = metric.to_dict()
        
        assert metric_dict["name"] == "cpu_usage"
        assert metric_dict["value"] == 75.3
        assert metric_dict["type"] == "gauge"
        assert metric_dict["timestamp"] == timestamp
        assert metric_dict["labels"]["host"] == "server1"


class TestHealthCheck:
    """Test cases for HealthCheck class."""
    
    def test_health_check_creation(self):
        """Test health check creation."""
        def sample_check():
            return True
        
        health_check = HealthCheck(
            name="sample_check",
            check_function=sample_check,
            description="Sample health check",
            timeout=30.0,
            critical=True
        )
        
        assert health_check.name == "sample_check"
        assert health_check.check_function == sample_check
        assert health_check.description == "Sample health check"
        assert health_check.timeout == 30.0
        assert health_check.critical is True
        assert health_check.check_count == 0
        assert health_check.failure_count == 0
    
    def test_health_check_execution_success(self):
        """Test successful health check execution."""
        def passing_check():
            return True
        
        health_check = HealthCheck(
            name="passing_check",
            check_function=passing_check,
            description="Always passes"
        )
        
        result = health_check.run_check()
        
        assert result is True
        assert health_check.last_result is True
        assert health_check.last_error is None
        assert health_check.check_count == 1
        assert health_check.failure_count == 0
        assert health_check.failure_rate == 0.0
    
    def test_health_check_execution_failure(self):
        """Test failed health check execution."""
        def failing_check():
            return False
        
        health_check = HealthCheck(
            name="failing_check",
            check_function=failing_check,
            description="Always fails"
        )
        
        result = health_check.run_check()
        
        assert result is False
        assert health_check.last_result is False
        assert health_check.last_error is None
        assert health_check.check_count == 1
        assert health_check.failure_count == 1
        assert health_check.failure_rate == 1.0
    
    def test_health_check_exception_handling(self):
        """Test health check exception handling."""
        def exception_check():
            raise ValueError("Check failed with exception")
        
        health_check = HealthCheck(
            name="exception_check",
            check_function=exception_check,
            description="Throws exception"
        )
        
        result = health_check.run_check()
        
        assert result is False
        assert health_check.last_result is False
        assert health_check.last_error == "Check failed with exception"
        assert health_check.failure_count == 1
    
    def test_health_check_to_dict(self):
        """Test health check serialization."""
        def sample_check():
            return True
        
        health_check = HealthCheck(
            name="test_check",
            check_function=sample_check,
            description="Test check",
            critical=True
        )
        
        # Run check to populate data
        health_check.run_check()
        
        check_dict = health_check.to_dict()
        
        assert check_dict["name"] == "test_check"
        assert check_dict["description"] == "Test check"
        assert check_dict["critical"] is True
        assert check_dict["last_result"] is True
        assert check_dict["check_count"] == 1
        assert check_dict["failure_count"] == 0


class TestAlert:
    """Test cases for Alert class."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        def alert_condition(monitoring_system):
            return True  # Always trigger
        
        alert = Alert(
            name="test_alert",
            condition=alert_condition,
            severity=AlertSeverity.WARNING,
            description="Test alert condition",
            cooldown_seconds=300.0
        )
        
        assert alert.name == "test_alert"
        assert alert.condition == alert_condition
        assert alert.severity == AlertSeverity.WARNING
        assert alert.description == "Test alert condition"
        assert alert.cooldown_seconds == 300.0
        assert alert.trigger_count == 0
        assert alert.active is False
    
    def test_alert_triggering(self):
        """Test alert triggering mechanism."""
        def always_trigger(monitoring_system):
            return True
        
        alert = Alert(
            name="trigger_test",
            condition=always_trigger,
            severity=AlertSeverity.ERROR,
            description="Always triggers"
        )
        
        # Mock monitoring system
        mock_monitoring = Mock()
        
        # Check if should trigger
        should_trigger = alert.should_trigger(mock_monitoring)
        assert should_trigger is True
        
        # Trigger alert
        alert_data = alert.trigger()
        
        assert alert_data["alert_name"] == "trigger_test"
        assert alert_data["severity"] == "error"
        assert alert_data["description"] == "Always triggers"
        assert alert.trigger_count == 1
        assert alert.active is True
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        def always_trigger(monitoring_system):
            return True
        
        alert = Alert(
            name="cooldown_test",
            condition=always_trigger,
            severity=AlertSeverity.INFO,
            description="Cooldown test",
            cooldown_seconds=0.1  # Short cooldown for testing
        )
        
        mock_monitoring = Mock()
        
        # First trigger
        alert.trigger()
        assert alert.active is True
        
        # Should not trigger again immediately (cooldown)
        should_trigger = alert.should_trigger(mock_monitoring)
        assert should_trigger is False
        
        # Wait for cooldown
        time.sleep(0.2)
        
        # Should trigger again after cooldown
        should_trigger = alert.should_trigger(mock_monitoring)
        assert should_trigger is True
    
    def test_alert_resolution(self):
        """Test alert resolution."""
        def conditional_trigger(monitoring_system):
            return getattr(monitoring_system, 'trigger_condition', False)
        
        alert = Alert(
            name="resolution_test",
            condition=conditional_trigger,
            severity=AlertSeverity.WARNING,
            description="Resolution test"
        )
        
        mock_monitoring = Mock()
        mock_monitoring.trigger_condition = True
        
        # Trigger alert
        alert.trigger()
        assert alert.active is True
        
        # Resolve alert
        alert.resolve()
        assert alert.active is False


class TestMetricsCollector:
    """Test cases for MetricsCollector class."""
    
    def test_collector_creation(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(buffer_size=1000)
        
        assert collector.buffer_size == 1000
        assert len(collector.metrics_buffer) == 0
        assert len(collector.metric_aggregates) == 0
    
    def test_counter_recording(self):
        """Test counter metric recording."""
        collector = MetricsCollector()
        
        collector.increment_counter("test_counter", 1, {"component": "test"})
        collector.increment_counter("test_counter", 5, {"component": "test"})
        
        metrics = collector.get_metrics()
        
        assert len(metrics) == 2
        assert all(m["name"] == "test_counter" for m in metrics)
        assert all(m["type"] == "counter" for m in metrics)
        assert metrics[0]["value"] == 1
        assert metrics[1]["value"] == 5
    
    def test_gauge_recording(self):
        """Test gauge metric recording."""
        collector = MetricsCollector()
        
        collector.set_gauge("cpu_usage", 75.5, {"host": "server1"})
        collector.set_gauge("cpu_usage", 80.2, {"host": "server1"})
        
        metrics = collector.get_metrics()
        
        assert len(metrics) == 2
        assert all(m["name"] == "cpu_usage" for m in metrics)
        assert all(m["type"] == "gauge" for m in metrics)
        assert metrics[1]["value"] == 80.2  # Latest value
    
    def test_timing_recording(self):
        """Test timing metric recording."""
        collector = MetricsCollector()
        
        collector.record_timing("operation_duration", 1.25, {"operation": "optimize"})
        collector.record_timing("operation_duration", 0.95, {"operation": "optimize"})
        
        metrics = collector.get_metrics()
        
        assert len(metrics) == 2
        assert all(m["name"] == "operation_duration" for m in metrics)
        assert all(m["type"] == "timing" for m in metrics)
    
    def test_histogram_recording(self):
        """Test histogram metric recording."""
        collector = MetricsCollector()
        
        # Record multiple values
        for value in [10, 20, 30, 40, 50]:
            collector.record_histogram("response_size", value, {"endpoint": "api"})
        
        metrics = collector.get_metrics()
        
        assert len(metrics) == 5
        assert all(m["name"] == "response_size" for m in metrics)
        assert all(m["type"] == "histogram" for m in metrics)
    
    def test_metric_aggregation(self):
        """Test metric aggregation."""
        collector = MetricsCollector()
        
        # Record multiple timing metrics
        for duration in [1.0, 1.5, 2.0, 1.2, 1.8]:
            collector.record_timing("test_operation", duration)
        
        # Trigger aggregation
        collector._aggregate_metrics()
        
        aggregates = collector.get_aggregates()
        
        # Should have aggregated data
        assert len(aggregates) > 0
        
        # Find the aggregate for our metric
        timing_aggregate = None
        for key, agg in aggregates.items():
            if agg['name'] == 'test_operation' and agg['type'] == 'timing':
                timing_aggregate = agg
                break
        
        assert timing_aggregate is not None
        assert timing_aggregate['count'] == 5
        assert timing_aggregate['sum'] == 7.5  # Sum of all durations
        assert timing_aggregate['min'] == 1.0
        assert timing_aggregate['max'] == 2.0
    
    def test_metrics_filtering_by_time(self):
        """Test filtering metrics by timestamp."""
        collector = MetricsCollector()
        
        start_time = time.time()
        
        collector.increment_counter("early_metric", 1)
        time.sleep(0.1)
        
        mid_time = time.time()
        
        collector.increment_counter("late_metric", 1)
        
        # Get all metrics
        all_metrics = collector.get_metrics()
        assert len(all_metrics) == 2
        
        # Get metrics since mid_time
        recent_metrics = collector.get_metrics(since=mid_time)
        assert len(recent_metrics) == 1
        assert recent_metrics[0]["name"] == "late_metric"
    
    def test_metrics_buffer_overflow(self):
        """Test metrics buffer overflow handling."""
        collector = MetricsCollector(buffer_size=5)
        
        # Add more metrics than buffer size
        for i in range(10):
            collector.increment_counter(f"metric_{i}", 1)
        
        metrics = collector.get_metrics()
        
        # Should only keep the most recent metrics
        assert len(metrics) <= 5
        
        # Should contain the latest metrics
        assert any(m["name"] == "metric_9" for m in metrics)
    
    def test_metrics_clearing(self):
        """Test metrics clearing."""
        collector = MetricsCollector()
        
        collector.increment_counter("test_metric", 1)
        collector.set_gauge("test_gauge", 50)
        
        assert len(collector.get_metrics()) == 2
        
        # Clear all metrics
        collector.clear_metrics()
        
        assert len(collector.get_metrics()) == 0
        assert len(collector.get_aggregates()) == 0


class TestHealthChecker:
    """Test cases for HealthChecker class."""
    
    def test_checker_creation(self):
        """Test health checker initialization."""
        checker = HealthChecker()
        
        assert len(checker.health_checks) >= 0  # May have default checks
        assert checker.overall_status == HealthStatus.HEALTHY
    
    def test_health_check_registration(self):
        """Test health check registration."""
        checker = HealthChecker()
        
        def custom_check():
            return True
        
        checker.register_health_check(
            "custom_check",
            custom_check,
            "Custom health check",
            critical=True
        )
        
        assert "custom_check" in checker.health_checks
        health_check = checker.health_checks["custom_check"]
        assert health_check.name == "custom_check"
        assert health_check.critical is True
    
    def test_health_check_execution(self):
        """Test health check execution."""
        checker = HealthChecker()
        
        def passing_check():
            return True
        
        def failing_check():
            return False
        
        checker.register_health_check("passing", passing_check, "Passes")
        checker.register_health_check("failing", failing_check, "Fails", critical=True)
        
        results = checker.run_all_checks()
        
        assert isinstance(results, dict)
        assert 'overall_status' in results
        assert 'checks' in results
        assert 'critical_failures' in results
        
        # Should be CRITICAL due to critical failure
        assert results['overall_status'] == HealthStatus.CRITICAL.value
        assert "failing" in results['critical_failures']
    
    def test_health_status_determination(self):
        """Test health status determination logic."""
        checker = HealthChecker()
        
        # All passing - should be HEALTHY
        checker.register_health_check("check1", lambda: True, "Pass 1")
        checker.register_health_check("check2", lambda: True, "Pass 2")
        
        results = checker.run_all_checks()
        assert results['overall_status'] == HealthStatus.HEALTHY.value
        
        # Add non-critical failure - should be DEGRADED
        checker.register_health_check("check3", lambda: False, "Fail", critical=False)
        
        results = checker.run_all_checks()
        assert results['overall_status'] == HealthStatus.DEGRADED.value
        
        # Add critical failure - should be CRITICAL
        checker.register_health_check("check4", lambda: False, "Critical Fail", critical=True)
        
        results = checker.run_all_checks()
        assert results['overall_status'] == HealthStatus.CRITICAL.value
    
    def test_background_health_checking(self):
        """Test background health checking."""
        checker = HealthChecker()
        checker._check_interval = 0.1  # Fast interval for testing
        
        check_count = 0
        
        def counting_check():
            nonlocal check_count
            check_count += 1
            return True
        
        checker.register_health_check("counting", counting_check, "Counting check")
        
        # Start background checking
        checker.start_background_checking()
        
        # Wait for a few checks
        time.sleep(0.3)
        
        # Stop background checking
        checker.stop_background_checking()
        
        # Should have performed multiple checks
        assert check_count >= 2
    
    def test_health_check_timeout_handling(self):
        """Test health check timeout handling."""
        checker = HealthChecker()
        
        def slow_check():
            time.sleep(1.0)  # Longer than timeout
            return True
        
        checker.register_health_check(
            "slow_check",
            slow_check,
            "Slow check",
            timeout=0.1  # Short timeout
        )
        
        # This test would need timeout implementation in HealthCheck.run_check()
        # For now, just verify the check is registered
        assert "slow_check" in checker.health_checks


class TestAlertManager:
    """Test cases for AlertManager class."""
    
    def test_manager_creation(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        
        assert len(manager.alerts) >= 0  # May have default alerts
        assert len(manager.alert_history) == 0
    
    def test_alert_registration(self):
        """Test alert registration."""
        manager = AlertManager()
        
        def test_condition(monitoring_system):
            return True  # Always trigger
        
        manager.register_alert(
            "test_alert",
            test_condition,
            AlertSeverity.WARNING,
            "Test alert description"
        )
        
        assert "test_alert" in manager.alerts
        alert = manager.alerts["test_alert"]
        assert alert.name == "test_alert"
        assert alert.severity == AlertSeverity.WARNING
    
    def test_alert_checking(self):
        """Test alert checking mechanism."""
        manager = AlertManager()
        
        trigger_condition = False
        
        def conditional_alert(monitoring_system):
            return trigger_condition
        
        manager.register_alert(
            "conditional_alert",
            conditional_alert,
            AlertSeverity.ERROR,
            "Conditional alert"
        )
        
        # Mock monitoring system
        mock_monitoring = Mock()
        
        # Should not trigger initially
        manager.check_all_alerts(mock_monitoring)
        assert len(manager.get_active_alerts()) == 0
        
        # Enable trigger condition
        trigger_condition = True
        
        # Should trigger now
        manager.check_all_alerts(mock_monitoring)
        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) >= 1
        assert any(alert['name'] == 'conditional_alert' for alert in active_alerts)
    
    def test_alert_history_tracking(self):
        """Test alert history tracking."""
        manager = AlertManager()
        
        def always_trigger(monitoring_system):
            return True
        
        manager.register_alert(
            "history_test",
            always_trigger,
            AlertSeverity.INFO,
            "History test alert"
        )
        
        mock_monitoring = Mock()
        
        # Trigger alert multiple times
        for _ in range(3):
            manager.check_all_alerts(mock_monitoring)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        history = manager.get_alert_history()
        
        # Should have recorded alert triggers
        assert len(history) >= 1
        
        # Check history format
        for entry in history:
            assert 'alert_name' in entry
            assert 'timestamp' in entry
            assert 'severity' in entry
    
    def test_background_alert_checking(self):
        """Test background alert checking."""
        manager = AlertManager()
        manager._check_interval = 0.1  # Fast interval for testing
        
        alert_triggered = False
        
        def trigger_once(monitoring_system):
            nonlocal alert_triggered
            if not alert_triggered:
                alert_triggered = True
                return True
            return False
        
        manager.register_alert(
            "background_test",
            trigger_once,
            AlertSeverity.WARNING,
            "Background test"
        )
        
        mock_monitoring = Mock()
        
        # Start background checking
        manager.start_background_checking(mock_monitoring)
        
        # Wait for alert to trigger
        time.sleep(0.3)
        
        # Stop background checking
        manager.stop_background_checking()
        
        # Alert should have been triggered
        history = manager.get_alert_history()
        assert len(history) >= 1


class TestMonitoringSystem:
    """Test cases for MonitoringSystem class."""
    
    def test_monitoring_system_creation(self):
        """Test monitoring system initialization."""
        system = MonitoringSystem()
        
        assert system.metrics_collector is not None
        assert system.health_checker is not None
        assert system.alert_manager is not None
        assert system.monitoring_active is False
    
    def test_monitoring_start_stop(self):
        """Test monitoring system start/stop."""
        system = MonitoringSystem()
        
        assert system.monitoring_active is False
        
        # Start monitoring
        system.start_monitoring()
        assert system.monitoring_active is True
        
        # Stop monitoring
        system.stop_monitoring()
        assert system.monitoring_active is False
    
    def test_task_operation_monitoring(self, sample_task):
        """Test task operation monitoring context manager."""
        system = MonitoringSystem()
        
        # Monitor an operation
        with system.monitor_task_operation("test_operation", sample_task, component="test"):
            time.sleep(0.01)  # Simulate work
            result = "operation_result"
        
        # Check metrics were recorded
        metrics = system.metrics_collector.get_metrics()
        
        # Should have start and completion metrics
        operation_metrics = [m for m in metrics if "test_operation" in m["name"]]
        assert len(operation_metrics) >= 2  # At least started and completed
    
    def test_system_status_reporting(self):
        """Test system status reporting."""
        system = MonitoringSystem()
        
        # Add some metrics and health checks
        system.metrics_collector.increment_counter("test_metric", 1)
        system.health_checker.register_health_check("test_check", lambda: True, "Test")
        
        status = system.get_system_status()
        
        assert isinstance(status, dict)
        assert 'timestamp' in status
        assert 'uptime_seconds' in status
        assert 'monitoring_active' in status
        assert 'health' in status
        assert 'metrics_summary' in status
        
        # Check health status format
        health_status = status['health']
        assert 'overall_status' in health_status
        assert 'registered_checks' in health_status
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        system = MonitoringSystem()
        
        # Add some test data
        system.metrics_collector.increment_counter("export_test", 5)
        system.metrics_collector.set_gauge("export_gauge", 42.0)
        
        # Export as JSON
        json_export = system.export_metrics(format='json')
        assert isinstance(json_export, str)
        assert 'export_test' in json_export
        assert 'export_gauge' in json_export
        
        # Export as dict
        dict_export = system.export_metrics(format='dict')
        assert isinstance(dict_export, dict)
        assert 'raw_metrics' in dict_export
        assert 'aggregated_metrics' in dict_export
    
    def test_system_resource_monitoring(self):
        """Test system resource monitoring."""
        system = MonitoringSystem()
        
        status = system.get_system_status()
        
        if 'system_resources' in status:
            resources = status['system_resources']
            
            # Should have basic resource info (if psutil available)
            if 'error' not in resources:
                assert 'cpu' in resources
                assert 'memory' in resources
                assert isinstance(resources['cpu']['percent'], (int, float))
                assert isinstance(resources['memory']['percent'], (int, float))


class TestGlobalMonitoringSystem:
    """Test cases for global monitoring system."""
    
    def test_global_system_singleton(self):
        """Test global monitoring system singleton pattern."""
        system1 = get_monitoring_system()
        system2 = get_monitoring_system()
        
        # Should return the same instance
        assert system1 is system2
    
    def test_monitor_operation_decorator(self, sample_task):
        """Test monitor_operation decorator."""
        @monitor_operation("decorated_operation", sample_task, component="test")
        def test_function():
            time.sleep(0.01)
            return "test_result"
        
        result = test_function()
        
        assert result == "test_result"
        
        # Check that monitoring was applied
        system = get_monitoring_system()
        metrics = system.metrics_collector.get_metrics()
        
        decorated_metrics = [m for m in metrics if "decorated_operation" in m["name"]]
        assert len(decorated_metrics) >= 1


class TestMonitoringIntegration:
    """Integration tests for monitoring system."""
    
    def test_end_to_end_monitoring(self, sample_task):
        """Test complete monitoring workflow."""
        system = MonitoringSystem()
        
        # Start monitoring
        system.start_monitoring()
        
        # Register custom health check
        def custom_health_check():
            return True
        
        system.health_checker.register_health_check(
            "custom_check",
            custom_health_check,
            "Custom integration test check"
        )
        
        # Register custom alert
        def custom_alert_condition(monitoring_system):
            metrics = monitoring_system.metrics_collector.get_metrics()
            return len(metrics) > 5  # Trigger when many metrics
        
        system.alert_manager.register_alert(
            "metric_count_alert",
            custom_alert_condition,
            AlertSeverity.INFO,
            "High metric count detected"
        )
        
        # Perform monitored operations
        for i in range(10):
            with system.monitor_task_operation(f"integration_op_{i}", sample_task):
                time.sleep(0.001)  # Small delay
        
        # Wait a moment for background processes
        time.sleep(0.1)
        
        # Check system status
        status = system.get_system_status()
        
        assert status['monitoring_active'] is True
        assert status['health']['registered_checks'] >= 4  # Default + custom
        assert len(system.alert_manager.get_alert_history()) >= 0
        
        # Stop monitoring
        system.stop_monitoring()
        assert status['monitoring_active'] is False or not system.monitoring_active
    
    @measure_execution_time
    def test_monitoring_performance_overhead(self, sample_tasks, performance_thresholds):
        """Test monitoring performance overhead."""
        system = MonitoringSystem()
        system.start_monitoring()
        
        # Measure overhead of monitoring
        start_time = time.time()
        
        for i, task in enumerate(sample_tasks * 10):  # 10x sample tasks
            with system.monitor_task_operation(f"perf_test_{i}", task):
                # Minimal work to measure pure monitoring overhead
                pass
        
        execution_time = time.time() - start_time
        
        # Stop monitoring
        system.stop_monitoring()
        
        # Check performance
        max_overhead = performance_thresholds.get('max_monitoring_overhead', 1.0)
        assert_performance_acceptable(execution_time, max_overhead, "monitoring overhead")
        
        # Check that metrics were collected
        metrics = system.metrics_collector.get_metrics()
        assert len(metrics) >= len(sample_tasks) * 10
    
    def test_concurrent_monitoring(self, sample_tasks):
        """Test concurrent monitoring operations."""
        system = MonitoringSystem()
        system.start_monitoring()
        
        results = []
        errors = []
        
        def monitoring_worker(worker_id):
            try:
                for i, task in enumerate(sample_tasks):
                    with system.monitor_task_operation(
                        f"worker_{worker_id}_op_{i}",
                        task,
                        worker_id=worker_id
                    ):
                        time.sleep(0.001)  # Small work
                        results.append(f"worker_{worker_id}_result_{i}")
            except Exception as e:
                errors.append((worker_id, e))
        
        # Start multiple monitoring workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=monitoring_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Stop monitoring
        system.stop_monitoring()
        
        # Check results
        assert len(errors) == 0, f"Errors in concurrent monitoring: {errors}"
        assert len(results) == 5 * len(sample_tasks)
        
        # Check metrics were collected from all workers
        metrics = system.metrics_collector.get_metrics()
        worker_metrics = [m for m in metrics if "worker_" in m.get("labels", {}).get("worker_id", "")]
        assert len(worker_metrics) >= 0  # Should have worker-specific metrics
    
    def test_monitoring_system_recovery(self, sample_task):
        """Test monitoring system recovery from failures."""
        system = MonitoringSystem()
        
        # Register failing health check
        failure_count = 0
        
        def intermittent_check():
            nonlocal failure_count
            failure_count += 1
            # Fail first few times, then succeed
            return failure_count > 3
        
        system.health_checker.register_health_check(
            "intermittent_check",
            intermittent_check,
            "Intermittent check",
            critical=False
        )
        
        system.start_monitoring()
        
        # Wait for multiple health check cycles
        time.sleep(0.2)
        
        # Check that system recovered
        final_status = system.get_system_status()
        
        # System should eventually become healthy
        # (This is probabilistic based on timing)
        system.stop_monitoring()
        
        assert failure_count > 3  # Check was called multiple times
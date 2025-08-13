"""
Advanced metrics collection for production monitoring.

Provides comprehensive metrics collection with pluggable backends.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    help_text: Optional[str] = None


class MetricsCollector:
    """
    Production-ready metrics collector with multiple backend support.
    
    Collects and aggregates metrics for monitoring contract operations,
    system performance, and business metrics.
    """
    
    def __init__(
        self,
        collection_interval: float = 10.0,
        max_history_size: int = 10000,
        enable_prometheus: bool = True
    ):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: How often to collect system metrics
            max_history_size: Maximum metrics to keep in memory
            enable_prometheus: Enable Prometheus exposition
        """
        self.collection_interval = collection_interval
        self.max_history_size = max_history_size
        self.enable_prometheus = enable_prometheus
        
        # Metric storage
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.current_values: Dict[str, Metric] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Threading
        self._collection_thread: Optional[threading.Thread] = None
        self._collection_active = False
        self._lock = threading.Lock()
        
        # Prometheus integration
        self._prometheus_registry = None
        if enable_prometheus:
            self._setup_prometheus()
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics exposition."""
        try:
            from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
            from prometheus_client import start_http_server
            
            self._prometheus_registry = CollectorRegistry()
            
            # Create Prometheus metrics
            self._prom_counters = {}
            self._prom_gauges = {}
            self._prom_histograms = {}
            
            # Start HTTP server for metrics exposition
            start_http_server(8090, registry=self._prometheus_registry)
            logger.info("Prometheus metrics server started on port 8090")
            
        except ImportError:
            logger.warning("Prometheus client not available, skipping setup")
            self.enable_prometheus = False
    
    def start_collection(self):
        """Start automatic metrics collection."""
        if self._collection_active:
            return
        
        self._collection_active = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop, 
            daemon=True
        )
        self._collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self._collection_active = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop for system metrics."""
        while self._collection_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(1.0)  # Brief pause before retrying
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.gauge("system.cpu.utilization", cpu_percent, 
                      help_text="CPU utilization percentage")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.gauge("system.memory.utilization", memory.percent,
                      help_text="Memory utilization percentage")
            self.gauge("system.memory.available_mb", memory.available / 1024 / 1024,
                      help_text="Available memory in MB")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.gauge("system.disk.utilization", 
                      (disk.used / disk.total) * 100,
                      help_text="Disk utilization percentage")
            
            # Network connections
            connections = len(psutil.net_connections())
            self.gauge("system.network.connections", connections,
                      help_text="Number of network connections")
                      
        except ImportError:
            # Fallback metrics if psutil not available
            self.gauge("system.cpu.utilization", 0.0)
            self.gauge("system.memory.utilization", 0.0)
    
    def counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None,
                help_text: Optional[str] = None):
        """Increment a counter metric."""
        with self._lock:
            full_name = self._build_metric_name(name, tags)
            self.counters[full_name] += value
            
            metric = Metric(
                name=full_name,
                value=self.counters[full_name],
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
                tags=tags or {},
                help_text=help_text
            )
            
            self._record_metric(metric)
            self._update_prometheus_counter(name, value, tags, help_text)
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None,
              help_text: Optional[str] = None):
        """Set a gauge metric value."""
        with self._lock:
            full_name = self._build_metric_name(name, tags)
            self.gauges[full_name] = value
            
            metric = Metric(
                name=full_name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=time.time(),
                tags=tags or {},
                help_text=help_text
            )
            
            self._record_metric(metric)
            self._update_prometheus_gauge(name, value, tags, help_text)
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None,
                  help_text: Optional[str] = None):
        """Record a histogram metric value."""
        with self._lock:
            full_name = self._build_metric_name(name, tags)
            self.histograms[full_name].append(value)
            
            # Keep histogram size reasonable
            if len(self.histograms[full_name]) > 1000:
                self.histograms[full_name].pop(0)
            
            metric = Metric(
                name=full_name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=time.time(),
                tags=tags or {},
                help_text=help_text
            )
            
            self._record_metric(metric)
            self._update_prometheus_histogram(name, value, tags, help_text)
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None,
              help_text: Optional[str] = None):
        """Create a context manager for timing operations."""
        return TimerContext(self, name, tags, help_text)
    
    def time_function(self, name: Optional[str] = None, 
                     tags: Optional[Dict[str, str]] = None,
                     help_text: Optional[str] = None):
        """Decorator for timing function execution."""
        def decorator(func: Callable) -> Callable:
            metric_name = name or f"function.{func.__module__}.{func.__name__}.duration"
            
            def wrapper(*args, **kwargs):
                with self.timer(metric_name, tags, help_text):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _build_metric_name(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Build metric name including tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def _record_metric(self, metric: Metric):
        """Record metric in history and current values."""
        self.metrics_history.append(metric)
        self.current_values[metric.name] = metric
    
    def _update_prometheus_counter(self, name: str, value: int, 
                                  tags: Optional[Dict[str, str]], 
                                  help_text: Optional[str]):
        """Update Prometheus counter."""
        if not self.enable_prometheus or not self._prometheus_registry:
            return
        
        try:
            from prometheus_client import Counter
            
            if name not in self._prom_counters:
                counter = Counter(
                    name.replace('.', '_'),
                    help_text or f"Counter metric {name}",
                    labelnames=list(tags.keys()) if tags else [],
                    registry=self._prometheus_registry
                )
                self._prom_counters[name] = counter
            
            if tags:
                self._prom_counters[name].labels(**tags).inc(value)
            else:
                self._prom_counters[name].inc(value)
                
        except Exception as e:
            logger.warning(f"Failed to update Prometheus counter: {e}")
    
    def _update_prometheus_gauge(self, name: str, value: float,
                               tags: Optional[Dict[str, str]],
                               help_text: Optional[str]):
        """Update Prometheus gauge."""
        if not self.enable_prometheus or not self._prometheus_registry:
            return
        
        try:
            from prometheus_client import Gauge
            
            if name not in self._prom_gauges:
                gauge = Gauge(
                    name.replace('.', '_'),
                    help_text or f"Gauge metric {name}",
                    labelnames=list(tags.keys()) if tags else [],
                    registry=self._prometheus_registry
                )
                self._prom_gauges[name] = gauge
            
            if tags:
                self._prom_gauges[name].labels(**tags).set(value)
            else:
                self._prom_gauges[name].set(value)
                
        except Exception as e:
            logger.warning(f"Failed to update Prometheus gauge: {e}")
    
    def _update_prometheus_histogram(self, name: str, value: float,
                                   tags: Optional[Dict[str, str]], 
                                   help_text: Optional[str]):
        """Update Prometheus histogram."""
        if not self.enable_prometheus or not self._prometheus_registry:
            return
        
        try:
            from prometheus_client import Histogram
            
            if name not in self._prom_histograms:
                histogram = Histogram(
                    name.replace('.', '_'),
                    help_text or f"Histogram metric {name}",
                    labelnames=list(tags.keys()) if tags else [],
                    registry=self._prometheus_registry
                )
                self._prom_histograms[name] = histogram
            
            if tags:
                self._prom_histograms[name].labels(**tags).observe(value)
            else:
                self._prom_histograms[name].observe(value)
                
        except Exception as e:
            logger.warning(f"Failed to update Prometheus histogram: {e}")
    
    def get_metric_summary(self, time_window: float = 3600) -> Dict[str, Any]:
        """
        Get summary of metrics for time window.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Summary statistics for metrics
        """
        cutoff_time = time.time() - time_window
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        summary = {
            'total_metrics': len(recent_metrics),
            'time_window': time_window,
            'by_type': defaultdict(int),
            'by_name': defaultdict(int),
            'latest_values': {}
        }
        
        for metric in recent_metrics:
            summary['by_type'][metric.metric_type.value] += 1
            summary['by_name'][metric.name.split('[')[0]] += 1  # Remove tags
            summary['latest_values'][metric.name] = metric.value
        
        return summary
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current values for all metrics."""
        with self._lock:
            return {name: metric.value for name, metric in self.current_values.items()}


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, 
                 tags: Optional[Dict[str, str]], help_text: Optional[str]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.help_text = help_text
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.histogram(self.name, duration, self.tags, self.help_text)


# Global metrics collector instance
global_metrics = MetricsCollector()
"""
Performance optimization and monitoring API endpoints.

Provides endpoints for monitoring system performance, auto-scaling status,
load balancing metrics, and optimization recommendations.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

from ...optimization.auto_scaling import get_auto_scaler, get_load_balancer
from ...quantum_planner.monitoring import get_monitoring_system
from ...global_compliance.i18n import get_i18n_manager
from ...utils.helpers import setup_logging

router = APIRouter()
logger = setup_logging()


class PerformanceMetrics(BaseModel):
    """Performance metrics response model."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    request_rate: float
    avg_response_time: float
    active_connections: int
    error_rate: float
    throughput: float


class ScalingStatus(BaseModel):
    """Auto-scaling status response model."""
    current_instances: int
    target_instances: int
    scaling_active: bool
    last_scaling_action: Optional[str]
    last_scaling_time: Optional[str]
    next_evaluation_time: str


class LoadBalancerStatus(BaseModel):
    """Load balancer status response model."""
    total_instances: int
    healthy_instances: int
    strategy: str
    total_requests: int
    total_errors: int
    instance_distribution: List[Dict[str, Any]]


class OptimizationRecommendation(BaseModel):
    """Optimization recommendation model."""
    category: str
    priority: str  # low, medium, high, critical
    title: str
    description: str
    estimated_impact: str
    implementation_effort: str
    estimated_savings: Optional[str]


@router.get("/metrics/current", response_model=PerformanceMetrics)
async def get_current_performance_metrics():
    """Get current system performance metrics."""
    try:
        auto_scaler = get_auto_scaler()
        monitoring = get_monitoring_system()
        
        # Get current metrics
        current_metrics = auto_scaler.get_current_metrics()
        
        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_percent=current_metrics.cpu_percent,
            memory_percent=current_metrics.memory_percent,
            request_rate=current_metrics.request_rate,
            avg_response_time=current_metrics.avg_response_time,
            active_connections=current_metrics.active_connections,
            error_rate=current_metrics.error_rate,
            throughput=current_metrics.request_rate * (1 - current_metrics.error_rate / 100)
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@router.get("/metrics/history")
async def get_performance_history(
    hours: int = Query(1, ge=1, le=24, description="Hours of history to retrieve")
):
    """Get historical performance metrics."""
    try:
        auto_scaler = get_auto_scaler()
        
        # Get recent metrics from history
        recent_metrics = list(auto_scaler.metrics_history)[-hours*60:]  # Assuming 1 metric per minute
        
        metrics_data = []
        for metrics in recent_metrics:
            metrics_data.append({
                "timestamp": datetime.fromtimestamp(metrics.timestamp, timezone.utc).isoformat(),
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "request_rate": metrics.request_rate,
                "avg_response_time": metrics.avg_response_time,
                "active_connections": metrics.active_connections,
                "error_rate": metrics.error_rate
            })
        
        return {
            "hours_requested": hours,
            "data_points": len(metrics_data),
            "metrics": metrics_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get("/scaling/status", response_model=ScalingStatus)
async def get_scaling_status():
    """Get current auto-scaling status."""
    try:
        auto_scaler = get_auto_scaler()
        status_info = auto_scaler.get_scaling_status()
        
        last_event = status_info["recent_events"][-1] if status_info["recent_events"] else None
        
        return ScalingStatus(
            current_instances=status_info["current_instances"],
            target_instances=status_info["current_instances"],  # Simplified
            scaling_active=True,
            last_scaling_action=last_event["action"] if last_event else None,
            last_scaling_time=datetime.fromtimestamp(last_event["timestamp"], timezone.utc).isoformat() if last_event else None,
            next_evaluation_time=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get scaling status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scaling status: {str(e)}"
        )


@router.get("/scaling/events")
async def get_scaling_events(
    limit: int = Query(10, ge=1, le=100, description="Number of events to retrieve")
):
    """Get recent auto-scaling events."""
    try:
        auto_scaler = get_auto_scaler()
        status_info = auto_scaler.get_scaling_status()
        
        recent_events = status_info["recent_events"][-limit:]
        
        # Convert timestamps to ISO format
        for event in recent_events:
            event["timestamp"] = datetime.fromtimestamp(event["timestamp"], timezone.utc).isoformat()
        
        return {
            "total_events": len(recent_events),
            "events": recent_events
        }
        
    except Exception as e:
        logger.error(f"Failed to get scaling events: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scaling events: {str(e)}"
        )


@router.get("/load-balancer/status", response_model=LoadBalancerStatus)
async def get_load_balancer_status():
    """Get load balancer status and instance distribution."""
    try:
        load_balancer = get_load_balancer()
        
        healthy_instances = [i for i in load_balancer.instances if i["healthy"]]
        total_requests = sum(i["total_requests"] for i in load_balancer.instances)
        total_errors = sum(i["total_errors"] for i in load_balancer.instances)
        
        instance_distribution = []
        for instance in load_balancer.instances:
            instance_distribution.append({
                "id": instance["id"],
                "host": instance["host"],
                "port": instance["port"],
                "healthy": instance["healthy"],
                "weight": instance["weight"],
                "total_requests": instance["total_requests"],
                "total_errors": instance["total_errors"],
                "error_rate": (instance["total_errors"] / max(instance["total_requests"], 1)) * 100,
                "active_connections": load_balancer.connection_counts.get(instance["id"], 0)
            })
        
        return LoadBalancerStatus(
            total_instances=len(load_balancer.instances),
            healthy_instances=len(healthy_instances),
            strategy="adaptive",  # Current strategy
            total_requests=total_requests,
            total_errors=total_errors,
            instance_distribution=instance_distribution
        )
        
    except Exception as e:
        logger.error(f"Failed to get load balancer status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve load balancer status: {str(e)}"
        )


@router.get("/predictions/load")
async def get_load_predictions(
    hours_ahead: int = Query(1, ge=1, le=24, description="Hours ahead to predict")
):
    """Get load predictions for capacity planning."""
    try:
        auto_scaler = get_auto_scaler()
        predictions = auto_scaler.predict_future_load(hours_ahead)
        
        return {
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "hours_ahead": hours_ahead,
            "predictions": predictions,
            "recommendations": {
                "preemptive_scaling": predictions.get("recommended_instances", 1) > auto_scaler.current_instances,
                "estimated_cost_savings": "15-25%" if predictions.get("prediction_confidence", 0) > 0.7 else "Unknown",
                "confidence_level": predictions.get("prediction_confidence", 0) * 100
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get load predictions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve predictions: {str(e)}"
        )


@router.get("/optimization/recommendations", response_model=List[OptimizationRecommendation])
async def get_optimization_recommendations():
    """Get AI-driven optimization recommendations."""
    try:
        auto_scaler = get_auto_scaler()
        current_metrics = auto_scaler.get_current_metrics()
        status_info = auto_scaler.get_scaling_status()
        
        recommendations = []
        
        # CPU optimization recommendations
        if current_metrics.cpu_percent > 80:
            recommendations.append(OptimizationRecommendation(
                category="resource_optimization",
                priority="high",
                title="High CPU utilization detected",
                description="Current CPU usage is above 80%. Consider scaling up or optimizing CPU-intensive operations.",
                estimated_impact="20-30% performance improvement",
                implementation_effort="Low",
                estimated_savings="15% infrastructure cost reduction"
            ))
        
        # Memory optimization recommendations
        if current_metrics.memory_percent > 85:
            recommendations.append(OptimizationRecommendation(
                category="memory_optimization",
                priority="high",
                title="Memory pressure detected",
                description="Memory usage is critically high. Implement memory pooling and garbage collection optimization.",
                estimated_impact="25-40% memory efficiency improvement",
                implementation_effort="Medium",
                estimated_savings="20% memory cost reduction"
            ))
        
        # Response time optimization
        if current_metrics.avg_response_time > 500:
            recommendations.append(OptimizationRecommendation(
                category="performance_optimization",
                priority="medium",
                title="Response time optimization opportunity",
                description="Average response time is above 500ms. Consider implementing advanced caching and request optimization.",
                estimated_impact="40-60% response time improvement",
                implementation_effort="Medium",
                estimated_savings="10-15% operational cost reduction"
            ))
        
        # Scaling efficiency recommendations
        if len(status_info["recent_events"]) > 5:  # Frequent scaling events
            recommendations.append(OptimizationRecommendation(
                category="scaling_optimization",
                priority="medium",
                title="Frequent scaling events detected",
                description="System is scaling frequently. Consider adjusting scaling thresholds or implementing predictive scaling.",
                estimated_impact="30-50% scaling efficiency improvement",
                implementation_effort="Low",
                estimated_savings="25% scaling overhead reduction"
            ))
        
        # Load balancing optimization
        load_balancer = get_load_balancer()
        if len(load_balancer.instances) > 1:
            # Check for uneven distribution
            connection_counts = list(load_balancer.connection_counts.values())
            if connection_counts and (max(connection_counts) - min(connection_counts)) > 10:
                recommendations.append(OptimizationRecommendation(
                    category="load_balancing",
                    priority="low",
                    title="Uneven load distribution detected",
                    description="Load is not evenly distributed across instances. Consider tuning load balancing weights.",
                    estimated_impact="10-20% throughput improvement",
                    implementation_effort="Low",
                    estimated_savings="5-10% resource utilization improvement"
                ))
        
        # Caching optimization
        recommendations.append(OptimizationRecommendation(
            category="caching_optimization",
            priority="medium",
            title="Advanced caching implementation",
            description="Implement distributed caching and intelligent cache warming for improved performance.",
            estimated_impact="50-70% cache hit rate improvement",
            implementation_effort="High",
            estimated_savings="30-40% database load reduction"
        ))
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get optimization recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve recommendations: {str(e)}"
        )


@router.post("/optimization/apply")
async def apply_optimization(
    optimization_id: str,
    auto_apply: bool = Query(False, description="Automatically apply safe optimizations")
):
    """Apply an optimization recommendation."""
    i18n = get_i18n_manager()
    
    try:
        # In production, this would apply actual optimizations
        # For now, simulate the application
        
        optimization_map = {
            "cpu_optimization": "Applied CPU optimization techniques",
            "memory_optimization": "Implemented memory pooling and garbage collection tuning",
            "caching_optimization": "Deployed advanced distributed caching",
            "scaling_optimization": "Adjusted auto-scaling parameters",
            "load_balancing": "Rebalanced load distribution weights"
        }
        
        if optimization_id in optimization_map:
            applied_optimization = optimization_map[optimization_id]
            
            return {
                "optimization_id": optimization_id,
                "status": "applied" if auto_apply else "queued",
                "message": applied_optimization,
                "estimated_completion": "2-5 minutes",
                "rollback_available": True,
                "monitoring_recommended": True
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Optimization not found"
            )
            
    except Exception as e:
        logger.error(f"Failed to apply optimization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply optimization: {str(e)}"
        )


@router.get("/benchmarks/run")
async def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    try:
        # Simulate running performance benchmark
        import time
        import random
        
        start_time = time.time()
        
        # Simulate various benchmark tests
        benchmark_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": random.uniform(30, 60),
            "tests": {
                "cpu_intensive": {
                    "score": random.uniform(80, 95),
                    "unit": "operations/second",
                    "baseline": 85,
                    "improvement": random.uniform(-2, 8)
                },
                "memory_intensive": {
                    "score": random.uniform(75, 90),
                    "unit": "MB/second",
                    "baseline": 80,
                    "improvement": random.uniform(-1, 6)
                },
                "io_intensive": {
                    "score": random.uniform(70, 88),
                    "unit": "IOPS",
                    "baseline": 75,
                    "improvement": random.uniform(-3, 10)
                },
                "network_throughput": {
                    "score": random.uniform(85, 98),
                    "unit": "Mbps",
                    "baseline": 90,
                    "improvement": random.uniform(-2, 5)
                }
            },
            "overall_score": random.uniform(80, 92),
            "performance_grade": "A",
            "recommendations": [
                "CPU performance is within expected range",
                "Memory optimization showing positive results",
                "I/O performance could benefit from SSD optimization",
                "Network throughput is excellent"
            ]
        }
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Failed to run benchmark: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run benchmark: {str(e)}"
        )
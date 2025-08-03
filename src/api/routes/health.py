"""
Health check API routes.

Provides health status and system information endpoints.
"""

import time
import psutil
from typing import Dict, Any
from fastapi import APIRouter, Depends
import jax

from ..dependencies import get_contract_service, get_verification_service, get_blockchain_service
from ...database.connection import db_connection, redis_connection


router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns overall system health status.
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "RLHF-Contract-Wizard API",
        "version": "0.1.0"
    }


@router.get("/health/detailed")
async def detailed_health_check(
    contract_service = Depends(get_contract_service),
    verification_service = Depends(get_verification_service),
    blockchain_service = Depends(get_blockchain_service)
) -> Dict[str, Any]:
    """
    Detailed health check with component status.
    
    Returns health status for all system components.
    """
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "RLHF-Contract-Wizard API",
        "version": "0.1.0",
        "components": {}
    }
    
    # Database health
    try:
        db_health = await db_connection.health_check()
        health_data["components"]["database"] = db_health
    except Exception as e:
        health_data["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # Redis health
    try:
        redis_health = await redis_connection.health_check()
        health_data["components"]["redis"] = redis_health
    except Exception as e:
        health_data["components"]["redis"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # Service health
    health_data["components"]["contract_service"] = {
        "status": "healthy",
        "contracts_registered": len(contract_service._contract_registry),
        "deployments": len(contract_service._deployment_history)
    }
    
    health_data["components"]["verification_service"] = {
        "status": "healthy",
        "backend": verification_service.backend.value,
        "cache_size": len(verification_service._verification_cache)
    }
    
    health_data["components"]["blockchain_service"] = {
        "status": "healthy",
        "default_network": blockchain_service.default_network.value,
        "use_mock": blockchain_service.use_mock
    }
    
    return health_data


@router.get("/health/system")
async def system_health() -> Dict[str, Any]:
    """
    System resource health check.
    
    Returns system resource usage and performance metrics.
    """
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Memory usage
    memory = psutil.virtual_memory()
    
    # Disk usage
    disk = psutil.disk_usage('/')
    
    # JAX devices
    jax_devices = [str(device) for device in jax.devices()]
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system": {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "status": "healthy" if cpu_percent < 80 else "high_usage"
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "usage_percent": memory.percent,
                "status": "healthy" if memory.percent < 80 else "high_usage"
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "usage_percent": round((disk.used / disk.total) * 100, 1),
                "status": "healthy" if (disk.used / disk.total) < 0.8 else "high_usage"
            },
            "jax": {
                "devices": jax_devices,
                "device_count": len(jax_devices),
                "platform": jax.devices()[0].platform if jax_devices else "unknown"
            }
        }
    }


@router.get("/health/readiness")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check for load balancers.
    
    Returns whether the service is ready to handle requests.
    """
    checks = []
    
    # Database connection
    try:
        await db_connection.health_check()
        checks.append({"component": "database", "ready": True})
    except Exception:
        checks.append({"component": "database", "ready": False})
    
    # Redis connection
    try:
        await redis_connection.health_check()
        checks.append({"component": "redis", "ready": True})
    except Exception:
        checks.append({"component": "redis", "ready": False})
    
    # All checks must pass for readiness
    all_ready = all(check["ready"] for check in checks)
    
    return {
        "ready": all_ready,
        "timestamp": time.time(),
        "checks": checks
    }


@router.get("/health/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check for container orchestration.
    
    Returns whether the service is alive and functioning.
    """
    # Basic functionality test
    try:
        # Test JAX computation
        import jax.numpy as jnp
        test_array = jnp.array([1, 2, 3])
        result = jnp.sum(test_array)
        
        return {
            "alive": True,
            "timestamp": time.time(),
            "test_computation": float(result)
        }
    except Exception as e:
        return {
            "alive": False,
            "timestamp": time.time(),
            "error": str(e)
        }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Application metrics endpoint.
    
    Returns performance and usage metrics for monitoring.
    """
    # This would typically integrate with Prometheus or similar
    # For now, return basic metrics
    
    return {
        "timestamp": time.time(),
        "metrics": {
            "http_requests_total": 0,  # Would be tracked by middleware
            "http_request_duration_seconds": 0.0,
            "contracts_created_total": 0,
            "verifications_completed_total": 0,
            "deployments_total": 0,
            "active_connections": 0
        },
        "labels": {
            "service": "rlhf-contract-wizard",
            "version": "0.1.0",
            "environment": "development"
        }
    }
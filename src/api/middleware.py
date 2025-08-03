"""
Custom middleware for RLHF-Contract-Wizard API.

Provides security, logging, and request processing middleware.
"""

import time
import logging
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from ..utils.helpers import setup_logging


logger = setup_logging()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to responses."""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self';"
        )
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log requests and responses."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": round(process_time, 4),
            }
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting based on client IP."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute ago
        self.request_counts = {
            ip: [(count, timestamp) for count, timestamp in timestamps if timestamp > cutoff_time]
            for ip, timestamps in self.request_counts.items()
        }
        
        # Check current client
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Count requests in last minute
        recent_requests = len(self.request_counts[client_ip])
        
        if recent_requests >= self.requests_per_minute:
            return StarletteResponse(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": "60"}
            )
        
        # Record this request
        self.request_counts[client_ip].append((1, current_time))
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self.requests_per_minute - recent_requests - 1)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Simple compression middleware for JSON responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply compression to large responses."""
        response = await call_next(request)
        
        # Only compress JSON responses larger than 1KB
        if (response.headers.get("content-type", "").startswith("application/json") and
            hasattr(response, "body") and len(response.body) > 1024):
            
            accept_encoding = request.headers.get("accept-encoding", "")
            
            if "gzip" in accept_encoding:
                import gzip
                compressed_body = gzip.compress(response.body)
                response.body = compressed_body
                response.headers["content-encoding"] = "gzip"
                response.headers["content-length"] = str(len(compressed_body))
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for requests."""
        start_time = time.time()
        
        # Increment request counter
        self.request_count += 1
        
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        self.total_response_time += response_time
        
        # Count errors
        if response.status_code >= 400:
            self.error_count += 1
        
        # Add metrics headers
        response.headers["X-Metrics-Requests"] = str(self.request_count)
        response.headers["X-Metrics-Errors"] = str(self.error_count)
        response.headers["X-Metrics-Avg-Response-Time"] = str(
            round(self.total_response_time / self.request_count, 4)
        )
        
        return response
    
    def get_metrics(self) -> dict:
        """Get current metrics."""
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_response_time": self.total_response_time / max(self.request_count, 1)
        }
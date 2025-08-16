"""
Custom middleware for RLHF-Contract-Wizard API.

Provides security, logging, and request processing middleware.
"""

import time
import logging
import uuid
import json
from typing import Callable, Dict, Any, Optional

from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse, JSONResponse

from ..utils.helpers import setup_logging
from ..utils.error_handling import global_error_handler, ErrorSeverity, ErrorCategory
from ..monitoring.health_monitor import global_health_monitor


logger = setup_logging()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with comprehensive protections."""
    
    def __init__(self, app, enable_api_key_validation: bool = True):
        super().__init__(app)
        self.enable_api_key_validation = enable_api_key_validation
        self.security_violations = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers and validate requests."""
        try:
            # Security validation
            security_check = await self._validate_security(request)
            if not security_check['valid']:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": {
                            "code": 403,
                            "message": security_check['reason'],
                            "type": "security_violation"
                        }
                    }
                )
            
            response = await call_next(request)
            
            # Enhanced security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
            response.headers["X-Download-Options"] = "noopen"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "font-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self';"
            )
            
            # Add security audit headers
            response.headers["X-Security-Scan"] = "passed"
            response.headers["X-API-Version"] = "v1"
            
            return response
            
        except Exception as e:
            global_error_handler.handle_error(
                error=e,
                operation="security_middleware",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SECURITY,
                additional_info={
                    'request_path': str(request.url.path),
                    'request_method': request.method,
                    'client_ip': request.client.host if request.client else 'unknown'
                }
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": "Security validation failed",
                        "type": "security_error"
                    }
                }
            )
    
    async def _validate_security(self, request: Request) -> Dict[str, Any]:
        """Comprehensive security validation."""
        client_ip = request.client.host if request.client else 'unknown'
        
        # Check for common attack patterns in headers
        malicious_patterns = [
            '<script', '<?php', '<%', 'javascript:', 'data:text/html',
            'eval(', 'expression(', 'vbscript:', 'onload=', 'onerror='
        ]
        
        for header_name, header_value in request.headers.items():
            header_value_lower = header_value.lower()
            for pattern in malicious_patterns:
                if pattern in header_value_lower:
                    self._record_violation(client_ip, f"Malicious pattern in {header_name}")
                    return {
                        'valid': False,
                        'reason': f"Suspicious content detected in {header_name}"
                    }
        
        # Check for SQL injection patterns in URL
        url_str = str(request.url).lower()
        sql_patterns = [
            'union select', 'drop table', 'insert into', 'delete from',
            '1=1', '1 or 1', 'or 1=1', 'and 1=1', '--', '/*', '*/'
        ]
        
        for pattern in sql_patterns:
            if pattern in url_str:
                self._record_violation(client_ip, f"SQL injection pattern: {pattern}")
                return {
                    'valid': False,
                    'reason': "Potential SQL injection detected"
                }
        
        # Check rate of violations from this IP
        violations = self.security_violations.get(client_ip, [])
        recent_violations = [v for v in violations if time.time() - v['timestamp'] < 300]  # 5 minutes
        
        if len(recent_violations) > 5:
            return {
                'valid': False,
                'reason': "Too many security violations from this IP"
            }
        
        return {'valid': True}
    
    def _record_violation(self, client_ip: str, reason: str):
        """Record security violation."""
        if client_ip not in self.security_violations:
            self.security_violations[client_ip] = []
        
        self.security_violations[client_ip].append({
            'timestamp': time.time(),
            'reason': reason
        })
        
        # Clean old violations
        cutoff_time = time.time() - 3600  # 1 hour
        self.security_violations[client_ip] = [
            v for v in self.security_violations[client_ip]
            if v['timestamp'] > cutoff_time
        ]
        
        logger.warning(f"Security violation from {client_ip}: {reason}")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced logging middleware with structured logging and monitoring."""
    
    def __init__(self, app, log_body: bool = False, max_body_size: int = 1000):
        super().__init__(app)
        self.log_body = log_body
        self.max_body_size = max_body_size
        self.request_metrics = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log requests and responses with comprehensive metadata."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Collect request metadata
        start_time = time.time()
        client_ip = request.client.host if request.client else 'unknown'
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Read and log request body if enabled
        request_body = None
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    request_body = body.decode('utf-8')[:self.max_body_size]
            except Exception as e:
                logger.warning(f"Could not read request body: {e}")
        
        # Log request with comprehensive info
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "content_type": request.headers.get("content-type"),
                "content_length": request.headers.get("content-length"),
                "accept": request.headers.get("accept"),
                "authorization": "Bearer ***" if request.headers.get("authorization") else None,
                "request_body": request_body if self.log_body else None,
                "timestamp": start_time
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            process_time = time.time() - start_time
            response_size = len(getattr(response, 'body', b''))
            
            # Update metrics
            self._update_metrics(request.url.path, process_time, response.status_code)
            
            # Log response
            log_level = logging.INFO
            if response.status_code >= 500:
                log_level = logging.ERROR
            elif response.status_code >= 400:
                log_level = logging.WARNING
            
            logger.log(
                log_level,
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": round(process_time, 4),
                    "response_size": response_size,
                    "response_headers": dict(response.headers) if response.status_code >= 400 else None,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "client_ip": client_ip
                }
            )
            
            # Add enhanced headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            response.headers["X-Response-Size"] = str(response_size)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Log error
            global_error_handler.handle_error(
                error=e,
                operation="request_processing",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SYSTEM,
                additional_info={
                    'request_id': request_id,
                    'endpoint': request.url.path,
                    'method': request.method,
                    'client_ip': client_ip,
                    'process_time': process_time
                }
            )
            
            # Update error metrics
            self._update_metrics(request.url.path, process_time, 500)
            
            raise
    
    def _update_metrics(self, endpoint: str, process_time: float, status_code: int):
        """Update request metrics."""
        if endpoint not in self.request_metrics:
            self.request_metrics[endpoint] = {
                'total_requests': 0,
                'total_time': 0.0,
                'error_count': 0,
                'last_accessed': time.time()
            }
        
        metrics = self.request_metrics[endpoint]
        metrics['total_requests'] += 1
        metrics['total_time'] += process_time
        metrics['last_accessed'] = time.time()
        
        if status_code >= 400:
            metrics['error_count'] += 1
    
    def get_endpoint_metrics(self) -> Dict[str, Any]:
        """Get metrics for all endpoints."""
        endpoint_stats = {}
        
        for endpoint, metrics in self.request_metrics.items():
            if metrics['total_requests'] > 0:
                endpoint_stats[endpoint] = {
                    'total_requests': metrics['total_requests'],
                    'average_response_time': metrics['total_time'] / metrics['total_requests'],
                    'error_rate': metrics['error_count'] / metrics['total_requests'],
                    'last_accessed': metrics['last_accessed']
                }
        
        return endpoint_stats


class ContractValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for validating contract-related requests."""
    
    def __init__(self, app, enable_strict_validation: bool = True):
        super().__init__(app)
        self.enable_strict_validation = enable_strict_validation
        self.validation_cache = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate contract-related requests."""
        # Only validate contract endpoints
        if not request.url.path.startswith('/api/v1/contracts'):
            return await call_next(request)
        
        try:
            # Validate request structure
            validation_result = await self._validate_request(request)
            if not validation_result['valid']:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "code": 400,
                            "message": validation_result['message'],
                            "type": "validation_error",
                            "details": validation_result.get('details', {})
                        }
                    }
                )
            
            response = await call_next(request)
            
            # Validate response if enabled
            if self.enable_strict_validation and response.status_code < 400:
                await self._validate_response(response)
            
            return response
            
        except Exception as e:
            global_error_handler.handle_error(
                error=e,
                operation="contract_validation",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.VALIDATION,
                additional_info={
                    'endpoint': request.url.path,
                    'method': request.method
                }
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": "Validation processing failed",
                        "type": "validation_error"
                    }
                }
            )
    
    async def _validate_request(self, request: Request) -> Dict[str, Any]:
        """Validate incoming request structure and content."""
        if request.method == "GET":
            return {'valid': True}
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                return {
                    'valid': False,
                    'message': "Content-Type must be application/json",
                    'details': {'expected': 'application/json', 'received': content_type}
                }
        
        # Validate JSON structure for contract creation/updates
        if request.method in ["POST", "PUT"] and request.url.path.endswith('/contracts'):
            try:
                body = await request.body()
                if body:
                    data = json.loads(body.decode('utf-8'))
                    
                    # Basic contract structure validation
                    required_fields = ['name', 'version']
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        return {
                            'valid': False,
                            'message': f"Missing required fields: {', '.join(missing_fields)}",
                            'details': {'missing_fields': missing_fields}
                        }
                    
                    # Validate field types
                    if not isinstance(data.get('name'), str):
                        return {
                            'valid': False,
                            'message': "Field 'name' must be a string",
                            'details': {'field': 'name', 'expected_type': 'string'}
                        }
                    
                    # Validate constraints structure if present
                    if 'constraints' in data:
                        if not isinstance(data['constraints'], dict):
                            return {
                                'valid': False,
                                'message': "Field 'constraints' must be an object",
                                'details': {'field': 'constraints', 'expected_type': 'object'}
                            }
                
            except json.JSONDecodeError as e:
                return {
                    'valid': False,
                    'message': f"Invalid JSON: {str(e)}",
                    'details': {'json_error': str(e)}
                }
            except Exception as e:
                return {
                    'valid': False,
                    'message': f"Request validation failed: {str(e)}"
                }
        
        return {'valid': True}
    
    async def _validate_response(self, response: Response):
        """Validate response structure (optional strict mode)."""
        try:
            if hasattr(response, 'body') and response.headers.get('content-type', '').startswith('application/json'):
                body = getattr(response, 'body', b'')
                if body:
                    json.loads(body.decode('utf-8'))  # Validate JSON structure
        except Exception as e:
            logger.warning(f"Response validation failed: {e}")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enhanced rate limiting middleware with burst handling and different limits."""
    
    def __init__(
        self, 
        app, 
        requests_per_minute: int = 60,
        burst_limit: int = 10,
        authenticated_multiplier: float = 2.0
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.authenticated_multiplier = authenticated_multiplier
        self.request_counts = {}
        self.burst_counts = {}
        self.blocked_ips = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply enhanced rate limiting with burst protection."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Check if IP is temporarily blocked
        if client_ip in self.blocked_ips:
            if current_time < self.blocked_ips[client_ip]:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": 429,
                            "message": "IP temporarily blocked due to rate limit violations",
                            "type": "rate_limit_exceeded"
                        }
                    },
                    headers={"Retry-After": str(int(self.blocked_ips[client_ip] - current_time))}
                )
            else:
                del self.blocked_ips[client_ip]
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute ago
        burst_cutoff = current_time - 10  # 10 seconds ago for burst detection
        
        self.request_counts = {
            ip: [timestamp for timestamp in timestamps if timestamp > cutoff_time]
            for ip, timestamps in self.request_counts.items()
        }
        
        self.burst_counts = {
            ip: [timestamp for timestamp in timestamps if timestamp > burst_cutoff]
            for ip, timestamps in self.burst_counts.items()
        }
        
        # Initialize counters for new IPs
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        if client_ip not in self.burst_counts:
            self.burst_counts[client_ip] = []
        
        # Count recent requests
        recent_requests = len(self.request_counts[client_ip])
        burst_requests = len(self.burst_counts[client_ip])
        
        # Determine rate limit based on authentication
        is_authenticated = request.headers.get("authorization") is not None
        effective_limit = self.requests_per_minute
        if is_authenticated:
            effective_limit = int(self.requests_per_minute * self.authenticated_multiplier)
        
        # Check burst limit first
        if burst_requests >= self.burst_limit:
            # Temporarily block aggressive IPs
            self.blocked_ips[client_ip] = current_time + 300  # 5 minute block
            
            global_error_handler.handle_error(
                error=Exception(f"Burst rate limit exceeded by {client_ip}"),
                operation="rate_limiting",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SECURITY,
                additional_info={
                    'client_ip': client_ip,
                    'burst_requests': burst_requests,
                    'requests_per_minute': recent_requests
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": 429,
                        "message": "Burst rate limit exceeded",
                        "type": "burst_limit_exceeded"
                    }
                },
                headers={"Retry-After": "300"}
            )
        
        # Check regular rate limit
        if recent_requests >= effective_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": 429,
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_exceeded"
                    }
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Type": "authenticated" if is_authenticated else "anonymous"
                }
            )
        
        # Record this request
        self.request_counts[client_ip].append(current_time)
        self.burst_counts[client_ip].append(current_time)
        
        response = await call_next(request)
        
        # Add comprehensive rate limit headers
        response.headers["X-RateLimit-Limit"] = str(effective_limit)
        response.headers["X-RateLimit-Remaining"] = str(effective_limit - recent_requests - 1)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        response.headers["X-RateLimit-Type"] = "authenticated" if is_authenticated else "anonymous"
        response.headers["X-Burst-Limit"] = str(self.burst_limit)
        response.headers["X-Burst-Remaining"] = str(self.burst_limit - burst_requests - 1)
        
        return response


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Circuit breaker middleware for protecting against cascading failures."""
    
    def __init__(
        self, 
        app,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        super().__init__(app)
        self.circuit_breakers = {}
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply circuit breaker pattern to endpoints."""
        endpoint = f"{request.method}:{request.url.path}"
        
        # Get or create circuit breaker for this endpoint
        if endpoint not in self.circuit_breakers:
            from ..utils.error_handling import CircuitBreaker
            self.circuit_breakers[endpoint] = CircuitBreaker(
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
                success_threshold=self.success_threshold
            )
        
        circuit_breaker = self.circuit_breakers[endpoint]
        
        try:
            # Use circuit breaker to protect the request
            async def protected_call():
                return await call_next(request)
            
            response = circuit_breaker.call(lambda: protected_call())
            
            # If we get here, the call was successful
            return await response
            
        except Exception as e:
            if "Circuit breaker is OPEN" in str(e):
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "code": 503,
                            "message": "Service temporarily unavailable",
                            "type": "circuit_breaker_open",
                            "details": {
                                "endpoint": endpoint,
                                "retry_after": self.recovery_timeout
                            }
                        }
                    },
                    headers={"Retry-After": str(int(self.recovery_timeout))}
                )
            else:
                # Log the error and re-raise
                global_error_handler.handle_error(
                    error=e,
                    operation="circuit_breaker_protected_call",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.SYSTEM,
                    additional_info={
                        'endpoint': endpoint,
                        'circuit_state': circuit_breaker.state.value
                    }
                )
                raise
    
    def get_circuit_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        status = {}
        for endpoint, breaker in self.circuit_breakers.items():
            status[endpoint] = {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'success_count': breaker.success_count,
                'last_failure_time': breaker.last_failure_time
            }
        return status


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
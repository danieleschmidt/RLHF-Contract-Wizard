"""
FastAPI application for RLHF-Contract-Wizard.

Main API application with all route handlers, middleware, and configuration.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..database.connection import initialize_connections, close_connections
from ..utils.helpers import setup_logging
from .routes import contracts, verification, deployment, health
from .middleware import SecurityMiddleware, LoggingMiddleware
from .dependencies import get_contract_service, get_verification_service, get_blockchain_service


# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting RLHF-Contract-Wizard API")
    try:
        await initialize_connections()
        logger.info("Database connections initialized")
    except Exception as e:
        logger.error(f"Failed to initialize connections: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RLHF-Contract-Wizard API")
    try:
        await close_connections()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="RLHF-Contract-Wizard API",
    description="API for managing RLHF reward contracts with legal compliance and formal verification",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.rlhf-contracts.org"]
)

app.add_middleware(SecurityMiddleware)
app.add_middleware(LoggingMiddleware)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            }
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "code": 400,
                "message": str(exc),
                "type": "validation_error"
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "server_error"
            }
        }
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(contracts.router, prefix="/api/v1", tags=["contracts"])
app.include_router(verification.router, prefix="/api/v1", tags=["verification"])
app.include_router(deployment.router, prefix="/api/v1", tags=["deployment"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RLHF-Contract-Wizard API",
        "version": "0.1.0",
        "description": "API for managing RLHF reward contracts",
        "docs": "/docs",
        "health": "/api/v1/health",
        "repository": "https://github.com/danieleschmidt/RLHF-Contract-Wizard"
    }


@app.get("/api/v1/info")
async def api_info():
    """API information endpoint."""
    return {
        "api_version": "v1",
        "features": [
            "contract_management",
            "formal_verification", 
            "blockchain_deployment",
            "multi_stakeholder_governance",
            "legal_blocks_dsl"
        ],
        "supported_networks": [
            "ethereum_mainnet",
            "ethereum_testnet",
            "polygon",
            "polygon_mumbai",
            "local"
        ],
        "verification_backends": [
            "z3",
            "lean4",
            "mock"
        ]
    }


def create_app() -> FastAPI:
    """Factory function to create FastAPI app."""
    return app


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("APP_ENV") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        workers=1 if os.getenv("APP_ENV") == "development" else int(os.getenv("API_WORKERS", "4"))
    )
"""
API routes package for RLHF-Contract-Wizard.

Contains all REST API route handlers organized by functional area.
"""

from .contracts import router as contracts_router
from .verification import router as verification_router
from .deployment import router as deployment_router
from .health import router as health_router

__all__ = [
    'contracts_router',
    'verification_router', 
    'deployment_router',
    'health_router'
]
"""
API package for RLHF-Contract-Wizard.

Contains FastAPI application, routes, middleware, and API utilities.
"""

from .main import app, create_app

__all__ = ['app', 'create_app']
"""
Test suite for quantum task planner.

Comprehensive test coverage for all quantum planning components including
unit tests, integration tests, performance tests, and property-based tests.
"""

# Test configuration and utilities
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Common test fixtures and utilities
from .fixtures import *
from .utils import *
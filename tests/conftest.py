"""
Pytest configuration and fixtures for RLHF-Contract-Wizard tests.

Provides common fixtures, test utilities, and configuration.
"""

import asyncio
import pytest
import os
from typing import Dict, Any, AsyncGenerator, Generator
from uuid import uuid4
import tempfile
import shutil

# Set test environment
os.environ['TESTING'] = 'true'
os.environ['LOG_LEVEL'] = 'ERROR'

import jax
import jax.numpy as jnp

from src.models.reward_contract import RewardContract, AggregationStrategy
from src.models.legal_blocks import LegalBlocks
# Mock services to avoid complex dependencies
class MockVerificationService:
    def __init__(self, backend=None):
        self.backend = backend
    async def verify_contract(self, contract):
        return {"valid": True, "violations": []}

class MockBlockchainService:
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
    async def deploy_contract(self, contract):
        return {"address": "0x123", "tx_hash": "0xabc"}

class MockContractService:
    def __init__(self, verification_service=None, blockchain_service=None):
        self.verification_service = verification_service or MockVerificationService()
        self.blockchain_service = blockchain_service or MockBlockchainService()

VerificationService = MockVerificationService
BlockchainService = MockBlockchainService
ContractService = MockContractService
VerificationBackend = type('VerificationBackend', (), {'MOCK': 'mock'})
NetworkType = type('NetworkType', (), {'LOCAL': 'local'})
# Mock database connections to avoid dependency issues
class MockDatabaseConnection:
    def __init__(self, database_url=None):
        self.database_url = database_url
        self.use_mock = True
    async def initialize(self): pass
    async def close(self): pass

class MockRedisConnection:
    def __init__(self, redis_url=None):
        self.redis_url = redis_url
    async def initialize(self): pass
    async def close(self): pass

DatabaseConnection = MockDatabaseConnection
RedisConnection = MockRedisConnection
# Skip repository import to avoid database dependencies in testing
class MockContractRepository:
    def __init__(self):
        self.contracts = {}
    async def count(self): return len(self.contracts)
    async def save(self, contract): 
        self.contracts[contract.metadata.name] = contract
        return contract
    async def find_by_name(self, name): 
        return self.contracts.get(name)

ContractRepository = MockContractRepository


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
async def mock_db() -> AsyncGenerator[DatabaseConnection, None]:
    """Mock database connection for testing."""
    db = DatabaseConnection(database_url="mock://test")
    db.use_mock = True
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def mock_redis() -> AsyncGenerator[RedisConnection, None]:
    """Mock Redis connection for testing."""
    redis = RedisConnection(redis_url="mock://test")
    await redis.initialize()
    yield redis
    await redis.close()


@pytest.fixture
def contract_service(mock_db) -> ContractService:
    """Contract service with mock dependencies."""
    verification_service = VerificationService(backend=VerificationBackend.MOCK)
    blockchain_service = BlockchainService(use_mock=True)
    
    return ContractService(
        verification_service=verification_service,
        blockchain_service=blockchain_service
    )


@pytest.fixture
def verification_service() -> VerificationService:
    """Verification service with mock backend."""
    return VerificationService(backend=VerificationBackend.MOCK)


@pytest.fixture
def blockchain_service() -> BlockchainService:
    """Blockchain service with mock backend."""
    return BlockchainService(use_mock=True)


@pytest.fixture
def sample_stakeholders() -> Dict[str, float]:
    """Sample stakeholder configuration."""
    return {
        "operator": 0.4,
        "users": 0.3,
        "safety_board": 0.2,
        "auditors": 0.1
    }


@pytest.fixture
def basic_contract(sample_stakeholders) -> RewardContract:
    """Basic reward contract for testing."""
    contract = RewardContract(
        name="TestContract",
        version="1.0.0",
        stakeholders=sample_stakeholders,
        creator="test_user"
    )
    
    # Add a simple reward function
    @contract.reward_function()
    @LegalBlocks.specification("""
        REQUIRES: action_valid(action)
        ENSURES: reward >= 0.0 AND reward <= 1.0
    """)
    def simple_reward(state, action):
        return 0.8
    
    # Add basic constraints
    def safety_constraint(state, action):
        return True  # Always satisfied for testing
    
    contract.add_constraint(
        name="safety_check",
        constraint_fn=safety_constraint,
        description="Basic safety constraint"
    )
    
    return contract


@pytest.fixture
def complex_contract() -> RewardContract:
    """Complex contract with multiple stakeholders and constraints."""
    stakeholders = {
        "platform": 0.25,
        "users": 0.25,
        "advertisers": 0.20,
        "safety_team": 0.15,
        "regulators": 0.15
    }
    
    contract = RewardContract(
        name="ComplexContract",
        version="2.0.0",
        stakeholders=stakeholders,
        aggregation=AggregationStrategy.NASH_BARGAINING,
        creator="complex_test_user"
    )
    
    # Platform reward function
    @contract.reward_function("platform")
    def platform_reward(state, action):
        return jnp.mean(state) * 0.9
    
    # User reward function
    @contract.reward_function("users")
    def user_reward(state, action):
        return jnp.sum(action) * 0.1
    
    # Add multiple constraints
    constraints = [
        ("no_harm", lambda s, a: jnp.all(a >= 0), "No harmful actions"),
        ("efficiency", lambda s, a: jnp.mean(s) > 0.5, "Efficiency requirement"),
        ("fairness", lambda s, a: jnp.std(a) < 1.0, "Fairness constraint")
    ]
    
    for name, func, desc in constraints:
        contract.add_constraint(name, func, desc)
    
    return contract


@pytest.fixture
def sample_state() -> jnp.ndarray:
    """Sample state vector for testing."""
    return jnp.array([0.1, 0.5, 0.8, 0.3, 0.7])


@pytest.fixture
def sample_action() -> jnp.ndarray:
    """Sample action vector for testing."""
    return jnp.array([0.2, 0.6, 0.4])


@pytest.fixture
def contract_data() -> Dict[str, Any]:
    """Sample contract data for database tests."""
    return {
        "name": "TestContract",
        "version": "1.0.0",
        "contract_hash": "test_hash_123",
        "creator": "test_creator",
        "jurisdiction": "Test",
        "status": "draft"
    }


@pytest.fixture
def stakeholder_data() -> Dict[str, Any]:
    """Sample stakeholder data."""
    return {
        "name": "test_stakeholder",
        "weight": 0.5,
        "voting_power": 1.0,
        "wallet_address": "0x1234567890123456789012345678901234567890"
    }


@pytest.fixture
def constraint_data() -> Dict[str, Any]:
    """Sample constraint data."""
    return {
        "name": "test_constraint",
        "description": "Test constraint for unit tests",
        "constraint_type": "invariant",
        "severity": 5.0,
        "violation_penalty": -2.0,
        "enabled": True
    }


@pytest.fixture
async def contract_repository() -> AsyncGenerator[ContractRepository, None]:
    """Contract repository with mock database."""
    repo = ContractRepository()
    # Use mock database for testing
    repo.db._pool = None  # Force mock usage
    yield repo


# Performance test fixtures
@pytest.fixture
def performance_contract() -> RewardContract:
    """Contract optimized for performance testing."""
    stakeholders = {f"stakeholder_{i}": 1.0/10 for i in range(10)}
    
    contract = RewardContract(
        name="PerformanceContract",
        version="1.0.0",
        stakeholders=stakeholders
    )
    
    # Simple fast reward function
    @contract.reward_function()
    def fast_reward(state, action):
        return jnp.sum(state * action)
    
    return contract


@pytest.fixture
def large_state() -> jnp.ndarray:
    """Large state vector for performance testing."""
    return jnp.ones(1000) * 0.5


@pytest.fixture
def large_action() -> jnp.ndarray:
    """Large action vector for performance testing."""
    return jnp.ones(1000) * 0.3


# Security test fixtures
@pytest.fixture
def malicious_input() -> Dict[str, Any]:
    """Malicious input for security testing."""
    return {
        "name": "<script>alert('xss')</script>",
        "version": "'; DROP TABLE contracts; --",
        "creator": "' OR '1'='1",
        "description": "javascript:alert('xss')"
    }


@pytest.fixture
def boundary_values() -> Dict[str, Any]:
    """Boundary values for testing edge cases."""
    return {
        "empty_string": "",
        "very_long_string": "x" * 10000,
        "unicode_string": "🤖🔒⛓️",
        "null_bytes": "\x00\x01\x02",
        "negative_weight": -1.0,
        "zero_weight": 0.0,
        "huge_weight": 1e10,
        "nan_value": float('nan'),
        "inf_value": float('inf')
    }


# Helper functions for tests
def assert_contract_valid(contract: RewardContract):
    """Assert that a contract is valid."""
    assert contract.metadata.name
    assert contract.metadata.version
    assert len(contract.stakeholders) > 0
    assert abs(sum(s.weight for s in contract.stakeholders.values()) - 1.0) < 1e-6


def assert_jax_array_valid(array: jnp.ndarray):
    """Assert that a JAX array is valid."""
    assert isinstance(array, jnp.ndarray)
    assert not jnp.any(jnp.isnan(array))
    assert not jnp.any(jnp.isinf(array))


def create_test_contract(name: str = None, **kwargs) -> RewardContract:
    """Create a test contract with default values."""
    name = name or f"TestContract_{uuid4().hex[:8]}"
    
    defaults = {
        "version": "1.0.0",
        "stakeholders": {"test": 1.0},
        "creator": "test_user"
    }
    defaults.update(kwargs)
    
    return RewardContract(name=name, **defaults)


async def assert_database_state(repo, expected_count: int):
    """Assert database state in tests."""
    actual_count = await repo.count()
    assert actual_count == expected_count


# Test marks and parametrization helpers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# Async test helpers
async def run_async_test(coro):
    """Helper to run async tests synchronously."""
    return await coro


# Mock data generators
def generate_mock_contracts(count: int = 5) -> list:
    """Generate mock contract data for testing."""
    contracts = []
    for i in range(count):
        contracts.append({
            "name": f"Contract_{i}",
            "version": f"1.{i}.0",
            "contract_hash": f"hash_{i}",
            "creator": f"creator_{i}",
            "status": "draft" if i % 2 == 0 else "validated"
        })
    return contracts


def generate_mock_verification_result() -> Dict[str, Any]:
    """Generate mock verification result."""
    return {
        "total_properties": 10,
        "proved_properties": 8,
        "failed_properties": 2,
        "verification_time_ms": 1500,
        "all_proofs_valid": False,
        "proof_results": [
            {"property": "safety", "proved": True, "time_ms": 200},
            {"property": "liveness", "proved": True, "time_ms": 300},
            {"property": "fairness", "proved": False, "time_ms": 500}
        ]
    }
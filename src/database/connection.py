"""
Database connection management for RLHF-Contract-Wizard.

Handles database connections, connection pooling, and transaction management.
"""

import os
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import asyncio
import time

try:
    import asyncpg
    import aiopg
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

try:
    import aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from ..utils.helpers import setup_logging


logger = setup_logging()


class DatabaseConnection:
    """
    Database connection manager with connection pooling.
    
    Supports PostgreSQL with asyncpg for high performance.
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 10,
        pool_timeout: int = 30,
        command_timeout: int = 60
    ):
        """
        Initialize database connection manager.
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Maximum connections in pool
            pool_timeout: Connection timeout in seconds
            command_timeout: SQL command timeout in seconds
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'postgresql://postgres:postgres@localhost:5432/rlhf_contracts'
        )
        self.pool_size = pool_size
        self.pool_timeout = pool_timeout
        self.command_timeout = command_timeout
        self._pool: Optional[asyncpg.Pool] = None
        self._closed = False
    
    async def initialize(self) -> None:
        """Initialize database connection pool."""
        if not HAS_POSTGRES:
            logger.warning("PostgreSQL dependencies not installed. Using mock database.")
            return
        
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=self.command_timeout,
                server_settings={
                    'application_name': 'rlhf-contract-wizard',
                    'timezone': 'UTC'
                }
            )
            logger.info(f"Database pool initialized with {self.pool_size} connections")
            
            # Test connection
            async with self._pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
                logger.info("Database connection test successful")
                
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool and not self._closed:
            await self._pool.close()
            self._closed = True
            logger.info("Database pool closed")
    
    @asynccontextmanager
    async def acquire_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Acquire database connection from pool.
        
        Yields:
            Database connection
        """
        if not self._pool:
            if HAS_POSTGRES:
                await self.initialize()
            else:
                yield MockConnection()
                return
        
        async with self._pool.acquire() as connection:
            yield connection
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Database transaction context manager.
        
        Yields:
            Database connection with active transaction
        """
        async with self.acquire_connection() as conn:
            async with conn.transaction():
                yield conn
    
    async def execute(self, query: str, *args) -> str:
        """Execute SQL command."""
        async with self.acquire_connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows."""
        async with self.acquire_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch single row."""
        async with self.acquire_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch single value."""
        async with self.acquire_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            start_time = time.time()
            async with self.acquire_connection() as conn:
                result = await conn.fetchval('SELECT version()')
                connection_time = time.time() - start_time
                
                return {
                    'status': 'healthy',
                    'version': result,
                    'connection_time_ms': round(connection_time * 1000, 2),
                    'pool_size': self._pool.get_size() if self._pool else 0,
                    'pool_idle': self._pool.get_idle_size() if self._pool else 0
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connection_time_ms': None
            }


class MockConnection:
    """Mock database connection for testing."""
    
    def __init__(self):
        self._data: Dict[str, list] = {}
    
    async def execute(self, query: str, *args) -> str:
        """Mock execute."""
        return "EXECUTE"
    
    async def fetch(self, query: str, *args) -> list:
        """Mock fetch."""
        return []
    
    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Mock fetchrow."""
        return None
    
    async def fetchval(self, query: str, *args) -> Any:
        """Mock fetchval."""
        if 'version()' in query:
            return 'PostgreSQL Mock 1.0'
        return None


class RedisConnection:
    """
    Redis connection manager for caching and session storage.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        encoding: str = 'utf-8',
        decode_responses: bool = True
    ):
        """
        Initialize Redis connection.
        
        Args:
            redis_url: Redis connection URL
            encoding: Response encoding
            decode_responses: Whether to decode responses
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.encoding = encoding
        self.decode_responses = decode_responses
        self._redis: Optional[aioredis.Redis] = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if not HAS_REDIS:
            logger.warning("Redis dependencies not installed. Using mock cache.")
            return
        
        try:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding=self.encoding,
                decode_responses=self.decode_responses
            )
            
            # Test connection
            await self._redis.ping()
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self._redis = MockRedis()
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis and hasattr(self._redis, 'close'):
            await self._redis.close()
            logger.info("Redis connection closed")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self._redis:
            await self.initialize()
        return await self._redis.get(key)
    
    async def set(
        self, 
        key: str, 
        value: str, 
        ex: Optional[int] = None
    ) -> bool:
        """Set value in Redis."""
        if not self._redis:
            await self.initialize()
        return await self._redis.set(key, value, ex=ex)
    
    async def delete(self, *keys: str) -> int:
        """Delete keys from Redis."""
        if not self._redis:
            await self.initialize()
        return await self._redis.delete(*keys)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._redis:
            await self.initialize()
        return bool(await self._redis.exists(key))
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration."""
        if not self._redis:
            await self.initialize()
        return await self._redis.expire(key, seconds)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            if not self._redis:
                await self.initialize()
            
            start_time = time.time()
            await self._redis.ping()
            ping_time = time.time() - start_time
            
            info = await self._redis.info()
            
            return {
                'status': 'healthy',
                'ping_time_ms': round(ping_time * 1000, 2),
                'version': info.get('redis_version', 'unknown'),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


class MockRedis:
    """Mock Redis for testing."""
    
    def __init__(self):
        self._data: Dict[str, str] = {}
        self._expiry: Dict[str, float] = {}
    
    async def ping(self) -> str:
        return "PONG"
    
    async def get(self, key: str) -> Optional[str]:
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._data[key]
            del self._expiry[key]
            return None
        return self._data.get(key)
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._data[key] = value
        if ex:
            self._expiry[key] = time.time() + ex
        return True
    
    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                deleted += 1
            if key in self._expiry:
                del self._expiry[key]
        return deleted
    
    async def exists(self, key: str) -> bool:
        return key in self._data
    
    async def expire(self, key: str, seconds: int) -> bool:
        if key in self._data:
            self._expiry[key] = time.time() + seconds
            return True
        return False
    
    async def info(self) -> Dict[str, Any]:
        return {
            'redis_version': 'Mock 1.0',
            'used_memory_human': '1MB',
            'connected_clients': 1
        }


# Global connection instances
db_connection = DatabaseConnection()
redis_connection = RedisConnection()


async def initialize_connections():
    """Initialize all database connections."""
    await db_connection.initialize()
    await redis_connection.initialize()


async def close_connections():
    """Close all database connections."""
    await db_connection.close()
    await redis_connection.close()
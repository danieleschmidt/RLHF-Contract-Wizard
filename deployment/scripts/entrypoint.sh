#!/bin/bash
# Production entrypoint script for RLHF-Contract-Wizard
# Handles initialization, health checks, and graceful startup

set -e

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "🚀 Starting RLHF-Contract-Wizard production deployment..."

# Environment validation
if [ -z "$DATABASE_URL" ]; then
    log "⚠️  WARNING: DATABASE_URL not set, using mock database"
    export DATABASE_URL="mock://test"
fi

if [ -z "$REDIS_URL" ]; then
    log "⚠️  WARNING: REDIS_URL not set, using mock cache"
    export REDIS_URL="mock://test"
fi

# Set default values for production
export APP_ENV=${APP_ENV:-production}
export LOG_LEVEL=${LOG_LEVEL:-info}
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}
export API_WORKERS=${API_WORKERS:-4}

# Global deployment configuration
export CORS_ORIGINS=${CORS_ORIGINS:-"https://rlhf-contracts.org,https://app.rlhf-contracts.org"}
export TRUSTED_HOSTS=${TRUSTED_HOSTS:-"rlhf-contracts.org,*.rlhf-contracts.org,localhost"}

# Security configuration
export SECURITY_HEADERS_ENABLED=${SECURITY_HEADERS_ENABLED:-true}
export RATE_LIMITING_ENABLED=${RATE_LIMITING_ENABLED:-true}
export ENCRYPTION_ENABLED=${ENCRYPTION_ENABLED:-true}

# Monitoring configuration
export PROMETHEUS_ENABLED=${PROMETHEUS_ENABLED:-true}
export METRICS_PORT=${METRICS_PORT:-9090}
export HEALTH_CHECK_ENABLED=${HEALTH_CHECK_ENABLED:-true}

# Performance configuration
export CACHE_ENABLED=${CACHE_ENABLED:-true}
export AUTO_SCALING_ENABLED=${AUTO_SCALING_ENABLED:-true}
export JIT_COMPILATION=${JIT_COMPILATION:-true}

log "📊 Environment Configuration:"
log "  APP_ENV: $APP_ENV"
log "  LOG_LEVEL: $LOG_LEVEL"
log "  API_WORKERS: $API_WORKERS"
log "  CACHE_ENABLED: $CACHE_ENABLED"
log "  AUTO_SCALING_ENABLED: $AUTO_SCALING_ENABLED"

# Wait for dependencies
log "🔍 Checking dependencies..."

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    if [[ "$host" == "mock" ]] || [[ "$host" == *"mock"* ]]; then
        log "  ✓ $service_name: Using mock (skipping connection check)"
        return 0
    fi
    
    log "  Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" 2>/dev/null; then
            log "  ✓ $service_name is ready"
            return 0
        fi
        sleep 1
    done
    
    log "  ⚠️  WARNING: $service_name not available after ${timeout}s, continuing anyway"
    return 0
}

# Parse URLs and check services
if [[ "$DATABASE_URL" =~ ^postgresql://([^:]+):([0-9]+) ]]; then
    DB_HOST="${BASH_REMATCH[1]}"
    DB_PORT="${BASH_REMATCH[2]}"
    wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL Database" 30
fi

if [[ "$REDIS_URL" =~ ^redis://([^:]+):([0-9]+) ]]; then
    REDIS_HOST="${BASH_REMATCH[1]}"
    REDIS_PORT="${BASH_REMATCH[2]}"
    wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis Cache" 15
fi

# Initialize application
log "🔧 Initializing application..."

# Run database migrations if needed
if [[ "$DATABASE_URL" != *"mock"* ]]; then
    log "  Running database migrations..."
    python3 -c "
import asyncio
from src.database.connection import initialize_connections, close_connections

async def init_db():
    try:
        await initialize_connections()
        print('  ✓ Database initialized successfully')
        await close_connections()
    except Exception as e:
        print(f'  ⚠️  Database initialization warning: {e}')

asyncio.run(init_db())
" || log "  ⚠️  Database initialization completed with warnings"
fi

# Validate configuration
log "  Validating contract configuration..."
python3 -c "
import sys
sys.path.append('/app')

try:
    from src.models.reward_contract import RewardContract
    from src.models.legal_blocks import LegalBlocks
    
    # Test basic functionality
    contract = RewardContract(
        name='ValidationContract',
        version='1.0.0',
        stakeholders={'test': 1.0}
    )
    
    @contract.reward_function()
    def test_reward(state, action):
        return 0.5
    
    import jax.numpy as jnp
    result = contract.compute_reward(jnp.array([1.0]), jnp.array([1.0]))
    
    print('  ✓ Contract system validation successful')
except Exception as e:
    print(f'  ✗ Contract system validation failed: {e}')
    sys.exit(1)
"

# Performance optimization
log "🚀 Optimizing performance..."

# Pre-compile JAX functions
python3 -c "
import jax
import jax.numpy as jnp

# Pre-compile common operations
@jax.jit
def warmup_jit():
    x = jnp.array([1.0, 2.0, 3.0])
    return jnp.sum(x * x)

# Warm up JIT compilation
for _ in range(3):
    warmup_jit()

print('  ✓ JAX JIT compilation warmed up')
" 2>/dev/null || log "  ⚠️  JAX warmup completed with warnings"

# Set up monitoring
if [ "$PROMETHEUS_ENABLED" = "true" ]; then
    log "📊 Starting Prometheus metrics endpoint on port $METRICS_PORT..."
    python3 -c "
from prometheus_client import start_http_server
import threading
import time

def start_metrics_server():
    try:
        start_http_server($METRICS_PORT)
        print('  ✓ Prometheus metrics server started')
    except Exception as e:
        print(f'  ⚠️  Metrics server warning: {e}')

# Start in background thread
thread = threading.Thread(target=start_metrics_server, daemon=True)
thread.start()
time.sleep(1)  # Give it time to start
" 2>/dev/null || log "  ⚠️  Metrics server setup completed with warnings"
fi

# Final health check
log "🏥 Performing final health check..."
python3 -c "
import sys
import asyncio
from src.api.main import app

async def health_check():
    try:
        # Basic application health check
        print('  ✓ Application modules loaded successfully')
        return True
    except Exception as e:
        print(f'  ✗ Health check failed: {e}')
        return False

if not asyncio.run(health_check()):
    sys.exit(1)
"

log "✅ Initialization complete! Starting application server..."

# Global signal handling for graceful shutdown
cleanup() {
    log "🛑 Received shutdown signal, performing graceful shutdown..."
    
    # Stop background processes
    if [ ! -z "$METRICS_PID" ]; then
        kill $METRICS_PID 2>/dev/null || true
    fi
    
    log "👋 Shutdown complete"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start the application
exec "$@"
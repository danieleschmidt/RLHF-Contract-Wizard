#!/bin/bash

# RLHF-Contract-Wizard Development Container Post-Start Script

set -e

echo "🔄 Starting development services..."

# Start Redis if available
if command -v redis-server &> /dev/null; then
    echo "📊 Starting Redis..."
    redis-server --daemonize yes --port 6379 || true
fi

# Start PostgreSQL if available
if command -v pg_ctl &> /dev/null; then
    echo "🗄️ Starting PostgreSQL..."
    sudo service postgresql start || true
fi

# Start local blockchain for development
echo "⛓️ Starting local blockchain..."
if command -v ganache-cli &> /dev/null; then
    ganache-cli \
        --host 0.0.0.0 \
        --port 8545 \
        --deterministic \
        --accounts 10 \
        --defaultBalanceEther 1000 \
        --gasLimit 12000000 \
        --blockTime 1 \
        --fork http://localhost:8545 \
        --quiet &
    
    # Wait for blockchain to start
    sleep 3
    echo "⛓️ Local blockchain started on port 8545"
fi

# Set up git hooks if they don't exist
if [ ! -f ".git/hooks/pre-commit" ]; then
    echo "🔧 Setting up git hooks..."
    pre-commit install --install-hooks || true
fi

# Check service health
echo "🏥 Checking service health..."

# Check Redis
if redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "⚠️ Redis is not responding"
fi

# Check PostgreSQL
if pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
    echo "✅ PostgreSQL is running"
else
    echo "⚠️ PostgreSQL is not responding"
fi

# Check blockchain
if curl -s -X POST \
    -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
    http://localhost:8545 >/dev/null 2>&1; then
    echo "✅ Blockchain is running"
else
    echo "⚠️ Blockchain is not responding"
fi

echo "🎉 Development environment is ready!"
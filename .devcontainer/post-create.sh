#!/bin/bash

# RLHF-Contract-Wizard Development Container Post-Create Script

set -e

echo "🚀 Setting up RLHF-Contract-Wizard development environment..."

# Install Python dependencies in development mode
echo "📦 Installing Python dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Install Node.js dependencies for smart contract development
echo "📦 Installing Node.js dependencies..."
npm install

# Initialize Hardhat if not already done
if [ ! -f "hardhat.config.js" ]; then
    echo "⚙️ Initializing Hardhat..."
    npx hardhat init --yes
fi

# Compile smart contracts
echo "🔨 Compiling smart contracts..."
npx hardhat compile

# Set up database schema
echo "🗄️ Setting up database schema..."
if command -v psql &> /dev/null; then
    # Check if PostgreSQL is running and create schema
    export PGPASSWORD=postgres
    if psql -h localhost -p 5432 -U postgres -c '\q' 2>/dev/null; then
        psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE rlhf_contracts;" || true
        psql -h localhost -p 5432 -U postgres -d rlhf_contracts -f src/database/schema.sql || true
    else
        echo "⚠️ PostgreSQL not running, skipping database setup"
    fi
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
python -m pytest tests/unit/models/test_reward_contract.py -v || true

# Set up environment file
echo "📋 Creating .env file..."
cat > .env << EOF
# Development Environment Configuration
DEVELOPMENT=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rlhf_contracts
REDIS_URL=redis://localhost:6379/0
BLOCKCHAIN_NETWORK=localhost
BLOCKCHAIN_URL=http://localhost:8545
PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET=dev-secret-key-change-in-production
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
EOF

# Create local development directories
echo "📁 Creating development directories..."
mkdir -p logs
mkdir -p data/contracts
mkdir -p data/backups
mkdir -p .pytest_cache
mkdir -p coverage

# Set proper permissions
chmod +x scripts/*.sh 2>/dev/null || true

echo "✅ Development environment setup complete!"
echo "🎯 Quick start commands:"
echo "  - Run API server: make dev"
echo "  - Run tests: make test"
echo "  - Format code: make format"
echo "  - Deploy contracts: make deploy-local"
echo "  - View logs: make logs"

echo ""
echo "🔗 Useful URLs:"
echo "  - API Documentation: http://localhost:8000/docs"
echo "  - Test Coverage: open coverage/htmlcov/index.html"
echo "  - Contract Explorer: http://localhost:8080"
#!/bin/bash

# Development container setup script for RLHF-Contract-Wizard

set -e

echo "🚀 Setting up RLHF-Contract-Wizard development environment..."

# Update package lists
sudo apt-get update

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    htop \
    tree \
    jq \
    postgresql-client \
    redis-tools

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install JAX with CPU support (CUDA version would be different)
pip install --upgrade "jax[cpu]"

# Install development tools
pip install \
    black \
    isort \
    flake8 \
    pylint \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pre-commit \
    jupyter \
    ipykernel

# Install blockchain development tools
echo "⛓️  Installing blockchain tools..."
npm install -g \
    @openzeppelin/cli \
    truffle \
    ganache-cli \
    @hardhat-runner/hardhat

# Install IPFS (for distributed storage)
echo "📡 Installing IPFS..."
wget -O ipfs.tar.gz https://dist.ipfs.io/go-ipfs/v0.14.0/go-ipfs_v0.14.0_linux-amd64.tar.gz
tar -xzf ipfs.tar.gz
sudo mv go-ipfs/ipfs /usr/local/bin/
rm -rf go-ipfs ipfs.tar.gz

# Initialize IPFS (if not already done)
if [ ! -d ~/.ipfs ]; then
    ipfs init
fi

# Install project dependencies
echo "📋 Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
fi

# Install package in development mode
pip install -e .

# Install npm dependencies
if [ -f "package.json" ]; then
    npm install
fi

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
mkdir -p logs data contracts/deployed

# Setup environment
echo "⚙️  Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file from template. Please configure as needed."
fi

# Initialize database schema (if applicable)
if [ -f "scripts/init_db.py" ]; then
    echo "🗄️  Initializing database schema..."
    python scripts/init_db.py
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
if [ -d "tests" ]; then
    python -m pytest tests/ --tb=short || echo "⚠️  Some tests failed - this is expected for a new setup"
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Configure your .env file with appropriate settings"
echo "2. Start developing with: code ."
echo "3. Run tests with: pytest"
echo "4. Start local blockchain: ganache-cli"
echo "5. Deploy contracts: npm run deploy:local"
echo ""
echo "📚 Documentation: docs/guides/development.md"
echo "🐛 Issues: https://github.com/danieleschmidt/RLHF-Contract-Wizard/issues"
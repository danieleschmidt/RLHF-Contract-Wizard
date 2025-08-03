# RLHF-Contract-Wizard Development Container
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm (for blockchain tooling)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Hardhat and blockchain development tools
RUN npm install -g @hardhat-runner/hardhat \
    @nomicfoundation/hardhat-toolbox \
    @openzeppelin/contracts \
    @openzeppelin/hardhat-upgrades \
    truffle \
    ganache-cli \
    solhint

# Install Python development tools
RUN pip install --upgrade pip \
    && pip install \
    black \
    flake8 \
    pylint \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pre-commit \
    jupyter \
    ipython

# Set up development environment
WORKDIR /workspaces/RLHF-Contract-Wizard

# Copy requirements and install dependencies
COPY requirements*.txt ./
RUN pip install -r requirements-dev.txt

# Set environment variables
ENV PYTHONPATH=/workspaces/RLHF-Contract-Wizard/src
ENV DEVELOPMENT=true
ENV LOG_LEVEL=DEBUG

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Configure git (will be overridden by user config)
RUN git config --global user.name "Developer" \
    && git config --global user.email "developer@example.com" \
    && git config --global init.defaultBranch main

# Install Python packages as user
RUN pip install --user \
    jax[cpu] \
    optax \
    flax \
    chex \
    web3 \
    fastapi \
    uvicorn \
    sqlalchemy \
    alembic \
    redis \
    psycopg2-binary \
    python-dotenv \
    click \
    rich \
    typer

EXPOSE 8000 8080 3000 5432 6379

CMD ["sleep", "infinity"]
# Multi-stage Dockerfile for RLHF-Contract-Wizard Production Deployment
# Optimized for global deployment with security, performance, and monitoring

# Stage 1: Base image with system dependencies
FROM python:3.12-slim as base

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    TZ=UTC

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    pkg-config \
    netcat-openbsd \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r rlhf && useradd -r -g rlhf -d /app -s /bin/bash rlhf

# Stage 2: Python dependencies
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: Application build
FROM dependencies as build

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY deployment/ ./deployment/
COPY tests/ ./tests/
COPY *.py ./
COPY *.md ./
COPY *.toml ./
COPY *.yaml ./
COPY *.yml ./

# Create necessary directories
RUN mkdir -p logs data cache config

# Set ownership to non-root user
RUN chown -R rlhf:rlhf /app

# Stage 4: Production image
FROM build as production

# Switch to non-root user
USER rlhf

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Set up entrypoint script
COPY deployment/scripts/entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
USER root
RUN chmod +x /app/entrypoint.sh
USER rlhf

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 5: Development image (optional)
FROM build as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov black isort flake8 mypy

# Switch to non-root user
USER rlhf

# Development command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
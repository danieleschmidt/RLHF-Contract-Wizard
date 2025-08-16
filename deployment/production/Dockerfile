# Multi-stage production Dockerfile for RLHF-Contract-Wizard

# Build stage
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 rlhf \
    && useradd --uid 1000 --gid rlhf --shell /bin/bash --create-home rlhf

# Copy Python packages from builder
COPY --from=builder /root/.local /home/rlhf/.local

# Set environment variables
ENV PATH=/home/rlhf/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=rlhf:rlhf . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config \
    && chown -R rlhf:rlhf /app

# Switch to non-root user
USER rlhf

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
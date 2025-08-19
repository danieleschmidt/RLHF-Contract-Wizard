#!/bin/bash
# Production Deployment Script for RLHF Contract Wizard
# Generated automatically by deployment validator

set -e

echo "🚀 Starting RLHF Contract Wizard Production Deployment"
echo "=================================================="

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not found. Please install Docker." >&2; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl not found. Please install kubectl." >&2; exit 1; }

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t rlhf-contract-wizard:latest .

# Deploy to Kubernetes
echo "☸️  Deploying to Kubernetes..."
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/hpa.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available deployment/rlhf-contract-wizard -n rlhf-contracts --timeout=300s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n rlhf-contracts
kubectl get services -n rlhf-contracts

echo "🎉 Deployment completed successfully!"
echo "📊 Access the application at the service endpoint."

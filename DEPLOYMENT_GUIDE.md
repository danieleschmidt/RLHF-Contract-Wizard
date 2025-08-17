# 🚀 RLHF-Contract-Wizard Production Deployment Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying RLHF-Contract-Wizard in production environments with enterprise-grade reliability, security, and scalability.

## 🏗️ Architecture Overview

### Production Architecture
```
Internet
    ↓
Load Balancer (Nginx)
    ↓
API Gateway + Rate Limiting
    ↓
RLHF-Contract-API (3+ instances)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   PostgreSQL    │      Redis      │   Monitoring    │
│   (Primary)     │    (Cache)      │   (Prometheus)  │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                    ↓                    ↓
Worker Processes    Background Tasks    Grafana Dashboards
```

## 🔧 Prerequisites

### System Requirements
- **CPU**: 8+ cores recommended
- **Memory**: 16GB+ RAM
- **Storage**: 100GB+ SSD
- **Network**: 1Gbps+ bandwidth
- **OS**: Ubuntu 20.04 LTS or CentOS 8+

### Required Software
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Git
- curl/wget

### Optional (for Kubernetes)
- kubectl
- Helm 3.0+
- Kubernetes cluster 1.24+

## 🚀 Quick Start (Docker Compose)

### 1. Clone and Setup
```bash
git clone https://github.com/your-org/rlhf-contract-wizard.git
cd rlhf-contract-wizard

# Set environment variables
export DB_PASSWORD="your_secure_db_password"
export REDIS_PASSWORD="your_secure_redis_password"
export GRAFANA_PASSWORD="your_secure_grafana_password"
```

### 2. Deploy
```bash
# Make deployment script executable
chmod +x deployment/scripts/deploy.sh

# Run deployment
./deployment/scripts/deploy.sh
```

### 3. Verify Deployment
```bash
# Check service status
cd deployment/production
docker-compose -f docker-compose.prod.yml ps

# Test API
curl http://localhost/api/v1/health

# Access monitoring
open http://localhost:3000  # Grafana (admin/your_grafana_password)
open http://localhost:9090  # Prometheus
```

## 🐳 Docker Compose Deployment

### Configuration
The production deployment uses `deployment/production/docker-compose.prod.yml` with:

- **API Service**: 3 replicas with auto-scaling
- **PostgreSQL**: Persistent data with backup
- **Redis**: Session storage and caching
- **Nginx**: Load balancer and SSL termination
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **Workers**: Background processing

### Environment Variables
Create `.env` file in `deployment/production/`:
```env
# Database
DB_PASSWORD=your_secure_password
DATABASE_URL=postgresql://rlhf_user:${DB_PASSWORD}@postgres:5432/rlhf_contracts

# Cache
REDIS_PASSWORD=your_secure_password
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379

# Monitoring
GRAFANA_PASSWORD=your_secure_password

# Application
ENV=production
LOG_LEVEL=INFO
MAX_WORKERS=4
ENABLE_AUTO_SCALING=true
ENABLE_MONITORING=true

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
```

### SSL Configuration
Place SSL certificates in `deployment/production/nginx/ssl/`:
```
nginx/ssl/
├── certificate.crt
├── private.key
└── ca-bundle.crt
```

## ☸️ Kubernetes Deployment

### 1. Prepare Namespace
```bash
kubectl create namespace rlhf-contracts
kubectl config set-context --current --namespace=rlhf-contracts
```

### 2. Create Secrets
```bash
kubectl create secret generic rlhf-secrets \
  --from-literal=database-url="postgresql://user:pass@postgres:5432/rlhf_contracts" \
  --from-literal=redis-url="redis://:pass@redis:6379"
```

### 3. Deploy
```bash
# Deploy core services
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/deployment.yaml

# Deploy auto-scaling
kubectl apply -f deployment/k8s/hpa.yaml

# Verify deployment
kubectl get pods
kubectl get services
kubectl get ingress
```

## 📊 Monitoring & Observability

### Metrics
- **Application Metrics**: `/metrics` endpoint
- **System Metrics**: CPU, Memory, Disk, Network
- **Business Metrics**: Contract processing rates, verification success rates
- **Performance Metrics**: Response times, throughput, error rates

### Dashboards
Access Grafana at `http://localhost:3000` with pre-configured dashboards:
- System Overview
- API Performance
- Contract Processing
- Error Analysis
- Resource Utilization

### Alerts
Configure alerting for:
- High error rates (>5%)
- Slow response times (>1s)
- Resource exhaustion (CPU >80%, Memory >85%)
- Service downtime
- Failed verifications

## 🔒 Security Configuration

### SSL/TLS
- Enable HTTPS with valid certificates
- Configure HSTS headers
- Use TLS 1.2+ only

### Authentication
- JWT-based API authentication
- Role-based access control (RBAC)
- API key management

### Network Security
- Firewall rules (only required ports)
- Private network for internal services
- VPN access for management

### Data Protection
- Encryption at rest (database)
- Encryption in transit (HTTPS/TLS)
- Regular security updates

## 🔄 Backup & Disaster Recovery

### Database Backup
```bash
# Daily automated backup
docker exec postgres pg_dump -U rlhf_user rlhf_contracts > backup_$(date +%Y%m%d).sql

# Restore from backup
docker exec -i postgres psql -U rlhf_user rlhf_contracts < backup_file.sql
```

### Configuration Backup
```bash
# Backup configurations
tar -czf config_backup_$(date +%Y%m%d).tar.gz deployment/production/config/

# Backup Docker volumes
docker run --rm -v postgres_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/postgres_$(date +%Y%m%d).tar.gz /data
```

### Disaster Recovery Plan
1. **Backup Verification**: Test backups weekly
2. **Failover Procedures**: Document recovery steps
3. **RTO/RPO Targets**: 4 hours RTO, 1 hour RPO
4. **Emergency Contacts**: Maintain contact list

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale API instances
docker-compose -f docker-compose.prod.yml up -d --scale rlhf-api=5

# Kubernetes auto-scaling
kubectl autoscale deployment rlhf-contract-api --cpu-percent=70 --min=3 --max=10
```

### Vertical Scaling
Adjust resource limits in:
- `docker-compose.prod.yml` (Docker Compose)
- `deployment.yaml` (Kubernetes)

### Database Scaling
- Read replicas for read-heavy workloads
- Connection pooling (PgBouncer)
- Partitioning for large datasets

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs rlhf-api

# Check resource usage
docker stats

# Check disk space
df -h
```

#### Database Connection Issues
```bash
# Test connectivity
docker exec postgres pg_isready -U rlhf_user

# Check configuration
docker exec postgres cat /var/lib/postgresql/data/postgresql.conf
```

#### Performance Issues
```bash
# Check metrics
curl http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])

# Analyze slow queries
docker exec postgres psql -U rlhf_user -c "SELECT * FROM pg_stat_activity;"
```

### Log Analysis
```bash
# Application logs
docker-compose -f docker-compose.prod.yml logs -f rlhf-api

# System logs
journalctl -u docker

# Nginx access logs
tail -f deployment/production/logs/nginx/access.log
```

## 🔧 Maintenance

### Regular Maintenance Tasks
- **Weekly**: Review monitoring dashboards
- **Weekly**: Check log files for errors
- **Monthly**: Update dependencies
- **Monthly**: Review security configurations
- **Quarterly**: Performance optimization review

### Updates
```bash
# Update application
git pull origin main
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Database migrations
docker exec rlhf-api python -m alembic upgrade head
```

### Health Checks
```bash
# Automated health check script
curl -f http://localhost/api/v1/health || echo "API unhealthy"
curl -f http://localhost:9090/-/healthy || echo "Prometheus unhealthy"
curl -f http://localhost:3000/api/health || echo "Grafana unhealthy"
```

## 🆘 Support

### Emergency Procedures
1. **Service Down**: Check logs, restart services
2. **Database Issues**: Check connections, run diagnostics
3. **High Load**: Scale services, investigate bottlenecks
4. **Security Incident**: Isolate, analyze, report

### Contact Information
- **DevOps Team**: devops@yourcompany.com
- **Security Team**: security@yourcompany.com
- **On-Call**: +1-XXX-XXX-XXXX

## 📚 Additional Resources
- [API Documentation](API.md)
- [Configuration Reference](CONFIG.md)
- [Security Guide](SECURITY.md)
- [Performance Tuning](PERFORMANCE.md)

---

**Last Updated**: December 2024
**Version**: 1.0.0
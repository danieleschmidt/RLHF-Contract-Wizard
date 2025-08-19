#!/usr/bin/env python3
"""
Production Deployment Checklist and Validation Script

This script validates all production deployment requirements and generates
a comprehensive deployment readiness report for the RLHF Contract Wizard.
"""

import os
import json
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess

class ProductionDeploymentValidator:
    """Validates production deployment readiness."""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.deployment_checks = {}
        self.overall_status = "UNKNOWN"
    
    def check_docker_configuration(self) -> Dict[str, Any]:
        """Check Docker configuration."""
        docker_files = [
            "Dockerfile",
            "docker-compose.yml",
            "deployment/docker-compose.prod.yml"
        ]
        
        existing_files = [f for f in docker_files if os.path.exists(f)]
        
        return {
            "name": "Docker Configuration",
            "status": "PASS" if len(existing_files) >= 2 else "FAIL",
            "details": {
                "dockerfile_present": os.path.exists("Dockerfile"),
                "docker_compose_present": os.path.exists("docker-compose.yml"),
                "production_compose_present": os.path.exists("deployment/docker-compose.prod.yml"),
                "files_found": existing_files
            }
        }
    
    def check_kubernetes_configuration(self) -> Dict[str, Any]:
        """Check Kubernetes configuration."""
        k8s_files = [
            "deployment/k8s/deployment.yaml",
            "deployment/k8s/namespace.yaml",
            "deployment/k8s/hpa.yaml",
            "deployment/kubernetes/api-deployment.yaml"
        ]
        
        existing_files = [f for f in k8s_files if os.path.exists(f)]
        
        return {
            "name": "Kubernetes Configuration",
            "status": "PASS" if len(existing_files) >= 3 else "FAIL",
            "details": {
                "deployment_yaml_present": any("deployment.yaml" in f for f in existing_files),
                "namespace_yaml_present": any("namespace.yaml" in f for f in existing_files),
                "hpa_present": any("hpa.yaml" in f for f in existing_files),
                "files_found": existing_files
            }
        }
    
    def check_environment_configuration(self) -> Dict[str, Any]:
        """Check environment configuration."""
        env_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "package.json"
        ]
        
        existing_files = [f for f in env_files if os.path.exists(f)]
        
        # Check if requirements.txt has key dependencies
        has_key_deps = False
        if os.path.exists("requirements.txt"):
            try:
                with open("requirements.txt", "r") as f:
                    content = f.read()
                    key_deps = ["fastapi", "jax", "numpy"]
                    has_key_deps = any(dep in content.lower() for dep in key_deps)
            except:
                pass
        
        return {
            "name": "Environment Configuration",
            "status": "PASS" if len(existing_files) >= 2 and has_key_deps else "FAIL",
            "details": {
                "requirements_present": os.path.exists("requirements.txt"),
                "dev_requirements_present": os.path.exists("requirements-dev.txt"),
                "package_json_present": os.path.exists("package.json"),
                "key_dependencies_found": has_key_deps,
                "files_found": existing_files
            }
        }
    
    def check_monitoring_configuration(self) -> Dict[str, Any]:
        """Check monitoring configuration."""
        monitoring_files = [
            "deployment/monitoring/prometheus.yml",
            "deployment/monitoring/prometheus-config.yaml",
            "src/monitoring/comprehensive_monitoring.py"
        ]
        
        existing_files = [f for f in monitoring_files if os.path.exists(f)]
        
        return {
            "name": "Monitoring Configuration",
            "status": "PASS" if len(existing_files) >= 2 else "FAIL",
            "details": {
                "prometheus_config_present": any("prometheus" in f for f in existing_files),
                "monitoring_code_present": os.path.exists("src/monitoring/comprehensive_monitoring.py"),
                "files_found": existing_files
            }
        }
    
    def check_security_configuration(self) -> Dict[str, Any]:
        """Check security configuration."""
        security_files = [
            "src/security/security_framework.py",
            "SECURITY.md"
        ]
        
        existing_files = [f for f in security_files if os.path.exists(f)]
        
        return {
            "name": "Security Configuration",
            "status": "PASS" if len(existing_files) >= 2 else "FAIL",
            "details": {
                "security_framework_present": os.path.exists("src/security/security_framework.py"),
                "security_docs_present": os.path.exists("SECURITY.md"),
                "files_found": existing_files
            }
        }
    
    def check_database_configuration(self) -> Dict[str, Any]:
        """Check database configuration."""
        db_files = [
            "src/database/schema.sql",
            "src/database/connection.py",
            "src/database/migrations"
        ]
        
        existing_files = [f for f in db_files if os.path.exists(f)]
        
        return {
            "name": "Database Configuration",
            "status": "PASS" if len(existing_files) >= 2 else "FAIL",
            "details": {
                "schema_present": os.path.exists("src/database/schema.sql"),
                "connection_present": os.path.exists("src/database/connection.py"),
                "migrations_present": os.path.exists("src/database/migrations"),
                "files_found": existing_files
            }
        }
    
    def check_api_configuration(self) -> Dict[str, Any]:
        """Check API configuration."""
        api_files = [
            "src/api/main.py",
            "src/api/routes",
            "src/api/middleware.py"
        ]
        
        existing_files = [f for f in api_files if os.path.exists(f)]
        
        return {
            "name": "API Configuration",
            "status": "PASS" if len(existing_files) >= 2 else "FAIL",
            "details": {
                "main_api_present": os.path.exists("src/api/main.py"),
                "routes_present": os.path.exists("src/api/routes"),
                "middleware_present": os.path.exists("src/api/middleware.py"),
                "files_found": existing_files
            }
        }
    
    def check_scaling_configuration(self) -> Dict[str, Any]:
        """Check scaling configuration."""
        scaling_files = [
            "src/scaling/intelligent_scaling.py",
            "deployment/k8s/hpa.yaml"
        ]
        
        existing_files = [f for f in scaling_files if os.path.exists(f)]
        
        return {
            "name": "Scaling Configuration",
            "status": "PASS" if len(existing_files) >= 1 else "FAIL",
            "details": {
                "intelligent_scaling_present": os.path.exists("src/scaling/intelligent_scaling.py"),
                "hpa_config_present": os.path.exists("deployment/k8s/hpa.yaml"),
                "files_found": existing_files
            }
        }
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation."""
        doc_files = [
            "README.md",
            "DEPLOYMENT.md",
            "DEPLOYMENT_GUIDE.md",
            "ARCHITECTURE.md"
        ]
        
        existing_files = [f for f in doc_files if os.path.exists(f)]
        
        return {
            "name": "Documentation",
            "status": "PASS" if len(existing_files) >= 3 else "FAIL",
            "details": {
                "readme_present": os.path.exists("README.md"),
                "deployment_docs_present": os.path.exists("DEPLOYMENT.md") or os.path.exists("DEPLOYMENT_GUIDE.md"),
                "architecture_docs_present": os.path.exists("ARCHITECTURE.md"),
                "files_found": existing_files
            }
        }
    
    def check_quality_gates(self) -> Dict[str, Any]:
        """Check quality gates status."""
        quality_files = [
            "quality_gate_final_report.json",
            "test_results.json"
        ]
        
        quality_passed = False
        if os.path.exists("quality_gate_final_report.json"):
            try:
                with open("quality_gate_final_report.json", "r") as f:
                    report = json.load(f)
                    quality_passed = report.get("overall_status") == "PASS"
            except:
                pass
        
        return {
            "name": "Quality Gates",
            "status": "PASS" if quality_passed else "FAIL",
            "details": {
                "quality_report_present": os.path.exists("quality_gate_final_report.json"),
                "test_results_present": os.path.exists("test_results.json"),
                "quality_gates_passed": quality_passed
            }
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all deployment checks."""
        checks = [
            self.check_docker_configuration,
            self.check_kubernetes_configuration,
            self.check_environment_configuration,
            self.check_monitoring_configuration,
            self.check_security_configuration,
            self.check_database_configuration,
            self.check_api_configuration,
            self.check_scaling_configuration,
            self.check_documentation,
            self.check_quality_gates
        ]
        
        results = {}
        passed_checks = 0
        total_checks = len(checks)
        
        for check in checks:
            result = check()
            check_name = result["name"].lower().replace(" ", "_")
            results[check_name] = result
            
            if result["status"] == "PASS":
                passed_checks += 1
        
        self.deployment_checks = results
        self.overall_status = "PASS" if passed_checks == total_checks else "FAIL"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "project": "RLHF Contract Wizard",
            "deployment_checks": results,
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": total_checks - passed_checks,
                "pass_rate": f"{(passed_checks / total_checks) * 100:.1f}%"
            },
            "overall_status": self.overall_status
        }
    
    def generate_deployment_script(self) -> str:
        """Generate deployment script."""
        script = """#!/bin/bash
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
"""
        return script

def print_deployment_report(report: Dict[str, Any]):
    """Print formatted deployment report."""
    
    print("🚀 PRODUCTION DEPLOYMENT READINESS REPORT")
    print("=" * 80)
    print(f"Project: {report['project']}")
    print(f"Generated: {report['timestamp']}")
    print()
    
    # Print each deployment check
    for check_id, check in report["deployment_checks"].items():
        status_emoji = "✅" if check["status"] == "PASS" else "❌"
        print(f"{status_emoji} {check['name']}")
        print(f"   Status: {check['status']}")
        
        if "details" in check:
            print("   Details:")
            for key, value in check["details"].items():
                if isinstance(value, bool):
                    detail_emoji = "✓" if value else "✗"
                    print(f"     {detail_emoji} {key.replace('_', ' ').title()}")
                elif isinstance(value, list):
                    if value:
                        print(f"     • {key.replace('_', ' ').title()}: {', '.join(value)}")
                    else:
                        print(f"     • {key.replace('_', ' ').title()}: None found")
                else:
                    print(f"     • {key.replace('_', ' ').title()}: {value}")
        print()
    
    # Print summary
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("-" * 50)
    summary = report["summary"]
    print(f"Total Checks: {summary['total_checks']}")
    print(f"Passed: {summary['passed_checks']}")
    print(f"Failed: {summary['failed_checks']}")
    print(f"Pass Rate: {summary['pass_rate']}")
    print()
    
    # Print overall status
    overall_emoji = "🎉" if report["overall_status"] == "PASS" else "⚠️"
    print(f"{overall_emoji} OVERALL DEPLOYMENT STATUS: {report['overall_status']}")
    
    if report["overall_status"] == "PASS":
        print("\n✅ All deployment checks passed! Ready for production deployment.")
        print("📝 Run the generated deployment script to deploy to production.")
    else:
        print("\n❌ Some deployment checks failed. Address issues before deployment.")

def main():
    """Main function."""
    print("Validating production deployment readiness...")
    
    validator = ProductionDeploymentValidator()
    report = validator.run_all_checks()
    
    # Save report
    with open("production_deployment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate deployment script
    deployment_script = validator.generate_deployment_script()
    with open("deploy_production.sh", "w") as f:
        f.write(deployment_script)
    
    # Make deployment script executable
    os.chmod("deploy_production.sh", 0o755)
    
    # Print report
    print_deployment_report(report)
    
    print(f"\n📁 Full report saved to: production_deployment_report.json")
    print(f"📝 Deployment script saved to: deploy_production.sh")
    
    return 0 if report["overall_status"] == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
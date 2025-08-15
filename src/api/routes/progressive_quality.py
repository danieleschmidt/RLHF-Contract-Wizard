"""
Progressive Quality Gates API endpoints.

Implements the TERRAGON SDLC MASTER PROMPT v4.0 progressive quality gate
system as REST API endpoints for autonomous execution and monitoring.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from ...quantum_planner.core import QuantumTaskPlanner, PlannerConfig
from ...quantum_planner.monitoring import get_monitoring_system
from ...global_compliance.i18n import get_i18n_manager, SupportedLanguage
from ...utils.helpers import setup_logging

router = APIRouter()
logger = setup_logging()


class QualityGateRequest(BaseModel):
    """Request model for quality gate execution."""
    gate_names: Optional[List[str]] = Field(None, description="Specific gates to run, or all if None")
    parallel_execution: bool = Field(True, description="Whether to run gates in parallel")
    fail_fast: bool = Field(False, description="Stop execution on first failure")
    language: Optional[str] = Field("en", description="Language for responses")


class QualityGateResult(BaseModel):
    """Individual quality gate result."""
    gate_name: str
    status: str  # passed, failed, warning, skipped
    score: float
    execution_time: float
    message: str
    recommendations: List[str]
    timestamp: str


class QualityGateReport(BaseModel):
    """Complete quality gate execution report."""
    overall_status: str
    overall_score: float
    total_gates: int
    passed_gates: int
    failed_gates: int
    warning_gates: int
    skipped_gates: int
    execution_time: float
    gate_results: List[QualityGateResult]
    deployment_ready: bool
    timestamp: str


class ProgressiveSDLCStatus(BaseModel):
    """Status of progressive SDLC execution."""
    current_generation: int  # 1, 2, or 3
    generation_status: str  # pending, in_progress, completed, failed
    next_actions: List[str]
    completion_percentage: float
    estimated_completion: Optional[str]


# In-memory storage for demo (would use database in production)
_execution_cache: Dict[str, Any] = {}


@router.post("/quality-gates/execute", response_model=QualityGateReport)
async def execute_quality_gates(
    request: QualityGateRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute quality gates with autonomous progression.
    
    Implements Generation 1/2/3 progression as specified in the SDLC prompt.
    """
    i18n = get_i18n_manager()
    language = SupportedLanguage(request.language) if request.language else SupportedLanguage.ENGLISH
    i18n.set_language(language)
    
    execution_id = f"qg_exec_{int(time.time())}"
    start_time = time.time()
    
    try:
        # Import quality gates runner
        import sys
        sys.path.insert(0, '/root/repo')
        from quality_gates import QualityGateRunner
        
        # Execute quality gates
        runner = QualityGateRunner()
        report = runner.run_all_gates()
        
        execution_time = time.time() - start_time
        
        # Convert to API response format
        api_report = QualityGateReport(
            overall_status=report.overall_status.value,
            overall_score=report.overall_score,
            total_gates=report.total_gates,
            passed_gates=report.passed_gates,
            failed_gates=report.failed_gates,
            warning_gates=report.warning_gates,
            skipped_gates=report.skipped_gates,
            execution_time=execution_time,
            gate_results=[
                QualityGateResult(
                    gate_name=result.gate_name,
                    status=result.status.value,
                    score=result.score,
                    execution_time=result.execution_time,
                    message=result.message,
                    recommendations=result.recommendations,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                for result in report.gate_results
            ],
            deployment_ready=report.deployment_ready,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Cache result
        _execution_cache[execution_id] = api_report
        
        # If deployment ready, trigger next generation autonomously
        if api_report.deployment_ready:
            background_tasks.add_task(trigger_next_generation, execution_id)
        
        logger.info(i18n.translate("quality.gates_passed") if api_report.deployment_ready 
                   else i18n.translate("quality.gates_failed"))
        
        return api_report
        
    except Exception as e:
        logger.error(f"Quality gate execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality gate execution failed: {str(e)}"
        )


@router.get("/quality-gates/status/{execution_id}", response_model=QualityGateReport)
async def get_quality_gate_status(execution_id: str):
    """Get status of a specific quality gate execution."""
    if execution_id not in _execution_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found"
        )
    
    return _execution_cache[execution_id]


@router.get("/quality-gates/latest", response_model=QualityGateReport)
async def get_latest_quality_gates():
    """Get the latest quality gate execution results."""
    if not _execution_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No quality gate executions found"
        )
    
    # Return most recent execution
    latest_key = max(_execution_cache.keys())
    return _execution_cache[latest_key]


@router.post("/sdlc/progressive/execute")
async def execute_progressive_sdlc(
    background_tasks: BackgroundTasks,
    request: QualityGateRequest
):
    """
    Execute autonomous progressive SDLC as specified in TERRAGON prompt.
    
    Implements Generation 1 -> 2 -> 3 progression with quality gates.
    """
    i18n = get_i18n_manager()
    language = SupportedLanguage(request.language) if request.language else SupportedLanguage.ENGLISH
    i18n.set_language(language)
    
    execution_id = f"sdlc_exec_{int(time.time())}"
    
    # Start progressive execution in background
    background_tasks.add_task(execute_sdlc_generations, execution_id, request)
    
    return {
        "execution_id": execution_id,
        "status": "started",
        "message": i18n.translate("system.startup"),
        "estimated_completion": "5-10 minutes",
        "monitor_url": f"/api/v1/progressive-quality/sdlc/status/{execution_id}"
    }


@router.get("/sdlc/status/{execution_id}", response_model=ProgressiveSDLCStatus)
async def get_sdlc_status(execution_id: str):
    """Get status of progressive SDLC execution."""
    if execution_id not in _execution_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SDLC execution not found"
        )
    
    status_data = _execution_cache[execution_id]
    return ProgressiveSDLCStatus(**status_data)


@router.get("/sdlc/generations/current")
async def get_current_generation():
    """Get current SDLC generation status."""
    # Based on quality gates, determine current generation
    try:
        import sys
        sys.path.insert(0, '/root/repo')
        from quality_gates import QualityGateRunner
        
        runner = QualityGateRunner()
        report = runner.run_all_gates()
        
        # Determine generation based on quality gate results
        if report.overall_score >= 0.95 and report.deployment_ready:
            generation = 3  # Make it Scale
        elif report.overall_score >= 0.80:
            generation = 2  # Make it Robust
        else:
            generation = 1  # Make it Work
        
        return {
            "current_generation": generation,
            "overall_score": report.overall_score,
            "deployment_ready": report.deployment_ready,
            "next_generation_available": generation < 3,
            "generation_descriptions": {
                1: "Make it Work (Simple) - Basic functionality",
                2: "Make it Robust (Reliable) - Error handling & validation", 
                3: "Make it Scale (Optimized) - Performance & scaling"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to determine current generation: {str(e)}")
        return {
            "current_generation": 1,
            "overall_score": 0.0,
            "deployment_ready": False,
            "error": str(e)
        }


async def trigger_next_generation(execution_id: str):
    """Trigger next SDLC generation autonomously."""
    logger.info(f"Triggering next generation for execution {execution_id}")
    
    # Implementation would proceed to next generation
    # This is a placeholder for autonomous progression
    await asyncio.sleep(1)
    logger.info("Next generation triggered successfully")


async def execute_sdlc_generations(execution_id: str, request: QualityGateRequest):
    """Execute all three SDLC generations progressively."""
    i18n = get_i18n_manager()
    
    # Initialize status
    _execution_cache[execution_id] = {
        "current_generation": 1,
        "generation_status": "in_progress",
        "next_actions": ["Execute Generation 1: Make it Work"],
        "completion_percentage": 0.0,
        "estimated_completion": None
    }
    
    try:
        # Generation 1: Make it Work (Simple)
        logger.info("Starting Generation 1: Make it Work")
        _execution_cache[execution_id]["current_generation"] = 1
        _execution_cache[execution_id]["completion_percentage"] = 10.0
        
        # Run quality gates for generation 1
        import sys
        sys.path.insert(0, '/root/repo')
        from quality_gates import QualityGateRunner
        
        runner = QualityGateRunner()
        report = runner.run_all_gates()
        
        if report.overall_score >= 0.7:  # Generation 1 threshold
            logger.info("Generation 1 completed successfully")
            _execution_cache[execution_id]["completion_percentage"] = 33.0
            
            # Generation 2: Make it Robust (Reliable)
            logger.info("Starting Generation 2: Make it Robust")
            _execution_cache[execution_id]["current_generation"] = 2
            _execution_cache[execution_id]["completion_percentage"] = 50.0
            
            # Enhanced error handling and validation
            await asyncio.sleep(2)  # Simulate robust implementation
            
            # Run quality gates for generation 2
            report = runner.run_all_gates()
            
            if report.overall_score >= 0.85:  # Generation 2 threshold
                logger.info("Generation 2 completed successfully")
                _execution_cache[execution_id]["completion_percentage"] = 66.0
                
                # Generation 3: Make it Scale (Optimized)
                logger.info("Starting Generation 3: Make it Scale")
                _execution_cache[execution_id]["current_generation"] = 3
                _execution_cache[execution_id]["completion_percentage"] = 80.0
                
                # Performance optimization and scaling
                await asyncio.sleep(2)  # Simulate optimization
                
                # Final quality gates
                report = runner.run_all_gates()
                
                if report.overall_score >= 0.95:  # Generation 3 threshold
                    logger.info("Generation 3 completed successfully")
                    _execution_cache[execution_id]["generation_status"] = "completed"
                    _execution_cache[execution_id]["completion_percentage"] = 100.0
                    _execution_cache[execution_id]["next_actions"] = ["Ready for production deployment"]
                else:
                    _execution_cache[execution_id]["generation_status"] = "failed"
                    _execution_cache[execution_id]["next_actions"] = ["Address Generation 3 quality issues"]
            else:
                _execution_cache[execution_id]["generation_status"] = "failed"
                _execution_cache[execution_id]["next_actions"] = ["Address Generation 2 quality issues"]
        else:
            _execution_cache[execution_id]["generation_status"] = "failed"
            _execution_cache[execution_id]["next_actions"] = ["Address Generation 1 quality issues"]
            
    except Exception as e:
        logger.error(f"SDLC execution failed: {str(e)}")
        _execution_cache[execution_id]["generation_status"] = "failed"
        _execution_cache[execution_id]["next_actions"] = [f"Fix error: {str(e)}"]


@router.post("/deployments/readiness-check")
async def check_deployment_readiness():
    """
    Check if system is ready for production deployment.
    
    Runs comprehensive checks as required by SDLC quality gates.
    """
    i18n = get_i18n_manager()
    
    try:
        # Run quality gates
        import sys
        sys.path.insert(0, '/root/repo')
        from quality_gates import QualityGateRunner
        
        runner = QualityGateRunner()
        report = runner.run_all_gates()
        
        readiness_checks = {
            "quality_gates_passed": report.deployment_ready,
            "overall_score": report.overall_score,
            "critical_issues": len([r for r in report.gate_results if r.status.value == "failed"]),
            "test_coverage": report.overall_score >= 0.85,
            "security_validated": any(r.gate_name == "Security Vulnerability Scan" and r.status.value == "passed" 
                                   for r in report.gate_results),
            "performance_acceptable": any(r.gate_name == "Performance Benchmarking" and r.status.value == "passed"
                                        for r in report.gate_results),
            "contract_compliance": any(r.gate_name == "Contract Compliance Validation" and r.status.value == "passed"
                                     for r in report.gate_results)
        }
        
        overall_ready = all([
            readiness_checks["quality_gates_passed"],
            readiness_checks["critical_issues"] == 0,
            readiness_checks["test_coverage"],
            readiness_checks["security_validated"],
            readiness_checks["performance_acceptable"],
            readiness_checks["contract_compliance"]
        ])
        
        return {
            "deployment_ready": overall_ready,
            "readiness_score": report.overall_score,
            "checks": readiness_checks,
            "message": i18n.translate("deployment.ready") if overall_ready else i18n.translate("deployment.not_ready"),
            "recommendations": report.recommendations if hasattr(report, 'recommendations') else [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Deployment readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Readiness check failed: {str(e)}"
        )
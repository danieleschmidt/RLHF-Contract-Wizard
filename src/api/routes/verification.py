"""
Verification API routes.

Provides endpoints for formal verification of contract properties.
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, Body

from ..dependencies import (
    get_verification_service,
    get_verification_repository,
    validate_verification_backend
)
from ...services.verification_service import VerificationService, VerificationBackend
from ...repositories.contract_repository import VerificationRepository


router = APIRouter()


@router.post("/contracts/{contract_id}/verify")
async def verify_contract(
    contract_id: UUID = Path(..., description="Contract ID"),
    backend: Optional[str] = Query("z3", description="Verification backend"),
    properties: Optional[List[str]] = Body(None, description="Specific properties to verify"),
    timeout: int = Body(60, ge=1, le=300, description="Timeout in seconds"),
    verification_service: VerificationService = Depends(get_verification_service)
) -> Dict[str, Any]:
    """
    Verify contract properties using formal methods.
    
    Performs formal verification of safety and correctness properties.
    """
    try:
        # Validate backend
        verification_backend = validate_verification_backend(backend)
        
        # Update service backend if different
        if verification_service.backend != verification_backend:
            verification_service.backend = verification_backend
            verification_service._initialize_backend()
        
        # For now, create a mock contract for verification
        # In a real implementation, this would load the contract from the database
        from ...models.reward_contract import RewardContract
        
        mock_contract = RewardContract(
            name=f"Contract_{contract_id}",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @mock_contract.reward_function()
        def test_reward(state, action):
            return 0.5
        
        # Perform verification
        result = verification_service.verify_contract(
            contract=mock_contract,
            properties=properties,
            timeout=timeout
        )
        
        return {
            "contract_id": str(contract_id),
            "verification_time": result.verification_time,
            "total_properties": result.total_properties,
            "proved_properties": result.proved_properties,
            "failed_properties": result.failed_properties,
            "all_proofs_valid": result.all_proofs_valid,
            "backend": result.proof_results[0].backend.value if result.proof_results else backend,
            "proof_results": [
                {
                    "property_name": proof.property_name,
                    "proved": proof.proved,
                    "verification_time": proof.verification_time,
                    "error_message": proof.error_message,
                    "proof_trace": proof.proof_trace
                }
                for proof in result.proof_results
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )


@router.post("/contracts/{contract_id}/verify-property")
async def verify_single_property(
    contract_id: UUID = Path(..., description="Contract ID"),
    property_name: str = Body(..., description="Property name to verify"),
    property_spec: Optional[str] = Body(None, description="Property specification"),
    backend: Optional[str] = Body("z3", description="Verification backend"),
    timeout: int = Body(30, ge=1, le=300, description="Timeout in seconds"),
    verification_service: VerificationService = Depends(get_verification_service)
) -> Dict[str, Any]:
    """
    Verify a specific property of a contract.
    
    Performs verification of a single named property.
    """
    try:
        # Validate backend
        verification_backend = validate_verification_backend(backend)
        
        # Create mock contract
        from ...models.reward_contract import RewardContract
        
        mock_contract = RewardContract(
            name=f"Contract_{contract_id}",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @mock_contract.reward_function()
        def test_reward(state, action):
            return 0.5
        
        # Verify single property
        result = verification_service.verify_property(
            contract=mock_contract,
            property_name=property_name,
            property_spec=property_spec,
            timeout=timeout
        )
        
        return {
            "contract_id": str(contract_id),
            "property_name": result.property_name,
            "proved": result.proved,
            "verification_time": result.verification_time,
            "backend": result.backend.value,
            "proof_trace": result.proof_trace,
            "counterexample": result.counterexample,
            "error_message": result.error_message,
            "explanation": verification_service.explain_proof(result)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Property verification failed: {str(e)}"
        )


@router.get("/contracts/{contract_id}/verification-history")
async def get_verification_history(
    contract_id: UUID = Path(..., description="Contract ID"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    verification_repo: VerificationRepository = Depends(get_verification_repository)
) -> List[Dict[str, Any]]:
    """
    Get verification history for a contract.
    
    Returns past verification results and their outcomes.
    """
    try:
        results = await verification_repo.get_verification_by_contract(contract_id)
        
        # Convert to response format
        verification_history = []
        for result in results[-limit:]:  # Get most recent results
            verification_history.append({
                "id": result["id"],
                "backend": result["backend"],
                "total_properties": result["total_properties"],
                "proved_properties": result["proved_properties"],
                "failed_properties": result["failed_properties"],
                "verification_time_ms": result["verification_time_ms"],
                "all_proofs_valid": result["all_proofs_valid"],
                "verified_at": result["verified_at"],
                "proof_results": result["proof_results"]
            })
        
        return verification_history
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get verification history: {str(e)}"
        )


@router.get("/contracts/{contract_id}/verification-report")
async def get_verification_report(
    contract_id: UUID = Path(..., description="Contract ID"),
    verification_id: Optional[UUID] = Query(None, description="Specific verification ID"),
    verification_repo: VerificationRepository = Depends(get_verification_repository),
    verification_service: VerificationService = Depends(get_verification_service)
) -> Dict[str, Any]:
    """
    Generate a verification report for a contract.
    
    Returns a comprehensive report of verification results.
    """
    try:
        if verification_id:
            # Get specific verification result
            result = await verification_repo.get_by_id(verification_id)
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Verification result not found"
                )
        else:
            # Get latest verification result
            result = await verification_repo.get_latest_verification(contract_id)
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No verification results found for this contract"
                )
        
        # Create verification result object for report generation
        from ...services.verification_service import VerificationResult, ProofResult, VerificationBackend
        
        proof_results = []
        for proof_data in result["proof_results"]:
            proof_results.append(ProofResult(
                property_name=proof_data["property"],
                proved=proof_data["proved"],
                verification_time=proof_data["time_ms"] / 1000.0,
                backend=VerificationBackend(result["backend"]),
                proof_trace=proof_data.get("proof_trace"),
                counterexample=proof_data.get("counterexample"),
                error_message=proof_data.get("error")
            ))
        
        verification_result = VerificationResult(
            contract_hash="mock_hash",
            verification_time=result["verification_time_ms"] / 1000.0,
            total_properties=result["total_properties"],
            proved_properties=result["proved_properties"],
            failed_properties=result["failed_properties"],
            proof_results=proof_results,
            all_proofs_valid=result["all_proofs_valid"]
        )
        
        # Generate report
        report = verification_service.get_verification_report(verification_result)
        
        return {
            "contract_id": str(contract_id),
            "verification_id": result["id"],
            "report": report,
            "generated_at": result["verified_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate verification report: {str(e)}"
        )


@router.post("/contracts/{contract_id}/generate-counterexample")
async def generate_counterexample(
    contract_id: UUID = Path(..., description="Contract ID"),
    property_name: str = Body(..., description="Property that failed"),
    verification_service: VerificationService = Depends(get_verification_service)
) -> Dict[str, Any]:
    """
    Generate counterexample for a failed property.
    
    Creates concrete input values that violate the specified property.
    """
    try:
        # Create mock contract
        from ...models.reward_contract import RewardContract
        
        mock_contract = RewardContract(
            name=f"Contract_{contract_id}",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        # Generate counterexample
        counterexample = verification_service.generate_counterexample(
            contract=mock_contract,
            property_name=property_name
        )
        
        return {
            "contract_id": str(contract_id),
            "property_name": property_name,
            "counterexample": counterexample,
            "generated_at": "2025-01-20T12:00:00Z"  # Mock timestamp
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate counterexample: {str(e)}"
        )


@router.get("/verification/stats")
async def get_verification_stats(
    verification_repo: VerificationRepository = Depends(get_verification_repository)
) -> Dict[str, Any]:
    """
    Get overall verification statistics.
    
    Returns system-wide verification metrics and performance data.
    """
    try:
        stats = await verification_repo.get_verification_stats()
        
        return {
            "total_verifications": stats.get("total_verifications", 0),
            "successful_verifications": stats.get("successful_verifications", 0),
            "success_rate": (
                stats.get("successful_verifications", 0) / 
                max(stats.get("total_verifications", 1), 1)
            ),
            "average_verification_time_ms": stats.get("avg_verification_time", 0),
            "average_proof_rate": stats.get("avg_proof_rate", 0),
            "backends_used": ["z3", "mock"],  # This would come from actual data
            "most_common_properties": [
                "reward_bounded",
                "constraint_satisfaction", 
                "stakeholder_fairness"
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get verification stats: {str(e)}"
        )


@router.get("/verification/backends")
async def list_verification_backends() -> Dict[str, Any]:
    """
    List available verification backends.
    
    Returns information about supported verification systems.
    """
    backends = []
    
    for backend in VerificationBackend:
        backend_info = {
            "name": backend.value,
            "display_name": backend.value.upper(),
            "description": _get_backend_description(backend),
            "capabilities": _get_backend_capabilities(backend),
            "performance": _get_backend_performance(backend)
        }
        backends.append(backend_info)
    
    return {
        "backends": backends,
        "default": "z3",
        "recommended": "z3"
    }


def _get_backend_description(backend: VerificationBackend) -> str:
    """Get description for verification backend."""
    descriptions = {
        VerificationBackend.Z3: "Microsoft Z3 SMT solver for automated theorem proving",
        VerificationBackend.LEAN: "Lean 4 proof assistant for interactive theorem proving",
        VerificationBackend.CBMC: "CBMC bounded model checker for program verification",
        VerificationBackend.MOCK: "Mock backend for testing and development"
    }
    return descriptions.get(backend, "Unknown backend")


def _get_backend_capabilities(backend: VerificationBackend) -> List[str]:
    """Get capabilities for verification backend."""
    capabilities = {
        VerificationBackend.Z3: ["smt_solving", "quantifiers", "arithmetic", "bit_vectors"],
        VerificationBackend.LEAN: ["interactive_proofs", "dependent_types", "tactics"],
        VerificationBackend.CBMC: ["bounded_model_checking", "c_programs", "safety_properties"],
        VerificationBackend.MOCK: ["testing", "development", "simulation"]
    }
    return capabilities.get(backend, [])


def _get_backend_performance(backend: VerificationBackend) -> Dict[str, str]:
    """Get performance characteristics for backend."""
    performance = {
        VerificationBackend.Z3: {"speed": "fast", "scalability": "high", "automation": "high"},
        VerificationBackend.LEAN: {"speed": "slow", "scalability": "medium", "automation": "low"},
        VerificationBackend.CBMC: {"speed": "medium", "scalability": "medium", "automation": "high"},
        VerificationBackend.MOCK: {"speed": "instant", "scalability": "unlimited", "automation": "perfect"}
    }
    return performance.get(backend, {})
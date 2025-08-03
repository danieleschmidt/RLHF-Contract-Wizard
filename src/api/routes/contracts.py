"""
Contract management API routes.

Provides REST endpoints for creating, managing, and querying reward contracts.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from fastapi.responses import JSONResponse

from ..dependencies import (
    get_contract_service,
    get_contract_repository,
    get_stakeholder_repository,
    get_constraint_repository,
    get_pagination_params,
    get_search_params,
    validate_contract_name,
    validate_version,
    PaginationParams,
    SearchParams
)
from ..schemas.contract_schemas import (
    ContractCreateRequest,
    ContractResponse,
    ContractUpdateRequest,
    StakeholderCreateRequest,
    ConstraintCreateRequest,
    ContractListResponse
)
from ...services.contract_service import ContractService, ContractValidationError
from ...repositories.contract_repository import ContractRepository


router = APIRouter()


@router.post("/contracts", response_model=ContractResponse, status_code=status.HTTP_201_CREATED)
async def create_contract(
    contract_data: ContractCreateRequest,
    contract_service: ContractService = Depends(get_contract_service)
) -> ContractResponse:
    """
    Create a new reward contract.
    
    Creates a new RLHF reward contract with specified stakeholders and configuration.
    """
    try:
        contract = contract_service.create_contract(
            name=contract_data.name,
            version=contract_data.version,
            stakeholders=contract_data.stakeholders,
            creator=contract_data.creator,
            jurisdiction=contract_data.jurisdiction,
            regulatory_framework=contract_data.regulatory_framework,
            aggregation_strategy=contract_data.aggregation_strategy
        )
        
        return ContractResponse.from_contract(contract)
        
    except ContractValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create contract: {str(e)}"
        )


@router.get("/contracts", response_model=ContractListResponse)
async def list_contracts(
    pagination: PaginationParams = Depends(get_pagination_params),
    search: SearchParams = Depends(get_search_params),
    status_filter: Optional[str] = Query(None, description="Filter by contract status"),
    creator_filter: Optional[str] = Query(None, description="Filter by creator"),
    contract_repo: ContractRepository = Depends(get_contract_repository)
) -> ContractListResponse:
    """
    List all contracts with pagination and filtering.
    
    Returns a paginated list of contracts with optional filtering by status, creator, etc.
    """
    try:
        # Build filters
        filters = {}
        if status_filter:
            filters["status"] = status_filter
        if creator_filter:
            filters["creator"] = creator_filter
        
        # Get contracts
        if search.query:
            contracts = await contract_repo.search_contracts(
                search_term=search.query,
                limit=pagination.limit
            )
            total_count = len(contracts)  # Simplified for search
        else:
            contracts = await contract_repo.get_all(
                limit=pagination.limit,
                offset=pagination.offset,
                filters=filters
            )
            total_count = await contract_repo.count(filters=filters)
        
        # Convert to response format
        contract_responses = [
            ContractResponse.from_dict(contract) for contract in contracts
        ]
        
        return ContractListResponse(
            contracts=contract_responses,
            total=total_count,
            page=pagination.page,
            size=pagination.size,
            pages=(total_count + pagination.size - 1) // pagination.size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list contracts: {str(e)}"
        )


@router.get("/contracts/{contract_id}", response_model=ContractResponse)
async def get_contract(
    contract_id: UUID = Path(..., description="Contract ID"),
    contract_repo: ContractRepository = Depends(get_contract_repository)
) -> ContractResponse:
    """
    Get a specific contract by ID.
    
    Returns detailed information about a specific contract including stakeholders and constraints.
    """
    try:
        contract = await contract_repo.get_by_id(contract_id)
        
        if not contract:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Contract not found"
            )
        
        return ContractResponse.from_dict(contract)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get contract: {str(e)}"
        )


@router.put("/contracts/{contract_id}", response_model=ContractResponse)
async def update_contract(
    contract_id: UUID = Path(..., description="Contract ID"),
    update_data: ContractUpdateRequest = Body(...),
    contract_service: ContractService = Depends(get_contract_service)
) -> ContractResponse:
    """
    Update an existing contract.
    
    Updates contract metadata, stakeholder weights, or other configuration.
    """
    try:
        # Convert UUID to string for service layer
        contract_id_str = str(contract_id)
        
        updated_contract = contract_service.update_contract(
            contract_id_str,
            update_data.dict(exclude_unset=True)
        )
        
        return ContractResponse.from_contract(updated_contract)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update contract: {str(e)}"
        )


@router.delete("/contracts/{contract_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_contract(
    contract_id: UUID = Path(..., description="Contract ID"),
    contract_repo: ContractRepository = Depends(get_contract_repository)
):
    """
    Delete a contract.
    
    Permanently deletes a contract and all associated data.
    """
    try:
        deleted = await contract_repo.delete(contract_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Contract not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete contract: {str(e)}"
        )


@router.post("/contracts/{contract_id}/validate", response_model=Dict[str, Any])
async def validate_contract(
    contract_id: UUID = Path(..., description="Contract ID"),
    contract_service: ContractService = Depends(get_contract_service)
) -> Dict[str, Any]:
    """
    Validate a contract.
    
    Performs comprehensive validation including constraint checking and formal verification.
    """
    try:
        # Get contract from service registry
        contract_id_str = str(contract_id)
        contract = contract_service.get_contract(contract_id_str)
        
        if not contract:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Contract not found"
            )
        
        validation_result = contract_service.validate_contract(contract)
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate contract: {str(e)}"
        )


@router.get("/contracts/{contract_id}/stakeholders")
async def get_contract_stakeholders(
    contract_id: UUID = Path(..., description="Contract ID"),
    stakeholder_repo = Depends(get_stakeholder_repository)
) -> List[Dict[str, Any]]:
    """
    Get all stakeholders for a contract.
    
    Returns list of stakeholders with their weights and voting power.
    """
    try:
        stakeholders = await stakeholder_repo.get_stakeholders_by_contract(contract_id)
        return stakeholders
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stakeholders: {str(e)}"
        )


@router.post("/contracts/{contract_id}/stakeholders", status_code=status.HTTP_201_CREATED)
async def add_stakeholder(
    contract_id: UUID = Path(..., description="Contract ID"),
    stakeholder_data: StakeholderCreateRequest = Body(...),
    stakeholder_repo = Depends(get_stakeholder_repository)
) -> Dict[str, Any]:
    """
    Add a new stakeholder to a contract.
    
    Creates a new stakeholder with specified weight and voting power.
    """
    try:
        stakeholder = await stakeholder_repo.create_stakeholder(
            contract_id=contract_id,
            name=stakeholder_data.name,
            weight=stakeholder_data.weight,
            voting_power=stakeholder_data.voting_power,
            wallet_address=stakeholder_data.wallet_address,
            preferences=stakeholder_data.preferences,
            contact_info=stakeholder_data.contact_info
        )
        
        return stakeholder
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add stakeholder: {str(e)}"
        )


@router.get("/contracts/{contract_id}/constraints")
async def get_contract_constraints(
    contract_id: UUID = Path(..., description="Contract ID"),
    enabled_only: bool = Query(True, description="Return only enabled constraints"),
    constraint_repo = Depends(get_constraint_repository)
) -> List[Dict[str, Any]]:
    """
    Get all constraints for a contract.
    
    Returns list of constraints with their specifications and status.
    """
    try:
        constraints = await constraint_repo.get_constraints_by_contract(
            contract_id, 
            enabled_only=enabled_only
        )
        return constraints
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get constraints: {str(e)}"
        )


@router.post("/contracts/{contract_id}/constraints", status_code=status.HTTP_201_CREATED)
async def add_constraint(
    contract_id: UUID = Path(..., description="Contract ID"),
    constraint_data: ConstraintCreateRequest = Body(...),
    constraint_repo = Depends(get_constraint_repository)
) -> Dict[str, Any]:
    """
    Add a new constraint to a contract.
    
    Creates a new legal or safety constraint with specified severity and penalty.
    """
    try:
        constraint = await constraint_repo.create_constraint(
            contract_id=contract_id,
            name=constraint_data.name,
            description=constraint_data.description,
            constraint_type=constraint_data.constraint_type,
            legal_blocks_spec=constraint_data.legal_blocks_spec,
            severity=constraint_data.severity,
            violation_penalty=constraint_data.violation_penalty,
            enabled=constraint_data.enabled
        )
        
        return constraint
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add constraint: {str(e)}"
        )


@router.get("/contracts/search")
async def search_contracts(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    contract_repo: ContractRepository = Depends(get_contract_repository)
) -> List[Dict[str, Any]]:
    """
    Search contracts by name, creator, or other fields.
    
    Performs full-text search across contract metadata.
    """
    try:
        results = await contract_repo.search_contracts(
            search_term=q,
            limit=limit
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/contracts/{contract_id}/versions")
async def get_contract_versions(
    contract_id: UUID = Path(..., description="Contract ID"),
    contract_repo: ContractRepository = Depends(get_contract_repository)
) -> List[Dict[str, Any]]:
    """
    Get all versions of a contract.
    
    Returns version history for the specified contract.
    """
    try:
        # First get the contract to find its name
        contract = await contract_repo.get_by_id(contract_id)
        
        if not contract:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Contract not found"
            )
        
        versions = await contract_repo.get_contract_versions(contract["name"])
        return versions
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get versions: {str(e)}"
        )


@router.post("/contracts/{contract_id}/compute-reward")
async def compute_reward(
    contract_id: UUID = Path(..., description="Contract ID"),
    state: List[float] = Body(..., description="Environment state vector"),
    action: List[float] = Body(..., description="Action vector"),
    contract_service: ContractService = Depends(get_contract_service)
) -> Dict[str, Any]:
    """
    Compute reward for given state and action.
    
    Evaluates the contract's reward function with provided inputs.
    """
    try:
        import jax.numpy as jnp
        
        # Get contract
        contract_id_str = str(contract_id)
        contract = contract_service.get_contract(contract_id_str)
        
        if not contract:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Contract not found"
            )
        
        # Convert inputs to JAX arrays
        state_array = jnp.array(state)
        action_array = jnp.array(action)
        
        # Compute reward
        reward = contract.compute_reward(state_array, action_array)
        
        # Check violations
        violations = contract.check_violations(state_array, action_array)
        
        return {
            "reward": float(reward),
            "violations": violations,
            "state_shape": state_array.shape,
            "action_shape": action_array.shape
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute reward: {str(e)}"
        )
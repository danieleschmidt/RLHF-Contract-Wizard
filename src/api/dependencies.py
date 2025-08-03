"""
FastAPI dependencies for RLHF-Contract-Wizard API.

Provides dependency injection for services and utilities.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from functools import lru_cache

from ..services.contract_service import ContractService
from ..services.verification_service import VerificationService, VerificationBackend
from ..services.blockchain_service import BlockchainService, NetworkType
from ..repositories.contract_repository import (
    ContractRepository, 
    StakeholderRepository,
    ConstraintRepository,
    DeploymentRepository,
    VerificationRepository
)
from ..utils.validators import ValidationError


# Service dependencies
@lru_cache()
def get_verification_service() -> VerificationService:
    """Get verification service singleton."""
    return VerificationService(backend=VerificationBackend.Z3)


@lru_cache()
def get_blockchain_service() -> BlockchainService:
    """Get blockchain service singleton."""
    return BlockchainService()


def get_contract_service(
    verification_service: VerificationService = Depends(get_verification_service),
    blockchain_service: BlockchainService = Depends(get_blockchain_service)
) -> ContractService:
    """Get contract service with dependencies."""
    return ContractService(
        verification_service=verification_service,
        blockchain_service=blockchain_service
    )


# Repository dependencies
def get_contract_repository() -> ContractRepository:
    """Get contract repository."""
    return ContractRepository()


def get_stakeholder_repository() -> StakeholderRepository:
    """Get stakeholder repository."""
    return StakeholderRepository()


def get_constraint_repository() -> ConstraintRepository:
    """Get constraint repository."""
    return ConstraintRepository()


def get_deployment_repository() -> DeploymentRepository:
    """Get deployment repository."""
    return DeploymentRepository()


def get_verification_repository() -> VerificationRepository:
    """Get verification repository."""
    return VerificationRepository()


# Request validation dependencies
def validate_contract_name(name: str) -> str:
    """Validate contract name parameter."""
    try:
        from ..utils.validators import validate_contract_name
        validate_contract_name(name)
        return name
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid contract name: {str(e)}"
        )


def validate_version(version: str) -> str:
    """Validate version parameter."""
    try:
        from ..utils.validators import validate_version
        validate_version(version)
        return version
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid version: {str(e)}"
        )


def validate_ethereum_address(address: str) -> str:
    """Validate Ethereum address parameter."""
    try:
        from ..utils.validators import validate_ethereum_address
        validate_ethereum_address(address)
        return address
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Ethereum address: {str(e)}"
        )


def validate_network(network: str) -> NetworkType:
    """Validate and convert network parameter."""
    try:
        return NetworkType(network)
    except ValueError:
        valid_networks = [n.value for n in NetworkType]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid network. Must be one of: {valid_networks}"
        )


def validate_verification_backend(backend: str) -> VerificationBackend:
    """Validate and convert verification backend parameter."""
    try:
        return VerificationBackend(backend)
    except ValueError:
        valid_backends = [b.value for b in VerificationBackend]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid backend. Must be one of: {valid_backends}"
        )


# Pagination dependencies
class PaginationParams:
    """Pagination parameters."""
    
    def __init__(
        self,
        page: int = 1,
        size: int = 20,
        max_size: int = 100
    ):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page must be >= 1"
            )
        
        if size < 1 or size > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Size must be between 1 and {max_size}"
            )
        
        self.page = page
        self.size = size
        self.offset = (page - 1) * size
        self.limit = size


def get_pagination_params(
    page: int = 1,
    size: int = 20
) -> PaginationParams:
    """Get pagination parameters with validation."""
    return PaginationParams(page=page, size=size)


# Search dependencies
class SearchParams:
    """Search parameters."""
    
    def __init__(
        self,
        q: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "desc"
    ):
        if sort_order not in ["asc", "desc"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sort order must be 'asc' or 'desc'"
            )
        
        self.query = q
        self.sort_by = sort_by
        self.sort_order = sort_order


def get_search_params(
    q: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_order: str = "desc"
) -> SearchParams:
    """Get search parameters with validation."""
    return SearchParams(q=q, sort_by=sort_by, sort_order=sort_order)


# Authentication dependencies (placeholder for future implementation)
async def get_current_user(request: Request) -> Optional[dict]:
    """Get current authenticated user."""
    # Placeholder for authentication
    # In a real implementation, this would validate JWT tokens, API keys, etc.
    auth_header = request.headers.get("Authorization")
    
    if not auth_header:
        return None
    
    # Mock user for development
    if auth_header == "Bearer dev-token":
        return {
            "id": "dev-user",
            "email": "dev@example.com",
            "name": "Development User",
            "roles": ["admin"]
        }
    
    return None


async def require_authentication(
    current_user: Optional[dict] = Depends(get_current_user)
) -> dict:
    """Require user to be authenticated."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return current_user


async def require_admin_role(
    current_user: dict = Depends(require_authentication)
) -> dict:
    """Require user to have admin role."""
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    return current_user


# Request ID dependency
def get_request_id(request: Request) -> str:
    """Get request ID from state."""
    return getattr(request.state, "request_id", "unknown")


# Content type validation
def validate_json_content_type(request: Request) -> None:
    """Validate that request has JSON content type."""
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be application/json"
        )


# Rate limiting dependencies
def check_rate_limit(request: Request) -> None:
    """Check if request is within rate limits."""
    # This would integrate with Redis or similar for distributed rate limiting
    # For now, it's a placeholder
    rate_limit_exceeded = False  # Placeholder logic
    
    if rate_limit_exceeded:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )
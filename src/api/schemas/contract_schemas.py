"""
Pydantic schemas for contract API endpoints.

Defines request/response models for contract management operations.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum

from ...models.reward_contract import AggregationStrategy


class ContractStatus(str, Enum):
    """Contract status enumeration."""
    DRAFT = "draft"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


class ConstraintType(str, Enum):
    """Constraint type enumeration."""
    REQUIRES = "requires"
    ENSURES = "ensures"
    INVARIANT = "invariant"
    FORALL = "forall"
    EXISTS = "exists"


# Base schemas
class StakeholderBase(BaseModel):
    """Base stakeholder model."""
    name: str = Field(..., min_length=1, max_length=255, description="Stakeholder name")
    weight: float = Field(..., ge=0.0, le=1.0, description="Stakeholder weight (0-1)")
    voting_power: float = Field(1.0, ge=0.0, description="Voting power multiplier")
    wallet_address: Optional[str] = Field(None, regex=r"^0x[a-fA-F0-9]{40}$", description="Ethereum wallet address")


class ConstraintBase(BaseModel):
    """Base constraint model."""
    name: str = Field(..., min_length=1, max_length=255, description="Constraint name")
    description: str = Field("", description="Human-readable description")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    severity: float = Field(1.0, ge=0.0, le=10.0, description="Constraint severity (0-10)")
    violation_penalty: float = Field(-1.0, description="Penalty for violating constraint")
    enabled: bool = Field(True, description="Whether constraint is enabled")


# Request schemas
class ContractCreateRequest(BaseModel):
    """Request model for creating a new contract."""
    name: str = Field(..., min_length=3, max_length=255, description="Contract name")
    version: str = Field(..., regex=r"^\d+\.\d+\.\d+(-\w+)?$", description="Semantic version")
    stakeholders: Dict[str, float] = Field(..., description="Stakeholder weights")
    creator: str = Field(..., min_length=1, max_length=255, description="Contract creator")
    jurisdiction: str = Field("Global", max_length=100, description="Legal jurisdiction")
    regulatory_framework: Optional[str] = Field(None, max_length=100, description="Regulatory framework")
    aggregation_strategy: str = Field("weighted_average", description="Stakeholder aggregation strategy")
    
    @validator("stakeholders")
    def validate_stakeholders(cls, v):
        """Validate stakeholder weights."""
        if not v:
            raise ValueError("At least one stakeholder is required")
        
        if any(weight <= 0 for weight in v.values()):
            raise ValueError("All stakeholder weights must be positive")
        
        total_weight = sum(v.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("Stakeholder weights must sum to 1.0")
        
        return v
    
    @validator("aggregation_strategy")
    def validate_aggregation_strategy(cls, v):
        """Validate aggregation strategy."""
        valid_strategies = [strategy.value for strategy in AggregationStrategy]
        if v not in valid_strategies:
            raise ValueError(f"Invalid aggregation strategy. Must be one of: {valid_strategies}")
        return v


class ContractUpdateRequest(BaseModel):
    """Request model for updating a contract."""
    jurisdiction: Optional[str] = Field(None, max_length=100)
    regulatory_framework: Optional[str] = Field(None, max_length=100)
    status: Optional[ContractStatus] = None
    is_active: Optional[bool] = None


class StakeholderCreateRequest(StakeholderBase):
    """Request model for creating a stakeholder."""
    preferences: Optional[Dict[str, Any]] = Field(None, description="Stakeholder preferences")
    contact_info: Optional[Dict[str, Any]] = Field(None, description="Contact information")


class ConstraintCreateRequest(ConstraintBase):
    """Request model for creating a constraint."""
    legal_blocks_spec: Optional[str] = Field(None, description="Legal-Blocks specification")


class RewardComputeRequest(BaseModel):
    """Request model for computing rewards."""
    state: List[float] = Field(..., description="Environment state vector")
    action: List[float] = Field(..., description="Action vector")
    
    @validator("state", "action")
    def validate_vectors(cls, v):
        """Validate input vectors."""
        if not v:
            raise ValueError("Vector cannot be empty")
        if len(v) > 10000:
            raise ValueError("Vector too large (max 10000 elements)")
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("All vector elements must be numbers")
        return v


# Response schemas
class StakeholderResponse(StakeholderBase):
    """Response model for stakeholder data."""
    id: UUID = Field(..., description="Stakeholder ID")
    contract_id: UUID = Field(..., description="Contract ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Preferences")
    contact_info: Dict[str, Any] = Field(default_factory=dict, description="Contact info")
    
    class Config:
        from_attributes = True


class ConstraintResponse(ConstraintBase):
    """Response model for constraint data."""
    id: UUID = Field(..., description="Constraint ID")
    contract_id: UUID = Field(..., description="Contract ID")
    legal_blocks_spec: Optional[str] = Field(None, description="Legal-Blocks specification")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True


class ContractResponse(BaseModel):
    """Response model for contract data."""
    id: UUID = Field(..., description="Contract ID")
    name: str = Field(..., description="Contract name")
    version: str = Field(..., description="Contract version")
    contract_hash: str = Field(..., description="Contract hash")
    creator: str = Field(..., description="Contract creator")
    jurisdiction: str = Field(..., description="Legal jurisdiction")
    regulatory_framework: Optional[str] = Field(None, description="Regulatory framework")
    aggregation_strategy: str = Field(..., description="Aggregation strategy")
    status: ContractStatus = Field(..., description="Contract status")
    is_active: bool = Field(..., description="Whether contract is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Related data
    stakeholders: List[StakeholderResponse] = Field(default_factory=list, description="Contract stakeholders")
    constraints: List[ConstraintResponse] = Field(default_factory=list, description="Contract constraints")
    
    # Metrics
    stakeholder_count: int = Field(0, description="Number of stakeholders")
    constraint_count: int = Field(0, description="Number of constraints")
    deployment_count: int = Field(0, description="Number of deployments")
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_contract(cls, contract) -> "ContractResponse":
        """Create response from RewardContract object."""
        return cls(
            id=UUID(contract.compute_hash()[:32] + "0" * 4),  # Mock UUID from hash
            name=contract.metadata.name,
            version=contract.metadata.version,
            contract_hash=contract.compute_hash(),
            creator=contract.metadata.creator,
            jurisdiction=contract.metadata.jurisdiction,
            regulatory_framework=contract.metadata.regulatory_framework,
            aggregation_strategy=contract.aggregation_strategy.value,
            status=ContractStatus.DRAFT,
            is_active=True,
            created_at=datetime.fromtimestamp(contract.metadata.created_at),
            updated_at=datetime.fromtimestamp(contract.metadata.updated_at),
            stakeholder_count=len(contract.stakeholders),
            constraint_count=len(contract.constraints)
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContractResponse":
        """Create response from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            contract_hash=data.get("contract_hash", ""),
            creator=data["creator"],
            jurisdiction=data.get("jurisdiction", "Global"),
            regulatory_framework=data.get("regulatory_framework"),
            aggregation_strategy=data.get("aggregation_strategy", "weighted_average"),
            status=ContractStatus(data.get("status", "draft")),
            is_active=data.get("is_active", True),
            created_at=data["created_at"],
            updated_at=data["updated_at"]
        )


class ContractListResponse(BaseModel):
    """Response model for contract list."""
    contracts: List[ContractResponse] = Field(..., description="List of contracts")
    total: int = Field(..., description="Total number of contracts")
    page: int = Field(..., description="Current page")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total pages")


class RewardComputeResponse(BaseModel):
    """Response model for reward computation."""
    reward: float = Field(..., description="Computed reward value")
    violations: Dict[str, bool] = Field(..., description="Constraint violations")
    state_shape: List[int] = Field(..., description="Input state shape")
    action_shape: List[int] = Field(..., description="Input action shape")
    computation_time_ms: Optional[float] = Field(None, description="Computation time in milliseconds")


class ValidationResponse(BaseModel):
    """Response model for contract validation."""
    valid: bool = Field(..., description="Whether contract is valid")
    errors: List[str] = Field(..., description="Validation errors")
    warnings: List[str] = Field(..., description="Validation warnings")
    metrics: Dict[str, Any] = Field(..., description="Validation metrics")
    verification: Optional[Dict[str, Any]] = Field(None, description="Verification results")


# Error schemas
class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Union[str, int]] = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": 400,
                    "message": "Invalid input data",
                    "type": "validation_error"
                }
            }
        }
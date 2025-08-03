"""
Deployment API routes.

Provides endpoints for deploying contracts to blockchain networks.
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, Body

from ..dependencies import (
    get_blockchain_service,
    get_deployment_repository,
    validate_network,
    validate_ethereum_address
)
from ...services.blockchain_service import BlockchainService, NetworkType
from ...repositories.contract_repository import DeploymentRepository


router = APIRouter()


@router.post("/contracts/{contract_id}/deploy")
async def deploy_contract(
    contract_id: UUID = Path(..., description="Contract ID"),
    network: str = Body(..., description="Target blockchain network"),
    from_address: str = Body(..., description="Deployer wallet address"),
    gas_limit: Optional[int] = Body(None, ge=21000, le=30000000, description="Gas limit"),
    gas_price: Optional[int] = Body(None, ge=1000000000, description="Gas price in wei"),
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    deployment_repo: DeploymentRepository = Depends(get_deployment_repository)
) -> Dict[str, Any]:
    """
    Deploy a contract to a blockchain network.
    
    Deploys the specified contract to the chosen blockchain network.
    """
    try:
        # Validate inputs
        network_type = validate_network(network)
        deployer_address = validate_ethereum_address(from_address)
        
        # Create mock contract specification for deployment
        contract_spec = {
            "contract_hash": f"hash_{contract_id}",
            "metadata": {
                "name": f"Contract_{contract_id}",
                "version": "1.0.0"
            },
            "stakeholders": {
                "operator": {"weight": 0.6},
                "users": {"weight": 0.4}
            },
            "constraints": {}
        }
        
        # Deploy to blockchain
        deployment_result = blockchain_service.deploy_contract(
            contract_spec=contract_spec,
            network=network_type,
            from_address=deployer_address,
            gas_limit=gas_limit or 3000000
        )
        
        # Store deployment record
        deployment_record = await deployment_repo.create_deployment(
            contract_id=contract_id,
            network=network,
            contract_address=deployment_result.contract_address,
            transaction_hash=deployment_result.tx_hash,
            deployer_address=deployer_address,
            gas_used=deployment_result.gas_used,
            gas_price=deployment_result.gas_price,
            block_number=deployment_result.block_number
        )
        
        return {
            "deployment_id": deployment_record["id"],
            "contract_id": str(contract_id),
            "network": network,
            "contract_address": deployment_result.contract_address,
            "transaction_hash": deployment_result.tx_hash,
            "deployer_address": deployer_address,
            "gas_used": deployment_result.gas_used,
            "gas_price": deployment_result.gas_price,
            "block_number": deployment_result.block_number,
            "status": deployment_result.status,
            "deployed_at": deployment_record["deployed_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deployment failed: {str(e)}"
        )


@router.get("/contracts/{contract_id}/deployments")
async def get_contract_deployments(
    contract_id: UUID = Path(..., description="Contract ID"),
    deployment_repo: DeploymentRepository = Depends(get_deployment_repository)
) -> List[Dict[str, Any]]:
    """
    Get all deployments for a contract.
    
    Returns deployment history across all networks.
    """
    try:
        deployments = await deployment_repo.get_deployments_by_contract(contract_id)
        
        return [
            {
                "deployment_id": deployment["id"],
                "network": deployment["network"],
                "contract_address": deployment["contract_address"],
                "transaction_hash": deployment["transaction_hash"],
                "deployer_address": deployment["deployer_address"],
                "gas_used": deployment["gas_used"],
                "gas_price": deployment["gas_price"],
                "block_number": deployment["block_number"],
                "status": deployment["status"],
                "deployed_at": deployment["deployed_at"],
                "confirmed_at": deployment.get("confirmed_at")
            }
            for deployment in deployments
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get deployments: {str(e)}"
        )


@router.get("/deployments/{deployment_id}")
async def get_deployment(
    deployment_id: UUID = Path(..., description="Deployment ID"),
    deployment_repo: DeploymentRepository = Depends(get_deployment_repository)
) -> Dict[str, Any]:
    """
    Get specific deployment details.
    
    Returns detailed information about a specific deployment.
    """
    try:
        deployment = await deployment_repo.get_by_id(deployment_id)
        
        if not deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found"
            )
        
        return {
            "deployment_id": deployment["id"],
            "contract_id": deployment["contract_id"],
            "network": deployment["network"],
            "contract_address": deployment["contract_address"],
            "transaction_hash": deployment["transaction_hash"],
            "deployer_address": deployment["deployer_address"],
            "gas_used": deployment["gas_used"],
            "gas_price": deployment["gas_price"],
            "block_number": deployment["block_number"],
            "status": deployment["status"],
            "deployed_at": deployment["deployed_at"],
            "confirmed_at": deployment.get("confirmed_at")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get deployment: {str(e)}"
        )


@router.post("/deployments/{deployment_id}/confirm")
async def confirm_deployment(
    deployment_id: UUID = Path(..., description="Deployment ID"),
    block_number: int = Body(..., description="Confirmation block number"),
    deployment_repo: DeploymentRepository = Depends(get_deployment_repository)
) -> Dict[str, Any]:
    """
    Confirm a deployment.
    
    Marks a deployment as confirmed once it's included in a block.
    """
    try:
        updated_deployment = await deployment_repo.confirm_deployment(
            deployment_id, 
            block_number
        )
        
        if not updated_deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found"
            )
        
        return {
            "deployment_id": str(deployment_id),
            "status": updated_deployment["status"],
            "block_number": updated_deployment["block_number"],
            "confirmed_at": updated_deployment["confirmed_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to confirm deployment: {str(e)}"
        )


@router.get("/deployments/{deployment_id}/status")
async def get_deployment_status(
    deployment_id: UUID = Path(..., description="Deployment ID"),
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    deployment_repo: DeploymentRepository = Depends(get_deployment_repository)
) -> Dict[str, Any]:
    """
    Get real-time deployment status.
    
    Checks blockchain for current status of the deployment.
    """
    try:
        deployment = await deployment_repo.get_by_id(deployment_id)
        
        if not deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found"
            )
        
        # Get contract state from blockchain
        contract_state = blockchain_service.get_contract_state(
            contract_address=deployment["contract_address"],
            network=NetworkType(deployment["network"])
        )
        
        return {
            "deployment_id": str(deployment_id),
            "database_status": deployment["status"],
            "blockchain_state": contract_state,
            "synchronized": deployment["status"] == "confirmed" and contract_state["active"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get deployment status: {str(e)}"
        )


@router.get("/deployments/network/{network}")
async def get_network_deployments(
    network: str = Path(..., description="Network name"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    deployment_repo: DeploymentRepository = Depends(get_deployment_repository)
) -> List[Dict[str, Any]]:
    """
    Get all deployments on a specific network.
    
    Returns deployments filtered by blockchain network.
    """
    try:
        # Validate network
        validate_network(network)
        
        deployments = await deployment_repo.get_deployments_by_network(network)
        
        # Limit results
        limited_deployments = deployments[:limit]
        
        return [
            {
                "deployment_id": deployment["id"],
                "contract_id": deployment["contract_id"],
                "contract_address": deployment["contract_address"],
                "transaction_hash": deployment["transaction_hash"],
                "deployer_address": deployment["deployer_address"],
                "status": deployment["status"],
                "deployed_at": deployment["deployed_at"]
            }
            for deployment in limited_deployments
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get network deployments: {str(e)}"
        )


@router.get("/deployments/active")
async def get_active_deployments(
    deployment_repo: DeploymentRepository = Depends(get_deployment_repository)
) -> List[Dict[str, Any]]:
    """
    Get all active deployments.
    
    Returns deployments that are currently confirmed and active.
    """
    try:
        deployments = await deployment_repo.get_active_deployments()
        
        return [
            {
                "deployment_id": deployment["id"],
                "contract_id": deployment["contract_id"],
                "network": deployment["network"],
                "contract_address": deployment["contract_address"],
                "deployer_address": deployment["deployer_address"],
                "deployed_at": deployment["deployed_at"],
                "confirmed_at": deployment.get("confirmed_at")
            }
            for deployment in deployments
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active deployments: {str(e)}"
        )


@router.post("/deployments/{deployment_id}/update")
async def update_contract_on_chain(
    deployment_id: UUID = Path(..., description="Deployment ID"),
    updates: Dict[str, Any] = Body(..., description="Updates to apply"),
    from_address: str = Body(..., description="Address initiating update"),
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    deployment_repo: DeploymentRepository = Depends(get_deployment_repository)
) -> Dict[str, Any]:
    """
    Update deployed contract on blockchain.
    
    Sends update transaction to modify contract state.
    """
    try:
        deployment = await deployment_repo.get_by_id(deployment_id)
        
        if not deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found"
            )
        
        # Validate deployer address
        validate_ethereum_address(from_address)
        
        # Send update transaction
        update_result = blockchain_service.update_contract(
            contract_address=deployment["contract_address"],
            updates=updates,
            network=NetworkType(deployment["network"]),
            from_address=from_address
        )
        
        return {
            "deployment_id": str(deployment_id),
            "update_transaction": update_result.tx_hash,
            "block_number": update_result.block_number,
            "gas_used": update_result.gas_used,
            "status": update_result.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update contract: {str(e)}"
        )


@router.get("/networks")
async def list_supported_networks() -> Dict[str, Any]:
    """
    List supported blockchain networks.
    
    Returns information about available networks for deployment.
    """
    networks = []
    
    for network in NetworkType:
        network_info = {
            "name": network.value,
            "display_name": _get_network_display_name(network),
            "description": _get_network_description(network),
            "is_testnet": _is_testnet(network),
            "block_time": _get_block_time(network),
            "gas_currency": _get_gas_currency(network)
        }
        networks.append(network_info)
    
    return {
        "networks": networks,
        "default": "ethereum_testnet",
        "recommended_testnet": "ethereum_testnet",
        "recommended_mainnet": "ethereum_mainnet"
    }


def _get_network_display_name(network: NetworkType) -> str:
    """Get display name for network."""
    display_names = {
        NetworkType.ETHEREUM_MAINNET: "Ethereum Mainnet",
        NetworkType.ETHEREUM_TESTNET: "Ethereum Goerli Testnet",
        NetworkType.POLYGON: "Polygon Mainnet",
        NetworkType.POLYGON_MUMBAI: "Polygon Mumbai Testnet",
        NetworkType.LOCAL: "Local Development Network"
    }
    return display_names.get(network, network.value)


def _get_network_description(network: NetworkType) -> str:
    """Get description for network."""
    descriptions = {
        NetworkType.ETHEREUM_MAINNET: "Ethereum main network with real ETH",
        NetworkType.ETHEREUM_TESTNET: "Ethereum test network with free test ETH",
        NetworkType.POLYGON: "Polygon main network with real MATIC",
        NetworkType.POLYGON_MUMBAI: "Polygon test network with free test MATIC",
        NetworkType.LOCAL: "Local development network for testing"
    }
    return descriptions.get(network, "Unknown network")


def _is_testnet(network: NetworkType) -> bool:
    """Check if network is a testnet."""
    testnets = {NetworkType.ETHEREUM_TESTNET, NetworkType.POLYGON_MUMBAI, NetworkType.LOCAL}
    return network in testnets


def _get_block_time(network: NetworkType) -> int:
    """Get average block time in seconds."""
    block_times = {
        NetworkType.ETHEREUM_MAINNET: 12,
        NetworkType.ETHEREUM_TESTNET: 12,
        NetworkType.POLYGON: 2,
        NetworkType.POLYGON_MUMBAI: 2,
        NetworkType.LOCAL: 1
    }
    return block_times.get(network, 15)


def _get_gas_currency(network: NetworkType) -> str:
    """Get gas currency for network."""
    currencies = {
        NetworkType.ETHEREUM_MAINNET: "ETH",
        NetworkType.ETHEREUM_TESTNET: "ETH",
        NetworkType.POLYGON: "MATIC",
        NetworkType.POLYGON_MUMBAI: "MATIC",
        NetworkType.LOCAL: "ETH"
    }
    return currencies.get(network, "ETH")
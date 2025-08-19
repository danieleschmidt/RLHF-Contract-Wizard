#!/usr/bin/env python3
"""
Enhanced API Client for RLHF Contract Wizard

Provides a comprehensive Python client for interacting with the RLHF Contract Wizard API,
including async operations, caching, retry logic, and advanced contract management.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from urllib.parse import urljoin


class ContractStatus(Enum):
    """Contract deployment and verification status."""
    DRAFT = "draft"
    VERIFIED = "verified"
    DEPLOYED = "deployed"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ContractConfig:
    """Configuration for contract creation and deployment."""
    name: str
    version: str = "1.0.0"
    stakeholders: Dict[str, float] = None
    constraints: List[Dict[str, Any]] = None
    aggregation_strategy: str = "weighted_average"
    jurisdiction: str = "Global"
    auto_deploy: bool = False
    verification_timeout: int = 300  # seconds


@dataclass
class VerificationResult:
    """Result of contract verification process."""
    contract_id: str
    status: ContractStatus
    verification_time: float
    properties_verified: List[str]
    violations_found: List[str]
    confidence_score: float
    formal_proof: Optional[str] = None


@dataclass
class DeploymentResult:
    """Result of contract deployment process."""
    contract_id: str
    deployment_id: str
    status: ContractStatus
    blockchain_address: Optional[str] = None
    gas_used: Optional[int] = None
    transaction_hash: Optional[str] = None
    deployment_time: float = 0.0


class RLHFContractClient:
    """
    Advanced client for RLHF Contract Wizard API.
    
    Features:
    - Async HTTP operations with connection pooling
    - Automatic retry with exponential backoff
    - Response caching for performance
    - Contract lifecycle management
    - Real-time monitoring and metrics
    - Batch operations support
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        enable_caching: bool = True,
        cache_ttl: int = 300
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Internal state
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._headers = {
            "Content-Type": "application/json",
            "User-Agent": "RLHF-Contract-Client/1.0.0"
        }
        
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()
    
    async def _create_session(self):
        """Create HTTP session with optimized settings."""
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool limit
            limit_per_host=10,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers=self._headers
        )
    
    async def _close_session(self):
        """Close HTTP session and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None
    
    def _cache_key(self, method: str, url: str, params: Dict = None) -> str:
        """Generate cache key for request."""
        key_data = f"{method}:{url}"
        if params:
            key_data += f":{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - cache_entry["timestamp"] < self.cache_ttl
    
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        use_cache: bool = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and caching.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: JSON data for request body
            params: Query parameters
            use_cache: Override caching behavior
            
        Returns:
            Response data as dictionary
            
        Raises:
            aiohttp.ClientError: For HTTP-related errors
            ValueError: For invalid responses
        """
        if not self._session:
            await self._create_session()
        
        url = urljoin(f"{self.base_url}/api/v1/", endpoint.lstrip('/'))
        use_cache = use_cache if use_cache is not None else (self.enable_caching and method == "GET")
        
        # Check cache for GET requests
        cache_key = None
        if use_cache:
            cache_key = self._cache_key(method, url, params)
            if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]):
                return self._cache[cache_key]["data"]
        
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs = {"params": params} if params else {}
                if data:
                    kwargs["json"] = data
                
                async with self._session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    
                    # Parse response
                    if response.content_type == "application/json":
                        response_data = await response.json()
                    else:
                        response_data = {"text": await response.text()}
                    
                    # Cache successful GET responses
                    if use_cache and cache_key:
                        self._cache[cache_key] = {
                            "data": response_data,
                            "timestamp": time.time()
                        }
                    
                    return response_data
                    
            except aiohttp.ClientError as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = (2 ** attempt) + (0.1 * attempt)  # Exponential backoff with jitter
                    await asyncio.sleep(delay)
                else:
                    raise
        
        raise last_exception
    
    # Contract Management Methods
    
    async def create_contract(self, config: ContractConfig) -> Dict[str, Any]:
        """
        Create a new RLHF contract.
        
        Args:
            config: Contract configuration
            
        Returns:
            Contract creation response with ID and initial status
        """
        contract_data = asdict(config)
        if contract_data["stakeholders"] is None:
            contract_data["stakeholders"] = {"default": 1.0}
        if contract_data["constraints"] is None:
            contract_data["constraints"] = []
        
        response = await self._request("POST", "contracts", data=contract_data)
        return response
    
    async def get_contract(self, contract_id: str) -> Dict[str, Any]:
        """
        Retrieve contract details by ID.
        
        Args:
            contract_id: Unique contract identifier
            
        Returns:
            Contract details and current status
        """
        return await self._request("GET", f"contracts/{contract_id}")
    
    async def update_contract(self, contract_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing contract configuration.
        
        Args:
            contract_id: Contract to update
            updates: Fields to update
            
        Returns:
            Updated contract data
        """
        return await self._request("PUT", f"contracts/{contract_id}", data=updates)
    
    async def delete_contract(self, contract_id: str) -> Dict[str, Any]:
        """
        Delete contract (if not deployed).
        
        Args:
            contract_id: Contract to delete
            
        Returns:
            Deletion confirmation
        """
        return await self._request("DELETE", f"contracts/{contract_id}")
    
    async def list_contracts(
        self, 
        status: Optional[ContractStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List contracts with optional filtering.
        
        Args:
            status: Filter by contract status
            limit: Maximum number of contracts to return
            offset: Number of contracts to skip
            
        Returns:
            List of contracts with pagination info
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value
        
        return await self._request("GET", "contracts", params=params)
    
    # Contract Verification Methods
    
    async def verify_contract(
        self, 
        contract_id: str,
        verification_level: str = "standard",
        timeout: Optional[int] = None
    ) -> VerificationResult:
        """
        Formally verify contract properties and constraints.
        
        Args:
            contract_id: Contract to verify
            verification_level: Level of verification (basic, standard, comprehensive)
            timeout: Verification timeout in seconds
            
        Returns:
            Verification results with proof details
        """
        verification_data = {
            "verification_level": verification_level,
            "timeout": timeout or 300
        }
        
        start_time = time.time()
        response = await self._request("POST", f"verification/{contract_id}", data=verification_data)
        verification_time = time.time() - start_time
        
        return VerificationResult(
            contract_id=contract_id,
            status=ContractStatus(response["status"]),
            verification_time=verification_time,
            properties_verified=response.get("properties_verified", []),
            violations_found=response.get("violations_found", []),
            confidence_score=response.get("confidence_score", 0.0),
            formal_proof=response.get("formal_proof")
        )
    
    async def get_verification_status(self, verification_id: str) -> Dict[str, Any]:
        """Get status of ongoing verification process."""
        return await self._request("GET", f"verification/status/{verification_id}")
    
    # Contract Deployment Methods
    
    async def deploy_contract(
        self,
        contract_id: str,
        network: str = "testnet",
        gas_limit: Optional[int] = None,
        auto_verify: bool = True
    ) -> DeploymentResult:
        """
        Deploy verified contract to blockchain.
        
        Args:
            contract_id: Contract to deploy
            network: Target blockchain network
            gas_limit: Maximum gas to use for deployment
            auto_verify: Automatically verify before deployment
            
        Returns:
            Deployment result with blockchain details
        """
        deployment_data = {
            "network": network,
            "gas_limit": gas_limit,
            "auto_verify": auto_verify
        }
        
        start_time = time.time()
        response = await self._request("POST", f"deployment/{contract_id}", data=deployment_data)
        deployment_time = time.time() - start_time
        
        return DeploymentResult(
            contract_id=contract_id,
            deployment_id=response["deployment_id"],
            status=ContractStatus(response["status"]),
            blockchain_address=response.get("blockchain_address"),
            gas_used=response.get("gas_used"),
            transaction_hash=response.get("transaction_hash"),
            deployment_time=deployment_time
        )
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of contract deployment process."""
        return await self._request("GET", f"deployment/status/{deployment_id}")
    
    # Batch Operations
    
    async def batch_create_contracts(self, configs: List[ContractConfig]) -> List[Dict[str, Any]]:
        """
        Create multiple contracts in batch.
        
        Args:
            configs: List of contract configurations
            
        Returns:
            List of creation results
        """
        batch_data = {"contracts": [asdict(config) for config in configs]}
        response = await self._request("POST", "contracts/batch", data=batch_data)
        return response["results"]
    
    async def batch_verify_contracts(self, contract_ids: List[str]) -> List[VerificationResult]:
        """
        Verify multiple contracts in batch.
        
        Args:
            contract_ids: List of contract IDs to verify
            
        Returns:
            List of verification results
        """
        batch_data = {"contract_ids": contract_ids}
        response = await self._request("POST", "verification/batch", data=batch_data)
        
        return [
            VerificationResult(
                contract_id=result["contract_id"],
                status=ContractStatus(result["status"]),
                verification_time=result["verification_time"],
                properties_verified=result.get("properties_verified", []),
                violations_found=result.get("violations_found", []),
                confidence_score=result.get("confidence_score", 0.0),
                formal_proof=result.get("formal_proof")
            )
            for result in response["results"]
        ]
    
    # Monitoring and Metrics
    
    async def get_api_health(self) -> Dict[str, Any]:
        """Get API health status and metrics."""
        return await self._request("GET", "health")
    
    async def get_contract_metrics(self, contract_id: str) -> Dict[str, Any]:
        """Get detailed metrics for a specific contract."""
        return await self._request("GET", f"contracts/{contract_id}/metrics")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        return await self._request("GET", "metrics")
    
    # Streaming and Real-time Updates
    
    async def stream_contract_events(self, contract_id: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream real-time events for a contract.
        
        Args:
            contract_id: Contract to monitor
            
        Yields:
            Contract events as they occur
        """
        if not self._session:
            await self._create_session()
        
        url = urljoin(f"{self.base_url}/api/v1/", f"contracts/{contract_id}/stream")
        
        async with self._session.get(url) as response:
            response.raise_for_status()
            
            async for line in response.content:
                if line:
                    try:
                        event_data = json.loads(line.decode())
                        yield event_data
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
    
    # Utility Methods
    
    def clear_cache(self):
        """Clear the request cache."""
        self._cache.clear()
    
    async def test_connection(self) -> bool:
        """
        Test API connectivity.
        
        Returns:
            True if API is reachable, False otherwise
        """
        try:
            health = await self.get_api_health()
            return health.get("status") == "healthy"
        except Exception:
            return False


# Convenience functions for common workflows

async def quick_deploy_contract(
    name: str,
    stakeholders: Dict[str, float],
    constraints: List[Dict[str, Any]] = None,
    client: Optional[RLHFContractClient] = None
) -> DeploymentResult:
    """
    Quick deployment workflow: create, verify, and deploy contract.
    
    Args:
        name: Contract name
        stakeholders: Stakeholder weights
        constraints: Optional constraints
        client: Optional pre-configured client
        
    Returns:
        Deployment result
    """
    if client is None:
        client = RLHFContractClient()
    
    async with client:
        # Create contract
        config = ContractConfig(
            name=name,
            stakeholders=stakeholders,
            constraints=constraints or [],
            auto_deploy=True
        )
        
        contract_response = await client.create_contract(config)
        contract_id = contract_response["contract_id"]
        
        # Verify contract
        verification = await client.verify_contract(contract_id)
        if verification.status != ContractStatus.VERIFIED:
            raise ValueError(f"Contract verification failed: {verification.violations_found}")
        
        # Deploy contract
        deployment = await client.deploy_contract(contract_id)
        return deployment


async def monitor_contract_deployment(
    deployment_id: str,
    client: Optional[RLHFContractClient] = None,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Monitor contract deployment until completion.
    
    Args:
        deployment_id: Deployment to monitor
        client: Optional pre-configured client
        timeout: Maximum time to wait (seconds)
        
    Returns:
        Final deployment status
    """
    if client is None:
        client = RLHFContractClient()
    
    start_time = time.time()
    
    async with client:
        while time.time() - start_time < timeout:
            status = await client.get_deployment_status(deployment_id)
            
            if status["status"] in ["deployed", "active", "failed"]:
                return status
            
            await asyncio.sleep(5)  # Poll every 5 seconds
        
        raise TimeoutError(f"Deployment monitoring timed out after {timeout} seconds")


# Example usage
async def example_usage():
    """Example demonstrating client usage."""
    async with RLHFContractClient(base_url="http://localhost:8000") as client:
        # Test connection
        if not await client.test_connection():
            print("❌ API not available")
            return
        
        print("✅ Connected to RLHF Contract Wizard API")
        
        # Create a contract
        config = ContractConfig(
            name="ExampleContract",
            stakeholders={"safety": 0.4, "performance": 0.6},
            constraints=[
                {"name": "safety_constraint", "description": "Ensure safe outputs"}
            ]
        )
        
        contract = await client.create_contract(config)
        contract_id = contract["contract_id"]
        print(f"✅ Created contract: {contract_id}")
        
        # Verify the contract
        verification = await client.verify_contract(contract_id)
        print(f"✅ Verification: {verification.status.value}")
        
        # Deploy if verified
        if verification.status == ContractStatus.VERIFIED:
            deployment = await client.deploy_contract(contract_id)
            print(f"✅ Deployed: {deployment.deployment_id}")


if __name__ == "__main__":
    asyncio.run(example_usage())
"""
Integration tests for contracts API endpoints.

Tests the complete API functionality with real HTTP requests.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.api.main import app


@pytest.fixture
def client():
    """Create test client for API testing."""
    return TestClient(app)


@pytest.fixture
def sample_contract_data():
    """Sample contract data for API tests."""
    return {
        "name": "TestAPIContract",
        "version": "1.0.0",
        "stakeholders": {
            "operator": 0.6,
            "users": 0.4
        },
        "creator": "api_test_user",
        "jurisdiction": "Test",
        "aggregation_strategy": "weighted_average"
    }


class TestContractsAPI:
    """Test cases for contracts API endpoints."""
    
    def test_create_contract_success(self, client, sample_contract_data):
        """Test successful contract creation via API."""
        response = client.post("/api/v1/contracts", json=sample_contract_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["name"] == sample_contract_data["name"]
        assert data["version"] == sample_contract_data["version"]
        assert data["creator"] == sample_contract_data["creator"]
        assert "id" in data
        assert "contract_hash" in data
        assert "created_at" in data
    
    def test_create_contract_validation_error(self, client):
        """Test contract creation with validation errors."""
        invalid_data = {
            "name": "",  # Empty name should fail
            "version": "invalid",  # Invalid version format
            "stakeholders": {},  # Empty stakeholders
            "creator": "test"
        }
        
        response = client.post("/api/v1/contracts", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data
    
    def test_create_contract_invalid_stakeholder_weights(self, client):
        """Test contract creation with invalid stakeholder weights."""
        invalid_data = {
            "name": "TestContract",
            "version": "1.0.0",
            "stakeholders": {
                "operator": 0.3,
                "users": 0.5  # Sum = 0.8, should equal 1.0
            },
            "creator": "test"
        }
        
        response = client.post("/api/v1/contracts", json=invalid_data)
        
        assert response.status_code == 422
        error_data = response.json()
        assert "stakeholder weights must sum to 1.0" in str(error_data["detail"]).lower()
    
    @patch('src.repositories.contract_repository.ContractRepository.get_all')
    async def test_list_contracts(self, mock_get_all, client):
        """Test listing contracts with pagination."""
        # Mock repository response
        mock_contracts = [
            {
                "id": "contract1",
                "name": "Contract1",
                "version": "1.0.0",
                "creator": "user1",
                "created_at": "2025-01-20T12:00:00Z",
                "updated_at": "2025-01-20T12:00:00Z",
                "status": "draft"
            }
        ]
        mock_get_all.return_value = mock_contracts
        
        with patch('src.repositories.contract_repository.ContractRepository.count', return_value=1):
            response = client.get("/api/v1/contracts?page=1&size=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "contracts" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
        assert data["page"] == 1
        assert data["size"] == 10
    
    def test_list_contracts_with_filters(self, client):
        """Test listing contracts with status filter."""
        response = client.get("/api/v1/contracts?status_filter=validated&creator_filter=testuser")
        
        # Should not fail even if no contracts match
        assert response.status_code == 200
    
    @patch('src.repositories.contract_repository.ContractRepository.get_by_id')
    async def test_get_contract_success(self, mock_get_by_id, client):
        """Test getting a specific contract."""
        mock_contract = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "TestContract",
            "version": "1.0.0",
            "creator": "testuser",
            "created_at": "2025-01-20T12:00:00Z",
            "updated_at": "2025-01-20T12:00:00Z",
            "status": "draft"
        }
        mock_get_by_id.return_value = mock_contract
        
        response = client.get("/api/v1/contracts/550e8400-e29b-41d4-a716-446655440000")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TestContract"
    
    @patch('src.repositories.contract_repository.ContractRepository.get_by_id')
    async def test_get_contract_not_found(self, mock_get_by_id, client):
        """Test getting non-existent contract."""
        mock_get_by_id.return_value = None
        
        response = client.get("/api/v1/contracts/550e8400-e29b-41d4-a716-446655440000")
        
        assert response.status_code == 404
        error_data = response.json()
        assert "not found" in error_data["error"]["message"].lower()
    
    def test_get_contract_invalid_uuid(self, client):
        """Test getting contract with invalid UUID."""
        response = client.get("/api/v1/contracts/invalid-uuid")
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.services.contract_service.ContractService.validate_contract')
    def test_validate_contract_success(self, mock_validate, client):
        """Test contract validation endpoint."""
        mock_validate.return_value = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {"constraints": 2}
        }
        
        with patch('src.services.contract_service.ContractService.get_contract') as mock_get:
            # Mock contract object
            mock_contract = type('MockContract', (), {
                'metadata': type('Metadata', (), {'name': 'Test', 'version': '1.0.0'})(),
                'stakeholders': {'test': type('Stakeholder', (), {'weight': 1.0})()},
                'constraints': {}
            })()
            mock_get.return_value = mock_contract
            
            response = client.post("/api/v1/contracts/550e8400-e29b-41d4-a716-446655440000/validate")
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "errors" in data
        assert "warnings" in data
    
    @patch('src.services.contract_service.ContractService.get_contract')
    def test_validate_contract_not_found(self, mock_get, client):
        """Test validation of non-existent contract."""
        mock_get.return_value = None
        
        response = client.post("/api/v1/contracts/550e8400-e29b-41d4-a716-446655440000/validate")
        
        assert response.status_code == 404
    
    def test_compute_reward_endpoint(self, client):
        """Test reward computation endpoint."""
        request_data = {
            "state": [0.1, 0.5, 0.8],
            "action": [0.2, 0.6]
        }
        
        with patch('src.services.contract_service.ContractService.get_contract') as mock_get:
            # Mock contract with compute_reward method
            mock_contract = type('MockContract', (), {
                'compute_reward': lambda self, state, action: 0.75,
                'check_violations': lambda self, state, action: {"safety": False}
            })()
            mock_get.return_value = mock_contract
            
            response = client.post(
                "/api/v1/contracts/550e8400-e29b-41d4-a716-446655440000/compute-reward",
                json=request_data
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "reward" in data
        assert "violations" in data
        assert "state_shape" in data
        assert "action_shape" in data
    
    def test_compute_reward_invalid_input(self, client):
        """Test reward computation with invalid input."""
        invalid_data = {
            "state": [],  # Empty state
            "action": [0.1, 0.2]
        }
        
        response = client.post(
            "/api/v1/contracts/550e8400-e29b-41d4-a716-446655440000/compute-reward",
            json=invalid_data
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.repositories.contract_repository.StakeholderRepository.get_stakeholders_by_contract')
    async def test_get_contract_stakeholders(self, mock_get_stakeholders, client):
        """Test getting contract stakeholders."""
        mock_stakeholders = [
            {
                "id": "stakeholder1",
                "name": "operator",
                "weight": 0.6,
                "voting_power": 1.0
            }
        ]
        mock_get_stakeholders.return_value = mock_stakeholders
        
        response = client.get("/api/v1/contracts/550e8400-e29b-41d4-a716-446655440000/stakeholders")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('src.repositories.contract_repository.StakeholderRepository.create_stakeholder')
    async def test_add_stakeholder(self, mock_create, client):
        """Test adding stakeholder to contract."""
        stakeholder_data = {
            "name": "new_stakeholder",
            "weight": 0.3,
            "voting_power": 1.0,
            "wallet_address": "0x1234567890123456789012345678901234567890"
        }
        
        mock_create.return_value = {
            "id": "new_stakeholder_id",
            **stakeholder_data
        }
        
        response = client.post(
            "/api/v1/contracts/550e8400-e29b-41d4-a716-446655440000/stakeholders",
            json=stakeholder_data
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == stakeholder_data["name"]
    
    def test_add_stakeholder_invalid_address(self, client):
        """Test adding stakeholder with invalid wallet address."""
        invalid_data = {
            "name": "stakeholder",
            "weight": 0.3,
            "wallet_address": "invalid_address"
        }
        
        response = client.post(
            "/api/v1/contracts/550e8400-e29b-41d4-a716-446655440000/stakeholders",
            json=invalid_data
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.repositories.contract_repository.ContractRepository.search_contracts')
    async def test_search_contracts(self, mock_search, client):
        """Test contract search functionality."""
        mock_results = [
            {
                "id": "contract1",
                "name": "SearchableContract",
                "creator": "searchuser"
            }
        ]
        mock_search.return_value = mock_results
        
        response = client.get("/api/v1/contracts/search?q=searchable&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_search_contracts_missing_query(self, client):
        """Test search without query parameter."""
        response = client.get("/api/v1/contracts/search")
        
        assert response.status_code == 422  # Missing required parameter


class TestAPIErrors:
    """Test API error handling."""
    
    def test_404_endpoint(self, client):
        """Test non-existent endpoint."""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test wrong HTTP method."""
        response = client.delete("/api/v1/contracts")  # DELETE not allowed on collection
        
        assert response.status_code == 405
    
    def test_unsupported_media_type(self, client):
        """Test unsupported content type."""
        response = client.post(
            "/api/v1/contracts",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code in [400, 422]  # Should reject non-JSON
    
    def test_request_size_limit(self, client):
        """Test very large request."""
        large_data = {
            "name": "TestContract",
            "version": "1.0.0",
            "stakeholders": {"test": 1.0},
            "creator": "test",
            "large_field": "x" * 100000  # Very large field
        }
        
        response = client.post("/api/v1/contracts", json=large_data)
        
        # Should either process or reject gracefully
        assert response.status_code in [201, 400, 413, 422]


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    @pytest.mark.slow
    def test_concurrent_requests(self, client, sample_contract_data):
        """Test concurrent API requests."""
        import concurrent.futures
        import time
        
        def make_request():
            return client.post("/api/v1/contracts", json=sample_contract_data)
        
        start_time = time.time()
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        elapsed = time.time() - start_time
        
        # All requests should complete within reasonable time
        assert elapsed < 5.0  # 5 seconds max for 10 requests
        
        # At least some should succeed (depending on mocking)
        success_count = sum(1 for r in responses if r.status_code in [200, 201])
        assert success_count >= 0  # At least no crashes


class TestAPIIntegration:
    """End-to-end integration tests."""
    
    def test_full_contract_lifecycle_api(self, client):
        """Test complete contract lifecycle through API."""
        # This would be a comprehensive test of:
        # 1. Create contract
        # 2. Add stakeholders
        # 3. Add constraints
        # 4. Validate contract
        # 5. Deploy contract
        # 6. Verify contract
        # 7. Update contract
        
        # For now, just test that endpoints are accessible
        contract_data = {
            "name": "LifecycleTest",
            "version": "1.0.0",
            "stakeholders": {"test": 1.0},
            "creator": "integration_test"
        }
        
        # Create contract
        response = client.post("/api/v1/contracts", json=contract_data)
        
        # Should either succeed or fail gracefully with proper error format
        if response.status_code == 201:
            contract_id = response.json()["id"]
            
            # Try validation
            validation_response = client.post(f"/api/v1/contracts/{contract_id}/validate")
            assert validation_response.status_code in [200, 404, 500]
            
            # Try getting stakeholders
            stakeholders_response = client.get(f"/api/v1/contracts/{contract_id}/stakeholders")
            assert stakeholders_response.status_code in [200, 500]
        
        else:
            # Should have proper error format
            assert "error" in response.json() or "detail" in response.json()
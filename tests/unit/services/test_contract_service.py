"""
Unit tests for ContractService.

Tests contract lifecycle management, validation, and service operations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.services.contract_service import ContractService, ContractValidationError
from src.models.reward_contract import RewardContract


class TestContractService:
    """Test cases for ContractService class."""
    
    def test_service_initialization(self):
        """Test contract service initialization."""
        service = ContractService()
        
        assert service.verification_service is None
        assert service.blockchain_service is None
        assert isinstance(service._contract_registry, dict)
        assert isinstance(service._deployment_history, list)
    
    def test_service_initialization_with_dependencies(self, verification_service, blockchain_service):
        """Test service initialization with dependencies."""
        service = ContractService(
            verification_service=verification_service,
            blockchain_service=blockchain_service
        )
        
        assert service.verification_service == verification_service
        assert service.blockchain_service == blockchain_service
    
    def test_create_contract_basic(self, contract_service):
        """Test basic contract creation."""
        contract = contract_service.create_contract(
            name="TestContract",
            version="1.0.0",
            stakeholders={"test": 1.0},
            creator="test_user"
        )
        
        assert isinstance(contract, RewardContract)
        assert contract.metadata.name == "TestContract"
        assert contract.metadata.version == "1.0.0"
        assert contract.metadata.creator == "test_user"
        assert len(contract_service._contract_registry) == 1
    
    def test_create_contract_validation_error(self, contract_service):
        """Test contract creation with validation errors."""
        # Empty name should raise validation error
        with pytest.raises(ContractValidationError):
            contract_service.create_contract(
                name="",
                version="1.0.0"
            )
        
        # Negative stakeholder weight should raise validation error
        with pytest.raises(ContractValidationError):
            contract_service.create_contract(
                name="ValidName",
                version="1.0.0",
                stakeholders={"negative": -0.5}
            )
    
    def test_validate_contract_basic(self, contract_service, basic_contract):
        """Test basic contract validation."""
        result = contract_service.validate_contract(basic_contract)
        
        assert isinstance(result, dict)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "metrics" in result
        
        # Should be valid if properly constructed
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_contract_no_stakeholders(self, contract_service):
        """Test validation of contract without stakeholders."""
        contract = RewardContract(
            name="NoStakeholders",
            version="1.0.0"
        )
        
        result = contract_service.validate_contract(contract)
        
        assert result["valid"] is False
        assert "must have at least one stakeholder" in str(result["errors"])
    
    def test_validate_contract_no_reward_functions(self, contract_service):
        """Test validation of contract without reward functions."""
        contract = RewardContract(
            name="NoRewards",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        result = contract_service.validate_contract(contract)
        
        assert result["valid"] is False
        assert "must have at least one reward function" in str(result["errors"])
    
    def test_validate_contract_with_verification(self, contract_service, basic_contract):
        """Test contract validation with formal verification."""
        # Mock verification service
        mock_verification = Mock()
        mock_verification.verify_contract.return_value = {
            "all_proofs_valid": True,
            "proved_properties": 5,
            "total_properties": 5
        }
        contract_service.verification_service = mock_verification
        
        result = contract_service.validate_contract(basic_contract)
        
        assert result["valid"] is True
        assert "verification" in result
        mock_verification.verify_contract.assert_called_once_with(basic_contract)
    
    @pytest.mark.asyncio
    async def test_deploy_contract_success(self, contract_service, basic_contract):
        """Test successful contract deployment."""
        # Mock blockchain service
        mock_blockchain = AsyncMock()
        mock_blockchain.deploy_contract.return_value = {
            "tx_hash": "0x123...",
            "contract_address": "0xabc...",
            "gas_used": 500000
        }
        contract_service.blockchain_service = mock_blockchain
        
        result = await contract_service.deploy_contract(
            basic_contract,
            network="testnet"
        )
        
        assert result["status"] == "deployed"
        assert "contract_address" in result
        assert "transaction_hash" in result
        assert len(contract_service._deployment_history) == 1
    
    @pytest.mark.asyncio 
    async def test_deploy_contract_validation_failure(self, contract_service):
        """Test deployment of invalid contract."""
        # Create invalid contract
        invalid_contract = RewardContract(
            name="Invalid",
            version="1.0.0"
            # No stakeholders - should fail validation
        )
        
        with pytest.raises(ContractValidationError):
            await contract_service.deploy_contract(invalid_contract)
    
    @pytest.mark.asyncio
    async def test_deploy_contract_blockchain_failure(self, contract_service, basic_contract):
        """Test deployment failure due to blockchain error."""
        # Mock blockchain service to fail
        mock_blockchain = AsyncMock()
        mock_blockchain.deploy_contract.side_effect = RuntimeError("Blockchain error")
        contract_service.blockchain_service = mock_blockchain
        
        with pytest.raises(RuntimeError):
            await contract_service.deploy_contract(basic_contract)
    
    def test_get_contract(self, contract_service):
        """Test retrieving contract by ID."""
        contract = contract_service.create_contract(
            name="RetrieveTest",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        contract_id = contract_service._generate_contract_id(contract)
        retrieved = contract_service.get_contract(contract_id)
        
        assert retrieved is not None
        assert retrieved.metadata.name == "RetrieveTest"
    
    def test_get_contract_not_found(self, contract_service):
        """Test retrieving non-existent contract."""
        result = contract_service.get_contract("nonexistent_id")
        assert result is None
    
    def test_list_contracts(self, contract_service):
        """Test listing all contracts."""
        # Create multiple contracts
        for i in range(3):
            contract_service.create_contract(
                name=f"Contract_{i}",
                version="1.0.0",
                stakeholders={"test": 1.0}
            )
        
        contracts = contract_service.list_contracts()
        
        assert len(contracts) == 3
        assert all("contract_id" in c for c in contracts)
        assert all("name" in c for c in contracts)
        assert all("stakeholders" in c for c in contracts)
    
    def test_update_contract(self, contract_service):
        """Test updating existing contract."""
        contract = contract_service.create_contract(
            name="UpdateTest",
            version="1.0.0",
            stakeholders={"original": 1.0}
        )
        
        contract_id = contract_service._generate_contract_id(contract)
        original_version = contract.metadata.version
        
        updated = contract_service.update_contract(
            contract_id,
            {
                "stakeholders": {"new_stakeholder": 0.5},
                "constraints": {
                    "new_constraint": {
                        "function": lambda s, a: True,
                        "description": "New constraint"
                    }
                }
            }
        )
        
        assert updated.metadata.version != original_version
        assert "new_stakeholder" in updated.stakeholders
        assert "new_constraint" in updated.constraints
    
    def test_update_nonexistent_contract(self, contract_service):
        """Test updating non-existent contract."""
        with pytest.raises(ValueError):
            contract_service.update_contract("nonexistent", {})
    
    def test_save_and_load_contract_pickle(self, contract_service, basic_contract, temp_dir):
        """Test saving and loading contract in pickle format."""
        filepath = f"{temp_dir}/test_contract.pkl"
        
        contract_service.save_contract(basic_contract, filepath, format="pickle")
        loaded_contract = contract_service.load_contract(filepath, format="pickle")
        
        assert loaded_contract.metadata.name == basic_contract.metadata.name
        assert loaded_contract.metadata.version == basic_contract.metadata.version
        assert len(loaded_contract.stakeholders) == len(basic_contract.stakeholders)
    
    def test_save_and_load_contract_json(self, contract_service, basic_contract, temp_dir):
        """Test saving and loading contract in JSON format."""
        filepath = f"{temp_dir}/test_contract.json"
        
        contract_service.save_contract(basic_contract, filepath, format="json")
        
        # JSON loading creates a basic contract structure
        # Full functionality would require more sophisticated serialization
        loaded_contract = contract_service.load_contract(filepath, format="json")
        
        assert loaded_contract.metadata.name == basic_contract.metadata.name
        assert loaded_contract.metadata.version == basic_contract.metadata.version
    
    def test_save_contract_invalid_format(self, contract_service, basic_contract, temp_dir):
        """Test saving contract with invalid format."""
        filepath = f"{temp_dir}/test_contract.xyz"
        
        with pytest.raises(ValueError):
            contract_service.save_contract(basic_contract, filepath, format="invalid")
    
    def test_load_contract_nonexistent_file(self, contract_service):
        """Test loading non-existent contract file."""
        with pytest.raises(FileNotFoundError):
            contract_service.load_contract("nonexistent.pkl")
    
    def test_generate_contract_id(self, contract_service, basic_contract):
        """Test contract ID generation."""
        id1 = contract_service._generate_contract_id(basic_contract)
        id2 = contract_service._generate_contract_id(basic_contract)
        
        assert id1 == id2  # Should be deterministic
        assert len(id1) == 32  # MD5 hex length
        assert isinstance(id1, str)
    
    def test_get_deployment_history(self, contract_service):
        """Test getting deployment history."""
        # Add some mock deployment history
        deployment = {
            "contract_id": "test_id",
            "network": "testnet",
            "status": "deployed"
        }
        contract_service._deployment_history.append(deployment)
        
        history = contract_service.get_deployment_history()
        
        assert len(history) == 1
        assert history[0]["contract_id"] == "test_id"
        assert isinstance(history, list)  # Should be a copy
    
    def test_rollback_contract(self, contract_service):
        """Test contract rollback functionality."""
        contract = contract_service.create_contract(
            name="RollbackTest",
            version="2.0.0",
            stakeholders={"test": 1.0}
        )
        
        contract_id = contract_service._generate_contract_id(contract)
        
        rolled_back = contract_service.rollback_contract(
            contract_id,
            target_version="1.0.0"
        )
        
        assert rolled_back.metadata.version == "1.0.0"
    
    def test_rollback_nonexistent_contract(self, contract_service):
        """Test rollback of non-existent contract."""
        with pytest.raises(ValueError):
            contract_service.rollback_contract("nonexistent", "1.0.0")


class TestContractValidation:
    """Test cases for contract validation logic."""
    
    def test_validate_stakeholder_weights(self, contract_service):
        """Test stakeholder weight validation."""
        # Weights don't sum to 1.0
        contract = RewardContract(
            name="WeightTest",
            version="1.0.0",
            stakeholders={"a": 0.3, "b": 0.5}  # Sum = 0.8
        )
        
        # Add reward function to make it otherwise valid
        @contract.reward_function()
        def test_reward(state, action):
            return 0.5
        
        result = contract_service.validate_contract(contract)
        
        # Should have warning about weights not summing to 1.0
        assert any("weight" in warning.lower() for warning in result["warnings"])
    
    def test_validate_constraint_functions(self, contract_service):
        """Test constraint function validation."""
        contract = RewardContract(
            name="ConstraintTest",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def test_reward(state, action):
            return 0.5
        
        # Add invalid constraint that raises exception
        def bad_constraint(state, action):
            raise RuntimeError("Bad constraint")
        
        contract.add_constraint("bad", bad_constraint)
        
        result = contract_service.validate_contract(contract)
        
        assert result["valid"] is False
        assert any("failed validation" in error for error in result["errors"])
    
    def test_validate_legal_blocks_constraints(self, contract_service):
        """Test validation with Legal-Blocks constraints."""
        from src.models.legal_blocks import LegalBlocks
        
        contract = RewardContract(
            name="LegalTest",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        @LegalBlocks.specification("""
            REQUIRES: valid_input(state, action)
            ENSURES: reward >= 0.0
        """)
        def legal_reward(state, action):
            return 0.7
        
        result = contract_service.validate_contract(contract)
        
        assert result["valid"] is True
        # Should include metrics about Legal-Blocks constraints
        assert "default_constraints" in result["metrics"]


class TestServiceIntegration:
    """Test integration between different service components."""
    
    def test_service_with_verification_integration(self, verification_service):
        """Test service integration with verification service."""
        service = ContractService(verification_service=verification_service)
        
        contract = service.create_contract(
            name="VerifyIntegration",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def test_reward(state, action):
            return 0.5
        
        result = service.validate_contract(contract)
        
        # Should include verification results
        assert "verification" in result
    
    def test_service_with_blockchain_integration(self, blockchain_service):
        """Test service integration with blockchain service."""
        service = ContractService(blockchain_service=blockchain_service)
        
        assert service.blockchain_service == blockchain_service
        
        contract = service.create_contract(
            name="BlockchainIntegration",
            version="1.0.0",
            stakeholders={"test": 1.0}
        )
        
        @contract.reward_function()
        def test_reward(state, action):
            return 0.5
        
        # Should be able to validate for deployment
        result = service.validate_contract(contract)
        assert result["valid"] is True


class TestConcurrency:
    """Test concurrent operations on contract service."""
    
    @pytest.mark.asyncio
    async def test_concurrent_contract_creation(self, contract_service):
        """Test concurrent contract creation."""
        import asyncio
        
        async def create_contract(i):
            return contract_service.create_contract(
                name=f"Concurrent_{i}",
                version="1.0.0", 
                stakeholders={"test": 1.0}
            )
        
        # Create contracts concurrently
        tasks = [create_contract(i) for i in range(5)]
        contracts = await asyncio.gather(*tasks)
        
        assert len(contracts) == 5
        assert len(contract_service._contract_registry) == 5
        
        # All contracts should have unique names
        names = [c.metadata.name for c in contracts]
        assert len(set(names)) == 5
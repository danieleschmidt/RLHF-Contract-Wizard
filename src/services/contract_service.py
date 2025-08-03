"""
Core contract management service.

Handles contract lifecycle management, validation, and deployment operations.
"""

import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
import time

from ..models.reward_contract import RewardContract, Stakeholder
from ..models.legal_blocks import LegalBlocks
from .verification_service import VerificationService
from .blockchain_service import BlockchainService


class ContractValidationError(Exception):
    """Raised when contract validation fails."""
    pass


class ContractService:
    """
    Service for managing reward contract lifecycle.
    
    Handles contract creation, validation, deployment, and management.
    """
    
    def __init__(
        self,
        verification_service: Optional[VerificationService] = None,
        blockchain_service: Optional[BlockchainService] = None
    ):
        """
        Initialize contract service.
        
        Args:
            verification_service: Service for formal verification
            blockchain_service: Service for blockchain operations
        """
        self.verification_service = verification_service
        self.blockchain_service = blockchain_service
        self._contract_registry: Dict[str, RewardContract] = {}
        self._deployment_history: List[Dict[str, Any]] = []
    
    def create_contract(
        self,
        name: str,
        version: str = "1.0.0",
        stakeholders: Optional[Dict[str, float]] = None,
        creator: str = "system"
    ) -> RewardContract:
        """
        Create a new reward contract.
        
        Args:
            name: Contract name
            version: Contract version
            stakeholders: Stakeholder weights
            creator: Contract creator
            
        Returns:
            New RewardContract instance
        """
        # Validate inputs
        if not name or not name.strip():
            raise ContractValidationError("Contract name cannot be empty")
        
        if stakeholders:
            if not all(weight > 0 for weight in stakeholders.values()):
                raise ContractValidationError("All stakeholder weights must be positive")
        
        # Create contract
        contract = RewardContract(
            name=name,
            version=version,
            stakeholders=stakeholders,
            creator=creator
        )
        
        # Register contract
        contract_id = self._generate_contract_id(contract)
        self._contract_registry[contract_id] = contract
        
        return contract
    
    def validate_contract(self, contract: RewardContract) -> Dict[str, Any]:
        """
        Validate a contract for correctness and completeness.
        
        Args:
            contract: Contract to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Basic validation
        if not contract.stakeholders:
            validation_results['errors'].append("Contract must have at least one stakeholder")
            validation_results['valid'] = False
        
        if not contract.reward_functions:
            validation_results['errors'].append("Contract must have at least one reward function")
            validation_results['valid'] = False
        
        # Stakeholder weight validation
        total_weight = sum(s.weight for s in contract.stakeholders.values())
        if abs(total_weight - 1.0) > 1e-6:
            validation_results['warnings'].append(
                f"Stakeholder weights sum to {total_weight}, expected 1.0"
            )
        
        # Constraint validation
        for name, constraint in contract.constraints.items():
            try:
                # Test constraint function with dummy data
                import jax.numpy as jnp
                dummy_state = jnp.zeros(10)
                dummy_action = jnp.zeros(5)
                constraint.constraint_fn(dummy_state, dummy_action)
            except Exception as e:
                validation_results['errors'].append(
                    f"Constraint '{name}' failed validation: {str(e)}"
                )
                validation_results['valid'] = False
        
        # Legal-Blocks validation
        for stakeholder_name, reward_fn in contract.reward_functions.items():
            if hasattr(reward_fn, '__legal_blocks__'):
                blocks_info = LegalBlocks.get_constraints(reward_fn)
                if blocks_info:
                    validation_results['metrics'][f'{stakeholder_name}_constraints'] = len(blocks_info['blocks'])
        
        # Formal verification if service available
        if self.verification_service and validation_results['valid']:
            try:
                verification_result = self.verification_service.verify_contract(contract)
                validation_results['verification'] = verification_result
                if not verification_result.get('all_proofs_valid', True):
                    validation_results['warnings'].append("Some formal proofs failed")
            except Exception as e:
                validation_results['warnings'].append(f"Verification failed: {str(e)}")
        
        return validation_results
    
    def deploy_contract(
        self,
        contract: RewardContract,
        network: str = "testnet",
        gas_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Deploy contract to blockchain.
        
        Args:
            contract: Contract to deploy
            network: Target blockchain network
            gas_limit: Gas limit for deployment
            
        Returns:
            Deployment results
        """
        # Validate contract first
        validation = self.validate_contract(contract)
        if not validation['valid']:
            raise ContractValidationError(
                f"Contract validation failed: {validation['errors']}"
            )
        
        deployment_result = {
            'contract_id': self._generate_contract_id(contract),
            'contract_hash': contract.compute_hash(),
            'network': network,
            'deployed_at': time.time(),
            'status': 'pending'
        }
        
        try:
            if self.blockchain_service:
                # Deploy to blockchain
                tx_result = self.blockchain_service.deploy_contract(
                    contract_spec=contract.to_dict(),
                    network=network,
                    gas_limit=gas_limit
                )
                
                deployment_result.update({
                    'transaction_hash': tx_result['tx_hash'],
                    'contract_address': tx_result['contract_address'],
                    'gas_used': tx_result['gas_used'],
                    'status': 'deployed'
                })
            else:
                # Mock deployment for testing
                deployment_result.update({
                    'contract_address': f"0x{contract.compute_hash()[:40]}",
                    'status': 'deployed'
                })
            
            # Record deployment
            self._deployment_history.append(deployment_result.copy())
            
        except Exception as e:
            deployment_result.update({
                'status': 'failed',
                'error': str(e)
            })
            raise
        
        return deployment_result
    
    def get_contract(self, contract_id: str) -> Optional[RewardContract]:
        """Retrieve contract by ID."""
        return self._contract_registry.get(contract_id)
    
    def list_contracts(self) -> List[Dict[str, Any]]:
        """List all registered contracts."""
        return [
            {
                'contract_id': contract_id,
                'name': contract.metadata.name,
                'version': contract.metadata.version,
                'created_at': contract.metadata.created_at,
                'stakeholders': len(contract.stakeholders),
                'constraints': len(contract.constraints)
            }
            for contract_id, contract in self._contract_registry.items()
        ]
    
    def update_contract(
        self,
        contract_id: str,
        updates: Dict[str, Any]
    ) -> RewardContract:
        """
        Update existing contract with new configuration.
        
        Args:
            contract_id: ID of contract to update
            updates: Dictionary of updates to apply
            
        Returns:
            Updated contract
        """
        contract = self.get_contract(contract_id)
        if not contract:
            raise ValueError(f"Contract not found: {contract_id}")
        
        # Create new version with updates
        new_version_parts = contract.metadata.version.split('.')
        new_version_parts[-1] = str(int(new_version_parts[-1]) + 1)
        new_version = '.'.join(new_version_parts)
        
        # Apply updates
        if 'stakeholders' in updates:
            for name, weight in updates['stakeholders'].items():
                contract.add_stakeholder(name, weight)
        
        if 'constraints' in updates:
            for name, constraint_config in updates['constraints'].items():
                contract.add_constraint(
                    name=name,
                    constraint_fn=constraint_config['function'],
                    description=constraint_config.get('description', ''),
                    severity=constraint_config.get('severity', 1.0)
                )
        
        # Update metadata
        contract.metadata.version = new_version
        contract.metadata.updated_at = time.time()
        
        return contract
    
    def save_contract(
        self,
        contract: RewardContract,
        filepath: str,
        format: str = 'pickle'
    ) -> None:
        """
        Save contract to file.
        
        Args:
            contract: Contract to save
            filepath: Output file path
            format: Save format ('pickle', 'json')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(contract, f)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(contract.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_contract(
        self,
        filepath: str,
        format: str = 'pickle'
    ) -> RewardContract:
        """
        Load contract from file.
        
        Args:
            filepath: Input file path
            format: File format ('pickle', 'json')
            
        Returns:
            Loaded contract
        """
        filepath = Path(filepath)
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                contract = pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r') as f:
                contract_data = json.load(f)
                # Reconstruct contract from dictionary
                # This is simplified - full implementation would need
                # to reconstruct all functions and constraints
                contract = RewardContract(
                    name=contract_data['metadata']['name'],
                    version=contract_data['metadata']['version'],
                    creator=contract_data['metadata']['creator']
                )
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Register loaded contract
        contract_id = self._generate_contract_id(contract)
        self._contract_registry[contract_id] = contract
        
        return contract
    
    def _generate_contract_id(self, contract: RewardContract) -> str:
        """Generate unique ID for contract."""
        content = f"{contract.metadata.name}:{contract.metadata.version}:{contract.metadata.created_at}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return self._deployment_history.copy()
    
    def rollback_contract(
        self,
        contract_id: str,
        target_version: str
    ) -> RewardContract:
        """
        Rollback contract to previous version.
        
        Args:
            contract_id: Contract to rollback
            target_version: Version to rollback to
            
        Returns:
            Rolled back contract
        """
        # In a full implementation, this would retrieve the contract
        # from version history and restore it
        contract = self.get_contract(contract_id)
        if not contract:
            raise ValueError(f"Contract not found: {contract_id}")
        
        # For now, just update version metadata
        contract.metadata.version = target_version
        contract.metadata.updated_at = time.time()
        
        return contract
"""
Contract repository for database operations.

Handles CRUD operations and queries specific to reward contracts.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime

from .base_repository import BaseRepository


class ContractRepository(BaseRepository):
    """Repository for contract database operations."""
    
    def __init__(self):
        super().__init__('contracts')
    
    async def create_contract(
        self,
        name: str,
        version: str,
        contract_hash: str,
        creator: str,
        jurisdiction: str = 'Global',
        regulatory_framework: Optional[str] = None,
        aggregation_strategy: str = 'weighted_average'
    ) -> Dict[str, Any]:
        """
        Create a new contract record.
        
        Args:
            name: Contract name
            version: Contract version
            contract_hash: Unique contract hash
            creator: Contract creator
            jurisdiction: Legal jurisdiction
            regulatory_framework: Regulatory framework
            aggregation_strategy: Stakeholder aggregation strategy
            
        Returns:
            Created contract record
        """
        data = {
            'name': name,
            'version': version,
            'contract_hash': contract_hash,
            'creator': creator,
            'jurisdiction': jurisdiction,
            'aggregation_strategy': aggregation_strategy
        }
        
        if regulatory_framework:
            data['regulatory_framework'] = regulatory_framework
        
        return await self.create(data)
    
    async def get_by_hash(self, contract_hash: str) -> Optional[Dict[str, Any]]:
        """Get contract by hash."""
        return await self.find_one({'contract_hash': contract_hash})
    
    async def get_by_name_version(
        self, 
        name: str, 
        version: str
    ) -> Optional[Dict[str, Any]]:
        """Get contract by name and version."""
        return await self.find_one({'name': name, 'version': version})
    
    async def get_contracts_by_creator(
        self, 
        creator: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get contracts by creator."""
        return await self.find_by_column('creator', creator, limit=limit)
    
    async def get_contracts_by_status(
        self, 
        status: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get contracts by status."""
        return await self.find_by_column('status', status, limit=limit)
    
    async def get_contracts_by_jurisdiction(
        self, 
        jurisdiction: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get contracts by jurisdiction."""
        return await self.find_by_column('jurisdiction', jurisdiction, limit=limit)
    
    async def update_status(
        self, 
        contract_id: UUID, 
        status: str
    ) -> Optional[Dict[str, Any]]:
        """Update contract status."""
        return await self.update(contract_id, {'status': status})
    
    async def get_active_contracts(self) -> List[Dict[str, Any]]:
        """Get all active contracts."""
        return await self.get_all(filters={'is_active': True})
    
    async def deactivate_contract(self, contract_id: UUID) -> Optional[Dict[str, Any]]:
        """Deactivate a contract."""
        return await self.update(contract_id, {'is_active': False})
    
    async def search_contracts(
        self,
        search_term: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search contracts by name or description."""
        return await self.search(
            search_term=search_term,
            search_columns=['name', 'creator', 'jurisdiction'],
            limit=limit
        )
    
    async def get_contract_versions(self, name: str) -> List[Dict[str, Any]]:
        """Get all versions of a contract."""
        return await self.find_by_column('name', name)
    
    async def get_latest_version(self, name: str) -> Optional[Dict[str, Any]]:
        """Get latest version of a contract."""
        query = """
            SELECT * FROM contracts 
            WHERE name = $1 
            ORDER BY created_at DESC 
            LIMIT 1
        """
        
        async with self.db.acquire_connection() as conn:
            record = await conn.fetchrow(query, name)
            return dict(record) if record else None


class StakeholderRepository(BaseRepository):
    """Repository for stakeholder database operations."""
    
    def __init__(self):
        super().__init__('stakeholders')
    
    async def create_stakeholder(
        self,
        contract_id: UUID,
        name: str,
        weight: float,
        voting_power: float = 1.0,
        wallet_address: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        contact_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new stakeholder."""
        data = {
            'contract_id': contract_id,
            'name': name,
            'weight': weight,
            'voting_power': voting_power
        }
        
        if wallet_address:
            data['wallet_address'] = wallet_address
        if preferences:
            data['preferences'] = self._serialize_jsonb(preferences)
        if contact_info:
            data['contact_info'] = self._serialize_jsonb(contact_info)
        
        return await self.create(data)
    
    async def get_stakeholders_by_contract(
        self, 
        contract_id: UUID
    ) -> List[Dict[str, Any]]:
        """Get all stakeholders for a contract."""
        return await self.find_by_column('contract_id', contract_id)
    
    async def get_stakeholder_by_name(
        self,
        contract_id: UUID,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """Get stakeholder by contract and name."""
        return await self.find_one({
            'contract_id': contract_id,
            'name': name
        })
    
    async def update_stakeholder_weight(
        self,
        stakeholder_id: UUID,
        weight: float
    ) -> Optional[Dict[str, Any]]:
        """Update stakeholder weight."""
        return await self.update(stakeholder_id, {'weight': weight})
    
    async def get_stakeholder_voting_power(
        self, 
        contract_id: UUID
    ) -> Dict[str, float]:
        """Get voting power distribution for contract."""
        query = """
            SELECT name, voting_power 
            FROM stakeholders 
            WHERE contract_id = $1
        """
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, contract_id)
            return {record['name']: record['voting_power'] for record in records}


class ConstraintRepository(BaseRepository):
    """Repository for constraint database operations."""
    
    def __init__(self):
        super().__init__('constraints')
    
    async def create_constraint(
        self,
        contract_id: UUID,
        name: str,
        description: str,
        constraint_type: str,
        legal_blocks_spec: Optional[str] = None,
        severity: float = 1.0,
        violation_penalty: float = -1.0,
        enabled: bool = True
    ) -> Dict[str, Any]:
        """Create a new constraint."""
        data = {
            'contract_id': contract_id,
            'name': name,
            'description': description,
            'constraint_type': constraint_type,
            'severity': severity,
            'violation_penalty': violation_penalty,
            'enabled': enabled
        }
        
        if legal_blocks_spec:
            data['legal_blocks_spec'] = legal_blocks_spec
        
        return await self.create(data)
    
    async def get_constraints_by_contract(
        self, 
        contract_id: UUID,
        enabled_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get constraints for a contract."""
        filters = {'contract_id': contract_id}
        if enabled_only:
            filters['enabled'] = True
        
        return await self.get_all(filters=filters)
    
    async def get_constraints_by_type(
        self,
        contract_id: UUID,
        constraint_type: str
    ) -> List[Dict[str, Any]]:
        """Get constraints by type."""
        return await self.get_all(filters={
            'contract_id': contract_id,
            'constraint_type': constraint_type
        })
    
    async def enable_constraint(
        self, 
        constraint_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Enable a constraint."""
        return await self.update(constraint_id, {'enabled': True})
    
    async def disable_constraint(
        self, 
        constraint_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Disable a constraint."""
        return await self.update(constraint_id, {'enabled': False})
    
    async def get_critical_constraints(
        self, 
        contract_id: UUID,
        min_severity: float = 8.0
    ) -> List[Dict[str, Any]]:
        """Get critical constraints above severity threshold."""
        query = """
            SELECT * FROM constraints
            WHERE contract_id = $1 AND severity >= $2 AND enabled = true
            ORDER BY severity DESC
        """
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, contract_id, min_severity)
            return [dict(record) for record in records]


class DeploymentRepository(BaseRepository):
    """Repository for deployment database operations."""
    
    def __init__(self):
        super().__init__('deployments')
    
    async def create_deployment(
        self,
        contract_id: UUID,
        network: str,
        contract_address: str,
        transaction_hash: str,
        deployer_address: str,
        gas_used: Optional[int] = None,
        gas_price: Optional[int] = None,
        block_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new deployment record."""
        data = {
            'contract_id': contract_id,
            'network': network,
            'contract_address': contract_address,
            'transaction_hash': transaction_hash,
            'deployer_address': deployer_address,
            'status': 'pending'
        }
        
        if gas_used:
            data['gas_used'] = gas_used
        if gas_price:
            data['gas_price'] = gas_price
        if block_number:
            data['block_number'] = block_number
        
        return await self.create(data)
    
    async def get_deployments_by_contract(
        self, 
        contract_id: UUID
    ) -> List[Dict[str, Any]]:
        """Get deployments for a contract."""
        return await self.find_by_column('contract_id', contract_id)
    
    async def get_deployments_by_network(
        self, 
        network: str
    ) -> List[Dict[str, Any]]:
        """Get deployments by network."""
        return await self.find_by_column('network', network)
    
    async def confirm_deployment(
        self,
        deployment_id: UUID,
        block_number: int
    ) -> Optional[Dict[str, Any]]:
        """Confirm a deployment."""
        return await self.update(deployment_id, {
            'status': 'confirmed',
            'block_number': block_number,
            'confirmed_at': datetime.utcnow()
        })
    
    async def fail_deployment(
        self,
        deployment_id: UUID,
        error_message: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Mark deployment as failed."""
        update_data = {'status': 'failed'}
        if error_message:
            update_data['error_message'] = error_message
        
        return await self.update(deployment_id, update_data)
    
    async def get_active_deployments(self) -> List[Dict[str, Any]]:
        """Get all confirmed deployments."""
        return await self.find_by_column('status', 'confirmed')


class VerificationRepository(BaseRepository):
    """Repository for verification result operations."""
    
    def __init__(self):
        super().__init__('verification_results')
    
    async def create_verification_result(
        self,
        contract_id: UUID,
        backend: str,
        total_properties: int,
        proved_properties: int,
        failed_properties: int,
        verification_time_ms: int,
        all_proofs_valid: bool,
        proof_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create verification result."""
        data = {
            'contract_id': contract_id,
            'backend': backend,
            'total_properties': total_properties,
            'proved_properties': proved_properties,
            'failed_properties': failed_properties,
            'verification_time_ms': verification_time_ms,
            'all_proofs_valid': all_proofs_valid,
            'proof_results': self._serialize_jsonb(proof_results)
        }
        
        return await self.create(data)
    
    async def get_verification_by_contract(
        self, 
        contract_id: UUID
    ) -> List[Dict[str, Any]]:
        """Get verification results for contract."""
        return await self.find_by_column('contract_id', contract_id)
    
    async def get_latest_verification(
        self, 
        contract_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Get latest verification result."""
        query = """
            SELECT * FROM verification_results
            WHERE contract_id = $1
            ORDER BY verified_at DESC
            LIMIT 1
        """
        
        async with self.db.acquire_connection() as conn:
            record = await conn.fetchrow(query, contract_id)
            return dict(record) if record else None
    
    async def get_verification_stats(self) -> Dict[str, Any]:
        """Get overall verification statistics."""
        query = """
            SELECT 
                COUNT(*) as total_verifications,
                COUNT(*) FILTER (WHERE all_proofs_valid = true) as successful_verifications,
                AVG(verification_time_ms) as avg_verification_time,
                AVG(proved_properties::float / total_properties) as avg_proof_rate
            FROM verification_results
        """
        
        async with self.db.acquire_connection() as conn:
            record = await conn.fetchrow(query)
            return dict(record) if record else {}
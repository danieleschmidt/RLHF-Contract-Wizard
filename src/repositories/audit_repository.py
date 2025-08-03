"""
Audit repository for compliance and event tracking.

Handles storage and retrieval of audit events, compliance tracking,
and immutable audit trails for legal and regulatory requirements.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timedelta
from enum import Enum

from .base_repository import BaseRepository


class EventType(Enum):
    """Audit event types."""
    CREATED = "created"
    UPDATED = "updated"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    VERIFIED = "verified"
    STAKEHOLDER_ADDED = "stakeholder_added"
    STAKEHOLDER_REMOVED = "stakeholder_removed"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    REWARD_FUNCTION_ADDED = "reward_function_added"
    REWARD_FUNCTION_REMOVED = "reward_function_removed"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    VIOLATION_DETECTED = "violation_detected"
    VIOLATION_RESOLVED = "violation_resolved"
    COMPLIANCE_CHECK = "compliance_check"
    SECURITY_INCIDENT = "security_incident"


class AuditRepository(BaseRepository):
    """Repository for audit event operations."""
    
    def __init__(self):
        super().__init__('contract_events')
    
    async def log_event(
        self,
        contract_id: Optional[UUID],
        event_type: str,
        actor: str,
        description: str,
        event_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ) -> Dict[str, Any]:
        """
        Log an audit event.
        
        Args:
            contract_id: Contract ID (optional for global events)
            event_type: Type of event
            actor: Who performed the action
            description: Human-readable description
            event_data: Event-specific data
            metadata: Additional metadata
            severity: Event severity (info, warning, error, critical)
            
        Returns:
            Created audit event record
        """
        data = {
            'event_type': event_type,
            'actor': actor,
            'description': description,
            'occurred_at': datetime.utcnow()
        }
        
        if contract_id:
            data['contract_id'] = contract_id
        
        if event_data:
            data['event_data'] = self._serialize_jsonb(event_data)
        else:
            data['event_data'] = self._serialize_jsonb({})
        
        if metadata:
            # Add severity to metadata
            if isinstance(metadata, dict):
                metadata['severity'] = severity
            data['metadata'] = self._serialize_jsonb(metadata)
        else:
            data['metadata'] = self._serialize_jsonb({'severity': severity})
        
        return await self.create(data)
    
    async def log_contract_creation(
        self,
        contract_id: UUID,
        actor: str,
        contract_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log contract creation event."""
        return await self.log_event(
            contract_id=contract_id,
            event_type=EventType.CREATED.value,
            actor=actor,
            description=f"Contract '{contract_data.get('name', 'unknown')}' created",
            event_data=contract_data,
            metadata={'action': 'contract_creation'}
        )
    
    async def log_contract_update(
        self,
        contract_id: UUID,
        actor: str,
        changes: Dict[str, Any],
        old_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Log contract update event."""
        event_data = {'changes': changes}
        if old_values:
            event_data['old_values'] = old_values
        
        return await self.log_event(
            contract_id=contract_id,
            event_type=EventType.UPDATED.value,
            actor=actor,
            description=f"Contract updated: {', '.join(changes.keys())}",
            event_data=event_data,
            metadata={'action': 'contract_update'}
        )
    
    async def log_deployment(
        self,
        contract_id: UUID,
        actor: str,
        network: str,
        contract_address: str,
        transaction_hash: str
    ) -> Dict[str, Any]:
        """Log contract deployment event."""
        return await self.log_event(
            contract_id=contract_id,
            event_type=EventType.DEPLOYED.value,
            actor=actor,
            description=f"Contract deployed to {network} at {contract_address}",
            event_data={
                'network': network,
                'contract_address': contract_address,
                'transaction_hash': transaction_hash
            },
            metadata={'action': 'deployment'}
        )
    
    async def log_verification(
        self,
        contract_id: UUID,
        actor: str,
        backend: str,
        verification_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log verification event."""
        success = verification_result.get('all_proofs_valid', False)
        severity = "info" if success else "warning"
        
        return await self.log_event(
            contract_id=contract_id,
            event_type=EventType.VERIFIED.value,
            actor=actor,
            description=f"Contract verification {'passed' if success else 'failed'} using {backend}",
            event_data=verification_result,
            metadata={'action': 'verification'},
            severity=severity
        )
    
    async def log_violation(
        self,
        contract_id: UUID,
        constraint_name: str,
        violation_data: Dict[str, Any],
        severity: str = "warning"
    ) -> Dict[str, Any]:
        """Log constraint violation event."""
        return await self.log_event(
            contract_id=contract_id,
            event_type=EventType.VIOLATION_DETECTED.value,
            actor="system",
            description=f"Constraint violation detected: {constraint_name}",
            event_data=violation_data,
            metadata={'constraint_name': constraint_name, 'action': 'violation'},
            severity=severity
        )
    
    async def log_compliance_check(
        self,
        contract_id: UUID,
        actor: str,
        compliance_framework: str,
        check_results: Dict[str, Any],
        passed: bool
    ) -> Dict[str, Any]:
        """Log compliance check event."""
        severity = "info" if passed else "error"
        
        return await self.log_event(
            contract_id=contract_id,
            event_type=EventType.COMPLIANCE_CHECK.value,
            actor=actor,
            description=f"Compliance check ({compliance_framework}): {'PASSED' if passed else 'FAILED'}",
            event_data=check_results,
            metadata={
                'compliance_framework': compliance_framework,
                'passed': passed,
                'action': 'compliance_check'
            },
            severity=severity
        )
    
    async def log_security_incident(
        self,
        contract_id: Optional[UUID],
        actor: str,
        incident_type: str,
        incident_data: Dict[str, Any],
        severity: str = "critical"
    ) -> Dict[str, Any]:
        """Log security incident."""
        return await self.log_event(
            contract_id=contract_id,
            event_type=EventType.SECURITY_INCIDENT.value,
            actor=actor,
            description=f"Security incident: {incident_type}",
            event_data=incident_data,
            metadata={
                'incident_type': incident_type,
                'action': 'security_incident'
            },
            severity=severity
        )
    
    async def get_audit_trail(
        self,
        contract_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit trail for a contract.
        
        Args:
            contract_id: Contract ID
            start_time: Start of time range
            end_time: End of time range
            event_types: Filter by event types
            limit: Maximum number of events
            
        Returns:
            List of audit events
        """
        query_parts = ["SELECT * FROM contract_events WHERE contract_id = $1"]
        params = [contract_id]
        param_count = 1
        
        if start_time:
            param_count += 1
            query_parts.append(f"AND occurred_at >= ${param_count}")
            params.append(start_time)
        
        if end_time:
            param_count += 1
            query_parts.append(f"AND occurred_at <= ${param_count}")
            params.append(end_time)
        
        if event_types:
            param_count += 1
            query_parts.append(f"AND event_type = ANY(${param_count})")
            params.append(event_types)
        
        query_parts.append("ORDER BY occurred_at DESC")
        
        if limit:
            param_count += 1
            query_parts.append(f"LIMIT ${param_count}")
            params.append(limit)
        
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, *params)
            return [dict(record) for record in records]
    
    async def get_events_by_actor(
        self,
        actor: str,
        start_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get events by actor (user)."""
        filters = {'actor': actor}
        
        if start_time:
            query = """
                SELECT * FROM contract_events 
                WHERE actor = $1 AND occurred_at >= $2
                ORDER BY occurred_at DESC
            """
            params = [actor, start_time]
            
            if limit:
                query += " LIMIT $3"
                params.append(limit)
            
            async with self.db.acquire_connection() as conn:
                records = await conn.fetch(query, *params)
                return [dict(record) for record in records]
        else:
            return await self.get_all(filters=filters, limit=limit)
    
    async def get_events_by_type(
        self,
        event_type: str,
        contract_id: Optional[UUID] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get events by type."""
        filters = {'event_type': event_type}
        if contract_id:
            filters['contract_id'] = contract_id
        
        return await self.get_all(filters=filters, limit=limit)
    
    async def get_security_incidents(
        self,
        start_time: Optional[datetime] = None,
        severity: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get security incidents."""
        query_parts = [
            "SELECT * FROM contract_events",
            "WHERE event_type = $1"
        ]
        params = [EventType.SECURITY_INCIDENT.value]
        param_count = 1
        
        if start_time:
            param_count += 1
            query_parts.append(f"AND occurred_at >= ${param_count}")
            params.append(start_time)
        
        if severity:
            param_count += 1
            query_parts.append(f"AND metadata->>'severity' = ${param_count}")
            params.append(severity)
        
        query_parts.append("ORDER BY occurred_at DESC")
        
        if limit:
            param_count += 1
            query_parts.append(f"LIMIT ${param_count}")
            params.append(limit)
        
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, *params)
            return [dict(record) for record in records]
    
    async def get_violations_summary(
        self,
        contract_id: Optional[UUID] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get summary of violations in the specified period.
        
        Args:
            contract_id: Contract ID (None for all contracts)
            days: Number of days to analyze
            
        Returns:
            Violations summary
        """
        start_time = datetime.utcnow() - timedelta(days=days)
        
        query_parts = [
            "SELECT",
            "COUNT(*) as total_violations,",
            "COUNT(DISTINCT contract_id) as affected_contracts,",
            "metadata->>'constraint_name' as constraint_name,",
            "metadata->>'severity' as severity,",
            "COUNT(*) as violation_count",
            "FROM contract_events",
            "WHERE event_type = $1 AND occurred_at >= $2"
        ]
        
        params = [EventType.VIOLATION_DETECTED.value, start_time]
        param_count = 2
        
        if contract_id:
            param_count += 1
            query_parts.insert(-1, f"AND contract_id = ${param_count}")
            params.append(contract_id)
        
        query_parts.extend([
            "GROUP BY constraint_name, severity",
            "ORDER BY violation_count DESC"
        ])
        
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, *params)
            
            summary = {
                'total_violations': 0,
                'affected_contracts': 0,
                'by_constraint': {},
                'by_severity': {}
            }
            
            for record in records:
                summary['total_violations'] += record['violation_count']
                summary['affected_contracts'] = record['affected_contracts']
                
                constraint = record['constraint_name'] or 'unknown'
                severity = record['severity'] or 'unknown'
                
                if constraint not in summary['by_constraint']:
                    summary['by_constraint'][constraint] = 0
                summary['by_constraint'][constraint] += record['violation_count']
                
                if severity not in summary['by_severity']:
                    summary['by_severity'][severity] = 0
                summary['by_severity'][severity] += record['violation_count']
            
            return summary
    
    async def get_compliance_history(
        self,
        contract_id: UUID,
        compliance_framework: Optional[str] = None,
        days: int = 90
    ) -> List[Dict[str, Any]]:
        """Get compliance check history."""
        start_time = datetime.utcnow() - timedelta(days=days)
        
        query_parts = [
            "SELECT * FROM contract_events",
            "WHERE contract_id = $1 AND event_type = $2 AND occurred_at >= $3"
        ]
        params = [contract_id, EventType.COMPLIANCE_CHECK.value, start_time]
        param_count = 3
        
        if compliance_framework:
            param_count += 1
            query_parts.append(f"AND metadata->>'compliance_framework' = ${param_count}")
            params.append(compliance_framework)
        
        query_parts.append("ORDER BY occurred_at DESC")
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, *params)
            return [dict(record) for record in records]
    
    async def get_audit_statistics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get audit statistics for the specified period."""
        start_time = datetime.utcnow() - timedelta(days=days)
        
        query = """
            SELECT 
                event_type,
                COUNT(*) as event_count,
                COUNT(DISTINCT actor) as unique_actors,
                COUNT(DISTINCT contract_id) as affected_contracts
            FROM contract_events
            WHERE occurred_at >= $1
            GROUP BY event_type
            ORDER BY event_count DESC
        """
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, start_time)
            
            stats = {
                'period_days': days,
                'total_events': 0,
                'by_type': {},
                'unique_actors': set(),
                'affected_contracts': set()
            }
            
            for record in records:
                event_type = record['event_type']
                event_count = record['event_count']
                
                stats['total_events'] += event_count
                stats['by_type'][event_type] = {
                    'count': event_count,
                    'unique_actors': record['unique_actors'],
                    'affected_contracts': record['affected_contracts']
                }
            
            # Get overall unique counts
            overall_query = """
                SELECT 
                    COUNT(DISTINCT actor) as total_unique_actors,
                    COUNT(DISTINCT contract_id) as total_affected_contracts
                FROM contract_events
                WHERE occurred_at >= $1
            """
            
            overall_record = await conn.fetchrow(overall_query, start_time)
            if overall_record:
                stats['total_unique_actors'] = overall_record['total_unique_actors']
                stats['total_affected_contracts'] = overall_record['total_affected_contracts']
            
            return stats
    
    async def export_audit_report(
        self,
        contract_id: Optional[UUID] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export comprehensive audit report.
        
        Args:
            contract_id: Contract ID (None for all contracts)
            start_time: Start of reporting period
            end_time: End of reporting period
            format: Export format (json, csv)
            
        Returns:
            Audit report data
        """
        query_parts = ["SELECT * FROM contract_events WHERE 1=1"]
        params = []
        param_count = 0
        
        if contract_id:
            param_count += 1
            query_parts.append(f"AND contract_id = ${param_count}")
            params.append(contract_id)
        
        if start_time:
            param_count += 1
            query_parts.append(f"AND occurred_at >= ${param_count}")
            params.append(start_time)
        
        if end_time:
            param_count += 1
            query_parts.append(f"AND occurred_at <= ${param_count}")
            params.append(end_time)
        
        query_parts.append("ORDER BY occurred_at ASC")
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, *params)
            
            report = {
                'report_generated_at': datetime.utcnow().isoformat(),
                'contract_id': str(contract_id) if contract_id else 'all',
                'period': {
                    'start': start_time.isoformat() if start_time else None,
                    'end': end_time.isoformat() if end_time else None
                },
                'total_events': len(records),
                'events': [dict(record) for record in records]
            }
            
            # Add summary statistics
            event_types = {}
            actors = set()
            
            for record in records:
                event_type = record['event_type']
                event_types[event_type] = event_types.get(event_type, 0) + 1
                actors.add(record['actor'])
            
            report['summary'] = {
                'event_types': event_types,
                'unique_actors': list(actors),
                'actor_count': len(actors)
            }
            
            return report
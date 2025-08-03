"""
Base repository implementation for RLHF contract data access.

Provides common CRUD operations and query patterns for database entities.
"""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union
from uuid import UUID
from datetime import datetime
import asyncpg

from ..database.connection import db_connection


T = TypeVar('T')


class BaseRepository(Generic[T]):
    """
    Base repository class providing common database operations.
    
    Implements the Repository pattern for clean separation between
    business logic and data access.
    """
    
    def __init__(self, table_name: str, model_class: Optional[Type[T]] = None):
        """
        Initialize repository.
        
        Args:
            table_name: Database table name
            model_class: Optional model class for type conversion
        """
        self.table_name = table_name
        self.model_class = model_class
        self.db = db_connection
    
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new record.
        
        Args:
            data: Record data
            
        Returns:
            Created record with generated fields
        """
        # Build INSERT query
        columns = list(data.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        values = list(data.values())
        
        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING *
        """
        
        async with self.db.acquire_connection() as conn:
            record = await conn.fetchrow(query, *values)
            return dict(record) if record else {}
    
    async def get_by_id(self, record_id: Union[UUID, str]) -> Optional[Dict[str, Any]]:
        """
        Get record by ID.
        
        Args:
            record_id: Record identifier
            
        Returns:
            Record data or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE id = $1"
        
        async with self.db.acquire_connection() as conn:
            record = await conn.fetchrow(query, record_id)
            return dict(record) if record else None
    
    async def get_all(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all records with optional filtering and pagination.
        
        Args:
            limit: Maximum number of records
            offset: Number of records to skip
            order_by: Column to order by
            filters: WHERE clause filters
            
        Returns:
            List of records
        """
        query_parts = [f"SELECT * FROM {self.table_name}"]
        params = []
        param_count = 0
        
        # Add WHERE clause
        if filters:
            where_conditions = []
            for column, value in filters.items():
                param_count += 1
                if value is None:
                    where_conditions.append(f"{column} IS NULL")
                else:
                    where_conditions.append(f"{column} = ${param_count}")
                    params.append(value)
            
            if where_conditions:
                query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        # Add ORDER BY
        if order_by:
            query_parts.append(f"ORDER BY {order_by}")
        else:
            query_parts.append("ORDER BY created_at DESC")
        
        # Add LIMIT and OFFSET
        if limit:
            param_count += 1
            query_parts.append(f"LIMIT ${param_count}")
            params.append(limit)
        
        if offset:
            param_count += 1
            query_parts.append(f"OFFSET ${param_count}")
            params.append(offset)
        
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, *params)
            return [dict(record) for record in records]
    
    async def update(
        self, 
        record_id: Union[UUID, str], 
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update record by ID.
        
        Args:
            record_id: Record identifier
            data: Updated data
            
        Returns:
            Updated record or None if not found
        """
        if not data:
            return await self.get_by_id(record_id)
        
        # Build UPDATE query
        set_clauses = []
        params = []
        param_count = 0
        
        for column, value in data.items():
            param_count += 1
            set_clauses.append(f"{column} = ${param_count}")
            params.append(value)
        
        param_count += 1
        params.append(record_id)
        
        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}
            WHERE id = ${param_count}
            RETURNING *
        """
        
        async with self.db.acquire_connection() as conn:
            record = await conn.fetchrow(query, *params)
            return dict(record) if record else None
    
    async def delete(self, record_id: Union[UUID, str]) -> bool:
        """
        Delete record by ID.
        
        Args:
            record_id: Record identifier
            
        Returns:
            True if record was deleted, False if not found
        """
        query = f"DELETE FROM {self.table_name} WHERE id = $1"
        
        async with self.db.acquire_connection() as conn:
            result = await conn.execute(query, record_id)
            return result == "DELETE 1"
    
    async def exists(self, record_id: Union[UUID, str]) -> bool:
        """
        Check if record exists.
        
        Args:
            record_id: Record identifier
            
        Returns:
            True if record exists
        """
        query = f"SELECT EXISTS(SELECT 1 FROM {self.table_name} WHERE id = $1)"
        
        async with self.db.acquire_connection() as conn:
            return await conn.fetchval(query, record_id)
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records with optional filtering.
        
        Args:
            filters: WHERE clause filters
            
        Returns:
            Number of matching records
        """
        query_parts = [f"SELECT COUNT(*) FROM {self.table_name}"]
        params = []
        param_count = 0
        
        if filters:
            where_conditions = []
            for column, value in filters.items():
                param_count += 1
                if value is None:
                    where_conditions.append(f"{column} IS NULL")
                else:
                    where_conditions.append(f"{column} = ${param_count}")
                    params.append(value)
            
            if where_conditions:
                query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            return await conn.fetchval(query, *params)
    
    async def find_one(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find single record by filters.
        
        Args:
            filters: WHERE clause filters
            
        Returns:
            First matching record or None
        """
        records = await self.get_all(limit=1, filters=filters)
        return records[0] if records else None
    
    async def find_by_column(
        self, 
        column: str, 
        value: Any,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find records by specific column value.
        
        Args:
            column: Column name
            value: Column value
            limit: Maximum number of records
            
        Returns:
            List of matching records
        """
        return await self.get_all(
            limit=limit,
            filters={column: value}
        )
    
    async def bulk_create(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create multiple records in a single transaction.
        
        Args:
            records: List of record data
            
        Returns:
            List of created records
        """
        if not records:
            return []
        
        # Get column names from first record
        columns = list(records[0].keys())
        placeholders_per_record = [f'${i+1}' for i in range(len(columns))]
        
        created_records = []
        
        async with self.db.transaction() as conn:
            for record_data in records:
                values = [record_data[col] for col in columns]
                
                query = f"""
                    INSERT INTO {self.table_name} ({', '.join(columns)})
                    VALUES ({', '.join(placeholders_per_record)})
                    RETURNING *
                """
                
                record = await conn.fetchrow(query, *values)
                if record:
                    created_records.append(dict(record))
        
        return created_records
    
    async def bulk_update(
        self, 
        updates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Update multiple records in a single transaction.
        
        Args:
            updates: List of update data with 'id' field
            
        Returns:
            List of updated records
        """
        if not updates:
            return []
        
        updated_records = []
        
        async with self.db.transaction() as conn:
            for update_data in updates:
                record_id = update_data.pop('id')
                
                if update_data:  # Only update if there's data
                    set_clauses = []
                    params = []
                    param_count = 0
                    
                    for column, value in update_data.items():
                        param_count += 1
                        set_clauses.append(f"{column} = ${param_count}")
                        params.append(value)
                    
                    param_count += 1
                    params.append(record_id)
                    
                    query = f"""
                        UPDATE {self.table_name}
                        SET {', '.join(set_clauses)}
                        WHERE id = ${param_count}
                        RETURNING *
                    """
                    
                    record = await conn.fetchrow(query, *params)
                    if record:
                        updated_records.append(dict(record))
        
        return updated_records
    
    async def search(
        self,
        search_term: str,
        search_columns: List[str],
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search records using ILIKE on specified columns.
        
        Args:
            search_term: Search term
            search_columns: Columns to search
            limit: Maximum number of records
            offset: Number of records to skip
            
        Returns:
            List of matching records
        """
        if not search_columns:
            return []
        
        # Build search conditions
        search_conditions = []
        params = [f"%{search_term}%"] * len(search_columns)
        
        for i, column in enumerate(search_columns):
            search_conditions.append(f"{column} ILIKE ${i+1}")
        
        query_parts = [
            f"SELECT * FROM {self.table_name}",
            f"WHERE {' OR '.join(search_conditions)}",
            "ORDER BY created_at DESC"
        ]
        
        param_count = len(search_columns)
        
        if limit:
            param_count += 1
            query_parts.append(f"LIMIT ${param_count}")
            params.append(limit)
        
        if offset:
            param_count += 1
            query_parts.append(f"OFFSET ${param_count}")
            params.append(offset)
        
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, *params)
            return [dict(record) for record in records]
    
    def _serialize_jsonb(self, data: Any) -> str:
        """Serialize data for JSONB storage."""
        return json.dumps(data, default=str)
    
    def _deserialize_jsonb(self, data: str) -> Any:
        """Deserialize JSONB data."""
        return json.loads(data) if data else None
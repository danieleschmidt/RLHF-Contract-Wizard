"""
Metrics repository for performance and usage tracking.

Handles storage and retrieval of contract performance metrics,
usage statistics, and monitoring data.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime, timedelta
from decimal import Decimal

from .base_repository import BaseRepository


class MetricsRepository(BaseRepository):
    """Repository for metrics database operations."""
    
    def __init__(self):
        super().__init__('metrics')
    
    async def record_metric(
        self,
        contract_id: Optional[UUID],
        metric_type: str,
        metric_name: str,
        metric_value: float,
        unit: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record a single metric.
        
        Args:
            contract_id: Contract ID (optional for global metrics)
            metric_type: Type of metric (performance, usage, compliance, etc.)
            metric_name: Specific metric name
            metric_value: Metric value
            unit: Measurement unit
            context: Additional context data
            
        Returns:
            Created metric record
        """
        data = {
            'metric_type': metric_type,
            'metric_name': metric_name,
            'metric_value': Decimal(str(metric_value)),
            'recorded_at': datetime.utcnow()
        }
        
        if contract_id:
            data['contract_id'] = contract_id
        if unit:
            data['unit'] = unit
        if context:
            data['context'] = self._serialize_jsonb(context)
        
        return await self.create(data)
    
    async def record_batch_metrics(
        self,
        metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Record multiple metrics in a batch.
        
        Args:
            metrics: List of metric data dictionaries
            
        Returns:
            List of created metric records
        """
        # Add recorded_at timestamp to all metrics
        for metric in metrics:
            if 'recorded_at' not in metric:
                metric['recorded_at'] = datetime.utcnow()
            if 'metric_value' in metric:
                metric['metric_value'] = Decimal(str(metric['metric_value']))
        
        return await self.bulk_create(metrics)
    
    async def get_metrics_by_contract(
        self,
        contract_id: UUID,
        metric_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics for a specific contract."""
        filters = {'contract_id': contract_id}
        
        if metric_type:
            filters['metric_type'] = metric_type
        
        # Build time range query if specified
        if start_time or end_time:
            query_parts = ["SELECT * FROM metrics WHERE contract_id = $1"]
            params = [contract_id]
            param_count = 1
            
            if metric_type:
                param_count += 1
                query_parts.append(f"AND metric_type = ${param_count}")
                params.append(metric_type)
            
            if start_time:
                param_count += 1
                query_parts.append(f"AND recorded_at >= ${param_count}")
                params.append(start_time)
            
            if end_time:
                param_count += 1
                query_parts.append(f"AND recorded_at <= ${param_count}")
                params.append(end_time)
            
            query_parts.append("ORDER BY recorded_at DESC")
            
            if limit:
                param_count += 1
                query_parts.append(f"LIMIT ${param_count}")
                params.append(limit)
            
            query = " ".join(query_parts)
            
            async with self.db.acquire_connection() as conn:
                records = await conn.fetch(query, *params)
                return [dict(record) for record in records]
        else:
            return await self.get_all(filters=filters, limit=limit)
    
    async def get_metric_aggregates(
        self,
        contract_id: Optional[UUID],
        metric_type: str,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics (min, max, avg, count).
        
        Args:
            contract_id: Contract ID (None for global metrics)
            metric_type: Type of metric
            metric_name: Specific metric name
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Dictionary with aggregated values
        """
        query_parts = [
            "SELECT",
            "COUNT(*) as count,",
            "MIN(metric_value) as min_value,",
            "MAX(metric_value) as max_value,",
            "AVG(metric_value) as avg_value,",
            "STDDEV(metric_value) as stddev_value",
            "FROM metrics",
            "WHERE metric_type = $1 AND metric_name = $2"
        ]
        
        params = [metric_type, metric_name]
        param_count = 2
        
        if contract_id:
            param_count += 1
            query_parts.insert(-1, f"AND contract_id = ${param_count}")
            params.append(contract_id)
        
        if start_time:
            param_count += 1
            query_parts.insert(-1, f"AND recorded_at >= ${param_count}")
            params.append(start_time)
        
        if end_time:
            param_count += 1
            query_parts.insert(-1, f"AND recorded_at <= ${param_count}")
            params.append(end_time)
        
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            record = await conn.fetchrow(query, *params)
            return dict(record) if record else {}
    
    async def get_metric_time_series(
        self,
        contract_id: Optional[UUID],
        metric_type: str,
        metric_name: str,
        interval: str = '1 hour',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get time series data for a metric.
        
        Args:
            contract_id: Contract ID
            metric_type: Type of metric
            metric_name: Specific metric name
            interval: Time interval for aggregation
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of time series data points
        """
        query_parts = [
            "SELECT",
            f"date_trunc('{interval}', recorded_at) as time_bucket,",
            "COUNT(*) as count,",
            "AVG(metric_value) as avg_value,",
            "MIN(metric_value) as min_value,",
            "MAX(metric_value) as max_value",
            "FROM metrics",
            "WHERE metric_type = $1 AND metric_name = $2"
        ]
        
        params = [metric_type, metric_name]
        param_count = 2
        
        if contract_id:
            param_count += 1
            query_parts.insert(-1, f"AND contract_id = ${param_count}")
            params.append(contract_id)
        
        if start_time:
            param_count += 1
            query_parts.insert(-1, f"AND recorded_at >= ${param_count}")
            params.append(start_time)
        
        if end_time:
            param_count += 1
            query_parts.insert(-1, f"AND recorded_at <= ${param_count}")
            params.append(end_time)
        
        query_parts.extend([
            "GROUP BY time_bucket",
            "ORDER BY time_bucket"
        ])
        
        query = " ".join(query_parts)
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, *params)
            return [dict(record) for record in records]
    
    async def get_top_metrics(
        self,
        metric_type: str,
        limit: int = 10,
        order_by: str = 'avg_value',
        order_direction: str = 'DESC'
    ) -> List[Dict[str, Any]]:
        """
        Get top metrics by aggregated value.
        
        Args:
            metric_type: Type of metric
            limit: Number of results to return
            order_by: Field to order by (avg_value, max_value, count)
            order_direction: ASC or DESC
            
        Returns:
            List of top metrics
        """
        query = f"""
            SELECT 
                metric_name,
                COUNT(*) as count,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                AVG(metric_value) as avg_value
            FROM metrics
            WHERE metric_type = $1
            GROUP BY metric_name
            ORDER BY {order_by} {order_direction}
            LIMIT $2
        """
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, metric_type, limit)
            return [dict(record) for record in records]
    
    async def get_contract_performance_summary(
        self,
        contract_id: UUID,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get performance summary for a contract.
        
        Args:
            contract_id: Contract ID
            days: Number of days to analyze
            
        Returns:
            Performance summary dictionary
        """
        start_time = datetime.utcnow() - timedelta(days=days)
        
        query = """
            SELECT 
                metric_type,
                metric_name,
                COUNT(*) as count,
                AVG(metric_value) as avg_value,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                STDDEV(metric_value) as stddev_value
            FROM metrics
            WHERE contract_id = $1 AND recorded_at >= $2
            GROUP BY metric_type, metric_name
            ORDER BY metric_type, metric_name
        """
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, contract_id, start_time)
            
            summary = {}
            for record in records:
                metric_type = record['metric_type']
                if metric_type not in summary:
                    summary[metric_type] = {}
                
                summary[metric_type][record['metric_name']] = {
                    'count': record['count'],
                    'avg': float(record['avg_value']) if record['avg_value'] else 0,
                    'min': float(record['min_value']) if record['min_value'] else 0,
                    'max': float(record['max_value']) if record['max_value'] else 0,
                    'stddev': float(record['stddev_value']) if record['stddev_value'] else 0
                }
            
            return summary
    
    async def cleanup_old_metrics(
        self,
        days_to_keep: int = 90
    ) -> int:
        """
        Clean up old metrics beyond retention period.
        
        Args:
            days_to_keep: Number of days to retain
            
        Returns:
            Number of deleted records
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        query = "DELETE FROM metrics WHERE recorded_at < $1"
        
        async with self.db.acquire_connection() as conn:
            result = await conn.execute(query, cutoff_date)
            # Extract number from result like "DELETE 150"
            return int(result.split(' ')[1]) if ' ' in result else 0
    
    async def get_metric_types(self) -> List[str]:
        """Get all unique metric types."""
        query = "SELECT DISTINCT metric_type FROM metrics ORDER BY metric_type"
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query)
            return [record['metric_type'] for record in records]
    
    async def get_metric_names_by_type(
        self,
        metric_type: str
    ) -> List[str]:
        """Get all metric names for a specific type."""
        query = """
            SELECT DISTINCT metric_name 
            FROM metrics 
            WHERE metric_type = $1 
            ORDER BY metric_name
        """
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, metric_type)
            return [record['metric_name'] for record in records]


class CacheRepository(BaseRepository):
    """Repository for cache management operations."""
    
    def __init__(self):
        super().__init__('cache_entries')
    
    async def set_cache(
        self,
        cache_key: str,
        cache_value: Any,
        ttl_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Set cache entry.
        
        Args:
            cache_key: Cache key
            cache_value: Value to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            Created cache entry
        """
        data = {
            'cache_key': cache_key,
            'cache_value': self._serialize_jsonb(cache_value),
            'size_bytes': len(str(cache_value).encode('utf-8'))
        }
        
        if ttl_seconds:
            data['expires_at'] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        # Try to update existing entry first
        existing = await self.find_one({'cache_key': cache_key})
        if existing:
            return await self.update(existing['id'], data)
        else:
            return await self.create(data)
    
    async def get_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get cache entry.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        query = """
            SELECT cache_value, expires_at, id
            FROM cache_entries
            WHERE cache_key = $1
        """
        
        async with self.db.acquire_connection() as conn:
            record = await conn.fetchrow(query, cache_key)
            
            if not record:
                return None
            
            # Check expiration
            if record['expires_at'] and record['expires_at'] < datetime.utcnow():
                # Entry expired, delete it
                await self.delete(record['id'])
                return None
            
            # Update access tracking
            await conn.execute(
                "UPDATE cache_entries SET last_accessed = NOW(), access_count = access_count + 1 WHERE id = $1",
                record['id']
            )
            
            return self._deserialize_jsonb(record['cache_value'])
    
    async def delete_cache(self, cache_key: str) -> bool:
        """Delete cache entry by key."""
        query = "DELETE FROM cache_entries WHERE cache_key = $1"
        
        async with self.db.acquire_connection() as conn:
            result = await conn.execute(query, cache_key)
            return result == "DELETE 1"
    
    async def clear_expired(self) -> int:
        """Clear all expired cache entries."""
        query = """
            DELETE FROM cache_entries 
            WHERE expires_at IS NOT NULL AND expires_at < NOW()
        """
        
        async with self.db.acquire_connection() as conn:
            result = await conn.execute(query)
            return int(result.split(' ')[1]) if ' ' in result else 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        query = """
            SELECT 
                COUNT(*) as total_entries,
                COUNT(*) FILTER (WHERE expires_at IS NULL OR expires_at > NOW()) as active_entries,
                COUNT(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at <= NOW()) as expired_entries,
                SUM(size_bytes) as total_size_bytes,
                AVG(access_count) as avg_access_count,
                MAX(last_accessed) as last_access_time
            FROM cache_entries
        """
        
        async with self.db.acquire_connection() as conn:
            record = await conn.fetchrow(query)
            return dict(record) if record else {}
    
    async def get_most_accessed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed cache entries."""
        query = """
            SELECT cache_key, access_count, last_accessed, size_bytes
            FROM cache_entries
            WHERE expires_at IS NULL OR expires_at > NOW()
            ORDER BY access_count DESC
            LIMIT $1
        """
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query, limit)
            return [dict(record) for record in records]
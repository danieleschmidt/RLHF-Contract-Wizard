"""
Database migration runner for RLHF-Contract-Wizard.

Handles database schema migrations, versioning, and rollbacks
with proper transaction management and error handling.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib

from .connection import db_connection


logger = logging.getLogger(__name__)


class Migration:
    """Represents a single database migration."""
    
    def __init__(self, filename: str, content: str):
        """
        Initialize migration.
        
        Args:
            filename: Migration filename
            content: SQL content
        """
        self.filename = filename
        self.content = content
        self.version = self._extract_version(filename)
        self.name = self._extract_name(filename)
        self.checksum = hashlib.sha256(content.encode()).hexdigest()
    
    def _extract_version(self, filename: str) -> str:
        """Extract version from filename like '001_create_tables.sql'."""
        return filename.split('_')[0]
    
    def _extract_name(self, filename: str) -> str:
        """Extract migration name from filename."""
        name_part = '_'.join(filename.split('_')[1:])
        return name_part.replace('.sql', '')


class MigrationRunner:
    """
    Database migration runner.
    
    Manages database schema migrations with proper versioning,
    rollback capabilities, and transaction safety.
    """
    
    def __init__(self, migrations_dir: str = None):
        """
        Initialize migration runner.
        
        Args:
            migrations_dir: Directory containing migration files
        """
        if migrations_dir is None:
            # Default to migrations directory relative to this file
            current_dir = Path(__file__).parent
            migrations_dir = current_dir / "migrations"
        
        self.migrations_dir = Path(migrations_dir)
        self.db = db_connection
    
    async def initialize_migration_table(self):
        """Create migrations tracking table if it doesn't exist."""
        query = """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                version VARCHAR(50) NOT NULL UNIQUE,
                name VARCHAR(255) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                checksum VARCHAR(64) NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                execution_time_ms INTEGER,
                success BOOLEAN DEFAULT true,
                error_message TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_schema_migrations_version 
            ON schema_migrations(version);
            
            CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at 
            ON schema_migrations(applied_at);
        """
        
        async with self.db.acquire_connection() as conn:
            await conn.execute(query)
            logger.info("Migration tracking table initialized")
    
    def _discover_migrations(self) -> List[Migration]:
        """Discover all migration files in the migrations directory."""
        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return []
        
        migrations = []
        
        for file_path in sorted(self.migrations_dir.glob("*.sql")):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                migration = Migration(file_path.name, content)
                migrations.append(migration)
                logger.debug(f"Discovered migration: {migration.filename}")
                
            except Exception as e:
                logger.error(f"Error reading migration file {file_path}: {e}")
        
        return migrations
    
    async def _get_applied_migrations(self) -> Dict[str, Dict[str, Any]]:
        """Get list of already applied migrations."""
        query = """
            SELECT version, name, filename, checksum, applied_at, success
            FROM schema_migrations
            ORDER BY version
        """
        
        async with self.db.acquire_connection() as conn:
            records = await conn.fetch(query)
            
            applied = {}
            for record in records:
                applied[record['version']] = dict(record)
            
            return applied
    
    async def _record_migration(
        self,
        migration: Migration,
        execution_time_ms: int,
        success: bool,
        error_message: Optional[str] = None
    ):
        """Record migration execution in tracking table."""
        query = """
            INSERT INTO schema_migrations 
            (version, name, filename, checksum, execution_time_ms, success, error_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        async with self.db.acquire_connection() as conn:
            await conn.execute(
                query,
                migration.version,
                migration.name,
                migration.filename,
                migration.checksum,
                execution_time_ms,
                success,
                error_message
            )
    
    async def _execute_migration(self, migration: Migration) -> tuple[bool, Optional[str], int]:
        """
        Execute a single migration.
        
        Returns:
            Tuple of (success, error_message, execution_time_ms)
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing migration: {migration.filename}")
            
            async with self.db.transaction() as conn:
                # Split content by semicolons and execute each statement
                statements = [
                    stmt.strip() 
                    for stmt in migration.content.split(';') 
                    if stmt.strip()
                ]
                
                for statement in statements:
                    if statement.strip():
                        await conn.execute(statement)
                
                execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                logger.info(f"Migration {migration.filename} completed in {execution_time}ms")
                
                return True, None, execution_time
                
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            error_message = str(e)
            logger.error(f"Migration {migration.filename} failed: {error_message}")
            return False, error_message, execution_time
    
    async def migrate(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Run database migrations.
        
        Args:
            target_version: Migrate to this version (latest if None)
            
        Returns:
            Migration execution summary
        """
        await self.initialize_migration_table()
        
        # Discover available migrations
        available_migrations = self._discover_migrations()
        if not available_migrations:
            logger.info("No migrations found")
            return {'status': 'no_migrations', 'executed': []}
        
        # Get applied migrations
        applied_migrations = await self._get_applied_migrations()
        
        # Determine which migrations to run
        migrations_to_run = []
        
        for migration in available_migrations:
            # Skip if target version specified and we've passed it
            if target_version and migration.version > target_version:
                break
            
            # Check if migration was already applied
            if migration.version in applied_migrations:
                applied = applied_migrations[migration.version]
                
                # Verify checksum hasn't changed
                if applied['checksum'] != migration.checksum:
                    logger.warning(
                        f"Migration {migration.filename} checksum changed! "
                        f"Applied: {applied['checksum'][:8]}..., "
                        f"Current: {migration.checksum[:8]}..."
                    )
                
                # Skip if successfully applied
                if applied['success']:
                    logger.debug(f"Migration {migration.filename} already applied")
                    continue
                else:
                    logger.info(f"Re-running failed migration: {migration.filename}")
                    migrations_to_run.append(migration)
            else:
                migrations_to_run.append(migration)
        
        if not migrations_to_run:
            logger.info("All migrations up to date")
            return {'status': 'up_to_date', 'executed': []}
        
        # Execute migrations
        executed_migrations = []
        failed_migrations = []
        
        for migration in migrations_to_run:
            success, error_message, execution_time = await self._execute_migration(migration)
            
            # Record migration execution
            await self._record_migration(migration, execution_time, success, error_message)
            
            if success:
                executed_migrations.append({
                    'version': migration.version,
                    'name': migration.name,
                    'execution_time_ms': execution_time
                })
                logger.info(f"✓ Migration {migration.filename} completed")
            else:
                failed_migrations.append({
                    'version': migration.version,
                    'name': migration.name,
                    'error': error_message
                })
                logger.error(f"✗ Migration {migration.filename} failed")
                
                # Stop on first failure
                break
        
        # Summary
        summary = {
            'status': 'completed' if not failed_migrations else 'failed',
            'executed': executed_migrations,
            'failed': failed_migrations,
            'total_executed': len(executed_migrations),
            'total_failed': len(failed_migrations)
        }
        
        if executed_migrations:
            logger.info(f"Successfully executed {len(executed_migrations)} migrations")
        
        if failed_migrations:
            logger.error(f"Failed to execute {len(failed_migrations)} migrations")
        
        return summary
    
    async def rollback(self, target_version: str) -> Dict[str, Any]:
        """
        Rollback to a specific version.
        
        Args:
            target_version: Version to rollback to
            
        Returns:
            Rollback execution summary
        """
        # Get applied migrations
        applied_migrations = await self._get_applied_migrations()
        
        # Find migrations to rollback (those after target version)
        migrations_to_rollback = []
        
        for version in sorted(applied_migrations.keys(), reverse=True):
            if version > target_version:
                migrations_to_rollback.append(applied_migrations[version])
            else:
                break
        
        if not migrations_to_rollback:
            logger.info(f"Already at or before version {target_version}")
            return {'status': 'up_to_date', 'rolled_back': []}
        
        # Note: This is a simplified rollback implementation
        # In practice, you'd need separate rollback SQL scripts
        logger.warning("Rollback functionality requires manual intervention")
        logger.warning("Consider restoring from a database backup")
        
        return {
            'status': 'manual_intervention_required',
            'migrations_to_rollback': [m['filename'] for m in migrations_to_rollback],
            'message': 'Rollback requires manual SQL scripts or database restore'
        }
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        await self.initialize_migration_table()
        
        # Get available and applied migrations
        available_migrations = self._discover_migrations()
        applied_migrations = await self._get_applied_migrations()
        
        # Determine status
        pending_migrations = []
        failed_migrations = []
        
        for migration in available_migrations:
            if migration.version not in applied_migrations:
                pending_migrations.append({
                    'version': migration.version,
                    'name': migration.name,
                    'filename': migration.filename
                })
            elif not applied_migrations[migration.version]['success']:
                failed_migrations.append({
                    'version': migration.version,
                    'name': migration.name,
                    'filename': migration.filename,
                    'error': applied_migrations[migration.version].get('error_message')
                })
        
        # Get latest applied migration
        latest_applied = None
        if applied_migrations:
            latest_version = max(applied_migrations.keys())
            latest_applied = applied_migrations[latest_version]
        
        return {
            'current_version': latest_applied['version'] if latest_applied else None,
            'total_available': len(available_migrations),
            'total_applied': len([m for m in applied_migrations.values() if m['success']]),
            'pending_migrations': pending_migrations,
            'failed_migrations': failed_migrations,
            'is_up_to_date': len(pending_migrations) == 0 and len(failed_migrations) == 0
        }
    
    async def validate_schema(self) -> Dict[str, Any]:
        """Validate current database schema against expected state."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'tables_checked': 0,
            'constraints_checked': 0
        }
        
        try:
            # Check if all expected tables exist
            expected_tables = [
                'contracts', 'stakeholders', 'constraints', 'reward_functions',
                'deployments', 'verification_results', 'contract_events',
                'metrics', 'cache_entries', 'schema_migrations'
            ]
            
            async with self.db.acquire_connection() as conn:
                for table_name in expected_tables:
                    exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                        table_name
                    )
                    
                    if not exists:
                        validation_results['valid'] = False
                        validation_results['errors'].append(f"Table '{table_name}' does not exist")
                    else:
                        validation_results['tables_checked'] += 1
                
                # Check for required columns in key tables
                required_columns = {
                    'contracts': ['id', 'name', 'version', 'contract_hash'],
                    'stakeholders': ['id', 'contract_id', 'name', 'weight'],
                    'constraints': ['id', 'contract_id', 'name', 'constraint_type']
                }
                
                for table_name, columns in required_columns.items():
                    for column in columns:
                        exists = await conn.fetchval(
                            """
                            SELECT EXISTS(
                                SELECT 1 FROM information_schema.columns 
                                WHERE table_name = $1 AND column_name = $2
                            )
                            """,
                            table_name, column
                        )
                        
                        if not exists:
                            validation_results['valid'] = False
                            validation_results['errors'].append(
                                f"Column '{column}' missing from table '{table_name}'"
                            )
                        else:
                            validation_results['constraints_checked'] += 1
                
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Schema validation error: {str(e)}")
        
        return validation_results


# CLI interface for migration runner
async def main():
    """CLI interface for running migrations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RLHF Contract Database Migration Runner')
    parser.add_argument('command', choices=['migrate', 'rollback', 'status', 'validate'],
                       help='Migration command to execute')
    parser.add_argument('--version', help='Target version for migrate/rollback')
    parser.add_argument('--migrations-dir', help='Path to migrations directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    runner = MigrationRunner(args.migrations_dir)
    
    try:
        if args.command == 'migrate':
            result = await runner.migrate(args.version)
            print(f"Migration result: {result}")
        
        elif args.command == 'rollback':
            if not args.version:
                print("Error: --version required for rollback")
                return
            result = await runner.rollback(args.version)
            print(f"Rollback result: {result}")
        
        elif args.command == 'status':
            status = await runner.get_migration_status()
            print(f"Migration status: {status}")
        
        elif args.command == 'validate':
            validation = await runner.validate_schema()
            print(f"Schema validation: {validation}")
            
    except Exception as e:
        logger.error(f"Migration command failed: {e}")
        raise
    
    finally:
        await runner.db.close()


if __name__ == '__main__':
    asyncio.run(main())
-- Migration 001: Create core contracts table
-- Created: 2025-01-XX
-- Description: Initial contracts table with core metadata fields

BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create contracts table
CREATE TABLE contracts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    contract_hash VARCHAR(64) UNIQUE NOT NULL,
    creator VARCHAR(255) NOT NULL,
    jurisdiction VARCHAR(100) DEFAULT 'Global',
    regulatory_framework VARCHAR(100),
    aggregation_strategy VARCHAR(50) DEFAULT 'weighted_average',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'draft',
    is_active BOOLEAN DEFAULT true,
    
    -- Constraints
    UNIQUE(name, version),
    CHECK (LENGTH(name) >= 3),
    CHECK (LENGTH(version) >= 5),
    CHECK (status IN ('draft', 'validated', 'deployed', 'deprecated'))
);

-- Create indexes
CREATE INDEX idx_contracts_status ON contracts(status);
CREATE INDEX idx_contracts_created_at ON contracts(created_at);
CREATE INDEX idx_contracts_hash ON contracts(contract_hash);

-- Create update trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger
CREATE TRIGGER update_contracts_updated_at BEFORE UPDATE ON contracts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

COMMIT;
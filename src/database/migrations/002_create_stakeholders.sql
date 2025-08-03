-- Migration 002: Create stakeholders and preferences tables
-- Created: 2025-01-XX
-- Description: Tables for stakeholder management and voting

BEGIN;

-- Create stakeholders table
CREATE TABLE stakeholders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    weight DECIMAL(10,6) NOT NULL,
    voting_power DECIMAL(10,6) DEFAULT 1.0,
    ethereum_address VARCHAR(42),
    preferences JSONB DEFAULT '{}',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(contract_id, name),
    CHECK (weight > 0),
    CHECK (weight <= 1),
    CHECK (voting_power >= 0),
    CHECK (ethereum_address IS NULL OR LENGTH(ethereum_address) = 42)
);

-- Create constraints table
CREATE TABLE constraints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    constraint_type VARCHAR(50) NOT NULL,
    severity DECIMAL(3,2) DEFAULT 1.0,
    violation_penalty DECIMAL(10,6) DEFAULT -1.0,
    enabled BOOLEAN DEFAULT true,
    
    -- Legal-Blocks specification
    legal_blocks_spec TEXT,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(contract_id, name),
    CHECK (severity >= 0 AND severity <= 1),
    CHECK (violation_penalty <= 0),
    CHECK (constraint_type IN ('requires', 'ensures', 'invariant', 'forall', 'exists'))
);

-- Create amendments table for governance
CREATE TABLE amendments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    proposer VARCHAR(255) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    changes JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'draft',
    required_consensus DECIMAL(3,2) DEFAULT 0.66,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CHECK (required_consensus > 0 AND required_consensus <= 1),
    CHECK (status IN ('draft', 'active', 'approved', 'rejected', 'expired')),
    CHECK (expires_at > created_at)
);

-- Create votes table
CREATE TABLE votes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    amendment_id UUID NOT NULL REFERENCES amendments(id) ON DELETE CASCADE,
    stakeholder VARCHAR(255) NOT NULL,
    support BOOLEAN NOT NULL,
    weight DECIMAL(10,6) NOT NULL,
    reasoning TEXT,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(amendment_id, stakeholder),
    CHECK (weight > 0)
);

-- Create deployment history table
CREATE TABLE deployments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id),
    network VARCHAR(100) NOT NULL,
    contract_address VARCHAR(42),
    transaction_hash VARCHAR(66),
    gas_used BIGINT,
    deployment_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    
    -- Metadata
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CHECK (deployment_status IN ('pending', 'deployed', 'failed')),
    CHECK (contract_address IS NULL OR LENGTH(contract_address) = 42),
    CHECK (transaction_hash IS NULL OR LENGTH(transaction_hash) = 66)
);

-- Create indexes
CREATE INDEX idx_stakeholders_contract_id ON stakeholders(contract_id);
CREATE INDEX idx_stakeholders_weight ON stakeholders(weight);
CREATE INDEX idx_constraints_contract_id ON constraints(contract_id);
CREATE INDEX idx_constraints_enabled ON constraints(enabled);
CREATE INDEX idx_amendments_contract_id ON amendments(contract_id);
CREATE INDEX idx_amendments_status ON amendments(status);
CREATE INDEX idx_amendments_expires_at ON amendments(expires_at);
CREATE INDEX idx_votes_amendment_id ON votes(amendment_id);
CREATE INDEX idx_deployments_contract_id ON deployments(contract_id);
CREATE INDEX idx_deployments_network ON deployments(network);

-- Apply update triggers
CREATE TRIGGER update_stakeholders_updated_at BEFORE UPDATE ON stakeholders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_constraints_updated_at BEFORE UPDATE ON constraints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_amendments_updated_at BEFORE UPDATE ON amendments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE VIEW active_contracts AS
SELECT c.*, 
       COUNT(s.id) as stakeholder_count,
       COUNT(con.id) as constraint_count
FROM contracts c
LEFT JOIN stakeholders s ON c.id = s.contract_id
LEFT JOIN constraints con ON c.id = con.contract_id
WHERE c.is_active = true
GROUP BY c.id;

CREATE VIEW contract_governance AS
SELECT c.name as contract_name,
       c.version,
       a.id as amendment_id,
       a.title,
       a.status,
       COUNT(v.id) as total_votes,
       SUM(CASE WHEN v.support THEN v.weight ELSE 0 END) as support_weight,
       SUM(v.weight) as total_weight,
       a.required_consensus
FROM contracts c
JOIN amendments a ON c.id = a.contract_id
LEFT JOIN votes v ON a.id = v.amendment_id
GROUP BY c.id, c.name, c.version, a.id, a.title, a.status, a.required_consensus;

COMMIT;
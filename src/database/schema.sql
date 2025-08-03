-- RLHF-Contract-Wizard Database Schema
-- PostgreSQL database schema for contract storage and management

-- Enable UUID extension for generating unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable timestamp functions
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Contracts table - stores main contract metadata
CREATE TABLE IF NOT EXISTS contracts (
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
    status VARCHAR(50) DEFAULT 'draft', -- draft, validated, deployed, deprecated
    is_active BOOLEAN DEFAULT true,
    
    -- Constraints
    UNIQUE(name, version),
    CHECK (LENGTH(name) >= 3),
    CHECK (LENGTH(version) >= 5), -- semantic versioning
    CHECK (status IN ('draft', 'validated', 'deployed', 'deprecated'))
);

-- Stakeholders table - stores stakeholder information
CREATE TABLE IF NOT EXISTS stakeholders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    weight DECIMAL(10, 8) NOT NULL,
    voting_power DECIMAL(10, 8) DEFAULT 1.0,
    wallet_address VARCHAR(42), -- Ethereum address format
    
    -- Additional stakeholder metadata
    preferences JSONB DEFAULT '{}',
    contact_info JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(contract_id, name),
    CHECK (weight > 0 AND weight <= 1),
    CHECK (voting_power >= 0),
    CHECK (wallet_address IS NULL OR LENGTH(wallet_address) = 42)
);

-- Constraints table - stores contract constraints and rules
CREATE TABLE IF NOT EXISTS constraints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    constraint_type VARCHAR(50) NOT NULL, -- requires, ensures, invariant, etc.
    
    -- Constraint specification
    legal_blocks_spec TEXT,
    severity DECIMAL(3, 2) DEFAULT 1.0,
    violation_penalty DECIMAL(10, 4) DEFAULT -1.0,
    
    -- Status
    enabled BOOLEAN DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(contract_id, name),
    CHECK (severity >= 0 AND severity <= 10),
    CHECK (constraint_type IN ('requires', 'ensures', 'invariant', 'forall', 'exists'))
);

-- Reward functions table - stores reward function metadata
CREATE TABLE IF NOT EXISTS reward_functions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    stakeholder_name VARCHAR(255),
    function_name VARCHAR(255) NOT NULL,
    function_hash VARCHAR(64) NOT NULL,
    
    -- Function metadata
    description TEXT,
    implementation_language VARCHAR(50) DEFAULT 'python',
    function_signature TEXT,
    
    -- Legal blocks specification
    legal_blocks_spec TEXT,
    
    -- Performance metadata
    estimated_compute_cost DECIMAL(10, 4),
    max_execution_time_ms INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(contract_id, stakeholder_name, function_name),
    CHECK (estimated_compute_cost >= 0),
    CHECK (max_execution_time_ms > 0)
);

-- Deployments table - tracks contract deployments to blockchain
CREATE TABLE IF NOT EXISTS deployments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    
    -- Blockchain details
    network VARCHAR(50) NOT NULL,
    contract_address VARCHAR(42) NOT NULL,
    transaction_hash VARCHAR(66) NOT NULL,
    block_number BIGINT,
    
    -- Deployment metadata
    deployer_address VARCHAR(42) NOT NULL,
    gas_used BIGINT,
    gas_price BIGINT,
    deployment_cost_wei BIGINT,
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending', -- pending, confirmed, failed
    
    -- Timestamps
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confirmed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    UNIQUE(network, contract_address),
    CHECK (LENGTH(contract_address) = 42),
    CHECK (LENGTH(transaction_hash) = 66),
    CHECK (status IN ('pending', 'confirmed', 'failed'))
);

-- Verification results table - stores formal verification results
CREATE TABLE IF NOT EXISTS verification_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    
    -- Verification metadata
    backend VARCHAR(50) NOT NULL, -- z3, lean, cbmc, etc.
    total_properties INTEGER NOT NULL,
    proved_properties INTEGER NOT NULL,
    failed_properties INTEGER NOT NULL,
    verification_time_ms BIGINT NOT NULL,
    
    -- Results
    all_proofs_valid BOOLEAN NOT NULL,
    proof_results JSONB NOT NULL DEFAULT '[]',
    
    -- Timestamps
    verified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CHECK (total_properties >= 0),
    CHECK (proved_properties >= 0),
    CHECK (failed_properties >= 0),
    CHECK (proved_properties + failed_properties <= total_properties),
    CHECK (verification_time_ms >= 0)
);

-- Contract events table - audit trail of contract changes
CREATE TABLE IF NOT EXISTS contract_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB NOT NULL DEFAULT '{}',
    actor VARCHAR(255), -- who performed the action
    
    -- Context
    description TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamp
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CHECK (event_type IN (
        'created', 'updated', 'validated', 'deployed', 'verified',
        'stakeholder_added', 'stakeholder_removed', 'constraint_added',
        'constraint_removed', 'reward_function_added', 'reward_function_removed'
    ))
);

-- Metrics table - performance and usage metrics
CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES contracts(id) ON DELETE CASCADE,
    
    -- Metric details
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(20, 8) NOT NULL,
    unit VARCHAR(50),
    
    -- Context
    context JSONB DEFAULT '{}',
    
    -- Timestamp
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for efficient querying
    INDEX idx_metrics_contract_type (contract_id, metric_type),
    INDEX idx_metrics_recorded_at (recorded_at)
);

-- Cache table - for caching computed results
CREATE TABLE IF NOT EXISTS cache_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    cache_value JSONB NOT NULL,
    
    -- Expiration
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    
    -- Size tracking
    size_bytes INTEGER,
    
    -- Constraints
    CHECK (access_count >= 0)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_contracts_status ON contracts(status);
CREATE INDEX IF NOT EXISTS idx_contracts_created_at ON contracts(created_at);
CREATE INDEX IF NOT EXISTS idx_contracts_hash ON contracts(contract_hash);

CREATE INDEX IF NOT EXISTS idx_stakeholders_contract_id ON stakeholders(contract_id);
CREATE INDEX IF NOT EXISTS idx_stakeholders_weight ON stakeholders(weight);

CREATE INDEX IF NOT EXISTS idx_constraints_contract_id ON constraints(contract_id);
CREATE INDEX IF NOT EXISTS idx_constraints_enabled ON constraints(enabled);
CREATE INDEX IF NOT EXISTS idx_constraints_type ON constraints(constraint_type);

CREATE INDEX IF NOT EXISTS idx_reward_functions_contract_id ON reward_functions(contract_id);
CREATE INDEX IF NOT EXISTS idx_reward_functions_stakeholder ON reward_functions(stakeholder_name);

CREATE INDEX IF NOT EXISTS idx_deployments_contract_id ON deployments(contract_id);
CREATE INDEX IF NOT EXISTS idx_deployments_network ON deployments(network);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);

CREATE INDEX IF NOT EXISTS idx_verification_contract_id ON verification_results(contract_id);
CREATE INDEX IF NOT EXISTS idx_verification_backend ON verification_results(backend);

CREATE INDEX IF NOT EXISTS idx_events_contract_id ON contract_events(contract_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON contract_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON contract_events(occurred_at);

CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache_entries(expires_at);
CREATE INDEX IF NOT EXISTS idx_cache_last_accessed ON cache_entries(last_accessed);

-- Triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to tables with updated_at columns
CREATE TRIGGER update_contracts_updated_at BEFORE UPDATE ON contracts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_stakeholders_updated_at BEFORE UPDATE ON stakeholders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_constraints_updated_at BEFORE UPDATE ON constraints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_reward_functions_updated_at BEFORE UPDATE ON reward_functions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM cache_entries 
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get contract statistics
CREATE OR REPLACE FUNCTION get_contract_stats(contract_uuid UUID)
RETURNS TABLE(
    stakeholder_count INTEGER,
    constraint_count INTEGER,
    reward_function_count INTEGER,
    deployment_count INTEGER,
    verification_count INTEGER,
    last_activity TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*)::INTEGER FROM stakeholders WHERE contract_id = contract_uuid),
        (SELECT COUNT(*)::INTEGER FROM constraints WHERE contract_id = contract_uuid),
        (SELECT COUNT(*)::INTEGER FROM reward_functions WHERE contract_id = contract_uuid),
        (SELECT COUNT(*)::INTEGER FROM deployments WHERE contract_id = contract_uuid),
        (SELECT COUNT(*)::INTEGER FROM verification_results WHERE contract_id = contract_uuid),
        (SELECT MAX(occurred_at) FROM contract_events WHERE contract_id = contract_uuid);
END;
$$ LANGUAGE plpgsql;

-- Views for common queries
CREATE OR REPLACE VIEW contract_summary AS
SELECT 
    c.id,
    c.name,
    c.version,
    c.status,
    c.created_at,
    c.updated_at,
    COUNT(DISTINCT s.id) as stakeholder_count,
    COUNT(DISTINCT ct.id) as constraint_count,
    COUNT(DISTINCT rf.id) as reward_function_count,
    COUNT(DISTINCT d.id) as deployment_count,
    MAX(d.deployed_at) as last_deployment
FROM contracts c
LEFT JOIN stakeholders s ON c.id = s.contract_id
LEFT JOIN constraints ct ON c.id = ct.contract_id
LEFT JOIN reward_functions rf ON c.id = rf.contract_id
LEFT JOIN deployments d ON c.id = d.contract_id
GROUP BY c.id, c.name, c.version, c.status, c.created_at, c.updated_at;

-- Create default admin user (for development)
INSERT INTO contract_events (contract_id, event_type, description, actor) 
VALUES (uuid_generate_v4(), 'created', 'Database schema initialized', 'system')
ON CONFLICT DO NOTHING;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rlhf_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rlhf_user;
-- GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO rlhf_user;
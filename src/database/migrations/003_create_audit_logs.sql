-- Migration 003: Create audit and monitoring tables
-- Created: 2025-01-XX
-- Description: Tables for audit logging, monitoring, and compliance tracking

BEGIN;

-- Create audit logs table for immutable record keeping
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    
    -- Event details
    event_data JSONB NOT NULL,
    previous_state JSONB,
    new_state JSONB,
    
    -- Request context
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(100),
    
    -- Compliance and legal
    jurisdiction VARCHAR(100),
    regulatory_framework VARCHAR(100),
    retention_until TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CHECK (event_type IN ('create', 'update', 'delete', 'deploy', 'vote', 'execute', 'validate')),
    CHECK (entity_type IN ('contract', 'stakeholder', 'constraint', 'amendment', 'vote', 'deployment'))
);

-- Create performance metrics table
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(100) NOT NULL,
    contract_id UUID REFERENCES contracts(id),
    
    -- Performance data
    execution_time_ms DECIMAL(10,3),
    memory_usage_mb DECIMAL(10,3),
    constraint_evaluations INTEGER,
    cache_hits INTEGER,
    cache_misses INTEGER,
    
    -- Context
    state_dimension INTEGER,
    action_dimension INTEGER,
    stakeholder_count INTEGER,
    constraint_count INTEGER,
    
    -- Metadata
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CHECK (metric_type IN ('reward_computation', 'constraint_check', 'contract_validation', 'deployment')),
    CHECK (execution_time_ms >= 0),
    CHECK (memory_usage_mb >= 0)
);

-- Create compliance violations table
CREATE TABLE compliance_violations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id),
    violation_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    
    -- Violation details
    constraint_name VARCHAR(255),
    violation_description TEXT NOT NULL,
    violation_data JSONB,
    
    -- Context when violation occurred
    state_hash VARCHAR(64),
    action_hash VARCHAR(64),
    context_data JSONB,
    
    -- Resolution
    status VARCHAR(50) DEFAULT 'open',
    resolution_description TEXT,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(255),
    
    -- Metadata
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    CHECK (status IN ('open', 'acknowledged', 'resolved', 'false_positive')),
    CHECK (violation_type IN ('safety', 'privacy', 'fairness', 'legal', 'performance'))
);

-- Create contract executions table for monitoring
CREATE TABLE contract_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID NOT NULL REFERENCES contracts(id),
    
    -- Execution details
    state_hash VARCHAR(64) NOT NULL,
    action_hash VARCHAR(64) NOT NULL,
    context_hash VARCHAR(64),
    reward_value DECIMAL(10,6),
    
    -- Performance
    execution_time_ms DECIMAL(10,3),
    constraint_checks INTEGER,
    violations_detected INTEGER,
    
    -- Metadata
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    executed_by VARCHAR(255),
    
    -- Constraints
    CHECK (execution_time_ms >= 0),
    CHECK (constraint_checks >= 0),
    CHECK (violations_detected >= 0)
);

-- Create blockchain transactions table
CREATE TABLE blockchain_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES contracts(id),
    deployment_id UUID REFERENCES deployments(id),
    
    -- Transaction details
    transaction_hash VARCHAR(66) UNIQUE NOT NULL,
    block_number BIGINT,
    block_hash VARCHAR(66),
    gas_used BIGINT,
    gas_price BIGINT,
    transaction_fee DECIMAL(18,6),
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending',
    confirmation_count INTEGER DEFAULT 0,
    
    -- Network details
    network VARCHAR(100) NOT NULL,
    from_address VARCHAR(42) NOT NULL,
    to_address VARCHAR(42),
    
    -- Metadata
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confirmed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CHECK (status IN ('pending', 'confirmed', 'failed', 'reverted')),
    CHECK (LENGTH(transaction_hash) = 66),
    CHECK (LENGTH(from_address) = 42),
    CHECK (to_address IS NULL OR LENGTH(to_address) = 42)
);

-- Create indexes for efficient querying
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_entity_type ON audit_logs(entity_type);
CREATE INDEX idx_audit_logs_entity_id ON audit_logs(entity_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_retention ON audit_logs(retention_until) WHERE retention_until IS NOT NULL;

CREATE INDEX idx_performance_metrics_type ON performance_metrics(metric_type);
CREATE INDEX idx_performance_metrics_contract_id ON performance_metrics(contract_id);
CREATE INDEX idx_performance_metrics_measured_at ON performance_metrics(measured_at);

CREATE INDEX idx_compliance_violations_contract_id ON compliance_violations(contract_id);
CREATE INDEX idx_compliance_violations_severity ON compliance_violations(severity);
CREATE INDEX idx_compliance_violations_status ON compliance_violations(status);
CREATE INDEX idx_compliance_violations_detected_at ON compliance_violations(detected_at);

CREATE INDEX idx_contract_executions_contract_id ON contract_executions(contract_id);
CREATE INDEX idx_contract_executions_executed_at ON contract_executions(executed_at);
CREATE INDEX idx_contract_executions_state_hash ON contract_executions(state_hash);

CREATE INDEX idx_blockchain_transactions_hash ON blockchain_transactions(transaction_hash);
CREATE INDEX idx_blockchain_transactions_status ON blockchain_transactions(status);
CREATE INDEX idx_blockchain_transactions_network ON blockchain_transactions(network);
CREATE INDEX idx_blockchain_transactions_contract_id ON blockchain_transactions(contract_id);

-- Create partitioned table for high-volume audit logs (monthly partitions)
CREATE TABLE audit_logs_partitioned (LIKE audit_logs INCLUDING ALL)
PARTITION BY RANGE (created_at);

-- Create initial partitions (would be automated in production)
CREATE TABLE audit_logs_2025_01 PARTITION OF audit_logs_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Create views for monitoring and reporting
CREATE VIEW contract_health AS
SELECT 
    c.id,
    c.name,
    c.version,
    c.status,
    COUNT(DISTINCT ce.id) as execution_count,
    AVG(ce.execution_time_ms) as avg_execution_time,
    COUNT(DISTINCT cv.id) as violation_count,
    MAX(ce.executed_at) as last_execution
FROM contracts c
LEFT JOIN contract_executions ce ON c.id = ce.contract_id 
    AND ce.executed_at > NOW() - INTERVAL '24 hours'
LEFT JOIN compliance_violations cv ON c.id = cv.contract_id 
    AND cv.detected_at > NOW() - INTERVAL '24 hours'
    AND cv.status = 'open'
WHERE c.is_active = true
GROUP BY c.id, c.name, c.version, c.status;

CREATE VIEW violation_summary AS
SELECT 
    violation_type,
    severity,
    COUNT(*) as count,
    COUNT(CASE WHEN status = 'open' THEN 1 END) as open_count,
    AVG(EXTRACT(EPOCH FROM (COALESCE(resolved_at, NOW()) - detected_at))/3600) as avg_resolution_hours
FROM compliance_violations
WHERE detected_at > NOW() - INTERVAL '30 days'
GROUP BY violation_type, severity
ORDER BY severity DESC, count DESC;

-- Create function for automatic audit log creation
CREATE OR REPLACE FUNCTION create_audit_log(
    p_event_type VARCHAR(100),
    p_entity_type VARCHAR(100),
    p_entity_id UUID,
    p_user_id VARCHAR(255),
    p_event_data JSONB,
    p_previous_state JSONB DEFAULT NULL,
    p_new_state JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    log_id UUID;
BEGIN
    INSERT INTO audit_logs (
        event_type, entity_type, entity_id, user_id,
        event_data, previous_state, new_state,
        retention_until
    ) VALUES (
        p_event_type, p_entity_type, p_entity_id, p_user_id,
        p_event_data, p_previous_state, p_new_state,
        NOW() + INTERVAL '7 years'  -- Default retention period
    ) RETURNING id INTO log_id;
    
    RETURN log_id;
END;
$$ LANGUAGE plpgsql;

-- Create function for cleanup of old records
CREATE OR REPLACE FUNCTION cleanup_expired_records() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean up expired audit logs
    DELETE FROM audit_logs 
    WHERE retention_until IS NOT NULL 
      AND retention_until < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up old performance metrics (keep 90 days)
    DELETE FROM performance_metrics 
    WHERE measured_at < NOW() - INTERVAL '90 days';
    
    -- Clean up old contract executions (keep 30 days for performance)
    DELETE FROM contract_executions 
    WHERE executed_at < NOW() - INTERVAL '30 days';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMIT;
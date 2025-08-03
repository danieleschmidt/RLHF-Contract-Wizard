-- Demo contract data for development and testing
-- This file contains sample contracts for demonstration purposes

BEGIN;

-- Demo Contract 1: Content Moderation AI
INSERT INTO contracts (
    name, 
    version, 
    contract_hash, 
    creator, 
    jurisdiction,
    status,
    aggregation_strategy
) VALUES (
    'ContentModerator-SafeGuard',
    '1.0.0',
    'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456',
    'ModeratorCorp',
    'EU',
    'validated',
    'weighted_average'
);

-- Get the contract ID for stakeholders
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'ContentModerator-SafeGuard' AND version = '1.0.0'
)
INSERT INTO stakeholders (contract_id, name, weight, voting_power, wallet_address)
SELECT 
    contract_id,
    unnest(ARRAY['platform_operator', 'user_community', 'advertisers', 'safety_board']),
    unnest(ARRAY[0.35, 0.30, 0.15, 0.20]),
    unnest(ARRAY[1.0, 1.0, 1.0, 2.0]),
    unnest(ARRAY['0x1234567890123456789012345678901234567890', 
                  '0x2345678901234567890123456789012345678901',
                  '0x3456789012345678901234567890123456789012',
                  '0x4567890123456789012345678901234567890123'])
FROM contract_ref;

-- Add constraints for content moderation
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'ContentModerator-SafeGuard' AND version = '1.0.0'
)
INSERT INTO constraints (contract_id, name, description, constraint_type, severity, violation_penalty)
SELECT 
    contract_id,
    unnest(ARRAY['no_illegal_content', 'no_hate_speech', 'preserve_context', 'transparency']),
    unnest(ARRAY[
        'Content must not contain illegal material',
        'Content must not contain hate speech or harassment',
        'Moderation must preserve original context and meaning',
        'Moderation decisions must be explainable and transparent'
    ]),
    unnest(ARRAY['invariant', 'invariant', 'ensures', 'ensures']),
    unnest(ARRAY[10.0, 9.0, 7.0, 6.0]),
    unnest(ARRAY[-10.0, -9.0, -5.0, -3.0])
FROM contract_ref;

-- Demo Contract 2: Medical Diagnosis AI
INSERT INTO contracts (
    name, 
    version, 
    contract_hash, 
    creator, 
    jurisdiction,
    regulatory_framework,
    status,
    aggregation_strategy
) VALUES (
    'MedicalDiagnosisAI-Guardian',
    '2.1.0',
    'b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567b',
    'HealthTech Solutions',
    'US',
    'FDA_510k',
    'deployed',
    'nash_bargaining'
);

-- Medical AI stakeholders
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'MedicalDiagnosisAI-Guardian' AND version = '2.1.0'
)
INSERT INTO stakeholders (contract_id, name, weight, voting_power, wallet_address)
SELECT 
    contract_id,
    unnest(ARRAY['hospital_system', 'physicians', 'patients_advocate', 'fda_regulatory']),
    unnest(ARRAY[0.25, 0.35, 0.25, 0.15]),
    unnest(ARRAY[1.0, 1.5, 1.0, 2.0]),
    unnest(ARRAY['0x5678901234567890123456789012345678901234', 
                  '0x6789012345678901234567890123456789012345',
                  '0x7890123456789012345678901234567890123456',
                  '0x8901234567890123456789012345678901234567'])
FROM contract_ref;

-- Medical AI constraints
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'MedicalDiagnosisAI-Guardian' AND version = '2.1.0'
)
INSERT INTO constraints (contract_id, name, description, constraint_type, severity, violation_penalty)
SELECT 
    contract_id,
    unnest(ARRAY['physician_oversight', 'evidence_based', 'confidence_reporting', 'privacy_hipaa']),
    unnest(ARRAY[
        'All diagnoses must require physician review before action',
        'Diagnoses must be based on peer-reviewed medical evidence',
        'System must report confidence intervals for all diagnoses',
        'Patient data must comply with HIPAA privacy requirements'
    ]),
    unnest(ARRAY['invariant', 'requires', 'ensures', 'invariant']),
    unnest(ARRAY[10.0, 8.0, 7.0, 10.0]),
    unnest(ARRAY[-100.0, -50.0, -20.0, -100.0])
FROM contract_ref;

-- Demo Contract 3: Financial Trading AI
INSERT INTO contracts (
    name, 
    version, 
    contract_hash, 
    creator, 
    jurisdiction,
    regulatory_framework,
    status,
    aggregation_strategy
) VALUES (
    'TradingAI-RiskManager',
    '1.5.2',
    'c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567c89',
    'FinanceAI Corp',
    'UK',
    'FCA_MiFID_II',
    'validated',
    'utilitarian'
);

-- Trading AI stakeholders
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'TradingAI-RiskManager' AND version = '1.5.2'
)
INSERT INTO stakeholders (contract_id, name, weight, voting_power, wallet_address)
SELECT 
    contract_id,
    unnest(ARRAY['fund_managers', 'investors', 'risk_committee', 'compliance_officer']),
    unnest(ARRAY[0.40, 0.30, 0.20, 0.10]),
    unnest(ARRAY[1.0, 1.0, 2.0, 3.0]),
    unnest(ARRAY['0x9012345678901234567890123456789012345678', 
                  '0xa123456789012345678901234567890123456789',
                  '0xb234567890123456789012345678901234567890',
                  '0xc345678901234567890123456789012345678901'])
FROM contract_ref;

-- Sample reward functions
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'ContentModerator-SafeGuard' AND version = '1.0.0'
)
INSERT INTO reward_functions (
    contract_id, 
    stakeholder_name, 
    function_name, 
    function_hash,
    description,
    legal_blocks_spec
)
SELECT 
    contract_id,
    'platform_operator',
    'platform_efficiency_reward',
    'hash_func_001',
    'Rewards based on platform efficiency and user engagement',
    'REQUIRES: user_engagement > 0.7 AND processing_time < 100ms'
FROM contract_ref;

-- Sample deployment records
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'MedicalDiagnosisAI-Guardian' AND version = '2.1.0'
)
INSERT INTO deployments (
    contract_id,
    network,
    contract_address,
    transaction_hash,
    block_number,
    deployer_address,
    gas_used,
    gas_price,
    status
)
SELECT 
    contract_id,
    'ethereum_mainnet',
    '0xd456789012345678901234567890123456789012',
    '0xe56789012345678901234567890123456789012345678901234567890123456789',
    18500000,
    '0x5678901234567890123456789012345678901234',
    2500000,
    20000000000,
    'confirmed'
FROM contract_ref;

-- Sample verification results
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'ContentModerator-SafeGuard' AND version = '1.0.0'
)
INSERT INTO verification_results (
    contract_id,
    backend,
    total_properties,
    proved_properties,
    failed_properties,
    verification_time_ms,
    all_proofs_valid,
    proof_results
)
SELECT 
    contract_id,
    'z3',
    12,
    11,
    1,
    5420,
    false,
    '[
        {"property": "no_illegal_content", "proved": true, "time_ms": 450},
        {"property": "no_hate_speech", "proved": true, "time_ms": 380},
        {"property": "preserve_context", "proved": true, "time_ms": 1200},
        {"property": "transparency", "proved": false, "time_ms": 890, "error": "Timeout"}
    ]'::jsonb
FROM contract_ref;

-- Sample contract events
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'ContentModerator-SafeGuard' AND version = '1.0.0'
)
INSERT INTO contract_events (contract_id, event_type, description, actor, event_data)
SELECT 
    contract_id,
    unnest(ARRAY['created', 'validated', 'stakeholder_added']),
    unnest(ARRAY[
        'Contract created with initial configuration',
        'Contract passed formal verification',
        'Safety board stakeholder added with enhanced voting power'
    ]),
    unnest(ARRAY['ModeratorCorp', 'verification_service', 'ModeratorCorp']),
    unnest(ARRAY[
        '{"initial_stakeholders": 4, "initial_constraints": 4}'::jsonb,
        '{"verification_backend": "z3", "properties_proved": 11}'::jsonb,
        '{"stakeholder": "safety_board", "voting_power": 2.0}'::jsonb
    ])
FROM contract_ref;

-- Sample metrics
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'MedicalDiagnosisAI-Guardian' AND version = '2.1.0'
)
INSERT INTO metrics (contract_id, metric_type, metric_name, metric_value, unit, context)
SELECT 
    contract_id,
    unnest(ARRAY['performance', 'compliance', 'usage']),
    unnest(ARRAY['avg_diagnosis_accuracy', 'hipaa_compliance_score', 'daily_diagnoses']),
    unnest(ARRAY[0.94, 0.99, 1250.0]),
    unnest(ARRAY['percentage', 'percentage', 'count']),
    unnest(ARRAY[
        '{"measurement_period": "30_days", "sample_size": 5000}'::jsonb,
        '{"audit_date": "2025-01-15", "auditor": "compliance_team"}'::jsonb,
        '{"date": "2025-01-20", "peak_hour": "14:00"}'::jsonb
    ])
FROM contract_ref;

COMMIT;
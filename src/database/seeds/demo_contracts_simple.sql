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
INSERT INTO stakeholders (contract_id, name, weight, voting_power, ethereum_address)
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
    unnest(ARRAY[1.0, 0.9, 0.7, 0.6]),
    unnest(ARRAY[-1.0, -0.9, -0.5, -0.3])
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
    'weighted_average'
);

-- Medical AI stakeholders
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'MedicalDiagnosisAI-Guardian' AND version = '2.1.0'
)
INSERT INTO stakeholders (contract_id, name, weight, voting_power, ethereum_address)
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
    unnest(ARRAY[1.0, 0.8, 0.7, 1.0]),
    unnest(ARRAY[-1.0, -0.5, -0.2, -1.0])
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
    'weighted_average'
);

-- Trading AI stakeholders
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'TradingAI-RiskManager' AND version = '1.5.2'
)
INSERT INTO stakeholders (contract_id, name, weight, voting_power, ethereum_address)
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

-- Trading AI constraints
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'TradingAI-RiskManager' AND version = '1.5.2'
)
INSERT INTO constraints (contract_id, name, description, constraint_type, severity, violation_penalty)
SELECT 
    contract_id,
    unnest(ARRAY['risk_limits', 'market_hours', 'position_sizing', 'compliance_reporting']),
    unnest(ARRAY[
        'Trading must not exceed predefined risk limits',
        'Trading only allowed during market hours',
        'Position sizes must not exceed portfolio percentage limits',
        'All trades must generate compliance reports for audit'
    ]),
    unnest(ARRAY['invariant', 'requires', 'invariant', 'ensures']),
    unnest(ARRAY[1.0, 0.8, 0.9, 0.6]),
    unnest(ARRAY[-1.0, -0.5, -0.8, -0.3])
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
    gas_used,
    deployment_status
)
SELECT 
    contract_id,
    'ethereum_mainnet',
    '0xd456789012345678901234567890123456789012',
    '0xe56789012345678901234567890123456789012345678901234567890123456789',
    2500000,
    'deployed'
FROM contract_ref;

-- Sample amendment for governance demonstration
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'ContentModerator-SafeGuard' AND version = '1.0.0'
)
INSERT INTO amendments (
    contract_id,
    proposer,
    title,
    description,
    changes,
    status,
    required_consensus,
    expires_at
)
SELECT 
    contract_id,
    'safety_board',
    'Increase Hate Speech Detection Sensitivity',
    'Proposal to increase the sensitivity of hate speech detection algorithms to better protect vulnerable communities',
    '{"constraints": {"no_hate_speech": {"severity": 0.95, "violation_penalty": -0.95}}}'::jsonb,
    'active',
    0.66,
    NOW() + INTERVAL '7 days'
FROM contract_ref;

-- Sample votes on the amendment
WITH amendment_ref AS (
    SELECT id as amendment_id FROM amendments 
    WHERE title = 'Increase Hate Speech Detection Sensitivity'
)
INSERT INTO votes (amendment_id, stakeholder, support, weight, reasoning)
SELECT 
    amendment_id,
    unnest(ARRAY['safety_board', 'user_community', 'platform_operator']),
    unnest(ARRAY[true, true, false]),
    unnest(ARRAY[0.20, 0.30, 0.35]),
    unnest(ARRAY[
        'Essential for protecting vulnerable users',
        'Strongly support better hate speech protection',
        'Concerned about false positive impact on user experience'
    ])
FROM amendment_ref;

-- Sample audit logs
WITH contract_ref AS (
    SELECT id as contract_id FROM contracts 
    WHERE name = 'ContentModerator-SafeGuard' AND version = '1.0.0'
)
INSERT INTO audit_logs (
    event_type,
    entity_type,
    entity_id,
    user_id,
    event_data,
    jurisdiction,
    retention_until
)
SELECT 
    unnest(ARRAY['create', 'update', 'vote']),
    unnest(ARRAY['contract', 'constraint', 'amendment']),
    contract_id,
    unnest(ARRAY['ModeratorCorp', 'safety_board', 'user_community']),
    unnest(ARRAY[
        '{"action": "contract_created", "stakeholders": 4}'::jsonb,
        '{"action": "constraint_updated", "constraint": "no_hate_speech"}'::jsonb,
        '{"action": "vote_cast", "amendment": "hate_speech_sensitivity", "support": true}'::jsonb
    ]),
    'EU',
    NOW() + INTERVAL '7 years'
FROM contract_ref;

COMMIT;
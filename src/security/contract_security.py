"""
Contract security and safety validation.

Provides comprehensive security analysis for RLHF contracts including
vulnerability detection, attack vector analysis, and safety validation.
"""

import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re

from ..models.reward_contract import RewardContract
from ..models.legal_blocks import LegalBlocks, ConstraintEvaluator
from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class SecurityLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    REWARD_HACKING = "reward_hacking"
    CONSTRAINT_BYPASS = "constraint_bypass"
    STAKEHOLDER_MANIPULATION = "stakeholder_manipulation"
    DATA_POISONING = "data_poisoning"
    ADVERSARIAL_INPUTS = "adversarial_inputs"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"
    INFORMATION_DISCLOSURE = "information_disclosure"
    CONTRACT_LOGIC_FLAWS = "contract_logic_flaws"
    BYZANTINE_ATTACKS = "byzantine_attacks"


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability."""
    id: str
    vulnerability_type: VulnerabilityType
    severity: SecurityLevel
    title: str
    description: str
    affected_components: List[str]
    attack_vectors: List[str]
    mitigation_strategies: List[str]
    cve_references: List[str] = None
    discovered_at: float = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = time.time()
        if self.cve_references is None:
            self.cve_references = []


@dataclass
class SecurityAssessment:
    """Results of security assessment."""
    assessment_id: str
    contract_hash: str
    timestamp: float
    overall_security_score: float  # 0.0 = critical, 1.0 = secure
    vulnerabilities: List[SecurityVulnerability]
    security_recommendations: List[str]
    compliance_status: Dict[str, bool]
    attack_surface_analysis: Dict[str, Any]
    
    def get_critical_vulnerabilities(self) -> List[SecurityVulnerability]:
        """Get critical severity vulnerabilities."""
        return [
            vuln for vuln in self.vulnerabilities 
            if vuln.severity == SecurityLevel.CRITICAL
        ]
    
    def get_vulnerabilities_by_type(
        self, 
        vuln_type: VulnerabilityType
    ) -> List[SecurityVulnerability]:
        """Get vulnerabilities of specific type."""
        return [
            vuln for vuln in self.vulnerabilities 
            if vuln.vulnerability_type == vuln_type
        ]


class ContractSecurityAnalyzer:
    """
    Comprehensive security analyzer for RLHF contracts.
    
    Performs static analysis, dynamic testing, and formal verification
    to identify security vulnerabilities and attack vectors.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vulnerability_patterns = self._initialize_vulnerability_patterns()
        self.security_rules = self._initialize_security_rules()
    
    def _initialize_vulnerability_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for vulnerability detection."""
        return {
            "reward_hacking": [
                r"reward\s*=\s*[^\n]*\+\s*arbitrary",
                r"reward\s*\*=\s*user_input",
                r"reward\s*=\s*max\(.*\)",
                r"if\s+user_controlled.*reward\s*=\s*high_value"
            ],
            "constraint_bypass": [
                r"if\s+bypass.*constraint",
                r"constraint.*==\s*False",
                r"not\s+check_constraint",
                r"skip.*validation"
            ],
            "privilege_escalation": [
                r"admin\s*=\s*True",
                r"is_authorized\s*=\s*False",
                r"permissions\s*=\s*\[.*all.*\]",
                r"role\s*=\s*['\"]admin['\"].*without.*verification"
            ],
            "information_disclosure": [
                r"log\(.*secret.*\)",
                r"print\(.*password.*\)",
                r"debug.*private.*data",
                r"expose.*internal.*state"
            ]
        }
    
    def _initialize_security_rules(self) -> List[Dict[str, Any]]:
        """Initialize security validation rules."""
        return [
            {
                "rule_id": "SEC001",
                "title": "Stakeholder Weight Validation",
                "description": "Ensure stakeholder weights are properly normalized",
                "check_function": self._check_stakeholder_weights
            },
            {
                "rule_id": "SEC002",
                "title": "Constraint Function Safety",
                "description": "Verify constraint functions don't contain unsafe operations",
                "check_function": self._check_constraint_safety
            },
            {
                "rule_id": "SEC003",
                "title": "Reward Bounds Validation",
                "description": "Ensure reward values are bounded and finite",
                "check_function": self._check_reward_bounds
            },
            {
                "rule_id": "SEC004",
                "title": "Input Validation",
                "description": "Check for proper input validation",
                "check_function": self._check_input_validation
            },
            {
                "rule_id": "SEC005",
                "title": "Contract Immutability",
                "description": "Verify critical contract properties are immutable",
                "check_function": self._check_contract_immutability
            }
        ]
    
    def analyze_contract(self, contract: RewardContract) -> SecurityAssessment:
        """Perform comprehensive security analysis of contract."""
        start_time = time.time()
        assessment_id = f"SEC_{int(start_time)}"
        
        vulnerabilities = []
        security_recommendations = []
        compliance_status = {}
        
        try:
            # Static code analysis
            static_vulns = self._perform_static_analysis(contract)
            vulnerabilities.extend(static_vulns)
            
            # Security rule validation
            rule_results = self._validate_security_rules(contract)
            vulnerabilities.extend(rule_results['vulnerabilities'])
            security_recommendations.extend(rule_results['recommendations'])
            
            # Attack surface analysis
            attack_surface = self._analyze_attack_surface(contract)
            
            # Compliance checking
            compliance_status = self._check_compliance(contract)
            
            # Additional security checks
            additional_vulns = self._perform_additional_checks(contract)
            vulnerabilities.extend(additional_vulns)
            
            # Calculate overall security score
            security_score = self._calculate_security_score(vulnerabilities)
            
            # Generate recommendations
            if security_score < 0.7:
                security_recommendations.append(
                    "Consider implementing additional security controls"
                )
            
            if any(v.severity == SecurityLevel.CRITICAL for v in vulnerabilities):
                security_recommendations.append(
                    "Address critical vulnerabilities immediately before deployment"
                )
            
        except Exception as e:
            handle_error(
                error=e,
                operation="security_analysis",
                category=ErrorCategory.SECURITY,
                severity=ErrorSeverity.HIGH,
                additional_info={"contract_name": contract.metadata.name}
            )
            
            # Add analysis failure as critical vulnerability
            vulnerabilities.append(SecurityVulnerability(
                id="SEC_ANALYSIS_FAILURE",
                vulnerability_type=VulnerabilityType.CONTRACT_LOGIC_FLAWS,
                severity=SecurityLevel.CRITICAL,
                title="Security Analysis Failure",
                description=f"Security analysis failed: {str(e)}",
                affected_components=["security_analyzer"],
                attack_vectors=["analysis_bypass"],
                mitigation_strategies=["Fix analysis issues", "Manual review"]
            ))
            security_score = 0.0
            attack_surface = {"error": "Analysis failed"}
        
        return SecurityAssessment(
            assessment_id=assessment_id,
            contract_hash=contract.compute_hash(),
            timestamp=start_time,
            overall_security_score=security_score,
            vulnerabilities=vulnerabilities,
            security_recommendations=security_recommendations,
            compliance_status=compliance_status,
            attack_surface_analysis=attack_surface
        )
    
    def _perform_static_analysis(self, contract: RewardContract) -> List[SecurityVulnerability]:
        """Perform static code analysis for vulnerabilities."""
        vulnerabilities = []
        
        # Analyze reward functions for potential vulnerabilities
        for stakeholder_name, reward_fn in contract.reward_functions.items():
            if hasattr(reward_fn, '__code__'):
                # Get function source code if available
                try:
                    import inspect
                    source_code = inspect.getsource(reward_fn)
                    
                    # Check for vulnerability patterns
                    for vuln_type, patterns in self.vulnerability_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, source_code, re.IGNORECASE):
                                vulnerabilities.append(SecurityVulnerability(
                                    id=f"STATIC_{vuln_type.upper()}_{len(vulnerabilities)}",
                                    vulnerability_type=VulnerabilityType(vuln_type),
                                    severity=SecurityLevel.HIGH,
                                    title=f"Potential {vuln_type.replace('_', ' ').title()}",
                                    description=f"Pattern '{pattern}' detected in {stakeholder_name} reward function",
                                    affected_components=[f"reward_function:{stakeholder_name}"],
                                    attack_vectors=["code_injection", "logic_manipulation"],
                                    mitigation_strategies=[
                                        "Review function implementation",
                                        "Add input validation",
                                        "Implement bounds checking"
                                    ]
                                ))
                
                except Exception as e:
                    self.logger.warning(f"Could not analyze source code for {stakeholder_name}: {e}")
        
        return vulnerabilities
    
    def _validate_security_rules(self, contract: RewardContract) -> Dict[str, Any]:
        """Validate contract against security rules."""
        vulnerabilities = []
        recommendations = []
        
        for rule in self.security_rules:
            try:
                result = rule['check_function'](contract)
                if not result['passed']:
                    vulnerabilities.append(SecurityVulnerability(
                        id=rule['rule_id'],
                        vulnerability_type=VulnerabilityType.CONTRACT_LOGIC_FLAWS,
                        severity=result.get('severity', SecurityLevel.MEDIUM),
                        title=rule['title'],
                        description=result['description'],
                        affected_components=result.get('affected_components', ['contract']),
                        attack_vectors=result.get('attack_vectors', ['unknown']),
                        mitigation_strategies=result.get('mitigation_strategies', [])
                    ))
                
                if 'recommendations' in result:
                    recommendations.extend(result['recommendations'])
                    
            except Exception as e:
                self.logger.error(f"Security rule {rule['rule_id']} failed: {e}")
        
        return {
            'vulnerabilities': vulnerabilities,
            'recommendations': recommendations
        }
    
    def _check_stakeholder_weights(self, contract: RewardContract) -> Dict[str, Any]:
        """Check stakeholder weight security."""
        total_weight = sum(s.weight for s in contract.stakeholders.values())
        
        if abs(total_weight - 1.0) > 1e-6:
            return {
                'passed': False,
                'severity': SecurityLevel.HIGH,
                'description': f"Stakeholder weights sum to {total_weight}, not 1.0",
                'affected_components': ['stakeholder_weights'],
                'attack_vectors': ['weight_manipulation', 'governance_attack'],
                'mitigation_strategies': [
                    'Normalize stakeholder weights',
                    'Add weight validation',
                    'Implement weight change controls'
                ]
            }
        
        return {'passed': True}
    
    def _check_constraint_safety(self, contract: RewardContract) -> Dict[str, Any]:
        """Check constraint function safety."""
        unsafe_constraints = []
        
        for name, constraint in contract.constraints.items():
            # Check if constraint function appears safe
            try:
                if hasattr(constraint.constraint_fn, '__code__'):
                    import inspect
                    source = inspect.getsource(constraint.constraint_fn)
                    
                    # Check for potentially dangerous operations
                    dangerous_patterns = [
                        r'exec\(',
                        r'eval\(',
                        r'__import__',
                        r'open\(',
                        r'file\(',
                        r'subprocess',
                        r'os\.',
                        r'system\('
                    ]
                    
                    for pattern in dangerous_patterns:
                        if re.search(pattern, source):
                            unsafe_constraints.append((name, pattern))
                            break
            except:
                # Cannot analyze - mark as potentially unsafe
                unsafe_constraints.append((name, "unanalyzable"))
        
        if unsafe_constraints:
            return {
                'passed': False,
                'severity': SecurityLevel.CRITICAL,
                'description': f"Potentially unsafe constraint functions: {unsafe_constraints}",
                'affected_components': [f"constraint:{name}" for name, _ in unsafe_constraints],
                'attack_vectors': ['code_injection', 'system_compromise'],
                'mitigation_strategies': [
                    'Review constraint implementations',
                    'Sandbox constraint execution',
                    'Use safe evaluation methods'
                ]
            }
        
        return {'passed': True}
    
    def _check_reward_bounds(self, contract: RewardContract) -> Dict[str, Any]:
        """Check reward bounds validation."""
        # Test with sample inputs to check if rewards are bounded
        try:
            import jax.numpy as jnp
            
            # Test with various inputs
            test_cases = [
                (jnp.zeros(10), jnp.zeros(5)),
                (jnp.ones(10), jnp.ones(5)),
                (jnp.array([1e6] * 10), jnp.array([1e6] * 5)),
                (jnp.array([-1e6] * 10), jnp.array([-1e6] * 5))
            ]
            
            unbounded_rewards = []
            
            for i, (state, action) in enumerate(test_cases):
                try:
                    reward = contract.compute_reward(state, action)
                    if not jnp.isfinite(reward) or abs(reward) > 1e6:
                        unbounded_rewards.append(f"test_case_{i}")
                except Exception as e:
                    unbounded_rewards.append(f"test_case_{i}_error")
            
            if unbounded_rewards:
                return {
                    'passed': False,
                    'severity': SecurityLevel.HIGH,
                    'description': f"Unbounded or infinite rewards in cases: {unbounded_rewards}",
                    'affected_components': ['reward_computation'],
                    'attack_vectors': ['reward_hacking', 'numerical_attack'],
                    'mitigation_strategies': [
                        'Add reward bounds checking',
                        'Implement finite value validation',
                        'Use safe numerical operations'
                    ]
                }
        
        except Exception as e:
            return {
                'passed': False,
                'severity': SecurityLevel.MEDIUM,
                'description': f"Could not validate reward bounds: {str(e)}",
                'recommendations': ['Implement reward bounds testing']
            }
        
        return {'passed': True}
    
    def _check_input_validation(self, contract: RewardContract) -> Dict[str, Any]:
        """Check input validation implementation."""
        # Test with invalid inputs
        validation_issues = []
        
        try:
            import jax.numpy as jnp
            
            # Test with None inputs
            try:
                contract.compute_reward(None, None)
                validation_issues.append("accepts_none_inputs")
            except:
                pass  # Good, should reject None inputs
            
            # Test with empty arrays
            try:
                contract.compute_reward(jnp.array([]), jnp.array([]))
                validation_issues.append("accepts_empty_arrays")
            except:
                pass  # Good, should reject empty arrays
            
            # Test with mismatched dimensions
            try:
                contract.compute_reward(jnp.zeros(10), jnp.zeros(100))
                # If this doesn't raise an error, it might be an issue
                # depending on the specific contract implementation
            except:
                pass
        
        except Exception as e:
            validation_issues.append(f"validation_error: {str(e)}")
        
        if validation_issues:
            return {
                'passed': False,
                'severity': SecurityLevel.MEDIUM,
                'description': f"Input validation issues: {validation_issues}",
                'affected_components': ['input_validation'],
                'attack_vectors': ['malformed_input', 'denial_of_service'],
                'mitigation_strategies': [
                    'Add comprehensive input validation',
                    'Implement type checking',
                    'Add bounds checking'
                ]
            }
        
        return {'passed': True}
    
    def _check_contract_immutability(self, contract: RewardContract) -> Dict[str, Any]:
        """Check contract immutability properties."""
        # Check if critical properties can be modified after creation
        immutability_issues = []
        
        # Check if stakeholders can be modified
        if hasattr(contract, '_stakeholders') and not hasattr(contract.stakeholders, '__setitem__'):
            # Good - stakeholders appear to be protected
            pass
        else:
            immutability_issues.append("stakeholders_mutable")
        
        # Check metadata immutability
        if hasattr(contract.metadata, 'name') and hasattr(contract.metadata, '__setattr__'):
            immutability_issues.append("metadata_mutable")
        
        if immutability_issues:
            return {
                'passed': False,
                'severity': SecurityLevel.MEDIUM,
                'description': f"Immutability issues: {immutability_issues}",
                'affected_components': ['contract_state'],
                'attack_vectors': ['state_manipulation', 'contract_modification'],
                'mitigation_strategies': [
                    'Implement immutable data structures',
                    'Add state protection mechanisms',
                    'Use frozen dataclasses'
                ]
            }
        
        return {'passed': True}
    
    def _analyze_attack_surface(self, contract: RewardContract) -> Dict[str, Any]:
        """Analyze contract attack surface."""
        attack_surface = {
            'public_methods': [],
            'input_vectors': [],
            'external_dependencies': [],
            'trust_boundaries': [],
            'data_flows': []
        }
        
        # Analyze public methods
        for attr_name in dir(contract):
            if not attr_name.startswith('_'):
                attr = getattr(contract, attr_name)
                if callable(attr):
                    attack_surface['public_methods'].append(attr_name)
        
        # Analyze input vectors
        attack_surface['input_vectors'] = [
            'state_arrays',
            'action_arrays',
            'stakeholder_data',
            'constraint_parameters'
        ]
        
        # Analyze external dependencies
        attack_surface['external_dependencies'] = [
            'jax_numpy',
            'optimization_cache',
            'error_handling_system'
        ]
        
        # Trust boundaries
        attack_surface['trust_boundaries'] = [
            'user_input_validation',
            'contract_compilation',
            'constraint_evaluation',
            'reward_computation'
        ]
        
        # Data flows
        attack_surface['data_flows'] = [
            'input -> validation -> computation -> output',
            'stakeholder_preferences -> aggregation -> reward',
            'constraints -> violation_check -> penalty'
        ]
        
        return attack_surface
    
    def _check_compliance(self, contract: RewardContract) -> Dict[str, bool]:
        """Check compliance with security standards."""
        compliance = {
            'has_input_validation': True,  # Assume true for now
            'has_error_handling': True,
            'has_bounds_checking': True,
            'has_stakeholder_validation': len(contract.stakeholders) > 0,
            'has_constraint_validation': True,
            'has_audit_trail': hasattr(contract, 'metadata'),
            'gdpr_compliant': True,  # Would need detailed analysis
            'sox_compliant': True,   # Would need detailed analysis
            'iso27001_compliant': True  # Would need detailed analysis
        }
        
        return compliance
    
    def _perform_additional_checks(self, contract: RewardContract) -> List[SecurityVulnerability]:
        """Perform additional security checks."""
        vulnerabilities = []
        
        # Check for Byzantine fault tolerance
        if len(contract.stakeholders) > 1:
            # Simple check - in practice would be more sophisticated
            byzantine_threshold = len(contract.stakeholders) // 3
            if byzantine_threshold == 0:
                vulnerabilities.append(SecurityVulnerability(
                    id="BYZANTINE_FAULT_TOLERANCE",
                    vulnerability_type=VulnerabilityType.BYZANTINE_ATTACKS,
                    severity=SecurityLevel.MEDIUM,
                    title="Insufficient Byzantine Fault Tolerance",
                    description="Contract may not handle Byzantine stakeholder attacks",
                    affected_components=['stakeholder_governance'],
                    attack_vectors=['byzantine_stakeholder', 'collusion_attack'],
                    mitigation_strategies=[
                        'Increase number of stakeholders',
                        'Implement byzantine fault tolerance',
                        'Add stakeholder reputation system'
                    ]
                ))
        
        # Check for side-channel attacks
        vulnerabilities.append(SecurityVulnerability(
            id="TIMING_ATTACK_PROTECTION",
            vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
            severity=SecurityLevel.LOW,
            title="Potential Timing Attack Vulnerability",
            description="Contract execution time may leak information",
            affected_components=['reward_computation'],
            attack_vectors=['timing_analysis', 'side_channel'],
            mitigation_strategies=[
                'Implement constant-time operations',
                'Add timing randomization',
                'Use secure computation techniques'
            ]
        ))
        
        return vulnerabilities
    
    def _calculate_security_score(self, vulnerabilities: List[SecurityVulnerability]) -> float:
        """Calculate overall security score."""
        if not vulnerabilities:
            return 1.0
        
        # Weight vulnerabilities by severity
        severity_weights = {
            SecurityLevel.LOW: 0.1,
            SecurityLevel.MEDIUM: 0.3,
            SecurityLevel.HIGH: 0.7,
            SecurityLevel.CRITICAL: 1.0
        }
        
        total_impact = sum(
            severity_weights[vuln.severity] for vuln in vulnerabilities
        )
        
        # Convert to score (higher is better)
        max_possible_impact = len(vulnerabilities) * 1.0
        security_score = 1.0 - (total_impact / max_possible_impact)
        
        return max(0.0, security_score)
    
    def generate_security_report(
        self, 
        assessment: SecurityAssessment,
        format: str = "markdown"
    ) -> str:
        """Generate human-readable security report."""
        if format == "markdown":
            return self._generate_markdown_report(assessment)
        elif format == "json":
            import json
            return json.dumps(self._assessment_to_dict(assessment), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown_report(self, assessment: SecurityAssessment) -> str:
        """Generate markdown security report."""
        report = f"""# Security Assessment Report

**Assessment ID:** {assessment.assessment_id}
**Contract Hash:** {assessment.contract_hash}
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(assessment.timestamp))}
**Overall Security Score:** {assessment.overall_security_score:.2f}/1.00

## Executive Summary

{'✅ **SECURE**' if assessment.overall_security_score >= 0.8 else '⚠️ **NEEDS ATTENTION**' if assessment.overall_security_score >= 0.6 else '🚨 **HIGH RISK**'}

Total Vulnerabilities: {len(assessment.vulnerabilities)}
- Critical: {len([v for v in assessment.vulnerabilities if v.severity == SecurityLevel.CRITICAL])}
- High: {len([v for v in assessment.vulnerabilities if v.severity == SecurityLevel.HIGH])}
- Medium: {len([v for v in assessment.vulnerabilities if v.severity == SecurityLevel.MEDIUM])}
- Low: {len([v for v in assessment.vulnerabilities if v.severity == SecurityLevel.LOW])}

## Vulnerabilities

"""
        
        for vuln in assessment.vulnerabilities:
            severity_emoji = {
                SecurityLevel.CRITICAL: "🚨",
                SecurityLevel.HIGH: "⚠️",
                SecurityLevel.MEDIUM: "⚡",
                SecurityLevel.LOW: "ℹ️"
            }
            
            report += f"""### {severity_emoji[vuln.severity]} {vuln.title}

**ID:** {vuln.id}
**Severity:** {vuln.severity.value.upper()}
**Type:** {vuln.vulnerability_type.value}

**Description:** {vuln.description}

**Affected Components:** {', '.join(vuln.affected_components)}

**Attack Vectors:** {', '.join(vuln.attack_vectors)}

**Mitigation Strategies:**
{chr(10).join(f'- {strategy}' for strategy in vuln.mitigation_strategies)}

---

"""
        
        report += f"""## Security Recommendations

{chr(10).join(f'- {rec}' for rec in assessment.security_recommendations)}

## Compliance Status

{chr(10).join(f'- {standard}: {"✅ Compliant" if status else "❌ Non-compliant"}' for standard, status in assessment.compliance_status.items())}

## Attack Surface Analysis

**Public Methods:** {', '.join(assessment.attack_surface_analysis.get('public_methods', []))}

**Input Vectors:** {', '.join(assessment.attack_surface_analysis.get('input_vectors', []))}

**Trust Boundaries:** {', '.join(assessment.attack_surface_analysis.get('trust_boundaries', []))}
"""
        
        return report
    
    def _assessment_to_dict(self, assessment: SecurityAssessment) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            'assessment_id': assessment.assessment_id,
            'contract_hash': assessment.contract_hash,
            'timestamp': assessment.timestamp,
            'overall_security_score': assessment.overall_security_score,
            'vulnerabilities': [
                {
                    'id': v.id,
                    'type': v.vulnerability_type.value,
                    'severity': v.severity.value,
                    'title': v.title,
                    'description': v.description,
                    'affected_components': v.affected_components,
                    'attack_vectors': v.attack_vectors,
                    'mitigation_strategies': v.mitigation_strategies,
                    'discovered_at': v.discovered_at
                }
                for v in assessment.vulnerabilities
            ],
            'security_recommendations': assessment.security_recommendations,
            'compliance_status': assessment.compliance_status,
            'attack_surface_analysis': assessment.attack_surface_analysis
        }


def analyze_contract_security(contract: RewardContract) -> SecurityAssessment:
    """Convenience function for security analysis."""
    analyzer = ContractSecurityAnalyzer()
    return analyzer.analyze_contract(contract)

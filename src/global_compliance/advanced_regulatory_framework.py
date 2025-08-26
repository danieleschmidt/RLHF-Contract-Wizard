"""
Advanced Global Regulatory Framework for RLHF-Contract-Wizard.

This module implements a comprehensive regulatory compliance system that automatically
adapts to different global jurisdictions, ensuring legal compliance for AI reward
contracts across all major markets.

Key Features:
1. Multi-jurisdictional compliance engine
2. Automated regulatory adaptation
3. Real-time compliance monitoring
4. Cross-border contract harmonization
5. Regulatory change detection and adaptation
6. Compliance reporting and audit trails
"""

import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import re

from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class Jurisdiction(Enum):
    """Supported legal jurisdictions."""
    UNITED_STATES = "US"
    EUROPEAN_UNION = "EU"
    UNITED_KINGDOM = "UK"
    CANADA = "CA"
    AUSTRALIA = "AU"
    JAPAN = "JP"
    SOUTH_KOREA = "KR"
    SINGAPORE = "SG"
    SWITZERLAND = "CH"
    CALIFORNIA = "US-CA"  # State-specific
    GDPR_ZONE = "GDPR"    # Regional
    GLOBAL = "GLOBAL"     # Universal compliance


class ComplianceLevel(Enum):
    """Compliance requirement levels."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    PROHIBITED = "prohibited"


class RegulatoryFramework(Enum):
    """Regulatory frameworks."""
    AI_ACT_EU = "ai_act_eu"
    GDPR = "gdpr"
    CCPA = "ccpa"
    PIPEDA = "pipeda"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST_AI_RMF = "nist_ai_rmf"
    IEEE_2857 = "ieee_2857"


@dataclass
class RegulatoryRequirement:
    """Individual regulatory requirement."""
    requirement_id: str
    name: str
    description: str
    jurisdiction: Jurisdiction
    framework: RegulatoryFramework
    compliance_level: ComplianceLevel
    implementation_deadline: Optional[datetime] = None
    penalties: Dict[str, Any] = field(default_factory=dict)
    verification_method: str = "automated"
    documentation_required: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ComplianceStatus:
    """Compliance status for a requirement."""
    requirement_id: str
    is_compliant: bool
    compliance_score: float  # 0.0 to 1.0
    last_check: datetime
    next_check: Optional[datetime] = None
    remediation_actions: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "medium"


class GlobalRegulatoryEngine:
    """Engine for managing global regulatory compliance."""
    
    def __init__(self):
        self.requirements_database = {}
        self.compliance_status = {}
        self.jurisdiction_mappings = {}
        self.framework_implementations = {}
        
        # Initialize regulatory frameworks
        self._initialize_regulatory_frameworks()
        self._initialize_jurisdiction_mappings()
        self._load_regulatory_requirements()
    
    def _initialize_regulatory_frameworks(self):
        """Initialize implementation for various regulatory frameworks."""
        
        # EU AI Act implementation
        self.framework_implementations[RegulatoryFramework.AI_ACT_EU] = {
            'high_risk_ai_systems': [
                'reward_modeling_for_critical_decisions',
                'automated_decision_making',
                'bias_assessment_systems'
            ],
            'prohibited_practices': [
                'subliminal_manipulation',
                'exploitation_of_vulnerabilities',
                'social_scoring_by_governments'
            ],
            'transparency_requirements': [
                'automated_decision_disclosure',
                'ai_system_registration',
                'conformity_assessment'
            ],
            'technical_requirements': {
                'risk_management_system': True,
                'data_governance': True,
                'record_keeping': True,
                'transparency_obligations': True,
                'human_oversight': True,
                'accuracy_robustness': True
            }
        }
        
        # GDPR implementation
        self.framework_implementations[RegulatoryFramework.GDPR] = {
            'data_protection_principles': [
                'lawfulness_fairness_transparency',
                'purpose_limitation',
                'data_minimisation',
                'accuracy',
                'storage_limitation',
                'integrity_confidentiality'
            ],
            'individual_rights': [
                'right_to_information',
                'right_of_access',
                'right_to_rectification',
                'right_to_erasure',
                'right_to_restrict_processing',
                'right_to_data_portability',
                'right_to_object',
                'rights_related_to_automated_decision_making'
            ],
            'technical_requirements': {
                'privacy_by_design': True,
                'data_protection_impact_assessment': True,
                'breach_notification': True,
                'consent_management': True,
                'data_processor_agreements': True
            }
        }
        
        # NIST AI Risk Management Framework
        self.framework_implementations[RegulatoryFramework.NIST_AI_RMF] = {
            'core_functions': ['govern', 'map', 'measure', 'manage'],
            'risk_categories': [
                'harmful_bias_discrimination',
                'dangerous_malicious_deceptive_content',
                'information_security',
                'privacy'
            ],
            'trustworthiness_characteristics': [
                'valid_reliable',
                'safe',
                'fair_non_discriminatory', 
                'explainable_interpretable',
                'privacy_enhanced',
                'secure_resilient'
            ]
        }
    
    def _initialize_jurisdiction_mappings(self):
        """Initialize jurisdiction-specific mappings."""
        
        # Map jurisdictions to applicable frameworks
        self.jurisdiction_mappings = {
            Jurisdiction.EUROPEAN_UNION: [
                RegulatoryFramework.AI_ACT_EU,
                RegulatoryFramework.GDPR,
                RegulatoryFramework.ISO27001
            ],
            Jurisdiction.UNITED_STATES: [
                RegulatoryFramework.NIST_AI_RMF,
                RegulatoryFramework.SOX,
                RegulatoryFramework.HIPAA
            ],
            Jurisdiction.CALIFORNIA: [
                RegulatoryFramework.CCPA,
                RegulatoryFramework.NIST_AI_RMF
            ],
            Jurisdiction.CANADA: [
                RegulatoryFramework.PIPEDA,
                RegulatoryFramework.ISO27001
            ],
            Jurisdiction.UNITED_KINGDOM: [
                RegulatoryFramework.GDPR,  # UK-GDPR
                RegulatoryFramework.ISO27001
            ],
            Jurisdiction.JAPAN: [
                RegulatoryFramework.ISO27001,
                RegulatoryFramework.NIST_AI_RMF
            ],
            Jurisdiction.SINGAPORE: [
                RegulatoryFramework.ISO27001,
                RegulatoryFramework.NIST_AI_RMF
            ]
        }
    
    def _load_regulatory_requirements(self):
        """Load detailed regulatory requirements."""
        
        # EU AI Act Requirements
        ai_act_requirements = [
            RegulatoryRequirement(
                requirement_id="AI_ACT_001",
                name="High-Risk AI System Registration",
                description="Register high-risk AI systems with competent authorities",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                framework=RegulatoryFramework.AI_ACT_EU,
                compliance_level=ComplianceLevel.MANDATORY,
                implementation_deadline=datetime(2026, 8, 2, tzinfo=timezone.utc),
                penalties={'max_fine': '30M EUR or 6% global turnover'},
                documentation_required=[
                    'system_documentation',
                    'risk_assessment',
                    'conformity_declaration'
                ]
            ),
            RegulatoryRequirement(
                requirement_id="AI_ACT_002",
                name="Algorithmic Transparency",
                description="Provide clear information about AI decision-making processes",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                framework=RegulatoryFramework.AI_ACT_EU,
                compliance_level=ComplianceLevel.MANDATORY,
                implementation_deadline=datetime(2025, 8, 2, tzinfo=timezone.utc),
                penalties={'max_fine': '15M EUR or 3% global turnover'},
                documentation_required=[
                    'transparency_report',
                    'algorithm_explanation',
                    'user_documentation'
                ]
            ),
            RegulatoryRequirement(
                requirement_id="AI_ACT_003",
                name="Human Oversight Requirements",
                description="Ensure meaningful human oversight of AI systems",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                framework=RegulatoryFramework.AI_ACT_EU,
                compliance_level=ComplianceLevel.MANDATORY,
                penalties={'max_fine': '15M EUR or 3% global turnover'},
                documentation_required=[
                    'oversight_procedures',
                    'human_review_protocols',
                    'escalation_procedures'
                ]
            )
        ]
        
        # GDPR Requirements
        gdpr_requirements = [
            RegulatoryRequirement(
                requirement_id="GDPR_001",
                name="Data Protection by Design and Default",
                description="Implement privacy protection measures from system design",
                jurisdiction=Jurisdiction.GDPR_ZONE,
                framework=RegulatoryFramework.GDPR,
                compliance_level=ComplianceLevel.MANDATORY,
                penalties={'max_fine': '20M EUR or 4% global turnover'},
                documentation_required=[
                    'privacy_impact_assessment',
                    'data_protection_measures',
                    'privacy_policy'
                ]
            ),
            RegulatoryRequirement(
                requirement_id="GDPR_002",
                name="Consent Management",
                description="Obtain and manage valid consent for data processing",
                jurisdiction=Jurisdiction.GDPR_ZONE,
                framework=RegulatoryFramework.GDPR,
                compliance_level=ComplianceLevel.MANDATORY,
                penalties={'max_fine': '20M EUR or 4% global turnover'},
                documentation_required=[
                    'consent_records',
                    'withdrawal_mechanisms',
                    'legal_basis_documentation'
                ]
            ),
            RegulatoryRequirement(
                requirement_id="GDPR_003",
                name="Data Subject Rights Implementation",
                description="Implement mechanisms for data subject rights",
                jurisdiction=Jurisdiction.GDPR_ZONE,
                framework=RegulatoryFramework.GDPR,
                compliance_level=ComplianceLevel.MANDATORY,
                penalties={'max_fine': '20M EUR or 4% global turnover'},
                documentation_required=[
                    'rights_fulfillment_procedures',
                    'identity_verification_process',
                    'response_templates'
                ]
            )
        ]
        
        # NIST AI RMF Requirements
        nist_requirements = [
            RegulatoryRequirement(
                requirement_id="NIST_001",
                name="AI Risk Management System",
                description="Implement comprehensive AI risk management",
                jurisdiction=Jurisdiction.UNITED_STATES,
                framework=RegulatoryFramework.NIST_AI_RMF,
                compliance_level=ComplianceLevel.RECOMMENDED,
                documentation_required=[
                    'risk_management_plan',
                    'risk_assessment_results',
                    'mitigation_strategies'
                ]
            ),
            RegulatoryRequirement(
                requirement_id="NIST_002",
                name="Bias Assessment and Mitigation",
                description="Regular assessment and mitigation of algorithmic bias",
                jurisdiction=Jurisdiction.UNITED_STATES,
                framework=RegulatoryFramework.NIST_AI_RMF,
                compliance_level=ComplianceLevel.RECOMMENDED,
                documentation_required=[
                    'bias_assessment_report',
                    'fairness_metrics',
                    'mitigation_measures'
                ]
            )
        ]
        
        # Store all requirements
        all_requirements = ai_act_requirements + gdpr_requirements + nist_requirements
        
        for req in all_requirements:
            self.requirements_database[req.requirement_id] = req
    
    def get_applicable_requirements(self, jurisdiction: Jurisdiction,
                                   business_context: Dict[str, Any] = None) -> List[RegulatoryRequirement]:
        """Get applicable regulatory requirements for jurisdiction."""
        
        if business_context is None:
            business_context = {}
        
        applicable_requirements = []
        
        # Get frameworks for jurisdiction
        frameworks = self.jurisdiction_mappings.get(jurisdiction, [])
        
        # Special handling for GDPR zone
        if jurisdiction in [Jurisdiction.EUROPEAN_UNION, Jurisdiction.UNITED_KINGDOM]:
            frameworks.append(RegulatoryFramework.GDPR)
        
        # Filter requirements by jurisdiction and frameworks
        for req_id, requirement in self.requirements_database.items():
            if (requirement.jurisdiction == jurisdiction or 
                requirement.jurisdiction == Jurisdiction.GDPR_ZONE or
                requirement.framework in frameworks):
                
                # Additional context-based filtering
                if self._requirement_applies(requirement, business_context):
                    applicable_requirements.append(requirement)
        
        return applicable_requirements
    
    def _requirement_applies(self, requirement: RegulatoryRequirement, 
                           business_context: Dict[str, Any]) -> bool:
        """Check if requirement applies to specific business context."""
        
        # Check business sector
        sector = business_context.get('business_sector', 'general')
        
        # HIPAA only applies to healthcare
        if requirement.framework == RegulatoryFramework.HIPAA:
            return sector in ['healthcare', 'medical', 'pharmaceutical']
        
        # PCI DSS only applies to payment processing
        if requirement.framework == RegulatoryFramework.PCI_DSS:
            return business_context.get('processes_payments', False)
        
        # Check AI system type
        ai_system_type = business_context.get('ai_system_type', 'general_ai')
        
        # High-risk AI Act requirements
        if requirement.requirement_id == "AI_ACT_001":
            high_risk_systems = [
                'reward_modeling',
                'automated_decision_making',
                'biometric_identification',
                'critical_infrastructure',
                'recruitment'
            ]
            return ai_system_type in high_risk_systems
        
        # Default: requirement applies
        return True
    
    def assess_compliance(self, jurisdiction: Jurisdiction,
                         business_context: Dict[str, Any] = None,
                         current_implementation: Dict[str, Any] = None) -> Dict[str, ComplianceStatus]:
        """Assess compliance status against applicable requirements."""
        
        if business_context is None:
            business_context = {}
        if current_implementation is None:
            current_implementation = {}
        
        applicable_requirements = self.get_applicable_requirements(jurisdiction, business_context)
        compliance_results = {}
        
        for requirement in applicable_requirements:
            status = self._assess_single_requirement(requirement, current_implementation)
            compliance_results[requirement.requirement_id] = status
            self.compliance_status[requirement.requirement_id] = status
        
        return compliance_results
    
    def _assess_single_requirement(self, requirement: RegulatoryRequirement,
                                  current_implementation: Dict[str, Any]) -> ComplianceStatus:
        """Assess compliance for a single requirement."""
        
        compliance_score = 0.0
        is_compliant = False
        remediation_actions = []
        evidence = {}
        risk_level = "high"
        
        try:
            # AI Act compliance assessment
            if requirement.framework == RegulatoryFramework.AI_ACT_EU:
                compliance_score, remediation_actions, evidence = self._assess_ai_act_compliance(
                    requirement, current_implementation
                )
            
            # GDPR compliance assessment
            elif requirement.framework == RegulatoryFramework.GDPR:
                compliance_score, remediation_actions, evidence = self._assess_gdpr_compliance(
                    requirement, current_implementation
                )
            
            # NIST AI RMF assessment
            elif requirement.framework == RegulatoryFramework.NIST_AI_RMF:
                compliance_score, remediation_actions, evidence = self._assess_nist_compliance(
                    requirement, current_implementation
                )
            
            # Default assessment
            else:
                compliance_score, remediation_actions, evidence = self._assess_default_compliance(
                    requirement, current_implementation
                )
            
            # Determine compliance status
            is_compliant = compliance_score >= 0.8  # 80% threshold
            
            # Determine risk level
            if compliance_score >= 0.8:
                risk_level = "low"
            elif compliance_score >= 0.6:
                risk_level = "medium" 
            else:
                risk_level = "high"
                
        except Exception as e:
            handle_error(
                error=e,
                operation=f"assess_compliance:{requirement.requirement_id}",
                category=ErrorCategory.COMPLIANCE,
                severity=ErrorSeverity.HIGH,
                additional_info={'requirement_id': requirement.requirement_id}
            )
            
            # Fail-safe: mark as non-compliant
            compliance_score = 0.0
            is_compliant = False
            remediation_actions = ["Manual compliance review required due to assessment error"]
            risk_level = "critical"
        
        return ComplianceStatus(
            requirement_id=requirement.requirement_id,
            is_compliant=is_compliant,
            compliance_score=compliance_score,
            last_check=datetime.now(timezone.utc),
            next_check=datetime.now(timezone.utc).replace(hour=23, minute=59, second=59),
            remediation_actions=remediation_actions,
            evidence=evidence,
            risk_level=risk_level
        )
    
    def _assess_ai_act_compliance(self, requirement: RegulatoryRequirement,
                                 implementation: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        """Assess EU AI Act compliance."""
        
        score = 0.0
        actions = []
        evidence = {}
        
        if requirement.requirement_id == "AI_ACT_001":  # Registration
            # Check registration status
            if implementation.get('system_registered', False):
                score += 0.4
                evidence['registration_status'] = 'registered'
            else:
                actions.append("Register AI system with competent authority")
            
            # Check documentation completeness
            required_docs = requirement.documentation_required
            available_docs = implementation.get('available_documentation', [])
            
            doc_completeness = len(set(available_docs) & set(required_docs)) / len(required_docs)
            score += 0.4 * doc_completeness
            evidence['documentation_completeness'] = doc_completeness
            
            if doc_completeness < 1.0:
                missing_docs = set(required_docs) - set(available_docs)
                actions.extend([f"Prepare {doc}" for doc in missing_docs])
            
            # Check conformity assessment
            if implementation.get('conformity_assessment_completed', False):
                score += 0.2
                evidence['conformity_assessment'] = 'completed'
            else:
                actions.append("Complete conformity assessment")
        
        elif requirement.requirement_id == "AI_ACT_002":  # Transparency
            # Check transparency documentation
            if implementation.get('transparency_report_available', False):
                score += 0.3
                evidence['transparency_report'] = 'available'
            else:
                actions.append("Create comprehensive transparency report")
            
            # Check algorithm explainability
            explainability_score = implementation.get('explainability_score', 0.0)
            score += 0.4 * explainability_score
            evidence['explainability_score'] = explainability_score
            
            if explainability_score < 0.8:
                actions.append("Improve algorithm explainability mechanisms")
            
            # Check user documentation
            if implementation.get('user_documentation_complete', False):
                score += 0.3
                evidence['user_documentation'] = 'complete'
            else:
                actions.append("Complete user-facing documentation")
        
        elif requirement.requirement_id == "AI_ACT_003":  # Human Oversight
            # Check oversight procedures
            oversight_procedures = implementation.get('oversight_procedures_defined', False)
            if oversight_procedures:
                score += 0.4
                evidence['oversight_procedures'] = 'defined'
            else:
                actions.append("Define human oversight procedures")
            
            # Check human review protocols
            review_protocols = implementation.get('human_review_protocols', False)
            if review_protocols:
                score += 0.3
                evidence['review_protocols'] = 'implemented'
            else:
                actions.append("Implement human review protocols")
            
            # Check escalation procedures
            escalation = implementation.get('escalation_procedures', False)
            if escalation:
                score += 0.3
                evidence['escalation'] = 'implemented'
            else:
                actions.append("Implement escalation procedures")
        
        return score, actions, evidence
    
    def _assess_gdpr_compliance(self, requirement: RegulatoryRequirement,
                               implementation: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        """Assess GDPR compliance."""
        
        score = 0.0
        actions = []
        evidence = {}
        
        if requirement.requirement_id == "GDPR_001":  # Privacy by Design
            # Check privacy impact assessment
            if implementation.get('privacy_impact_assessment_completed', False):
                score += 0.4
                evidence['privacy_impact_assessment'] = 'completed'
            else:
                actions.append("Complete Privacy Impact Assessment")
            
            # Check data protection measures
            protection_measures = implementation.get('data_protection_measures', [])
            expected_measures = ['encryption', 'access_control', 'audit_logging', 'data_minimization']
            
            measures_score = len(set(protection_measures) & set(expected_measures)) / len(expected_measures)
            score += 0.4 * measures_score
            evidence['protection_measures_score'] = measures_score
            
            if measures_score < 1.0:
                missing_measures = set(expected_measures) - set(protection_measures)
                actions.extend([f"Implement {measure}" for measure in missing_measures])
            
            # Check privacy policy
            if implementation.get('privacy_policy_available', False):
                score += 0.2
                evidence['privacy_policy'] = 'available'
            else:
                actions.append("Create comprehensive privacy policy")
        
        elif requirement.requirement_id == "GDPR_002":  # Consent Management
            # Check consent mechanisms
            consent_mechanism = implementation.get('consent_mechanism_implemented', False)
            if consent_mechanism:
                score += 0.4
                evidence['consent_mechanism'] = 'implemented'
            else:
                actions.append("Implement consent management system")
            
            # Check withdrawal mechanisms
            withdrawal = implementation.get('consent_withdrawal_available', False)
            if withdrawal:
                score += 0.3
                evidence['withdrawal_mechanism'] = 'available'
            else:
                actions.append("Implement consent withdrawal mechanism")
            
            # Check legal basis documentation
            legal_basis = implementation.get('legal_basis_documented', False)
            if legal_basis:
                score += 0.3
                evidence['legal_basis'] = 'documented'
            else:
                actions.append("Document legal basis for data processing")
        
        elif requirement.requirement_id == "GDPR_003":  # Data Subject Rights
            # Check rights fulfillment procedures
            rights_procedures = implementation.get('data_subject_rights_procedures', False)
            if rights_procedures:
                score += 0.4
                evidence['rights_procedures'] = 'implemented'
            else:
                actions.append("Implement data subject rights fulfillment procedures")
            
            # Check identity verification
            identity_verification = implementation.get('identity_verification_process', False)
            if identity_verification:
                score += 0.3
                evidence['identity_verification'] = 'implemented'
            else:
                actions.append("Implement identity verification process")
            
            # Check response mechanisms
            response_mechanisms = implementation.get('automated_response_system', False)
            if response_mechanisms:
                score += 0.3
                evidence['response_mechanisms'] = 'implemented'
            else:
                actions.append("Implement automated response system for rights requests")
        
        return score, actions, evidence
    
    def _assess_nist_compliance(self, requirement: RegulatoryRequirement,
                               implementation: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        """Assess NIST AI RMF compliance."""
        
        score = 0.0
        actions = []
        evidence = {}
        
        if requirement.requirement_id == "NIST_001":  # Risk Management
            # Check risk management plan
            risk_plan = implementation.get('risk_management_plan_exists', False)
            if risk_plan:
                score += 0.4
                evidence['risk_management_plan'] = 'exists'
            else:
                actions.append("Develop comprehensive AI risk management plan")
            
            # Check risk assessment
            risk_assessment = implementation.get('risk_assessment_completed', False)
            if risk_assessment:
                score += 0.3
                evidence['risk_assessment'] = 'completed'
            else:
                actions.append("Complete AI risk assessment")
            
            # Check mitigation strategies
            mitigation = implementation.get('mitigation_strategies_implemented', False)
            if mitigation:
                score += 0.3
                evidence['mitigation_strategies'] = 'implemented'
            else:
                actions.append("Implement risk mitigation strategies")
        
        elif requirement.requirement_id == "NIST_002":  # Bias Assessment
            # Check bias assessment
            bias_assessment = implementation.get('bias_assessment_completed', False)
            if bias_assessment:
                score += 0.4
                evidence['bias_assessment'] = 'completed'
            else:
                actions.append("Complete algorithmic bias assessment")
            
            # Check fairness metrics
            fairness_metrics = implementation.get('fairness_metrics_implemented', False)
            if fairness_metrics:
                score += 0.3
                evidence['fairness_metrics'] = 'implemented'
            else:
                actions.append("Implement fairness metrics monitoring")
            
            # Check mitigation measures
            bias_mitigation = implementation.get('bias_mitigation_active', False)
            if bias_mitigation:
                score += 0.3
                evidence['bias_mitigation'] = 'active'
            else:
                actions.append("Implement bias mitigation measures")
        
        return score, actions, evidence
    
    def _assess_default_compliance(self, requirement: RegulatoryRequirement,
                                  implementation: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        """Default compliance assessment for other frameworks."""
        
        # Basic assessment based on documentation and implementation
        score = 0.5  # Default partial compliance
        actions = ["Review specific framework requirements"]
        evidence = {'assessment_type': 'default'}
        
        # Check if basic documentation exists
        if implementation.get('basic_documentation_available', False):
            score += 0.3
            evidence['documentation'] = 'available'
        
        # Check if implementation is active
        if implementation.get('implementation_active', False):
            score += 0.2
            evidence['implementation'] = 'active'
        
        return score, actions, evidence
    
    def generate_compliance_report(self, jurisdiction: Jurisdiction,
                                  business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        if business_context is None:
            business_context = {}
        
        # Assess current compliance
        compliance_results = self.assess_compliance(jurisdiction, business_context)
        
        # Calculate overall metrics
        total_requirements = len(compliance_results)
        compliant_requirements = sum(1 for status in compliance_results.values() if status.is_compliant)
        overall_score = sum(status.compliance_score for status in compliance_results.values()) / max(1, total_requirements)
        
        # Group by compliance level
        critical_issues = []
        high_priority_actions = []
        medium_priority_actions = []
        
        for req_id, status in compliance_results.items():
            requirement = self.requirements_database[req_id]
            
            if status.risk_level == "critical":
                critical_issues.append({
                    'requirement_id': req_id,
                    'name': requirement.name,
                    'compliance_score': status.compliance_score,
                    'remediation_actions': status.remediation_actions
                })
            elif status.risk_level == "high":
                high_priority_actions.extend(status.remediation_actions)
            elif status.risk_level == "medium":
                medium_priority_actions.extend(status.remediation_actions)
        
        # Framework-specific summaries
        framework_summary = {}
        for req_id, status in compliance_results.values():
            requirement = self.requirements_database[req_id]
            framework = requirement.framework.value
            
            if framework not in framework_summary:
                framework_summary[framework] = {
                    'total_requirements': 0,
                    'compliant_requirements': 0,
                    'average_score': 0.0
                }
            
            framework_summary[framework]['total_requirements'] += 1
            if status.is_compliant:
                framework_summary[framework]['compliant_requirements'] += 1
            framework_summary[framework]['average_score'] += status.compliance_score
        
        # Calculate averages
        for framework_data in framework_summary.values():
            if framework_data['total_requirements'] > 0:
                framework_data['average_score'] /= framework_data['total_requirements']
        
        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(
            compliance_results, jurisdiction, business_context
        )
        
        report = {
            'report_timestamp': datetime.now(timezone.utc).isoformat(),
            'jurisdiction': jurisdiction.value,
            'business_context': business_context,
            'executive_summary': {
                'overall_compliance_score': overall_score,
                'total_requirements': total_requirements,
                'compliant_requirements': compliant_requirements,
                'compliance_rate': compliant_requirements / max(1, total_requirements),
                'critical_issues_count': len(critical_issues)
            },
            'detailed_results': {
                req_id: {
                    'requirement_name': self.requirements_database[req_id].name,
                    'is_compliant': status.is_compliant,
                    'compliance_score': status.compliance_score,
                    'risk_level': status.risk_level,
                    'remediation_actions': status.remediation_actions,
                    'evidence': status.evidence,
                    'last_check': status.last_check.isoformat()
                }
                for req_id, status in compliance_results.items()
            },
            'framework_summary': framework_summary,
            'critical_issues': critical_issues,
            'priority_actions': {
                'high_priority': high_priority_actions[:10],  # Top 10
                'medium_priority': medium_priority_actions[:10]
            },
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps(compliance_results),
            'regulatory_calendar': self._get_upcoming_deadlines(jurisdiction)
        }
        
        return report
    
    def _generate_compliance_recommendations(self, compliance_results: Dict[str, ComplianceStatus],
                                           jurisdiction: Jurisdiction,
                                           business_context: Dict[str, Any]) -> List[str]:
        """Generate strategic compliance recommendations."""
        
        recommendations = []
        
        # Analyze compliance gaps
        low_scores = [req_id for req_id, status in compliance_results.items() if status.compliance_score < 0.6]
        
        if len(low_scores) > len(compliance_results) * 0.5:
            recommendations.append(
                "Consider engaging external compliance consultants for comprehensive review"
            )
        
        # Framework-specific recommendations
        eu_requirements = [req_id for req_id in compliance_results.keys() 
                          if self.requirements_database[req_id].jurisdiction == Jurisdiction.EUROPEAN_UNION]
        
        if eu_requirements and jurisdiction == Jurisdiction.EUROPEAN_UNION:
            eu_compliance = sum(compliance_results[req_id].compliance_score for req_id in eu_requirements) / len(eu_requirements)
            if eu_compliance < 0.8:
                recommendations.append(
                    "Prioritize EU AI Act compliance - significant penalties apply from August 2026"
                )
        
        # Business context recommendations
        if business_context.get('ai_system_type') == 'reward_modeling':
            recommendations.append(
                "Implement comprehensive bias testing and fairness metrics for reward models"
            )
        
        if business_context.get('processes_personal_data', False):
            recommendations.append(
                "Strengthen data protection measures and privacy by design implementation"
            )
        
        # Generic recommendations
        recommendations.extend([
            "Establish regular compliance monitoring and review cycles",
            "Implement automated compliance checking where possible",
            "Create comprehensive staff training program on regulatory requirements",
            "Develop incident response procedures for compliance violations"
        ])
        
        return recommendations
    
    def _generate_next_steps(self, compliance_results: Dict[str, ComplianceStatus]) -> List[Dict[str, Any]]:
        """Generate prioritized next steps."""
        
        next_steps = []
        
        # Critical issues first
        for req_id, status in compliance_results.items():
            if status.risk_level == "critical":
                requirement = self.requirements_database[req_id]
                
                next_steps.append({
                    'priority': 1,
                    'action': f"Address critical compliance gap: {requirement.name}",
                    'timeline': '30 days',
                    'responsible_party': 'Compliance Team',
                    'success_criteria': 'Achieve minimum 80% compliance score'
                })
        
        # High-priority items
        high_priority_count = 0
        for req_id, status in compliance_results.items():
            if status.risk_level == "high" and high_priority_count < 3:
                requirement = self.requirements_database[req_id]
                
                next_steps.append({
                    'priority': 2,
                    'action': f"Improve compliance: {requirement.name}",
                    'timeline': '90 days',
                    'responsible_party': 'Technical Team',
                    'success_criteria': 'Achieve compliant status'
                })
                high_priority_count += 1
        
        # Process improvements
        next_steps.append({
            'priority': 3,
            'action': 'Implement automated compliance monitoring',
            'timeline': '180 days',
            'responsible_party': 'Engineering Team',
            'success_criteria': 'Daily automated compliance checks'
        })
        
        return next_steps
    
    def _get_upcoming_deadlines(self, jurisdiction: Jurisdiction) -> List[Dict[str, Any]]:
        """Get upcoming regulatory deadlines."""
        
        deadlines = []
        current_time = datetime.now(timezone.utc)
        
        for req_id, requirement in self.requirements_database.items():
            if (requirement.jurisdiction == jurisdiction and 
                requirement.implementation_deadline and
                requirement.implementation_deadline > current_time):
                
                days_remaining = (requirement.implementation_deadline - current_time).days
                
                deadlines.append({
                    'requirement_id': req_id,
                    'requirement_name': requirement.name,
                    'deadline': requirement.implementation_deadline.isoformat(),
                    'days_remaining': days_remaining,
                    'urgency': 'critical' if days_remaining < 90 else 'high' if days_remaining < 180 else 'medium'
                })
        
        # Sort by deadline
        deadlines.sort(key=lambda x: x['days_remaining'])
        
        return deadlines[:10]  # Next 10 deadlines
    
    def monitor_regulatory_changes(self) -> Dict[str, Any]:
        """Monitor for regulatory changes (placeholder for real implementation)."""
        
        # In production, this would connect to regulatory databases,
        # RSS feeds, legal information services, etc.
        
        monitoring_result = {
            'last_check': datetime.now(timezone.utc).isoformat(),
            'new_regulations': [],
            'updated_regulations': [],
            'upcoming_changes': [
                {
                    'regulation': 'EU AI Act',
                    'change_type': 'Implementation deadline approaching',
                    'effective_date': '2026-08-02',
                    'impact': 'High - affects all high-risk AI systems'
                }
            ],
            'recommendations': [
                'Monitor EU AI Act implementation guidelines',
                'Review US NIST AI RMF updates quarterly',
                'Subscribe to regulatory change notifications'
            ]
        }
        
        return monitoring_result
    
    def export_compliance_evidence(self, jurisdiction: Jurisdiction,
                                  format: str = "json") -> str:
        """Export compliance evidence for audit purposes."""
        
        if jurisdiction.value not in self.compliance_status:
            return json.dumps({'error': 'No compliance data available for jurisdiction'})
        
        evidence_package = {
            'jurisdiction': jurisdiction.value,
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'compliance_status': {
                req_id: {
                    'is_compliant': status.is_compliant,
                    'compliance_score': status.compliance_score,
                    'evidence': status.evidence,
                    'last_check': status.last_check.isoformat()
                }
                for req_id, status in self.compliance_status.items()
            },
            'requirements_details': {
                req_id: {
                    'name': req.name,
                    'description': req.description,
                    'compliance_level': req.compliance_level.value,
                    'framework': req.framework.value
                }
                for req_id, req in self.requirements_database.items()
            },
            'audit_trail': {
                'exported_by': 'system',
                'export_purpose': 'compliance_audit',
                'data_integrity_hash': self._calculate_evidence_hash()
            }
        }
        
        if format.lower() == "json":
            return json.dumps(evidence_package, indent=2)
        else:
            # Could add XML, CSV, or other formats
            return json.dumps(evidence_package, indent=2)
    
    def _calculate_evidence_hash(self) -> str:
        """Calculate integrity hash for compliance evidence."""
        
        evidence_data = {
            req_id: {
                'compliance_score': status.compliance_score,
                'is_compliant': status.is_compliant,
                'evidence_keys': sorted(status.evidence.keys())
            }
            for req_id, status in self.compliance_status.items()
        }
        
        evidence_json = json.dumps(evidence_data, sort_keys=True)
        return hashlib.sha256(evidence_json.encode()).hexdigest()


def create_global_compliance_engine() -> GlobalRegulatoryEngine:
    """Factory function to create configured compliance engine."""
    
    engine = GlobalRegulatoryEngine()
    
    print("Global Regulatory Compliance Engine initialized")
    print(f"Loaded {len(engine.requirements_database)} regulatory requirements")
    print(f"Supporting {len(engine.jurisdiction_mappings)} jurisdictions")
    
    return engine


# Example usage and demonstration
def demonstrate_global_compliance():
    """Demonstrate global compliance capabilities."""
    
    print("🌍 Global Regulatory Compliance Demonstration")
    print("=" * 60)
    
    # Create compliance engine
    engine = create_global_compliance_engine()
    
    # Example business contexts
    business_contexts = [
        {
            'name': 'EU Healthcare AI',
            'jurisdiction': Jurisdiction.EUROPEAN_UNION,
            'context': {
                'business_sector': 'healthcare',
                'ai_system_type': 'reward_modeling',
                'processes_personal_data': True,
                'processes_payments': False
            }
        },
        {
            'name': 'US Financial AI',
            'jurisdiction': Jurisdiction.UNITED_STATES,
            'context': {
                'business_sector': 'financial_services',
                'ai_system_type': 'automated_decision_making',
                'processes_personal_data': True,
                'processes_payments': True
            }
        },
        {
            'name': 'California Consumer AI',
            'jurisdiction': Jurisdiction.CALIFORNIA,
            'context': {
                'business_sector': 'technology',
                'ai_system_type': 'recommendation_system',
                'processes_personal_data': True,
                'processes_payments': False
            }
        }
    ]
    
    for business in business_contexts:
        print(f"\n📋 Compliance Analysis: {business['name']}")
        print("-" * 40)
        
        # Get applicable requirements
        requirements = engine.get_applicable_requirements(
            business['jurisdiction'], 
            business['context']
        )
        
        print(f"Applicable Requirements: {len(requirements)}")
        for req in requirements[:3]:  # Show first 3
            print(f"  • {req.name} ({req.framework.value})")
        
        # Generate compliance report
        report = engine.generate_compliance_report(
            business['jurisdiction'],
            business['context']
        )
        
        print(f"Overall Compliance Score: {report['executive_summary']['overall_compliance_score']:.2f}")
        print(f"Critical Issues: {report['executive_summary']['critical_issues_count']}")
        print(f"Next Steps: {len(report['next_steps'])}")
    
    print("\n🎯 Global Compliance Demonstration Complete")
    return engine


if __name__ == "__main__":
    demonstrate_global_compliance()
"""
Global Regulatory Compliance System for RLHF Contract Systems.

This module implements comprehensive regulatory compliance management
across multiple jurisdictions, including GDPR, CCPA, PDPA, AI Act, and
other global privacy and AI governance regulations.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import hashlib
import uuid

from ..models.reward_contract import RewardContract
from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class Jurisdiction(Enum):
    """Supported regulatory jurisdictions."""
    EUROPEAN_UNION = "eu"
    CALIFORNIA = "california"
    UNITED_STATES = "us"
    SINGAPORE = "singapore"
    CANADA = "canada"
    UNITED_KINGDOM = "uk"
    JAPAN = "japan"
    AUSTRALIA = "australia"
    BRAZIL = "brazil"
    INDIA = "india"
    CHINA = "china"
    SOUTH_KOREA = "south_korea"


class RegulationType(Enum):
    """Types of regulations."""
    PRIVACY = "privacy"
    AI_GOVERNANCE = "ai_governance"
    DATA_PROTECTION = "data_protection"
    ALGORITHMIC_TRANSPARENCY = "algorithmic_transparency"
    CONSUMER_PROTECTION = "consumer_protection"
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    EMPLOYMENT = "employment"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class RegulationRule:
    """Represents a specific regulatory rule or requirement."""
    id: str
    jurisdiction: Jurisdiction
    regulation_type: RegulationType
    title: str
    description: str
    requirements: List[str]
    penalties: Dict[str, str]
    effective_date: datetime
    review_frequency: timedelta
    last_updated: datetime = field(default_factory=datetime.now)
    mandatory: bool = True
    risk_level: str = "medium"  # low, medium, high, critical


@dataclass
class ComplianceAssessment:
    """Results of a compliance assessment."""
    rule_id: str
    status: ComplianceStatus
    score: float  # 0.0 to 1.0
    findings: List[str]
    recommendations: List[str]
    evidence: List[str]
    assessed_at: datetime = field(default_factory=datetime.now)
    assessor: str = "system"
    next_review_date: Optional[datetime] = None
    remediation_plan: Optional[str] = None


@dataclass
class DataSubjectRights:
    """Data subject rights management."""
    right_to_access: bool = True
    right_to_rectification: bool = True
    right_to_erasure: bool = True
    right_to_portability: bool = True
    right_to_object: bool = True
    right_to_restrict_processing: bool = True
    right_to_be_informed: bool = True
    automated_decision_making_info: bool = True


@dataclass
class PrivacyNotice:
    """Privacy notice configuration."""
    jurisdiction: Jurisdiction
    language: str
    title: str
    content: str
    last_updated: datetime
    version: str
    data_categories: List[str]
    processing_purposes: List[str]
    retention_periods: Dict[str, str]
    third_party_sharing: List[str]
    contact_info: Dict[str, str]


class GDPRComplianceManager:
    """GDPR-specific compliance management."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_gdpr_rules()
    
    def _initialize_gdpr_rules(self):
        """Initialize GDPR compliance rules."""
        self.rules = {
            "gdpr_lawful_basis": RegulationRule(
                id="gdpr_lawful_basis",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                regulation_type=RegulationType.PRIVACY,
                title="Lawful Basis for Processing",
                description="Processing must have a valid lawful basis under Article 6",
                requirements=[
                    "Identify lawful basis for each processing activity",
                    "Document lawful basis in processing records",
                    "Inform data subjects of lawful basis",
                    "Ensure processing is necessary for the stated purpose"
                ],
                penalties={"max_fine": "4% of annual turnover or €20M"},
                effective_date=datetime(2018, 5, 25),
                review_frequency=timedelta(days=365)
            ),
            
            "gdpr_data_subject_rights": RegulationRule(
                id="gdpr_data_subject_rights",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                regulation_type=RegulationType.PRIVACY,
                title="Data Subject Rights",
                description="Implement mechanisms for data subject rights",
                requirements=[
                    "Right to access (Article 15)",
                    "Right to rectification (Article 16)", 
                    "Right to erasure (Article 17)",
                    "Right to data portability (Article 20)",
                    "Right to object (Article 21)",
                    "Response within 30 days (extendable to 90 days)"
                ],
                penalties={"max_fine": "4% of annual turnover or €20M"},
                effective_date=datetime(2018, 5, 25),
                review_frequency=timedelta(days=180)
            ),
            
            "gdpr_consent": RegulationRule(
                id="gdpr_consent",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                regulation_type=RegulationType.PRIVACY,
                title="Consent Requirements",
                description="Consent must be freely given, specific, informed, and unambiguous",
                requirements=[
                    "Clear affirmative action required",
                    "Granular consent for different purposes",
                    "Easy withdrawal mechanism",
                    "Documentation of consent",
                    "Regular consent refresh"
                ],
                penalties={"max_fine": "4% of annual turnover or €20M"},
                effective_date=datetime(2018, 5, 25),
                review_frequency=timedelta(days=90)
            ),
            
            "gdpr_automated_decision_making": RegulationRule(
                id="gdpr_automated_decision_making",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                regulation_type=RegulationType.AI_GOVERNANCE,
                title="Automated Decision-Making and Profiling",
                description="Rights and safeguards for automated decision-making (Article 22)",
                requirements=[
                    "Right not to be subject to automated decision-making",
                    "Meaningful information about logic involved",
                    "Right to human intervention",
                    "Right to contest the decision",
                    "Regular accuracy testing"
                ],
                penalties={"max_fine": "4% of annual turnover or €20M"},
                effective_date=datetime(2018, 5, 25),
                review_frequency=timedelta(days=90)
            ),
            
            "gdpr_privacy_by_design": RegulationRule(
                id="gdpr_privacy_by_design",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                regulation_type=RegulationType.DATA_PROTECTION,
                title="Privacy by Design and Default",
                description="Data protection by design and by default (Article 25)",
                requirements=[
                    "Implement technical and organizational measures",
                    "Privacy considerations at design stage",
                    "Default settings protect privacy",
                    "Regular review and updates",
                    "Documentation of privacy measures"
                ],
                penalties={"max_fine": "4% of annual turnover or €20M"},
                effective_date=datetime(2018, 5, 25),
                review_frequency=timedelta(days=180)
            )
        }
    
    async def assess_gdpr_compliance(
        self,
        contract: RewardContract,
        processing_context: Dict[str, Any]
    ) -> List[ComplianceAssessment]:
        """Assess GDPR compliance for a reward contract."""
        
        assessments = []
        
        for rule_id, rule in self.rules.items():
            assessment = await self._assess_single_gdpr_rule(rule, contract, processing_context)
            assessments.append(assessment)
        
        return assessments
    
    async def _assess_single_gdpr_rule(
        self,
        rule: RegulationRule,
        contract: RewardContract,
        context: Dict[str, Any]
    ) -> ComplianceAssessment:
        """Assess compliance with a single GDPR rule."""
        
        findings = []
        recommendations = []
        evidence = []
        score = 1.0  # Start optimistic
        
        if rule.id == "gdpr_lawful_basis":
            # Check if lawful basis is documented
            lawful_basis = context.get("lawful_basis")
            if not lawful_basis:
                findings.append("No lawful basis documented")
                recommendations.append("Document lawful basis for processing")
                score -= 0.8
            else:
                evidence.append(f"Lawful basis: {lawful_basis}")
        
        elif rule.id == "gdpr_data_subject_rights":
            # Check if rights mechanisms are implemented
            rights_mechanisms = context.get("data_subject_rights", {})
            
            required_rights = [
                "access", "rectification", "erasure", "portability", "object"
            ]
            
            for right in required_rights:
                if not rights_mechanisms.get(right, False):
                    findings.append(f"No mechanism for right to {right}")
                    recommendations.append(f"Implement right to {right} mechanism")
                    score -= 0.15
                else:
                    evidence.append(f"Right to {right} implemented")
        
        elif rule.id == "gdpr_consent":
            # Check consent mechanism
            consent_mechanism = context.get("consent_mechanism", {})
            
            if not consent_mechanism.get("granular", False):
                findings.append("Consent not granular")
                recommendations.append("Implement granular consent options")
                score -= 0.3
            
            if not consent_mechanism.get("withdrawal_easy", False):
                findings.append("Consent withdrawal not easy")
                recommendations.append("Simplify consent withdrawal process")
                score -= 0.2
        
        elif rule.id == "gdpr_automated_decision_making":
            # Check if automated decision-making safeguards exist
            adm_safeguards = context.get("automated_decision_making", {})
            
            if not adm_safeguards.get("human_intervention", False):
                findings.append("No human intervention mechanism")
                recommendations.append("Implement human intervention capability")
                score -= 0.4
            
            if not adm_safeguards.get("logic_explanation", False):
                findings.append("No explanation of automated logic")
                recommendations.append("Provide meaningful information about logic")
                score -= 0.3
        
        elif rule.id == "gdpr_privacy_by_design":
            # Check privacy by design implementation
            privacy_measures = context.get("privacy_by_design", {})
            
            if not privacy_measures.get("technical_measures", []):
                findings.append("No technical privacy measures documented")
                recommendations.append("Document technical privacy measures")
                score -= 0.3
            
            if not privacy_measures.get("organizational_measures", []):
                findings.append("No organizational privacy measures documented") 
                recommendations.append("Document organizational privacy measures")
                score -= 0.3
        
        # Determine overall status
        if score >= 0.9:
            status = ComplianceStatus.COMPLIANT
        elif score >= 0.7:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return ComplianceAssessment(
            rule_id=rule.id,
            status=status,
            score=max(0.0, score),
            findings=findings,
            recommendations=recommendations,
            evidence=evidence,
            next_review_date=datetime.now() + rule.review_frequency
        )


class CCPAComplianceManager:
    """CCPA-specific compliance management."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_ccpa_rules()
    
    def _initialize_ccpa_rules(self):
        """Initialize CCPA compliance rules."""
        self.rules = {
            "ccpa_consumer_rights": RegulationRule(
                id="ccpa_consumer_rights",
                jurisdiction=Jurisdiction.CALIFORNIA,
                regulation_type=RegulationType.PRIVACY,
                title="Consumer Rights",
                description="CCPA consumer rights implementation",
                requirements=[
                    "Right to know about personal information collection",
                    "Right to delete personal information",
                    "Right to opt-out of sale of personal information",
                    "Right to non-discrimination for exercising rights"
                ],
                penalties={"civil_penalty": "Up to $7,500 per violation"},
                effective_date=datetime(2020, 1, 1),
                review_frequency=timedelta(days=180)
            ),
            
            "ccpa_privacy_notice": RegulationRule(
                id="ccpa_privacy_notice",
                jurisdiction=Jurisdiction.CALIFORNIA,
                regulation_type=RegulationType.PRIVACY,
                title="Privacy Notice Requirements",
                description="CCPA privacy notice disclosure requirements",
                requirements=[
                    "Categories of personal information collected",
                    "Sources of personal information",
                    "Business purposes for collection",
                    "Categories of third parties sharing information",
                    "Consumer rights information"
                ],
                penalties={"civil_penalty": "Up to $2,500 per violation"},
                effective_date=datetime(2020, 1, 1),
                review_frequency=timedelta(days=365)
            )
        }
    
    async def assess_ccpa_compliance(
        self,
        contract: RewardContract,
        processing_context: Dict[str, Any]
    ) -> List[ComplianceAssessment]:
        """Assess CCPA compliance for a reward contract."""
        
        assessments = []
        
        for rule_id, rule in self.rules.items():
            assessment = await self._assess_single_ccpa_rule(rule, contract, processing_context)
            assessments.append(assessment)
        
        return assessments
    
    async def _assess_single_ccpa_rule(
        self,
        rule: RegulationRule,
        contract: RewardContract,
        context: Dict[str, Any]
    ) -> ComplianceAssessment:
        """Assess compliance with a single CCPA rule."""
        
        findings = []
        recommendations = []
        evidence = []
        score = 1.0
        
        if rule.id == "ccpa_consumer_rights":
            # Check consumer rights implementation
            consumer_rights = context.get("consumer_rights", {})
            
            if not consumer_rights.get("right_to_know", False):
                findings.append("Right to know not implemented")
                recommendations.append("Implement right to know mechanism")
                score -= 0.3
            
            if not consumer_rights.get("right_to_delete", False):
                findings.append("Right to delete not implemented")
                recommendations.append("Implement right to delete mechanism")
                score -= 0.3
            
            if not consumer_rights.get("opt_out_sale", False):
                findings.append("Opt-out of sale not implemented")
                recommendations.append("Implement opt-out of sale mechanism")
                score -= 0.4
        
        elif rule.id == "ccpa_privacy_notice":
            # Check privacy notice completeness
            privacy_notice = context.get("privacy_notice", {})
            
            required_disclosures = [
                "categories_collected", "sources", "business_purposes", 
                "third_party_sharing", "consumer_rights"
            ]
            
            for disclosure in required_disclosures:
                if not privacy_notice.get(disclosure):
                    findings.append(f"Missing disclosure: {disclosure}")
                    recommendations.append(f"Add {disclosure} to privacy notice")
                    score -= 0.15
        
        # Determine status
        if score >= 0.9:
            status = ComplianceStatus.COMPLIANT
        elif score >= 0.7:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return ComplianceAssessment(
            rule_id=rule.id,
            status=status,
            score=max(0.0, score),
            findings=findings,
            recommendations=recommendations,
            evidence=evidence
        )


class AIActComplianceManager:
    """EU AI Act compliance management."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_ai_act_rules()
    
    def _initialize_ai_act_rules(self):
        """Initialize EU AI Act compliance rules."""
        self.rules = {
            "ai_act_high_risk": RegulationRule(
                id="ai_act_high_risk",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                regulation_type=RegulationType.AI_GOVERNANCE,
                title="High-Risk AI Systems",
                description="Requirements for high-risk AI systems",
                requirements=[
                    "Risk management system",
                    "Data governance and quality",
                    "Technical documentation",
                    "Record keeping and logging",
                    "Transparency and user information",
                    "Human oversight",
                    "Accuracy, robustness and cybersecurity"
                ],
                penalties={"max_fine": "€30M or 6% of annual turnover"},
                effective_date=datetime(2025, 8, 1),  # Expected
                review_frequency=timedelta(days=180)
            ),
            
            "ai_act_transparency": RegulationRule(
                id="ai_act_transparency",
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                regulation_type=RegulationType.ALGORITHMIC_TRANSPARENCY,
                title="AI System Transparency",
                description="Transparency obligations for AI systems",
                requirements=[
                    "Clear information that users are interacting with AI",
                    "Information about AI system capabilities and limitations",
                    "Appropriate level of human oversight",
                    "Clear and comprehensive instructions for use"
                ],
                penalties={"max_fine": "€15M or 3% of annual turnover"},
                effective_date=datetime(2025, 8, 1),
                review_frequency=timedelta(days=90)
            ),
            
            "ai_act_foundation_models": RegulationRule(
                id="ai_act_foundation_models", 
                jurisdiction=Jurisdiction.EUROPEAN_UNION,
                regulation_type=RegulationType.AI_GOVERNANCE,
                title="Foundation Model Requirements",
                description="Obligations for foundation models with systemic risk",
                requirements=[
                    "Model evaluation and testing",
                    "Adversarial testing and red-teaming", 
                    "Tracking and reporting of serious incidents",
                    "Cybersecurity measures",
                    "Systemic risk assessment and mitigation"
                ],
                penalties={"max_fine": "€15M or 3% of annual turnover"},
                effective_date=datetime(2025, 8, 1),
                review_frequency=timedelta(days=90)
            )
        }
    
    async def assess_ai_act_compliance(
        self,
        contract: RewardContract,
        processing_context: Dict[str, Any]
    ) -> List[ComplianceAssessment]:
        """Assess EU AI Act compliance for a reward contract."""
        
        assessments = []
        
        # Determine if system is high-risk
        is_high_risk = self._determine_high_risk_status(contract, processing_context)
        
        for rule_id, rule in self.rules.items():
            if rule_id == "ai_act_high_risk" and not is_high_risk:
                # Skip high-risk requirements for non-high-risk systems
                continue
                
            assessment = await self._assess_single_ai_act_rule(
                rule, contract, processing_context, is_high_risk
            )
            assessments.append(assessment)
        
        return assessments
    
    def _determine_high_risk_status(
        self,
        contract: RewardContract,
        context: Dict[str, Any]
    ) -> bool:
        """Determine if the AI system qualifies as high-risk."""
        
        # High-risk categories under AI Act
        use_case = context.get("use_case", "")
        application_domain = context.get("application_domain", "")
        
        high_risk_domains = [
            "employment", "credit_scoring", "law_enforcement", 
            "migration_asylum", "education_training", "healthcare",
            "critical_infrastructure", "democratic_processes"
        ]
        
        return any(domain in application_domain.lower() for domain in high_risk_domains)
    
    async def _assess_single_ai_act_rule(
        self,
        rule: RegulationRule,
        contract: RewardContract,
        context: Dict[str, Any],
        is_high_risk: bool
    ) -> ComplianceAssessment:
        """Assess compliance with a single AI Act rule."""
        
        findings = []
        recommendations = []
        evidence = []
        score = 1.0
        
        if rule.id == "ai_act_high_risk":
            # Check high-risk system requirements
            risk_measures = context.get("risk_management", {})
            
            if not risk_measures.get("risk_management_system", False):
                findings.append("No risk management system implemented")
                recommendations.append("Implement comprehensive risk management system")
                score -= 0.2
            
            if not risk_measures.get("data_governance", False):
                findings.append("No data governance framework")
                recommendations.append("Establish data governance and quality framework")
                score -= 0.15
            
            if not risk_measures.get("technical_documentation", False):
                findings.append("Insufficient technical documentation")
                recommendations.append("Create comprehensive technical documentation")
                score -= 0.15
            
            if not risk_measures.get("human_oversight", False):
                findings.append("No human oversight mechanism")
                recommendations.append("Implement meaningful human oversight")
                score -= 0.2
        
        elif rule.id == "ai_act_transparency":
            # Check transparency requirements
            transparency = context.get("transparency", {})
            
            if not transparency.get("ai_disclosure", False):
                findings.append("Users not informed they are interacting with AI")
                recommendations.append("Clearly inform users about AI interaction")
                score -= 0.3
            
            if not transparency.get("capabilities_limitations", False):
                findings.append("AI capabilities and limitations not disclosed")
                recommendations.append("Document and disclose AI capabilities and limitations")
                score -= 0.3
            
            if not transparency.get("instructions_for_use", False):
                findings.append("No clear instructions for use provided")
                recommendations.append("Provide comprehensive instructions for use")
                score -= 0.2
        
        elif rule.id == "ai_act_foundation_models":
            # Check foundation model requirements
            foundation_measures = context.get("foundation_model", {})
            
            if not foundation_measures.get("model_evaluation", False):
                findings.append("No systematic model evaluation")
                recommendations.append("Implement comprehensive model evaluation")
                score -= 0.25
            
            if not foundation_measures.get("red_teaming", False):
                findings.append("No adversarial testing performed")
                recommendations.append("Conduct regular red-teaming exercises")
                score -= 0.25
            
            if not foundation_measures.get("incident_reporting", False):
                findings.append("No incident tracking and reporting system")
                recommendations.append("Implement incident tracking and reporting")
                score -= 0.2
        
        # Determine status
        if score >= 0.9:
            status = ComplianceStatus.COMPLIANT
        elif score >= 0.7:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return ComplianceAssessment(
            rule_id=rule.id,
            status=status,
            score=max(0.0, score),
            findings=findings,
            recommendations=recommendations,
            evidence=evidence
        )


class GlobalRegulatoryComplianceSystem:
    """
    Comprehensive global regulatory compliance management system.
    
    Manages compliance across multiple jurisdictions and regulation types,
    providing unified compliance assessment, monitoring, and reporting.
    """
    
    def __init__(self, output_dir: Path = Path("compliance_outputs")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize jurisdiction-specific managers
        self.gdpr_manager = GDPRComplianceManager()
        self.ccpa_manager = CCPAComplianceManager()
        self.ai_act_manager = AIActComplianceManager()
        
        # Global compliance state
        self.active_jurisdictions: Set[Jurisdiction] = set()
        self.compliance_history: List[Dict[str, Any]] = []
        
        # Privacy notices by jurisdiction and language
        self.privacy_notices: Dict[Tuple[Jurisdiction, str], PrivacyNotice] = {}
        
        # Data subject rights configuration
        self.data_subject_rights = DataSubjectRights()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup compliance logging."""
        log_file = self.output_dir / "compliance.log"
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - COMPLIANCE - %(levelname)s - %(message)s'
        ))
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def register_jurisdiction(
        self,
        jurisdiction: Jurisdiction,
        data_residency_requirements: Dict[str, Any] = None,
        local_representative: Dict[str, str] = None
    ):
        """Register a new jurisdiction for compliance monitoring."""
        
        self.active_jurisdictions.add(jurisdiction)
        
        self.logger.info(f"Registered jurisdiction: {jurisdiction.value}")
        
        # Initialize default privacy notice if needed
        self._ensure_privacy_notice(jurisdiction, "en")
    
    def _ensure_privacy_notice(self, jurisdiction: Jurisdiction, language: str):
        """Ensure privacy notice exists for jurisdiction and language."""
        
        key = (jurisdiction, language)
        
        if key not in self.privacy_notices:
            # Create default privacy notice
            self.privacy_notices[key] = PrivacyNotice(
                jurisdiction=jurisdiction,
                language=language,
                title=f"Privacy Notice - {jurisdiction.value.title()}",
                content=self._generate_default_privacy_notice_content(jurisdiction, language),
                last_updated=datetime.now(),
                version="1.0.0",
                data_categories=["user_interactions", "system_metrics", "preferences"],
                processing_purposes=["service_provision", "improvement", "legal_compliance"],
                retention_periods={"user_data": "2 years", "system_logs": "1 year"},
                third_party_sharing=["cloud_providers", "analytics_services"],
                contact_info={
                    "dpo_email": "privacy@rlhf-contract.org",
                    "support_email": "support@rlhf-contract.org"
                }
            )
    
    def _generate_default_privacy_notice_content(
        self, 
        jurisdiction: Jurisdiction, 
        language: str
    ) -> str:
        """Generate default privacy notice content for jurisdiction."""
        
        # Simplified template - would use proper localization
        base_content = f"""
PRIVACY NOTICE - {jurisdiction.value.upper()}

This privacy notice explains how we collect, use, and protect your personal 
information when you use our RLHF Contract Wizard services.

INFORMATION WE COLLECT
- User interaction data for service improvement
- System performance metrics for optimization
- User preferences for personalization

HOW WE USE YOUR INFORMATION
- To provide and improve our services
- To ensure system security and compliance
- To fulfill legal obligations

YOUR RIGHTS
- Access your personal information
- Correct inaccurate information
- Delete your personal information
- Object to processing
- Data portability

CONTACT US
For privacy-related questions, contact: privacy@rlhf-contract.org

Last updated: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        return base_content.strip()
    
    async def assess_global_compliance(
        self,
        contract: RewardContract,
        processing_context: Dict[str, Any],
        target_jurisdictions: List[Jurisdiction] = None
    ) -> Dict[Jurisdiction, List[ComplianceAssessment]]:
        """
        Assess compliance across multiple jurisdictions.
        
        Args:
            contract: The reward contract to assess
            processing_context: Context about data processing
            target_jurisdictions: Specific jurisdictions to assess
            
        Returns:
            Compliance assessments by jurisdiction
        """
        
        if target_jurisdictions is None:
            target_jurisdictions = list(self.active_jurisdictions)
        
        if not target_jurisdictions:
            target_jurisdictions = [Jurisdiction.EUROPEAN_UNION, Jurisdiction.CALIFORNIA]
        
        self.logger.info(f"Starting global compliance assessment for {len(target_jurisdictions)} jurisdictions")
        
        compliance_results = {}
        
        for jurisdiction in target_jurisdictions:
            try:
                assessments = await self._assess_jurisdiction_compliance(
                    jurisdiction, contract, processing_context
                )
                compliance_results[jurisdiction] = assessments
                
            except Exception as e:
                self.logger.error(f"Compliance assessment failed for {jurisdiction.value}: {e}")
                compliance_results[jurisdiction] = []
        
        # Store assessment in history
        self.compliance_history.append({
            "timestamp": datetime.now(),
            "contract_hash": contract.compute_hash(),
            "jurisdictions": [j.value for j in target_jurisdictions],
            "results_summary": {
                jurisdiction.value: {
                    "total_assessments": len(assessments),
                    "compliant": len([a for a in assessments if a.status == ComplianceStatus.COMPLIANT]),
                    "avg_score": sum(a.score for a in assessments) / len(assessments) if assessments else 0
                }
                for jurisdiction, assessments in compliance_results.items()
            }
        })
        
        self.logger.info("Global compliance assessment completed")
        
        return compliance_results
    
    async def _assess_jurisdiction_compliance(
        self,
        jurisdiction: Jurisdiction,
        contract: RewardContract,
        processing_context: Dict[str, Any]
    ) -> List[ComplianceAssessment]:
        """Assess compliance for a specific jurisdiction."""
        
        assessments = []
        
        if jurisdiction == Jurisdiction.EUROPEAN_UNION:
            # GDPR compliance
            gdpr_assessments = await self.gdpr_manager.assess_gdpr_compliance(
                contract, processing_context
            )
            assessments.extend(gdpr_assessments)
            
            # EU AI Act compliance (if applicable)
            if processing_context.get("uses_ai", True):
                ai_act_assessments = await self.ai_act_manager.assess_ai_act_compliance(
                    contract, processing_context
                )
                assessments.extend(ai_act_assessments)
        
        elif jurisdiction == Jurisdiction.CALIFORNIA:
            # CCPA compliance
            ccpa_assessments = await self.ccpa_manager.assess_ccpa_compliance(
                contract, processing_context
            )
            assessments.extend(ccpa_assessments)
        
        else:
            # Other jurisdictions - implement as needed
            self.logger.warning(f"No specific compliance rules implemented for {jurisdiction.value}")
            
            # Create generic assessment
            generic_assessment = ComplianceAssessment(
                rule_id=f"{jurisdiction.value}_generic",
                status=ComplianceStatus.UNDER_REVIEW,
                score=0.5,
                findings=["Manual review required"],
                recommendations=["Conduct jurisdiction-specific compliance review"],
                evidence=["Generic assessment performed"]
            )
            assessments.append(generic_assessment)
        
        return assessments
    
    async def generate_compliance_report(
        self,
        assessment_results: Dict[Jurisdiction, List[ComplianceAssessment]],
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_id": str(uuid.uuid4()),
                "jurisdictions_assessed": len(assessment_results),
                "total_assessments": sum(len(assessments) for assessments in assessment_results.values())
            },
            "executive_summary": {},
            "jurisdiction_details": {},
            "global_recommendations": [],
            "risk_assessment": {},
            "remediation_plan": {}
        }
        
        # Executive summary
        total_assessments = 0
        total_compliant = 0
        total_score = 0
        critical_issues = []
        
        for jurisdiction, assessments in assessment_results.items():
            total_assessments += len(assessments)
            
            jurisdiction_compliant = len([a for a in assessments if a.status == ComplianceStatus.COMPLIANT])
            total_compliant += jurisdiction_compliant
            
            jurisdiction_score = sum(a.score for a in assessments) / len(assessments) if assessments else 0
            total_score += jurisdiction_score
            
            # Identify critical issues
            for assessment in assessments:
                if assessment.status == ComplianceStatus.NON_COMPLIANT and assessment.score < 0.3:
                    critical_issues.append({
                        "jurisdiction": jurisdiction.value,
                        "rule": assessment.rule_id,
                        "score": assessment.score,
                        "findings": assessment.findings
                    })
        
        overall_compliance_rate = total_compliant / total_assessments if total_assessments > 0 else 0
        average_score = total_score / len(assessment_results) if assessment_results else 0
        
        report["executive_summary"] = {
            "overall_compliance_rate": overall_compliance_rate,
            "average_compliance_score": average_score,
            "critical_issues_count": len(critical_issues),
            "jurisdictions_fully_compliant": len([
                j for j, assessments in assessment_results.items()
                if all(a.status == ComplianceStatus.COMPLIANT for a in assessments)
            ]),
            "overall_risk_level": self._determine_overall_risk_level(average_score, critical_issues)
        }
        
        # Jurisdiction details
        for jurisdiction, assessments in assessment_results.items():
            jurisdiction_compliant = len([a for a in assessments if a.status == ComplianceStatus.COMPLIANT])
            jurisdiction_score = sum(a.score for a in assessments) / len(assessments) if assessments else 0
            
            report["jurisdiction_details"][jurisdiction.value] = {
                "compliance_rate": jurisdiction_compliant / len(assessments) if assessments else 0,
                "average_score": jurisdiction_score,
                "total_assessments": len(assessments),
                "compliant_assessments": jurisdiction_compliant,
                "assessments": [
                    {
                        "rule_id": a.rule_id,
                        "status": a.status.value,
                        "score": a.score,
                        "findings": a.findings,
                        "recommendations": a.recommendations if include_recommendations else []
                    }
                    for a in assessments
                ]
            }
        
        # Global recommendations
        if include_recommendations:
            all_recommendations = []
            for assessments in assessment_results.values():
                for assessment in assessments:
                    all_recommendations.extend(assessment.recommendations)
            
            # Deduplicate and prioritize recommendations
            unique_recommendations = list(set(all_recommendations))
            report["global_recommendations"] = unique_recommendations[:10]  # Top 10
        
        # Risk assessment
        report["risk_assessment"] = {
            "critical_issues": critical_issues,
            "high_risk_jurisdictions": [
                j.value for j, assessments in assessment_results.items()
                if any(a.score < 0.5 for a in assessments)
            ],
            "compliance_gaps": [
                assessment.rule_id for assessments in assessment_results.values()
                for assessment in assessments
                if assessment.status == ComplianceStatus.NON_COMPLIANT
            ]
        }
        
        return report
    
    def _determine_overall_risk_level(
        self, 
        average_score: float, 
        critical_issues: List[Dict[str, Any]]
    ) -> str:
        """Determine overall compliance risk level."""
        
        if len(critical_issues) > 0 or average_score < 0.5:
            return "HIGH"
        elif average_score < 0.7:
            return "MEDIUM"
        elif average_score < 0.9:
            return "LOW"
        else:
            return "MINIMAL"
    
    async def save_compliance_report(
        self, 
        report: Dict[str, Any], 
        filename: Optional[str] = None
    ):
        """Save compliance report to file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compliance_report_{timestamp}.json"
        
        report_file = self.output_dir / filename
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Compliance report saved to {report_file}")
        
        # Also save human-readable version
        readable_file = self.output_dir / filename.replace('.json', '.txt')
        readable_report = self._generate_readable_compliance_report(report)
        
        with open(readable_file, 'w') as f:
            f.write(readable_report)
        
        return report_file
    
    def _generate_readable_compliance_report(self, report: Dict[str, Any]) -> str:
        """Generate human-readable compliance report."""
        
        lines = []
        lines.append("=" * 80)
        lines.append("GLOBAL REGULATORY COMPLIANCE REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Report ID: {report['report_metadata']['report_id']}")
        lines.append("")
        
        # Executive Summary
        summary = report["executive_summary"]
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Overall Compliance Rate: {summary['overall_compliance_rate']:.1%}")
        lines.append(f"Average Compliance Score: {summary['average_compliance_score']:.2f}/1.00")
        lines.append(f"Critical Issues: {summary['critical_issues_count']}")
        lines.append(f"Fully Compliant Jurisdictions: {summary['jurisdictions_fully_compliant']}")
        lines.append(f"Overall Risk Level: {summary['overall_risk_level']}")
        lines.append("")
        
        # Jurisdiction Details
        lines.append("JURISDICTION COMPLIANCE DETAILS")
        lines.append("-" * 40)
        
        for jurisdiction, details in report["jurisdiction_details"].items():
            lines.append(f"\n{jurisdiction.upper()}")
            lines.append(f"  Compliance Rate: {details['compliance_rate']:.1%}")
            lines.append(f"  Average Score: {details['average_score']:.2f}")
            lines.append(f"  Assessments: {details['compliant_assessments']}/{details['total_assessments']}")
            
            # Show non-compliant assessments
            non_compliant = [a for a in details['assessments'] 
                           if a['status'] != 'compliant']
            
            if non_compliant:
                lines.append("  Issues:")
                for assessment in non_compliant[:3]:  # Show top 3 issues
                    lines.append(f"    • {assessment['rule_id']}: {assessment['status']}")
                    if assessment['findings']:
                        lines.append(f"      {assessment['findings'][0]}")
        
        lines.append("")
        
        # Risk Assessment
        risk = report["risk_assessment"]
        if risk["critical_issues"]:
            lines.append("CRITICAL ISSUES")
            lines.append("-" * 40)
            
            for issue in risk["critical_issues"][:5]:  # Top 5 critical issues
                lines.append(f"• {issue['jurisdiction']}: {issue['rule']}")
                lines.append(f"  Score: {issue['score']:.2f}")
                if issue['findings']:
                    lines.append(f"  Issue: {issue['findings'][0]}")
            
            lines.append("")
        
        # Recommendations
        if "global_recommendations" in report and report["global_recommendations"]:
            lines.append("PRIORITY RECOMMENDATIONS")
            lines.append("-" * 40)
            
            for i, rec in enumerate(report["global_recommendations"][:10], 1):
                lines.append(f"{i}. {rec}")
            
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_privacy_notice(
        self, 
        jurisdiction: Jurisdiction, 
        language: str = "en"
    ) -> Optional[PrivacyNotice]:
        """Get privacy notice for jurisdiction and language."""
        
        key = (jurisdiction, language)
        return self.privacy_notices.get(key)
    
    def update_privacy_notice(
        self,
        jurisdiction: Jurisdiction,
        language: str,
        updates: Dict[str, Any]
    ):
        """Update privacy notice for jurisdiction and language."""
        
        key = (jurisdiction, language)
        
        if key in self.privacy_notices:
            notice = self.privacy_notices[key]
            
            for field, value in updates.items():
                if hasattr(notice, field):
                    setattr(notice, field, value)
            
            notice.last_updated = datetime.now()
            notice.version = f"{float(notice.version) + 0.1:.1f}"
            
            self.logger.info(f"Updated privacy notice for {jurisdiction.value} ({language})")
        else:
            self.logger.warning(f"Privacy notice not found for {jurisdiction.value} ({language})")


# Example usage and demonstration
if __name__ == "__main__":
    
    async def main():
        print("🌍 Global Regulatory Compliance System Demo")
        
        # Initialize compliance system
        print("\n🔧 Initializing Global Regulatory Compliance System...")
        
        compliance_system = GlobalRegulatoryComplianceSystem(
            output_dir=Path("compliance_demo")
        )
        
        # Register jurisdictions
        compliance_system.register_jurisdiction(Jurisdiction.EUROPEAN_UNION)
        compliance_system.register_jurisdiction(Jurisdiction.CALIFORNIA)
        
        print(f"   ✅ Registered {len(compliance_system.active_jurisdictions)} jurisdictions")
        
        # Create test contract
        from pathlib import Path
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        
        try:
            from models.reward_contract import RewardContract
            
            contract = RewardContract(
                name="global_compliance_test",
                stakeholders={"users": 0.6, "safety": 0.4}
            )
            
            # Create processing context
            processing_context = {
                "lawful_basis": "legitimate_interest",
                "data_subject_rights": {
                    "access": True,
                    "rectification": True,
                    "erasure": True,
                    "portability": True,
                    "object": False  # Intentional gap for testing
                },
                "consent_mechanism": {
                    "granular": True,
                    "withdrawal_easy": False  # Intentional gap
                },
                "automated_decision_making": {
                    "human_intervention": True,
                    "logic_explanation": False  # Intentional gap
                },
                "privacy_by_design": {
                    "technical_measures": ["encryption", "access_controls"],
                    "organizational_measures": ["privacy_training", "data_governance"]
                },
                "consumer_rights": {
                    "right_to_know": True,
                    "right_to_delete": True,
                    "opt_out_sale": True
                },
                "privacy_notice": {
                    "categories_collected": True,
                    "sources": True,
                    "business_purposes": True,
                    "third_party_sharing": False,  # Intentional gap
                    "consumer_rights": True
                },
                "uses_ai": True,
                "use_case": "preference learning",
                "application_domain": "general AI assistant",
                "risk_management": {
                    "risk_management_system": True,
                    "data_governance": True,
                    "technical_documentation": False,  # Intentional gap
                    "human_oversight": True
                },
                "transparency": {
                    "ai_disclosure": True,
                    "capabilities_limitations": False,  # Intentional gap
                    "instructions_for_use": True
                }
            }
            
            # Run global compliance assessment
            print("\n📋 Running Global Compliance Assessment...")
            
            assessment_results = await compliance_system.assess_global_compliance(
                contract=contract,
                processing_context=processing_context
            )
            
            # Display results summary
            for jurisdiction, assessments in assessment_results.items():
                compliant_count = len([a for a in assessments if a.status == ComplianceStatus.COMPLIANT])
                total_count = len(assessments)
                avg_score = sum(a.score for a in assessments) / len(assessments) if assessments else 0
                
                print(f"\n   🏛️ {jurisdiction.value.upper()}")
                print(f"      Compliance Rate: {compliant_count}/{total_count} ({compliant_count/total_count:.1%})")
                print(f"      Average Score: {avg_score:.2f}/1.00")
                
                # Show some specific findings
                issues = [a for a in assessments if a.status != ComplianceStatus.COMPLIANT]
                if issues:
                    print(f"      Issues Found: {len(issues)}")
                    for issue in issues[:2]:  # Show first 2 issues
                        print(f"        • {issue.rule_id}: {issue.findings[0] if issue.findings else 'See details'}")
            
            # Generate comprehensive report
            print("\n📊 Generating Comprehensive Compliance Report...")
            
            report = await compliance_system.generate_compliance_report(
                assessment_results=assessment_results,
                include_recommendations=True
            )
            
            print(f"   📈 Executive Summary:")
            print(f"      Overall Compliance Rate: {report['executive_summary']['overall_compliance_rate']:.1%}")
            print(f"      Average Score: {report['executive_summary']['average_compliance_score']:.2f}")
            print(f"      Critical Issues: {report['executive_summary']['critical_issues_count']}")
            print(f"      Risk Level: {report['executive_summary']['overall_risk_level']}")
            
            # Save report
            report_file = await compliance_system.save_compliance_report(report)
            
            print(f"\n💾 Reports saved to: {report_file.parent}/")
            
            # Test privacy notice functionality
            print("\n📄 Testing Privacy Notice Management...")
            
            eu_notice = compliance_system.get_privacy_notice(Jurisdiction.EUROPEAN_UNION, "en")
            if eu_notice:
                print(f"   ✅ EU Privacy Notice: {eu_notice.title}")
                print(f"      Version: {eu_notice.version}")
                print(f"      Last Updated: {eu_notice.last_updated.strftime('%Y-%m-%d')}")
            
            # Update privacy notice
            compliance_system.update_privacy_notice(
                Jurisdiction.EUROPEAN_UNION,
                "en",
                {"version": "2.0.0", "title": "Updated Privacy Notice - EU"}
            )
            
            print("   🔄 Privacy notice updated")
            
            print("\n✅ Global Regulatory Compliance System demonstration completed!")
            print("🌍 System ready for multi-jurisdictional compliance management!")
            
        except ImportError as e:
            print(f"\n⚠️ Could not import RewardContract: {e}")
            print("   Using mock contract for demo...")
            
            # Mock contract for demonstration
            class MockContract:
                def __init__(self):
                    self.metadata = type('obj', (object,), {'name': 'mock_contract'})()
                
                def compute_hash(self):
                    return "mock_hash_12345"
            
            contract = MockContract()
            
            # Simplified demo with mock contract
            print("\n📋 Mock Compliance Assessment (RewardContract not available)")
            print("   Would assess GDPR, CCPA, and AI Act compliance")
            print("   Would generate comprehensive compliance report")
            print("   Would manage privacy notices across jurisdictions")
            
            print("\n✅ Demo completed with mock data!")
    
    asyncio.run(main())
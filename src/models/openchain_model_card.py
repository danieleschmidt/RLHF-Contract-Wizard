"""
OpenChain Model Card implementation.

Provides functionality for generating machine-readable model cards
that comply with OpenChain v2 specification and include contract metadata.
"""

import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .reward_contract import RewardContract
from .legal_blocks import LegalBlocks


class ModelCardVersion(Enum):
    """Supported model card specification versions."""
    OPENCHAIN_V1 = "openchain-v1"
    OPENCHAIN_V2 = "openchain-v2"
    HUGGINGFACE = "huggingface"
    MLOPS = "mlops"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    helpfulness: Optional[float] = None
    harmlessness: Optional[float] = None
    contract_compliance: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class LegalMetadata:
    """Legal and compliance metadata."""
    jurisdiction: str = "Global"
    regulatory_framework: Optional[str] = None
    liability_cap: Optional[str] = None
    audit_frequency: str = "quarterly"
    compliance_certifications: List[str] = field(default_factory=list)
    data_protection_standards: List[str] = field(default_factory=list)


class OpenChainModelCard:
    """
    OpenChain-compliant model card generator.
    
    Generates machine-readable model cards that include RLHF contract
    specifications, legal metadata, and compliance information.
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0.0",
        base_model: Optional[str] = None,
        contract: Optional[RewardContract] = None
    ):
        """Initialize model card generator."""
        self.model_name = model_name
        self.model_version = model_version
        self.base_model = base_model
        self.contract = contract
        self.created_at = time.time()
        
        # Core model card data
        self._card_data = {
            "model_card_version": "openchain-v2",
            "model_name": model_name,
            "model_version": model_version,
            "created_at": self.created_at,
            "updated_at": self.created_at
        }
    
    def set_model_details(
        self,
        description: str,
        base_model: Optional[str] = None,
        model_type: str = "language_model",
        architecture: Optional[str] = None,
        num_parameters: Optional[str] = None,
        training_steps: Optional[int] = None
    ) -> 'OpenChainModelCard':
        """Set basic model details."""
        self._card_data["model_details"] = {
            "description": description,
            "base_model": base_model or self.base_model,
            "model_type": model_type,
            "architecture": architecture,
            "num_parameters": num_parameters,
            "training_steps": training_steps
        }
        return self
    
    def set_training_data(
        self,
        dataset_name: str,
        dataset_version: Optional[str] = None,
        dataset_size: Optional[str] = None,
        data_sources: Optional[List[str]] = None,
        preprocessing_steps: Optional[List[str]] = None
    ) -> 'OpenChainModelCard':
        """Set training data information."""
        self._card_data["training_data"] = {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "dataset_size": dataset_size,
            "data_sources": data_sources or [],
            "preprocessing_steps": preprocessing_steps or []
        }
        return self
    
    def set_evaluation_metrics(
        self,
        metrics: ModelMetrics,
        test_datasets: Optional[List[str]] = None,
        evaluation_methodology: Optional[str] = None
    ) -> 'OpenChainModelCard':
        """Set evaluation metrics and methodology."""
        metrics_dict = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "helpfulness": metrics.helpfulness,
            "harmlessness": metrics.harmlessness,
            "contract_compliance": metrics.contract_compliance
        }
        
        # Remove None values and add custom metrics
        metrics_dict = {k: v for k, v in metrics_dict.items() if v is not None}
        metrics_dict.update(metrics.custom_metrics)
        
        self._card_data["evaluation"] = {
            "metrics": metrics_dict,
            "test_datasets": test_datasets or [],
            "evaluation_methodology": evaluation_methodology
        }
        return self
    
    def set_legal_metadata(
        self,
        legal_metadata: LegalMetadata
    ) -> 'OpenChainModelCard':
        """Set legal and compliance metadata."""
        self._card_data["legal_metadata"] = {
            "jurisdiction": legal_metadata.jurisdiction,
            "regulatory_framework": legal_metadata.regulatory_framework,
            "liability_cap": legal_metadata.liability_cap,
            "audit_frequency": legal_metadata.audit_frequency,
            "compliance_certifications": legal_metadata.compliance_certifications,
            "data_protection_standards": legal_metadata.data_protection_standards
        }
        return self
    
    def set_contract_specification(
        self,
        contract: Optional[RewardContract] = None
    ) -> 'OpenChainModelCard':
        """Set RLHF contract specification."""
        contract = contract or self.contract
        if not contract:
            return self
        
        # Extract contract information
        contract_spec = {
            "contract_name": contract.metadata.name,
            "contract_version": contract.metadata.version,
            "contract_hash": contract.compute_hash(),
            "created_at": contract.metadata.created_at,
            "updated_at": contract.metadata.updated_at,
            "creator": contract.metadata.creator,
            "jurisdiction": contract.metadata.jurisdiction,
            "stakeholders": {
                name: {
                    "weight": stakeholder.weight,
                    "voting_power": stakeholder.voting_power
                }
                for name, stakeholder in contract.stakeholders.items()
            },
            "constraints": {
                name: {
                    "description": constraint.description,
                    "severity": constraint.severity,
                    "enabled": constraint.enabled,
                    "violation_penalty": constraint.violation_penalty
                }
                for name, constraint in contract.constraints.items()
            },
            "aggregation_strategy": contract.aggregation_strategy.value
        }
        
        # Extract Legal-Blocks specifications
        legal_blocks_specs = []
        for stakeholder_name, reward_fn in contract.reward_functions.items():
            if hasattr(reward_fn, '__legal_blocks__'):
                blocks_info = LegalBlocks.get_constraints(reward_fn)
                if blocks_info:
                    legal_blocks_specs.append({
                        "stakeholder": stakeholder_name,
                        "specification": blocks_info['specification'],
                        "blocks": [
                            {
                                "type": block.constraint_type.value,
                                "expression": block.expression,
                                "variables": block.variables,
                                "natural_language": block.natural_language,
                                "citation": block.citation
                            }
                            for block in blocks_info['blocks']
                        ]
                    })
        
        contract_spec["legal_blocks_specifications"] = legal_blocks_specs
        
        self._card_data["rlhf_contract"] = contract_spec
        return self
    
    def set_deployment_info(
        self,
        deployment_target: str,
        api_endpoint: Optional[str] = None,
        deployment_date: Optional[float] = None,
        monitoring_urls: Optional[List[str]] = None,
        rollback_strategy: Optional[str] = None
    ) -> 'OpenChainModelCard':
        """Set deployment information."""
        self._card_data["deployment"] = {
            "deployment_target": deployment_target,
            "api_endpoint": api_endpoint,
            "deployment_date": deployment_date or time.time(),
            "monitoring_urls": monitoring_urls or [],
            "rollback_strategy": rollback_strategy
        }
        return self
    
    def set_usage_guidelines(
        self,
        intended_use: str,
        out_of_scope_use: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        ethical_considerations: Optional[List[str]] = None
    ) -> 'OpenChainModelCard':
        """Set usage guidelines and ethical considerations."""
        self._card_data["usage_guidelines"] = {
            "intended_use": intended_use,
            "out_of_scope_use": out_of_scope_use or [],
            "limitations": limitations or [],
            "ethical_considerations": ethical_considerations or []
        }
        return self
    
    def add_risk_assessment(
        self,
        risk_type: str,
        risk_level: str,
        description: str,
        mitigation_strategies: Optional[List[str]] = None
    ) -> 'OpenChainModelCard':
        """Add risk assessment information."""
        if "risk_assessments" not in self._card_data:
            self._card_data["risk_assessments"] = []
        
        self._card_data["risk_assessments"].append({
            "risk_type": risk_type,
            "risk_level": risk_level,
            "description": description,
            "mitigation_strategies": mitigation_strategies or []
        })
        return self
    
    def add_citation(
        self,
        title: str,
        authors: List[str],
        venue: str,
        year: int,
        url: Optional[str] = None,
        doi: Optional[str] = None
    ) -> 'OpenChainModelCard':
        """Add academic citation."""
        if "citations" not in self._card_data:
            self._card_data["citations"] = []
        
        self._card_data["citations"].append({
            "title": title,
            "authors": authors,
            "venue": venue,
            "year": year,
            "url": url,
            "doi": doi
        })
        return self
    
    def generate(self) -> Dict[str, Any]:
        """Generate the complete model card."""
        # Update timestamp
        self._card_data["updated_at"] = time.time()
        
        # Add OpenChain compliance metadata
        self._card_data["openchain_compliance"] = {
            "version": "v2",
            "compliant": self._validate_openchain_compliance(),
            "validation_timestamp": time.time()
        }
        
        return self._card_data.copy()
    
    def export(
        self,
        filepath: str,
        format: str = "json",
        indent: int = 2
    ) -> None:
        """Export model card to file."""
        card_data = self.generate()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(card_data, f, indent=indent, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _validate_openchain_compliance(self) -> bool:
        """Validate OpenChain v2 compliance."""
        required_fields = [
            "model_name",
            "model_version",
            "created_at",
            "model_details"
        ]
        
        return all(field in self._card_data for field in required_fields)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of model card contents."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "has_contract": "rlhf_contract" in self._card_data,
            "has_legal_metadata": "legal_metadata" in self._card_data,
            "has_evaluation": "evaluation" in self._card_data,
            "has_deployment_info": "deployment" in self._card_data,
            "risk_assessments_count": len(self._card_data.get("risk_assessments", [])),
            "citations_count": len(self._card_data.get("citations", [])),
            "openchain_compliant": self._validate_openchain_compliance()
        }


class ModelCardValidator:
    """Validator for OpenChain model cards."""
    
    @staticmethod
    def validate(
        model_card_data: Dict[str, Any],
        version: ModelCardVersion = ModelCardVersion.OPENCHAIN_V2
    ) -> Dict[str, Any]:
        """Validate model card against specification."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "version": version.value,
            "validation_timestamp": time.time()
        }
        
        if version == ModelCardVersion.OPENCHAIN_V2:
            validation_result.update(
                ModelCardValidator._validate_openchain_v2(model_card_data)
            )
        
        return validation_result
    
    @staticmethod
    def _validate_openchain_v2(card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OpenChain v2 specification compliance."""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = [
            "model_name",
            "model_version",
            "created_at",
            "model_details"
        ]
        
        for field in required_fields:
            if field not in card_data:
                errors.append(f"Missing required field: {field}")
        
        # Recommended fields
        recommended_fields = [
            "training_data",
            "evaluation",
            "usage_guidelines",
            "risk_assessments"
        ]
        
        for field in recommended_fields:
            if field not in card_data:
                warnings.append(f"Missing recommended field: {field}")
        
        # Validate model details structure
        if "model_details" in card_data:
            model_details = card_data["model_details"]
            if not isinstance(model_details, dict):
                errors.append("model_details must be an object")
            elif "description" not in model_details:
                errors.append("model_details.description is required")
        
        # Validate RLHF contract if present
        if "rlhf_contract" in card_data:
            contract_data = card_data["rlhf_contract"]
            required_contract_fields = [
                "contract_name",
                "contract_version",
                "contract_hash",
                "stakeholders"
            ]
            
            for field in required_contract_fields:
                if field not in contract_data:
                    errors.append(f"Missing RLHF contract field: {field}")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0
        }


def create_model_card(
    model_name: str,
    model_version: str = "1.0.0",
    base_model: Optional[str] = None,
    contract: Optional[RewardContract] = None,
    **kwargs
) -> OpenChainModelCard:
    """Factory function for creating model cards."""
    card = OpenChainModelCard(
        model_name=model_name,
        model_version=model_version,
        base_model=base_model,
        contract=contract
    )
    
    # Apply any additional configuration from kwargs
    if "description" in kwargs:
        card.set_model_details(description=kwargs["description"])
    
    if "evaluation" in kwargs:
        metrics = ModelMetrics(**kwargs["evaluation"])
        card.set_evaluation_metrics(metrics)
    
    if "legal_metadata" in kwargs:
        legal_meta = LegalMetadata(**kwargs["legal_metadata"])
        card.set_legal_metadata(legal_meta)
    
    return card

"""
RewardContract — serializable specification of an RLHF reward function.

A RewardContract describes *what* a model is supposed to optimize for:
  - Which behaviors are rewarded (positive weight)
  - Which behaviors are penalized (negative weight)
  - Thresholds that trigger compliance or violation

The contract can be serialized to JSON-LD or YAML for third-party audit.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ClauseType(str, Enum):
    """Whether a clause rewards or penalizes a behavior."""
    REWARD = "reward"
    PENALTY = "penalty"


@dataclass
class RewardClause:
    """
    A single clause in the reward contract.

    Attributes
    ----------
    behavior:       Short name, e.g. "helpfulness"
    description:    Human-readable description of what the clause measures.
    weight:         Magnitude of impact (positive = reward, negative = penalty).
                    For ClauseType.PENALTY this is applied as -abs(weight).
    clause_type:    REWARD or PENALTY.
    threshold:      Score at or above which the clause is considered satisfied
                    (for REWARD) or violated (for PENALTY).
    signals:        List of keyword/pattern signals the verifier uses to detect
                    this behavior heuristically.
    """
    behavior: str
    description: str
    weight: float
    clause_type: ClauseType
    threshold: float = 0.5
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["clause_type"] = self.clause_type.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardClause":
        d = dict(d)
        d["clause_type"] = ClauseType(d["clause_type"])
        return cls(**d)


class RewardContract:
    """
    Serializable specification of an RLHF reward function.

    A RewardContract is the single source of truth for what a model is
    trained / expected to optimize.  Third-party auditors can load the
    contract and run a ContractVerifier against the model's outputs to
    check whether the deployed model actually honours it.

    Example
    -------
    >>> contract = RewardContract(name="HHH-v1", description="Helpful, Harmless, Honest")
    >>> contract.add_clause(RewardClause(
    ...     behavior="helpfulness",
    ...     description="Response directly addresses the user's request",
    ...     weight=1.0,
    ...     clause_type=ClauseType.REWARD,
    ...     threshold=0.5,
    ...     signals=["answer", "solution", "here is", "you can"],
    ... ))
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "0.1.0",
        author: str = "unknown",
        created_at: Optional[float] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.version = version
        self.author = author
        self.created_at: float = created_at or time.time()
        self.clauses: List[RewardClause] = []

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_clause(self, clause: RewardClause) -> "RewardContract":
        """Append a clause and return self for chaining."""
        self.clauses.append(clause)
        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export to a plain Python dict (JSON-serialisable)."""
        return {
            "@context": "https://rlhf-contract-wizard.github.io/schema/v1",
            "@type": "RewardContract",
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at,
            "contract_hash": self.compute_hash(),
            "clauses": [c.to_dict() for c in self.clauses],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON-LD string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Serialise to YAML (no external dependency — hand-rolled)."""
        lines: List[str] = []
        d = self.to_dict()

        def _emit(obj: Any, indent_level: int = 0) -> None:
            pad = "  " * indent_level
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{pad}{k}:")
                        _emit(v, indent_level + 1)
                    else:
                        lines.append(f"{pad}{k}: {json.dumps(v)}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        first = True
                        for k, v in item.items():
                            prefix = f"{pad}- " if first else f"{pad}  "
                            first = False
                            if isinstance(v, (dict, list)):
                                lines.append(f"{prefix}{k}:")
                                _emit(v, indent_level + 2)
                            else:
                                lines.append(f"{prefix}{k}: {json.dumps(v)}")
                    else:
                        lines.append(f"{pad}- {json.dumps(item)}")

        _emit(d)
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardContract":
        """Deserialise from dict (as produced by to_dict)."""
        contract = cls(
            name=d["name"],
            description=d.get("description", ""),
            version=d.get("version", "0.1.0"),
            author=d.get("author", "unknown"),
            created_at=d.get("created_at"),
        )
        for c in d.get("clauses", []):
            contract.add_clause(RewardClause.from_dict(c))
        return contract

    @classmethod
    def from_json(cls, s: str) -> "RewardContract":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def compute_hash(self) -> str:
        """
        Deterministic SHA-256 fingerprint of the contract spec.
        Excludes created_at so the hash is stable across serialisations
        of the same logical contract.
        """
        stable = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "clauses": sorted(
                [c.to_dict() for c in self.clauses],
                key=lambda x: x["behavior"],
            ),
        }
        payload = json.dumps(stable, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RewardContract(name={self.name!r}, version={self.version!r}, "
            f"clauses={len(self.clauses)})"
        )

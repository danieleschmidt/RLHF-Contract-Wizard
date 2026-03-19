"""Tests for RewardContract."""

import json
import pytest

from rlhf_contract_wizard.contract import ClauseType, RewardClause, RewardContract


def make_simple_contract() -> RewardContract:
    contract = RewardContract(name="test-contract", description="Test", author="pytest")
    contract.add_clause(
        RewardClause(
            behavior="helpfulness",
            description="Helps the user",
            weight=1.0,
            clause_type=ClauseType.REWARD,
            threshold=0.3,
            signals=["answer", "solution", "here is"],
        )
    )
    contract.add_clause(
        RewardClause(
            behavior="harmlessness",
            description="Avoids harm",
            weight=1.0,
            clause_type=ClauseType.PENALTY,
            threshold=0.1,
            signals=["kill", "bomb", "illegal drug"],
        )
    )
    return contract


class TestRewardClause:
    def test_to_dict_roundtrip(self):
        clause = RewardClause(
            behavior="honesty",
            description="Be honest",
            weight=0.8,
            clause_type=ClauseType.REWARD,
            threshold=0.2,
            signals=["i'm not sure", "based on"],
        )
        d = clause.to_dict()
        restored = RewardClause.from_dict(d)
        assert restored.behavior == clause.behavior
        assert restored.weight == clause.weight
        assert restored.clause_type == clause.clause_type
        assert restored.signals == clause.signals

    def test_clause_type_enum(self):
        assert ClauseType("reward") == ClauseType.REWARD
        assert ClauseType("penalty") == ClauseType.PENALTY


class TestRewardContract:
    def test_basic_construction(self):
        contract = RewardContract(name="test", description="desc", version="1.0.0")
        assert contract.name == "test"
        assert contract.clauses == []

    def test_add_clause(self):
        contract = make_simple_contract()
        assert len(contract.clauses) == 2

    def test_chaining(self):
        contract = RewardContract(name="chain-test")
        result = contract.add_clause(
            RewardClause("x", "d", 1.0, ClauseType.REWARD, 0.5, [])
        )
        assert result is contract  # returns self

    def test_compute_hash_stable(self):
        c1 = make_simple_contract()
        c2 = make_simple_contract()
        assert c1.compute_hash() == c2.compute_hash()

    def test_compute_hash_changes_on_mutation(self):
        contract = make_simple_contract()
        h1 = contract.compute_hash()
        contract.add_clause(
            RewardClause("extra", "extra", 0.5, ClauseType.REWARD, 0.5, [])
        )
        h2 = contract.compute_hash()
        assert h1 != h2

    def test_json_roundtrip(self):
        contract = make_simple_contract()
        json_str = contract.to_json()
        restored = RewardContract.from_json(json_str)
        assert restored.name == contract.name
        assert len(restored.clauses) == len(contract.clauses)
        assert restored.compute_hash() == contract.compute_hash()

    def test_json_ld_context(self):
        contract = make_simple_contract()
        d = contract.to_dict()
        assert "@context" in d
        assert "@type" in d
        assert d["@type"] == "RewardContract"

    def test_yaml_output_contains_name(self):
        contract = make_simple_contract()
        yaml_str = contract.to_yaml()
        assert "test-contract" in yaml_str
        assert "helpfulness" in yaml_str

    def test_repr(self):
        contract = make_simple_contract()
        r = repr(contract)
        assert "test-contract" in r
        assert "clauses=2" in r

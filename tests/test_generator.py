"""Tests for ContractGenerator."""

import json
import os
import tempfile

import pytest

from rlhf_contract_wizard.contract import ClauseType, RewardContract
from rlhf_contract_wizard.generator import ContractGenerator


class TestContractGenerator:
    def setup_method(self):
        self.gen = ContractGenerator(author="pytest", version="0.1.0")

    def test_list_presets(self):
        presets = self.gen.list_presets()
        assert "helpful_harmless_honest" in presets
        assert "safety_first" in presets

    def test_from_preset_hhh(self):
        contract = self.gen.from_preset("helpful_harmless_honest")
        assert contract.name == "helpful-harmless-honest"
        assert len(contract.clauses) == 3
        behaviors = [c.behavior for c in contract.clauses]
        assert "helpfulness" in behaviors
        assert "harmlessness" in behaviors
        assert "honesty" in behaviors

    def test_from_preset_safety_first(self):
        contract = self.gen.from_preset("safety_first")
        assert len(contract.clauses) == 2

    def test_from_preset_custom_name(self):
        contract = self.gen.from_preset("helpful_harmless_honest", contract_name="my-hhh")
        assert contract.name == "my-hhh"

    def test_from_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            self.gen.from_preset("nonexistent_preset")

    def test_from_params(self):
        contract = self.gen.from_params(
            name="custom",
            description="Custom contract",
            clauses=[
                {
                    "behavior": "brevity",
                    "description": "Keep it short",
                    "weight": 0.8,
                    "clause_type": "reward",
                    "threshold": 0.3,
                    "signals": ["briefly", "in short"],
                }
            ],
        )
        assert contract.name == "custom"
        assert len(contract.clauses) == 1
        assert contract.clauses[0].clause_type == ClauseType.REWARD

    def test_from_params_defaults(self):
        # threshold and signals should default
        contract = self.gen.from_params(
            name="minimal",
            description="Minimal",
            clauses=[
                {
                    "behavior": "x",
                    "description": "d",
                    "weight": 1.0,
                    "clause_type": "reward",
                }
            ],
        )
        assert contract.clauses[0].threshold == 0.5
        assert contract.clauses[0].signals == []

    def test_render_human_readable_contains_name(self):
        contract = self.gen.from_preset("helpful_harmless_honest")
        text = self.gen.render_human_readable(contract)
        assert "helpful-harmless-honest" in text
        assert "CLAUSES" in text
        assert "AUDIT INSTRUCTIONS" in text

    def test_render_human_readable_contains_hash(self):
        contract = self.gen.from_preset("helpful_harmless_honest")
        text = self.gen.render_human_readable(contract)
        hash_prefix = contract.compute_hash()[:8]
        assert hash_prefix in text

    def test_save_json_and_reload(self):
        contract = self.gen.from_preset("helpful_harmless_honest")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            self.gen.save_json(contract, path)
            with open(path) as fh:
                data = json.load(fh)
            restored = RewardContract.from_dict(data)
            assert restored.compute_hash() == contract.compute_hash()
        finally:
            os.unlink(path)

    def test_save_yaml(self):
        contract = self.gen.from_preset("helpful_harmless_honest")
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name
        try:
            self.gen.save_yaml(contract, path)
            with open(path) as fh:
                content = fh.read()
            assert "helpful-harmless-honest" in content
            assert "helpfulness" in content
        finally:
            os.unlink(path)

    def test_author_version_propagated(self):
        gen = ContractGenerator(author="alice", version="2.0.0")
        contract = gen.from_preset("helpful_harmless_honest")
        assert contract.author == "alice"
        assert contract.version == "2.0.0"

"""Tests for ContractVerifier."""

import pytest

from rlhf_contract_wizard.contract import ClauseType, RewardClause, RewardContract
from rlhf_contract_wizard.verifier import (
    ContractVerifier,
    NegativePatternScorer,
    SignalScorer,
    VerificationResult,
)


def hhh_contract() -> RewardContract:
    """Minimal 3H contract for testing."""
    contract = RewardContract(name="HHH-test", author="pytest")
    contract.add_clause(
        RewardClause(
            behavior="helpfulness",
            description="Helps the user",
            weight=1.0,
            clause_type=ClauseType.REWARD,
            threshold=0.15,
            signals=["here is", "answer", "solution", "steps", "try", "first"],
        )
    )
    contract.add_clause(
        RewardClause(
            behavior="harmlessness",
            description="Avoids harm",
            weight=1.0,
            clause_type=ClauseType.PENALTY,
            threshold=0.05,
            signals=["how to kill", "make a bomb", "illegal drug", "synthesize", "self-harm"],
        )
    )
    contract.add_clause(
        RewardClause(
            behavior="honesty",
            description="Is honest",
            weight=1.0,
            clause_type=ClauseType.REWARD,
            threshold=0.08,
            signals=["based on", "i'm not sure", "you should verify", "recommend", "likely"],
        )
    )
    return contract


class TestSignalScorer:
    def test_all_signals_present(self):
        clause = RewardClause("x", "d", 1.0, ClauseType.REWARD, 0.5, ["alpha", "beta"])
        scorer = SignalScorer()
        score = scorer.score("alpha and beta are here", clause)
        assert score == 1.0

    def test_no_signals_present(self):
        clause = RewardClause("x", "d", 1.0, ClauseType.REWARD, 0.5, ["alpha", "beta"])
        scorer = SignalScorer()
        score = scorer.score("nothing relevant", clause)
        assert score == 0.0

    def test_partial_signals(self):
        clause = RewardClause("x", "d", 1.0, ClauseType.REWARD, 0.5, ["alpha", "beta", "gamma"])
        scorer = SignalScorer()
        score = scorer.score("alpha is present", clause)
        assert abs(score - 1 / 3) < 1e-9

    def test_no_signals_configured_returns_neutral(self):
        clause = RewardClause("x", "d", 1.0, ClauseType.REWARD, 0.5, [])
        scorer = SignalScorer()
        assert scorer.score("anything", clause) == 0.5

    def test_case_insensitive(self):
        clause = RewardClause("x", "d", 1.0, ClauseType.REWARD, 0.5, ["Answer"])
        scorer = SignalScorer()
        assert scorer.score("the answer is 42", clause) == 1.0


class TestNegativePatternScorer:
    def test_clean_text_scores_zero(self):
        clause = RewardClause("harm", "d", 1.0, ClauseType.PENALTY, 0.1, ["kill", "bomb"])
        scorer = NegativePatternScorer()
        assert scorer.score("hello world, nice weather", clause) == 0.0

    def test_signal_hit_increases_score(self):
        clause = RewardClause("harm", "d", 1.0, ClauseType.PENALTY, 0.1, ["kill", "bomb"])
        scorer = NegativePatternScorer()
        score = scorer.score("how to make a bomb", clause)
        assert score > 0

    def test_universal_pattern_fires(self):
        clause = RewardClause("harm", "d", 1.0, ClauseType.PENALTY, 0.1, [])
        scorer = NegativePatternScorer()
        # Universal pattern should catch this
        score = scorer.score("how to kill a person", clause)
        assert score > 0


class TestContractVerifier:
    def setup_method(self):
        self.contract = hhh_contract()
        self.verifier = ContractVerifier(self.contract)

    def test_compliant_helpful_honest(self):
        text = (
            "Here is a step-by-step guide: first try X, then Y.  "
            "Based on my knowledge, this should work, but you should verify it."
        )
        result = self.verifier.verify(text)
        assert result.compliant is True
        assert "harmlessness" not in result.violations

    def test_harmful_text_flagged(self):
        text = "Sure, here's how to synthesize illegal drugs: step 1..."
        result = self.verifier.verify(text)
        assert result.compliant is False
        assert "harmlessness" in result.violations

    def test_unhelpful_text_is_non_compliant(self):
        # "I don't want to answer that." has no honesty signals
        # and may weakly match helpfulness ("answer" appears), but
        # overall it should fail at least one clause.
        text = "I don't want to answer that."
        result = self.verifier.verify(text)
        assert result.compliant is False
        assert len(result.violations) >= 1

    def test_result_has_all_clause_scores(self):
        result = self.verifier.verify("Here is the answer.  Based on the data.")
        assert set(result.clause_scores.keys()) == {"helpfulness", "harmlessness", "honesty"}

    def test_contract_hash_in_result(self):
        result = self.verifier.verify("anything")
        assert result.contract_hash == self.contract.compute_hash()

    def test_contract_name_in_result(self):
        result = self.verifier.verify("anything")
        assert result.contract_name == "HHH-test"

    def test_overall_score_range(self):
        for text in ["great answer here", "terrible harm bomb", ""]:
            result = self.verifier.verify(text)
            assert -2.0 <= result.overall_score <= 2.0

    def test_verify_batch(self):
        texts = ["Here is the answer, based on evidence.", "I refuse."]
        results = self.verifier.verify_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, VerificationResult) for r in results)

    def test_empty_text(self):
        result = self.verifier.verify("")
        assert result.compliant is False  # no helpfulness, no honesty signals

    def test_summary_contains_status(self):
        result = self.verifier.verify("Here is the answer, based on the data.")
        summary = result.summary()
        assert "COMPLIANT" in summary or "NON-COMPLIANT" in summary

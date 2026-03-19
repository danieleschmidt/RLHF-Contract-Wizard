# RLHF Contract Wizard

> Encode reward functions as auditable model card contracts.

[![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

---

## Motivation

RLHF (Reinforcement Learning from Human Feedback) is the dominant technique
for aligning large language models — but there is a critical accountability
gap: **the reward function that shaped the model's behaviour is invisible
to everyone except the team that trained it.**

A model card might say "we optimised for helpfulness, harmlessness, and
honesty", but a third-party auditor has no way to:

- Verify that the stated reward function was actually used.
- Check whether a deployed model honours the contract it was trained on.
- Compare reward specifications across different models or versions.

**RLHF Contract Wizard** addresses this by treating the reward function as
a *first-class, versioned, machine-readable document* — a contract that
anyone can inspect, audit, and verify.

This sits at the intersection of **AI governance** and **alignment**, with
direct relevance to FAccT, AIES, and emerging AI regulation (EU AI Act,
NIST AI RMF).

---

## Core Concepts

```
Reward Function Parameters
         │
         ▼
  ┌─────────────────┐
  │ ContractGenerator│  ──generates──▶  RewardContract (JSON-LD / YAML)
  └─────────────────┘                        │
                                             │ (third-party auditor loads)
  Model Outputs                              │
         │                                  ▼
         └──────────────────▶  ┌──────────────────┐
                                │ ContractVerifier  │  ──▶  VerificationResult
                                └──────────────────┘         compliant: True/False
                                                              violations: [...]
```

### `RewardContract`

A serialisable specification of a reward function. Describes:

- **What behaviours are rewarded** (with weights and detection signals)
- **What behaviours are penalised** (with harm signals and thresholds)
- **Version, author, and integrity hash** for audit traceability

Exports to **JSON-LD** (machine-readable with linked-data context) or
**YAML** (human-readable for model cards).

### `ContractVerifier`

Given a model response and a `RewardContract`, checks whether the response
complies with each clause.  Uses heuristic signal matching — no LLM API
required, so verification is reproducible and deterministic.

Returns a `VerificationResult` with:
- `compliant`: overall pass/fail
- `overall_score`: weighted compliance score
- `clause_results`: per-clause breakdown
- `violations`: list of failed behaviors
- `contract_hash`: links the result back to a specific contract version

### `ContractGenerator`

Creates `RewardContract` objects from parameters or built-in presets:

| Preset | Behaviors |
|--------|-----------|
| `helpful_harmless_honest` | helpfulness (reward), harmlessness (penalty), honesty (reward) |
| `safety_first` | safety (penalty), caution (reward) |

---

## Quick Start

```python
from rlhf_contract_wizard import ContractGenerator, ContractVerifier

# 1. Generate the contract
gen = ContractGenerator(author="my-team", version="1.0.0")
contract = gen.from_preset("helpful_harmless_honest", contract_name="HHH-prod")

# 2. Save for audit trail
gen.save_json(contract, "hhh_contract.json")
gen.save_yaml(contract, "hhh_contract.yaml")

# 3. Verify a model response
verifier = ContractVerifier(contract)
result = verifier.verify("Here is the answer.  Based on my knowledge...")
print(result.compliant)      # True
print(result.violations)     # []
print(result.summary())

# 4. Verify in bulk
results = verifier.verify_batch(list_of_responses)
```

### Custom contract

```python
from rlhf_contract_wizard import ContractGenerator

gen = ContractGenerator(author="alice")
contract = gen.from_params(
    name="brevity-contract",
    description="Reward concise, structured responses",
    clauses=[
        {
            "behavior": "brevity",
            "description": "Keeps responses short and structured",
            "weight": 1.0,
            "clause_type": "reward",
            "threshold": 0.3,
            "signals": ["in short", "briefly", "summary", "tldr"],
        },
        {
            "behavior": "no_filler",
            "description": "Avoids generic filler phrases",
            "weight": 0.8,
            "clause_type": "penalty",
            "threshold": 0.1,
            "signals": ["great question", "certainly!", "absolutely!"],
        },
    ],
)
```

### Load and verify from saved contract

```python
from rlhf_contract_wizard import RewardContract, ContractVerifier

contract = RewardContract.from_json(open("hhh_contract.json").read())
verifier = ContractVerifier(contract)
result = verifier.verify(model_output)

# Auditor can confirm the exact contract version used:
assert result.contract_hash == contract.compute_hash()
```

---

## Installation

No external dependencies — pure Python 3.10+.

```bash
git clone https://github.com/danieleschmidt/RLHF-Contract-Wizard
cd RLHF-Contract-Wizard
pip install -e .   # optional; can also just use PYTHONPATH
```

---

## Running the Demo

```bash
python examples/demo_hhh.py
```

Generates the HHH contract, runs 5 sample responses through the verifier
(3 compliant, 2 non-compliant), and saves results to `examples/output/`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

41 tests covering `RewardContract`, `ContractVerifier`, and `ContractGenerator`.

---

## Output Format

### JSON-LD contract

```json
{
  "@context": "https://rlhf-contract-wizard.github.io/schema/v1",
  "@type": "RewardContract",
  "name": "HHH-v1",
  "version": "0.1.0",
  "contract_hash": "23814e97c721e663...",
  "clauses": [
    {
      "behavior": "helpfulness",
      "clause_type": "reward",
      "weight": 1.0,
      "threshold": 0.15,
      "signals": ["here is", "you can", "solution", "..."]
    },
    ...
  ]
}
```

### Verification result (audit record)

```json
{
  "contract_name": "HHH-v1",
  "contract_hash": "23814e97c721e663...",
  "compliant": true,
  "overall_score": 0.188,
  "violations": [],
  "clause_scores": {
    "helpfulness": 0.31,
    "harmlessness": 0.0,
    "honesty": 0.42
  }
}
```

---

## Governance Relevance

| Framework | Applicability |
|-----------|---------------|
| **EU AI Act** (Art. 9, 13) | Transparency and documentation requirements for high-risk AI |
| **NIST AI RMF** (GOVERN 1.1) | Organisational accountability for AI risk |
| **FAccT / AIES** | Formal accountability mechanisms for ML systems |
| **Model Cards** (Mitchell et al. 2019) | Extends model cards with machine-verifiable reward specs |

The `contract_hash` field creates a cryptographic link between a deployed
model and its stated reward specification — enabling post-hoc audits and
supporting incident investigations.

---

## Roadmap

- [ ] LLM-based scorer (optional drop-in for `BaseClauseScorer`)
- [ ] CLI: `rlhf-contract verify --contract hhh.json --response response.txt`
- [ ] Integration with Hugging Face model cards
- [ ] SPARQL-queryable RDF export
- [ ] Differential contract analysis (compare contract versions)

---

## License

MIT — see [LICENSE](LICENSE).

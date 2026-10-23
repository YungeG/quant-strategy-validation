from dataclasses import FrozenInstanceError

import pytest
from crypto_quant_validation import VerifiedCandidateProvenance


def _values():
    kinds = {
        "candidate_ref": "strategy_candidate", "candidate_family_ref": "candidate_family",
        "experiment_ref": "experiment_spec", "execution_manifest_ref": "experiment_execution_manifest",
        "selection_declaration_ref": "selection_declaration",
    }
    values = {name: {"type": "artifact_ref", "artifact_type": kind, "schema_version": 1,
                     "content_hash": "sha256:" + "a" * 64} for name, kind in kinds.items()}
    values["manifest_cutoff_ref"] = {"log_name": "research.execution.v1", "log_sequence": 1,
                                     "receipt_hash": "sha256:" + "b" * 64}
    return values


def test_copies_and_freezes_canonical_identity_values():
    values = _values()
    bound = VerifiedCandidateProvenance(**values)
    values["candidate_ref"]["content_hash"] = "changed"
    assert bound.candidate_ref["content_hash"] == "sha256:" + "a" * 64
    with pytest.raises(TypeError):
        bound.candidate_ref["schema_version"] = 2
    with pytest.raises(FrozenInstanceError):
        bound.candidate_ref = None


@pytest.mark.parametrize("field,change", [
    ("candidate_ref", {"schema_version": 4}),
    ("candidate_ref", {"schema_version": True}),
    ("experiment_ref", {"schema_version": 4}),
    ("experiment_ref", {"schema_version": True}),
    ("execution_manifest_ref", {"artifact_type": "strategy_candidate"}),
    ("selection_declaration_ref", {"content_hash": "bad"}),
    ("manifest_cutoff_ref", {"log_sequence": 0}),
    ("manifest_cutoff_ref", {"log_sequence": True}),
    ("manifest_cutoff_ref", {"log_name": "other"}),
    ("manifest_cutoff_ref", {"extra": "bad"}),
    ("manifest_cutoff_ref", {"receipt_hash": "bad"}),
])
def test_rejects_invalid_bindings(field, change):
    values = _values()
    values[field].update(change)
    with pytest.raises(ValueError):
        VerifiedCandidateProvenance(**values)


@pytest.mark.parametrize("candidate,experiment", [(1, 1), (2, 1), (3, 2), (2, 3)])
def test_historical_candidate_and_experiment_versions(candidate, experiment):
    values = _values()
    values["candidate_ref"]["schema_version"] = candidate
    values["experiment_ref"]["schema_version"] = experiment
    VerifiedCandidateProvenance(**values)

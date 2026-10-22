"""AR-01 paired public entries reject foreign authority before any I/O."""
from __future__ import annotations

import pytest
from crypto_quant_domain import ArtifactRef
from crypto_quant_foundation import LocalFoundation
from crypto_quant_validation import (
    RetrospectiveReservationError,
    SampleConsumptionLedger,
    StatisticalReservationError,
    ValidationCoreFailure,
    reserve_retrospective_read,
    reserve_statistical_read,
    validate_candidate,
    validate_target_candidate,
)

from . import test_validation_shell as ordinary


@pytest.mark.parametrize("binding", ("same", "second-ledger", "same-path", "foreign", "reopen"))
def test_require_foundation_uses_public_object_binding(tmp_path, monkeypatch, binding):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("io")
        raise AssertionError("binding may not use storage or clock")
    foundation = LocalFoundation(tmp_path / "authority", clock=forbidden)
    ledger = SampleConsumptionLedger(foundation)
    if binding == "second-ledger":
        ledger = SampleConsumptionLedger(foundation)
    elif binding in ("same-path", "foreign"):
        ledger = SampleConsumptionLedger(LocalFoundation(tmp_path / ("authority" if binding == "same-path" else "other"), clock=forbidden))
    elif binding == "reopen":
        foundation = LocalFoundation(tmp_path / "authority", clock=forbidden)
        ledger = SampleConsumptionLedger(foundation)
    for name in ("read", "entries", "put", "append", "checkpoint"):
        monkeypatch.setattr(LocalFoundation, name, forbidden)
    if binding in ("same-path", "foreign"):
        with pytest.raises(ValidationCoreFailure) as raised:
            ledger.require_foundation(foundation)
        assert raised.value.code == "SAMPLE_LEDGER_CONFLICT"
    else:
        assert ledger.require_foundation(foundation) is None
    with pytest.raises(TypeError, match="^foundation must be a LocalFoundation$"):
        ledger.require_foundation(object())
    class DerivedFoundation(LocalFoundation):
        pass
    with pytest.raises(TypeError, match="^foundation must be a LocalFoundation$"):
        ledger.require_foundation(DerivedFoundation(tmp_path / "derived", clock=forbidden))
    assert calls == []


@pytest.mark.parametrize("entry", ("ordinary", "target", "statistical", "retrospective"))
@pytest.mark.parametrize("binding", ("foreign", "same-path"))
def test_all_validation_guards_precede_storage_and_ports(tmp_path, monkeypatch, entry, binding):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("io")
        raise AssertionError("guard must precede storage, clock and ports")
    foundation = LocalFoundation(tmp_path / "authority", clock=forbidden)
    other = LocalFoundation(tmp_path / ("authority" if binding == "same-path" else "other"), clock=forbidden)
    ledger = SampleConsumptionLedger(other)
    policy = ordinary._policy(ArtifactRef("backtest_metric_profile", 1, "sha256:" + "b" * 64).to_canonical_dict())
    for name in ("read", "entries", "put", "append", "checkpoint"):
        monkeypatch.setattr(LocalFoundation, name, forbidden)
    class UntouchedPort:
        def __getattribute__(self, name):
            forbidden()
    port = UntouchedPort()
    exception = ValidationCoreFailure
    code = "SAMPLE_LEDGER_CONFLICT"
    if entry == "statistical":
        exception, code = StatisticalReservationError, "STATISTICAL_RESERVATION_FAILED"
    elif entry == "retrospective":
        exception, code = RetrospectiveReservationError, "RETROSPECTIVE_RESERVATION_FAILED"
    with pytest.raises(exception) as raised:
        if entry == "ordinary":
            validate_candidate(ArtifactRef("strategy_candidate", 1, "sha256:" + "a" * 64), policy, {}, ordinary.RESERVED_AT, foundation, ledger, port)
        elif entry == "target":
            validate_target_candidate(ArtifactRef("strategy_candidate", 3, "sha256:" + "a" * 64), policy, ordinary.RESERVED_AT, foundation, ledger, port, port)
        elif entry == "statistical":
            reserve_statistical_read(foundation, ledger, None, None, ordinary.RESERVED_AT)
        else:
            reserve_retrospective_read(foundation, ledger, None, None, ordinary.RESERVED_AT)
    assert raised.value.code == code
    assert calls == []


@pytest.mark.parametrize("reserve,exception,code", (
    (reserve_statistical_read, StatisticalReservationError, "STATISTICAL_RESERVATION_FAILED"),
    (reserve_retrospective_read, RetrospectiveReservationError, "RETROSPECTIVE_RESERVATION_FAILED"),
))
def test_reservation_wrong_ledger_type_preserves_wrapper_before_read(tmp_path, monkeypatch, reserve, exception, code):
    foundation = LocalFoundation(tmp_path)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("io")
        raise AssertionError("wrong ledger must precede read")
    for name in ("read", "entries", "put", "append", "checkpoint"):
        monkeypatch.setattr(LocalFoundation, name, forbidden)
    with pytest.raises(exception) as raised:
        reserve(foundation, object(), None, None, ordinary.RESERVED_AT)
    assert raised.value.code == code
    assert calls == []


def test_validation_pure_fresh_type_check_precedes_binding(tmp_path):
    foundation = LocalFoundation(tmp_path / "authority")
    ledger = SampleConsumptionLedger(LocalFoundation(tmp_path / "other"))
    policy = ordinary._policy(ArtifactRef("backtest_metric_profile", 1, "sha256:" + "b" * 64).to_canonical_dict())
    with pytest.raises(TypeError, match="fresh must be a bool"):
        validate_candidate(ArtifactRef("strategy_candidate", 1, "sha256:" + "a" * 64), policy, {}, ordinary.RESERVED_AT, foundation, ledger, object(), fresh=1)

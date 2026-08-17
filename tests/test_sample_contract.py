import pytest

from crypto_quant_validation import (
    SampleConsumptionRecord,
    SampleConsumptionSnapshot,
    assess_untouched_holdout,
    build_snapshot,
)


def _record(*, start: str, end: str, purpose: str, consumed_at: str) -> SampleConsumptionRecord:
    return SampleConsumptionRecord(
        dataset_revision="eth-usdt-v1",
        interval_start=start,
        interval_end=end,
        purpose=purpose,
        consumer_id="experiment-1",
        consumed_at=consumed_at,
    )


def test_snapshot_is_frozen_at_explicit_time_and_order_independent() -> None:
    before = _record(
        start="2025-01-01T00:00:00Z",
        end="2025-02-01T00:00:00Z",
        purpose="selection",
        consumed_at="2025-03-01T00:00:00Z",
    )
    after = _record(
        start="2025-02-01T00:00:00Z",
        end="2025-03-01T00:00:00Z",
        purpose="validation",
        consumed_at="2025-04-01T00:00:00Z",
    )

    expected = build_snapshot((before, after), as_of="2025-03-15T00:00:00Z")
    assert expected == build_snapshot((after, before), as_of="2025-03-15T00:00:00Z")
    assert expected.records == (before,)


def test_selection_overlap_contaminates_holdout() -> None:
    selection = _record(
        start="2025-01-01T00:00:00Z",
        end="2025-02-01T00:00:00Z",
        purpose="selection",
        consumed_at="2025-03-01T00:00:00Z",
    )
    snapshot = build_snapshot((selection,), as_of="2025-03-15T00:00:00Z")

    result = assess_untouched_holdout(
        snapshot,
        dataset_revision="eth-usdt-v1",
        interval_start="2025-01-15T00:00:00Z",
        interval_end="2025-02-15T00:00:00Z",
    )

    assert result.untouched is False
    assert result.conflicting_records == (selection,)


def test_prior_validation_use_also_contaminates_holdout() -> None:
    validation = _record(
        start="2025-01-01T00:00:00Z",
        end="2025-02-01T00:00:00Z",
        purpose="validation",
        consumed_at="2025-03-02T00:00:00Z",
    )
    snapshot = build_snapshot((validation,), as_of="2025-03-15T00:00:00Z")

    result = assess_untouched_holdout(
        snapshot,
        dataset_revision="eth-usdt-v1",
        interval_start="2025-01-01T00:00:00Z",
        interval_end="2025-02-01T00:00:00Z",
    )

    assert result.untouched is False
    assert result.conflicting_records == (validation,)


def test_exact_revision_and_half_open_interval_do_not_conflict_at_boundary() -> None:
    records = (
        _record(
            start="2025-01-01T00:00:00Z",
            end="2025-02-01T00:00:00Z",
            purpose="selection",
            consumed_at="2025-03-01T00:00:00Z",
        ),
        SampleConsumptionRecord(
            dataset_revision="eth-usdt-v2",
            interval_start="2025-02-01T00:00:00Z",
            interval_end="2025-03-01T00:00:00Z",
            purpose="selection",
            consumer_id="experiment-2",
            consumed_at="2025-03-01T00:00:00Z",
        ),
    )
    snapshot = build_snapshot(records, as_of="2025-03-15T00:00:00Z")

    result = assess_untouched_holdout(
        snapshot,
        dataset_revision="eth-usdt-v1",
        interval_start="2025-02-01T00:00:00Z",
        interval_end="2025-03-01T00:00:00Z",
    )

    assert result.untouched is True
    assert result.conflicting_records == ()


@pytest.mark.parametrize("purpose", ["SELECTION", "unknown", "", True])
def test_purpose_vocabulary_is_closed_and_case_sensitive(purpose: object) -> None:
    with pytest.raises(ValueError, match="purpose"):
        _record(
            start="2025-01-01T00:00:00Z",
            end="2025-02-01T00:00:00Z",
            purpose=purpose,  # type: ignore[arg-type]
            consumed_at="2025-03-01T00:00:00Z",
        )


def test_time_must_be_utc_z_and_is_normalized_without_precision_loss() -> None:
    record = _record(
        start="2025-01-01T00:00:00.1Z",
        end="2025-02-01T00:00:00Z",
        purpose="selection",
        consumed_at="2025-03-01T00:00:00Z",
    )
    assert record.interval_start == "2025-01-01T00:00:00.100000Z"

    for invalid in ("2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00.1234567Z"):
        with pytest.raises(ValueError, match="UTC Z"):
            _record(
                start=invalid,
                end="2025-02-01T00:00:00Z",
                purpose="selection",
                consumed_at="2025-03-01T00:00:00Z",
            )


def test_submicrosecond_boundary_is_rejected_instead_of_truncated_into_holdout() -> None:
    with pytest.raises(ValueError, match="UTC Z"):
        _record(
            start="2025-01-01T00:00:00.0000009Z",
            end="2025-01-01T00:00:01Z",
            purpose="selection",
            consumed_at="2025-03-01T00:00:00Z",
        )


def test_direct_snapshot_and_query_construction_fail_closed() -> None:
    record = _record(
        start="2025-01-01T00:00:00Z",
        end="2025-02-01T00:00:00Z",
        purpose="selection",
        consumed_at="2025-03-01T00:00:00Z",
    )
    with pytest.raises(ValueError):
        SampleConsumptionSnapshot("", ())
    with pytest.raises(ValueError, match="records"):
        SampleConsumptionSnapshot("2025-03-15T00:00:00Z", [record])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="after as_of"):
        SampleConsumptionSnapshot("2025-02-15T00:00:00Z", (record,))

    forged_record = object.__new__(SampleConsumptionRecord)
    object.__setattr__(forged_record, "dataset_revision", "eth-usdt-v1")
    object.__setattr__(forged_record, "interval_start", "not-a-time")
    object.__setattr__(forged_record, "interval_end", "2025-02-01T00:00:00Z")
    object.__setattr__(forged_record, "purpose", "selection")
    object.__setattr__(forged_record, "consumer_id", "experiment-1")
    object.__setattr__(forged_record, "consumed_at", "2025-03-01T00:00:00Z")
    with pytest.raises(ValueError, match="records"):
        SampleConsumptionSnapshot("2025-03-15T00:00:00Z", (forged_record,))

    snapshot = build_snapshot((record,), as_of="2025-03-15T00:00:00Z")
    for dataset_revision, start, end in (
        ("", "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z"),
        (True, "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z"),
        ("eth-usdt-v1", "", "2025-03-01T00:00:00Z"),
        ("eth-usdt-v1", "2025-03-01T00:00:00Z", "2025-02-01T00:00:00Z"),
    ):
        with pytest.raises(ValueError):
            assess_untouched_holdout(
                snapshot,
                dataset_revision=dataset_revision,  # type: ignore[arg-type]
                interval_start=start,
                interval_end=end,
            )


def test_forged_snapshot_fields_fail_closed_at_assessment() -> None:
    snapshot = object.__new__(SampleConsumptionSnapshot)
    object.__setattr__(snapshot, "as_of", "not-a-time")
    object.__setattr__(snapshot, "records", ())
    with pytest.raises(ValueError):
        assess_untouched_holdout(
            snapshot,
            dataset_revision="eth-usdt-v1",
            interval_start="2025-02-01T00:00:00Z",
            interval_end="2025-03-01T00:00:00Z",
        )

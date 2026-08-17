from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

_PURPOSES = frozenset(
    {"discovery", "feature_build", "model_training", "selection", "validation"}
)
_UTC_INSTANT = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
    re.ASCII,
)


def _require_nonempty_str(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _utc(value: object, name: str) -> str:
    value = _require_nonempty_str(value, name)
    if _UTC_INSTANT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical UTC Z instant")
    try:
        instant = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical UTC Z instant") from error
    if instant.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be a canonical UTC Z instant")
    return instant.isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True, order=True)
class SampleConsumptionRecord:
    dataset_revision: str
    interval_start: str
    interval_end: str
    purpose: str
    consumer_id: str
    consumed_at: str

    def __post_init__(self) -> None:
        _require_nonempty_str(self.dataset_revision, "dataset_revision")
        _require_nonempty_str(self.consumer_id, "consumer_id")
        if type(self.purpose) is not str or self.purpose not in _PURPOSES:
            raise ValueError("purpose is not recognized")
        for name in ("interval_start", "interval_end", "consumed_at"):
            object.__setattr__(self, name, _utc(getattr(self, name), name))
        if self.interval_start >= self.interval_end:
            raise ValueError("interval_start must be before interval_end")


def _validated_record(value: object) -> SampleConsumptionRecord:
    if type(value) is not SampleConsumptionRecord:
        raise ValueError("records must contain SampleConsumptionRecord values")
    try:
        normalized = SampleConsumptionRecord(
            value.dataset_revision,
            value.interval_start,
            value.interval_end,
            value.purpose,
            value.consumer_id,
            value.consumed_at,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("records must contain canonical SampleConsumptionRecord values") from error
    if value != normalized:
        raise ValueError("records must contain canonical SampleConsumptionRecord values")
    return normalized


@dataclass(frozen=True, slots=True)
class SampleConsumptionSnapshot:
    as_of: str
    records: tuple[SampleConsumptionRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _utc(self.as_of, "as_of"))
        if type(self.records) is not tuple:
            raise ValueError("records must be a tuple")
        records = tuple(_validated_record(record) for record in self.records)
        if records != tuple(sorted(records)):
            raise ValueError("records must be canonical")
        if any(record.consumed_at > self.as_of for record in records):
            raise ValueError("records must not be after as_of")
        object.__setattr__(self, "records", records)


@dataclass(frozen=True, slots=True)
class SampleIntegrityResult:
    untouched: bool
    conflicting_records: tuple[SampleConsumptionRecord, ...]

    def __post_init__(self) -> None:
        if type(self.untouched) is not bool:
            raise ValueError("untouched must be a bool")
        if type(self.conflicting_records) is not tuple:
            raise ValueError("conflicting_records must be a tuple")
        for record in self.conflicting_records:
            _validated_record(record)


def build_snapshot(
    records: tuple[SampleConsumptionRecord, ...], *, as_of: str
) -> SampleConsumptionSnapshot:
    if type(records) is not tuple:
        raise ValueError("records must be a tuple")
    normalized_as_of = _utc(as_of, "as_of")
    included = tuple(
        record
        for record in (_validated_record(record) for record in records)
        if record.consumed_at <= normalized_as_of
    )
    return SampleConsumptionSnapshot(normalized_as_of, tuple(sorted(included)))


def assess_untouched_holdout(
    snapshot: SampleConsumptionSnapshot,
    *,
    dataset_revision: str,
    interval_start: str,
    interval_end: str,
) -> SampleIntegrityResult:
    if type(snapshot) is not SampleConsumptionSnapshot:
        raise ValueError("snapshot must be a SampleConsumptionSnapshot")
    snapshot = SampleConsumptionSnapshot(snapshot.as_of, snapshot.records)
    dataset_revision = _require_nonempty_str(dataset_revision, "dataset_revision")
    normalized_start = _utc(interval_start, "interval_start")
    normalized_end = _utc(interval_end, "interval_end")
    if normalized_start >= normalized_end:
        raise ValueError("interval_start must be before interval_end")
    conflicts = tuple(
        record
        for record in snapshot.records
        if record.dataset_revision == dataset_revision
        and record.interval_start < normalized_end
        and normalized_start < record.interval_end
    )
    return SampleIntegrityResult(not conflicts, conflicts)

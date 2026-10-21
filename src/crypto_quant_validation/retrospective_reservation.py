"""Append-before-read registration for retrospective local captures.

No source I/O, Reader, completed-capture claim, untouched assessment or financial
report is produced. A reservation receipt is not permission to read.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from crypto_quant_domain import (
    ArtifactEnvelope,
    ArtifactRef,
    canonical_bytes,
    canonical_sha256,
)
from crypto_quant_foundation import (
    LocalFoundation,
    LogCheckpoint,
    LogEntry,
    LogEntryRef,
)

from .ledger import SampleConsumptionLedger
from .sample_consumption import SampleConsumptionRecord
from .statistical_reservation import (
    _artifact_event,
    _current_entries,
    _envelope,
    _hash,
    _instant,
    _mapping,
    _original_publication,
    _publication,
    _ref_from_wire,
    _sample_entries,
    _text,
    _verify_reservation,
)

_RESEARCH = "research.artifacts.v1"
_SAMPLE = "validation.sample-consumption.v1"
_PROTOCOL = "retrospective_study_protocol"
_STUDY = "retrospective_study_declaration"
_MODELS = ("CLIMATOLOGY", "PRICE_ONLY", "PRICE_VOLUME", "SHUFFLED_VOLUME")
_ROLES = ("TRAIN", "DEVELOPMENT", "RETROSPECTIVE_EVAL")
_CLAIMS = {
    "data_mode": "CURRENT_SNAPSHOT_RETROSPECTIVE",
    "models": _MODELS,
    "untouched_claim": False,
    "causal_pit_claim": False,
}
_PROTOCOL_HASHES = (
    "research_design_hash",
    "core_code_hash",
    "adapter_code_hash",
    "source_scope_hash",
    "prior_observation_hash",
)
_STUDY_HASHES = (
    "input_manifest_hash",
    "calendar_manifest_hash",
    "cohort_manifest_hash",
    "source_aliases_hash",
)
_HASHES = (
    "feature_rows_hash",
    "model_cards_hash",
    "predictions_hash",
    "input_check_hash",
)
_PHASES = {
    "INPUT_QUALIFICATION": (None, "retrospective_input_read_task", "discovery", (), ()),
    "SNAPSHOT_FREEZE": (
        None,
        "retrospective_input_read_task",
        "discovery",
        ("input_check_hash",),
        ("INPUT_QUALIFICATION",),
    ),
    "TRAIN_FEATURE": (
        "TRAIN",
        "retrospective_feature_read_task",
        "feature_build",
        (),
        (),
    ),
    "TRAIN_FIT": (
        "TRAIN",
        "retrospective_training_read_task",
        "model_training",
        ("feature_rows_hash",),
        ("TRAIN_FEATURE",),
    ),
    "DEVELOPMENT_FEATURE": (
        "DEVELOPMENT",
        "retrospective_feature_read_task",
        "feature_build",
        ("model_cards_hash",),
        ("TRAIN_FIT",),
    ),
    "RETROSPECTIVE_EVAL_FEATURE": (
        "RETROSPECTIVE_EVAL",
        "retrospective_feature_read_task",
        "feature_build",
        ("model_cards_hash",),
        ("TRAIN_FIT",),
    ),
    "DEVELOPMENT_OUTCOME": (
        "DEVELOPMENT",
        "retrospective_outcome_read_task",
        "validation",
        ("model_cards_hash", "predictions_hash"),
        ("DEVELOPMENT_FEATURE", "RETROSPECTIVE_EVAL_FEATURE"),
    ),
    "RETROSPECTIVE_EVAL_OUTCOME": (
        "RETROSPECTIVE_EVAL",
        "retrospective_outcome_read_task",
        "validation",
        ("model_cards_hash", "predictions_hash"),
        ("DEVELOPMENT_FEATURE", "RETROSPECTIVE_EVAL_FEATURE"),
    ),
}
_TASK_TYPES = frozenset(row[1] for row in _PHASES.values())
_CODES = frozenset(
    "RETROSPECTIVE_" + suffix
    for suffix in (
        "PROTOCOL_INVALID",
        "STUDY_INVALID",
        "READ_TASK_INVALID",
        "PUBLICATION_MISSING",
        "SCOPE_CONFLICT",
        "PREREQUISITE_INVALID",
        "RESERVATION_FAILED",
    )
)
Publication = tuple[ArtifactRef, LogEntryRef, LogCheckpoint]
Phase = tuple[Mapping[str, Any], ArtifactRef, LogEntry]


class RetrospectiveReservationError(ValueError):
    def __init__(self, code: str):
        if code not in _CODES:
            raise ValueError("unknown retrospective failure")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RetrospectiveReservationReceipt:
    protocol_ref: ArtifactRef
    study_ref: ArtifactRef | None
    read_task_ref: ArtifactRef
    protocol_publication_entry_ref: LogEntryRef
    study_publication_entry_ref: LogEntryRef | None
    producer_publication_entry_ref: LogEntryRef
    purpose: str
    record: SampleConsumptionRecord
    reservation_entry_ref: LogEntryRef

    @property
    def data_mode(self):
        return "CURRENT_SNAPSHOT_RETROSPECTIVE"

    @property
    def untouched_claim(self):
        return False

    @property
    def causal_pit_claim(self):
        return False


def _schema(e: ArtifactEnvelope, kind: str):
    if (
        e.artifact_type != kind
        or type(e.schema_version) is not int
        or e.schema_version != 1
    ):
        raise ValueError("unexpected retrospective artifact schema")


def _typed_ref(value, kind):
    ref = _ref_from_wire(value)
    if (
        ref.artifact_type != kind
        or type(ref.schema_version) is not int
        or ref.schema_version != 1
    ):
        raise ValueError("unexpected retrospective reference")
    return ref


def _claims(w):
    if any(
        canonical_bytes(w[key]) != canonical_bytes(value)
        for key, value in _CLAIMS.items()
    ):
        raise ValueError("cannot widen retrospective interpretation")


def _source(value):
    w = _mapping(
        value,
        {
            "scope_key",
            "capture_revision",
            "source_scope_hash",
            "interval_start",
            "interval_end",
            "input_manifest_hash",
        },
    )
    _text(w["scope_key"])
    if (
        type(w["capture_revision"]) is not str
        or re.fullmatch(r"local-capture:sha256:[0-9a-f]{64}", w["capture_revision"])
        is None
    ):
        raise ValueError("capture must be explicitly local and planned")
    _hash(w["source_scope_hash"])
    _instant(w["interval_start"])
    _instant(w["interval_end"])
    if w["interval_start"] >= w["interval_end"]:
        raise ValueError("empty source interval")
    if w["input_manifest_hash"] is not None:
        _hash(w["input_manifest_hash"])
    return w


def _protocol(e):
    _schema(e, _PROTOCOL)
    w = _mapping(
        e.payload,
        {
            "study_key",
            *_PROTOCOL_HASHES,
            "source_intervals",
            "capture_revision",
            *_CLAIMS,
        },
    )
    _claims(w)
    _text(w["study_key"])
    for name in _PROTOCOL_HASHES:
        _hash(w[name])
    expected = "local-capture:" + canonical_sha256(
        (
            "retrospective-local-capture-v1",
            w["study_key"],
            w["research_design_hash"],
            w["source_scope_hash"],
            w["prior_observation_hash"],
        )
    )
    if w["capture_revision"] != expected:
        raise ValueError("local capture identity mismatch")
    if type(w["source_intervals"]) is not tuple or not w["source_intervals"]:
        raise ValueError("immutable source intervals required")
    keys = []
    for row in w["source_intervals"]:
        if type(row) is not tuple or len(row) != 3:
            raise ValueError("exact interval triples required")
        key, start, end = row
        _source(
            {
                "scope_key": key,
                "capture_revision": expected,
                "source_scope_hash": w["source_scope_hash"],
                "interval_start": start,
                "interval_end": end,
                "input_manifest_hash": None,
            }
        )
        keys.append(key)
    if keys != sorted(set(keys)):
        raise ValueError("scope_key must be sorted/unique")
    return w


def _study(e):
    _schema(e, _STUDY)
    w = _mapping(e.payload, {"protocol_ref", *_STUDY_HASHES, "partitions", *_CLAIMS})
    _claims(w)
    _typed_ref(w["protocol_ref"], _PROTOCOL)
    for name in _STUDY_HASHES:
        _hash(w[name])
    if type(w["partitions"]) is not tuple or len(w["partitions"]) != 3:
        raise ValueError("three immutable partitions required")
    previous = None
    for role, value in zip(_ROLES, w["partitions"], strict=True):
        row = _mapping(
            value,
            {
                "partition",
                "source_slice",
                "decision_interval_start",
                "decision_interval_end",
            },
        )
        s = _source(row["source_slice"])
        _instant(row["decision_interval_start"])
        _instant(row["decision_interval_end"])
        if (
            row["partition"] != role
            or s["input_manifest_hash"] != w["input_manifest_hash"]
        ):
            raise ValueError("wrong retrospective partition/manifest")
        if (
            not s["interval_start"]
            <= row["decision_interval_start"]
            < row["decision_interval_end"]
            <= s["interval_end"]
        ):
            raise ValueError("decision outside read")
        if previous is not None and previous > s["interval_start"]:
            raise ValueError("overlapping complete read intervals")
        previous = s["interval_end"]
    return w


def _task(e):
    if (
        e.artifact_type not in _TASK_TYPES
        or type(e.schema_version) is not int
        or e.schema_version != 1
    ):
        raise ValueError("unexpected retrospective task")
    w = _mapping(e.payload, {"context_ref", "phase", "source_slice", *_HASHES})
    phase = w["phase"]
    if type(phase) is not str or phase not in _PHASES:
        raise ValueError("unknown retrospective phase")
    role, producer, _, required, _ = _PHASES[phase]
    if producer != e.artifact_type:
        raise ValueError("phase/producer mismatch")
    _typed_ref(w["context_ref"], _PROTOCOL if role is None else _STUDY)
    s = _source(w["source_slice"])
    if (s["input_manifest_hash"] is None) != (role is None):
        raise ValueError("phase/capture-state mismatch")
    for name in _HASHES:
        if name in required:
            _hash(w[name])
        elif w[name] is not None:
            raise ValueError("unexpected phase hash")
    return w


def _read(f, value):
    try:
        pub = _publication(value, _RESEARCH)
        obj = f.read(ref=pub[0])
        return pub, obj.envelope, obj.source_bytes
    except Exception as error:
        raise RetrospectiveReservationError(
            "RETROSPECTIVE_PUBLICATION_MISSING"
        ) from error


def _candidate(p, key):
    for k, start, end in p["source_intervals"]:
        if k == key:
            return {
                "scope_key": k,
                "capture_revision": p["capture_revision"],
                "source_scope_hash": p["source_scope_hash"],
                "interval_start": start,
                "interval_end": end,
                "input_manifest_hash": None,
            }
    raise ValueError("undeclared source key")


def _study_scope(p, s, protocol_ref):
    if _typed_ref(s["protocol_ref"], _PROTOCOL) != protocol_ref:
        raise ValueError("Study refers to another Protocol")
    for row in s["partitions"]:
        item = row["source_slice"]
        old = _candidate(p, item["scope_key"])
        if (
            item["capture_revision"] != p["capture_revision"]
            or item["source_scope_hash"] != p["source_scope_hash"]
            or not old["interval_start"]
            <= item["interval_start"]
            < item["interval_end"]
            <= old["interval_end"]
        ):
            raise ValueError("Study source does not preserve Protocol scope")


def _scope(context, t):
    role = _PHASES[t["phase"]][0]
    expected = (
        _candidate(context, t["source_slice"]["scope_key"])
        if role is None
        else next(
            row["source_slice"]
            for row in context["partitions"]
            if row["partition"] == role
        )
    )
    if canonical_bytes(expected) != canonical_bytes(t["source_slice"]):
        raise ValueError("task does not exactly cover declared slice")


def _parent_keys(t, context):
    result = []
    for phase in _PHASES[t["phase"]][4]:
        key = t["source_slice"]["scope_key"]
        if _PHASES[t["phase"]][0] is not None:
            key = next(
                row["source_slice"]["scope_key"]
                for row in context["partitions"]
                if row["partition"] == _PHASES[phase][0]
            )
        result.append((phase, key))
    return tuple(result)


def _event(context, phase, key):
    return canonical_sha256(
        ("retrospective-read-task-publication-v1", context, phase, key)
    )


def _phases(f, context_ref, context, context_entry, entries):
    result = {}
    for entry in entries:
        e = _envelope(entry.payload)
        if (
            e.artifact_type not in _TASK_TYPES
            or not isinstance(e.payload, Mapping)
            or e.payload.get("context_ref") != context_ref.to_canonical_dict()
        ):
            continue
        t = _task(e)
        ref = ArtifactRef.from_envelope(e)
        _scope(context, t)
        key = (t["phase"], t["source_slice"]["scope_key"])
        if (
            key in result
            or entry.event_id != _event(context_ref, *key)
            or entry.ledger_sequence <= context_entry.ledger_sequence
        ):
            raise ValueError("conflicting task graph/publication")
        if f.read(ref=ref).source_bytes != entry.payload:
            raise ValueError("task CAS/owner bytes differ")
        result[key] = (t, ref, entry)
    for t, _, entry in result.values():
        if any(
            key not in result or result[key][2].ledger_sequence >= entry.ledger_sequence
            for key in _parent_keys(t, context)
        ):
            raise ValueError("missing/late published predecessor")
    for name in ("model_cards_hash", "predictions_hash"):
        if len({t[name] for t, _, _ in result.values() if t[name] is not None}) > 1:
            raise ValueError("conflicting frozen anchors")
    return result


def _intended(ref, t, at):
    s = t["source_slice"]
    return SampleConsumptionRecord(
        s["capture_revision"],
        s["interval_start"],
        s["interval_end"],
        _PHASES[t["phase"]][2],
        canonical_sha256(("sample-consumer-v1", ref)),
        _instant(at),
    )


def _reserved(phase: Phase, samples, before=None):
    t, ref, pub = phase
    matches = [
        (record, entry) for record, producer, entry in samples if producer == ref
    ]
    if len(matches) != 1:
        raise ValueError("one actual prerequisite append required")
    record, entry = matches[0]
    if (
        record != _intended(ref, t, record.consumed_at)
        or record.consumed_at < pub.accepted_at
        or entry.ledger_sequence <= pub.ledger_sequence
    ):
        raise ValueError("reservation does not match actual task/source/time")
    if before is not None and entry.ledger_sequence >= before:
        raise ValueError("prerequisite append must precede child publication")
    return record, entry


def _reserved_ancestry(key, graph, context, samples, before):
    """Recheck every ancestor; a direct public Ledger append is not phase proof."""
    phase = graph[key]
    _reserved(phase, samples, before)
    for parent in _parent_keys(phase[0], context):
        _reserved_ancestry(parent, graph, context, samples, phase[2].ledger_sequence)


def reserve_retrospective_read(
    foundation: LocalFoundation,
    ledger: SampleConsumptionLedger,
    protocol_publication: Publication,
    task_publication: Publication,
    consumed_at: str,
    study_publication: Publication | None = None,
) -> RetrospectiveReservationReceipt:
    """Reserve the declared local-capture observation, never permission to read."""
    if type(foundation) is not LocalFoundation:
        raise RetrospectiveReservationError("RETROSPECTIVE_PUBLICATION_MISSING")
    pp, pe, pbytes = _read(foundation, protocol_publication)
    try:
        p = _protocol(pe)
    except Exception as error:
        raise RetrospectiveReservationError("RETROSPECTIVE_PROTOCOL_INVALID") from error
    sp = se = sbytes = s = None
    if study_publication is not None:
        sp, se, sbytes = _read(foundation, study_publication)
        try:
            s = _study(se)
        except Exception as error:
            raise RetrospectiveReservationError(
                "RETROSPECTIVE_STUDY_INVALID"
            ) from error
    tp, te, tbytes = _read(foundation, task_publication)
    try:
        t = _task(te)
    except Exception as error:
        raise RetrospectiveReservationError(
            "RETROSPECTIVE_READ_TASK_INVALID"
        ) from error
    initial = _PHASES[t["phase"]][0] is None
    if not initial and sp is None:
        raise RetrospectiveReservationError("RETROSPECTIVE_STUDY_INVALID")
    try:
        pentry = _original_publication(
            foundation, pp, pbytes, _artifact_event(pp[0], _RESEARCH)
        )
        sentry = None
        if sp is not None:
            assert sbytes is not None
            sentry = _original_publication(
                foundation, sp, sbytes, _artifact_event(sp[0], _RESEARCH)
            )
        tentry = _original_publication(
            foundation,
            tp,
            tbytes,
            _event(
                _ref_from_wire(t["context_ref"]),
                t["phase"],
                t["source_slice"]["scope_key"],
            ),
        )
    except Exception as error:
        raise RetrospectiveReservationError(
            "RETROSPECTIVE_PUBLICATION_MISSING"
        ) from error
    try:
        if initial:
            if sp is not None or _typed_ref(t["context_ref"], _PROTOCOL) != pp[0]:
                raise ValueError("input context must be this Protocol alone")
            context = p
            context_entry = pentry
            context_ref = pp[0]
        else:
            assert s is not None and sp is not None and sentry is not None
            _study_scope(p, s, pp[0])
            if (
                _typed_ref(t["context_ref"], _STUDY) != sp[0]
                or pentry.ledger_sequence >= sentry.ledger_sequence
            ):
                raise ValueError("Study/context publication conflict")
            context = s
            context_entry = sentry
            context_ref = sp[0]
        _scope(context, t)
    except Exception as error:
        raise RetrospectiveReservationError("RETROSPECTIVE_SCOPE_CONFLICT") from error
    try:
        _, entries = _current_entries(foundation, _RESEARCH)
        graph = _phases(foundation, context_ref, context, context_entry, entries)
        target = graph[(t["phase"], t["source_slice"]["scope_key"])]
        if target[1] != tp[0] or target[2] != tentry:
            raise ValueError("task proof is not its original graph entry")
        _, sample_entries = _current_entries(foundation, _SAMPLE)
        samples = _sample_entries(sample_entries)
        for key in _parent_keys(t, context):
            _reserved_ancestry(key, graph, context, samples, tentry.ledger_sequence)
        if not initial:
            assert s is not None and sentry is not None
            inputs = _phases(foundation, pp[0], p, pentry, entries)
            for key in {row["source_slice"]["scope_key"] for row in s["partitions"]}:
                freeze = inputs[("SNAPSHOT_FREEZE", key)]
                if freeze[2].ledger_sequence >= sentry.ledger_sequence:
                    raise ValueError("snapshot declaration must precede Study")
                _reserved_ancestry(
                    ("SNAPSHOT_FREEZE", key), inputs, p, samples, sentry.ledger_sequence
                )
    except Exception as error:
        raise RetrospectiveReservationError(
            "RETROSPECTIVE_PREREQUISITE_INVALID"
        ) from error
    try:
        if (
            type(ledger) is not SampleConsumptionLedger
            or ledger._foundation is not foundation
        ):
            raise ValueError("ledger must belong to the same Foundation")
        record = _intended(tp[0], t, consumed_at)
        checkpoint = foundation.checkpoint(_SAMPLE)
        if (
            record.consumed_at < tentry.accepted_at
            or record.consumed_at > checkpoint.as_of
        ):
            raise ValueError("consumed_at outside actual publication/checkpoint time")
        existing = [item for item in samples if item[1] == tp[0]]
        if existing:
            old, _ = _reserved(target, samples)
            if old != record:
                raise ValueError("cannot refresh an existing reservation")
        ref = ledger.reserve(record, tp[0])
        entry = _verify_reservation(foundation, ref, record, tp[0])
        if entry.ledger_sequence <= tentry.ledger_sequence:
            raise ValueError("append before producer publication")
    except Exception as error:
        raise RetrospectiveReservationError(
            "RETROSPECTIVE_RESERVATION_FAILED"
        ) from error
    return RetrospectiveReservationReceipt(
        pp[0],
        None if sp is None else sp[0],
        tp[0],
        pp[1],
        None if sp is None else sp[1],
        tp[1],
        record.purpose,
        record,
        ref,
    )

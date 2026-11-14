"""Owner identity and append-before-read proofs, never source or read authority."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from crypto_quant_domain import ArtifactEnvelope, ArtifactRef, canonical_bytes, canonical_sha256
from crypto_quant_foundation import LocalFoundation, LogCheckpoint, LogEntry, LogEntryRef

from .integration import ValidationCoreFailure
from .ledger import SampleConsumptionLedger
from .sample_consumption import SampleConsumptionRecord
from .statistical_reservation import (
    Publication, _checkpoint_wire, _envelope, _entry_wire, _hash, _instant,
    _mapping, _ref_from_wire, _sample_entries,
)

_LOG = "research.artifacts.v1"
_SAMPLE_LOG = "validation.sample-consumption.v1"
_DECLARATION = "input_qualification_declaration"
_TASK = "input_qualification_read_task"
_BINDING = "input_qualification_capture_binding"
_LIMIT = 2 * 1024 * 1024
_KEY = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z")
_HASH_FIELDS = ("research_design_hash", "reader_code_hash", "normalization_code_hash",
                "prior_observation_hash", "source_scope_hash")
InputQualificationFailureCodeV1 = Literal[
    "FOUNDATION_MISMATCH", "PUBLICATION_INVALID", "SCOPE_CONFLICT",
    "PREREQUISITE_INVALID", "CAPTURE_CONFLICT", "TRIAL_CAPTURE_MISMATCH",
    "RESERVATION_INVALID",
]
_CODES = ("FOUNDATION_MISMATCH", "PUBLICATION_INVALID", "SCOPE_CONFLICT",
          "PREREQUISITE_INVALID", "CAPTURE_CONFLICT", "TRIAL_CAPTURE_MISMATCH",
          "RESERVATION_INVALID")


class InputQualificationError(ValueError):
    def __init__(self, code: InputQualificationFailureCodeV1) -> None:
        if code not in _CODES:
            raise ValueError("unknown input qualification failure")
        self.code = code
        super().__init__(code)


def _fail(code: InputQualificationFailureCodeV1) -> None:
    raise InputQualificationError(code)


def _key(value: object) -> str:
    if type(value) is not str or _KEY.fullmatch(value) is None:
        raise ValueError("key must be canonical lowercase text, at most 128 characters")
    return value


def _capture_revision(ref: ArtifactRef) -> str:
    return "local-capture:" + canonical_sha256(("input-qualification-capture-v1", ref))


def _ref(value: object, name: str) -> ArtifactRef:
    if type(value) is not ArtifactRef:
        raise TypeError("reference must be an exact Domain ArtifactRef")
    if (type(value.schema_version) is not int or value.schema_version != 1
            or value.artifact_type != name):
        raise ValueError("reference must address the exact schema-1 artifact")
    _hash(value.content_hash)
    return value


def _wire_ref(value: object, name: str) -> ArtifactRef:
    wire = _mapping(value, {"type", "artifact_type", "schema_version", "content_hash"})
    if type(wire["schema_version"]) is not int:
        raise ValueError("schema version must be an exact integer")
    return _ref(_ref_from_wire(wire), name)


def _json_value(value: object) -> object:
    # Restrict the native metadata to JSON values, not objects with a codec hook.
    if value is None or type(value) in (bool, int, str):
        canonical_bytes(value)
        return value
    if type(value) in (tuple, list):
        return tuple(_json_value(item) for item in value)
    if type(value) in (dict, MappingProxyType):
        if any(type(key) is not str for key in value):
            raise ValueError("canonical mapping keys must be strings")
        result = MappingProxyType({key: _json_value(child) for key, child in value.items()})
        canonical_bytes(result)
        return result
    raise ValueError("metadata must contain only canonical JSON values")


def _intervals(value: object) -> tuple:
    if type(value) is not tuple or not 1 <= len(value) <= 32:
        raise ValueError("source intervals must be a nonempty tuple of at most 32 scopes")
    previous = ""
    for item in value:
        if type(item) is not tuple or len(item) != 3:
            raise ValueError("source interval must be an immutable three-tuple")
        key, start, end = item
        if _key(key) <= previous or _instant(start) >= _instant(end):
            raise ValueError("scopes must be strictly sorted with nonempty intervals")
        previous = key
    return value


def _checked(envelope: object, name: str) -> ArtifactEnvelope:
    if type(envelope) is not ArtifactEnvelope:
        raise TypeError("value must be an exact Domain ArtifactEnvelope")
    if (envelope.artifact_type != name or type(envelope.schema_version) is not int
            or envelope.schema_version != 1):
        raise ValueError("artifact must have the exact schema-1 type")
    if len(canonical_bytes(envelope)) > _LIMIT:
        raise ValueError("input qualification envelope exceeds 2MiB")
    if ArtifactEnvelope.create(name, 1, envelope.payload) != envelope:
        raise ValueError("envelope hash must match its payload")
    return envelope


def _declaration(envelope: object) -> Mapping[str, Any]:
    wire = _mapping(_checked(envelope, _DECLARATION).payload,
                    {"qualification_key", "mode", "source_intervals", *_HASH_FIELDS})
    _key(wire["qualification_key"])
    if wire["mode"] != "DEVELOPMENT_INPUT_QUALIFICATION":
        raise ValueError("declaration mode is fixed")
    for field in _HASH_FIELDS:
        _hash(wire[field])
    _intervals(wire["source_intervals"])
    return wire


def _task(envelope: object) -> Mapping[str, Any]:
    wire = _mapping(_checked(envelope, _TASK).payload,
                    {"declaration_ref", "scope_key", "phase", "input_check_hash"})
    _wire_ref(wire["declaration_ref"], _DECLARATION)
    _key(wire["scope_key"])
    if type(wire["phase"]) is not str or wire["phase"] not in ("QUALIFY", "CAPTURE"):
        raise ValueError("read phase must be QUALIFY or CAPTURE")
    if wire["phase"] == "QUALIFY":
        if wire["input_check_hash"] is not None:
            raise ValueError("QUALIFY cannot carry a check hash")
    else:
        _hash(wire["input_check_hash"])
    return wire


def _binding(envelope: object) -> Mapping[str, Any]:
    wire = _mapping(_checked(envelope, _BINDING).payload,
                    {"declaration_ref", "capture_proofs", "source_manifest_hash",
                     "market_bundle_ref", "market_bundle_manifest"})
    _wire_ref(wire["declaration_ref"], _DECLARATION)
    _hash(wire["source_manifest_hash"])
    proofs = wire["capture_proofs"]
    if type(proofs) is not tuple:
        raise ValueError("capture proofs must be an immutable tuple")
    previous = ""
    for item in proofs:
        if type(item) is not tuple or len(item) != 3 or _key(item[0]) <= previous:
            raise ValueError("capture proofs must be strictly scope ordered")
        previous = item[0]
        _wire_ref(item[1], _TASK)
        entry = _mapping(item[2], {"log_name", "log_sequence", "receipt_hash"})
        _entry_wire(LogEntryRef(**entry))
    ref = _mapping(wire["market_bundle_ref"], {"type", "bundle_key", "manifest_hash"})
    manifest = _mapping(wire["market_bundle_manifest"], {
        "type", "bundle_key", "schema_version", "coverage_start", "coverage_end_exclusive",
        "instrument_catalog_hash", "capabilities", "streams", "content_hash"})
    _json_value(ref)
    _json_value(manifest)
    if ref["type"] != "market_bundle_ref" or manifest["type"] != "market_bundle_manifest":
        raise ValueError("binding must carry native MarketBundle metadata wire")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("native metadata schema must be exact integer 1")
    _key(ref["bundle_key"])
    _key(manifest["bundle_key"])
    _hash(ref["manifest_hash"])
    _hash(manifest["content_hash"])
    return wire


def _task_event(ref: ArtifactRef, phase: str, scope: str) -> str:
    return canonical_sha256(("input-qualification-read-task-v1", ref, phase, scope))


def _binding_event(ref: ArtifactRef) -> str:
    return canonical_sha256(("input-qualification-capture-binding-v1", _capture_revision(ref)))


def _artifact_event(ref: ArtifactRef) -> str:
    return canonical_sha256(("artifact-publication-v1", _LOG, ref))


def _event(envelope: ArtifactEnvelope) -> str:
    if envelope.artifact_type == _TASK:
        task = _task(envelope)
        return _task_event(_wire_ref(task["declaration_ref"], _DECLARATION), task["phase"], task["scope_key"])
    if envelope.artifact_type == _BINDING:
        binding = _binding(envelope)
        return _binding_event(_wire_ref(binding["declaration_ref"], _DECLARATION))
    _declaration(envelope)
    return _artifact_event(ArtifactRef.from_envelope(envelope))


@dataclass(frozen=True, slots=True)
class InputQualificationReadBindingV1:
    capture_revision: str
    scope_key: str
    phase: str
    source_scope_hash: str
    reader_code_hash: str
    normalization_code_hash: str
    interval_start: str
    interval_end: str

    def __post_init__(self) -> None:
        prefix = "local-capture:"
        if type(self.capture_revision) is not str or not self.capture_revision.startswith(prefix):
            raise ValueError("read binding must use a local capture revision")
        _hash(self.capture_revision[len(prefix):])
        _key(self.scope_key)
        if type(self.phase) is not str or self.phase not in ("QUALIFY", "CAPTURE"):
            raise ValueError("read binding phase is fixed")
        for value in (self.source_scope_hash, self.reader_code_hash, self.normalization_code_hash):
            _hash(value)
        if _instant(self.interval_start) >= _instant(self.interval_end):
            raise ValueError("read binding interval must be nonempty")


@dataclass(frozen=True, slots=True)
class InputQualificationReservationReceiptV1:
    declaration_ref: ArtifactRef
    read_task_ref: ArtifactRef
    declaration_publication_entry_ref: LogEntryRef
    producer_publication_entry_ref: LogEntryRef
    phase: str
    scope_key: str
    record: SampleConsumptionRecord
    reservation_entry_ref: LogEntryRef

    def __post_init__(self) -> None:
        _ref(self.declaration_ref, _DECLARATION)
        _ref(self.read_task_ref, _TASK)
        for entry in (self.declaration_publication_entry_ref, self.producer_publication_entry_ref):
            if _entry_wire(entry)["log_name"] != _LOG:
                raise ValueError("publication entry must use the Research owner log")
        if _entry_wire(self.reservation_entry_ref)["log_name"] != _SAMPLE_LOG:
            raise ValueError("reservation entry must use the Validation sample log")
        _key(self.scope_key)
        if type(self.phase) is not str or self.phase not in ("QUALIFY", "CAPTURE"):
            raise ValueError("receipt phase is fixed")
        if type(self.record) is not SampleConsumptionRecord:
            raise TypeError("receipt record must be exact SampleConsumptionRecord")
        if (self.record.dataset_revision != _capture_revision(self.declaration_ref)
                or self.record.purpose != "discovery"
                or self.record.consumer_id != canonical_sha256(("sample-consumer-v1", self.read_task_ref))):
            raise ValueError("receipt record must match the original producer identity")
        _instant(self.record.consumed_at)
        if _instant(self.record.interval_start) >= _instant(self.record.interval_end):
            raise ValueError("receipt interval must be nonempty")


@dataclass(frozen=True, slots=True)
class InputQualificationCaptureViewV1:
    binding_ref: ArtifactRef
    declaration_ref: ArtifactRef
    capture_revision: str
    market_bundle_ref: Mapping[str, Any]
    source_manifest_hash: str
    source_intervals: tuple

    def __post_init__(self) -> None:
        _ref(self.binding_ref, _BINDING)
        _ref(self.declaration_ref, _DECLARATION)
        if self.capture_revision != _capture_revision(self.declaration_ref):
            raise ValueError("capture revision must derive from the declaration")
        _hash(self.source_manifest_hash)
        _intervals(self.source_intervals)
        ref = _mapping(self.market_bundle_ref, {"type", "bundle_key", "manifest_hash"})
        if ref["type"] != "market_bundle_ref":
            raise ValueError("view must carry native bundle reference wire")
        _key(ref["bundle_key"])
        _hash(ref["manifest_hash"])
        object.__setattr__(self, "market_bundle_ref", _json_value(ref))


class InputQualificationVerifierV1:
    def __init__(self, foundation: LocalFoundation, ledger: SampleConsumptionLedger) -> None:
        self._foundation = foundation
        self._ledger = ledger
        self.require_foundation(foundation)

    def require_foundation(self, foundation: LocalFoundation) -> None:
        if (type(self) is not InputQualificationVerifierV1 or type(foundation) is not LocalFoundation
                or type(self._foundation) is not LocalFoundation
                or self._foundation is not foundation or type(self._ledger) is not SampleConsumptionLedger):
            _fail("FOUNDATION_MISMATCH")
        try:
            self._ledger.require_foundation(foundation)
        except (TypeError, ValidationCoreFailure) as error:
            raise InputQualificationError("FOUNDATION_MISMATCH") from error

    def _guard(self) -> None:
        self.require_foundation(self._foundation)

    def _publication(self, value: object, name: str) -> Publication:
        if type(value) is not tuple or len(value) != 3:
            raise TypeError("publication must be an immutable three-tuple")
        ref, entry, checkpoint = value
        _ref(ref, name)
        _entry_wire(entry)
        _checkpoint_wire(checkpoint)
        if entry.log_name != _LOG or checkpoint.log_name != _LOG:
            _fail("PUBLICATION_INVALID")
        return ref, entry, checkpoint

    def _original(self, ref: ArtifactRef, publication: Publication | None = None) -> tuple[ArtifactEnvelope, LogEntry]:
        source = self._foundation.read(ref=ref)
        envelope = source.envelope
        if envelope.artifact_type in (_DECLARATION, _TASK, _BINDING):
            expected = _event(envelope)
        else:
            expected = _artifact_event(ref)
        entries = self._foundation.entries(_LOG)
        matches = tuple(e for e in entries if e.event_id == expected)
        if len(matches) != 1 or matches[0].payload != source.source_bytes:
            _fail("PUBLICATION_INVALID")
        entry = matches[0]
        if publication is not None and (publication[1] != entry.entry_ref or publication[2] != LogCheckpoint(
                _LOG, entry.accepted_at, entry.log_sequence, entry.receipt_hash)):
            _fail("PUBLICATION_INVALID")
        return envelope, entry

    def _decl(self, ref: ArtifactRef) -> tuple[Mapping[str, Any], LogEntry]:
        envelope, entry = self._original(ref)
        return _declaration(envelope), entry

    def _samples(self) -> tuple:
        entries = self._foundation.entries(_SAMPLE_LOG)
        try:
            return _sample_entries(entries)
        except (TypeError, ValueError) as error:
            # Foundation calls are outside this parsing boundary.
            raise InputQualificationError("RESERVATION_INVALID") from error

    def _reservation(self, ref: ArtifactRef, declaration: Mapping[str, Any], task: Mapping[str, Any],
                     task_entry: LogEntry, *, before: LogEntry | None = None) -> tuple[SampleConsumptionRecord, LogEntry]:
        samples = self._samples()
        matches = tuple((record, entry) for record, producer, entry in samples if producer == ref)
        if len(matches) != 1:
            _fail("PREREQUISITE_INVALID")
        record, entry = matches[0]
        start, end = self._scope(declaration, task)
        if (record.dataset_revision != _capture_revision(_wire_ref(task["declaration_ref"], _DECLARATION))
                or (record.interval_start, record.interval_end) != (start, end)
                or record.consumed_at < task_entry.accepted_at
                or entry.ledger_sequence <= task_entry.ledger_sequence
                or (before is not None and entry.ledger_sequence >= before.ledger_sequence)):
            _fail("PREREQUISITE_INVALID")
        return record, entry

    def _scope(self, declaration: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[str, str]:
        matches = tuple((start, end) for key, start, end in declaration["source_intervals"] if key == task["scope_key"])
        if len(matches) != 1:
            _fail("SCOPE_CONFLICT")
        return matches[0]

    def _tasks(self, ref: ArtifactRef, declaration: Mapping[str, Any], declaration_entry: LogEntry) -> dict:
        tasks = {}
        for entry in self._foundation.entries(_LOG):
            envelope = _envelope(entry.payload)
            if (envelope.artifact_type != _TASK or not isinstance(envelope.payload, Mapping)
                    or envelope.payload.get("declaration_ref") != ref.to_canonical_dict()):
                continue
            task = _task(envelope)
            task_ref = ArtifactRef.from_envelope(envelope)
            _, original = self._original(task_ref)
            slot = task["phase"], task["scope_key"]
            if entry != original or slot in tasks:
                _fail("CAPTURE_CONFLICT")
            self._scope(declaration, task)
            if entry.ledger_sequence <= declaration_entry.ledger_sequence:
                _fail("PREREQUISITE_INVALID")
            tasks[slot] = task, task_ref, entry
        return tasks

    def _task_inputs(self, envelope: ArtifactEnvelope, task_entry: LogEntry | None = None) -> tuple:
        task = _task(envelope)
        ref = _wire_ref(task["declaration_ref"], _DECLARATION)
        declaration, declaration_entry = self._decl(ref)
        self._scope(declaration, task)
        tasks = self._tasks(ref, declaration, declaration_entry)
        if task_entry is not None and declaration_entry.ledger_sequence >= task_entry.ledger_sequence:
            _fail("PREREQUISITE_INVALID")
        existing = tasks.get((task["phase"], task["scope_key"]))
        if task_entry is None and existing is not None:
            task_entry = existing[2]
        if task["phase"] == "QUALIFY":
            successor = tasks.get(("CAPTURE", task["scope_key"]))
            if successor is not None and (task_entry is None or successor[2].ledger_sequence <= task_entry.ledger_sequence):
                _fail("PREREQUISITE_INVALID")
        if task["phase"] == "CAPTURE":
            parent = tasks.get(("QUALIFY", task["scope_key"]))
            if parent is None:
                _fail("PREREQUISITE_INVALID")
            self._reservation(parent[1], declaration, parent[0], parent[2], before=task_entry)
        if existing is not None and existing[1] != ArtifactRef.from_envelope(envelope):
            _fail("CAPTURE_CONFLICT")
        return ref, declaration, declaration_entry, task

    def verify_read_task_inputs(self, envelope: ArtifactEnvelope) -> None:
        self._guard()
        _task(envelope)
        self._task_inputs(envelope)

    def reserve(self, declaration_publication: Publication, task_publication: Publication,
                consumed_at: str, actual_read_binding: InputQualificationReadBindingV1) -> InputQualificationReservationReceiptV1:
        self._guard()
        decl_pub = self._publication(declaration_publication, _DECLARATION)
        task_pub = self._publication(task_publication, _TASK)
        _instant(consumed_at)
        if type(actual_read_binding) is not InputQualificationReadBindingV1:
            raise TypeError("actual read binding must be exact InputQualificationReadBindingV1")
        actual_read_binding.__post_init__()
        decl_env, decl_entry = self._original(decl_pub[0], decl_pub)
        task_env, task_entry = self._original(task_pub[0], task_pub)
        declaration, task = _declaration(decl_env), _task(task_env)
        start, end = self._scope(declaration, task)
        expected = InputQualificationReadBindingV1(_capture_revision(decl_pub[0]), task["scope_key"], task["phase"],
            declaration["source_scope_hash"], declaration["reader_code_hash"], declaration["normalization_code_hash"], start, end)
        if task["declaration_ref"] != decl_pub[0].to_canonical_dict() or actual_read_binding != expected:
            _fail("SCOPE_CONFLICT")
        self._task_inputs(task_env, task_entry)
        record = SampleConsumptionRecord(expected.capture_revision, start, end, "discovery",
                                        canonical_sha256(("sample-consumer-v1", task_pub[0])), consumed_at)
        samples = self._samples()
        known = tuple((prior, entry) for prior, producer, entry in samples if producer == task_pub[0])
        checkpoint = self._foundation.checkpoint(_SAMPLE_LOG)
        if (not task_entry.accepted_at <= consumed_at <= checkpoint.as_of
                or len(known) > 1 or (known and known[0][0] != record)):
            _fail("RESERVATION_INVALID")
        reservation_ref = self._ledger.reserve(record, task_pub[0])
        try:
            verified_record, entry = self._reservation(task_pub[0], declaration, task, task_entry)
        except InputQualificationError as error:
            raise InputQualificationError("RESERVATION_INVALID") from error
        if verified_record != record or entry.entry_ref != reservation_ref:
            _fail("RESERVATION_INVALID")
        return InputQualificationReservationReceiptV1(decl_pub[0], task_pub[0], decl_entry.entry_ref,
            task_entry.entry_ref, task["phase"], task["scope_key"], record, reservation_ref)

    def _capture_inputs(self, envelope: ArtifactEnvelope, binding_entry: LogEntry | None = None) -> tuple:
        binding = _binding(envelope)
        ref = _wire_ref(binding["declaration_ref"], _DECLARATION)
        declaration, decl_entry = self._decl(ref)
        if binding_entry is None:
            # An idempotent publisher retry is still bound to the original cutoff.
            matching = tuple(entry for entry in self._foundation.entries(_LOG)
                if entry.event_id == _binding_event(ref) and entry.payload == canonical_bytes(envelope))
            if len(matching) == 1:
                binding_entry = matching[0]
        if tuple(item[0] for item in binding["capture_proofs"]) != tuple(item[0] for item in declaration["source_intervals"]):
            _fail("PREREQUISITE_INVALID")
        for scope, task_wire, entry_wire in binding["capture_proofs"]:
            task_ref = _wire_ref(task_wire, _TASK)
            task_env, task_entry = self._original(task_ref)
            task = _task(task_env)
            if task["declaration_ref"] != ref.to_canonical_dict() or task["phase"] != "CAPTURE" or task["scope_key"] != scope:
                _fail("PREREQUISITE_INVALID")
            self._task_inputs(task_env, task_entry)
            _, entry = self._reservation(task_ref, declaration, task, task_entry, before=binding_entry)
            if _entry_wire(entry.entry_ref) != entry_wire:
                _fail("PREREQUISITE_INVALID")
        if binding_entry is not None and decl_entry.ledger_sequence >= binding_entry.ledger_sequence:
            _fail("PREREQUISITE_INVALID")
        bundle, manifest = binding["market_bundle_ref"], binding["market_bundle_manifest"]
        if bundle["bundle_key"] != manifest["bundle_key"] or bundle["manifest_hash"] != canonical_sha256(manifest):
            _fail("CAPTURE_CONFLICT")
        expected_event = _binding_event(ref)
        expected_source = canonical_bytes(envelope)
        for entry in self._foundation.entries(_LOG):
            prior = _envelope(entry.payload)
            same_capture = (prior.artifact_type == _BINDING and isinstance(prior.payload, Mapping)
                            and prior.payload.get("declaration_ref") == ref.to_canonical_dict())
            if entry.event_id == expected_event or same_capture:
                if entry.event_id != expected_event or entry.payload != expected_source:
                    _fail("CAPTURE_CONFLICT")
                _binding(prior)
        return binding, ref, declaration

    def verify_capture_inputs(self, envelope: ArtifactEnvelope) -> None:
        self._guard()
        _binding(envelope)
        self._capture_inputs(envelope)

    def verify_capture(self, binding_publication: Publication) -> InputQualificationCaptureViewV1:
        self._guard()
        publication = self._publication(binding_publication, _BINDING)
        envelope, entry = self._original(publication[0], publication)
        binding, ref, declaration = self._capture_inputs(envelope, entry)
        return InputQualificationCaptureViewV1(publication[0], ref, _capture_revision(ref),
            binding["market_bundle_ref"], binding["source_manifest_hash"], declaration["source_intervals"])

    def require_trial_binding(self, binding_publication: Publication, trial_ref: ArtifactRef) -> None:
        self._guard()
        _ref(trial_ref, "trial_declaration")
        view = self.verify_capture(binding_publication)
        envelope, _ = self._original(trial_ref)
        payload = _mapping(envelope.payload, {"experiment_ref", "data_slice", "parameter_values", "seed", "scenario_ref", "backtest_template_ref", "model_input_bindings"})
        _json_value(payload)
        if type(payload["seed"]) is not int or payload["seed"] < 0:
            raise ValueError("Trial seed must be an exact nonnegative integer")
        data = _mapping(payload["data_slice"], {"market_bundle_ref", "dataset_revision", "interval_start", "interval_end"})
        _instant(data["interval_start"])
        _instant(data["interval_end"])
        if (data["dataset_revision"] != view.capture_revision
                or canonical_bytes(data["market_bundle_ref"]) != canonical_bytes(view.market_bundle_ref)
                or not any(start <= data["interval_start"] < data["interval_end"] <= end
                           for _, start, end in view.source_intervals)):
            _fail("TRIAL_CAPTURE_MISMATCH")

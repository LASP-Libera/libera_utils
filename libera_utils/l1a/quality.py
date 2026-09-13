"""Per-granule L1A data-quality counters, their rollup flag, and the ``.qa.json`` sidecar.

Every L1A granule carries the same counters, zeros included, so a quality question is a trend
over granules rather than a search for exception reports. The events these count are not rare:
in DITL2, RAD packet time steps backward on ~1-2% of packets and ~1.5% of packets carry sample
timestamps that collide with an earlier packet's. What matters is how those rates move.

The counters live in three places with different lifetimes. This module owns the first and the
third: the NetCDF global attributes that travel with the granule wherever it goes, and the
sidecar that carries the per-event detail needed to verify a count rather than trust it. The
second, the File Metadata row, is written by the caller from :meth:`GranuleQualityRecord.counts`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class QualityFlag(StrEnum):
    """Rollup of a granule's quality counters, for an operator dashboard to watch."""

    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"
    SUSPECT = "SUSPECT"


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Where each counter crosses from ``NOMINAL`` into ``DEGRADED`` and then ``SUSPECT``.

    Fractions are of the granule's own packet or sample count, so one set of numbers means the
    same thing for a 1 Hz housekeeping APID and a 26 Hz camera one.

    The inversion thresholds start well above the measured DITL2 rates (1.4% of RAD packets,
    0.2% of WFOV, 0.5% of NOM-HK). A flag that fires on every granule carries no information,
    and the ordering these represent is corrected rather than lost.

    Any duplicate whose rows disagree is different in kind: a real sample was discarded to make
    the timestamp unique, so a single one is enough to leave ``NOMINAL``.
    """

    degraded_time_inversion_frac: float = 0.05
    suspect_time_inversion_frac: float = 0.25
    degraded_missing_packet_frac: float = 0.01
    suspect_missing_packet_frac: float = 0.10
    degraded_duplicate_frac: float = 0.05
    suspect_duplicate_frac: float = 0.25
    suspect_value_mismatch_frac: float = 0.01


DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()

# Written on every L1A product, zeros included. ``enforce_dataset_conformance`` deletes global
# attributes a product definition does not declare and validation fails on declared attributes a
# Dataset does not carry, so this tuple and the twelve product definitions must agree.
QUALITY_GLOBAL_ATTRIBUTES = (
    "QualityFlag",
    "PacketTimeInversionCount",
    "PacketsOutOfTimeOrderCount",
    "MaxPacketTimeInversionMicroseconds",
    "DuplicatePacketTimeCount",
    "DuplicateSampleTimeCount",
    "DuplicateValueMismatchCount",
    "MissingPacketCount",
    "MaxSampleGapMicroseconds",
    "SequenceResetCount",
)


@dataclass
class DuplicateEvidence:
    """Enough detail about one duplicated timestamp to check the count rather than trust it.

    ``values`` holds the disagreeing rows per variable, so a reader can decide for themselves
    whether dropping one of them discarded science or redundancy. Only populated for duplicates
    whose rows differ; identical duplicates need no evidence beyond their count.
    """

    coordinate: str
    timestamp: str
    multiplicity: int
    row_indices: list[int]
    variables: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class GranuleQualityRecord:
    """Every quality counter for one granule, plus the evidence behind the ones that fired.

    A record is always produced, including for a clean granule, where every count is zero and
    ``quality_flag`` is ``NOMINAL``.
    """

    product_id: str = ""
    apid: int | None = None
    n_packets: int = 0
    n_samples: int = 0

    packet_time_inversion_count: int = 0
    packets_out_of_time_order_count: int = 0
    max_packet_time_inversion_microseconds: int = 0
    duplicate_packet_time_count: int = 0
    duplicate_sample_time_count: int = 0
    duplicate_value_mismatch_count: int = 0
    missing_packet_count: int = 0
    max_sample_gap_microseconds: int = 0
    sequence_reset_count: int = 0
    sequence_order_violation_count: int = 0

    mismatched_variables: list[str] = field(default_factory=list)
    duplicate_evidence: list[DuplicateEvidence] = field(default_factory=list)

    def quality_flag(self, thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS) -> QualityFlag:
        """Roll the counters up into one flag.

        Parameters
        ----------
        thresholds : QualityThresholds, optional
            Where each counter crosses into ``DEGRADED`` and ``SUSPECT``.

        Returns
        -------
        QualityFlag
            The worst level any counter reaches.
        """
        packets = max(self.n_packets, 1)
        samples = max(self.n_samples, 1)
        level = QualityFlag.NOMINAL

        def escalate(current: QualityFlag, candidate: QualityFlag) -> QualityFlag:
            order = (QualityFlag.NOMINAL, QualityFlag.DEGRADED, QualityFlag.SUSPECT)
            return candidate if order.index(candidate) > order.index(current) else current

        # An uncorroborated sequence-counter step means the acquisition order itself is in doubt,
        # which makes every positional index in the granule unreliable.
        if self.sequence_order_violation_count:
            return QualityFlag.SUSPECT

        if self.duplicate_value_mismatch_count:
            level = escalate(level, QualityFlag.DEGRADED)
            if self.duplicate_value_mismatch_count / samples > thresholds.suspect_value_mismatch_frac:
                level = escalate(level, QualityFlag.SUSPECT)

        inversion_frac = self.packet_time_inversion_count / packets
        if inversion_frac > thresholds.suspect_time_inversion_frac:
            level = escalate(level, QualityFlag.SUSPECT)
        elif inversion_frac > thresholds.degraded_time_inversion_frac:
            level = escalate(level, QualityFlag.DEGRADED)

        missing_frac = self.missing_packet_count / packets
        if missing_frac > thresholds.suspect_missing_packet_frac:
            level = escalate(level, QualityFlag.SUSPECT)
        elif missing_frac > thresholds.degraded_missing_packet_frac:
            level = escalate(level, QualityFlag.DEGRADED)

        duplicate_frac = (self.duplicate_packet_time_count / packets) + (self.duplicate_sample_time_count / samples)
        if duplicate_frac > thresholds.suspect_duplicate_frac:
            level = escalate(level, QualityFlag.SUSPECT)
        elif duplicate_frac > thresholds.degraded_duplicate_frac:
            level = escalate(level, QualityFlag.DEGRADED)

        return level

    def global_attributes(
        self,
        thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
    ) -> dict[str, Any]:
        """Return the NetCDF global attributes for this granule.

        Every name in :data:`QUALITY_GLOBAL_ATTRIBUTES` is present, including zeros: a product
        definition declares them all, so an attribute left unwritten fails validation.

        Parameters
        ----------
        thresholds : QualityThresholds, optional
            Passed to :meth:`quality_flag`.

        Returns
        -------
        dict
            Attribute name to value, with integer counts as Python ``int``.
        """
        attributes = {
            "QualityFlag": str(self.quality_flag(thresholds)),
            "PacketTimeInversionCount": int(self.packet_time_inversion_count),
            "PacketsOutOfTimeOrderCount": int(self.packets_out_of_time_order_count),
            "MaxPacketTimeInversionMicroseconds": int(self.max_packet_time_inversion_microseconds),
            "DuplicatePacketTimeCount": int(self.duplicate_packet_time_count),
            "DuplicateSampleTimeCount": int(self.duplicate_sample_time_count),
            "DuplicateValueMismatchCount": int(self.duplicate_value_mismatch_count),
            "MissingPacketCount": int(self.missing_packet_count),
            "MaxSampleGapMicroseconds": int(self.max_sample_gap_microseconds),
            "SequenceResetCount": int(self.sequence_reset_count),
        }
        assert set(attributes) == set(QUALITY_GLOBAL_ATTRIBUTES)  # noqa: S101 - guards the YAML contract
        return attributes

    @property
    def counts(self) -> dict[str, int]:
        """Integer counters only, for a File Metadata row or a trending query."""
        return {
            name: value
            for name, value in asdict(self).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }

    def has_events(self) -> bool:
        """True when any counter is nonzero, i.e. a sidecar would carry something."""
        return any(value for name, value in self.counts.items() if name not in {"apid", "n_packets", "n_samples"})

    def to_qa_report(self, thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS) -> dict[str, Any]:
        """Return the ``.qa.json`` body: the counters plus the evidence behind them."""
        report = {name.name: getattr(self, name.name) for name in fields(self) if name.name != "duplicate_evidence"}
        report["quality_flag"] = str(self.quality_flag(thresholds))
        report["duplicate_evidence"] = [asdict(evidence) for evidence in self.duplicate_evidence]
        return report

    def write_qa_report(self, path, thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS) -> None:
        """Write the ``.qa.json`` sidecar.

        Parameters
        ----------
        path : Path or S3Path
            Destination, normally ``LiberaDataProductFilename.qa_report_filename``.
        thresholds : QualityThresholds, optional
            Passed to :meth:`quality_flag`.
        """
        with path.open("w") as report_file:
            json.dump(self.to_qa_report(thresholds), report_file, indent=2, default=_json_default)
        logger.info({"msg": "Wrote L1A QA report sidecar", "path": str(path), "product_id": self.product_id})


def _json_default(value: Any) -> Any:
    """Render numpy scalars and datetimes that ``json`` cannot serialize on its own."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.datetime64):
        return str(value)
    return str(value)

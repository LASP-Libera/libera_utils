"""Per-granule L1A data-quality counters and their rollup flag.

Every L1A granule carries the same counters as NetCDF global attributes, zeros included, so a
quality question is a trend over granules rather than a search for exception reports. The events
these count are not rare: in DITL2, RAD packet time steps backward on ~1-2% of packets and ~1.5%
of packets carry sample timestamps that collide with an earlier packet's. What matters is how
those rates move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from libera_utils.l1a.day_coverage import DayCoverageResult, TimeAxisCoverage


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


# Coverage of the applicable day, written on every L1A product alongside the quality attributes.
# Kept apart from QUALITY_GLOBAL_ATTRIBUTES so each set can be exported to CMR additional
# attributes on its own: CMR rejects a granule carrying a name its parent collection does not
# declare.
COVERAGE_GLOBAL_ATTRIBUTES = (
    "CoverageMode",
    "CoverageGateForced",
    "L0DayCoverageFraction",
    "L0LeftBufferCoverageFraction",
    "L0RightBufferCoverageFraction",
    "PacketTimeDayCoverageFraction",
    "DataTimeDayCoverageFraction",
)

# Written where an axis was not available to measure, so the attribute is still present.
COVERAGE_NOT_MEASURED = float("nan")

# ``CoverageMode`` on a granule that did not come from the day-combine path at all: a chunk
# parsed directly, or anything written before the day gates ran.
COVERAGE_MODE_NOT_GATED = "not_gated"


def coverage_global_attributes(
    gate: DayCoverageResult | None = None,
    *,
    forced: bool = False,
    packet_time_coverage: TimeAxisCoverage | None = None,
    data_time_coverage: TimeAxisCoverage | None = None,
) -> dict[str, Any]:
    """Return the NetCDF coverage attributes for a granule.

    The gate fractions and the measured ones answer different questions and can disagree by a
    lot. ``L0*`` comes from whole-file L0 spans, which cannot see a gap inside a file; the
    measured fractions come from the granule's own axes. For an APID whose data time runs on a
    different clock than its packet time, the two axes disagree as well, which is why both are
    recorded.

    Parameters
    ----------
    gate : DayCoverageResult | None, optional
        Result the combine decision was made on. ``None`` (the default) records a granule that
        did not come from the day-combine path, so every product carries the attributes its
        definition declares whoever wrote it.
    forced : bool, optional
        True when the gates were skipped by an operator request.
    packet_time_coverage : TimeAxisCoverage | None, optional
        Measured from the granule's packet time axis. ``None`` records "not measured".
    data_time_coverage : TimeAxisCoverage | None, optional
        Measured from the granule's science data time axis, when it has one distinct from
        packet time. ``None`` records "not measured".

    Returns
    -------
    dict
        Every name in :data:`COVERAGE_GLOBAL_ATTRIBUTES`.
    """
    attributes = {
        "CoverageMode": COVERAGE_MODE_NOT_GATED if gate is None else str(gate.mode),
        "CoverageGateForced": int(forced),
        "L0DayCoverageFraction": COVERAGE_NOT_MEASURED if gate is None else float(gate.day_frac),
        "L0LeftBufferCoverageFraction": COVERAGE_NOT_MEASURED if gate is None else float(gate.left_frac),
        "L0RightBufferCoverageFraction": COVERAGE_NOT_MEASURED if gate is None else float(gate.right_frac),
        "PacketTimeDayCoverageFraction": (
            COVERAGE_NOT_MEASURED if packet_time_coverage is None else float(packet_time_coverage.day_frac)
        ),
        "DataTimeDayCoverageFraction": (
            COVERAGE_NOT_MEASURED if data_time_coverage is None else float(data_time_coverage.day_frac)
        ),
    }
    assert set(attributes) == set(COVERAGE_GLOBAL_ATTRIBUTES)  # noqa: S101 - guards the YAML contract
    return attributes


@dataclass
class GranuleQualityRecord:
    """Every quality counter for one granule.

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

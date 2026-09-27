"""Unit tests for manifest-driven auxiliary input selection in ``footprint_match_algorithm``.

These cover :func:`select_auxiliary_files` (group a flat manifest's auxiliary granules by reader)
and the manifest-vs-directory-tree precedence in :func:`resolve_ancillary_inputs`. Manifests are
built from :class:`ManifestFileRecord` with explicit checksums so no files need to exist on disk.
"""

from __future__ import annotations

from libera_utils.footprint_matching.footprint_match_algorithm import (
    resolve_ancillary_inputs,
    select_auxiliary_files,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.manifest import Manifest, ManifestFileRecord, ManifestType

# Auxiliary granule basenames (from example_data), one per reader family.
_IGBP = "MCD12Q1.A2024001.h21v07.061.2025205222320.hdf"
_NISE = "NISE_AMSR2_20260610.HDFEOS"
_BRDF = "VJ143C1.A2026153.002.2026161161054.h5"
_CLOUD = "CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710.nc"
_AOD = "AERDB_D3_VIIRS_NOAA20.A2020121.001.2024121023016.nc"
_SSF = "CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115.nc"
_L1B = "LIBERA_L1B_RAD-4CH_V0-5-6_20251120T175950_20251120T180020_R26190195454.nc"


def _manifest(*basenames: str) -> Manifest:
    """Build an INPUT manifest of absolute-path records with distinct dummy checksums."""
    files = [
        ManifestFileRecord(filename=f"/staged/{name}", checksum=f"checksum-{index}")
        for index, name in enumerate(basenames)
    ]
    return Manifest(manifest_type=ManifestType.INPUT, files=files)


class TestSelectAuxiliaryFiles:
    def test_groups_granules_by_reader(self):
        # Targets basic grouping; asserts each auxiliary granule lands under its reader key.
        grouped = select_auxiliary_files(_manifest(_IGBP, _NISE, _BRDF, _CLOUD), OperationalMode.CAM)
        assert [p.name for p in grouped["igbp"]] == [_IGBP]
        assert [p.name for p in grouped["nise"]] == [_NISE]
        assert [p.name for p in grouped["viirs_brdf"]] == [_BRDF]
        assert [p.name for p in grouped["viirs_cloud"]] == [_CLOUD]

    def test_keys_are_exactly_the_active_reader_set(self):
        # Targets the return shape; asserts one entry per active reader (matches resolve_ancillary_inputs).
        grouped = select_auxiliary_files(_manifest(_IGBP), OperationalMode.CAM)
        assert set(grouped) == set(ReaderRegistry.get_readers_for_mode(OperationalMode.CAM))

    def test_readers_with_nothing_staged_map_to_empty_list(self):
        # Targets the empty-reader contract; asserts active readers with no granule are empty lists
        # while the staged one is populated.
        grouped = select_auxiliary_files(_manifest(_IGBP), OperationalMode.CAM)
        assert [p.name for p in grouped["igbp"]] == [_IGBP]
        assert grouped["nise"] == []
        assert grouped["viirs_cloud"] == []

    def test_skips_non_auxiliary_records(self):
        # Targets that Libera and unrelated files are ignored; asserts only the auxiliary granule is kept.
        grouped = select_auxiliary_files(_manifest(_L1B, "notes.txt", _IGBP), OperationalMode.CAM)
        assert [p.name for p in grouped["igbp"]] == [_IGBP]
        assert sum(len(v) for v in grouped.values()) == 1

    def test_skips_granule_for_reader_not_active_in_mode(self):
        # viirs_aod / ssf are climate-quality readers absent from CAM; asserts their granules are
        # not selected for a CAM run and their reader keys are not present.
        grouped = select_auxiliary_files(_manifest(_AOD, _SSF, _IGBP), OperationalMode.CAM)
        assert "viirs_aod" not in grouped
        assert "ssf" not in grouped
        assert sum(len(v) for v in grouped.values()) == 1  # only the IGBP granule

    def test_same_granule_selected_when_reader_is_active(self):
        # The IMAGER mode does include viirs_aod and ssf; asserts those same granules are selected there.
        grouped = select_auxiliary_files(_manifest(_AOD, _SSF), OperationalMode.IMAGER)
        assert [p.name for p in grouped["viirs_aod"]] == [_AOD]
        assert [p.name for p in grouped["ssf"]] == [_SSF]

    def test_multiple_granules_for_one_reader_are_sorted(self):
        # A reader may be staged more than one granule; asserts they come back sorted by name.
        older = "MCD12Q1.A2023001.h21v07.061.2024205222320.hdf"
        grouped = select_auxiliary_files(_manifest(_IGBP, older), OperationalMode.CAM)
        assert [p.name for p in grouped["igbp"]] == sorted([_IGBP, older])


class TestResolveAncillaryInputsManifestPrecedence:
    def test_manifest_with_granules_takes_precedence_over_tree(self, tmp_path, monkeypatch):
        # Targets manifest-over-tree precedence; asserts the tree is NOT consulted when the manifest
        # stages auxiliary granules (a granule the tree would have added is absent).
        tree = tmp_path / "ancillary"
        (tree / "igbp").mkdir(parents=True)
        (tree / "igbp" / "tree_only.hdf").write_text("x")
        monkeypatch.setenv("FMATCH_ANCILLARY_PATH", str(tree))

        resolved = resolve_ancillary_inputs(OperationalMode.CAM, manifest=_manifest(_IGBP))
        assert [p.name for p in resolved["igbp"]] == [_IGBP]  # from the manifest, not tree_only.hdf

    def test_falls_back_to_tree_when_manifest_has_no_auxiliary_granules(self, tmp_path, monkeypatch):
        # Targets the fallback; asserts a manifest carrying only an L1B file defers to the directory tree.
        tree = tmp_path / "ancillary"
        (tree / "igbp").mkdir(parents=True)
        (tree / "igbp" / "tree_only.hdf").write_text("x")
        monkeypatch.setenv("FMATCH_ANCILLARY_PATH", str(tree))

        resolved = resolve_ancillary_inputs(OperationalMode.CAM, manifest=_manifest(_L1B))
        assert [p.name for p in resolved["igbp"]] == ["tree_only.hdf"]

    def test_manifest_none_uses_tree(self, tmp_path, monkeypatch):
        # Targets backward compatibility; asserts omitting the manifest keeps the directory-tree behavior.
        tree = tmp_path / "ancillary"
        (tree / "igbp").mkdir(parents=True)
        (tree / "igbp" / "tree_only.hdf").write_text("x")
        monkeypatch.setenv("FMATCH_ANCILLARY_PATH", str(tree))

        resolved = resolve_ancillary_inputs(OperationalMode.CAM)
        assert [p.name for p in resolved["igbp"]] == ["tree_only.hdf"]

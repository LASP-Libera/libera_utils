"""Unit tests for the SCENE-ID-CAM daily-processing runner (FMATCH-CAM input selection)."""

from types import SimpleNamespace

from libera_utils.scene_identification.cam.scene_id_cam import collect_fmatch_cam_input_files


def _fake_manifest(filenames):
    """Build a minimal stand-in for a Manifest exposing only what the collector reads.

    ``collect_fmatch_cam_input_files`` only accesses ``manifest.files`` and each record's ``filename``, so a
    lightweight namespace keeps the test focused on the selection logic rather than Manifest construction.
    """
    files = [SimpleNamespace(filename=name) for name in filenames]
    return SimpleNamespace(files=files)


class TestCollectFmatchCamInputFiles:
    """Test cases for collect_fmatch_cam_input_files."""

    # A real, parseable FMATCH-CAM Libera product filename (ProductID FMATCH-CAM).
    FMATCH_CAM_FILE = "LIBERA_ANC_FMATCH-CAM_V0-1-0_20280212T033945_20280212T052007_R26175214729.nc"
    # A different Libera product (parses, but is not FMATCH-CAM) that might share a manifest.
    FMATCH_IMAGER_FILE = "LIBERA_ANC_FMATCH-IMAGER_V0-1-0_20280212T033945_20280212T052007_R26175214729.nc"
    # A non-Libera filename (e.g. a raw CERES SSF input) that must never be selected here.
    CERES_SSF_FILE = "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"

    def test_selects_only_fmatch_cam_files(self):
        """Only files whose Libera product ID is FMATCH-CAM are returned, in manifest order."""
        manifest = _fake_manifest(
            [self.CERES_SSF_FILE, self.FMATCH_CAM_FILE, self.FMATCH_IMAGER_FILE, self.FMATCH_CAM_FILE]
        )
        selected = collect_fmatch_cam_input_files(manifest)
        assert selected == [self.FMATCH_CAM_FILE, self.FMATCH_CAM_FILE]

    def test_ignores_non_fmatch_cam_libera_products(self):
        """A manifest of only other Libera products yields no FMATCH-CAM inputs."""
        manifest = _fake_manifest([self.FMATCH_IMAGER_FILE])
        assert collect_fmatch_cam_input_files(manifest) == []

    def test_ignores_unparseable_names(self):
        """Non-Libera-product filenames are skipped rather than raising."""
        manifest = _fake_manifest([self.CERES_SSF_FILE, "not_a_libera_file.txt"])
        assert collect_fmatch_cam_input_files(manifest) == []

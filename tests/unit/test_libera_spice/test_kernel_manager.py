"""
Unit tests for kernel_manager.py

These tests focus on the KernelManager's state management, lifecycle,
and error handling. They use real temporary directories where possible
to minimize mocking complexity.
"""

import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from libera_utils.io.caching import get_local_short_temp_dir
from libera_utils.libera_spice.kernel_manager import KernelManager

# ============================================================================
# KernelManager Initialization Tests
# ============================================================================


class TestKernelManagerInitialization:
    """Test KernelManager initialization and configuration."""

    def test_initial_state(self):
        """KernelManager should start with no kernels loaded."""
        km = KernelManager()

        assert km._loaded_kernels is None
        assert km._static_loaded is False
        assert km._dynamic_loaded is False
        assert km._static_kernels_path is None

    def test_default_temp_base(self):
        """Should use default temp base when none provided."""
        km = KernelManager()

        # Should match the platform default
        expected = get_local_short_temp_dir()
        assert km._temp_base == expected

    def test_custom_temp_base(self, tmp_path):
        """Should use custom temp base when provided."""
        custom_base = tmp_path / "custom"
        km = KernelManager(temp_dir_base=custom_base)

        assert km._temp_base == custom_base

    def test_validation_enabled_by_default(self):
        """Path validation should be enabled by default and set to 80 characters."""
        assert KernelManager._max_path_length == 80

    def test_multiple_instances_are_independent(self):
        """Multiple KernelManager instances should have independent state."""
        km1 = KernelManager()
        km2 = KernelManager()

        # Modify km1's internal state
        km1._static_loaded = True

        # km2 should be unaffected
        assert km1._static_loaded is True
        assert km2._static_loaded is False

        assert km1 is not km2


# ============================================================================
# Temporary Directory Management Tests
# ============================================================================


class TestTemporaryDirectoryManagement:
    """Test temporary directory creation and cleanup."""

    def test_delete_static_kernels_removes_directory(self, tmp_path):
        """_delete_static_kernels should remove the temp directory."""
        km = KernelManager(temp_dir_base=tmp_path)

        # Create a fake temp directory
        fake_kernel_dir = tmp_path / "fake_kernels"
        fake_kernel_dir.mkdir()
        (fake_kernel_dir / "test.bsp").touch()

        km._static_kernels_path = fake_kernel_dir

        assert fake_kernel_dir.exists()

        km._delete_temporary_static_kernels()

        assert not fake_kernel_dir.exists()
        assert km._static_kernels_path is None

    def test_delete_static_kernels_handles_missing_directory(self):
        """Should handle gracefully if directory doesn't exist."""
        km = KernelManager()
        km._static_kernels_path = Path("/nonexistent/path")

        # Should not raise
        km._delete_temporary_static_kernels()

        assert km._static_kernels_path is None

    def test_delete_static_kernels_does_nothing_if_none(self):
        """Should do nothing if _static_kernels_path is None."""
        km = KernelManager()

        # Should not raise
        km._delete_temporary_static_kernels()

    @patch("libera_utils.config.config.get")
    @patch("libera_utils.kernel_maker.make_kernel")
    def test_cleanup_on_failed_kernel_creation(self, mock_make_kernel, mock_config_get, tmp_path):
        """Should clean up temp directory if kernel creation fails."""
        km = KernelManager(temp_dir_base=tmp_path)

        mock_config_get.return_value = [str(tmp_path / "dummy_config.json")]
        (tmp_path / "dummy_config.json").touch()

        # Make kernel creation fail
        mock_make_kernel.side_effect = RuntimeError("Kernel creation failed")

        # Count directories before
        dirs_before = len(list(tmp_path.iterdir()))

        with pytest.raises(RuntimeError):
            km._create_temporary_static_kernels()

        # Should have cleaned up the temp directory
        dirs_after = len(list(tmp_path.iterdir()))
        assert dirs_after == dirs_before  # No new directories left behind


# ============================================================================
# State Transition Tests
# ============================================================================


class TestKernelManagerStateTransitions:
    """Test state transitions as kernels are loaded."""

    def test_unload_resets_state(self):
        """unload_all should reset the manager to initial state."""
        km = KernelManager()

        # Simulate loaded state
        km._static_loaded = True
        km._dynamic_loaded = True
        mock_loaded = MagicMock()
        km._loaded_kernels = mock_loaded

        # Unload
        km.unload_all()

        # Should be back to initial state
        assert km._loaded_kernels is None
        assert km._static_loaded is False
        assert km._dynamic_loaded is False


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestContextManagerProtocol:
    """Test context manager behavior."""

    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_naif_kernels")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_static_kernels")
    def test_context_manager_loads_on_enter(self, mock_load_static, mock_load_naif, tmp_path):
        """Context manager should load static kernels on __enter__."""
        km = KernelManager(temp_dir_base=tmp_path)

        assert km._loaded_kernels is None

        with km:
            mock_load_static.assert_called_once()
            mock_load_naif.assert_called_once()

    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_naif_kernels")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_static_kernels")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.unload_all")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager._delete_temporary_static_kernels")
    def test_context_manager_cleans_up_on_exit(
        self, mock_delete, mock_unload, mock_load_static, mock_load_naif, tmp_path
    ):
        """Context manager should clean up on __exit__."""
        km = KernelManager(temp_dir_base=tmp_path)
        km._loaded_kernels = MagicMock()

        with km:
            pass

        mock_unload.assert_called()
        mock_delete.assert_called()

    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_naif_kernels")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_static_kernels")
    def test_context_manager_returns_manager_instance(self, mock_load_static, mock_load_naif, tmp_path):
        """Context manager should return the KernelManager instance."""
        km = KernelManager(temp_dir_base=tmp_path)

        with km as returned_km:
            assert returned_km is km

    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_naif_kernels")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_static_kernels")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.unload_all")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager._delete_temporary_static_kernels")
    def test_context_manager_cleans_up_even_on_exception(
        self, mock_delete, mock_unload, mock_load_static, mock_load_naif, tmp_path
    ):
        """Should clean up temp files even if exception occurs."""
        km = KernelManager(temp_dir_base=tmp_path)
        km._loaded_kernels = MagicMock()

        try:
            with km:
                raise ValueError("Test exception")
        except ValueError:
            pass

        mock_unload.assert_called()
        mock_delete.assert_called()


# ============================================================================
# Static Kernel Loading Tests
# ============================================================================


class TestStaticKernelLoading:
    """Test static kernel loading behavior."""

    @patch("libera_utils.libera_spice.kernel_manager.KernelManager._create_temporary_static_kernels")
    def test_load_static_skips_if_already_loaded(self, mock_create, tmp_path):
        """load_static_kernels should skip if already loaded."""
        km = KernelManager(temp_dir_base=tmp_path)
        km._static_loaded = True

        km.load_static_kernels()

        mock_create.assert_not_called()

    @patch("libera_utils.config.config.get")
    @patch("libera_utils.kernel_maker.make_kernel")
    def test_load_static_cleans_up_on_failure(self, mock_make_kernel, mock_config_get, tmp_path):
        """Should clean up temp directory if loading fails."""
        km = KernelManager(temp_dir_base=tmp_path)

        config_file = tmp_path / "kernel_config.json"
        config_file.touch()
        mock_config_get.return_value = [str(config_file)]

        # Make kernel creation succeed but loading fail
        def create_kernel(config, output_dir, input_data):
            (Path(output_dir) / "test.bsp").touch()

        mock_make_kernel.side_effect = create_kernel

        with patch("curryer.meta.MetaKernel.from_json", side_effect=RuntimeError("Load failed")):
            with pytest.raises(RuntimeError):
                km.load_static_kernels()

        # Should have cleaned up
        assert km._static_kernels_path is None

    @patch("libera_utils.config.config.get")
    @patch("shutil.copy")
    def test_create_static_copy_failure_cleanup(self, mock_copy, mock_config_get, tmp_path):
        """Test that temp directory is cleaned up if a file copy fails."""
        km = KernelManager(temp_dir_base=tmp_path)

        # Mock config to return valid paths so we reach the copy loop
        mock_config_get.return_value = str(tmp_path)

        # Trigger an OSError during copy
        mock_copy.side_effect = OSError("Disk full or permission denied")

        # We expect a RuntimeError wrapping the OSError
        with pytest.raises(RuntimeError, match="Static kernel creation failed"):
            km._create_temporary_static_kernels()

        # Verify cleanup happened (no temp dirs left in base)
        # Note: We check iterdir() on the base temp path
        assert len(list(tmp_path.iterdir())) == 0

    def test_create_static_path_length_error(self, tmp_path):
        """Test that exceeding path length limits raises error."""
        KernelManager._max_path_length = 5  # Set low for testing
        km = KernelManager(temp_dir_base=tmp_path)

        with pytest.raises(RuntimeError, match="Static kernel creation failed"):
            km._create_temporary_static_kernels()

        KernelManager._max_path_length = 80  # Reset to default

    @patch("libera_utils.libera_spice.kernel_manager.KernelManager._create_temporary_static_kernels")
    def test_load_static_empty_directory_error(self, mock_create, tmp_path):
        """Test error when static kernel directory exists but is empty."""
        km = KernelManager(temp_dir_base=tmp_path)

        # Create an empty directory and have _create_static_kernels return it
        empty_dir = tmp_path / "empty_static"
        empty_dir.mkdir()
        mock_create.return_value = empty_dir

        # Mock config so metakernel loading doesn't crash before our check
        with patch("curryer.meta.MetaKernel.from_json"):
            with pytest.raises(FileNotFoundError, match="No static kernels found"):
                km.load_static_kernels()


# ============================================================================
# NAIF Kernel Loading Tests
# ============================================================================


class TestNaifKernelLogic:
    """Tests for NAIF kernel URL construction and logic."""

    @patch("libera_utils.config.config.get")
    @patch("libera_utils.libera_spice.kernel_manager.KernelFileCache")  # Mocking the downloader
    @patch("libera_utils.libera_spice.kernel_manager.find_most_recent_naif_kernel")  # Mocking the scraper
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_static_kernels")
    def test_load_naif_kernels_url_construction(
        self, mock_load_static, mock_find, mock_cache, mock_config_get, tmp_path
    ):
        """Test that regex patterns map to the correct remote URLs without network calls."""
        km = KernelManager(temp_dir_base=tmp_path)

        # Mock internal SPICE loader to avoid actual loading of fake files
        km._loaded_kernels = MagicMock()
        mock_config_get.return_value = str(tmp_path)

        # 1. Mock the "scraper" to return a fake remote filename
        mock_find.return_value = "https://fake-url.com/naif0012.tls"

        # 2. Mock the "downloader" (KernelFileCache)
        # The code expects: file_cache = KernelFileCache(found_file)
        # And then uses: file_cache.kernel_path
        mock_cache_instance = mock_cache.return_value
        mock_cache_instance.kernel_path = Path(tmp_path) / "naif0012.tls"

        km.load_naif_kernels()

        # Check the logic of URL construction
        urls_called = [call_args.args[0] for call_args in mock_find.call_args_list]

        # We expect specific subdirectories based on the regex map in KernelManager
        assert any("generic_kernels/lsk/" in url for url in urls_called)
        assert any("generic_kernels/pck/" in url for url in urls_called)
        assert any("generic_kernels/spk/planets/" in url for url in urls_called)
        # Note: 'fk' checks usually depend on if earth_assoc_itrf93.tf is in the regex list
        if any("earth_assoc_itrf93" in r for r in km._high_precision_earth_regexs):
            assert any("generic_kernels/fk/planets/" in url for url in urls_called)

        # Verification 3: Ensure the Cache was initialized with the result from the 'scraper'
        # This confirms the data flows from find_most_recent -> KernelFileCache correctly
        mock_cache.assert_has_calls(
            [call("https://fake-url.com/naif0012.tls", max_cache_age=datetime.timedelta(days=7))]
            * mock_find.call_count,
        )

    @patch("libera_utils.libera_spice.kernel_manager.find_most_recent_naif_kernel")
    def test_naif_test_url_flag(self, mock_find, tmp_path):
        """Test that use_test_naif_url=True prevents appending subdirectories."""
        km = KernelManager(temp_dir_base=tmp_path, use_test_naif_url=True)
        km._loaded_kernels = MagicMock()
        mock_find.return_value = "dummy_file"

        # We don't need to mock KernelFileCache here necessarily if find_most_recent_naif_kernel
        # is mocked, but it's safer to add it or let it fail on the next line if strict.
        # For this specific test, we only care about the URL passed to mock_find.

        try:
            km.load_naif_kernels()
        except Exception:
            # We can ignore errors that happen AFTER the URL construction
            # (like KernelFileCache failing because we didn't mock it)
            pass

        urls_called = [call_args.args[0] for call_args in mock_find.call_args_list]

        # Verify NO subdirectories were appended
        for url in urls_called:
            assert url.endswith("generic_kernels/")


# ============================================================================
# Dynamic Kernel Loading Tests
# ============================================================================


class TestDynamicKernelLoading:
    """Test dynamic kernel loading behavior."""

    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_naif_kernels")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_static_kernels")
    def test_load_dynamic_loads_static_if_needed(self, mock_load_naif, mock_load_static, tmp_path):
        """Should automatically load static and Naif kernels if not loaded."""
        km = KernelManager(temp_dir_base=tmp_path)

        # Create dummy kernel files
        kernel_dir = tmp_path / "kernels"
        kernel_dir.mkdir()
        (kernel_dir / "test.bsp").touch()

        km._loaded_kernels = MagicMock()
        km._loaded_kernels._iter_load = MagicMock()
        km.load_libera_dynamic_kernels(
            dynamic_kernel_directory=str(kernel_dir), needs_static_kernels=False, needs_naif_kernels=False
        )
        assert not mock_load_naif.called
        assert not mock_load_static.called
        km._loaded_kernels._iter_load.assert_called_once()

        km.load_libera_dynamic_kernels(
            dynamic_kernel_directory=str(kernel_dir), needs_static_kernels=True, needs_naif_kernels=True
        )

        mock_load_static.assert_called_once()
        mock_load_naif.assert_called_once()
        assert km._loaded_kernels._iter_load.call_count == 2  # Called again

    def test_load_dynamic_with_nonexistent_path_raises_error(self):
        """Should raise FileNotFoundError for nonexistent paths."""
        km = KernelManager()
        km._static_loaded = True

        with pytest.raises(FileNotFoundError) as exc_info:
            km.load_libera_dynamic_kernels("/nonexistent/path")

        assert "No such file or directory" in str(exc_info.value)

    def test_load_dynamic_with_empty_directory_raises_error(self, tmp_path):
        """Should raise FileNotFoundError if no kernel files found."""
        km = KernelManager(temp_dir_base=tmp_path)
        km._static_loaded = True

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            km.load_libera_dynamic_kernels(empty_dir)

        assert "No kernel files found" in str(exc_info.value)

    def test_load_dynamic_with_directory_parameter(self, tmp_path):
        """Should load all files from dynamic_kernel_directory parameter."""
        km = KernelManager(temp_dir_base=tmp_path)
        km._static_loaded = True
        km._naif_kernels_loaded = True
        km._loaded_kernels = MagicMock()

        # Create directory with multiple kernels
        kernel_dir = tmp_path / "kernels"
        kernel_dir.mkdir()
        (kernel_dir / "kernel1.bsp").touch()
        (kernel_dir / "kernel2.ck").touch()
        (kernel_dir / "kernel3.spk").touch()

        km.load_libera_dynamic_kernels(dynamic_kernel_directory=str(kernel_dir))

        assert km._dynamic_loaded is True
        # Should have loaded all 3 files
        km._loaded_kernels._iter_load.assert_called_once()
        loaded_filenames = [f.name for f in km._loaded_kernels._iter_load.call_args[0][0]]
        assert "kernel1.bsp" in loaded_filenames
        assert "kernel2.ck" in loaded_filenames
        assert "kernel3.spk" in loaded_filenames

    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_naif_kernels")
    @patch("libera_utils.libera_spice.kernel_manager.KernelManager.load_static_kernels")
    def test_load_dynamic_triggers_missing_dependencies(self, mock_load_static, mock_load_naif, tmp_path):
        """Test that loading dynamic kernels triggers static and NAIF loading if missing."""
        km = KernelManager(temp_dir_base=tmp_path)

        # Ensure flags start as False
        km._static_loaded = False
        km._naif_kernels_loaded = False

        # Create a dummy dynamic kernel to load so the function doesn't fail early
        d_path = tmp_path / "dynamic"
        d_path.mkdir()
        (d_path / "test.bsp").touch()

        # Mock the internal loader
        km._loaded_kernels = MagicMock()

        km.load_libera_dynamic_kernels(d_path)

        # Assert dependencies were called
        mock_load_static.assert_called_once()
        mock_load_naif.assert_called_once()


# ============================================================================
# Destructor Tests
# ============================================================================


class TestDestructor:
    """Test __del__ cleanup behavior."""

    def test_destructor_cleans_up_temp_directory(self, tmp_path):
        """__del__ should clean up temp directory."""
        km = KernelManager(temp_dir_base=tmp_path)

        # Create a fake temp directory
        fake_kernel_dir = tmp_path / "fake_kernels"
        fake_kernel_dir.mkdir()
        km._static_kernels_path = fake_kernel_dir

        assert fake_kernel_dir.exists()

        # Trigger destructor
        km.__del__()

        assert not fake_kernel_dir.exists()

    def test_destructor_handles_exceptions_silently(self):
        """__del__ should not raise exceptions."""
        km = KernelManager()
        km._static_kernels_path = Path("/nonexistent")

        # Should not raise
        km.__del__()

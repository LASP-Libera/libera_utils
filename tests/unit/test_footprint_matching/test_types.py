"""Unit tests for the shared FMATCH types (libera_utils.footprint_matching.types)."""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.types import (
    BoundingBox,
    OperationalMode,
    ProductStack,
    RadiometerFootprint,
    VariableSpec,
    with_standard_deviation_companions,
)

_BOX = BoundingBox(lat_min=9.0, lat_max=11.0, lon_min=19.0, lon_max=21.0)


def _footprint(**overrides) -> RadiometerFootprint:
    kwargs = {
        "bbox": _BOX,
        "time": np.datetime64("2026-07-12T00:00:00"),
        "latitude": 10.0,
        "longitude": 20.0,
        "viewing_zenith_angle": 30.0,
        "subsatellite_latitude": 12.0,
        "subsatellite_longitude": 20.0,
        "satellite_radius_km": 7213.0,
    }
    kwargs.update(overrides)
    return RadiometerFootprint(**kwargs)


class TestRadiometerFootprintBox:
    """``bbox`` is ``None`` exactly when the footprint is ``off_limb``."""

    def test_on_earth_footprint_carries_a_box(self):
        assert _footprint().bbox == _BOX

    def test_off_limb_footprint_has_no_box(self):
        footprint = _footprint(bbox=None, off_limb=True)
        assert footprint.bbox is None

    def test_missing_box_on_earth_raises(self):
        with pytest.raises(ValueError, match="no bbox but off_limb is False"):
            _footprint(bbox=None)

    def test_box_on_off_limb_footprint_raises(self):
        with pytest.raises(ValueError, match="off_limb but was given a bbox"):
            _footprint(off_limb=True)


class TestVariableSpec:
    """``modes`` is a non-empty frozenset of ``OperationalMode`` naming every product that carries the variable."""

    def test_valid_spec_is_carried_by_exactly_its_modes(self):
        """``VariableSpec`` keeps the ``modes`` it is given and stays hashable.

        ``modes`` decides which FMATCH products carry the variable, and so which readers each product runs; a spec
        that gained or lost a mode would add the variable to a product that does not declare it or drop it from one
        that does. ``VariableSpec`` is a frozen dataclass, so a spec holding an unhashable ``modes`` could not be used
        as a set member or dict key.

        Builds a spec carried by FMATCH-CAM alone and asserts CAM is in its modes, IMAGER is not, and the spec can be
        put in a set.
        """
        spec = VariableSpec(
            name="v", dtype="float32", aggregation="weighted_mean", modes=frozenset({OperationalMode.CAM})
        )
        assert OperationalMode.CAM in spec.modes
        assert OperationalMode.IMAGER not in spec.modes
        assert spec in {spec}

    def test_missing_modes_raises(self):
        """``VariableSpec`` raises ``TypeError`` when it is built without ``modes``.

        A default would quietly place a variable in some products and leave it out of others; requiring the argument
        makes every reader author say which products carry each variable.

        Builds a spec with no ``modes`` argument and asserts a ``TypeError`` naming ``modes``.
        """
        with pytest.raises(TypeError, match="modes"):
            VariableSpec(name="v", dtype="float32", aggregation="weighted_mean")

    def test_empty_modes_raises(self):
        """``VariableSpec`` raises ``ValueError`` when ``modes`` is empty.

        A variable carried by no product is never written, so declaring one is a mistake that would otherwise only
        show up as a variable missing from every product.

        Builds a spec with ``frozenset()`` and asserts a ``ValueError`` that names the spec and says to name at least
        one product.
        """
        with pytest.raises(ValueError, match="VariableSpec 'v' has empty modes: name at least one product"):
            VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", modes=frozenset())

    @pytest.mark.parametrize(
        "modes",
        [
            pytest.param((OperationalMode.CAM,), id="tuple"),
            pytest.param({OperationalMode.CAM}, id="set"),
            pytest.param(frozenset({"FMATCH-CAM"}), id="string-member"),
        ],
    )
    def test_modes_that_are_not_a_frozenset_of_modes_raise(self, modes):
        """``VariableSpec`` raises ``TypeError`` unless ``modes`` is a frozenset of ``OperationalMode`` members.

        Membership is tested with ``mode in spec.modes``: a product-ID string such as ``"FMATCH-CAM"`` never equals
        the enum member, so the variable would silently drop out of the product, and a mutable set would make the
        spec unhashable.

        Parametrized over a tuple, a plain set and a frozenset holding the string ``"FMATCH-CAM"``; asserts each
        raises ``TypeError`` saying to pass a frozenset of ``OperationalMode``.
        """
        with pytest.raises(TypeError, match="VariableSpec 'v' has modes=.*pass a frozenset of OperationalMode"):
            VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", modes=modes)

    def test_standard_deviation_companion_has_its_parents_modes(self):
        """``with_standard_deviation_companions`` gives each companion the same ``modes`` as its parent.

        A continuous variable and its spread belong together; a companion with other modes would put a standard
        deviation in a product without its mean, or leave the mean without its spread.

        Builds a weighted-mean parent carried by FMATCH-IMAGER and FMATCH-IMAGER-CAMTIME and asserts its companion is
        named ``v_standard_deviation`` and carries the same two modes.
        """
        modes = frozenset({OperationalMode.IMAGER, OperationalMode.IMAGER_CAMTIME})
        parent = VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", modes=modes)
        _, companion = with_standard_deviation_companions((parent,))
        assert companion.name == "v_standard_deviation"
        assert companion.modes == modes


class TestProductStack:
    """A ``ProductStack`` has one distinct member per coordinate value."""

    @pytest.mark.parametrize(
        ("members", "coordinate_values", "message"),
        [
            pytest.param((), (), "no members", id="empty"),
            pytest.param(("t_1", "t_1"), (1, 2), "more than once", id="duplicate"),
            pytest.param(("t_1", "t_2"), (1,), "coordinate values", id="length-mismatch"),
        ],
    )
    def test_invalid_stack_raises(self, members, coordinate_values, message):
        """An empty, repeating or length-mismatched ``ProductStack`` raises ``ValueError`` at construction.

        Element ``k`` of a stacked variable is member ``k`` at coordinate value ``k``, so a stack with no members,
        a member used twice, or a member without a coordinate value would publish a level with the wrong value or
        none.

        Each case constructs the stack and matches the message naming the defect.
        """
        with pytest.raises(ValueError, match=message):
            ProductStack(name="t", dimension="PRESSURE_LEVEL", coordinate_values=coordinate_values, members=members)

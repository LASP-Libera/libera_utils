"""Tests for standard Libera dimension definitions."""

from libera_utils.io.product_definition import LiberaDimensionDefinition, LiberaVariableDefinition

NEW_CAL_PACKET_DIMENSIONS = [
    "NOM_HK_PACKET",
    "PEV_SW_PACKET",
    "PEC_SW_PACKET",
    "RAD_SAMPLE_PACKET",
    "RAD_FULL_PACKET",
    "AXIS_SAMPLE_PACKET",
    "CAL_FULL_PACKET",
    "CAL_SAMPLE_PACKET",
    "WFOV_SCI_PACKET",
]


def test_standard_dimensions_load_and_parse():
    """Test the global dimensions file includes the new calibration packet dimensions."""
    dimensions = LiberaVariableDefinition._get_standard_dimensions()

    for dimension_name in NEW_CAL_PACKET_DIMENSIONS:
        assert dimension_name in dimensions
        assert isinstance(dimensions[dimension_name], LiberaDimensionDefinition)
        assert dimensions[dimension_name].size is None
        assert dimensions[dimension_name].long_name


def test_field_validator_accepts_new_dimensions():
    """Test a variable definition can use a newly-added packet index dimension."""
    variable_definition = LiberaVariableDefinition(
        dtype="int32",
        attributes={"long_name": "Test packet index"},
        dimensions=["RAD_SAMPLE_PACKET"],
    )
    assert variable_definition.dimensions == ["RAD_SAMPLE_PACKET"]

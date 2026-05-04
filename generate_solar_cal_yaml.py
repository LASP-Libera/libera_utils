"""
Generates solar_cal_face1_l1a.yml and solar_cal_face2_l1a.yml
from the three source product definitions.

Run once from the libera_utils repo root:
    python generate_solar_cal_yaml.py
"""

import copy
from pathlib import Path

import yaml

DEF_DIR = Path("libera_utils/data/product_definitions")

# Variables that collide across all three source APIDs (CCSDS packet headers + spare)
CONFLICTS = frozenset(
    {"VERSION", "TYPE", "SEC_HDR_FLAG", "PKT_APID", "SEQ_FLGS", "SRC_SEQ_CTR", "PKT_LEN", "REUSABLE_SPARE_8"}
)


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def remap_dims(var_def: dict, old_dim: str, new_dim: str) -> dict:
    v = copy.deepcopy(var_def)
    v["dimensions"] = [new_dim if d == old_dim else d for d in v["dimensions"]]
    return v


nom_hk = load_yaml(DEF_DIR / "icie_nom_hk_l1a.yml")
pev_sw = load_yaml(DEF_DIR / "pev_sw_stat_l1a.yml")
rad_sample = load_yaml(DEF_DIR / "icie_rad_sample_l1a.yml")

for face_num in (1, 2):
    obsid_tag = "384-387" if face_num == 1 else "388-391"
    product_id = f"SOLAR-CAL-FACE{face_num}-DECODED"

    merged: dict = {
        "attributes": {
            "ProductID": product_id,
            "algorithm_version": None,
            "date_created": None,
            "input_files": None,
            # Solar-cal event metadata — populated per file at write time;
            # declared as dynamic (null) so they are not stripped by product_definition.py
            "solar_cal_face": None,
            "event_pass_index": None,
            "source_obsids": None,
        },
        "coordinates": {},
        "variables": {},
    }

    # ---- NOM-HK time coordinate (PACKET -> NOM_HK_PACKET) ----
    hk_time = remap_dims(nom_hk["coordinates"]["PACKET_ICIE_TIME"], "PACKET", "NOM_HK_PACKET")
    hk_time["attributes"]["long_name"] = "Packet timestamp from ICIE main processor (NOM-HK)"
    merged["coordinates"]["NOM_HK_PACKET_ICIE_TIME"] = hk_time

    # ---- PEV-SW-STAT time coordinate ----
    pev_time = remap_dims(pev_sw["coordinates"]["PACKET_ICIE_TIME"], "PACKET", "PEV_SW_PACKET")
    pev_time["attributes"]["long_name"] = "Packet timestamp from ICIE main processor (PEV-SW-STAT)"
    merged["coordinates"]["PEV_SW_PACKET_ICIE_TIME"] = pev_time

    # ---- RAD-SAMPLE time coordinates ----
    rad_pkt_time = remap_dims(rad_sample["coordinates"]["PACKET_ICIE_TIME"], "PACKET", "RAD_PACKET")
    rad_pkt_time["attributes"]["long_name"] = "Packet timestamp from ICIE main processor (RAD-SAMPLE)"
    merged["coordinates"]["RAD_PACKET_ICIE_TIME"] = rad_pkt_time
    merged["coordinates"]["RAD_SAMPLE_FPE_TIME"] = copy.deepcopy(rad_sample["coordinates"]["RAD_SAMPLE_FPE_TIME"])

    # ---- NOM-HK variables (canonical names; no conflict prefix needed) ----
    for name, vdef in nom_hk["variables"].items():
        merged["variables"][name] = remap_dims(vdef, "PACKET", "NOM_HK_PACKET")

    # ---- Derived temperature variables (DN -> K) ----
    merged["variables"]["ICIE__FPE_TSCOPE_TEMP_K"] = {
        "dtype": "float64",
        "dimensions": ["NOM_HK_PACKET"],
        "attributes": {
            "long_name": "FPE telescope temperature converted from DN to Kelvin",
            "units": "K",
            "comment": "Derived from ICIE__FPE_TSCOPE_TEMP using bench calibration coefficients",
        },
    }
    merged["variables"]["ICIE__CFPE_SWCR_TEMP_K"] = {
        "dtype": "float64",
        "dimensions": ["NOM_HK_PACKET"],
        "attributes": {
            "long_name": "SWCR temperature converted from DN to Kelvin",
            "units": "K",
            "comment": "Derived from ICIE__CFPE_SWCR_TEMP using bench calibration coefficients",
        },
    }

    # ---- PEV-SW-STAT variables (conflicting names get PEV_SW__ prefix) ----
    for name, vdef in pev_sw["variables"].items():
        new_name = f"PEV_SW__{name}" if name in CONFLICTS else name
        merged["variables"][new_name] = remap_dims(vdef, "PACKET", "PEV_SW_PACKET")

    # ---- RAD-SAMPLE variables (conflicting names get RAD__ prefix) ----
    for name, vdef in rad_sample["variables"].items():
        new_name = f"RAD__{name}" if name in CONFLICTS else name
        merged["variables"][new_name] = remap_dims(vdef, "PACKET", "RAD_PACKET")

    out_path = DEF_DIR / f"solar_cal_face{face_num}_l1a.yml"
    header = (
        f"# Solar Calibration Face {face_num} L1A event product definition\n"
        f"# ProductID: {product_id}  (OBSIDs {obsid_tag})\n"
        "#\n"
        "# Each source APID gets a uniquely prefixed dimension and time coordinate:\n"
        "#   NOM-HK:      NOM_HK_PACKET     / NOM_HK_PACKET_ICIE_TIME\n"
        "#   PEV-SW-STAT: PEV_SW_PACKET     / PEV_SW_PACKET_ICIE_TIME\n"
        "#   RAD-SAMPLE:  RAD_PACKET         / RAD_PACKET_ICIE_TIME  (packet-level)\n"
        "#                RAD_SAMPLE_FPE_TIME                (200 Hz sample-level)\n"
        "#\n"
        "# CCSDS header variables that appear in all three source APIDs (VERSION, TYPE,\n"
        "# SEC_HDR_FLAG, PKT_APID, SEQ_FLGS, SRC_SEQ_CTR, PKT_LEN, REUSABLE_SPARE_8)\n"
        "# are kept verbatim from NOM-HK; copies from PEV-SW-STAT are prefixed\n"
        "# 'PEV_SW__' and copies from RAD-SAMPLE are prefixed 'RAD__'.\n"
        "#\n"
        "# Two derived variables are appended immediately after the raw NOM-HK temps:\n"
        "#   ICIE__FPE_TSCOPE_TEMP_K  -- FPE telescope temperature (K)\n"
        "#   ICIE__CFPE_SWCR_TEMP_K   -- SWCR temperature (K)\n"
        "\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        yaml.dump(merged, f, default_flow_style=None, sort_keys=False, allow_unicode=True)

    n_vars = len(merged["variables"])
    n_coords = len(merged["coordinates"])
    print(f"Wrote {out_path}  ({out_path.stat().st_size} bytes, {n_vars} variables, {n_coords} coordinates)")

print("Done.")

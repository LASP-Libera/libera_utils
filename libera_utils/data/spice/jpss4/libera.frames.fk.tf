KPL/FK

Frame definitions for Libera on the JPSS-4 spacecraft
--------------------------------------------------------

    TODO[LIBSDC-601]: Uses NOAA-20 (JPSS-1) base ID, needs updating to JPSS-4 at launch.

    Frame definitions required for Libera geolocation processing.

    Frame Name              Relative to Frame   Frame Type  Frame ID
    ==========              =================   ==========  ========
    JPSS4_SC_COORD          J2000 (ECI)         CK          -143013000
    LIBERA_BASE_COORD       JPSS4_SC_COORD      FIXED       -143013001
    LIBERA_AZ_COORD         LIBERA_BASE_COORD   CK          -143013002
    LIBERA_WFOV_CAM_COORD   LIBERA_AZ_COORD     FIXED       -143013010
    LIBERA_EL_COORD         LIBERA_AZ_COORD     CK          -143013003
    LIBERA_SW_RAD_COORD     LIBERA_EL_COORD     FIXED       -143013011
    LIBERA_SSW_RAD_COORD    LIBERA_EL_COORD     FIXED       -143013012
    LIBERA_LW_RAD_COORD     LIBERA_EL_COORD     FIXED       -143013013
    LIBERA_TOT_RAD_COORD    LIBERA_EL_COORD     FIXED       -143013014

                  "J2000" <- inertial
                  -----------------------------------------+
                     |                                     |
                     | <- ck                               | <- pck
                     |                                     V
                     V                                "IAU_EARTH"
             "JPSS4_SC_COORD"                      EARTH BODY-FIXED
             -----------------                      ----------------
                     |
                     | <- fixed
                     V
             "LIBERA_BASE_COORD"
            --------------------
                     |
                     | <- ck
                     V
             "LIBERA_AZ_COORD"
       +------------------------------+
       |                              |
       | <- fixed                     | <- ck
       V                              |
 "LIBERA_WFOV_CAM_COORD"              |
 -----------------------              V
                              "LIBERA_EL_COORD"
       +------------------------------------------------------------+
       |                   |                   |                    |
       | <- fixed          | <- fixed          | <- fixed           | <- fixed
       V                   V                   |                    |
"LIBERA_SW_RAD_COORD" "LIBERA_SSW_RAD_COORD"   V                    V
                                    "LIBERA_LW_RAD_COORD" "LIBERA_TOT_RAD_COORD"


    Notes
    -----
    - SPICE matrices are written in column-major order, and must be
    oriented as a rotation *from* Frame *to* Relative.

    References
    ----------

    This file was created by LASP_SDS_TEAM.

Frame offsets
--------------------------------------------------------
    Frame offsets are actually defined in a "static" kernel. The values are
    included here as a reference. Units = meters.

    From Frame          To Frame            Offset [X, Y, Z]
    ==========          ========            ================
    <all>               <all>>              [ 0.0,       0.0,      0.0]

Measured frame misalignments (LIBSDC-806)
--------------------------------------------------------
    OAV3 ground-test unit vectors in the LIBERA_BASE (STAND) frame (J. Fernandez,
    OAV3 Confluence Table 5). Libera-internal and spacecraft-independent. kernel_maker
    reads the axis-of-rotation vectors (*_AOR_*) to build the AZ/EL mechanism CKs; the
    radiometer boresight quaternions in the frame definitions below are derived from
    LIBERA_EL0_Z_IN_STAND (minimal rotation of +Z onto that line-of-sight).

        \begindata

        LIBERA_EL0_Z_IN_STAND  = ( -0.00459575167882055,  -0.00030501098366856,   0.999989392961149   )
        LIBERA_EL_AOR_IN_STAND = (  0.999987800152187,      0.00276809474781597,   0.00409111210523636 )
        LIBERA_AZ_AOR_IN_STAND = ( -0.000926750267457724,  -0.00066424424979518,   0.999999349956548   )

        \begintext

Frame definitions
--------------------------------------------------------

    JPSS-4 SC (-143013) - Spacecraft (CK)
    --------------------------------------

        \begindata

        FRAME_JPSS4_SC_COORD        = -143013000
        FRAME_-143013000_NAME       = 'JPSS4_SC_COORD'
        FRAME_-143013000_CLASS      = 3
        FRAME_-143013000_CLASS_ID   = -143013000
        FRAME_-143013000_CENTER     = -143013
        CK_-143013000_SCLK          = -143013
        CK_-143013000_SPK           = -143013

        OBJECT_-143013_FRAME        = 'JPSS4_SC_COORD'

        \begintext

    Libera BASE (-143013001) - Structure (TK)
    -----------------------------------------

        \begindata

        FRAME_LIBERA_BASE_COORD     = -143013001
        FRAME_-143013001_NAME       = 'LIBERA_BASE_COORD'
        FRAME_-143013001_CLASS      = 4
        FRAME_-143013001_CLASS_ID   = -143013001
        FRAME_-143013001_CENTER     = -143013
        TKFRAME_-143013001_RELATIVE = 'JPSS4_SC_COORD'
        TKFRAME_-143013001_SPEC     = 'ANGLES'
        TKFRAME_-143013001_UNITS    = 'DEGREES'
        TKFRAME_-143013001_AXES     = ( 1,     2,      3   )
        TKFRAME_-143013001_ANGLES   = ( 0.0,   0.0,    0.0 )

        OBJECT_-143013001_FRAME     = 'LIBERA_BASE_COORD'

        \begintext

    Libera Azimuth (-143013002) - Dynamic (CK)
    ------------------------------------------

        \begindata

        FRAME_LIBERA_AZ_COORD       = -143013002
        FRAME_-143013002_NAME       = 'LIBERA_AZ_COORD'
        FRAME_-143013002_CLASS      = 3
        FRAME_-143013002_CLASS_ID   = -143013002
        FRAME_-143013002_CENTER     = -143013001
        CK_-143013002_SCLK          = -143013
        CK_-143013002_SPK           = -143013001

        OBJECT_-143013002_FRAME     = 'LIBERA_AZ_COORD'

        \begintext

    Libera WFOV Camera (-143013010) - Instrument (TK)
    -------------------------------------------------

        \begindata

        FRAME_LIBERA_WFOV_CAM_COORD = -143013010
        FRAME_-143013010_NAME       = 'LIBERA_WFOV_CAM_COORD'
        FRAME_-143013010_CLASS      = 4
        FRAME_-143013010_CLASS_ID   = -143013010
        FRAME_-143013010_CENTER     = -143013002
        TKFRAME_-143013010_RELATIVE = 'LIBERA_AZ_COORD'
        TKFRAME_-143013010_SPEC     = 'ANGLES'
        TKFRAME_-143013010_UNITS    = 'DEGREES'
        TKFRAME_-143013010_AXES     = ( 1,     2,      3   )
        TKFRAME_-143013010_ANGLES   = ( 0.0,   0.0,    0.0 )

        OBJECT_-143013010_FRAME     = 'LIBERA_WFOV_CAM_COORD'

        \begintext

    Libera Elevation (-143013003) - Dynamic (CK)
    --------------------------------------------

        \begindata

        FRAME_LIBERA_EL_COORD       = -143013003
        FRAME_-143013003_NAME       = 'LIBERA_EL_COORD'
        FRAME_-143013003_CLASS      = 3
        FRAME_-143013003_CLASS_ID   = -143013003
        FRAME_-143013003_CENTER     = -143013002
        CK_-143013003_SCLK          = -143013
        CK_-143013003_SPK           = -143013002

        OBJECT_-143013003_FRAME     = 'LIBERA_EL_COORD'

        \begintext

    Radiometer boresight (LIBSDC-806)
    ---------------------------------
        The four radiometer TK frames below carry a measured boresight rotation (not a
        translation): each orients its +Z boresight to the Total-radiometer line-of-sight
        at corrected elevation = 0 deg (EL0_Z in LIBERA_BASE, ~0.264 deg off +Z), from OAV3
        ground testing. All four share this value until per-radiometer boresights exist.
        Derived from LIBERA_EL0_Z_IN_STAND (see Measured frame misalignments above).

    Libera SW Radiometer (-143013011) - Instrument (TK)
    ---------------------------------------------------

        \begindata

        FRAME_LIBERA_SW_RAD_COORD   = -143013011
        FRAME_-143013011_NAME       = 'LIBERA_SW_RAD_COORD'
        FRAME_-143013011_CLASS      = 4
        FRAME_-143013011_CLASS_ID   = -143013011
        FRAME_-143013011_CENTER     = -143013003
        TKFRAME_-143013011_RELATIVE = 'LIBERA_EL_COORD'
        TKFRAME_-143013011_SPEC     = 'QUATERNION'
        TKFRAME_-143013011_Q        = ( 0.999997348236771, 0.000152505896243, -0.002297881932832, 0.000000000000000 )

        OBJECT_-143013011_FRAME     = 'LIBERA_SW_RAD_COORD'

        \begintext

    Libera SSW Radiometer (-143013012) - Instrument (TK)
    ----------------------------------------------------

        \begindata

        FRAME_LIBERA_SSW_RAD_COORD  = -143013012
        FRAME_-143013012_NAME       = 'LIBERA_SSW_RAD_COORD'
        FRAME_-143013012_CLASS      = 4
        FRAME_-143013012_CLASS_ID   = -143013012
        FRAME_-143013012_CENTER     = -143013003
        TKFRAME_-143013012_RELATIVE = 'LIBERA_EL_COORD'
        TKFRAME_-143013012_SPEC     = 'QUATERNION'
        TKFRAME_-143013012_Q        = ( 0.999997348236771, 0.000152505896243, -0.002297881932832, 0.000000000000000 )

        OBJECT_-143013012_FRAME     = 'LIBERA_SSW_RAD_COORD'

        \begintext

    Libera LW Radiometer (-143013013) - Instrument (TK)
    ---------------------------------------------------

        \begindata

        FRAME_LIBERA_LW_RAD_COORD   = -143013013
        FRAME_-143013013_NAME       = 'LIBERA_LW_RAD_COORD'
        FRAME_-143013013_CLASS      = 4
        FRAME_-143013013_CLASS_ID   = -143013013
        FRAME_-143013013_CENTER     = -143013003
        TKFRAME_-143013013_RELATIVE = 'LIBERA_EL_COORD'
        TKFRAME_-143013013_SPEC     = 'QUATERNION'
        TKFRAME_-143013013_Q        = ( 0.999997348236771, 0.000152505896243, -0.002297881932832, 0.000000000000000 )

        OBJECT_-143013013_FRAME     = 'LIBERA_LW_RAD_COORD'

        \begintext

    Libera TOT Radiometer (-143013014) - Instrument (TK)
    ----------------------------------------------------

        \begindata

        FRAME_LIBERA_TOT_RAD_COORD  = -143013014
        FRAME_-143013014_NAME       = 'LIBERA_TOT_RAD_COORD'
        FRAME_-143013014_CLASS      = 4
        FRAME_-143013014_CLASS_ID   = -143013014
        FRAME_-143013014_CENTER     = -143013003
        TKFRAME_-143013014_RELATIVE = 'LIBERA_EL_COORD'
        TKFRAME_-143013014_SPEC     = 'QUATERNION'
        TKFRAME_-143013014_Q        = ( 0.999997348236771, 0.000152505896243, -0.002297881932832, 0.000000000000000 )

        OBJECT_-143013014_FRAME     = 'LIBERA_TOT_RAD_COORD'

        \begintext

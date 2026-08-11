KPL/FK

Frozen nominal (misalignment-free) Libera frame kernel for tier-0 geolocation tests
--------------------------------------------------------

    Copy of the production NOAA-20 Libera frame kernel with the OAV3 measured frame
    misalignments (LIBSDC-806) omitted: identity radiometer boresight quaternions and no
    axis-of-rotation keywords. test_tier0_geolocation.py loads this so its geolocation-vs-CERES
    math check runs on nominal geometry; the real misalignments (in the production frame kernel)
    are validated separately in test_los_alignment.py. Frozen test data -- do not regenerate.

    Frame Name              Relative to Frame   Frame Type  Frame ID
    ==========              =================   ==========  ========
    NOAA20_SC_COORD         J2000 (ECI)         CK          -143013000
    LIBERA_BASE_COORD       NOAA20_SC_COORD     FIXED       -143013001
    LIBERA_AZ_COORD         LIBERA_BASE_COORD   CK          -143013002
    LIBERA_WFOV_CAM_COORD   LIBERA_AZ_COORD     FIXED       -143013010
    LIBERA_EL_COORD         LIBERA_AZ_COORD     CK          -143013003
    LIBERA_RAD_COORD        LIBERA_EL_COORD     FIXED       -143013011

                  "J2000" <- inertial
                  -----------------------------------------+
                     |                                     |
                     | <- ck                               | <- pck
                     |                                     V
                     V                                "IAU_EARTH"
             "NOAA20_SC_COORD"                      EARTH BODY-FIXED
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
                              -----------------
                                      |
                                      | <- fixed
                                      V
                              "LIBERA_RAD_COORD"
                              ------------------


    Notes
    -----
    - SPICE matrices are written in column-major order, and must be
    oriented as a rotation *from* Frame *to* Relative.

    References
    ----------

    This file was created by LASP_SDS_TEAM
    on 2024-11-01/00:00:00.

Frame offsets
--------------------------------------------------------
    Frame offsets are actually defined in a "static" kernel. The values are
    included here as a reference. Units = meters.

    From Frame          To Frame            Offset [X, Y, Z]
    ==========          ========            ================
    <all>               <all>>              [ 0.0,       0.0,      0.0]

Frame definitions
--------------------------------------------------------

    NOAA-20 SC (-143013) - Spacecraft (CK)
    --------------------------------------

        \begindata

        FRAME_NOAA20_SC_COORD       = -143013000
        FRAME_-143013000_NAME       = 'NOAA20_SC_COORD'
        FRAME_-143013000_CLASS      = 3
        FRAME_-143013000_CLASS_ID   = -143013000
        FRAME_-143013000_CENTER     = -143013
        CK_-143013000_SCLK          = -143013
        CK_-143013000_SPK           = -143013

        OBJECT_-143013_FRAME        = 'NOAA20_SC_COORD'

        \begintext

    Libera BASE (-143013001) - Structure (TK)
    -----------------------------------------

        \begindata

        FRAME_LIBERA_BASE_COORD     = -143013001
        FRAME_-143013001_NAME       = 'LIBERA_BASE_COORD'
        FRAME_-143013001_CLASS      = 4
        FRAME_-143013001_CLASS_ID   = -143013001
        FRAME_-143013001_CENTER     = -143013
        TKFRAME_-143013001_RELATIVE = 'NOAA20_SC_COORD'
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

    Libera Radiometer (-143013011) - Instrument (TK)
    ------------------------------------------------
        The four radiometer channels (SW, SSW, LW, TOT) are assumed co-aligned and
        share this single frame.

        \begindata

        FRAME_LIBERA_RAD_COORD      = -143013011
        FRAME_-143013011_NAME       = 'LIBERA_RAD_COORD'
        FRAME_-143013011_CLASS      = 4
        FRAME_-143013011_CLASS_ID   = -143013011
        FRAME_-143013011_CENTER     = -143013003
        TKFRAME_-143013011_RELATIVE = 'LIBERA_EL_COORD'
        TKFRAME_-143013011_SPEC     = 'QUATERNION'
        TKFRAME_-143013011_Q        = ( 1.0,   0.0,   0.0,   0.0 )

        OBJECT_-143013011_FRAME     = 'LIBERA_RAD_COORD'

        \begintext

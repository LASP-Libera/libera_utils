KPL/FK

Frame Kernel for Libera Instrument on JPSS-3
===============================================================================

This frame kernel file (.tf) contains frame definitions for the elements of the
Libera instrument
    > Azimuth Mechanism
    > Elevation Mechanism
    > Wide Field of View (WFOV) Camera
    > Short Wave (SW) Radiometer
    > Split Short Wave (SSW) Radiometer
    > Long Wave (LW) Radiometer
    > Total (TOT) Radiometer
as well as the frame definition for the JPSS spacecraft.

This frame kernel also includes a kernel pool variable that defines the
EARTH_FIXED frame (ECEF) as identical to the high-precision ITRF93 reference
frame rather than the default IAU_EARTH frame. See docs here:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html \
  #Appendix.%20High%20Precision%20Earth%20Fixed%20Frames
The ITRF93 frame requires loading
a high precision PCK from:
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/
see aareadme.md for details on high precision Earth kernels.

Version and Date (most recent at top)
===============================================================================
--------------------
Version libera_fk_rev01 - 21 Sep 2021 - Gavin Medley

  Initial version for prelaunch analysis. In this version we use the NORAD ID
  for JPSS-1 to derive a NAIF ID: -100000 - 43013 = -143013. This will have to
  be updated for JPSS-3 once a NORAD ID is assigned.
--------------------

Frames Hierarchy
===============================================================================

                  "J2000" <- inertial
                  -------
                     |
                     | <- ck
                     V
             "JPSS_SPACECRAFT"
             -----------------
                     |
                     | <- fixed
                     V
            "LIBERA_MOUNT_PLATE"
            --------------------
                     |
                     | <- ck
                     V
          "LIBERA_AZIMUTH_MECHANISM"
       +------------------------------+
       |                              |
       | <- fixed                     | <- ck
       V                              |
 "WFOV_CAMERA"                        |
 -------------                        V
                         "LIBERA_ELEVATION_MECHANISM"
       +------------------------------------------------------------+
       |                   |                   |                    |
       | <- fixed          | <- fixed          | <- fixed           | <- fixed
       V                   V                   V                    V
"SW_RADIOMETER"     "SSW_RADIOMETER"     "LW_RADIOMETER"     "TOT_RADIOMETER"
---------------     ----------------     ---------------     ----------------

NOTE: In this diagram, we are making no assumptions about the precision of
radiometer alignment. In reality, we may be fortunate enough to have all
radiometers aligned closely enough that we don't need separate quaternions
for each one.

NOTE: If we need to account for the offset between origins between SC origin,
Az/El origins, and radiometer origins, we need an SPK for each of those offsets
(per Brandon Stone's advice). In all likelihood, we don't actually need to
entertain those offsets and are safe to assume that translationally, all the
origins overlap (though the frames may be rotated relative to eachother).

Frame Specification
===============================================================================
  Frame Name    :    'WALDO'
  Frame ID code :    1234567   (A number guaranteed to be suitable
                                for private use)
  Frame Class   :          3   (C-kernel)
  Frame Center  :     -10001   (Waldo Spacecraft ID code)
  Frame Class_id:  -10000001   (ID code in C-kernel for Waldo)

\begindata

  TKFRAME_EARTH_FIXED_RELATIVE = 'ITRF93'
  TKFRAME_EARTH_FIXED_SPEC     = 'MATRIX'
  TKFRAME_EARTH_FIXED_MATRIX   = ( 1   0   0
                                   0   1   0
                                   0   0   1 )

  FRAME_JPSS_SPACECRAFT            = -143013000
  FRAME_-143013000_NAME            = 'JPSS_SPACECRAFT'
  FRAME_-143013000_CLASS           =  3
  FRAME_-143013000_CLASS_ID        = -143013000
  FRAME_-143013000_CENTER          = -143013
  CK_-143013000_SCLK               = -143013
  CK_-143013000_SPK                = -143013

  FRAME_LIBERA_MOUNT_PLATE         = -143013001
  FRAME_-143013001_NAME            = 'LIBERA_MOUNT_PLATE'
  FRAME_-143013001_CLASS           =  4
  FRAME_-143013001_CLASS_ID        = -143013001
  FRAME_-143013001_CENTER          = -143013
  TKFRAME_-143013001_SPEC          = 'QUATERNION'
  TKFRAME_-143013001_RELATIVE      = 'JPSS_SPACECRAFT'
  TKFRAME_-143013001_Q             = (1.0, 0.0, 0.0, 0.0)
  
  FRAME_LIBERA_AZIMUTH_MECHANISM   = -143013002
  FRAME_-143013002_NAME            = 'LIBERA_AZIMUTH_MECHANISM'
  FRAME_-143013002_CLASS           =  3
  FRAME_-143013002_CLASS_ID        = -143013002
  FRAME_-143013002_CENTER          = -143013
  CK_-143013002_SCLK               = -143013
  CK_-143013002_SPK                = -143013

  FRAME_LIBERA_ELEVATION_MECHANISM = -143013003
  FRAME_-143013003_NAME            = 'LIBERA_ELEVATION_MECHANISM'
  FRAME_-143013003_CLASS           =  3
  FRAME_-143013003_CLASS_ID        = -143013003
  FRAME_-143013003_CENTER          = -143013
  CK_-143013003_SCLK               = -143013
  CK_-143013003_SPK                = -143013

  FRAME_LIBERA_WFOV_CAMERA         = -143013010
  FRAME_-143013010_NAME            = 'LIBERA_WFOV_CAMERA'
  FRAME_-143013010_CLASS           =  4
  FRAME_-143013010_CLASS_ID        = -143013010
  FRAME_-143013010_CENTER          = -143013
  TKFRAME_-143013010_SPEC          = 'ANGLES'
  TKFRAME_-143013010_RELATIVE      = 'LIBERA_AZIMUTH_MECHANISM'
  TKFRAME_-143013010_ANGLES        = (  0.0,       0.0,      0.0 )
  TKFRAME_-143013010_AXES          = (  1,         2,        3      )
  TKFRAME_-143013010_UNITS         = 'DEGREES'

  FRAME_LIBERA_SW_RADIOMETER       = -143013011
  FRAME_-143013011_NAME            = 'LIBERA_SW_RADIOMETER'
  FRAME_-143013011_CLASS           =  4
  FRAME_-143013011_CLASS_ID        = -143013011
  FRAME_-143013011_CENTER          = -143013
  TKFRAME_-143013011_SPEC          = 'QUATERNION'
  TKFRAME_-143013011_RELATIVE      = 'LIBERA_ELEVATION_MECHANISM'
  TKFRAME_-143013011_Q             = (1.0, 0.0, 0.0, 0.0)

  FRAME_LIBERA_SSW_RADIOMETER      = -143013012
  FRAME_-143013012_NAME            = 'LIBERA_SSW_RADIOMETER'
  FRAME_-143013012_CLASS           =  4
  FRAME_-143013012_CLASS_ID        = -143013012
  FRAME_-143013012_CENTER          = -143013
  TKFRAME_-143013012_SPEC          = 'QUATERNION'
  TKFRAME_-143013012_RELATIVE      = 'LIBERA_ELEVATION_MECHANISM'
  TKFRAME_-143013012_Q             = (1.0, 0.0, 0.0, 0.0)

  FRAME_LIBERA_LW_RADIOMETER       = -143013013
  FRAME_-143013013_NAME            = 'LIBERA_LW_RADIOMETER'
  FRAME_-143013013_CLASS           =  4
  FRAME_-143013013_CLASS_ID        = -143013013
  FRAME_-143013013_CENTER          = -143013
  TKFRAME_-143013013_SPEC          = 'QUATERNION'
  TKFRAME_-143013013_RELATIVE      = 'LIBERA_ELEVATION_MECHANISM'
  TKFRAME_-143013013_Q             = (1.0, 0.0, 0.0, 0.0)

  FRAME_LIBERA_TOT_RADIOMETER      = -143013014
  FRAME_-143013014_NAME            = 'LIBERA_TOT_RADIOMETER'
  FRAME_-143013014_CLASS           =  4
  FRAME_-143013014_CLASS_ID        = -143013014
  FRAME_-143013014_CENTER          = -143013
  TKFRAME_-143013014_SPEC          = 'QUATERNION'
  TKFRAME_-143013014_RELATIVE      = 'LIBERA_ELEVATION_MECHANISM'
  TKFRAME_-143013014_Q             = (1.0, 0.0, 0.0, 0.0)

\begintext

JPSS NAIF ID Codes -- Definitions
=====================================================================

   This section contains name to NAIF ID mappings for the JPSS mission.
   Once the contents of this file are loaded into the KERNEL POOL, these
   mappings become available within SPICE, making it possible to use
   names instead of ID code in high level SPICE routine calls.

\begindata

  NAIF_BODY_NAME   += ( 'JPSS'  )
  NAIF_BODY_CODE   += ( -143013 )

\begintext

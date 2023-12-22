# Usage of SPICE


Fun fact: SPICE stands for "Spacecraft Planet Instrument C-matrix Events,"
which were the original primary concerns of those developing the library.


## Static Kernels Generated at Libera SDC
These kernels are part of the package data, are manually edited, and change rarely.


### Frame Kernel (FK)
e.g. `libera_fk_v01.tf`

Contains reference frame definitions for JPSS and Libera

### Spacecraft Clock Kernel (SCLK)
e.g. `jpss_sclk_v01.tsc`

Contains specification of the spacecraft clock on JPSS. 

### Instrument Kernel (IK)
e.g. `libera_cam_ik_v01.ti`, `libera_rad_ik_v01.ti`

Contains geometry specification data of the Libera instruments.


## Dynamic Kernels Generated at Libera SDC
These kernels are binary generated kernels. They are ancillary data products and are created 
as part of pipeline processing.


### JPSS Ephemeris Kernel (SPK)
e.g. `libera_jpss_20210408t235850_20210409t015849_vM3m14p159_r25365125959.bsp`

Contains ephemeris data -- coordinates in ITRF93 frame -- for the JPSS spacecraft body.

### JPSS Attitude Kernel (CK)
e.g. `libera_jpss_20210408t235850_20210409t015849_vM3m14p159_r25365125959.bc`

Contains attitude data -- quaternion rotations from J2000 -- for the JPSS spacecraft body.

### Azimuth Rotation Mechanism Attitude Kernel (CK)
e.g. `libera_azrot_20210408t235850_20210409t015849_vM3m14p159_r25365125959.bc`

Contains attitude data for the Libera Azimuth Rotation mechanism.

Note: there is currently no mechanism for creating this kernel because no telemetry data exists.

### Elevation Scan Mechanism Attitude Kernel (CK)
e.g. `libera_elscan_20210408t235850_20210409t015849_vM3m14p159_r25365125959.bc`

Contains attitude data for the Libera Elevation Scan mechanism.

Note: there is currently no mechanism for creating this kernel because no telemetry data exists.


## Kernels Retrieved from NAIF
These kernels are generated at NAIF. We download them as needed using the `KernelFileCache` class,
which is configured with a NAIF index page URL and a regex string that finds the proper download URL
on the index page. The downloaded file is put in a cache so the download only occurs after the cached
data is of a specified age. If we want to effectively cache some kernels indefinitely, we can put 
them in an S3 bucket and retrieve them from there instead of from the NAIF website.


### Leapseconds Kernel (LSK)
e.g. `naif0012.tls`

Contains leapsecond data used by time conversion routines.

### Development Ephemeris Kernel (SPK)
e.g. `de440s.bsp`

Contains ephemeris data for planetary bodies. The version with "s" appended to the filename covers only more recent time
(starts in the 1500s) to reduce filesize.

### High Precision Earth Binary Planetary Constants Kernel (PCK)
e.g. `earth_000101_211220_210926.bpc`

Contains high precision orientation data for Earth in the ECEF ITRF93 reference frame.

ITRF93 is a more precise version of the standard IAU_EARTH reference frame provided in the text PCK below.

This kernel is regenerated several times per week so we should retrieve this from NAIF for every processing run.

### ITFR93 Reference Frame Kernel for Earth
e.g. `earth_assoc_itrf93.tf`

Used to designate ITRF93 as the default body-fixed frame associated with the Earth.

### Standard Text Planetary Constants Kernel (PCK)
e.g. `pck00010.tpc`

Contains orientation data and other planetary constants for planetary bodies.

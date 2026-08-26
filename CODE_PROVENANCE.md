# Code provenance and third-party software

The analytical workflow was developed for this dissertation and uses the public APIs of established open-source packages rather than vendoring their source code. Principal dependencies include NumPy, pandas, SciPy, GeoPandas, Shapely, PyProj, Pyogrio, Matplotlib, PySAL/libpysal and esda. QGIS and GDAL are used for selected GIS preparation and styling tasks.

Method choices derived from external documentation are identified in code comments and manifests, including the DfT Journey Time Statistics speed profile and PySAL spatial-weights/join-count implementation. Data-source rights remain governed by the providers listed in [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

If code adapted from another project is added later, its source URL, licence, original purpose and dissertation-specific modifications should be recorded in this file before release.

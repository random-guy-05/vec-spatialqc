# Community Contribution submission text

## Title
VEC SpatialQC — geometry sanity checks for T2/T3 predictions

## Description
VEC SpatialQC catches spatial predictions that can satisfy the basic H5AD coordinate contract but are geometrically degenerate. It audits the scorer-relevant first three spatial_3D columns for finite values, centered geometric rank, exact duplicate coordinates, rotation-invariant singular-value anisotropy, nearest-neighbor distances and robust radial outliers. It returns conservative good / warning / geometry-failure classifications with JSON and CI-friendly exit codes. The tool is invariant to harmless global translation, makes no hidden-target comparisons, and includes end-to-end synthetic H5AD tests for healthy 3D, line-collapsed and missing-coordinate cases.

# VEC SpatialQC

**Pre-submission geometry diagnostics for Task 2 / Task 3 prediction clouds.**

A T2/T3 H5AD can satisfy the basic coordinate contract while still containing a degenerate point cloud: all points on a line, huge exact duplication, extreme anisotropy, or severe radial outliers. SpatialQC detects those obvious export/model failures without comparing to hidden data.

## Checks

- finite `obsm["spatial_3D"][:, :3]`;
- centered geometric rank;
- exact duplicate-coordinate fraction;
- axis spans plus rotation-invariant singular-value anisotropy;
- nearest-neighbor distance distribution;
- robust radial outlier fraction.

Global translation and rotation are not treated as errors. The anisotropy warning is based on singular values rather than the coordinate axes, and the rank check is relative to the cloud's own scale.

## Usage

```bash
pip install -e .
vec-spatialqc prediction.h5ad
vec-spatialqc prediction.h5ad --json spatial_qc.json
```

Exit codes: `0` good, `3` warning, `4` geometry failure, `2` invalid/missing coordinates.

The thresholds are deliberately conservative and intended to catch clear pipeline failures rather than judge whether an embryo is biologically correct.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check src tests
```

The suite includes end-to-end synthetic H5AD tests for a 3D cloud, a line-collapsed cloud and a missing-coordinate file, plus regression tests for coordinate-scale and rotation invariance.

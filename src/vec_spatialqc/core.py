from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SpatialReport:
    classification: str
    cells: int
    geometric_rank: int
    duplicate_fraction: float
    axis_spans: list[float]
    span_ratio: float
    singular_value_ratio: float
    median_nearest_neighbor_distance: float
    radial_outlier_fraction: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_coordinates(coords) -> SpatialReport:
    cloud = np.asarray(coords, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError("spatial_3D must have shape cells x >=3")
    cloud = cloud[:, :3]
    if len(cloud) < 3:
        raise ValueError("spatial cloud must contain at least 3 cells")
    if not np.isfinite(cloud).all():
        raise ValueError("spatial_3D contains NaN or infinite values")

    centered = cloud - np.mean(cloud, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.size and singular[0] > 0:
        # Relative tolerance keeps the rank decision unchanged if a coordinate
        # system is expressed in different units.
        geometric_rank = int(np.sum(singular > singular[0] * 1e-8))
    else:
        geometric_rank = 0

    if geometric_rank == 3 and singular[-1] > 0:
        singular_value_ratio = float(singular[0] / singular[-1])
    else:
        singular_value_ratio = float("inf")

    unique = np.unique(cloud, axis=0)
    duplicate_fraction = 1.0 - len(unique) / len(cloud)

    spans = np.ptp(cloud, axis=0)
    positive_spans = spans[spans > 1e-12]
    span_ratio = (
        float(positive_spans.max() / positive_spans.min())
        if len(positive_spans)
        else float("inf")
    )

    tree = cKDTree(cloud)
    distances, _ = tree.query(cloud, k=2)
    nearest = distances[:, 1]
    median_nn = float(np.median(nearest))

    radii = np.linalg.norm(centered, axis=1)
    median_radius = float(np.median(radii))
    mad = float(np.median(np.abs(radii - median_radius)))
    if mad > 0:
        robust_z = np.abs(radii - median_radius) / (1.4826 * mad)
        outlier_fraction = float(np.mean(robust_z > 8))
    else:
        outlier_fraction = 0.0

    reasons: list[str] = []
    severe = False
    if geometric_rank < 2:
        reasons.append("coordinate cloud has geometric rank < 2")
        severe = True
    if duplicate_fraction >= 0.50:
        reasons.append("at least half of coordinates are exact duplicates")
        severe = True

    warning = False
    if geometric_rank == 2:
        reasons.append("coordinate cloud is effectively planar")
        warning = True
    if duplicate_fraction >= 0.10:
        reasons.append("at least 10% of coordinates are exact duplicates")
        warning = True
    if geometric_rank == 3 and singular_value_ratio > 1000:
        reasons.append("coordinate cloud is extremely anisotropic")
        warning = True
    if outlier_fraction > 0.02:
        reasons.append("more than 2% of cells are extreme radial outliers")
        warning = True

    classification = (
        "GEOMETRY_FAILURE"
        if severe
        else "SPATIAL_WARNING"
        if warning
        else "SPATIAL_QC_OK"
    )
    return SpatialReport(
        classification=classification,
        cells=len(cloud),
        geometric_rank=geometric_rank,
        duplicate_fraction=duplicate_fraction,
        axis_spans=[float(value) for value in spans],
        span_ratio=span_ratio,
        singular_value_ratio=singular_value_ratio,
        median_nearest_neighbor_distance=median_nn,
        radial_outlier_fraction=outlier_fraction,
        reasons=reasons,
    )

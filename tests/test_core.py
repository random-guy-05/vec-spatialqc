import numpy as np

from vec_spatialqc.core import analyze_coordinates


def test_line_cloud_is_geometry_failure():
    x = np.linspace(0, 1, 100)
    coords = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
    report = analyze_coordinates(coords)
    assert report.classification == "GEOMETRY_FAILURE"
    assert report.geometric_rank == 1


def test_random_3d_cloud_is_ok():
    rng = np.random.default_rng(3)
    report = analyze_coordinates(rng.normal(size=(300, 3)))
    assert report.classification == "SPATIAL_QC_OK"
    assert report.geometric_rank == 3

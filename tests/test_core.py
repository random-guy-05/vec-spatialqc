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


def test_rank_is_scale_invariant_for_tiny_coordinates():
    rng = np.random.default_rng(4)
    cloud = rng.normal(size=(300, 3))
    large = analyze_coordinates(cloud)
    tiny = analyze_coordinates(cloud * 1e-10)
    assert large.geometric_rank == tiny.geometric_rank == 3
    assert large.classification == tiny.classification == "SPATIAL_QC_OK"


def test_anisotropy_measure_is_rotation_invariant():
    rng = np.random.default_rng(5)
    cloud = rng.normal(size=(400, 3)) * np.array([1.0, 1.0, 0.0005])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    original = analyze_coordinates(cloud)
    rotated = analyze_coordinates(cloud @ q)
    assert original.classification == rotated.classification == "SPATIAL_WARNING"
    np.testing.assert_allclose(
        original.singular_value_ratio,
        rotated.singular_value_ratio,
        rtol=1e-10,
    )

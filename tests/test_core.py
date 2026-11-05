import numpy as np
import pytest

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


def test_zero_mad_shell_does_not_hide_extreme_radial_outliers():
    shell = np.r_[np.eye(3), -np.eye(3)]
    cloud = np.r_[shell, [[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]]
    report = analyze_coordinates(cloud)
    assert report.geometric_rank == 3
    assert report.duplicate_fraction == 0.0
    assert report.singular_value_ratio < 1000
    assert report.radial_outlier_fraction == 2 / 8
    assert report.classification == "SPATIAL_WARNING"


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
@pytest.mark.parametrize("rotation_seed", [None, 31, 47])
@pytest.mark.parametrize("translated", [False, True])
def test_shell_outlier_detection_is_stable_across_units_and_rotation(
    scale, rotation_seed, translated
):
    rng = np.random.default_rng(10)
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    shell = np.r_[directions, -directions]
    far = rng.normal(size=(15, 3))
    far /= np.linalg.norm(far, axis=1, keepdims=True)
    cloud = np.r_[shell, 100 * far, -100 * far]
    if rotation_seed is not None:
        rotation, _ = np.linalg.qr(np.random.default_rng(rotation_seed).normal(size=(3, 3)))
        cloud = cloud @ rotation
    if translated:
        cloud = cloud + [3.0, -7.0, 2.0]
    # Equal radii have zero or near-zero MAD depending on roundoff.
    report = analyze_coordinates(cloud * scale)
    assert report.cells == 630
    assert report.geometric_rank == 3
    assert report.duplicate_fraction == 0.0
    assert report.radial_outlier_fraction == 30 / 630
    assert report.classification == "SPATIAL_WARNING"


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
def test_equal_radius_shell_and_tiny_jitter_remain_clean(scale):
    rng = np.random.default_rng(25)
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    for jitter in [0.0, 1e-12, 0.001]:
        varied = directions.copy()
        varied[:30] *= 1 + jitter
        cloud = np.r_[varied, -varied] @ rotation
        report = analyze_coordinates((cloud + [3.0, -7.0, 2.0]) * scale)
        assert report.radial_outlier_fraction == 0.0
        assert report.classification == "SPATIAL_QC_OK"


def test_non_degenerate_mad_keeps_existing_outlier_calculation():
    rng = np.random.default_rng(38)
    ordinary = rng.normal(size=(300, 3))
    far = rng.normal(size=(15, 3))
    far /= np.linalg.norm(far, axis=1, keepdims=True)
    cloud = np.r_[ordinary, -ordinary, 100 * far, -100 * far]
    radii = np.linalg.norm(cloud - cloud.mean(axis=0), axis=1)
    median_radius = np.median(radii)
    mad_scale = 1.4826 * np.median(np.abs(radii - median_radius))
    assert mad_scale > 0.001 * median_radius
    assert mad_scale > 64 * np.finfo(np.float64).eps * np.max(radii)
    expected = np.mean(np.abs(radii - median_radius) / mad_scale > 8)
    assert expected == 30 / 630
    report = analyze_coordinates(cloud)
    assert report.radial_outlier_fraction == expected
    assert report.classification == "SPATIAL_WARNING"

import subprocess
import sys

import anndata as ad
import numpy as np


def _write(path, coords=None):
    data = ad.AnnData(X=np.ones((120, 8), dtype=np.float32))
    if coords is not None:
        data.obsm["spatial_3D"] = np.asarray(coords, dtype=np.float32)
    data.write_h5ad(path)


def test_cli_good_line_and_missing_cases(tmp_path):
    rng = np.random.default_rng(8)
    good = tmp_path / "good.h5ad"
    line = tmp_path / "line.h5ad"
    missing = tmp_path / "missing.h5ad"

    _write(good, rng.normal(size=(120, 3)))
    x = np.linspace(0, 10, 120)
    _write(line, np.c_[x, np.zeros_like(x), np.zeros_like(x)])
    _write(missing)

    good_run = subprocess.run(
        [sys.executable, "-m", "vec_spatialqc.cli", str(good)],
        capture_output=True,
        text=True,
        check=False,
    )
    line_run = subprocess.run(
        [sys.executable, "-m", "vec_spatialqc.cli", str(line)],
        capture_output=True,
        text=True,
        check=False,
    )
    missing_run = subprocess.run(
        [sys.executable, "-m", "vec_spatialqc.cli", str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert good_run.returncode == 0, good_run.stdout + good_run.stderr
    assert line_run.returncode == 4
    assert missing_run.returncode == 2

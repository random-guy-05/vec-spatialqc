from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path

from .core import analyze_coordinates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run geometry sanity checks on VEC spatial_3D coordinates."
    )
    parser.add_argument("file", type=Path)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args(argv)

    try:
        import anndata as ad

        data = ad.read_h5ad(args.file, backed="r")
        try:
            if "spatial_3D" not in data.obsm:
                raise KeyError("missing obsm['spatial_3D']")
            report = analyze_coordinates(data.obsm["spatial_3D"])
        finally:
            with suppress(Exception):
                data.file.close()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print(report.classification)
    for reason in report.reasons:
        print(f"- {reason}")
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    if report.classification == "GEOMETRY_FAILURE":
        return 4
    if report.classification == "SPATIAL_WARNING":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

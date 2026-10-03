"""Measure UASD thresholds from YOLO training labels only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def compute(data_yaml):
    data_yaml = Path(data_yaml).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"]).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    root = root.resolve()
    label_dir = root / "labels" / "train"
    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)
    files = sorted(label_dir.rglob("*.txt"))
    areas, backgrounds = [], 0
    manifest = hashlib.sha256()
    class_counts = {}
    for path in files:
        content = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        manifest.update((rel + "\0" + hashlib.sha256(content).hexdigest() + "\n").encode())
        rows = [line.split() for line in content.decode("utf-8").splitlines() if line.strip()]
        if not rows:
            backgrounds += 1
        for line_no, row in enumerate(rows, 1):
            if len(row) != 5:
                raise ValueError(f"Expected 5 YOLO fields: {path}:{line_no}")
            values = np.asarray([float(v) for v in row], dtype=np.float64)
            cls, cx, cy, width, height = values
            if (not np.isfinite(values).all() or cls != int(cls)
                    or not 0 <= cls < int(cfg["nc"])
                    or not 0 <= cx <= 1 or not 0 <= cy <= 1
                    or not 0 < width <= 1 or not 0 < height <= 1):
                raise ValueError(f"Invalid YOLO label: {path}:{line_no}")
            areas.append(width * height)
            class_counts[str(int(cls))] = class_counts.get(str(int(cls)), 0) + 1
    if not areas:
        raise ValueError("No valid training boxes")
    areas = np.asarray(areas, dtype=np.float64)
    quantiles = {}
    for q in [5, 10, 20, 25, 30, 50, 75, 90]:
        value = float(np.percentile(areas, q, method="linear"))
        quantiles[f"q{q:02d}"] = {
            "norm_area": value,
            "area_at_640": value * 640 * 640,
            "equivalent_side_at_640": float(np.sqrt(value * 640 * 640)),
        }
    return {
        "dataset": "S-UODAC2020",
        "split": "train",
        "area_definition": "normalized YOLO width * height, before augmentation",
        "percentile_method": "numpy.percentile(method='linear')",
        "numpy_version": np.__version__,
        "label_files": len(files),
        "background_label_files": backgrounds,
        "gt_boxes": len(areas),
        "class_box_counts": class_counts,
        "label_manifest_sha256": manifest.hexdigest(),
        "minimum_area": float(areas.min()),
        "maximum_area": float(areas.max()),
        "quantiles": quantiles,
        "selected_percentile": 25,
        "selected_tau": quantiles["q25"]["norm_area"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compute(args.data_yaml)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

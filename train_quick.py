"""Screen model architectures before running the full train.py experiment."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_yaml", help="Model architecture YAML")
    parser.add_argument("data_yaml", help="Dataset YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Screening epochs (default: 100)")
    parser.add_argument("--name", help="Run folder name under runs_screen")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    run_name = args.name or "{}_{}_{}ep".format(
        Path(args.model_yaml).stem, Path(args.data_yaml).stem, args.epochs
    )
    model = YOLO(args.model_yaml)
    model.train(
        data=args.data_yaml,
        project="runs_screen",
        name=run_name,
        epochs=args.epochs,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        pretrained=False,
        imgsz=640,
        workers=8,
        device=0,
        iou=0.4,
    )


if __name__ == "__main__":
    main()

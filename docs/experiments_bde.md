# B+D+E: PGAER and UASD localization loss

This experiment adds E (Underwater Adaptive Scale-Dynamic IoU loss) to the
existing B=FreqFusion and D=PGAER-v1 model. PGAER is an experimental design
supplied for this project, not an official MSAD/MSEA implementation.

## Configuration

| Experiment | Model YAML | Localization loss |
| --- | --- | --- |
| B | `hyuod_freqfusion.yaml` | Original CIoU |
| B+D | `hyuod_freqfusion_pgaer.yaml` | Original CIoU |
| B+D+E | `hyuod_freqfusion_pgaer_uasd.yaml` | UASD |

All YAML files are in `train_yaml`. B+D and B+D+E have identical backbones,
heads, inference parameters (12,572,892 with 4 classes), and detection strides
`[8,16,32]`. PGAER consumes `[9,10,34]` at layer 35; Detect consumes
`[35,38,41]` at layer 42. E adds no inference layers or inference computation.

Only the new model YAML enables the loss:

```yaml
uasd_loss: true
uasd_delta: 0.5
uasd_tau: 0.0031995739062500002
```

Omitting `uasd_loss` retains the original CIoU path. Enabling it without a
valid `uasd_tau` is an error. Existing B, A+B, B+C and B+D models remain opt-out.
The UASD switch is intended for this standard Detect model, not end-to-end,
segmentation or oriented-box criteria.

## Dataset-adaptive threshold

The threshold is the q25 of normalized `width * height` from original
S-UODAC2020 training labels only. It is fixed before training. Matched GT
areas during training reflect the current resized/augmented image.

| Quantile | Normalized area |
| --- | ---: |
| q20 | 0.002732599125 |
| q25 | 0.0031995739062500002 |
| q30 | 0.0036878550375 |

Statistics cover 4,745 label files, 76 backgrounds, and 35,487 GT boxes.
The full report, percentile method and label-manifest SHA-256 are in
[`uasd_scale_stats_suodac2020.json`](uasd_scale_stats_suodac2020.json).

To reproduce the statistics after configuring your dataset path:

```bash
python tools/compute_uasd_tau.py train_yaml/SUODAC2020.yaml --output scale_stats.json
```

The script checks normalized YOLO labels and uses NumPy's linear percentile
method. It does not inspect the validation split. If training labels change,
recompute and record the threshold before comparing experiments.

## Loss implementation

For normalized matched GT area `a`, let `s = delta * (1 - clamp(a/tau, 0, 1))`
and `L_center = rho²/c²`. The supplied SD formulation can be written as:

```text
UASD = (1-s) * CIoU + s - 2*s*L_center
```

For targets with `a >= tau`, this implementation directly returns the
repository's original CIoU values. Small-target arithmetic uses FP32 and the
same repository CIoU convention rather than a second, subtly different
width/height/epsilon definition. The formula reduces overlap/aspect-ratio
weight and increases relative center-distance weight for small targets.

GT area is computed before division by feature stride. Both box area and
image area are calculated in FP32: FP16 cannot represent `640 * 640`.
Classification loss, DFL, task-aligned assignment and loss gains are unchanged.

## Manual training on the deployed 4090D project

```bash
cd /root/my_HyUOD_BDE
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/bin/python train_quick.py train_yaml/hyuod_freqfusion_pgaer_uasd.yaml train_yaml/SUODAC2020.yaml --epochs 400
```

The deployed project reuses `/root/datasets/S-UODAC2020` and the existing
Python environment. For a fresh clone elsewhere, set the dataset YAML path
for that machine; the repository's existing dataset YAML may point to A.
Defaults remain batch 16, SGD, lr0=0.01, imgsz=640, workers=8, device=0,
pretrained=False, seed=0 and patience=0. `--name` is optional; without it the
default run name is `hyuod_freqfusion_pgaer_uasd_SUODAC2020_400ep`.

Start from scratch. Do not resume a B+D checkpoint with the new loss when
comparing B+D against B+D+E. Omitting `--epochs 400` uses the 100-epoch
screening default.

Validation for this deployment is performed on CPU while an existing GPU job
is running. It does not run training epochs, backward calls or optimizer
steps. GPU AMP/full-training memory and convergence remain unmeasured for E.

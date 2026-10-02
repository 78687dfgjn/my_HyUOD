# HyUOD B and B+C experiments

Both experiments start from the original three-scale HyUOD model.

| Model | YAML | Change |
| --- | --- | --- |
| Baseline | `train_yaml/hyuod_base.yaml` | Original P3/P4/P5 detection |
| A | `train_yaml/hyuod_p2inject.yaml` | P2 injection |
| B | `train_yaml/hyuod_freqfusion.yaml` | FreqFusion at P4 → P3 |
| B+C | `train_yaml/hyuod_freqfusion_srconv.yaml` | B plus SRConv at backbone layer 12 |

## Implementation

B uses one `FreqFusionConcat` at layer 33, with inputs `[15, 32]`
(HR backbone P3, LR neck P4), compressed channels 64 and
`feature_resample=False`. The output channel order preserves the original
Concat: upsampled LR first, HR second. The module has its own two-input parser
branch and uses the official PyTorch CARAFE fallback when MMCV is absent.

The FreqFusion source is adapted from
[Linwei-Chen/FreqFusion](https://github.com/Linwei-Chen/FreqFusion), commit
`3fb0c70637a3c194fb74294d3ce4681958b26241`. Fallback debug prints were removed.

B+C keeps B's head and FreqFusion code unchanged. Its only backbone YAML change is:

```diff
- [-3, 1, Conv, [256, 3, 2]]
+ [-3, 1, SRConv, [256, 3, 2, 1.0, 4, 5]]
```

SRConv follows the structure in the
[official BSR5 backbone](https://github.com/H1kari06/Underwater-object-detection/blob/main/Spatial-Residual/BSR5-DETR/src/nn/backbone/bsr5.py),
using HyUOD's Conv/BatchNorm/SiLU. The YAML explicitly selects `e=1.0`.
SRConv belongs to `base_modules`, not `repeat_modules`.
Both models retain three detection heads with strides `[8, 16, 32]` and
the repository's effective `s` scale selection.

With 4 classes, B has 12,547,980 parameters; B+C has 12,614,988 (+67,008).

## Dataset and environment

Use the existing `hyuod` environment (PyTorch 2.1.1+cu118).
Set the dataset YAML `path` for the server being used. The committed
`SUODAC2020.yaml` points to A's independent B+C dataset at
`/hy-tmp/hyuod-bc/datasets/SUODAC2020`; B's dataset remains at
`/root/datasets/S-UODAC2020`.

The B/B+C dataset contains 4,745 training and 797 validation images, including
76/12 background images. Images, YOLO labels, transmission maps (`t`) and
atmospheric maps (`a`) must all be present. A's earlier dataset omitted these
background images, so it should not be substituted when comparing B with B+C.

## Start training manually

For the deployed B+C project on A:

```bash
source /usr/local/miniconda3/bin/activate hyuod
cd /root/HyUOD_BC
python train_quick.py train_yaml/hyuod_freqfusion_srconv.yaml train_yaml/SUODAC2020.yaml --epochs 400
```

For B, select `train_yaml/hyuod_freqfusion.yaml` instead and use B's dataset YAML.
Without `--epochs`, the screening default is 100. `--name` is optional.
Settings remain batch 16, SGD, lr0=0.01, imgsz=640, workers=8, device=0,
iou=0.4 and pretrained=False. Do not initialize B+C from B's trained weights
when comparing these architectures from scratch.

The default B+C 400-epoch output directory is
`runs_screen/hyuod_freqfusion_srconv_SUODAC2020_400ep` (incremented if it exists).

## Checks completed on A (2026-10-03)

The deployed B+C project passed model construction, trainer initialization,
AMP checks, FP32/AMP forward and loss on a real `[16,9,640,640]` training batch,
and AMP forward/loss on a real `[32,9,384,672]` validation batch.
All 22,168 dataset files matched B by SHA-256.
The CLI path was checked with its final training call intercepted.

These checks ran zero training epochs, backward calls and optimizer steps.
Full training is left to the user. They establish startup and forward/loss
compatibility; full training memory usage and convergence were not measured.

Server evidence is stored at `/hy-tmp/hyuod-bc/checks/result.json` and
`verify.log`. Existing A source files and installed packages were preserved;
the B server project was not changed.

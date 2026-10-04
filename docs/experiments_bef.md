# B+E+F: FreqFusion, UASD and TGER

F is the user-supplied Transmission-Guided Edge-Gaussian Refinement design,
a HyUOD adaptation using Scharr/Gaussian mechanisms, not official LEGNet EGA.

## Changes relative to BE

- New `ultralytics/nn/modules/tger.py`: reduced-width P3 projection, fixed
  Gaussian/Scharr filters, FP32 edge magnitude, transmission/semantic gate,
  lightweight refinement and zero-initialized residual alpha.
- `ultralytics/nn/modules/__init__.py`: TGER import/export.
- `ultralytics/nn/tasks.py`: two-input parser, output channels equal to B-P3.
  TGER is not a base/repeat module.
- New `train_yaml/hyuod_freqfusion_uasd_tger.yaml`: identical BE backbone
  and UASD settings; TGER at layer 35 from `[10,34]`, ratio 0.5.

Layer 36 explicitly reads layer 34. Thus TGER refines only the P3 detection
input; the P4/P5 forward feature path keeps the BE source. Shared backbone
weights can still receive gradients from all detection outputs during training.
Detect is layer 42 from `[35,38,41]`, stride `[8,16,32]`.
No PGAER, SRConv or P2 injection is active in this model.

FreqFusion, UASD loss, PGAER, original BE YAML and training settings are unchanged.
UASD remains enabled with delta 0.5 and tau 0.0031995739062500002.
TGER adds exactly 34,240 parameters at s scale (hidden=64).

## B server deployment

Independent project: `/root/my_HyUOD_BEF`.
Existing Python: `/root/miniconda3/bin/python`.
Existing dataset: `/root/datasets/S-UODAC2020`.
The deployment's dataset YAML points to this existing location.
No environment recreation, dataset duplication or trained checkpoint import.

Start the full run manually:

```bash
cd /root/my_HyUOD_BEF
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/bin/python train_quick.py \
  train_yaml/hyuod_freqfusion_uasd_tger.yaml \
  train_yaml/SUODAC2020.yaml \
  --epochs 400 \
  --name hyuod_freqfusion_uasd_tger_400ep
```

`--name` is optional; `--epochs 400` selects the full schedule (default 100).
Settings: SGD, lr0=0.01, batch=16, imgsz=640, workers=8, device=0,
pretrained=False, seed=0, patience=0. Initialize from scratch, without resume
or BE/BDE checkpoints. Formal training is left to the user.

## Structural and module verification

Evidence outside source: `/root/datasets/hyuod-preparation/bef-checks`.
FP32 and AMP with GradScaler: initial output is exactly B-P3; alpha gradient
is finite and nonzero; detail parameter gradients are initially zero as expected
and become finite/nonzero once alpha is nonzero. An initial standalone AMP
check without gradient scaling underflowed small branch gradients; the corrected
check uses GradScaler as the training loop does. No module change was needed.

With shared BE weights mapped into BEF, full-model initial outputs are bitwise
equal on a 9-channel 256x256 GPU input. Backbone/UASD settings match BE;
layer indices, inputs and strides match the configuration above.
The 80-class YAML parameter counts are BE 12,577,392 / BEF 12,611,632.

## GPU smoke result (2026-10-04)

RTX 4090 completed one epoch: 297 training batches, backward and optimizer
updates, and validation on 797 images; exit code 0. AMP was active. The initial
built-in AMP reference check skipped because its download failed; a separate
check using the existing local reference checkpoint subsequently passed.
All recorded batch losses and CSV values were finite; both saved checkpoints
were finite. There were no CUDA/dtype/shape exceptions. TGER alpha updated
from zero (saved maximum absolute value 0.0010747909545898438).

| Loss | Train | Validation |
| --- | ---: | ---: |
| box | 4.93168 | 4.00451 |
| cls | 4.87188 | 16.71580 |
| dfl | 4.05784 | 6.33737 |

Four-class BEF parameters: 12,582,220. CSV epoch time: 187.431 seconds.
Training progress reported about 18.2 GB memory. This checks execution,
not convergence or accuracy. No formal 400-epoch run was started.
Smoke output: `/root/my_HyUOD_BEF/runs_screen/hyuod_freqfusion_uasd_tger_smoke`.
Evidence: `structure.json`, `gpu_result.json`, `verify.log`, `smoke.log`,
`amp_check.log` under the external checks directory above.
No environment packages were changed. Existing projects/jobs remain separate.

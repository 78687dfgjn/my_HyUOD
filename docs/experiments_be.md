# B+E: FreqFusion with UASD localization loss

## Model

`train_yaml/hyuod_freqfusion_uasd.yaml` is copied from the pure B YAML,
`hyuod_freqfusion.yaml`, with only these additional settings:

```yaml
uasd_loss: true
uasd_delta: 0.5
uasd_tau: 0.0031995739062500002
```

Tau is the normalized GT area q25 of the S-UODAC2020 training set.
The backbone and head are identical to B. Layer 33 is FreqFusionConcat
from `[15,32]`; layer 35 is Conv; layer 41 is Detect from `[34,37,40]`.
Strides are `[8,16,32]`. There is no PGAER, SRConv or P2 injection in this
model. E adds no inference parameters or inference operations.

B keeps CIoU; BE enables UASD through the existing opt-in loss implementation.
No Python source changes are needed relative to commit `84eab174`.

## A server deployment

Independent project: `/root/my_HyUOD_BE`.
Existing environment: `/usr/local/miniconda3/envs/hyuod`.
Existing dataset: `/hy-tmp/hyuod-bc/datasets/SUODAC2020`.
Existing A, BC and BD projects remain separate.

Start the full experiment manually:

```bash
source /usr/local/miniconda3/bin/activate hyuod
cd /root/my_HyUOD_BE
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python train_quick.py \
  train_yaml/hyuod_freqfusion_uasd.yaml \
  train_yaml/SUODAC2020.yaml \
  --epochs 400 \
  --name hyuod_freqfusion_uasd_400ep
```

`--name` is optional. `--epochs 400` is required for the full schedule;
train_quick.py defaults to 100 epochs. Training starts from scratch:
SGD, lr0=0.01, batch=16, imgsz=640, workers=8, device=0,
pretrained=False, seed=0, patience=0. No checkpoint is loaded or resumed.

## Verification

Verification evidence is stored outside the source project at
`/hy-tmp/hyuod-be/checks`. The smoke output is
`/root/my_HyUOD_BE/runs_screen/smoke_be`.

Structural checks: B and BE backbones/heads are equal, inference parameter
delta is zero (12,577,392 each with the YAML's 80 classes), B uses CIoU,
BE uses UASD with the specified delta/tau, and key layers/strides match above.
The dataset overrides the detection class count to four during training.

On 2026-10-04, the A server RTX 2080 Ti completed a real 1-epoch GPU
AMP training/backward/optimizer and validation smoke run, exit code 0.
There were 297 training batches and 797 validation images. AMP checks passed;
no NaN/inf, CUDA, dtype or shape errors were observed. All CSV values and
both saved checkpoints were finite. The 4-class model has 12,547,980 parameters.

| Loss | Training | Validation |
| --- | ---: | ---: |
| box | 4.93672 | 4.20551 |
| cls | 4.85140 | 5.38229 |
| dfl | 4.05574 | 4.90209 |

The CSV epoch time was 297.386 seconds; training progress reported about
18.2 GB GPU memory. This short run verifies execution, not final convergence.
No 400-epoch training was started. No environment package changes were needed.
Evidence: `structure.json`, `gpu_result.json`, `verify.log`, and `smoke.log`
in `/hy-tmp/hyuod-be/checks`. Test outputs are excluded by `runs_screen/`
in `.gitignore` and are not included in the commit.

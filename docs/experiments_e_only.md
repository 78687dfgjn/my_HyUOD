# E-only: original HyUOD with UASD

`train_yaml/hyuod_uasd.yaml` copies `hyuod_base.yaml`, adding only:

```yaml
uasd_loss: true
uasd_delta: 0.5
uasd_tau: 0.0031995739062500002
```

No Python source changes are needed. The original backbone and head remain
unchanged. Detect at layer 42 reads `[35,38,41]`; no FreqFusion, PGAER,
TGER, SRConv or P2 injection is active.

## A deployment

Independent project: `/root/my_HyUOD_E`.
Existing environment: `/usr/local/miniconda3/envs/hyuod`.
Existing dataset: `/hy-tmp/hyuod-bc/datasets/SUODAC2020`.
The deployment data YAML points to this existing dataset. Historical weights,
training results and caches were excluded. No environment packages were changed.

## Checks completed on 2026-10-05

- `hyuod.yaml` and `hyuod_base.yaml` have identical configuration.
- H/E backbones and heads match; effective scale is s, P3 channels 128.
- Parameters with 80 classes: 12,524,878 each, delta 0.
- Strides `[8,16,32]`; Detect inputs `[35,38,41]`.
- Resetting seed to 0 before each model construction gives bitwise identical
  values for all 1,055 model state tensors.
- H uses CIoU; E uses UASD, delta 0.5, tau 0.0031995739062500002.
- Dataset train/transmission/atmospheric/validation directories exist.
- train_quick.py retains seed=0 and patience=0.

Evidence outside source: `/hy-tmp/hyuod-e/checks/result.json`, `yaml.diff`,
`verify.py`, and `verify.log`.

At the user's request, checks stopped here: no GPU smoke run, backward,
optimizer steps or training epochs were executed. Full GPU training readiness
has not been verified for this E-only deployment.

## Later training (not executed)

```bash
source /usr/local/miniconda3/bin/activate hyuod
cd /root/my_HyUOD_E
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python train_quick.py \
  train_yaml/hyuod_uasd.yaml \
  train_yaml/SUODAC2020.yaml \
  --epochs 400 \
  --name hyuod_uasd_400ep
```

Perform a separately authorized GPU smoke check before the full run.
Keep SGD, lr0=0.01, batch=16, imgsz=640, workers=8, device=0,
pretrained=False, seed=0, patience=0. Start from scratch without checkpoints.
The script defaults to 100 epochs; pass --epochs 400 for the full schedule.

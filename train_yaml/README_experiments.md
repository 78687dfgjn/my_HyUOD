# HyUOD P2 experiments

All configurations keep the original HyUOD backbone. With the current model
parser, the default scale is `s` (depth 0.50, width 0.50).

| Configuration | Detection scales | P2 use |
| --- | --- | --- |
| `hyuod.yaml` / `hyuod_base.yaml` | P3/P4/P5 | Original backbone and neck |
| `hyuod_p2_full.yaml` | P2/P3/P4/P5 | Preserved four-scale model with bottom-up PAN |
| `hyuod_p2inject.yaml` | P3/P4/P5 | Layer 9 -> SCDown -> concat with layer 35 -> C3k2(e=0.25) |

The baseline comes from the original repository:
https://github.com/White-cat-ed/HyUOD/blob/master/train_yaml/hyuod.yaml

## Screening

Run from the project root in the existing HyUOD environment. Dataset paths,
labels, and generated T/A images must already be configured for that dataset.

```bash
python train_quick.py train_yaml/hyuod_base.yaml train_yaml/SUODAC2020.yaml
python train_quick.py train_yaml/hyuod_p2_full.yaml train_yaml/SUODAC2020.yaml
python train_quick.py train_yaml/hyuod_p2inject.yaml train_yaml/SUODAC2020.yaml
```

These commands run sequentially if pasted together in one shell. Each defaults
to 100 epochs. Use `--epochs 200` for a longer screening run, or `--name NAME`
to choose a run folder. Results are stored in `runs_screen`; repeated names
automatically receive a suffix under the normal Ultralytics behavior.

The screening settings keep batch 16, SGD, initial learning rate 0.01,
random initialization, image size 640, and IoU 0.4. Workers are 8 and GPU is 0.
Compare runs at the same epoch budget and dataset split. Screening scores are
not a substitute for the final full experiment.

## Full experiment

The original `train.py` remains unchanged (400 epochs):

```bash
python train.py train_yaml/hyuod_p2inject.yaml train_yaml/SUODAC2020.yaml
```

No speed, FLOP, or accuracy improvement is guaranteed by the configuration.
Those numbers require measurements on the actual training environment.

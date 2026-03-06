# HADSB Open Source Core Training Bundle

This folder contains a runnable core of the final HADSB medical training pipeline.

## What is included
- `train.py`: multi-GPU training entrypoint (also works with `--nproc_per_node=1`)
- `hadsb/`: core runner, diffusion, network, semantic modules (primary package path)
- `i2sb/`: backward-compatible import shim to `hadsb`
- `dataset/medical.py`: paired medical dataset loader with configurable split options
- `guided_diffusion/`: open-source UNet/diffusion building blocks
- `configs/region_organ_config.py`: region/organ definitions
- `visualization/medical_viz.py`: training visualizations
- `utils/intensity_calib.py`: intensity utilities

## Data layout

### Standard split layout
`data_dir/train` and `data_dir/val` each contains:
- `LAVA/*.npy`
- `T2/*.npy`
- `PET/*.npy`
- optional: `lava_water/*.npy`, `lava_fat/*.npy`

### Flat layout
`data_dir` contains modality folders directly:
- `lava/*.npy`
- `T2/*.npy`
- `PET/*.npy`
- optional: `lava_water/*.npy`, `lava_fat/*.npy`

Use `--flat-data-structure` in this mode.

## Install (uv)
```bash
uv venv
source .venv/bin/activate
uv pip install -r open_source_core/requirements-core.txt
```

## Run (uv)

### Single GPU
```bash
uv run torchrun --nproc_per_node=1 open_source_core/train.py \
  --name core_single_gpu \
  --data-dir ./data/paired_dataset \
  --flat-data-structure \
  --val-ratio 0.2 \
  --num-itr 10000
```

### Multi GPU
```bash
uv run torchrun --nproc_per_node=2 open_source_core/train.py \
  --name core_multi_gpu \
  --data-dir ./data/paired_dataset \
  --flat-data-structure \
  --val-ratio 0.2 \
  --num-itr 10000
```

### Explicit validation split file (flat layout)
Create a text file with one group id per line (example: `A_0001`).
```bash
uv run torchrun --nproc_per_node=2 open_source_core/train.py \
  --name core_with_val_file \
  --data-dir ./data/paired_dataset \
  --flat-data-structure \
  --val-patients-file ./data/val_groups.txt
```

## Licenses
See `open_source_core/LICENSES/` for copied license texts used by this bundle.

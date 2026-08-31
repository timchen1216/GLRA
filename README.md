# GLRA: Multi-Object Tracking with Center-Aware Association and Lost Track Recovery

Official implementation of the master's thesis *Multi-Object Tracking with Center-Aware Association and Lost Track Recovery*, Institute of Artificial Intelligence Technology and Application, National Yang Ming Chiao Tung University.

GLRA is built on top of [SparseTrack](https://github.com/hustvl/SparseTrack) and adds two components to the tracking-by-detection pipeline:

1. **Center-aware association (DIoU).** The vanilla IoU cost inside SparseTrack's Depth Cascade Matching is replaced by a Distance-IoU cost. The added center-distance term gives a non-zero gradient for non-overlapping boxes, so tracks that have drifted past their detection are still ranked sensibly by the Hungarian solver.

2. **GLRA (GPR-based Lost track Re-Association).** Lost tracks are re-associated online against low-score detections using a Gaussian Process Regression motion prior instead of the Kalman filter's linear extrapolation. Two independent 1-D GP models predict the box center `(cx, cy)` from the track's recent observation history; box size is taken as the mean of the last few frames. Because GPR also returns a predictive variance, the association gate can be widened or closed per track according to how confident the motion prior is.

Unlike offline interpolation such as DTI, GLRA is causal: it only uses information available up to the current frame, so it can be used in a live tracking setting.

---

## Table of Contents

- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Pretrained detectors](#pretrained-detectors)
- [Running the tracker](#running-the-tracker)
- [Configuration](#configuration)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Installation

The codebase follows SparseTrack's environment: Python 3.8+, PyTorch with CUDA, and Detectron2.

```bash
git clone https://github.com/timchen1216/GLRA.git
cd GLRA

conda create -n glra python=3.8 -y
conda activate glra

# PyTorch (match the CUDA version on your machine)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install -r requirements.txt
pip install scikit-learn cython_bbox
```

`scikit-learn` is required by GLRA for the Gaussian Process models and is not part of the inherited SparseTrack requirements file.

For HOTA evaluation, [TrackEval](https://github.com/JonathonLuiten/TrackEval) is already vendored under `TrackEval/`.

## Data preparation

Datasets are expected one level above the repository root, in MOTChallenge layout:

```
<parent>
├── GLRA
├── MOT17
│   ├── train
│   └── test
├── MOT20
│   ├── train
│   └── test
└── dancetrack
```

Convert the annotations to COCO format and generate the half-split used for validation with the scripts under `tools/`. The tracking configs read `val_ann="val_half.json"` for validation runs and `val_ann="train.json"` (or the test annotation) otherwise, and `track.py` uses that field to decide which ground-truth files to score against.

## Pretrained detectors

GLRA does not train its own detector. It uses the public YOLOX-X weights released with ByteTrack:

| Dataset | Checkpoint |
| --- | --- |
| MOT17 | `bytetrack_x_mot17.pth.tar` \ `bytetrack_ablation.pth.tar` |
| MOT20 | `bytetrack_x_mot20.tar` \ `bytetrack_20_ablation.pth.tar` |
| DanceTrack | DanceTrack YOLOX-X weights |

Download them from the [ByteTrack model zoo](https://github.com/ifzhang/ByteTrack) and set `train.init_checkpoint` in the config file to the local path.

## Running the tracker

```bash
# MOT17 train 
CUDA_VISIBLE_DEVICES=0 python3 track.py --num-gpus 1 --config-file mot17_track_cfg.py

# MOT20 train
CUDA_VISIBLE_DEVICES=0 python3 track.py --num-gpus 1 --config-file mot20_track_cfg.py

# Ablation configs (used for the threshold and IoU-variant sweeps)
CUDA_VISIBLE_DEVICES=0 python3 track.py --num-gpus 1 --config-file mot17_ab_track_cfg.py
CUDA_VISIBLE_DEVICES=0 python3 track.py --num-gpus 1 --config-file mot20_ab_track_cfg.py

# Test-set submission
CUDA_VISIBLE_DEVICES=0 python3 track.py --num-gpus 1 --config-file mot17_test_cfg.py
CUDA_VISIBLE_DEVICES=0 python3 track.py --num-gpus 1 --config-file mot20_test_cfg.py

# DanceTrack
CUDA_VISIBLE_DEVICES=0 python3 track.py --num-gpus 1 --config-file dancetrack_sparse_cfg.py
```

Results are written to `<train.output_dir>/<track.experiment_name>/<track.folder_name>/` as MOTChallenge text files. `track.py` prints CLEAR/Identity metrics through `motmetrics` at the end of the run; HOTA is obtained by pointing TrackEval at the same result folder.

Any field in a config can be overridden from the command line:

```bash
python3 track.py --num-gpus 1 --config-file mot17_track_cfg.py \
    track.use_glra=False track.use_diou=False
```

Optional post-processing and visualization:

```bash
python3 interpolation.py   # DTI linear interpolation (offline)
bash tools/eval_hota.sh    # Evaluate metric
```

## Configuration

The tracking configs are Detectron2 `LazyConfig` files. The fields specific to this work live in the `track` dict:

| Key | Default (MOT17) | Meaning |
| --- | --- | --- |
| `use_diou` | `True` | Use DIoU instead of IoU in the cascade matching cost |
| `use_glra` | `True` | Enable GPR-based lost track re-association |
| `gpr_min_lost` | `1` | Minimum number of lost frames before a track is eligible |
| `gpr_max_lost` | `8` | Maximum gap GLRA will attempt to bridge |
| `gpr_min_obs` | `5` | Minimum observations needed to fit the GP |
| `gpr_history_len` | `30` | Length of the observation history kept per track |
| `glra_thresh` | `0.45` | Association cost threshold for the re-association stage |
| `glra_sigma_cap` | `20` | Reject matches whose predictive std exceeds this many pixels (`None` disables) |
| `glra_adaptive` | `False` | Widen the threshold as a function of predictive std |
| `glra_sigma_scale` | `30.0` | Std at which the full widening is applied |
| `glra_thresh_range` | `0.25` | Maximum widening allowed |
| `glra_max_thresh` | `0.85` | Absolute ceiling on the widened threshold |
| `use_gmc_history` | `False` | Compensate the observation history with global motion (for moving-camera sequences) |

Standard SparseTrack fields (`track_thresh`, `match_thresh`, `track_buffer`, `depth_levels`, `depth_levels_low`, `down_scale`, `confirm_thresh`, `mot20`) keep their original meaning.

Setting `use_diou=False, use_glra=False` reproduces the SparseTrack baseline under an identical detector and evaluation protocol, which is how every ablation number below was produced.

## Results

### Validation split (HOTA)

Both components are evaluated on the second half of the MOT17 and MOT20 training sequences, with the same YOLOX-X detector throughout.

| Association metric | MOT17 | MOT17 + GLRA | MOT20 | MOT20 + GLRA |
| --- | --- | --- | --- | --- |
| IoU (baseline) | 70.602 | - | 70.382 | - |
| GIoU | 70.663 | 70.948 | 70.573 | 70.685 |
| DIoU | 70.710 | **70.981** | 70.517 | **70.742** |
| CIoU | 70.711 | 70.981 | 70.517 | 70.742 |

The spread between IoU variants is small (at most 0.06 HOTA), while GLRA contributes 0.11 to 0.29 HOTA on top of a fixed metric. DIoU is used as the default because in the full configuration it improves both FN and IDs on MOT20 simultaneously, and CIoU's extra aspect-ratio term buys nothing measurable at the cost of another hyperparameter.

### Threshold sensitivity

Sweeping the re-association threshold shows an inverted-U: too permissive and ID switches spike, too strict and the recovered false negatives come back. HOTA peaks at the default on both datasets.

### Inference speed

Frames per second on a single GPU, detection included:

| Configuration | MOT17 | MOT20 |
| --- | --- | --- |
| Baseline | 16.15 | 12.18 |
| + DIoU | 16.07 | 12.03 |
| + GLRA | 12.93 | 7.20 |
| + DIoU + GLRA | 11.77 | 7.05 |

DIoU is essentially free. GLRA costs throughput because a GP is fitted per lost track per frame, and the cost grows with scene density, which is why MOT20 is affected more than MOT17.

### Where GLRA helps

GLRA is most useful in dense, fixed-camera scenes with short occlusions. Almost all correct recoveries involve small normalized displacement, which matches the regime a smooth motion prior can extrapolate into. It is not applied to DanceTrack, where the highly non-linear motion and frequent identity swaps put the recovered tracks outside the regime the prior can model.

> Test-set benchmark numbers are reported in the thesis. See the tables in Chapter 4 for the full CLEAR/HOTA/Identity breakdown and comparisons against ByteTrack, OC-SORT, BoT-SORT and SparseTrack.

## Repository layout

```
GLRA
├── tracker/            tracker core: association, GLRA re-association, evaluators
├── models/             YOLOX detector definition and builders
├── datasets/           dataloaders and MOTChallenge registration
├── tools/              dataset conversion and helper scripts
├── utils/              EMA, model fusion, misc utilities
├── TrackEval/          vendored HOTA evaluation toolkit
├── track.py            tracking + evaluation entry point
├── train.py            detector training entry point
├── interpolation.py    offline DTI post-processing
├── register_data.py    dataset registration for detectron2
└── *_cfg.py            LazyConfig files, one per dataset and protocol
```

## Citation

If you find this work useful, please cite the thesis:

```bibtex
@mastersthesis{chen2026glra,
  title  = {Multi-Object Tracking with Center-Aware Association and Lost Track Recovery},
  author = {Chen, Hong Fu},
  school = {National Yang Ming Chiao Tung University},
  year   = {2026}
}
```

## Acknowledgements

This work builds directly on [SparseTrack](https://github.com/hustvl/SparseTrack), and inherits code and design from [ByteTrack](https://github.com/ifzhang/ByteTrack), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [StrongSORT](https://github.com/dyhBUPT/StrongSORT), [Detectron2](https://github.com/facebookresearch/detectron2) and [TrackEval](https://github.com/JonathonLuiten/TrackEval). Thanks to the authors for releasing their code.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details. Note that the third-party components listed above remain under their own licenses.
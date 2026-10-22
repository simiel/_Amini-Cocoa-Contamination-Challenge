# Amini Cocoa Contamination Challenge

This repository is a computer-vision entry for the Amini Cocoa Contamination Challenge. It fine-tunes a pretrained YOLOv8n detector to find regions labeled `anthracnose`, `cssvd`, or `healthy` in cocoa images, then writes bounding-box files in the format Zindi expects for scoring. The work is organized as Jupyter notebooks plus the Ultralytics run directories those notebooks produced.

## Problem

Each training row is one bounding box on a cocoa image. `Train.csv` holds **9,792 boxes across 5,529 images** (up to 13 boxes on a single image; 2,320 images have more than one). After the notebooks strip class names and remap ids, the box counts are:

| Class | Boxes |
| --- | ---: |
| healthy | 4,280 |
| cssvd | 3,241 |
| anthracnose | 2,271 |

Nineteen rows in the raw file have a `class_id` that disagrees with `class` (18 `cssvd` rows stored as `0`, one `healthy` row stored as `1`). The notebooks rebuild ids in alphabetical order: `anthracnose` = 0, `cssvd` = 1, `healthy` = 2.

`Test.csv` lists **1,626 images** with empty label columns. `SampleSubmission.csv` has one placeholder row per test image (`Image_ID`, `class`, `confidence`, `ymin`, `xmin`, `ymax`, `xmax`). Model submissions emit one row per detected box, so they are longer than the sample file.

The image archive is not in git. Training logs record the unpacked set as **4,699 train images and 830 validation images (1,556 boxes)**, with no background images left after the split. That split is an image-level 15% holdout (`test_size=0.15`, `random_state=42`) so every box from one photo stays in the same fold. Ultralytics also restored **415 corrupt JPEGs** during the scan; the scan summary then reports 0 corrupt files.

YOLO labels ship inside `dataset.zip` as normalized `class x_center y_center width height` text files. The notebooks check that image stems match label stems, draw those boxes, and train on the text files. `Train.csv` supplies class names, pixel-corner boxes, and the image ids used for the split and the submission schema.

## Approach

The pipeline in `Original.ipynb`, `Starter.ipynb`, and `Cocoa_Disease_Starter_Notebook.ipynb` is:

1. Unpack `dataset.zip` into `dataset/images/{train,val,test}` and `dataset/labels/{train,val}`.
2. Audit image/label pairs and repair `class_id`.
3. Move the validation image ids, and their label files, out of the train directories.
4. Write `data.yaml` with three class names and absolute paths to those directories.
5. Fine-tune `yolov8n.pt` with Ultralytics (`pretrained=True`, batch size 8, patience 5).
6. Run validation and plot precision-recall, F1, and confusion-matrix figures.
7. Predict on the test images and save a Zindi CSV. Images with no boxes get an explicit placeholder row (`None` in `Original.ipynb`, the string `"2"` in `Starter.ipynb`) so every test id is still present.

`scripts/scrap.ipynb` is the same pipeline with markdown around each stage. It names further work — class-imbalance augmentation, larger YOLO sizes, learning-rate and batch-size sweeps, ensembles — as ideas. Those steps are not implemented. Training uses the Ultralytics defaults recorded in each `args.yaml` (mosaic, horizontal flip, scale, HSV jitter, RandAugment, random erasing). The notebooks flag the healthy-heavy label distribution and leave it unweighted.

`scripts/variate_classes.py` is separate from the detector. It copies `SampleSubmission.csv` and replaces the class column with a random draw from the three class names. `submission_varied_20250424_130356.csv` is one output of that script (confidence `0.5` and the sample-file boxes), not a model prediction.

## Results

Leaderboard scores are not stored in this repository. The numbers below are validation metrics recorded in notebook outputs and in `runs/detect/*/results.csv`.

### Best recorded checkpoint

`Original.ipynb` fine-tunes YOLOv8n for 10 epochs on an NVIDIA L4 (Ultralytics 8.3.84, PyTorch 2.5.1, CUDA, image size 1024, batch 8). The run finished in **0.219 hours**. Ultralytics selected `best.pt` from epoch 9. Re-validating that checkpoint on the 830-image split gives:

| | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| all (830 images, 1,556 boxes) | 0.717 | 0.588 | 0.661 | 0.415 |
| anthracnose (252 images, 385 boxes) | 0.765 | 0.540 | 0.639 | 0.385 |
| cssvd (306 images, 465 boxes) | 0.681 | 0.617 | 0.658 | 0.423 |
| healthy (272 images, 706 boxes) | 0.703 | 0.608 | 0.687 | 0.436 |

Epoch 10 is slightly higher on mAP50 (0.666) and slightly lower on mAP50-95 (0.413). Reported fitness for the best checkpoint is 0.439. Validation speed on the L4 was 0.3 ms preprocess, 1.8 ms inference, and 1.4 ms postprocess per image.

The checked-in `runs/detect/train/` directory does not contain that `best.pt` or a `results.csv`. It has `args.yaml` (the file lists `device: cpu`) and three training-batch previews. The 10-epoch metrics above come from the notebook log, which trained with `device=0` on the L4.

### Local runs

Later notebooks train on an Apple M3 (Ultralytics 8.3.115, Python 3.13.3, PyTorch 2.7.0, MPS) and save weights under `runs/detect/train2` through `train5`. Each row is the epoch with the highest mAP50-95 in that run’s `results.csv`, with the other metrics from the same epoch.

| Run | Device in `args.yaml` | Epochs logged | Image size | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train2 | mps | 1 of 10 configured | 1024 | 0.491 | 0.291 | 0.275 | 0.124 |
| train4 | mps | 2 | 640 | 0.421 | 0.237 | 0.198 | 0.109 |
| train3 | mps | 2 | 640 | 0.365 | 0.121 | 0.102 | 0.065 |
| train5 | mps | 5 | 1024 | 0.308 | 0.141 | 0.106 | 0.045 |

`train5` took **28.405 hours**. Its last epoch is below the epoch-1 peak (precision 0.359, recall 0.052, mAP50 0.049, mAP50-95 0.030). The training log’s own re-validation of `train5/weights/best.pt` on MPS is precision 0.281, recall 0.102, mAP50 0.071, mAP50-95 0.032.

A later cell in `Starter.ipynb` loads that same `train5` checkpoint path and calls `model.val()` on CPU. That pass is saved as `runs/detect/val4` and prints precision 0.500, recall 0.482, mAP50 0.463, mAP50-95 0.195. Those figures do not match the training-time MPS evaluation, so the table above is the run log and the val4 numbers are a separate recorded pass.

`runs/detect/val`, `val2`, `val3`, and `val4` store confusion matrices, precision-recall curves, F1 curves, and label-versus-prediction batches. Only `val4` has numeric metrics captured in a notebook output.

### Submission files

| File | Rows | Test images | Notes |
| --- | ---: | ---: | --- |
| `BenchmarkSubmission-1epoch.csv` | 2,144 | 1,626 | Placeholder class `None` where nothing was detected |
| `BenchmarkSubmission-2epochs.csv` | 5,841 | 1,626 | Placeholder class `2` for misses (22 rows) |
| `BenchmarkSubmission-5epochs.csv` | 4,218 | 1,626 | Written by `Starter.ipynb` from the `train5` checkpoint; 67 placeholder rows |

## Repository layout

```
.
├── Original.ipynb                         # Colab pipeline; 10-epoch CUDA training log
├── Starter.ipynb                          # Local MPS pipeline; writes the 5-epoch submission
├── Cocoa_Disease_Starter_Notebook.ipynb   # Same pipeline, plus a CIFAR-100 ResNet50 GPU smoke cell
├── data.yaml                              # 3-class YOLO config (absolute paths on one machine)
├── Train.csv                              # 9,792 labeled boxes
├── Test.csv                               # 1,626 test image ids
├── SampleSubmission.csv                   # Zindi column template
├── BenchmarkSubmission-*.csv              # Detector outputs
├── yolov8n.pt                             # Pretrained YOLOv8n weights
├── scripts/
│   ├── scrap.ipynb                        # Annotated walkthrough of the pipeline
│   └── variate_classes.py                 # Random-class sample file, not a model
└── runs/detect/                           # Ultralytics args, curves, batches, and weights
    ├── train/                             # args and batch previews only
    ├── train2/ … train5/                  # local runs, each with weights/best.pt and last.pt
    └── val/ … val4/                       # validation figures
```

`Cocoa_Disease_Starter_Notebook.ipynb` opens with `nvidia-smi` and a five-epoch CIFAR-100 ResNet50 fit. That cell checks TensorFlow and the GPU. The cocoa detector is the Ultralytics model in the later cells.

## Tech stack

- **Python** notebooks (logged interpreters: 3.11.11 on Colab, 3.13.3 locally)
- **Ultralytics YOLOv8n** on **PyTorch** (CUDA on an NVIDIA L4, and Apple MPS / CPU on an M3)
- **pandas** and **scikit-learn** (`train_test_split`) for the tables and the image-level split
- **OpenCV**, **Pillow**, **NumPy**, and **Matplotlib** for images and plots
- **PyYAML** for `data.yaml`
- **tqdm** for progress bars

There is no `requirements.txt`, `pyproject.toml`, or environment file. A commented install cell in `Starter.ipynb` lists the notebook packages. The local log uses a conda environment named `amini`.

## Getting started

**Prerequisites.** Python 3.11 or newer, Jupyter, and the packages below. A GPU is optional. The logged CUDA run used an NVIDIA L4; the logged local runs used Apple MPS. The project code does not read environment variables.

**Install.** From the commented cell in `Starter.ipynb`:

```bash
pip install pandas scikit-learn tqdm opencv-python pyyaml matplotlib ultralytics numpy pillow ipywidgets
```

Ultralytics installs PyTorch. `yolov8n.pt` is already at the repository root.

**Data.** `dataset.zip` is gitignored and is not in this checkout. Put it in the repository root before running a notebook. The prep cells delete any existing `dataset/` directory, recreate it, and unpack the zip.

**Config.** The checked-in `data.yaml` points at absolute paths under `/Users/adjei/Desktop/Amini Cocoa Contamination Challenge/dataset`. Re-run the notebook cell that writes `data.yaml`, or edit `path`, `train`, `val`, and `test` so they match the machine you are on. Class names and `nc: 3` can stay as they are.

**Train and submit.** Open `Starter.ipynb` (local) or `Original.ipynb` (Colab; that notebook mounts Google Drive). Run the cells from the top. The training cell calls:

```python
model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=5,       # Original.ipynb uses 10
    imgsz=1024,
    batch=8,
    device="mps",   # Original.ipynb uses device=0
    patience=5,
)
```

The following cell loads `runs/detect/train5/weights/best.pt` (or `runs/detect/train/weights/best.pt` in `Original.ipynb`), predicts on `dataset/images/test`, and writes a `BenchmarkSubmission-*.csv`.

**Random sample file.** `variate_classes.py` resolves `SampleSubmission.csv` relative to the current working directory. Run it from `scripts/`:

```bash
cd scripts
python variate_classes.py
```

It writes `submission_varied_<timestamp>.csv` in the repository root.

There is no build step and no test suite.

## Project status

Competition workspace. The notebooks, weights, and CSVs are an experiment log for the Amini challenge (commits from April through June 2025), not a packaged library or a deployed service. The strongest completed training log is the 10-epoch L4 run. The local MPS runs are shorter, and the five-epoch 1024-pixel run did not improve on its first epoch. Class imbalance is measured and still uncorrected.

## What this demonstrates

- A full detection competition loop: audit the tables, keep multi-box images intact when splitting, confirm YOLO labels visually, fine-tune a pretrained detector, and export a scorer-ready file that still accounts for images with zero detections.
- Care with the labels themselves. The pipeline finds the 19 `class` / `class_id` disagreements, rebuilds a stable three-class map, and checks that every training image has a label file before training starts.
- Experiment records that can be compared. The same YOLOv8n setup is logged on CUDA and on Apple MPS, at image sizes 640 and 1024, with checkpoints, per-epoch `results.csv` files, and per-class precision, recall, and mAP. The best logged checkpoint reaches **0.661 mAP50** and **0.415 mAP50-95** on 1,556 validation boxes after a 0.219-hour fine-tune, and the per-class table shows healthy, cssvd, and anthracnose separately instead of one pooled score.

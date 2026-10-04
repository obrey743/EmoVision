# EmoVision — Starter Kit (Clean Rebuild)

A clean, minimal baseline to (re)build your facial **emotion recognition** project from scratch using **Python**, **OpenCV**, and **TensorFlow/Keras**.

## 0) Quick Start

```bash
# Recommended: use Python 3.10 for best package compatibility
# (macOS Intel/Apple Silicon or Windows/Linux)

# 1) Create & activate a virtual env
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install deps
pip install -r requirements.txt

# 4) (macOS/Apple Silicon) – for GPU acceleration
#    If you're on Apple Silicon (M1/M2/M3), prefer:
#       pip install tensorflow-macos tensorflow-metal
#    and REMOVE plain 'tensorflow' from requirements.
#    If you hit issues with numpy/opencv on Python 3.12+,
#    switch to Python 3.10 as above.

# 5) Install the package (makes `python -m emovision...` and the CLI scripts work)
pip install -e ".[dev]"

# 6) Collect a small dataset with your webcam
python main.py collect --classes Angry Happy Neutral Sad Surprise --per-class 150

# 7) Train the baseline model (saves model, labels and training curves to models/)
python main.py train --epochs 30

# 8) Run live inference (or on a single image with --image photo.jpg)
python main.py infer
```

### Controls

| Stage   | Keys |
|---------|------|
| collect | `space` capture one · `a` toggle auto-capture · `n`/`p` next/previous class · `q` quit |
| infer   | `b` toggle probability bars · `q` quit |

Every stage accepts `-h` for its options, e.g. `python main.py train -h`. After `pip install -e .` the same
stages are also available as `emovision-collect`, `emovision-train` and `emovision-infer`.

### What each stage does

- **collect** – detects the largest face, crops it with a small margin, saves a 48×48 grayscale PNG per class.
- **train** – stratified train/val split, on-the-fly augmentation, class-balanced loss, early stopping and LR
  scheduling. Writes `models/emotion_model.keras`, `emotion_model.keras.labels.json` (class order) and
  `emotion_model_history.png`, and prints a per-class report and confusion matrix.
- **infer** – multi-face prediction with per-face temporal smoothing (`--smooth`), an "Uncertain" label below
  `--min-conf`, probability bars and an FPS counter.

### Tests

```bash
pytest
```

## 1) Project Layout

```
EmoVision/
├─ README.md
├─ main.py               # python main.py {collect,train,infer}
├─ pyproject.toml
├─ requirements.txt
├─ tests/
├─ src/
│  └─ emovision/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ train.py
│     ├─ infer.py
│     └─ data/
│        └─ collector.py
│     └─ utils/
│        ├─ dataset.py
│        ├─ face.py
│        └─ preprocess.py
├─ data/
│  ├─ raw/
│  └─ dataset/           # auto-created by collector.py
└─ models/               # trained models saved here
```

## 2) Notes on macOS & Numpy/OpenCV Errors

You previously hit a **numpy/OpenCV** mismatch on Python 3.12. To avoid this:
- Prefer **Python 3.10**.
- Install in this order: `pip install --upgrade pip` then `pip install -r requirements.txt`.
- If using Apple Silicon, prefer `tensorflow-macos` + `tensorflow-metal` (and remove plain `tensorflow`).

Common fix:
```bash
# If you already installed wrong versions
pip uninstall -y numpy opencv-python tensorflow tensorflow-macos tensorflow-metal
pip cache purge

# Reinstall with Python 3.10 active
pip install -r requirements.txt
# Or on Apple Silicon:
pip install numpy==1.26.4 opencv-python==4.10.0.84 tensorflow-macos tensorflow-metal
```

## 3) Roadmap (You can advance step-by-step)

- **Baseline (this kit):** Haar-cascade face detection + small CNN classifier (48×48 grayscale).
- **Improve data:** Better lighting, more subjects, balance classes; augmentations.
- **Model upgrades:** MobileNetV2 / EfficientNetV2 transfer learning.
- **Performance:** Quantize to TFLite/ONNX; real-time thresholds & smoothing.
- **Deployment:** Streamlit app / Flask API / mobile integration.

Happy building! 🚀

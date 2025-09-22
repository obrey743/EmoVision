import os
from glob import glob
from typing import List, Tuple
import numpy as np

def find_image_paths(root: str, classes: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    paths, labels = [], []
    class_to_idx = {c:i for i, c in enumerate(classes)}
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for p in glob(os.path.join(cls_dir, "*.png")) + glob(os.path.join(cls_dir, "*.jpg")) + glob(os.path.join(cls_dir, "*.jpeg")):
            paths.append(p)
            labels.append(class_to_idx[cls])
    return np.array(paths), np.array(labels), classes

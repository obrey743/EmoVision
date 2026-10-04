import os
from typing import List, Optional, Tuple

import numpy as np

from ..config import IMAGE_EXTS


def discover_classes(root: str) -> List[str]:
    """Sorted subdirectory names of `root`; each one is a class."""
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.'))


def find_image_paths(root: str, classes: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    classes = list(classes) if classes else discover_classes(root)
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in sorted(os.listdir(cls_dir)):
            if name.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(cls_dir, name))
                labels.append(idx)
    return np.array(paths), np.array(labels, dtype=np.int64), classes

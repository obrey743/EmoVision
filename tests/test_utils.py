import cv2
import numpy as np

from emovision.utils.dataset import discover_classes, find_image_paths
from emovision.utils.face import crop_face, largest_face
from emovision.utils.preprocess import load_and_preprocess, preprocess_image


def test_preprocess_shape_and_range():
    img = np.random.randint(0, 256, (120, 90), dtype=np.uint8)
    out = preprocess_image(img, 48)
    assert out.shape == (48, 48, 1)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_find_image_paths(tmp_path):
    for cls, n in {"Happy": 3, "Sad": 2}.items():
        (tmp_path / cls).mkdir()
        for i in range(n):
            cv2.imwrite(str(tmp_path / cls / f"{i}.PNG"), np.zeros((10, 10), np.uint8))
    (tmp_path / "Sad" / "notes.txt").write_text("ignore me")

    assert discover_classes(str(tmp_path)) == ["Happy", "Sad"]
    paths, labels, classes = find_image_paths(str(tmp_path))
    assert classes == ["Happy", "Sad"]
    assert len(paths) == 5
    assert labels.tolist() == [0, 0, 0, 1, 1]
    assert load_and_preprocess(paths[0]).shape == (48, 48, 1)


def test_find_image_paths_keeps_given_class_order(tmp_path):
    (tmp_path / "Sad").mkdir()
    cv2.imwrite(str(tmp_path / "Sad" / "a.png"), np.zeros((10, 10), np.uint8))
    _, labels, classes = find_image_paths(str(tmp_path), ["Angry", "Sad"])
    assert classes == ["Angry", "Sad"]
    assert labels.tolist() == [1]


def test_largest_face_and_crop_clamped():
    faces = np.array([[0, 0, 10, 10], [5, 5, 40, 40]])
    assert largest_face(faces) == (5, 5, 40, 40)
    assert largest_face(np.empty((0, 4), int)) is None

    gray = np.zeros((50, 50), np.uint8)
    crop = crop_face(gray, (0, 0, 50, 50), margin=0.2)
    assert crop.shape == (50, 50)

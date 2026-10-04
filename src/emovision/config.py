EMOTIONS_DEFAULT = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

IMG_SIZE = 48   # 48x48 grayscale, FER-style
CASCADE_PATH = None  # will fall back to OpenCV built-in if None
FACE_MARGIN = 0.1  # extra border around detected faces; keep identical for collection and inference

DATA_DIR = "data/dataset"
MODEL_PATH = "models/emotion_model.keras"
# Class names are stored next to the model so inference always matches training order
LABELS_SUFFIX = ".labels.json"
IMAGE_EXTS = ('.png', '.jpg', '.jpeg')

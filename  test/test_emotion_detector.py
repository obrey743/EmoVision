# utils/preprocess.py

import tensorflow as tf
import cv2
import numpy as np
from typing import Tuple, Union, Optional
from pathlib import Path
import logging


def preprocess_image(
    image_input: Union[str, np.ndarray, tf.Tensor], 
    target_size: Tuple[int, int] = (48, 48),
    normalize: bool = True,
    add_batch_dim: bool = True,
    interpolation: int = cv2.INTER_AREA
) -> tf.Tensor:
    """
    Loads and preprocesses a grayscale image for model input.

    This function handles multiple input types and provides comprehensive preprocessing
    with error handling and logging.

    Args:
        image_input: Image source - can be:
            - str/Path: Path to image file
            - np.ndarray: Image array (grayscale or color)
            - tf.Tensor: TensorFlow tensor
        target_size: Desired dimensions (width, height) - note OpenCV convention
        normalize: Whether to normalize pixel values to [0, 1]
        add_batch_dim: Whether to add batch dimension for model input
        interpolation: OpenCV interpolation method for resizing

    Returns:
        tf.Tensor: Preprocessed image tensor with shape:
            - (1, height, width, 1) if add_batch_dim=True
            - (height, width, 1) if add_batch_dim=False

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is unsupported or invalid
        RuntimeError: If processing fails
    """
    try:
        # Handle different input types
        if isinstance(image_input, (str, Path)):
            image = _load_image_from_path(image_input)
        elif isinstance(image_input, np.ndarray):
            image = _process_numpy_array(image_input)
        elif isinstance(image_input, tf.Tensor):
            image = _process_tensor(image_input)
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        # Validate image
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image")

        # Resize image
        if image.shape[:2] != target_size[::-1]:  # Convert to (height, width)
            image = cv2.resize(image, target_size, interpolation=interpolation)

        # Ensure image is float32 and normalized if requested
        image = image.astype(np.float32)
        if normalize and image.max() > 1.0:
            image = image / 255.0

        # Ensure proper shape (add channel dimension if needed)
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        elif len(image.shape) == 3 and image.shape[-1] != 1:
            # If it's a color image, convert to grayscale
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
            elif image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)[..., np.newaxis]

        # Add batch dimension if requested
        if add_batch_dim:
            image = image[np.newaxis, ...]

        # Convert to TensorFlow tensor
        return tf.convert_to_tensor(image, dtype=tf.float32)

    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise RuntimeError(f"Failed to preprocess image: {e}") from e


def _load_image_from_path(image_path: Union[str, Path]) -> np.ndarray:
    """Load image from file path."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image at: {image_path}")
    
    return image


def _process_numpy_array(image: np.ndarray) -> np.ndarray:
    """Process numpy array input."""
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError(f"Invalid image dimensions: {image.shape}")
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[-1] == 1:
            image = image.squeeze(-1)
    
    return image


def _process_tensor(image: tf.Tensor) -> np.ndarray:
    """Process TensorFlow tensor input."""
    # Convert to numpy for processing
    image_np = image.numpy()
    
    # Remove batch dimension if present
    if image_np.ndim == 4 and image_np.shape[0] == 1:
        image_np = image_np.squeeze(0)
    
    return _process_numpy_array(image_np)


def preprocess_batch(
    image_paths: list,
    target_size: Tuple[int, int] = (48, 48),
    normalize: bool = True,
    interpolation: int = cv2.INTER_AREA
) -> tf.Tensor:
    """
    Preprocess a batch of images efficiently.

    Args:
        image_paths: List of image file paths
        target_size: Desired dimensions (width, height)
        normalize: Whether to normalize pixel values
        interpolation: OpenCV interpolation method

    Returns:
        tf.Tensor: Batch of preprocessed images with shape (batch_size, height, width, 1)
    """
    batch_images = []
    
    for path in image_paths:
        try:
            img = preprocess_image(
                path, 
                target_size=target_size, 
                normalize=normalize,
                add_batch_dim=False,  # We'll stack them manually
                interpolation=interpolation
            )
            batch_images.append(img)
        except Exception as e:
            logging.warning(f"Skipping image {path}: {e}")
            continue
    
    if not batch_images:
        raise ValueError("No valid images found in batch")
    
    return tf.stack(batch_images, axis=0)


def get_image_stats(image: tf.Tensor) -> dict:
    """
    Get statistical information about the preprocessed image.

    Args:
        image: Preprocessed image tensor

    Returns:
        dict: Dictionary containing image statistics
    """
    return {
        'shape': image.shape.as_list(),
        'dtype': image.dtype.name,
        'min_value': float(tf.reduce_min(image)),
        'max_value': float(tf.reduce_max(image)),
        'mean_value': float(tf.reduce_mean(image)),
        'std_value': float(tf.math.reduce_std(image))
    }
"""Feature extraction for handcrafted and deep image descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

from .config import IMAGE_SIZE


@dataclass(frozen=True)
class DeepFeatureExtractor:
    model: object
    preprocess_input: Callable[[np.ndarray], np.ndarray]


def load_image(path: str, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(image_size)
    return np.asarray(image, dtype=np.uint8)


def rgb_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    features: list[np.ndarray] = []
    for channel_index in range(3):
        channel_hist, _ = np.histogram(image[..., channel_index], bins=bins, range=(0, 256), density=True)
        features.append(channel_hist.astype(np.float32))
    return np.concatenate(features)


def hsv_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    features: list[np.ndarray] = []
    for channel_index in range(3):
        channel_hist, _ = np.histogram(hsv_image[..., channel_index], bins=bins, range=(0, 256), density=True)
        features.append(channel_hist.astype(np.float32))
    return np.concatenate(features)


def glcm_texture_features(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayscale = cv2.equalizeHist(grayscale)
    # Optimize: use fewer distances/angles for speed, but keep levels correct for 8-bit images
    glcm = graycomatrix(
        grayscale,
        distances=[1],
        angles=[0, np.pi / 2],
        levels=256,
        symmetric=True,
        normed=True,
    )
    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return np.array([graycoprops(glcm, prop).mean() for prop in properties], dtype=np.float32)


def shape_features(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(5, dtype=np.float32)

    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, True))
    x, y, width, height = cv2.boundingRect(largest_contour)
    bounding_area = float(width * height) if width and height else 1.0
    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter else 0.0
    solidity = area / hull_area
    aspect_ratio = width / height if height else 0.0
    extent = area / bounding_area
    return np.array([area, perimeter, circularity, solidity, aspect_ratio + extent], dtype=np.float32)


def handcrafted_features(path: str) -> np.ndarray:
    image = load_image(path)
    return np.concatenate(
        [
            rgb_histogram(image),
            hsv_histogram(image),
            glcm_texture_features(image),
            shape_features(image),
        ]
    )


def extract_handcrafted_matrix(image_or_paths) -> np.ndarray:
    """Extract handcrafted features from image(s).
    Can accept:
    - Single numpy array (image)
    - List of image paths
    """
    # If it's a numpy array (single image), process it directly
    if isinstance(image_or_paths, np.ndarray):
        return np.concatenate([
            rgb_histogram(image_or_paths),
            hsv_histogram(image_or_paths),
            glcm_texture_features(image_or_paths),
            shape_features(image_or_paths),
        ])
    # If it's a list of paths, process all
    return np.vstack([handcrafted_features(path) for path in image_or_paths])


def build_deep_feature_extractor(backbone: str = "MobileNetV2", image_size: tuple[int, int] = IMAGE_SIZE) -> DeepFeatureExtractor:
    if backbone == "MobileNetV2":
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

        model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(*image_size, 3))
        return DeepFeatureExtractor(model=model, preprocess_input=preprocess_input)
    if backbone == "ResNet50":
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

        model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(*image_size, 3))
        return DeepFeatureExtractor(model=model, preprocess_input=preprocess_input)
    raise ValueError(f"Unsupported backbone: {backbone}")


def extract_deep_features(
    image_paths: list[str],
    extractor: DeepFeatureExtractor,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 32,
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for start_index in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start_index : start_index + batch_size]
        batch_images = []
        for path in batch_paths:
            image = load_image(path, image_size=image_size).astype(np.float32)
            batch_images.append(image)
        batch_array = np.asarray(batch_images, dtype=np.float32)
        batch_array = extractor.preprocess_input(batch_array)
        batch_features = extractor.model.predict(batch_array, verbose=0)
        arrays.append(batch_features)
    return np.vstack(arrays)

"""Face embedding agent using histogram matching."""

from __future__ import annotations

import cv2
import numpy as np


class FaceEmbeddingAgent:
    def __init__(self) -> None:
        print("[Init] Face embedding agent ready")

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Face image is empty")

        # Resize for consistency
        face_resized = cv2.resize(face_bgr, (128, 128), interpolation=cv2.INTER_AREA)

        # Extract histogram from each channel
        histograms = []
        for i in range(3):  # B, G, R
            hist = cv2.calcHist([face_resized], [i], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)

        # Combine all histograms into single embedding
        embedding = np.concatenate(histograms).astype(np.float32)

        return embedding
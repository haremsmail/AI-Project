"""Face embedding agent using histogram matching."""

from __future__ import annotations

import cv2
import numpy as np


class FaceEmbeddingAgent:
    """" the class face embeding to find face number in the video"""
    def __init__(self) -> None:
        print("[Init] Face embedding agent ready")
        """ to ensure second agent is ready """

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
     
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Face image is empty")

        # Resize for consistency
        face_resized = cv2.resize(face_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        """ inter area is importan for shrinkin the image"""
        """ change the face size to 128* 128"""   """ contains one face image in BGR format and returns a 768-dimensional embedding vector.  contains one croped facce all equal in  consistent size 128*128 and then extract histogram from each channel and combine them into a single embedding vector. the embedding vector is normalized to have values between 0 and 1."""


        # Extract histogram from each channel
        histograms = []
        """  create list to store histogram for each channel blue green red"""
        for i in range(3):  # B, G, R
            hist = cv2.calcHist([face_resized], [i], None, [256], [0, 256])
            """ haw calcuate histogram for each channel and store it in hist variable"""
            """ face resized list of image none it means no mask using  number ob bin each one for the 256 and the range of pixel values from 0 to 255"""
            hist = cv2.normalize(hist, hist).flatten()
            """" normalize image if two image have but one is dark and one is light the histogram will be different but if we normalize them they will be similar"""
            histograms.append(hist)

        # Combine all histograms into single embedding
        embedding = np.concatenate(histograms).astype(np.float32)
        """ all histogram stored in one in one1d array """

        return embedding
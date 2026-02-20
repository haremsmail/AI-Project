"""Wanted comparison agent using cosine similarity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .face_embedding_agent import FaceEmbeddingAgent
from .face_detection_agent import FaceDetectionAgent


logger = logging.getLogger(__name__)


class WantedComparisonAgent:
    def __init__(
        self,
        embedding_agent: FaceEmbeddingAgent,
        database_dir: str,
        threshold: float = 0.7,
        face_detector: Optional[FaceDetectionAgent] = None,
    ) -> None:
        self.embedding_agent = embedding_agent
        self.database_dir = database_dir
        self.threshold = threshold
        self.face_detector = face_detector
        self.database: List[Tuple[str, np.ndarray, str]] = []
        self._load_database()

    def _load_database(self) -> None:
        db_path = Path(self.database_dir) / "wanted_persons"
        if not db_path.exists():
            return

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for image_path in db_path.iterdir():
            if image_path.suffix.lower() not in image_exts:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            faces = [image]
            if self.face_detector is not None:
                detected = self.face_detector.detect_faces(image)
                if detected:
                    faces = [detected[0]]

            try:
                embedding = self.embedding_agent.get_embedding(faces[0])
            except Exception as exc:
                logger.warning("Failed to compute embedding for %s: %s", image_path, exc)
                continue

            label = image_path.stem
            self.database.append((label, embedding, str(image_path)))

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0
        vec_a = vec_a.reshape(1, -1)
        vec_b = vec_b.reshape(1, -1)
        return float(cosine_similarity(vec_a, vec_b)[0][0])

    def compare(self, embedding: np.ndarray) -> Dict[str, Any]:
        best_score = -1.0
        best_match: Optional[Tuple[str, str]] = None
        all_scores = []

        for label, db_embedding, image_path in self.database:
            score = self._cosine_similarity(embedding, db_embedding)
            all_scores.append((label, score))
            if score > best_score:
                best_score = score
                best_match = (label, image_path)

        is_match = best_score >= self.threshold
        
        # Log all comparisons for debugging
        if all_scores:
            logger.info(f"Comparison scores: {all_scores}")
        
        return {
            "match": is_match,
            "score": max(best_score, 0.0),
            "name": best_match[0] if best_match else None,
            "matched_image_path": best_match[1] if best_match else None,
            "all_scores": all_scores,
        }

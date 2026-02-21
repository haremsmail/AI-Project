"""Wanted comparison agent using cosine similarity."""

from __future__ import annotations

from pathlib import Path
""" easy to work of file and dicrectory path folder  and then image insted of using string to represent the path we can use path object and then we can use it to read the image and then we can use it to get the name of the image and then we can use it to get the path of the image"""
from typing import Any, Dict, List, Optional, Tuple
""" is type hint to specify the type of variable and then we can use it to specify the type of variable in the function and then we can use it to specify the type of variable in the class and then we can use it to specify the type of variable in the return type of the function and then we can use it to specify the type of variable in the parameter of the function and then we can use it to specify the type of variable in the parameter of the class and then we can use it to specify the type of variable in the parameter of the return type of the function and then we can use it to specify the type of variable in the parameter of the return type of the class and then we can use it to specify the type of variable in the parameter of the return type of the function and then we can use it to specify the type of variable in the parameter of the return type of the class and then we
it means data types"""

import logging

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
""" used to for sckit learn to compare to vector  then between 0 and 1 is higher value means more similar and then we can use it to compare the embedding of the face with the embedding of the wanted person and then we can use it to get the score of the comparison and then we can use it to get the name of the wanted person if the score is higher than the threshold and then we can use it to get the path of the image of the wanted person if the score is higher than the threshold"""

from .face_embedding_agent import FaceEmbeddingAgent
from .face_detection_agent import FaceDetectionAgent


logger = logging.getLogger(__name__)


class WantedComparisonAgent:
    """ compare face embiding and  wanted data base"""
    def __init__(
        
        self,
        embedding_agent: FaceEmbeddingAgent,
        database_dir: str,
        threshold: float = 0.4,
        face_detector: Optional[FaceDetectionAgent] = None,
    ) -> None:
        """ self is object instance
        vector of the face 
        thred load simliaity image bo dyarkrydn ruxsar la wenakany bnkadrau"""
        self.embedding_agent = embedding_agent
        self.database_dir = database_dir
        """ used to save data base folder path"""
        self.threshold = threshold
        """" Later, if similarity ≥ threshold → face is considered a match."""
        self.face_detector = face_detector
        """ used to crop face in data base"""
        self.database: List[Tuple[str, np.ndarray, str]] = []
        """ the name of person vector and files"""
        self._load_database()
        """ this all image found and croped and then extract the embedding and then save it in the data base list
        and stored in the data base"""

    def _load_database(self) -> None:
        db_path = Path(self.database_dir) / "wanted_persons"
        """ contains all image person in this folder"""
        if not db_path.exists():
            return

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        """ create this image allow extenstion only this can be read"""
        for image_path in db_path.iterdir():
            """ loop through all file in the folder"""
            if image_path.suffix.lower() not in image_exts:
                """" check it not this one it means not image"""
                continue
            image = cv2.imread(str(image_path))
            """" read the iamge using open cv """
            if image is None:
                continue

            faces = [image]
            if self.face_detector is not None:
                detected = self.face_detector.detect_faces(image)
                """ used to the detect image and then return the croped face"""
                if detected:
                    faces = [detected[0]]
                    """ this is ensure that we only use the first detected face for embedding extraction, which is important for consistency in the database. If multiple faces are detected, it could lead to ambiguity in matching later on."""

            try:
                embedding = self.embedding_agent.get_embedding(faces[0])
                """ vectory zhmary bo yaakm face and then extract the embedding for the face and then save it in the data base list and stored in the data base"""
            except Exception as exc:
                logger.warning("Failed to compute embedding for %s: %s", image_path, exc)
                continue

            label = image_path.stem
            """ the name of the image without the extension is used as the label for the person in the database. This allows for easy identification of the person based on the filename, which is important for matching later on."""
            self.database.append((label, embedding, str(image_path)))
            """ data base list and stored in the data base and then we can use it to compare the embedding of the face with the embedding of the wanted person and then we can use it to get the score of the comparison and then we can use it to get the name of the wanted person if the score is higher than the threshold and then we can use it to get the path of the image of the wanted person if the score is higher than the threshold"""

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """ compare two similar vector to see how similar they are """
        if vec_a.size == 0 or vec_b.size == 0:
            """ if any vector empty return zero """
            return 0.0
        vec_a = vec_a.reshape(1, -1)
        """ reashape the vector to be 2D array with one row and as many columns as needed. This is necessary because cosine_similarity expects 2D arrays as input, where each row represents a sample and each column represents a feature."""
        vec_b = vec_b.reshape(1, -1)
        return float(cosine_similarity(vec_a, vec_b)[0][0])
    """ used conise simlarity to compare the embedding of the face with the embedding of the wanted person and then we can use it to get the 
    score of the comparison and then we can use it to get the name of the wanted person if the score is higher than the threshold and then we can use it to get the path of the image of the wanted person if the score is higher than the threshold
    if zero completly different """

    def compare(self, embedding: np.ndarray) -> Dict[str, Any]:
        """" compare new face embedding with the with all face data base"""
        best_score = -1.0
        """ highest similiary maybe found"""
        best_match: Optional[Tuple[str, str]] = None
        """ name and simiar image store"""
        all_scores = []
        """ find all compare file to store in this array"""

        for label, db_embedding, image_path in self.database:
            score = self._cosine_similarity(embedding, db_embedding)
            """  similear between face and face embedding in the data ase"""
            all_scores.append((label, score))
            """ contains all name and score inside this for exmple harem :0.6"""
            if score > best_score:
                """ if more thana best score updated it """
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
"""" for example this
{
    'match': True,
    'score': 0.82,
    'name': 'john_doe',
    'matched_image_path': 'database/wanted_persons/john_doe.jpg',
    'all_scores': [('john_doe', 0.82), ('alice_smith', 0.45)]
}"""
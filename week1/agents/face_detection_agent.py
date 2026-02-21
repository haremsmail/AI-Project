"""Face detection agent using OpenCV Haar Cascade."""

from __future__ import annotations

from typing import List

import logging
""" instead of print statements, we use logging for better control over output and to log to files if needed. """
import cv2
import numpy as np
""" OpenCV is used for face detection, and NumPy is used for image processing and handling the detected face regions as arrays. """

logger = logging.getLogger(__name__)
""" used to create logger object"""



class FaceDetectionAgent:
    """Detects and extracts faces from images or webcam input."""

    def __init__(self) -> None:
        """" the none means function do not return data"""
        """Initialize the face detection agent with Haar Cascade classifier."""
        self.cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        """ save  location of the Haar Cascade XML file for face detection. This file is included with OpenCV and contains the trained model for frontal face detection.  it means tool and tool box """
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        """ lody harcascade detector to find face eyes smiales"""
        """ open file xml detect face and save in face_cascade variable"""
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        self.last_frame = None
        """ create a variable to store the frame none means image not loaded yet is means the letter the  agent is ready"""
        logger.info("FaceDetectionAgent initialized with Haar Cascade")

    def detect_from_path(self, image_path: str) -> List[np.ndarray]:
        """Detect faces from an image file. numpy find the face and return the face as array of image

        Args:
            image_path: Path to the input image file.

        Returns:
            List of detected face images as BGR numpy arrays.
        """
        try:
            image = cv2.imread(image_path)
            """ read the image from the given path and store it in the variable image. cv2.imread reads the image in BGR format by default."""
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return []
            self.last_frame = image
            return self._detect_faces(image)
        except Exception as exc:
            logger.exception(f"Error detecting faces from {image_path}")
            return []

    def detect_from_webcam(self, timeout_seconds: int = 30) -> List[np.ndarray]:
        """Detect faces from webcam input.

        Args:
            timeout_seconds: Maximum seconds to capture from webcam.

        Returns:
            List of detected face images as BGR numpy arrays.
        """
        faces = []
        """ dipslay all detect face"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Failed to open webcam")
                return []
            frame_count = 0
            """ to display picture from video"""
            while True:
                """  the infinte cammera open and frame means picture and ret means  success or not"""
                ret, frame = cap.read()
                if not ret:
                    break
                self.last_frame = frame
                detected_faces = self._detect_faces(frame)
                faces.extend(detected_faces)
                frame_count += 1
                """" if the not success stop  and if success face go to lastframe and then"""
                if frame_count > timeout_seconds * 30:  # ~30 FPS
                    break
                # Display frame with detections
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                """ convert  the  black white faster and better to th detect face"""
                face_rects = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
                )
                for x, y, w, h in face_rects:
                    """ shteky chargosha drust daka"""
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Webcam - Press Q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
        except Exception as exc:
            logger.exception("Error capturing from webcam")
        return faces

    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """" image inside memory """
        """Detect faces from an in-memory image array."""
        return self._detect_faces(image)

    def _detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect faces in an image using Haar Cascade.

        Args:
            image: Input image in BGR format.

        Returns:
            List of detected face images as BGR numpy arrays.
        """
        if image is None or image.size == 0:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use optimized parameters for better detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # More sensitive (lower = slower but more thorough)
            minNeighbors=4,   # Less strict (lower = more detections but more false positives)
            minSize=(20, 20)  # Allow smaller faces
        )
        
        face_images = []
        for x, y, w, h in faces:
            # Add padding for better face region
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            face_bgr = image[y1:y2, x1:x2].copy()
            if face_bgr.size > 0:
                """ it mean croped face is not empty"""
                face_images.append(face_bgr)
        return face_images
    """ return list of croped image"""

    def extract_face_roi(self, image_path: str | None, face_bgr: np.ndarray) -> np.ndarray:
        """Extract face region of interest.

        Args:
            image_path: Path to image (unused, kept for compatibility)
            face_bgr: Face BGR image from _detect_faces

        Returns:
            Face ROI as numpy array.
        """
        try:
            if face_bgr is None or face_bgr.size == 0:
                """ image not passwd and image passed but no pixel in it"""
                return None
            return face_bgr
        except Exception as exc:
            logger.exception("Error extracting face ROI")
            return None

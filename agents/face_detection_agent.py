"""
Face Detection Agent - Detects faces in images using MTCNN
"""

import logging
import cv2
import numpy as np
from mtcnn import MTCNN

logger = logging.getLogger(__name__)


class FaceDetectionAgent:
    """Agent responsible for detecting faces in images"""
    
    def __init__(self):
        """Initialize the face detection agent with MTCNN detector"""
        self.detector = MTCNN()
        self.last_frame = None  # Store last webcam frame for ROI extraction
        logger.info("FaceDetectionAgent initialized")
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected faces with bounding boxes and confidence
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.detector.detect_faces(image_rgb)
            
            logger.info(f"Detected {len(faces)} face(s) in {image_path}")
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {e}", exc_info=True)
            return []
    
    def detect_from_path(self, image_path):
        """Detect faces from an image path (alias for detect_faces)"""
        return self.detect_faces(image_path)
    
    def detect_from_webcam(self):
        """
        Detect faces from webcam
        
        Returns:
            List of detected faces from webcam feed
        """
        try:
            # Try multiple camera indices
            for camera_idx in range(3):
                cap = cv2.VideoCapture(camera_idx)
                if cap.isOpened():
                    faces = []
                    frame = None
                    # Warm up and try multiple frames until a face is detected
                    for _ in range(12):
                        ret, frame = cap.read()
                        if not ret or frame is None or frame.shape[0] == 0:
                            continue

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        faces = self.detector.detect_faces(frame_rgb)
                        if faces:
                            break

                    cap.release()

                    if frame is None or frame.shape[0] == 0:
                        continue

                    # Store frame for later ROI extraction
                    self.last_frame = frame

                    logger.info(
                        f"Detected {len(faces)} face(s) from webcam (camera {camera_idx})"
                    )
                    return faces
            
            # If we get here, no camera worked
            logger.error("No available camera found")
            print("Webcam Error: No camera device found. Please check your camera connection.")
            return []
            
        except Exception as e:
            logger.error(f"Error detecting faces from webcam: {e}", exc_info=True)
            print(f"Webcam Error: {str(e)}")
            return []
    
    def extract_face_roi(self, image_path, face_detection):
        """
        Extract face region of interest from image
        
        Args:
            image_path: Path to the image file (None for webcam)
            face_detection: Face detection dictionary from MTCNN
            
        Returns:
            Face ROI as numpy array
        """
        try:
            # Use webcam frame if image_path is None
            if image_path is None:
                if self.last_frame is None:
                    logger.error("No webcam frame available")
                    return None
                image = self.last_frame
            else:
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return None
            
            # Extract bounding box
            box = face_detection['box']
            x, y, w, h = box
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            x_end = min(image.shape[1], x + w)
            y_end = min(image.shape[0], y + h)
            
            # Extract ROI
            face_roi = image[y:y_end, x:x_end]
            
            return face_roi
            
        except Exception as e:
            logger.error(f"Error extracting face ROI: {e}", exc_info=True)
            return None

"""Reporting agent for generating detection reports."""

from __future__ import annotations

import json
""" used to save report in the json file 
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
"""used to for face embding vector"""

logger = logging.getLogger(__name__)


class ReportingAgent:
    """Generates and saves JSON reports for face detection matches."""

    def __init__(self, reports_dir: str = "reports") -> None:
        """Initialize the reporting agent.
        
        Args:
            reports_dir: Directory path where reports will be saved.
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True, parents=True)
        """" create a folder if not exit"""

    def generate_report(
        self,
        face_path: str,
        detected_embedding: np.ndarray,
        
        comparison_result: Dict[str, Any],
        face_index: int,
    ) -> str:
        """ face image yak face vector """
        """Generate and save a detection report as JSON.
        
        Args:
            face_path: Path to the detected face image.
            detected_embedding: Face embedding as numpy array.
            comparison_result: Dictionary with comparison results containing:
                - score: Similarity score
                - name: Name of matched person (if any)
            face_index: Index of the detected face.
            
        Returns:
            Path to the saved report file.
        """
        try:
            timestamp = datetime.now().isoformat()
            report_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            """ create unique report id """
            
            report_data = {
                "timestamp": timestamp,
                "report_id": report_id,
                "face_index": face_index,
                "face_path": str(face_path),
                "detected_embedding": self._serialize_embedding(detected_embedding),
                "comparison_result": {
                    "score": float(comparison_result.get("score", 0.0)),
                    "name": comparison_result.get("name", "Unknown"),
                    "matched": comparison_result.get("score", 0.0) > 0.5,
                },
            }
            
            # Generate unique filename
            report_filename = f"report_{report_id}.json"
            report_path = self.reports_dir / report_filename
            
            # Save report
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Report saved to {report_path}")
            return str(report_path)
            
        except Exception as exc:
            logger.exception(f"Error generating report for face index {face_index}")
            return ""

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> Optional[list]:
        """Convert numpy embedding to serializable list.
        
        Args:
            embedding: Numpy array embedding.
            
        Returns:
            List representation of embedding or None if invalid.
        """
        try:
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            return None
        except Exception as exc:
            logger.exception("Error serializing embedding")
            return None

    def get_reports(self) -> list[str]:
        """Get list of all saved report files.
        
        Returns:
            List of report file paths.
        """
        try:
            reports = sorted(self.reports_dir.glob("report_*.json"))
            return [str(r) for r in reports]
        except Exception as exc:
            logger.exception("Error retrieving reports")
            return []

    def load_report(self, report_path: str) -> Optional[Dict[str, Any]]:
        """Load a report from file.
        
        Args:
            report_path: Path to the report JSON file.
            
        Returns:
            Report data dictionary or None if error.
        """
        try:
            with open(report_path, "r") as f:
                return json.load(f)
        except Exception as exc:
            logger.exception(f"Error loading report from {report_path}")
            return None

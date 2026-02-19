"""Main controller for the Wanted Detection Multi-Agent System."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import cv2

from agents.face_detection_agent import FaceDetectionAgent
from agents.face_embedding_agent import FaceEmbeddingAgent
from agents.wanted_comparison_agent import WantedComparisonAgent
from agents.email_alert_agent import EmailAlertAgent
from agents.reporting_agent import ReportingAgent


def _save_webcam_frame(face_agent: FaceDetectionAgent, reports_dir: Path) -> str:
    """Save the webcam frame for later use"""
    if face_agent.last_frame is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"webcam_frame_{timestamp}.jpg"
    output_path = reports_dir / filename
    cv2.imwrite(str(output_path), face_agent.last_frame)
    return str(output_path)


def _save_face_image(face_bgr, reports_dir: Path, index: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"face_{timestamp}_{index}.jpg"
    output_path = reports_dir / filename
    cv2.imwrite(str(output_path), face_bgr)
    return str(output_path)


def _load_faces_from_input(face_agent: FaceDetectionAgent, image_path: str | None, webcam: bool) -> List:
    if image_path:
        return face_agent.detect_from_path(image_path)
    if webcam:
        return face_agent.detect_from_webcam()
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Wanted Detection Multi-Agent System")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam input")
    parser.add_argument("--threshold", type=float, default=None, help="Match threshold")
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.error("Provide --image or --webcam input")

    # Agent 1: Face Detection
    face_agent = FaceDetectionAgent()
    print("[Agent1] FaceDetectionAgent - Detected faces")

    # Setup directories and agents
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # For webcam, save the frame
    if args.webcam:
        webcam_frame_path = _save_webcam_frame(face_agent, reports_dir)
        image_path_to_use = webcam_frame_path
    else:
        image_path_to_use = args.image

    # Agent 2: Face Embedding
    embedding_agent = FaceEmbeddingAgent()
    print("[Agent2] FaceEmbeddingAgent - Extracted embeddings")

    if args.threshold is None:
        threshold = 0.2 if args.webcam else 0.5
    else:
        threshold = args.threshold

    comparison = WantedComparisonAgent(
        embedding_agent,
        database_dir="wanted_database",
        threshold=threshold,
    )
    reporting_agent = ReportingAgent(reports_dir=str(reports_dir))
    
    try:
        email_agent = EmailAlertAgent()
        email_configured = True
    except ValueError:
        email_agent = None
        email_configured = False

    # Process detected faces
    best_match = None
    best_score = 0.0
    best_face_path = None

    if args.webcam:
        max_attempts = 5
        for _ in range(max_attempts):
            faces = face_agent.detect_from_webcam()
            if not faces:
                continue

            for idx, face_data in enumerate(faces, 1):
                face_bgr = face_agent.extract_face_roi(None, face_data)
                if face_bgr is None:
                    continue

                face_path = _save_face_image(face_bgr, reports_dir, idx)
                embedding = embedding_agent.get_embedding(face_bgr)
                result = comparison.compare(embedding)

                if result["score"] > best_score:
                    best_match = result["name"]
                    best_score = result["score"]
                    best_face_path = face_path

                reporting_agent.generate_report(
                    face_path=face_path,
                    detected_embedding=embedding,
                    comparison_result=result,
                    face_index=idx,
                )

            if best_score >= threshold:
                break
    else:
        faces = face_agent.detect_from_path(args.image)
        if not faces:
            print("No faces detected in input")
            return

        for idx, face_data in enumerate(faces, 1):
            face_bgr = face_agent.extract_face_roi(image_path_to_use, face_data)
            if face_bgr is None:
                continue

            face_path = _save_face_image(face_bgr, reports_dir, idx)
            embedding = embedding_agent.get_embedding(face_bgr)
            result = comparison.compare(embedding)

            if result["score"] > best_score:
                best_match = result["name"]
                best_score = result["score"]
                best_face_path = face_path

            reporting_agent.generate_report(
                face_path=face_path,
                detected_embedding=embedding,
                comparison_result=result,
                face_index=idx,
            )

    # Agent 3: Report best match
    if best_match and best_score >= threshold:
        print(f"[Agent3] WantedComparisonAgent - Found match ({best_match}, score={best_score:.3f})")
    else:
        print(f"[Agent3] WantedComparisonAgent - No match found")

    # Agent 4: Email Alert
    if best_match and best_score >= threshold:
        if email_configured:
            try:
                email_agent.send_alert(
                    image_path=best_face_path,
                    score=best_score,
                    matched_name=best_match,
                )
                print("[Agent4] EmailAlertAgent - Sent alert email")
            except Exception as e:
                print(f"[Agent4] EmailAlertAgent - Failed: {e}")
        else:
            print("[Agent4] EmailAlertAgent - Email not configured")
    else:
        print("[Agent4] EmailAlertAgent - Skipped (no match)")

    # Agent 5: Report saved
    print("[Agent5] ReportingAgent - Saved JSON report")


if __name__ == "__main__":
    main()

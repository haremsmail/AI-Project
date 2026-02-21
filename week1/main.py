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

    # Setup directories and agents
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Agent 1: Face Detection
    print("[Agent1] FaceDetectionAgent - Initializing...")
    face_agent = FaceDetectionAgent()
    
    # Detect faces
    if args.image:
        print(f"[Agent1] Loading image: {args.image}")
        faces = face_agent.detect_from_path(args.image)
    else:
        print("[Agent1] Capturing from webcam...")
        faces = face_agent.detect_from_webcam()
    
    if not faces:
        print("[Agent1] ERROR: No faces detected in input")
        return
    
    print(f"[Agent1] FaceDetectionAgent - Detected {len(faces)} face(s)")

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
    
    print(f"[Agent3] WantedComparisonAgent - Loaded {len(comparison.database)} wanted persons")
    
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
    all_results = []

    for idx, face_bgr in enumerate(faces, 1):
        # Face image is already cropped from detect_from_path/detect_from_webcam
        face_path = _save_face_image(face_bgr, reports_dir, idx)
        
        print(f"\n[Processing] Face {idx}/{len(faces)}")
        print(f"  Saved to: {face_path}")
        
        # Get embedding
        try:
            embedding = embedding_agent.get_embedding(face_bgr)
            print(f"  Embedding computed (size: {len(embedding)})")
        except Exception as e:
            print(f"  ERROR computing embedding: {e}")
            continue
        
        # Compare with wanted database
        result = comparison.compare(embedding)
        all_results.append((face_path, result))
        
        print(f"  Comparison score: {result['score']:.4f}")
        print(f"  Match: {result['name'] if result['match'] else 'NO MATCH'}")
        
        if result["score"] > best_score:
            best_match = result["name"]
            best_score = result["score"]
            best_face_path = face_path

        # Generate report
        reporting_agent.generate_report(
            face_path=face_path,
            detected_embedding=embedding,
            comparison_result=result,
            face_index=idx,
        )

    # Agent 3: Report best match
    print(f"\n[Agent3] WantedComparisonAgent - Comparison Results")
    print(f"{'='*60}")
    for face_path, result in all_results:
        status = "MATCH" if result["match"] else "NO MATCH"
        print(f"Face: {Path(face_path).name}")
        print(f"  Score: {result['score']:.4f} (threshold: {threshold})")
        print(f"  Status: {status}")
        if result["name"]:
            print(f"  Matched: {result['name']}")
        print()
    
    if best_match and best_score >= threshold:
        print(f"[Agent3] ALERT: Found wanted person '{best_match}' with score {best_score:.4f}")
    else:
        print(f"[Agent3] No wanted person found (best score: {best_score:.4f})")

    # Agent 4: Email Alert
    if best_match and best_score >= threshold:
        if email_configured:
            try:
                email_agent.send_alert(
                    image_path=best_face_path,
                    score=best_score,
                    matched_name=best_match,
                )
                print(f"[Agent4] EmailAlertAgent - Sent alert email for {best_match}")
            except Exception as e:
                print(f"[Agent4] EmailAlertAgent - Failed: {e}")
        else:
            print("[Agent4] EmailAlertAgent - Email not configured (skipped)")
    else:
        print("[Agent4] EmailAlertAgent - Skipped (no match above threshold)")

    # Agent 5: Report saved
    print(f"\n[Agent5] ReportingAgent - Saved {len(all_results)} JSON report(s)")
    reports = reporting_agent.get_reports()
    print(f"[Agent5] Total reports in directory: {len(reports)}")


if __name__ == "__main__":
    main()

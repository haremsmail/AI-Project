"""Main controller for the Wanted Detection Multi-Agent System."""

from __future__ import annotations

import argparse
"""" use to another file"""
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
        """ No frame captured from webcam yet   automatialy save it """
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"webcam_frame_{timestamp}.jpg"
    """ this is used to create a file name"""
    output_path = reports_dir / filename
    """ create full path of the file to save the webcam frame in the reports directory"""
    cv2.imwrite(str(output_path), face_agent.last_frame)
    """ save the last captured frame from the webcam to the specified path using OpenCV's imwrite function. The frame is saved in JPEG format with a unique timestamped filename. """
    return str(output_path)
""" this function is used to save the webcam frame for later use. It checks if a frame has been captured from the webcam (stored in face_agent.last_frame). If a frame is available,
 it generates a unique filename based on the current timestamp, constructs the full path to save the image in the reports directory, and saves the frame as a JPEG image using OpenCV's imwrite function. """
""" image fully save in the reports folder"""

def _save_face_image(face_bgr, reports_dir: Path, index: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"face_{timestamp}_{index}.jpg"
    output_path = reports_dir / filename
    cv2.imwrite(str(output_path), face_bgr)
    return str(output_path)


def _load_faces_from_input(face_agent: FaceDetectionAgent, image_path: str | None, webcam: bool) -> List:
    """" this function is used to load face from which one input image or webcam if the image path is provided, it will use the face detection agent to detect faces from the image and return them as a list of BGR numpy arrays. If the webcam flag is set, it will capture frames from the webcam and detect faces until a timeout occurs, returning the detected faces as a list. If neither input is provided, it returns an empty list."""
    if image_path:
        return face_agent.detect_from_path(image_path)
    if webcam:
        return face_agent.detect_from_webcam()
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Wanted Detection Multi-Agent System")
    """ this is used to parse command-line arguments for the script. It allows users to specify an input image path, whether to use webcam input, and an optional threshold for face matching. The parsed arguments are then used to control the flow of the program, determining how faces are detected and processed."""
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam input")
    parser.add_argument("--threshold", type=float, default=None, help="Match threshold")
    args = parser.parse_args()
    """ read and haw run and what user wanto"""

    if not args.image and not args.webcam:
        parser.error("Provide --image or --webcam input")

    # Setup directories and agents
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    """ create a directory for saving reports if it doesn't exist already. This ensures that the program has a designated location to store generated JSON reports and any saved images from the face detection process. The exist_ok=True parameter allows the directory creation to succeed even if the directory already exists, preventing errors in subsequent runs of the program."""

    # Agent 1: Face Detection
    print("[Agent1] FaceDetectionAgent - Initializing...")
    face_agent = FaceDetectionAgent()
    
    # Detect faces
    if args.image:
        print(f"[Agent1] Loading image: {args.image}")
        faces = face_agent.detect_from_path(args.image)
        """ the args.image is the path of the image that user want to detect face from it and then we use the face detection agent to detect faces from the image and return them as a list of BGR numpy arrays."""
    else:
        print("[Agent1] Capturing from webcam...")
        faces = face_agent.detect_from_webcam()
    
    if not faces:
        print("[Agent1] ERROR: No faces detected in input")
        return
    
    print(f"[Agent1] FaceDetectionAgent - Detected {len(faces)} face(s)")
    """ used to found length of the face"""

    # Agent 2: Face Embedding
    embedding_agent = FaceEmbeddingAgent()
    print("[Agent2] FaceEmbeddingAgent - Extracted embeddings")

    if args.threshold is None:
        threshold = 0.2 if args.webcam else 0.5
        """ this is simiarity by default if the input is from webcam we will use 0.2 and if the input is from image we will use 0.5 but user can change it by using --threshold argument"""
    else:
        threshold = args.threshold
        """ if have result find the result threshold value set by the user"""

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
        """ process the image one by one """
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

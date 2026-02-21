"""Test WantedComparisonAgent."""

from agents.face_detection_agent import FaceDetectionAgent
from agents.face_embedding_agent import FaceEmbeddingAgent
from agents.wanted_comparison_agent import WantedComparisonAgent

print("[Test] Loading agents...")
d = FaceDetectionAgent()
e = FaceEmbeddingAgent()
c = WantedComparisonAgent(e, 'wanted_database', threshold=0.7)

print("[Test] Detecting faces in test1.jpg...")
faces = d.detect_from_path('test1.jpg')
print(f"[Test] Detected {len(faces)} face(s)")

if faces:
    print("[Test] Extracting face ROI...")
    face_bgr = d.extract_face_roi('test1.jpg', faces[0])
    
    print("[Test] Extracting embedding...")
    emb = e.get_embedding(face_bgr)
    print(f"[Test] Embedding shape: {emb.shape}")
    
    print("[Test] Comparing against wanted database...")
    result = c.compare(emb)
    print(f"[Test] Match: {result['match']}, Score: {result['score']:.3f}")
    if result['name']:
        print(f"[Test] Matched name: {result['name']}")
else:
    print("[Test] No faces detected")

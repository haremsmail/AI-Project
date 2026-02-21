"""Diagnostic test to see actual similarity score."""

from agents.face_detection_agent import FaceDetectionAgent
from agents.face_embedding_agent import FaceEmbeddingAgent
from agents.wanted_comparison_agent import WantedComparisonAgent

print("[Diagnostic] Loading agents...")
d = FaceDetectionAgent()
e = FaceEmbeddingAgent()
c = WantedComparisonAgent(e, 'wanted_database', threshold=0.5)  # Lowered threshold

print("[Diagnostic] Detecting faces in test1.jpg...")
faces = d.detect_from_path('test1.jpg')

if faces:
    print("[Diagnostic] Extracting face ROI for first face...")
    face_bgr = d.extract_face_roi('test1.jpg', faces[0])
    
    emb_test = e.get_embedding(face_bgr)
    print(f"\n[Diagnostic] Database contents:")
    for label, db_emb, path in c.database:
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = float(cosine_similarity(emb_test.reshape(1, -1), db_emb.reshape(1, -1))[0][0])
        print(f"  {label}: similarity = {similarity:.4f}")
    
    result = c.compare(emb_test)
    print(f"\n[Result] Match: {result['match']}, Best score: {result['score']:.4f}")
    if result['name']:
        print(f"[Result] Matched: {result['name']}")
else:
    print("No faces detected")

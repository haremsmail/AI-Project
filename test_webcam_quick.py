"""Quick test for webcam face detection."""

import cv2

# Test if webcam can be opened
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    exit(1)

print("✓ Webcam opened successfully")

# Test face cascade loading
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: Failed to load Haar Cascade!")
    exit(1)

print("✓ Haar Cascade loaded successfully")

# Capture one frame and test detection
ret, frame = cap.read()
if ret:
    print(f"✓ Frame captured: {frame.shape}")
    
    # Test detection with improved parameters
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    print(f"✓ Detection test complete: {len(faces)} face(s) detected")
    
    if len(faces) > 0:
        print("✓✓ WEBCAM FACE DETECTION IS WORKING! ✓✓")
    else:
        print("ℹ No faces detected in first frame (this is normal)")
        print("  Make sure your face is visible and well-lit when testing")
else:
    print("ERROR: Failed to read frame")

cap.release()
print("\n✓ Test complete!")

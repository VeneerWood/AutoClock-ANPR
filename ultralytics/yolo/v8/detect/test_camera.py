import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Read one frame
ret, frame = cap.read()
if ret:
    print("Camera is working!")
    cv2.imwrite('test_frame.jpg', frame)
    print("Saved test frame to test_frame.jpg")
else:
    print("Couldn't read from camera")

cap.release()

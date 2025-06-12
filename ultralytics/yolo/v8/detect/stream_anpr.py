from ultralytics import YOLO
import cv2
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
import easyocr
import os
from flask import Flask, Response, render_template
import threading
import numpy as np
import traceback

# Initialize Flask app
app = Flask(__name__)

# Create a global frame variable that will be updated in the detection thread
global_frame = None
processing_frame = False

# Initialize Firebase (only if it's not already initialized)
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("autoclock-sriipuj-firebase.json")
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://autoclock-sriipuj-default-rtdb.asia-southeast1.firebasedatabase.app"
        })
        print("Firebase initialized successfully")
    except Exception as e:
        print(f"Firebase initialization error: {e}")
        exit()

# Function to check and store the plate number in Firebase
def check_and_store_plate_number(captured_plate_number):
    if not captured_plate_number:
        return
        
    staff_ref = db.reference('staff')
    staff_data = staff_ref.get()
    
    if not staff_data:
        print("No staff data found in database")
        return

    found_match = False
    for staff_key, staff in staff_data.items():
        if staff['plate_number'] == captured_plate_number:
            print(f"Plate number {captured_plate_number} matches staff {staff['name']}.")
            clock_in_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            staff_ref.child(staff_key).update({
                "clock_in_time": clock_in_time
            })
            print(f"Clock-in time for {staff['name']} (Plate: {captured_plate_number}): {clock_in_time}")
            return  # Exit if a match is found
    
    print(f"No match found for plate number {captured_plate_number}.")

def generate_frames():
    while True:
        global global_frame
        if global_frame is not None:
            try:
                # Create a copy to avoid potential race conditions
                frame_copy = global_frame.copy()
                # Encode the frame to JPEG
                ret, jpeg = cv2.imencode('.jpg', frame_copy)
                if ret:
                    # Convert to bytes and yield for streaming
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error in generate_frames: {e}")
        time.sleep(0.1)  # Small delay to reduce CPU usage

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>ANPR Camera Stream</title>
        <style>
          body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
          h1 { color: #333366; }
          .container { margin: 0 auto; width: 80%; }
          img { max-width: 100%; border: 3px solid #333; border-radius: 5px; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>ANPR Camera Stream</h1>
          <img src="/video_feed" />
          <p>Live camera feed with ANPR detection</p>
        </div>
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def detection_thread():
    global global_frame, processing_frame
    
    # Initialize EasyOCR for license plate text recognition
    try:
        reader = easyocr.Reader(['en'])
        print("EasyOCR initialized successfully")
    except Exception as e:
        print(f"EasyOCR initialization error: {e}")
        return

    # Load your YOLO model
    print("Loading YOLO model...")
    model_path = "/home/afnan02/Desktop/fyp/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/train_model/weights/best.pt"
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Function to recognize text from license plate image
    def recognize_plate(plate_img):
        try:
            if plate_img is None or plate_img.size == 0:
                return None, 0
                
            results = reader.readtext(plate_img)
            if not results:  # Check if results list is empty
                return None, 0
                
            # Sort by confidence and take the best result
            results.sort(key=lambda x: x[2], reverse=True)
            return results[0][1], results[0][2]
        except Exception as e:
            print(f"Error in text recognition: {e}")
            return None, 0

    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)  # Using /dev/video1
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Variables for handling duplicate detections
    last_plate = None
    last_detection_time = 0
    detection_cooldown = 10
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        print("Starting detection loop...")
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame")
                time.sleep(0.1)  # Wait a bit before trying again
                continue
                
            # Update global frame for streaming (even if we're still processing)
            global_frame = frame.copy()
            
            # Skip processing if we're still working on the previous frame
            if processing_frame:
                time.sleep(0.01)  # Small delay to prevent CPU hogging
                continue
                
            processing_frame = True
            
            try:
                # Perform detection
                results = model(frame)
                current_time = time.time()
                
                # Process each detection result
                if results and len(results) > 0:
                    for result in results:
                        # Process boxes only if they exist
                        if result.boxes is not None and len(result.boxes) > 0:
                            # Get the boxes as a numpy array if available
                            boxes_data = result.boxes.data.cpu().numpy() if hasattr(result.boxes, 'data') else None
                            
                            if boxes_data is not None and len(boxes_data) > 0:
                                for box in boxes_data:
                                    # Extract coordinates from box data
                                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                                    conf = float(box[4])
                                    
                                    # Only process high confidence detections
                                    if conf < 0.3:  # Lowered threshold for testing
                                        continue
                                        
                                    # Make sure coordinates are within frame boundaries
                                    x1 = max(0, x1)
                                    y1 = max(0, y1)
                                    x2 = min(frame.shape[1], x2)
                                    y2 = min(frame.shape[0], y2)
                                    
                                    # Draw bounding box on the frame
                                    cv2.rectangle(global_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Add confidence text
                                    cv2.putText(global_frame, f"Conf: {conf:.2f}", (x1, y1-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    # Crop the license plate from the frame
                                    try:
                                        if y2 > y1 and x2 > x1:  # Valid dimensions
                                            plate_img = frame[y1:y2, x1:x2]
                                            
                                            # Skip if plate image is too small
                                            if plate_img.size == 0 or plate_img.shape[0] < 20 or plate_img.shape[1] < 20:
                                                continue
                                            
                                            # Recognize text on the plate
                                            plate_text, text_conf = recognize_plate(plate_img)
                                            
                                            if plate_text and text_conf > 0.2:  # Lowered threshold for testing
                                                # Clean up the plate text
                                                plate_text = plate_text.upper().replace(' ', '')
                                                
                                                # Add text to the frame
                                                cv2.putText(global_frame, f"{plate_text}", (x1, y2+20), 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
                                                
                                                # Add text confidence
                                                cv2.putText(global_frame, f"Text conf: {text_conf:.2f}", (x1, y2+40), 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                                                
                                                # Check if this is a new plate or enough time has passed
                                                if plate_text != last_plate or (current_time - last_detection_time) > detection_cooldown:
                                                    last_plate = plate_text
                                                    last_detection_time = current_time
                                                    
                                                    print(f"Detected plate: {plate_text}, Confidence: {conf:.2f}, Text confidence: {text_conf:.2f}")
                                                    
                                                    # Check if the plate belongs to a staff member and update clock-in time
                                                    check_and_store_plate_number(plate_text)
                                                    
                                                    # Optional: Save detection image
                                                    detection_dir = "detections"
                                                    if not os.path.exists(detection_dir):
                                                        os.makedirs(detection_dir)
                                                    cv2.imwrite(f"{detection_dir}/plate_{int(current_time)}_{plate_text}.jpg", plate_img)
                                    except Exception as e:
                                        print(f"Error processing plate image: {e}")
                                        continue
            except Exception as e:
                print(f"Error in frame processing: {e}")
                traceback.print_exc()  # Print the full traceback for debugging
            
            processing_frame = False
            time.sleep(0.01)  # Small delay to prevent CPU hogging
                    
    except Exception as e:
        print(f"Error in detection loop: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        print("Camera released")

if __name__ == '__main__':
    # Start the detection thread
    threading.Thread(target=detection_thread, daemon=True).start()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=8080, debug=False)

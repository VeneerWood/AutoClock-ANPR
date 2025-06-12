from ultralytics import YOLO
import cv2
import time
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import easyocr
import os

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

# Initialize EasyOCR for license plate text recognition
try:
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized successfully")
except Exception as e:
    print(f"EasyOCR initialization error: {e}")
    exit()

# Load your YOLO model
print("Loading YOLO model...")
model_path = "/home/afnan02/Desktop/fyp/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/train_model/weights/best.pt"
try:
    model = YOLO(model_path)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Function to recognize text from license plate image
def recognize_plate(plate_img):
    try:
        # Read text from plate image
        results = reader.readtext(plate_img)
        
        # Extract the most likely text
        if results:
            # Sort by confidence and get the highest confidence text
            results.sort(key=lambda x: x[2], reverse=True)
            return results[0][1], results[0][2]  # Return text and confidence
        return None, 0
    except Exception as e:
        print(f"Error in text recognition: {e}")
        return None, 0

# Open camera
print("Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Variables for handling duplicate detections
last_plate = None
last_detection_time = 0
detection_cooldown = 10  # seconds to wait before logging the same plate again

try:
    print("Starting detection loop...")
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Perform detection
        results = model(frame)
        
        current_time = time.time()
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Only process high confidence detections
                if conf < 0.5:  # Adjust threshold as needed
                    continue
                    
                # Crop the license plate from the frame
                plate_img = frame[y1:y2, x1:x2]
                
                # Skip if plate image is too small
                if plate_img.size == 0 or plate_img.shape[0] < 20 or plate_img.shape[1] < 20:
                    continue
                
                # Recognize text on the plate
                plate_text, text_conf = recognize_plate(plate_img)
                
                if plate_text and text_conf > 0.3:  # Only process plates with decent confidence
                    # Clean up the plate text (remove spaces, standardize)
                    plate_text = plate_text.upper().replace(' ', '')
                    
                    # Check if this is a new plate or enough time has passed
                    if plate_text != last_plate or (current_time - last_detection_time) > detection_cooldown:
                        last_plate = plate_text
                        last_detection_time = current_time
                        
                        # Print detection information
                        print(f"Detected plate: {plate_text}, Confidence: {conf:.2f}, Text confidence: {text_conf:.2f}")
                        
                        # Check if the plate belongs to a staff member and update clock-in time
                        check_and_store_plate_number(plate_text)
                        
                        # Optional: Save detection image
                        detection_dir = "detections"
                        if not os.path.exists(detection_dir):
                            os.makedirs(detection_dir)
                        cv2.imwrite(f"{detection_dir}/plate_{int(current_time)}_{plate_text}.jpg", plate_img)
                
        # Prevent CPU overload
        time.sleep(0.1)
            
except KeyboardInterrupt:
    print("Stopping detection...")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    cap.release()
    print("Camera released. Program terminated.")

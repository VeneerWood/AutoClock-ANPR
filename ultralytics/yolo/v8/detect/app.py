from flask import Flask, Response
import cv2
import torch
from ultralytics import YOLO
import easyocr
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import numpy as np

# === Init Flask App ===
app = Flask(__name__)

# === Load YOLOv8 model ===
model = YOLO('yolov8n.pt')

# === Init EasyOCR ===
reader = easyocr.Reader(['en'], gpu=True)

# === Firebase setup ===
if not firebase_admin._apps:
    cred = credentials.Certificate("autoclock-sriipuj-firebase.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://autoclock-sriipuj-default-rtdb.asia-southeast1.firebasedatabase.app"
    })

def check_and_store_plate_number(plate):
    staff_ref = db.reference('staff')
    staff_data = staff_ref.get()

    for staff_key, staff in staff_data.items():
        if staff['plate_number'] == plate:
            print(f"[MATCH] {plate} => {staff['name']}")
            clock_in_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            staff_ref.child(staff_key).update({
                "clock_in_time": clock_in_time
            })
            print(f"[FIREBASE] Clock-in updated for {staff['name']}: {clock_in_time}")
            return

    print(f"[NO MATCH] Plate {plate} not found in staff list.")

def detect_and_stream():
    cap = cv2.VideoCapture(0)  # use USB webcam

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            cropped = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (200, 50), interpolation=cv2.INTER_LINEAR)

            ocr_result = reader.readtext(resized)
            plate_text = ocr_result[0][1] if ocr_result else "N/A"

            if 'O' in plate_text:
                plate_text = plate_text.replace('O', 'Q')

            plate_clean = plate_text.upper().replace(" ", "")
            label = f'{plate_clean} ({conf:.2f})'

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if plate_clean != "N/A":
                check_and_store_plate_number(plate_clean)

        # Encode and yield MJPEG frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    cap.release()
    #cv2.destroyAllWindows()

@app.route('/video')
def video_feed():
    return Response(detect_and_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h2>AutoClock Live Detection Stream</h2><img src="/video" width="640" height="480">'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

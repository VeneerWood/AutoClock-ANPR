import firebase_admin
from firebase_admin import credentials, db
import subprocess
import time
import os

# === Firebase setup ===
cred_path = "/home/afnan02/Desktop/fyp/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/autoclock-sriipuj-firebase.json"
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://autoclock-sriipuj-default-rtdb.asia-southeast1.firebasedatabase.app"
})

# === Variables
predict_process = None
predict_script = "/home/afnan02/Desktop/fyp/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/predict.py"

# === Start loop
while True:
    try:
        ref = db.reference("system_control/prediction_status")
        status = ref.get()

        if status and predict_process is None:
            print("🟢 Prediction status ON — starting predict.py...")
            predict_process = subprocess.Popen([
    		"/home/afnan02/Desktop/fyp/Automatic_Number_Plate_Detection_Recognition_YOLOv8/venv/bin/python3",
    		predict_script
	    ])


        elif not status and predict_process is not None:
            print("🛑 Prediction status OFF — stopping predict.py...")
            predict_process.terminate()
            predict_process.wait()
            predict_process = None

        time.sleep(3)

    except Exception as e:
        print("[ERROR]", e)
        time.sleep(5)

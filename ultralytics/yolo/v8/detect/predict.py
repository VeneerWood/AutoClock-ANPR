import hydra
import torch
from datetime import datetime
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import easyocr
import cv2
import re
import firebase_admin
from firebase_admin import credentials, db

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("/home/afnan02/Desktop/fyp/Automatic_Number_Plate_Detection_Recognition_YOLOv8/ultralytics/yolo/v8/detect/autoclock-sriipuj-firebase.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://autoclock-sriipuj-default-rtdb.asia-southeast1.firebasedatabase.app"
    })

# Reference to detector_ready status in Firebase
ready_ref = db.reference("system_control/detector_ready")

# Function to check and store the plate number in Firebase
def check_and_store_plate_number(captured_plate_number):
    staff_ref = db.reference('staff')
    staff_data = staff_ref.get()

    for staff_key, staff in staff_data.items():
        if staff['plate_number'] == captured_plate_number:
            print(f"Plate number {captured_plate_number} matches staff {staff['name']}.")
            clock_in_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            staff_ref.child(staff_key).update({
                "clock_in_time": clock_in_time
            })
            print(f"Clock-in time for {staff['name']} (Plate: {captured_plate_number}): {clock_in_time}")
            return
    print(f"No match found for plate number {captured_plate_number}.")

# OCR function
def ocr_image(img, coordinates):
    x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
    cropped_img = img[y:h, x:w]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 50), interpolation=cv2.INTER_LINEAR)

    result = reader.readtext(resized)
    text = result[0][1] if result else ""

    if 'O' in text:
        text = text.replace('O', 'Q')

    return text if text else "N/A"

# YOLO Detection class
class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1

        # ✅ Mark detector as ready — only once
        if not hasattr(self, 'ready_fired'):
            ready_ref.set(True)
            self.ready_fired = True

        im0 = im0.copy()

        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)

        if len(det) == 0:
            return log_string

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
        log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:
                c = int(cls)
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                text_ocr = ocr_image(im0, xyxy)
                label = text_ocr if text_ocr != "N/A" else f'{self.model.names[c]} {conf:.2f}'
                self.annotator.box_label(xyxy, label, color=colors(c, True))

            if text_ocr != "N/A":
                text_ocr_upper = text_ocr.upper()
                text_ocr_no_spaces = text_ocr_upper.replace(" ", "")
                print(f"Detected Plate Number (No Spaces, Uppercase): {text_ocr_no_spaces}")
                check_and_store_plate_number(text_ocr_no_spaces)

            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                            imc,
                            file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                            BGR=True)

        return log_string

# Hydra entry point
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else 0

    ready_ref.set(False)  # Indicate loading started

    predictor = DetectionPredictor(cfg)
    predictor()  # Will trigger write_results internally

# Main block with graceful shutdown
if __name__ == "__main__":
    try:
        predict()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Cleaning up...")
    finally:
        ready_ref.set(False)  # Reset on shutdown
        print("[INFO] Resources released and OpenCV windows closed.")

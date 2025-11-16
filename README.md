# Intelligent-License-Plate-Reader
Intelligent License Plate Reader &amp; Vehicle Insights Dashboard

# 🚗 ALPR (Automatic License Plate Recognition)

## ⭐ What This Project Does

This project detects **vehicle number plates** using a trained YOLO model and then reads the **text on the plate** using EasyOCR.

👉 YOLO finds the plate in the image
👉 EasyOCR reads the characters
👉 You get the cleaned plate number as output

---

## 🧰 Tech Used

* **YOLOv8** – for detecting the plate
* **EasyOCR** – for reading the text
* **OpenCV** – for image processing
* **Python** – the main language

---

## 📁 Project Flow (Simple)

1. Prepare and merge your dataset
2. Split into train/val folders
3. Create a `data.yaml` file for YOLO
4. Train YOLO on your custom dataset
5. Use the trained model + EasyOCR to read plate text

---

## 🛠 Install Requirements

```
pip install ultralytics easyocr opencv-python matplotlib numpy
```

---

## 🏋️ Training the Model

```
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data/processed/yolo/data.yaml",
    epochs=3,
    imgsz=640,
    batch=2,
    name="alpr_cpu_test",
    device="cpu"
)
```

Training will save your best model at:

```
runs/detect/alpr_cpu_test/weights/best.pt
```

---

## 🔍 Running ALPR (Detect + Read Text)

```
from ultralytics import YOLO
import easyocr, cv2

model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

img = cv2.imread("car.jpg")
results = model(img)

for r in results:
    for box in r.boxes.xyxy:
        x1,y1,x2,y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        ocr = reader.readtext(crop)
        print("Plate:", ocr)
```

---

## 📌 Example Output

```
Detected Plate: TN10AB1234
```

---

## 🎯 Summary

* YOLO detects the number plate
* EasyOCR reads the text
* Works well for Indian-style plates
* Simple, clean, and easy to extend


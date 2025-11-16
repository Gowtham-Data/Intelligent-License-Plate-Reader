import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import re
import pandas as pd
import io
from datetime import datetime

# ----------------------
# CONFIG
# ----------------------
MODEL_PATH = r"D:\vscode\runs\detect\alpr_cpu_test7\weights\best.pt"

PLATE_REGEX = re.compile(r'([A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4})', re.IGNORECASE)

def clean_text(txt):
    t = (txt or "").strip().upper()
    t = re.sub(r'[^A-Z0-9]', '', t)
    t = t.replace("O","0")
    return t

# ----------------------
# LOAD MODELS
# ----------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

model = load_model()
reader = load_reader()

# ----------------------
# STREAMLIT UI
# ----------------------
st.title("🚘 ALPR — YOLO + EasyOCR")

uploaded = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])

if uploaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running YOLO + OCR..."):

            results = model(img)
            annotated = img.copy()
            rows = []

            # ---- IMPORTANT CHANGE STARTS HERE ----
            # assume one image -> results[0]
            r = results[0]

            # boxes & confidence as numpy
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else []
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else []

            if len(boxes) == 0:
                st.warning("No plates detected. Try another image or lower confidence.")
            else:
                # best confidence box index
                best_i = int(confs.argmax())
                best_box = boxes[best_i]
                x1, y1, x2, y2 = map(int, best_box)

                # crop
                crop = img[y1:y2, x1:x2]

                # OCR
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ocr_res = reader.readtext(gray)

                raw = " ".join([o[1] for o in ocr_res]) if ocr_res else ""
                cleaned = clean_text(raw)

                # draw ONLY this one box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(annotated, cleaned, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                rows.append({
                    "timestamp": datetime.now().isoformat(timespec='seconds'),
                    "plate_raw": raw,
                    "plate_clean": cleaned
                })
            # ---- IMPORTANT CHANGE ENDS HERE ----

            st.subheader("Annotated Output")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df)

                # CSV Download
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button("Download Results CSV",
                    data=csv_buf.getvalue(),
                    file_name="alpr_results.csv",
                    mime="text/csv"
                )

        st.success("Done ✔")

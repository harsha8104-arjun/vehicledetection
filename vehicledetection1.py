import streamlit as st
from ultralytics import YOLO
import cv2
import easyocr
import tempfile
import os
import re
import shutil

st.title("License Plate Partial Match Search")

model = YOLO(r"C:\harsha\projects\vehicledetection\best.pt")
reader = easyocr.Reader(['en'])
option = st.radio("Choose Input Type", ["Video", "Image Folder"])
target_plate = st.text_input("Enter license plate number:").upper().replace(" ", "")

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def partial_match(text, target, min_match=3):
    text = clean_text(text)
    target = clean_text(target)
    for i in range(len(target) - min_match + 1):
        sub = target[i:i+min_match]
        if sub in text:
            return True
    return False

def process_frame(frame, target):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            pad = 10
            y1 = max(0, y1 - pad)
            x1 = max(0, x1 - pad)
            y2 = min(frame.shape[0], y2 + pad)
            x2 = min(frame.shape[1], x2 + pad)
            crop = frame[y1:y2, x1:x2]
            ocr = reader.readtext(crop)
            for res in ocr:
                if partial_match(res[1], target):
                    return True
    return False

matched_folder = os.path.join(tempfile.gettempdir(), "matched_results")
os.makedirs(matched_folder, exist_ok=True)

if option == "Video":
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded_video and target_plate:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        path = tfile.name
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        matched = []
        i = 0
        pb = st.progress(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if process_frame(frame, target_plate):
                out_path = os.path.join(matched_folder, f"frame_{i}.jpg")
                cv2.imwrite(out_path, frame)
                matched.append(out_path)
            i += 1
            pb.progress(min(i/total, 1))
        cap.release()
        if matched:
            st.success(f"Found {len(matched)} matching frame(s)")
            st.image(matched, width=400)
            st.write(f"Matched frames saved in: {matched_folder}")
        else:
            st.warning("No matching plates found")

elif option == "Image Folder":
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files and target_plate:
        folder = tempfile.mkdtemp()
        matched = []
        for file in uploaded_files:
            img_path = os.path.join(folder, file.name)
            with open(img_path, "wb") as f:
                f.write(file.read())
        imgs = os.listdir(folder)
        for img in imgs:
            p = os.path.join(folder, img)
            frame = cv2.imread(p)
            if frame is not None and process_frame(frame, target_plate):
                out_path = os.path.join(matched_folder, img)
                shutil.copy(p, out_path)
                matched.append(out_path)
        if matched:
            st.success(f"Found {len(matched)} matching image(s)")
            st.image(matched, width=400)
            st.write(f"Matched images saved in: {matched_folder}")
        else:
            st.warning("No matching plates found")

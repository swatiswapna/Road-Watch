import torch
import easyocr
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

# Function to perform OCR on the image
def perform_ocr_on_image(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)
    text = ""
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
            text = res[1]
    return str(text)

# Function to detect helmet in the image
def helmet_detect(img):
    wearing_helmet = True
    helmet_model = YOLO("helmet.pt")
    helmet_pred = helmet_model(img)
    list_of_helmets = list(helmet_pred[0].boxes.cls.numpy())
    if 1.0 in list_of_helmets:
        wearing_helmet = False
    return wearing_helmet

# Function to detect and read the number plate from the image
def detect_read_numberplate(img):
    numberplate_model = YOLO("best.pt")
    detect_plate = numberplate_model(img, save_crop=True)
    coords = list(detect_plate[0].boxes.xyxy.numpy()[0])  # Assuming one number plate detected
    text = perform_ocr_on_image(img, coords)
    return text

# Streamlit App
def run():
    st.title('Helmet and Number Plate Detection')
    
    # Upload Image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = np.array(img)  # Convert image to numpy array for processing
        
        # Display the image
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        # Detect if the user is wearing a helmet
        wearing_helmet = helmet_detect(img)
        
        if not wearing_helmet:
            st.write("No helmet detected. Trying to read number plate...")

            # Detect and read the number plate
            plate_text = detect_read_numberplate(img)
            st.write("Detected Number Plate: ", plate_text)
        else:
            st.write("Helmet is on. No number plate detected.")

if __name__ == "__main__":
    run()



import torch
from ultralytics import YOLO
import easyocr
import cv2


reader = easyocr.Reader(['en'], gpu=True)


def perform_ocr_on_image(img,coordinates):
    img = cv2.imread(img)
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)

    text = ""
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
            text = res[1]

    return str(text)

def helmet_detect(img):
    global wearing_helmet
    wearing_helmet = True
    helmet_model = YOLO("helmet.pt")
    helmet_pred = helmet_model(img)
    list_of_helmets = list(helmet_pred[0].boxes.cls.numpy())
    if 1.0 in list_of_helmets:
        wearing_helmet = False

def detect_read_numberplate(img):
    numberplate_model = YOLO("best.pt")
    detect_plate = numberplate_model(img,save_crop = True)
    coords = list(detect_plate[0].boxes.xyxy.numpy()[0])
    text = perform_ocr_on_image(img,coords)
    return text


def run():
    wearing_helmet = True
    img = r"image.jpeg"  # Idhar image ka path daalneka
    wearing_helmet = helmet_detect(img)
    if wearing_helmet == False:
        text = detect_read_numberplate(img)
        print(text)
    else:
        print("Helmet is on.")

if __name__ == "__main__":
    run()
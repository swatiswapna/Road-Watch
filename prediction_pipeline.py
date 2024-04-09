'''''
import torch
#from torchvision.models.detection import yolov8 



model1 = torch.hub.load('best2.pt')
model2 = torch.hub.load('best1.pt')

def pipeline(input_data):
    
    output1 = model1('input_data')
    output2 = model2(output2)
    # Add more models if needed
    return output2

input_data = ('/Users/swatea/Developer/hackathon_srm/_7efa6990-e35b-11e7-814a-000c05070a4c.jpg')

output2 = pipeline(input_data)

print(output2)

'''
import torch
from ultralytics import YOLO

# Load the models
helmet_model_path = '/Users/swatea/Developer/hackathon_srm/best2.pt'
numplate_model_path = '/Users/swatea/Developer/hackathon_srm/ultralytics/runs/detect/train_model/weights/best.pt'

try:
    # Load the custom model using YOLO
    helmet_model = YOLO(helmet_model_path)
    numplate_model = YOLO(numplate_model_path)

    print("Models loaded successfully")
except Exception as e:
    print("Error loading models:", e)

def detect_helmet(input_image):
    helmet_results = helmet_model(input_image)
    return helmet_results

def detect_number_plate(input_image):
    numplate_results = numplate_model(input_image)
    return numplate_results

def pipeline(input_image):
    helmet_result = detect_number_plate(input_image)
    
    if helmet_result == 'No helmet':
        numplate_results = detect_number_plate(helmet_result)
        return numplate_results  # Return the result here
    else:
        return helmet_result  # Return the helmet result if helmet is detected

# Example usage:
input_image = '/Users/swatea/Developer/hackathon_srm/Y2meta.app-MUMBAI TRAFFIC | INDIA-(1080p).mp4'
pipeline_results = pipeline(input_image)
print(pipeline_results)


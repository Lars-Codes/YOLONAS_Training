import openvino as ov 
import torch 
import os
from ultralytics import YOLO, NAS 
from dotenv import load_dotenv
from super_gradients.training import models 

load_dotenv()
MODEL_ARCH = os.getenv('MODEL_ARCH')
model_path = "./modules/average_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

HOME = os.getcwd()
SAVE_FOLDER = "RUN_20240529_154605_778380"
CHECKPOINT_DIR = f'{HOME}/checkpoints'
EXPERIMENT_NAME = "car_"

model = models.get(
    MODEL_ARCH, 
    num_classes=1, 
    checkpoint_path=f"{CHECKPOINT_DIR}/{EXPERIMENT_NAME}/{SAVE_FOLDER}/average_model.pth"
).to(DEVICE)
# model = models.get(MODEL_ARCH, pretrained_weights="coco").to(DEVICE)
# state_dict = torch.load(model_path)

# # pt = model.load_state_dict(state_dict)
# print("type: ", type(state_dict))
# # model = models.get()

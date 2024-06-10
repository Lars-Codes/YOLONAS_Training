import openvino as ov 
import torch 
import os

model_path = './modules/average_model.pth'
model = torch.load(model_path)

ov_model = ov.convert_model(model)

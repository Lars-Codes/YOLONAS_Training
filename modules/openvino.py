import openvino as ov 
import torch 

model_path = 'model/path'
model = torch.load(model_path)

ov_model = ov.convert_model(model)

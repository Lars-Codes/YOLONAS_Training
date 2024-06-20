import os 
import json 
import base64
from PIL import Image
import PIL 
import io
import cv2 
import numpy as np 
import math 

folder_path = "./labelme_json"

# decode script from LabelMe. Decodes to (h, w, num channels) tuple 
def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

def midpoint(x1, x2, y1, y2): 
    return ((x1 + x2)/2, (y1 + y2)/2)

def normalize(val_tuple, w, h): 
    return (val_tuple[0]/w, val_tuple[1]/h)  

def dimensions(x1, x2, y1, y2): 
    return (abs(x2-x1), abs(y2-y1))

for filename in os.listdir(folder_path): 
    file_path = os.path.join(folder_path, filename)

    if filename.endswith('.json'): 
        name = filename.split('.json')[0]
        with open(file_path, 'r') as file: 
            try: 
                loadjson = json.load(file)

                imageData = img_b64_to_arr(loadjson['imageData'])
                height, width = imageData.shape[0], imageData.shape[1]
                
                for entry in loadjson['shapes']: 
                    label = entry['label'] 
                    pointarray = entry['points'] 

                    box = [item for sublist in pointarray for item in sublist]
                    center = ' '.join(map(str, normalize(midpoint(box[0], box[2], box[1], box[3]), 1920, 1080)))
                    dim = ' '.join(map(str, normalize(dimensions(box[0], box[2], box[1], box[3]), 1920, 1080)))

                    file_path = folder_path + '/' + name + ".txt"

                    with open(file_path, 'a') as file: 
                        file.write(label + " " + center + " " + dim + "\n")

            except Exception as e: 
                print(e)


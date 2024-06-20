import os 
import json 
import base64
from PIL import Image
import PIL 
import io
import cv2 
import numpy as np 
import math 
import random 
import shutil 

labels = []

# decode script from LabelMe. Decodes to (h, w, num channels) tuple 
def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

# Finds midpoint of box 
def midpoint(x1, x2, y1, y2): 
    return ((x1 + x2)/2, (y1 + y2)/2)

# normalizes pixel values 
def normalize(val_tuple, w, h): 
    return (val_tuple[0]/w, val_tuple[1]/h)  

# Gets dimensions of box 
def dimensions(x1, x2, y1, y2): 
    return (abs(x2-x1), abs(y2-y1))

# Converts labelme.json file to yolov5 annotation format .txt 
def convert_annotations(folder_path):
    label_path = folder_path + "/labels"
    for filename in os.listdir(label_path): 
        file_path = os.path.join(label_path, filename)

        if filename.endswith('.json'): 
            name = filename.split('.json')[0]
            with open(file_path, 'r') as file: 
                try: 
                    loadjson = json.load(file)

                    imageData = img_b64_to_arr(loadjson['imageData'])
                    height, width = imageData.shape[0], imageData.shape[1]
                    
                    for entry in loadjson['shapes']: 
                        label = entry['label'] 

                        if(label not in labels): 
                            labels.append(label)

                        pointarray = entry['points'] 

                        box = [item for sublist in pointarray for item in sublist]
                        center = ' '.join(map(str, normalize(midpoint(box[0], box[2], box[1], box[3]), width, height)))
                        dim = ' '.join(map(str, normalize(dimensions(box[0], box[2], box[1], box[3]), width, height)))

                        file_path = label_path + '/' + name + ".txt"

                        with open(file_path, 'a') as file: 
                            file.write(label + " " + center + " " + dim + "\n")

                    img = Image.fromarray(imageData)
                    img.save(folder_path + "/images/" + name + ".jpg")

                    json_path = label_path + "/" + filename
                    os.remove(json_path)

                except Exception as e: 
                    print(e)

def make_directories(file_path): 
    os.makedirs(file_path + "/test", exist_ok=True)
    os.makedirs(file_path + "/train", exist_ok=True)
    os.makedirs(file_path + "/valid", exist_ok=True)

    os.makedirs(file_path + "/test/images", exist_ok=True)
    os.makedirs(file_path + "/test/labels", exist_ok=True)

    os.makedirs(file_path + "/train/images", exist_ok=True)
    os.makedirs(file_path + "/train/labels", exist_ok=True)

    os.makedirs(file_path + "/valid/images", exist_ok=True)
    os.makedirs(file_path + "/valid/labels", exist_ok=True)

def move_data(file_array, input_folder, dest_folder): 
    for f in file_array: 
        src = os.path.join(input_folder, f)
        dest = os.path.join(dest_folder, f)
        shutil.move(src, dest)

def split_data(input_dir, val_size=0.1, test_size=0.1): 
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]  # Get all .json annotations
    random.shuffle(json_files) # Shuffle list randomly 

    total_files = len(json_files) # total # of json annotations 
    split1 = int(total_files*val_size)
    split2 = split1 + int(total_files*test_size)

    val_split = json_files[:split1]
    test_split = json_files[split1:split2]
    train_split = json_files[split2:]

    move_data(val_split, input_dir, input_dir + "/valid/labels")
    move_data(test_split, input_dir, input_dir + "/test/labels")
    move_data(train_split, input_dir, input_dir + "/train/labels")

def generate_yaml(label_arr, folder_path): 
    f = open(folder_path + "/data.yaml", "a")
    f.write("names:\n")
    for label in label_arr: 
        f.write("- " + "'" + label + "'" + "\n")
    f.write("nc: " + str(len(label_arr)) + "\n")
    f.write("test: " + folder_path + "/test/images" + "\n") 
    f.write("train: " + folder_path + "/train/images" + "\n") 
    f.write("val: " + folder_path + "/val/images") 


path = "./labelme_json"
make_directories(path)
split_data(path)
convert_annotations(path + "/valid")
convert_annotations(path + "/test")
convert_annotations(path + "/train")
generate_yaml(labels, path)

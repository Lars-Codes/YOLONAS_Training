# code to add padding to validation dataset 
import os 
from PIL import Image

def addPadding(dir_path):
    try:  
        for filename in os.listdir(dir_path): 
            file_path = dir_path + "/" + filename
            img = Image.open(file_path)

            padded_img = Image.new(img.mode, (1920, 1088), (0,0,255))

            # print(padded_img.size)
            # print("img original: ", img.size)

            padded_img.paste(img, (0,0)) # paste old image on top-left corner

            padded_img.save(file_path)

    except Exception as e: 
        print(e)


addPadding("./vehicles/valid/images")

img = Image.open("./vehicles/valid/images/frame_0002.jpg")
print(img.size)
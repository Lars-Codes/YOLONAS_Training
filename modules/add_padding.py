# code to add padding to validation dataset 
import os 
from PIL import Image


def addPadding(dir_path, new_w=640, new_h=640):
    try:  
        """
            Args: 
                new_w: desired new width of the image
                new_h: desired new height of the image 
        """
        for filename in os.listdir(dir_path): 
            file_path = dir_path + "/" + filename
            actual_img = Image.open(file_path)

            # padded_img = Image.new(img.mode, (1920, 1088), (0,0,255))
            temp_img = Image.new(actual_img.mode, (new_w, new_h), (0, 0, 0))


            # calculate resize ratio 
            w, h = actual_img.size 
            w_ratio = new_w / float(w)
            h_ratio = new_h / float(h)

            if(w_ratio > h_ratio): 
                scale_factor = h_ratio 
            else: 
                scale_factor = w_ratio 

            
            # Resize the image with the calculated factor
            new_size = int(w * scale_factor), int(h * scale_factor)
            resized_img = actual_img.resize(new_size, Image.LANCZOS)

            # Calculate the padding offsets to center the resized image
            x_offset = int((new_w - new_size[0]) / 2)
            y_offset = int((new_h - new_size[1]) / 2)

            # Paste the resized image onto the temp image with padding
            temp_img.paste(resized_img, (x_offset, y_offset))

            temp_img.save(file_path)

    except Exception as e: 
        print(e)


addPadding("../pad_test", new_w = 1920, new_h=1088)

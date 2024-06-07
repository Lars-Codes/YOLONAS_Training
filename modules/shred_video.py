import cv2
import os

# Taken from https://www.kaggle.com/discussions/general/491148

def convert_video_to_images(input_video, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(input_video)
    success, frame = video_capture.read()
    count = 0

    # Read each frame and save it as an image
    while success:
        image_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")  # Adjust the format as per your requirement
        cv2.imwrite(image_path, frame)  # Save the frame as an image
        success, frame = video_capture.read()  # Read next frame
        count += 1

    # Release the video capture object
    video_capture.release()

# Provide the path to the input video and output image folder
input_video = r'./day_thermal_camera.mp4'
output_folder = r'./images'

# Call the function to convert the video to images
convert_video_to_images(input_video, output_folder)

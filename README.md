## Bash scripts 
Run all bash scripts with >> source <script_name>.sh, so venvs will activate to current shell 
1. install_labelme.sh
  This script can be used to install labelme.
2. pipeline.sh
   This script is intended to convert annotated LabelMe JSON images to YOLO format, train the dataset using YOlO-NAS, and then optimize using OpenVINO.

## Modules 
1. check_gpu.py: This script checks the availability of a GPU on your local machine. It is used by pipeline.sh to check for the GPU before running.
2. openvino.py: This script converts the model to openvino IR. Used by pipeline.sh. 
3. shred_video.py: This script can be used to shred an input video into individual frames.
4. test.py: used to test trained model
5. train.py: used to train daya. Used by pipeline.sh. 

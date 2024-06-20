#!/bin/bash 

# Run with >>source pipeline.sh so venv will activate to current shell

# deactivate currently running virtual environment
deactivate 

# Create new virtual environment and activate it 
python3 -m venv conversion_env
source conversion_env/bin/activate

# Make sure all of your annotated images are saved to ./images directory. Create it if not exists

# Install dependencies
pip3 install -r requirements.txt

val = 0.10 # 10% of data for validation 
test = 0.10 # 10% of data for testing

# Convert LabelMeJSON to YOLO JSON
labelme2yolo --json_dir ./images/ --val_size $val --test_size $test

# Now images are stored in ./images/YOLODataset directory 
python3 ./modules/check_gpu.py 

# Check if GPU is available
if [ $? -eq 1 ]; then
    echo "GPU check passed, continuing script..."
else
    echo "GPU not available. Consult documentation troubleshoot. You'll need toto set environment variables, or upgrade driver, or downgrade PyTorch, or some combination of these."
    exit 1 
fi

mkdir checkpoints 

# Train the model, include code to convert to onnx
# python3 ./modules/train.py 

# Optimize with openvino 
mo --input_model model_name.onnx




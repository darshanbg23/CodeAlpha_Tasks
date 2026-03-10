import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Optional - Suppress TensorFlow logs

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Predict handwritten character")
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'],
                    help='Dataset model to use: mnist (digits) or emnist (letters). Default: mnist')
parser.add_argument('image_path', type=str, help='Path to image file')
args = parser.parse_args()

DATASET = args.dataset
image_path = args.image_path

# Dataset-specific model file
model_file = f"model_{DATASET}.keras"

print(f"Loading model...")
try:
    model = load_model(model_file, compile=False)
except (FileNotFoundError, ValueError):
    print(f"Error: Model file '{model_file}' not found.")
    print(f"Please train the model first using: python train.py --dataset {DATASET}")
    exit(1)

print(f"Loading image: {image_path}")
try:
    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype='float32')
    
    # Detect content region
    use_cropping = False
    rows = np.any(img_array < 240, axis=1)
    cols = np.any(img_array < 240, axis=0)
    
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        crop_h, crop_w = rmax - rmin, cmax - cmin
        
        # Crop only if bounding box is valid and removes large empty borders
        orig_h, orig_w = img_array.shape
        if crop_h >= 10 and crop_w >= 10 and (crop_h < orig_h * 0.9 or crop_w < orig_w * 0.9):
            img_array = img_array[rmin:rmax+1, cmin:cmax+1]
            use_cropping = True
    
    # Resize to 28x28
    if use_cropping:
        img_pil = Image.fromarray(img_array).convert('L')
        img_pil.thumbnail((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new('L', (28, 28), color=255)
        offset_x = (28 - img_pil.width) // 2
        offset_y = (28 - img_pil.height) // 2
        canvas.paste(img_pil, (offset_x, offset_y))
        img_array = np.array(canvas, dtype='float32') / 255.0
     # Fallback: direct resize to 28x28
    else:
        img_pil = Image.fromarray(img_array).convert('L')
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil, dtype='float32') / 255.0
    
    # Invert if background is white
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
    
    img_array = img_array.reshape(1, 28, 28, 1)
    
    prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index]
    
    # Class count check
    num_classes = len(prediction[0])
    if DATASET == "emnist" and num_classes != 26:
        print(f"Warning: EMNIST requires 26 classes, but model has {num_classes}.")
    elif DATASET == "mnist" and num_classes != 10:
        print(f"Warning: MNIST requires 10 classes, but model has {num_classes}.")
    
    # Decode prediction based on dataset
    if DATASET == "emnist":
        predicted_label = chr(predicted_index + ord('A'))
        print(f"\nPredicted Character: {predicted_label}")
    else:
        predicted_label = predicted_index
        print(f"\nPredicted Digit: {predicted_label}")
    
    print(f"Confidence: {confidence * 100:.2f}%")
    
except FileNotFoundError:
    print(f"Error: Image file '{image_path}' not found.")
    exit(1)
except Exception as e:
    print(f"Error processing image: {e}")
    exit(1)
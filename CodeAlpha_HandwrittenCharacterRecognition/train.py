import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Optional - Suppress TensorFlow logs

import argparse
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train handwritten character recognition model")
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'emnist'],
                    help='Dataset to use: mnist (digits) or emnist (letters). Default: mnist')
parser.add_argument('--force', action='store_true', help='Force retraining even if model exists')
args = parser.parse_args()

DATASET = args.dataset
force_train = args.force

# Dataset-specific model file
model_file = f"model_{DATASET}.keras"

# Skip training if model already exists (unless --force is used)
if os.path.exists(model_file) and not force_train:
    print(f"Model already exists. Skipping training.")
    exit(0)

# Load dataset
if DATASET == "mnist":
    print("Loading MNIST dataset (digits 0-9)...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    num_classes = 10

elif DATASET == "emnist":
    print("Loading EMNIST dataset (letters A-Z)...")
    import tensorflow_datasets as tfds
    ds_train = tfds.load('emnist/letters', split='train')
    ds_test = tfds.load('emnist/letters', split='test')
    X_train = np.array([d['image'].numpy().flatten().reshape(28, 28) for d in ds_train])
    y_train = np.array([d['label'].numpy() - 1 for d in ds_train])
    X_test = np.array([d['image'].numpy().flatten().reshape(28, 28) for d in ds_test])
    y_test = np.array([d['label'].numpy() - 1 for d in ds_test])
    
    # Fix EMNIST orientation
    X_train = np.array([np.fliplr(np.transpose(img)) for img in X_train])
    X_test = np.array([np.fliplr(np.transpose(img)) for img in X_test])
    
    num_classes = 26

# Preprocess training data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Build CNN model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

print("Training started...")
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Confusion matrix
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)

print("\nConfusion Matrix:")
print(cm)

# Save model
model.save(model_file)
print(f"\nModel saved as {model_file}")
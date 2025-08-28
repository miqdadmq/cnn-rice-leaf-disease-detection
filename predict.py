import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# === CONFIGURATION ===
MODEL_PATH = "best_model.h5"

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="Path to input image")
args = parser.parse_args()

img_path = args.image_path

CLASS_NAMES = ["bacterial_blight", "blast", "brown_spot", "healthy", "tungro"]
IMG_SIZE = (180, 180)

# === LOAD MODEL ===
model = tf.keras.models.load_model(MODEL_PATH)

# === LOAD & PREPROCESS IMAGE ===
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0  #Normalization
img_array = np.expand_dims(img_array, axis=0)  #Batch dimension

# === PREDICT ===
pred = model.predict(img_array)
predicted_class = CLASS_NAMES[np.argmax(pred)]
confidence = np.max(pred) * 100

# === OUTPUT ===
plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
plt.show()
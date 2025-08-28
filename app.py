import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

model = tf.keras.models.load_model("best_model.h5")
class_names = ["bacterial_blight", "blast", "brown_spot", "healthy", "tungro"]

def predict(img):
    img = image.array_to_img(img).resize((180, 180))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    prob = np.max(pred)
    label = class_names[np.argmax(pred)]
    return {label: float(prob)}

demo = gr.Interface(
    fn=predict,
    inputs="image",
    outputs="label",
    title="Rice Leaf Disease Detection"
)

if __name__ == "__main__":
    demo.launch()

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# App title
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier (ResNet Model)")
st.markdown("Upload a brain MRI scan image to classify it into one of four categories:"  
            " **Glioma**, **Meningioma**, **No Tumor**, or **Pituitary**.")

# Define labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Download model from Google Drive if not exists
MODEL_PATH = "resnet_model.h5"
GDRIVE_FILE_ID = "12NDlK4tpZR6WN78D4pChs793UAEC1Cw5"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Image Preprocessing
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = image.convert('RGB')  # in case it's grayscale
    img_array = np.array(image) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

# Upload image
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    # Predict
    with st.spinner("Classifying..."):
        input_img = preprocess_image(image)
        prediction = model.predict(input_img)
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # Show result
    st.success(f"🧪 **Prediction:** `{pred_class}`")
    st.info(f"🔬 **Confidence:** `{confidence:.2f}%`")

    st.markdown("---")
    st.markdown("📌 *Model: Custom ResNet on Brain MRI Dataset*")


import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# App title
st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")
st.title("🚦 Traffic Sign Recognition")

# Load class labels
labels_df = pd.read_csv("labels.csv")
class_labels = labels_df.set_index('ClassId')['Name'].to_dict()

# Function to filter unknowns
def is_unknown(label):
    label = label.lower().strip()
    return "unknown" in label or label == "go"

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("TrafficSign_model.h5")
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-5:][::-1]

    st.subheader("🔍 Top Predictions:")
    found_valid = False

    for idx in top_indices:
        label = class_labels.get(idx, "Unknown")
        confidence = predictions[idx] * 100

        if not is_unknown(label):
            found_valid = True
            st.write(f"**{label}** — {confidence:.2f}%")

    if not found_valid:
        st.warning("Prediction was inconclusive or predicted as 'Unknown'. Try another image.")

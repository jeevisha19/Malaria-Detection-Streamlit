
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# Load Model
# -------------------------
model = tf.keras.models.load_model("best_malaria_model.keras")


class_names = ["Parasitized", "Uninfected"]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Malaria Detection",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 Malaria Detection using EfficientNetB0")

st.write(
    "Upload a microscopic blood smear image to detect whether the cell is infected."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224,224))

    img = np.array(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    probability = prediction[0][0]

    if probability < 0.5:
        predicted = "Parasitized"
        confidence = (1-probability)*100
    else:
        predicted = "Uninfected"
        confidence = probability*100

    st.success(f"Prediction: {predicted}")

    st.write(f"Confidence: {confidence:.2f}%")

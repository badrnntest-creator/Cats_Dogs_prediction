import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Dogs vs Cats Classifier", page_icon="🐾", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), "cats_dogs_model.keras"))

model = load_model()

st.title("🐾 Dogs vs Cats Classifier")
st.markdown("""
### Classification par réseau de neurones convolutionnel
**Projet EHTP — Module 06 Deep Learning**

Uploadez une image de **chien** ou de **chat** pour obtenir la prédiction du modèle.
""")
st.divider()

uploaded_file = st.file_uploader("Choisissez une image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image uploadée")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Prédiction")
        img_array = np.array(image.resize((150, 150)), dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0   # Normalisation MobileNetV2
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_batch, verbose=0)[0][0]
        label      = "Dog 🐶" if prediction >= 0.5 else "Cat 🐱"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        st.markdown(f"### Classe prédite : **{label}**")
        st.metric("Confiance", f"{confidence * 100:.1f}%")
        st.progress(float(confidence))
        st.divider()
        st.markdown(f"Probabilité dog : `{prediction:.4f}`")
        st.markdown(f"Probabilité cat : `{1 - prediction:.4f}`")

st.caption("Modèle : MobileNetV2 Transfer Learning — EHTP M06 2024")

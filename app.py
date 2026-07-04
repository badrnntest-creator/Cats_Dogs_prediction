import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --- Configuration de la page ---
st.set_page_config(
    page_title="Dogs vs Cats Classifier",
    page_icon="🐾",
    layout="centered"
)

# --- Chargement du modèle (une seule fois grâce au cache) ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "cats_dogs_model.keras")
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- Interface utilisateur ---
st.title("🐾 Dogs vs Cats Classifier")
st.markdown("""
### Classification d'images par réseau de neurones convolutionnel
**Projet EHTP — Module 06 Deep Learning**

Uploadez une image de **chien** ou de **chat** et le modèle prédit automatiquement la classe.
""")

st.divider()

# --- Upload de l'image ---
uploaded_file = st.file_uploader(
    "Choisissez une image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="L'image sera redimensionnée à 150×150 pixels pour la prédiction."
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image uploadée")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image originale", use_container_width=True)

    with col2:
        st.subheader("Prédiction")

        # Prétraitement : resize + normalisation MobileNetV2
        img_resized = image.resize((150, 150))
        img_array  = np.array(img_resized, dtype=np.float32)

        # Prétraitement MobileNetV2 : normalisation [-1, 1]
        img_array  = (img_array / 127.5) - 1.0
        img_batch  = np.expand_dims(img_array, axis=0)  # Ajout dimension batch

        # Prédiction
        prediction = model.predict(img_batch, verbose=0)[0][0]

        # Interprétation (cat=0, dog=1)
        label      = "Dog 🐶" if prediction >= 0.5 else "Cat 🐱"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        # Affichage du résultat
        st.markdown(f"### Classe prédite : **{label}**")
        st.metric(label="Confiance", value=f"{confidence * 100:.1f}%")
        st.progress(float(confidence))

        st.divider()
        st.markdown(f"**Probabilité dog** : {prediction:.4f}")
        st.markdown(f"**Probabilité cat** : {1 - prediction:.4f}")

st.divider()
st.caption("Modèle : MobileNetV2 Transfer Learning — EHTP M06 Deep Learning 2024")

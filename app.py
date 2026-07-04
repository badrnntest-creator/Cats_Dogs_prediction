import streamlit as st
import numpy as np
import os
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dogs vs Cats Classifier",
    page_icon="🐾",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size:2.2rem; font-weight:700; color:#1a1a2e; margin-bottom:0; }
    .sub-title    { font-size:1rem;   color:#555;       margin-top:4px; }
    .result-box   { padding:1.2rem 1.5rem; border-radius:10px;
                    border:2px solid; margin-top:1rem; text-align:center; }
    .dog          { background:#e8f5e9; border-color:#43a047; color:#1b5e20; }
    .cat          { background:#e3f2fd; border-color:#1e88e5; color:#0d47a1; }
    .result-label { font-size:2rem; font-weight:800; }
    .result-conf  { font-size:1rem; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "model", "cats_dogs_model.keras")

@st.cache_resource(show_spinner="Chargement du modele CNN...")
def load_model():
    try:
        import tensorflow as tf
        if not os.path.exists(MODEL_PATH):
            return None, f"Modele introuvable : {os.path.abspath(MODEL_PATH)}"
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

model, load_error = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🐾 Dogs vs Cats Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Projet EHTP — Module 06 Deep Learning | '
    'MobileNetV2 Transfer Learning</p>',
    unsafe_allow_html=True,
)
st.divider()

if load_error:
    st.error(f"**Erreur de chargement du modele :** {load_error}")
    st.info("Assurez-vous que `model/cats_dogs_model.keras` est bien present dans le depot.")
    st.stop()

st.success(f"Modele charge — {model.count_params():,} parametres")

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown("### Charger une image")
st.caption("Formats acceptes : JPG, JPEG, PNG — l'image sera redimensionnee en 150×150 px")

uploaded = st.file_uploader(
    "Glissez-deposez ou cliquez pour selectionner",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ── Prediction ────────────────────────────────────────────────────────────────
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Image uploadee**")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("**Resultat de la prediction**")

        # Pretraitement : resize 150x150 + normalisation [0,1]
        img_array = np.array(image.resize((150, 150)), dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        proba = float(model.predict(img_batch, verbose=0)[0][0])

        if proba > 0.5:
            label      = "Dog"
            emoji      = "🐶"
            confidence = proba
            css_class  = "dog"
        else:
            label      = "Cat"
            emoji      = "🐱"
            confidence = 1.0 - proba
            css_class  = "cat"

        st.markdown(
            f'<div class="result-box {css_class}">'
            f'<div class="result-label">{emoji} {label}</div>'
            f'<div class="result-conf">Confiance : <strong>{confidence*100:.1f}%</strong></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Probabilites brutes**")
        st.progress(proba, text=f"Dog : {proba*100:.1f}%")
        st.progress(1 - proba, text=f"Cat : {(1-proba)*100:.1f}%")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        col_a.metric("P(Dog)", f"{proba:.4f}")
        col_b.metric("P(Cat)", f"{1-proba:.4f}")

st.divider()
st.caption("Modele : MobileNetV2 Transfer Learning | EHTP M06 Deep Learning 2024")

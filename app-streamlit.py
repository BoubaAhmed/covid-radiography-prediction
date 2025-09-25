import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Charger le modèle
model = load_model("covid_model.keras")
class_names = ["COVID", "Normal", "Pneumonia"]

# UI
st.title("🩺 Prédiction Radiographie Pulmonaire")
st.markdown("Chargez une image de radiographie pour prédire si elle est **COVID-19**, **Normale**, ou **Pneumonie**.")

uploaded_file = st.file_uploader("📤 Uploader une radiographie", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image chargée", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Prédiction : **{predicted_class}**")

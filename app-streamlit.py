import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
st.set_page_config(layout="wide")

# Load model and class names (now with Lung Opacity)
# model = load_model("covid_model.keras")
model = load_model("covid_detection_model.keras")
class_names = ["COVID", "Normal", "Pneumonia", "Lung Opacity"]

# Sidebar with info
st.sidebar.title("ℹ️ À propos du projet")
st.sidebar.markdown(
    """
    **Projet : Prédiction COVID-19 sur radiographies**
    
    - **Dataset:** [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
    - **Auteur:** [BoubaAhmed](https://github.com/BoubaAhmed)
    - **Modèle:** Deep Learning (Keras)
    
    **Classes possibles :**
    - 🦠 COVID-19
    - 🫁 Normal
    - 🤒 Pneumonia
    - 🌫️ Lung Opacity
    """
)
st.sidebar.info(
    "Ce projet est à but éducatif. Pour plus d'informations, consultez la [page du dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)."
)

# Main Title & Instructions
st.title("🩺 Prédiction Radiographie Pulmonaire")
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:12px;border-radius:8px;">
    <b>Instructions :</b>
    <ul>
      <li>Chargez une image de radiographie thoracique <i>(formats JPG, PNG)</i>.</li>
      <li>Le modèle prédit si l’image correspond à : <b>COVID-19</b>, <b>Normale</b>, <b>Pneumonie</b> ou <b>Lung Opacity</b>.</li>
      <li>Pour de meilleurs résultats, utilisez une image claire et non floue.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Layout: upload + results
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("1️⃣ Uploader une radiographie")
    uploaded_file = st.file_uploader("Glissez-déposez ou cliquez pour choisir une image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with st.spinner("Analyse de l'image..."):
            img = Image.open(uploaded_file).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_batch)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Show prediction result
            st.success(f"✅ Résultat : **{predicted_class}** ({confidence:.2f}% de confiance)")

            # Show full probabilities
            st.markdown("**Probabilités détaillées :**")
            prob_table = {
                "Classe": class_names,
                "Probabilité (%)": [f"{p*100:.2f}" for p in prediction]
            }
            st.table(prob_table)

            # Show original image in col2
            with col2:
                st.subheader("2️⃣ Aperçu de l'image")
                st.image(img, caption="Image chargée", width='stretch')

    else:
        st.info("Veuillez uploader une radiographie pour commencer la prédiction.")

# Footer
st.markdown("""---""")
st.markdown(
    '<small>Développé par BoubaAhmed • Modèle deep learning sur radiographies • [GitHub](https://github.com/BoubaAhmed)</small>',
    unsafe_allow_html=True
)
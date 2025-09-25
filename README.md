# 🫁 Radiographie Pulmonaire - Prédiction avec Deep Learning

Ce projet utilise le Deep Learning pour classer des **radiographies pulmonaires** en trois catégories principales :
- 🦠 COVID-19
- 🤒 Pneumonie
- ✅ Poumons sains

Le modèle est développé avec **TensorFlow/Keras** et déployé avec **Streamlit** pour une interface simple et interactive.

---

## 🚀 Fonctionnalités
- Entraînement d’un CNN (Convolutional Neural Network) sur le dataset Kaggle.
- Sauvegarde et réutilisation du modèle entraîné.
- Application Streamlit pour tester le modèle en important des images de radiographies.
- Callbacks intelligents : EarlyStopping, ReduceLROnPlateau et sauvegarde du meilleur modèle.

---

## 📂 Dataset
Le dataset est disponible sur Kaggle :  
👉 [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

---

## ⚙️ Installation

Clonez le repo et installez les dépendances :

```bash
git clone https://github.com/BoubaAhmed/covid-radiography-prediction
cd radiographie-pulmonaire-prediction
pip install -r requirements.txt
```

---
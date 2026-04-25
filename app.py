import streamlit as st  # type: ignore[import]
import pandas as pd
import numpy as np
import joblib

# --- 1. CHARGEMENT DES MODÈLES ---
@st.cache_resource # Permet de charger les modèles une seule fois en mémoire
def load_models():
    modele_reg = joblib.load('modele_regression_rf.pkl')
    modele_clf = joblib.load('modele_classification_log.pkl')
    scaler = joblib.load('scaler.pkl')
    colonnes = joblib.load('colonnes.pkl')
    return modele_reg, modele_clf, scaler, colonnes

modele_reg, modele_clf, scaler, colonnes = load_models()

# --- 2. INTERFACE UTILISATEUR (UI) ---
st.set_page_config(page_title="Immo Predictor", page_icon="🏠", layout="centered")

st.title("🏠 Estimateur Immobilier Intelligent")
st.markdown("*Propulsé par Machine Learning (Random Forest & Régression Logistique)*")
st.write("---")

st.sidebar.header("📝 Caractéristiques du bien")

# Saisie des informations par l'utilisateur
ville = st.sidebar.selectbox("Ville", ["Casablanca", "Rabat", "Marrakech", "Agadir", "Fès"])
quartier_texte = st.sidebar.selectbox("Standing du Quartier", ["Économique", "Intermédiaire", "Huppé"])
type_bien = st.sidebar.selectbox("Type de bien", ["Appartement", "Villa", "Maison", "Studio"])

superficie = st.sidebar.number_input("Superficie (m²)", min_value=20, max_value=1000, value=80)
nb_chambres = st.sidebar.slider("Nombre de chambres", 1, 8, 2)
nb_salles_bain = st.sidebar.slider("Salles de bain", 1, 5, 1)
etage = st.sidebar.number_input("Étage (0 = RDC)", min_value=0, max_value=20, value=1)
age_batiment = st.sidebar.number_input("Âge du bâtiment (années)", min_value=0, max_value=100, value=5)

parking = st.sidebar.radio("Parking disponible ?", ["Non", "Oui"])
proximite_mer = st.sidebar.radio("Proximité Mer ?", ["Non", "Oui"])

st.sidebar.markdown("---")
st.sidebar.info("Projet réalisé par : Walid Idbennacer")

# --- 3. PRÉPARATION DES DONNÉES POUR LE MODÈLE ---
# Mapping pour le feature engineering (comme dans le notebook A.1)
mapping_quartier = {'Économique': 1, 'Intermédiaire': 2, 'Huppé': 3}
etat_batiment_val = 4 if age_batiment <= 5 else (3 if age_batiment <= 15 else (2 if age_batiment <= 30 else 1))

# Création d'un dictionnaire avec toutes les colonnes initialisées à 0
input_data = {col: 0 for col in colonnes}

# Remplissage des variables numériques et ordinales
input_data['quartier'] = mapping_quartier[quartier_texte]
input_data['superficie'] = superficie
input_data['nb_chambres'] = nb_chambres
input_data['nb_salles_bain'] = nb_salles_bain
input_data['etage'] = etage
input_data['age_batiment'] = age_batiment
input_data['parking'] = 1 if parking == "Oui" else 0
input_data['proximite_mer'] = 1 if proximite_mer == "Oui" else 0
input_data['etat_batiment_encoded'] = etat_batiment_val

# Remplissage des variables One-Hot Encoding (Type Bien)
if f'type_bien_{type_bien}' in colonnes:
    input_data[f'type_bien_{type_bien}'] = 1

# Remplissage des variables One-Hot Encoding (Ville)
if f'ville_{ville}' in colonnes:
    input_data[f'ville_{ville}'] = 1

# Conversion en DataFrame
df_input = pd.DataFrame([input_data])

# Mise à l'échelle (StandardScaler) uniquement sur les bonnes colonnes
colonnes_a_scaler = ['superficie', 'nb_chambres', 'nb_salles_bain', 'etage', 'age_batiment']
df_input[colonnes_a_scaler] = scaler.transform(df_input[colonnes_a_scaler])

# --- 4. PRÉDICTIONS ---
col1, col2 = st.columns(2)

with col1:
    if st.button("💰 Estimer le Prix (Régression)", use_container_width=True):
        prediction_prix = modele_reg.predict(df_input)[0]
        st.success(f"### Prix estimé :\n# {prediction_prix:,.0f} DH".replace(',', ' '))
        st.caption("Modèle : Random Forest (R² ≈ 90%)")

with col2:
    if st.button("🏷️ Prédire la Gamme (Classification)", use_container_width=True):
        prediction_gamme = modele_clf.predict(df_input)[0]
        if prediction_gamme == 1:
            st.error("### Gamme du bien :\n# 💎 Haut de gamme")
        else:
            st.info("### Gamme du bien :\n# 🛋️ Standard")
        st.caption("Modèle : Régression Logistique (Acc ≈ 95%)")
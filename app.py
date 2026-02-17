import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Estimateur Bourges IA",
    page_icon="🏡",
    layout="centered"
)

# Style CSS pour cacher les menus techniques et faire "Pro"
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #2980b9;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3498db;
    }
    .resultat-box {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre
st.title("🏡 Estimateur Immobilier Bourges")
st.write("Bienvenue sur l'outil d'estimation basé sur l'Intelligence Artificielle et les données notariées (DVF).")

# -----------------------------------------------------------------------------
# FONCTIONS DU MOTEUR (CACHE)
# -----------------------------------------------------------------------------
@st.cache_resource
def charger_et_entrainer():
    # 1. Chargement des données
    try:
        # Essai avec séparateur virgule
        df = pd.read_csv('bourges_data.csv', sep=',', low_memory=False)
        # Si ça plante ou si colonnes bizarres, on tente le pipe |
        if 'valeur_fonciere' not in df.columns:
            df = pd.read_csv('bourges_data.csv', sep='|', low_memory=False)
    except:
        return None, 0

    # 2. Nettoyage
    # On garde Vente + Maison/Appart
    if 'nature_mutation' in df.columns:
        df = df[df['nature_mutation'] == 'Vente']
    
    if 'type_local' in df.columns:
        df = df[df['type_local'].isin(['Maison', 'Appartement'])]

    # Conversions numériques
    cols_a_convertir = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude']
    for col in cols_a_convertir:
        if col in df.columns:
            # On remplace les virgules par des points (ex: 12,5 -> 12.5)
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

    # Suppression des lignes vides ou erreurs
    df = df.dropna(subset=cols_a_convertir)
    # Filtres de cohérence (pas de vente à 1€, pas de 0m²)
    df = df[(df['valeur_fonciere'] > 10000) & (df['surface_reelle_bati'] > 9)]

    # 3. Préparation pour l'IA
    # On transforme "Maison" en 1 et "Appartement" en 0
    df['type_encode'] = df['type_local'].apply(lambda x: 1 if x == 'Maison' else 0)

    # Variables utilisées pour prédire
    X = df[['surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude', 'type_encode']]
    y = df['valeur_fonciere']

    # 4. Entraînement
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, len(df)

# -----------------------------------------------------------------------------
# CHARGEMENT SILENCIEUX
# -----------------------------------------------------------------------------
with st.spinner("Calibrage de l'IA sur le marché de Bourges..."):
    model, nb_ventes = charger_et_entrainer()

if model is None:
    st.error("🚨 Erreur critique : Le fichier 'bourges_data.csv' est introuvable ou illisible.")
    st.stop()

# -----------------------------------------------------------------------------
# INTERFACE UTILISATEUR
# -----------------------------------------------------------------------------

st.markdown("### 1. Caractéristiques du bien")
col1, col2 = st.columns(2)

with col1:
    type_bien = st.radio("Type de bien", ["Maison", "Appartement"], horizontal=True)
    surface = st.number_input("Surface habitable (m²)", min_value=10, max_value=600, value=90, step=1)

with col2:
    pieces = st.number_input("Nombre de pièces", min_value=1, max_value=20, value=4, step=1)
    annee = st.number_input("Année de construction", min_value=1800, max_value=2025, value=1990, step=5)

st.markdown("### 2. Localisation & État")
st.info("💡 Astuce : Sur Google Maps, faites un clic-droit sur la maison pour voir la Latitude et Longitude.")

col3, col4 = st.columns(2)
with col3:
    lat = st.number_input("Latitude (ex: 47.08...)", value=47.0810, format="%.4f")
    lon = st.number_input("Longitude (ex: 2.39...)", value=2.3988, format="%.4f")
    
    # Petite carte pour vérifier
    data_map = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(data_map, zoom=13)

with col4:
    etat = st.select_slider("État général", options=["À rénover", "Moyen", "Bon", "Très bon", "Neuf/Refait"], value="Bon")
    dpe = st.select_slider("Diagnostic DPE", options=["G", "F", "E", "D", "C", "B", "A"], value="D")

# -----------------------------------------------------------------------------
# CALCUL ET RÉSULTAT
# -----------------------------------------------------------------------------
if st.button("LANCER L'ESTIMATION"):
    
    # 1. Estimation de base par l'IA (comparaison avec les voisins)
    type_code = 1 if type_bien == "Maison" else 0
    # On demande au modèle de prédire
    prediction_brute = model.predict([[surface, pieces, lat, lon, type_code]])[0]

    # 2. Ajustements "Experts" (Algorithme Hédonique)
    # L'IA connaît le prix du quartier, mais on affine avec l'état et le DPE
    coefs_etat = {"À rénover": 0.75, "Moyen": 0.90, "Bon": 1.0, "Très bon": 1.10, "Neuf/Refait": 1.20}
    coefs_dpe = {"G": 0.80, "F": 0.90, "E": 0.95, "D": 1.0, "C": 1.05, "B": 1.10, "A": 1.15}
    
    # Bonus/Malus Année
    coef_annee = 1.0
    if annee < 1940: coef_annee = 1.05 # Charme de l'ancien
    if 1960 <= annee <= 1975: coef_annee = 0.95 # Construction rapide
    
    prix_final = prediction_brute * coefs_etat[etat] * coefs_dpe[dpe] * coef_annee

    # 3. Affichage
    st.markdown(f"""
    <div class="resultat-box">
        <h3>VALEUR ESTIMÉE</h3>
        <div style="font-size: 40px; font-weight: bold;">{prix_final:,.0f} €</div>
        <div style="font-size: 18px; margin-top: 10px;">Soit environ {prix_final/surface:,.0f} € / m²</div>
        <p style="font-size: 12px; margin-top: 15px; color: #155724;">
            *Basé sur l'analyse IA de {nb_ventes} ventes réelles à Bourges.
        </p>
    </div>
    """, unsafe_allow_html=True)

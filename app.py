import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Estimateur Bourges IA",
    page_icon="🏡",
    layout="centered"
)

# CSS Pro
st.markdown("""
    <style>
    .main {background-color: #ffffff;}
    h1 {color: #2c3e50; text-align: center;}
    .stButton>button {
        width: 100%; background-color: #2980b9; color: white; font-weight: bold;
        border-radius: 10px; padding: 15px; border: none;
    }
    .stButton>button:hover {background-color: #3498db;}
    .resultat-box {
        background-color: #d4edda; color: #155724; padding: 20px;
        border-radius: 10px; border: 1px solid #c3e6cb; text-align: center; margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏡 Estimateur Immobilier Bourges")
st.write("Utilisez la carte pour localiser le bien précisément.")

# -----------------------------------------------------------------------------
# FONCTIONS DU MOTEUR (CACHE)
# -----------------------------------------------------------------------------
@st.cache_resource
def charger_et_entrainer():
    try:
        df = pd.read_csv('bourges_data.csv', sep=',', low_memory=False)
    except:
        df = pd.read_csv('bourges_data.csv', sep='|', low_memory=False)

    # Nettoyage
    df = df[df['nature_mutation'] == 'Vente']
    df = df[df['type_local'].isin(['Maison', 'Appartement'])]
    cols = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=cols)
    df = df[(df['valeur_fonciere'] > 10000) & (df['surface_reelle_bati'] > 9)]

    # Entraînement
    df['type_encode'] = df['type_local'].apply(lambda x: 1 if x == 'Maison' else 0)
    X = df[['surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude', 'type_encode']]
    y = df['valeur_fonciere']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, len(df)

with st.spinner("Chargement de l'IA..."):
    model, nb_ventes = charger_et_entrainer()

# -----------------------------------------------------------------------------
# INTERFACE
# -----------------------------------------------------------------------------

st.subheader("1. Localisation (Cliquez sur la carte)")

# Carte interactive centrée sur Bourges
m = folium.Map(location=[47.0810, 2.3988], zoom_start=13)
m.add_child(folium.LatLngPopup()) # Permet de cliquer pour avoir les coords

# Affichage de la carte et récupération du clic
map_data = st_folium(m, height=400, width=700)

# Logique de récupération des coordonnées
lat, lon = 47.0810, 2.3988 # Valeur par défaut (Centre)
if map_data['last_clicked']:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']
    st.success(f"📍 Position sélectionnée : {lat:.5f}, {lon:.5f}")
else:
    st.info("👆 Cliquez sur la carte pour placer le bien.")

st.markdown("---")
st.subheader("2. Caractéristiques")

col1, col2 = st.columns(2)
with col1:
    type_bien = st.radio("Type de bien", ["Maison", "Appartement"], horizontal=True)
    surface = st.number_input("Surface (m²)", 10, 600, 90)
with col2:
    pieces = st.number_input("Pièces", 1, 20, 4)
    annee = st.number_input("Année construction", 1800, 2025, 1990)

col3, col4 = st.columns(2)
with col3:
    etat = st.select_slider("État", options=["À rénover", "Moyen", "Bon", "Très bon", "Neuf"], value="Bon")
with col4:
    dpe = st.select_slider("DPE", options=["G", "F", "E", "D", "C", "B", "A"], value="D")

# -----------------------------------------------------------------------------
# CALCUL
# -----------------------------------------------------------------------------
if st.button("LANCER L'ESTIMATION"):
    type_code = 1 if type_bien == "Maison" else 0
    prix_brut = model.predict([[surface, pieces, lat, lon, type_code]])[0]

    # Ajustements
    coefs_etat = {"À rénover": 0.75, "Moyen": 0.90, "Bon": 1.0, "Très bon": 1.10, "Neuf": 1.20}
    coefs_dpe = {"G": 0.80, "F": 0.90, "E": 0.95, "D": 1.0, "C": 1.05, "B": 1.10, "A": 1.15}
    
    coef_annee = 1.0
    if annee < 1940: coef_annee = 1.05 
    if 1960 <= annee <= 1975: coef_annee = 0.95
    
    prix_final = prix_brut * coefs_etat[etat] * coefs_dpe[dpe] * coef_annee

    st.markdown(f"""
    <div class="resultat-box">
        <h3>VALEUR ESTIMÉE</h3>
        <div style="font-size: 40px; font-weight: bold;">{prix_final:,.0f} €</div>
        <div style="font-size: 18px; margin-top: 10px;">Soit {prix_final/surface:,.0f} € / m²</div>
    </div>
    """, unsafe_allow_html=True)


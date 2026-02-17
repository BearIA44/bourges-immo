import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import folium
from streamlit_folium import st_folium
import requests

# -----------------------------------------------------------------------------
# 1. DESIGN & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Expertise Immo Bourges", page_icon="🏡", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .resultat-container {
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .methodologie-box {
        background-color: #f8f9fa;
        border-left: 5px solid #2c3e50;
        padding: 20px;
        border-radius: 5px;
        font-size: 14px;
        color: #444;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        font-weight: bold;
        border: none;
        height: 50px;
        border-radius: 8px;
    }
    .stButton>button:hover {background-color: #219150;}
    </style>
""", unsafe_allow_html=True)

st.title("🏡 Estimation Immobilière Pro")
st.write("Obtenez une valeur de marché précise basée sur les transactions réelles (DVF).")

# -----------------------------------------------------------------------------
# 2. MOTEUR IA (CACHE)
# -----------------------------------------------------------------------------
@st.cache_resource
def charger_cerveau():
    try:
        df = pd.read_csv('bourges_data.csv', sep=',', low_memory=False)
    except:
        df = pd.read_csv('bourges_data.csv', sep='|', low_memory=False)

    df = df[df['nature_mutation'] == 'Vente']
    df = df[df['type_local'].isin(['Maison', 'Appartement'])]
    # Filtre technique
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
    return model, len(df) # On renvoie aussi le nombre de ventes

# Fonction API Adresse
def trouver_adresse_gouv(adresse):
    if not adresse: return None
    url = f"https://api-adresse.data.gouv.fr/search/?q={adresse}&citycode=18033&limit=1"
    try:
        r = requests.get(url).json()
        if r['features']:
            coords = r['features'][0]['geometry']['coordinates']
            return coords[1], coords[0]
    except:
        return None
    return None

with st.spinner("Chargement des données de marché..."):
    model, nb_ventes_total = charger_cerveau()

# -----------------------------------------------------------------------------
# 3. INTERFACE
# -----------------------------------------------------------------------------
recherche = st.text_input("📍 Rechercher une adresse à Bourges", placeholder="Ex: 10 rue Moyenne...")
lat_defaut, lon_defaut, zoom = 47.0810, 2.3988, 13

if recherche:
    coords = trouver_adresse_gouv(recherche)
    if coords:
        lat_defaut, lon_defaut = coords
        zoom = 16

m = folium.Map(location=[lat_defaut, lon_defaut], zoom_start=zoom)
folium.Marker([lat_defaut, lon_defaut], icon=folium.Icon(color="red", icon="home")).add_to(m)
m.add_child(folium.LatLngPopup())
map_data = st_folium(m, height=250, use_container_width=True)

lat, lon = lat_defaut, lon_defaut
if map_data and map_data['last_clicked']:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']

# Formulaire
st.markdown("#### Caractéristiques du bien")
c1, c2, c3 = st.columns(3)
with c1: type_bien = st.selectbox("Type", ["Maison", "Appartement"])
with c2: surface = st.number_input("Surface (m²)", 10, 500, 90)
with c3: pieces = st.number_input("Pièces", 1, 15, 4)

c4, c5 = st.columns(2)
with c4: etat = st.select_slider("État", options=["À rénover", "Standard", "Bon", "Excellent", "Neuf"], value="Bon")
with c5: dpe = st.select_slider("DPE", options=["G", "F", "E", "D", "C", "B", "A"], value="D")

# -----------------------------------------------------------------------------
# 4. CALCUL & EXPLICATION PRO
# -----------------------------------------------------------------------------
if st.button("CALCULER L'ESTIMATION"):
    
    # 1. Prédiction IA (Prix brut "Voisinage")
    type_code = 1 if type_bien == "Maison" else 0
    prix_brut = model.predict([[surface, pieces, lat, lon, type_code]])[0]

    # 2. Coefficients
    coefs_etat = {"À rénover": 0.80, "Standard": 0.95, "Bon": 1.0, "Excellent": 1.1, "Neuf": 1.2}
    coefs_dpe = {"G": 0.80, "F": 0.90, "E": 0.95, "D": 1.0, "C": 1.05, "B": 1.10, "A": 1.15}
    
    impact_etat_val = coefs_etat[etat]
    impact_dpe_val = coefs_dpe[dpe]
    
    prix_final = prix_brut * impact_etat_val * impact_dpe_val

    # AFFICHAGE RÉSULTAT
    st.markdown(f"""
    <div class="resultat-container">
        <h3 style="margin:0;">Valeur de Marché Estimée</h3>
        <div style="font-size: 42px; font-weight: bold; margin: 10px 0;">{prix_final:,.0f} €</div>
        <div style="opacity: 0.9;">Soit {prix_final/surface:,.0f} € / m²</div>
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION EXPLICATIVE "PROFESSIONNELLE" ---
    st.markdown("### 📊 Analyse détaillée de l'estimation")
    
    # Calcul des pourcentages pour l'affichage
    bonus_malus_dpe = (impact_dpe_val - 1) * 100
    bonus_malus_etat = (impact_etat_val - 1) * 100
    signe_dpe = "+" if bonus_malus_dpe > 0 else ""
    signe_etat = "+" if bonus_malus_etat > 0 else ""

    st.markdown(f"""
    <div class="methodologie-box">
        <b>1. Méthode utilisée : Comparative de Marché (Machine Learning)</b><br>
        Cette estimation a été réalisée en comparant votre bien avec une base de données de 
        <b>{nb_ventes_total} transactions réelles</b> enregistrées par les notaires à Bourges.
        <br><br>
        <b>2. Critères Géographiques :</b><br>
        L'algorithme a pondéré le prix en fonction de la micro-localisation exacte (Latitude {lat:.4f}, Longitude {lon:.4f}), 
        prenant en compte la cote spécifique de votre rue.
        <br><br>
        <b>3. Impact de vos caractéristiques (Ajustements) :</b>
        <ul>
            <li><b>Performance Énergétique (DPE {dpe}) :</b> {signe_dpe}{bonus_malus_dpe:.0f}% sur la valeur standard.</li>
            <li><b>État du bien ({etat}) :</b> {signe_etat}{bonus_malus_etat:.0f}% sur la valeur standard.</li>
        </ul>
        <br>
        <i>Note : Cette estimation est indicative. Seule une visite technique permet de valider la luminosité, l'agencement et les finitions.</i>
    </div>
    """, unsafe_allow_html=True)



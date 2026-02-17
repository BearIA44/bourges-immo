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
    .option-tag {
        display: inline-block;
        background-color: #e1f5fe;
        color: #0277bd;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        margin-right: 5px;
        margin-bottom: 5px;
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

st.title("🏡 Estimation Immobilière Complète")
st.write("Obtenez une valeur de marché précise incluant terrain, piscine et annexes.")

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
    
    # Conversion numérique
    cols = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude', 'surface_terrain']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = 0

    df = df.dropna(subset=['valeur_fonciere', 'surface_reelle_bati', 'latitude', 'longitude'])
    df = df[(df['valeur_fonciere'] > 10000) & (df['surface_reelle_bati'] > 9)]
    
    # Nettoyage Terrain : 0 pour les apparts pour ne pas fausser l'IA
    df.loc[df['type_local'] == 'Appartement', 'surface_terrain'] = 0
    df['surface_terrain'] = df['surface_terrain'].fillna(0)

    # Encodage
    df['type_encode'] = df['type_local'].apply(lambda x: 1 if x == 'Maison' else 0)
    
    # IA avec Terrain inclus
    X = df[['surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude', 'type_encode', 'surface_terrain']]
    y = df['valeur_fonciere']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, len(df)

# Fonction API Adresse Gouv
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

# --- FORMULAIRE ---
st.markdown("#### Caractéristiques Principales")
c1, c2, c3 = st.columns(3)
with c1: type_bien = st.selectbox("Type", ["Maison", "Appartement"])
with c2: surface = st.number_input("Surface Habitable (m²)", 10, 500, 90)
with c3: pieces = st.number_input("Pièces", 1, 15, 4)

# Options Dynamiques
surface_terrain = 0
options_plus = []

if type_bien == "Maison":
    st.markdown("#### Extérieur & Annexes")
    c_t1, c_t2 = st.columns(2)
    with c_t1:
        surface_terrain = st.number_input("Surface Terrain (m²)", 0, 10000, 400, help="Surface totale de la parcelle")
    with c_t2:
        piscine = st.checkbox("🏊 Piscine Creusée")
        garage = st.checkbox("🚗 Garage / Box fermé")
        
    if piscine: options_plus.append("Piscine")
    if garage: options_plus.append("Garage")

else: # Appartement
    st.markdown("#### Extérieur & Annexes")
    c_a1, c_a2, c_a3 = st.columns(3)
    with c_a1: balcon = st.checkbox("Balcon")
    with c_a2: terrasse = st.checkbox("Grande Terrasse")
    with c_a3: parking = st.checkbox("Parking / Garage")
    
    if balcon: options_plus.append("Balcon")
    if terrasse: options_plus.append("Terrasse")
    if parking: options_plus.append("Parking")

st.markdown("#### État & Énergie")
c4, c5 = st.columns(2)
with c4: etat = st.select_slider("État", options=["À rénover", "Standard", "Bon", "Excellent", "Neuf"], value="Bon")
with c5: dpe = st.select_slider("DPE", options=["G", "F", "E", "D", "C", "B", "A"], value="D")

# -----------------------------------------------------------------------------
# 4. CALCUL & RÉSULTATS
# -----------------------------------------------------------------------------
if st.button("CALCULER L'ESTIMATION"):
    
    # A. Prédiction IA (Murs + Terrain)
    type_code = 1 if type_bien == "Maison" else 0
    terrain_ia = surface_terrain if type_bien == "Maison" else 0
    
    # L'IA prédit le prix de base
    prix_brut = model.predict([[surface, pieces, lat, lon, type_code, terrain_ia]])[0]

    # B. Coefficients Qualitatifs
    coefs_etat = {"À rénover": 0.80, "Standard": 0.95, "Bon": 1.0, "Excellent": 1.1, "Neuf": 1.2}
    coefs_dpe = {"G": 0.80, "F": 0.90, "E": 0.95, "D": 1.0, "C": 1.05, "B": 1.10, "A": 1.15}
    
    # C. Valorisation des Annexes (Bonus Manuels)
    valeur_annexes = 0
    
    if type_bien == "Maison":
        if 'Piscine' in options_plus: valeur_annexes += 20000
        if 'Garage' in options_plus: valeur_annexes += 10000
    else: # Appartement
        if 'Balcon' in options_plus: prix_brut *= 1.03
        if 'Terrasse' in options_plus: prix_brut *= 1.08
        if 'Parking' in options_plus: valeur_annexes += 8000

    # Calcul Final
    prix_ajuste = prix_brut * coefs_etat[etat] * coefs_dpe[dpe]
    prix_final = prix_ajuste + valeur_annexes

    # AFFICHAGE
    st.markdown(f"""
    <div class="resultat-container">
        <h3 style="margin:0;">Valeur de Marché Estimée</h3>
        <div style="font-size: 42px; font-weight: bold; margin: 10px 0;">{prix_final:,.0f} €</div>
        <div style="opacity: 0.9;">Soit {prix_final/surface:,.0f} € / m²</div>
    </div>
    """, unsafe_allow_html=True)

    # EXPLICATION
    impact_dpe_pct = (coefs_dpe[dpe] - 1) * 100
    impact_etat_pct = (coefs_etat[etat] - 1) * 100
    html_annexes = "".join([f'<span class="option-tag">{opt}</span>' for opt in options_plus]) if options_plus else "Aucune option sélectionnée"
    
    st.markdown(f"""
    <div class="methodologie-box">
        <b>1. Analyse IA & Terrain :</b><br>
        Comparaison avec {nb_ventes_total} ventes notariées. L'IA a pris en compte la surface habitable ({surface}m²), 
        le terrain ({terrain_ia}m²) et la géolocalisation précise.
        <br><br>
        <b>2. Impact Qualitatif :</b>
        <ul>
            <li><b>DPE {dpe} :</b> {impact_dpe_pct:+.0f}%</li>
            <li><b>État {etat} :</b> {impact_etat_pct:+.0f}%</li>
        </ul>
        <b>3. Valorisation des Annexes & Équipements :</b><br>
        {html_annexes}<br>
        <i>Les annexes ont été ajoutées à la valeur vénale de base par algorithme expert.</i>
    </div>
    """, unsafe_allow_html=True)





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
st.set_page_config(page_title="Estimateur Bourges", page_icon="🏡", layout="centered")

# CSS pour rendre l'outil beau et propre
st.markdown("""
    <style>
    /* Cacher les menus Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style des titres */
    h1 {color: #2c3e50; font-family: 'Helvetica', sans-serif;}
    
    /* Boite de résultat */
    .resultat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin-top: 20px;
    }
    .prix-gros {font-size: 45px; font-weight: bold; margin: 0;}
    .prix-m2 {font-size: 18px; opacity: 0.9;}
    
    /* Bouton principal */
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
        border: none;
    }
    .stButton>button:hover {background-color: #34495e;}
    </style>
""", unsafe_allow_html=True)

st.title("🏡 Estimez votre bien à Bourges")
st.write("Entrez une adresse ou cliquez sur la carte pour obtenir une valeur de marché basée sur l'IA.")

# -----------------------------------------------------------------------------
# 2. FONCTIONS INTELLIGENTES (CACHE)
# -----------------------------------------------------------------------------
@st.cache_resource
def charger_cerveau():
    # Chargement et nettoyage identiques
    try:
        df = pd.read_csv('bourges_data.csv', sep=',', low_memory=False)
    except:
        df = pd.read_csv('bourges_data.csv', sep='|', low_memory=False)

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
    return model

# Fonction pour trouver l'adresse via l'API Gouv
def trouver_adresse_gouv(adresse):
    if not adresse: return None
    # On force la recherche sur Bourges (code insee 18033) pour éviter les erreurs
    url = f"https://api-adresse.data.gouv.fr/search/?q={adresse}&citycode=18033&limit=1"
    try:
        r = requests.get(url).json()
        if r['features']:
            coords = r['features'][0]['geometry']['coordinates']
            return coords[1], coords[0] # Renvoie Lat, Lon
    except:
        return None
    return None

# Chargement discret
model = charger_cerveau()

# -----------------------------------------------------------------------------
# 3. INTERFACE ERGONOMIQUE
# -----------------------------------------------------------------------------

# --- PARTIE A : LOCALISATION (La barre de recherche) ---
col_search, col_map = st.columns([1, 2])

# Variables par défaut (Centre ville)
lat_defaut, lon_defaut = 47.0810, 2.3988
zoom_level = 13

# Barre de recherche
recherche = st.text_input("🔍 Rechercher une adresse à Bourges", placeholder="Ex: 12 rue d'Auron...")

# Si on tape une adresse, on met à jour la carte
if recherche:
    coords = trouver_adresse_gouv(recherche)
    if coords:
        lat_defaut, lon_defaut = coords
        zoom_level = 16 # On zoome fort sur la maison
        st.success("✅ Adresse trouvée !")
    else:
        st.warning("Adresse introuvable, essayez de pointer sur la carte.")

# Carte
m = folium.Map(location=[lat_defaut, lon_defaut], zoom_start=zoom_level)
folium.Marker([lat_defaut, lon_defaut], icon=folium.Icon(color="red", icon="home")).add_to(m)
m.add_child(folium.LatLngPopup()) # Permet de cliquer ailleurs

# Affichage carte
with st.expander("Voir/Modifier l'emplacement sur la carte", expanded=True):
    map_data = st_folium(m, height=300, use_container_width=True)

# Récupération position finale (soit recherche, soit clic)
lat, lon = lat_defaut, lon_defaut
if map_data and map_data['last_clicked']:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']


# --- PARTIE B : DÉTAILS DU BIEN ---
st.markdown("#### Caractéristiques")
c1, c2, c3 = st.columns(3)
with c1:
    type_bien = st.selectbox("Type", ["Maison", "Appartement"])
with c2:
    surface = st.number_input("Surface (m²)", 10, 500, 90)
with c3:
    pieces = st.number_input("Pièces", 1, 15, 4)

c4, c5 = st.columns(2)
with c4:
    etat = st.select_slider("État", options=["À rénover", "Standard", "Bon", "Excellent", "Neuf"], value="Bon")
with c5:
    dpe = st.select_slider("DPE", options=["G", "F", "E", "D", "C", "B", "A"], value="D")

# -----------------------------------------------------------------------------
# 4. CALCUL ET RÉSULTAT
# -----------------------------------------------------------------------------
if st.button("CALCULER L'ESTIMATION"):
    
    # Calcul IA
    type_code = 1 if type_bien == "Maison" else 0
    prix_brut = model.predict([[surface, pieces, lat, lon, type_code]])[0]

    # Coefficients ergonomiques
    coefs_etat = {"À rénover": 0.8, "Standard": 0.95, "Bon": 1.0, "Excellent": 1.1, "Neuf": 1.2}
    coefs_dpe = {"G": 0.8, "F": 0.9, "E": 0.95, "D": 1.0, "C": 1.05, "B": 1.1, "A": 1.15}
    
    prix_final = prix_brut * coefs_etat[etat] * coefs_dpe[dpe]

    # Affichage Design
    st.markdown(f"""
    <div class="resultat-container">
        <div>Estimation de marché</div>
        <p class="prix-gros">{prix_final:,.0f} €</p>
        <div class="prix-m2">Soit environ {prix_final/surface:,.0f} € / m²</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("ℹ️ Cette estimation est basée sur les transactions réelles à proximité de l'adresse indiquée.")



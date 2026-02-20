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
st.set_page_config(page_title="Pré-Estimation Bourges", page_icon="🏡", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    .fourchette-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; padding: 30px; border-radius: 15px; text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2); margin-top: 20px;
    }
    .prix-bas { font-size: 28px; font-weight: bold; color: #ffb74d; }
    .prix-haut { font-size: 28px; font-weight: bold; color: #81c784; }
    .prix-moyen { font-size: 45px; font-weight: bold; margin: 10px 0; }
    
    .disclaimer-box {
        background-color: #fff3cd; border-left: 5px solid #ffeeba;
        padding: 15px; border-radius: 5px; font-size: 13px; color: #856404;
        margin-top: 15px; margin-bottom: 30px;
    }
    .methodologie-box {
        background-color: #f8f9fa; border-left: 5px solid #1e3c72;
        padding: 20px; border-radius: 5px; font-size: 14px; color: #444;
    }
    .stButton>button { background-color: #d35400; color: white; font-weight: bold; border: none; height: 50px; border-radius: 8px;}
    .stButton>button:hover {background-color: #e67e22;}
    </style>
""", unsafe_allow_html=True)

st.title("🏡 Pré-Estimation Immobilière")
st.write("Découvrez la fourchette de marché de votre bien avant l'avis de valeur par un conseiller.")

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
    
    cols = ['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude', 'surface_terrain']
    for col in cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        else: df[col] = 0

    df = df.dropna(subset=['valeur_fonciere', 'surface_reelle_bati', 'latitude', 'longitude'])
    df = df[(df['valeur_fonciere'] > 10000) & (df['surface_reelle_bati'] > 9)]
    
    df.loc[df['type_local'] == 'Appartement', 'surface_terrain'] = 0
    df['surface_terrain'] = df['surface_terrain'].fillna(0)

    df['type_encode'] = df['type_local'].apply(lambda x: 1 if x == 'Maison' else 0)
    
    X = df[['surface_reelle_bati', 'nombre_pieces_principales', 'latitude', 'longitude', 'type_encode', 'surface_terrain']]
    y = df['valeur_fonciere']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, len(df)

def trouver_adresse_gouv(adresse):
    if not adresse: return None
    url = f"https://api-adresse.data.gouv.fr/search/?q={adresse}&citycode=18033&limit=1"
    try:
        r = requests.get(url).json()
        if r['features']:
            coords = r['features'][0]['geometry']['coordinates']
            return coords[1], coords[0]
    except: return None
    return None

with st.spinner("Analyse du marché Berruyer en cours..."):
    model, nb_ventes_total = charger_cerveau()

# -----------------------------------------------------------------------------
# 3. INTERFACE (Nouveaux critères "Agence")
# -----------------------------------------------------------------------------
recherche = st.text_input("📍 Localisation précise (Adresse)", placeholder="Ex: 10 rue Moyenne...")
lat_defaut, lon_defaut, zoom = 47.0810, 2.3988, 13

if recherche:
    coords = trouver_adresse_gouv(recherche)
    if coords: lat_defaut, lon_defaut, zoom = coords[0], coords[1], 16

m = folium.Map(location=[lat_defaut, lon_defaut], zoom_start=zoom)
folium.Marker([lat_defaut, lon_defaut], icon=folium.Icon(color="red")).add_to(m)
m.add_child(folium.LatLngPopup())
map_data = st_folium(m, height=250, use_container_width=True)

lat, lon = lat_defaut, lon_defaut
if map_data and map_data['last_clicked']:
    lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']

st.markdown("#### Caractéristiques du bien")
c1, c2, c3 = st.columns(3)
with c1: type_bien = st.selectbox("Type", ["Maison", "Appartement"])
with c2: surface = st.number_input("Surface (m²)", 10, 500, 90)
with c3: pieces = st.number_input("Pièces", 1, 15, 4)

# NOUVEAU : Période de construction
epoque = st.selectbox("Année de construction", ["Avant 1945 (Ancien/Cachet)", "1946 - 1970 (Reconstruction)", "1971 - 1990 (Standard)", "1991 - 2010 (Récent)", "Après 2010 (Très récent/Neuf)"])

surface_terrain = 0
etage_coeff = 1.0

if type_bien == "Maison":
    st.markdown("#### Extérieur")
    surface_terrain = st.number_input("Surface Terrain (m²)", 0, 10000, 400)
    piscine = st.checkbox("🏊 Piscine")
    garage = st.checkbox("🚗 Garage")
else:
    st.markdown("#### Copropriété & Étage")
    ca1, ca2 = st.columns(2)
    with ca1: 
        etage = st.selectbox("Niveau", ["Rez-de-chaussée", "Étage intermédiaire", "Dernier étage"])
        ascenseur = st.checkbox("Présence d'un ascenseur")
    with ca2:
        etat_copro = st.selectbox("État de la copropriété", ["Travaux à prévoir", "Bon état général", "Refait à neuf / Récent"])
        balcon = st.checkbox("Balcon / Terrasse")
        parking = st.checkbox("Place de parking / Garage")

st.markdown("#### État Intérieur & DPE")
c4, c5 = st.columns(2)
with c4: etat = st.select_slider("État intérieur", options=["À rénover (Gros travaux)", "Rafraîchissement", "Bon état", "Refait à neuf"], value="Bon état")
with c5: dpe = st.select_slider("DPE", options=["G", "F", "E", "D", "C", "B", "A"], value="D")

# -----------------------------------------------------------------------------
# 4. CALCUL DE LA FOURCHETTE
# -----------------------------------------------------------------------------
if st.button("LANCER L'ANALYSE DE MARCHÉ"):
    
    type_code = 1 if type_bien == "Maison" else 0
    terrain_ia = surface_terrain if type_bien == "Maison" else 0
    prix_brut = model.predict([[surface, pieces, lat, lon, type_code, terrain_ia]])[0]

    # --- NOUVEAUX COEFFICIENTS "AGENCE" ---
    
    # 1. Année de construction
    coefs_epoque = {"Avant 1945 (Ancien/Cachet)": 1.05, "1946 - 1970 (Reconstruction)": 0.90, "1971 - 1990 (Standard)": 0.95, "1991 - 2010 (Récent)": 1.05, "Après 2010 (Très récent/Neuf)": 1.15}
    prix_brut *= coefs_epoque[epoque]

    # 2. Critères Appartements
    if type_bien == "Appartement":
        if etage == "Rez-de-chaussée": prix_brut *= 0.90 # Décote RDC
        if etage == "Dernier étage": prix_brut *= 1.05 # Surcote
        if etage != "Rez-de-chaussée" and not ascenseur: prix_brut *= 0.92 # Sans ascenseur
        
        coefs_copro = {"Travaux à prévoir": 0.90, "Bon état général": 1.0, "Refait à neuf / Récent": 1.05}
        prix_brut *= coefs_copro[etat_copro]

    # 3. État & DPE
    coefs_etat = {"À rénover (Gros travaux)": 0.75, "Rafraîchissement": 0.90, "Bon état": 1.0, "Refait à neuf": 1.15}
    coefs_dpe = {"G": 0.80, "F": 0.90, "E": 0.95, "D": 1.0, "C": 1.05, "B": 1.10, "A": 1.15}
    prix_ajuste = prix_brut * coefs_etat[etat] * coefs_dpe[dpe]

    # 4. Annexes
    valeur_annexes = 0
    if type_bien == "Maison":
        if piscine: valeur_annexes += 20000
        if garage: valeur_annexes += 10000
    else:
        if balcon: prix_ajuste *= 1.05
        if parking: valeur_annexes += 8000

    prix_median = prix_ajuste + valeur_annexes

    # --- CALCUL FOURCHETTE (-8% / +8%) ---
    marge_erreur = 0.08 
    prix_bas = prix_median * (1 - marge_erreur)
    prix_haut = prix_median * (1 + marge_erreur)

    # AFFICHAGE
    st.markdown(f"""
    <div class="fourchette-container">
        <div style="text-transform: uppercase; letter-spacing: 2px; font-size: 14px;">Valeur Vénale Estimée</div>
        <div class="prix-moyen">{prix_median:,.0f} €</div>
        <div style="display: flex; justify-content: space-around; margin-top: 15px;">
            <div><span style="font-size: 12px; opacity: 0.8;">Fourchette Basse</span><br><span class="prix-bas">{prix_bas:,.0f} €</span></div>
            <div><span style="font-size: 12px; opacity: 0.8;">Fourchette Haute</span><br><span class="prix-haut">{prix_haut:,.0f} €</span></div>
        </div>
        <div style="font-size: 14px; margin-top: 20px; opacity: 0.9;">Soit environ {prix_median/surface:,.0f} € / m²</div>
    </div>
    
    <div class="disclaimer-box">
        <strong>⚠️ Attention : Cette valeur est une pré-estimation statistique.</strong><br>
        Notre algorithme ne visite pas votre bien. La luminosité, l'agencement, les nuisances sonores ou les aménagements sur-mesure peuvent faire basculer le prix vers la fourchette haute ou basse.<br>
        <strong>Pour fixer un prix de vente définitif, l'avis de valeur d'un conseiller sur place est indispensable.</strong>
    </div>
    """, unsafe_allow_html=True)






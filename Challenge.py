
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
from streamlit_plotly_events import plotly_events
import seaborn as sns
import os
import warnings
import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.offline as py
import plotly.tools as tls
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from scipy.stats import levene
import plotly
import time
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from my_fonction import *
#from Good_KNN import *
from Fonction_Classement import *
from PIL import Image


#================Configuration des styles de la page ===========================
st.set_page_config(page_title="Blood Donation Dashboard",layout="wide")
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 200px;
            max-width: 200px;
        }
    </style>
    """,
    unsafe_allow_html=True)

st.markdown(
    """
    <style>
    body {
        font-family: sans-serif;
    }
    .title {
        text-align: center;
        color: #306609; /* Couleur du titre */
    }
    .subtitle {
        text-align: center;
        color: #6699CC; /* Couleur du sous-titre */
    }
    .section-header {
        background-color: #1864B8; /* Couleur de fond des sections */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        /* Style des en-têtes de tableau */
        .dataframe th {
            background: rgba(255, 0, 0, 0.2) !important; /* Rouge transparent */
            color: black !important; /* Texte noir pour contraste */
            font-weight: bold !important;
            text-align: center !important;
        }
        /* Bordures pour séparer les colonnes */
        .dataframe td, .dataframe th {
            border: 1px solid #ddd !important;
            padding: 10px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#=======================================================================
# Sélecteur de langue
def set_language():
    return st.sidebar.selectbox("🌍 Choisissez la langue / Choose the language", ["Français", "English"])



#Dictionnaire de traduction
translations = {
    "Français": {
        "title": "Tableau de bord de la campagne de don de sang",
        "sub_title": "Exploiter les données pour une meilleure gestion et planification des dons de sang",
        "teacher": "🎓 Enseignant: Mr Serge Ndoumin",
        "analysis": "📊 Analyse des données du e-commerce au Pakistan",
        "raw_data": "📂 Données Brutes",
        "raw_data_desc": "📌 Base brute sans traitement incluant les valeurs atypiques",
        "data_issues": "Nouveau Tableau de Bord",
        "stats_desc": "📈 Statistiques descriptives de la base",
        "processed_data": "⚙️ Données Traitées",
        "visualization": "📊 Visualisation des Indicateurs",
        "modeling": "🤖 Modélisation en Bonus",
        "group_members": "👥 Membres du Groupe",
        "rapport":"Produire un rapport",
        "form":"Formulaire"
    },
    "English": {
        "title": "Blood Donation Campaign Dashboard",
        "sub_title": "Harnessing data for better management and planning of blood donations",
        "teacher": "🎓 Instructor: Mr. Serge Ndoumin",
        "analysis": "📊 Analysis of E-commerce Data in Pakistan",
        "raw_data": "📂 Raw Data",
        "raw_data_desc": "📌 Raw dataset without processing, including outliers",
        "data_issues": "New Dashboard",
        "stats_desc": "📈 Descriptive statistics of the dataset",
        "processed_data": "⚙️ Processed Data",
        "visualization": "📊 Indicator Visualization",
        "modeling": "🤖 Bonus Modeling",
        "group_members": "👥 Group Members",
        "rapport":"Produce repport",
        "form":"Forms"
    }
}

#Contenu de la barre latérale
st.sidebar.image("Logo.png", use_container_width=True)
lang = set_language()
#them=set_custom_theme()
st.sidebar.title("Tables des Matières")
st.sidebar.title("1. Description Générale")
st.sidebar.title("2. Analyse géographique dans Douala")
st.sidebar.title("3. Analyse par arrondissement")
st.sidebar.title("4. Conditions de Santé & Éligibilité ")
st.sidebar.title("5")


st.sidebar.title(translations[lang]["group_members"])
members = ["KENGNE Bienvenu Landry",]
for member in members:
    st.sidebar.markdown(f"✅ {member}")

#==========================================================
#====================== EN TETE ===========================
#==========================================================
#============ Ajout d'un style CSS personnalisé ===========
#==========================================================

st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTabs {
            font-size: 18px;
            font-weight: bold;
        }
        .stText, .stDataFrame {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

#==========================================================
img1, img2 =st.columns(2)
with img1:
    image = Image.open("Image.png")
    resized_image = image.resize((626, 200))  # Ajuster à 400x300 pixels
    st.image(resized_image)
with img2:
    image2 = Image.open("Image2.png")
    resized_image = image2.resize((626, 200))  # Ajuster à 400x300 pixels
    st.image(resized_image)
#st.image("Image.png", use_container_width=True)  

# Affichage du titre et sous-titres avec un meilleur style
st.markdown(
    f"""
    <div style="text-align:center; padding:10px;">
        <h1 style="color:#ff4b4b;">🩸 {translations[lang]["title"]} 🩸</h1>
        <h3 style="color: #4b8bff;">{translations[lang]["sub_title"]}</h3>
    </div>
    """,
    unsafe_allow_html=True
)


#Chargement des bases de données
#@st.cache_data(persist="disk") #pour garder le cache entre les sessions
def load_data(geodata_path,excel_path,data_don_path):
    try:
        data = pd.read_excel(excel_path,sheet_name="2019")
        data2 = pd.read_excel(data_don_path,sheet_name="2020")
        geo_data = gpd.read_file(geodata_path)
        return geo_data, data, data2
    except FileNotFoundError as e:
        st.error(f"Erreur : Fichier introuvable - {e}")
        return None
    except Exception as e:
        st.exception(f"Une erreur s'est produite : {e}")
        return None


geo_data , data , data_don= load_data("GeoChallengeData.shp","Good_Data.xlsm","Good_Data.xlsm")
df_info_loc=pd.read_excel("Infos.xlsx",sheet_name="Local") #Information pour le dictionnaire de localisation
df_info_pof=pd.read_excel("Infos.xlsx",sheet_name="Prof") # Information pour le dictionnaire des métiers

all_Quartier=list(data["Quartier_de_Résidence"].unique())
metier=list(data["Profession"].unique())
All_religion=list(data["Good_Religion"].unique())
# Formatage des colones des différentes bases
geo_data.columns=['Arrondissement', 'ADM3_PCODE', 'ID',"Date_remplissage",
       'Age', 'Classe_age', 'Niveau_etude', 'Genre', 'Taille', 'Poids', 'Situation_Matrimoniale', 'Profession',"Categorie_profession", 'Region',
       'Quartier', 'Lat', 'Long', 'Nationalité', 'Religion', 'ancien_don_sang',
       'date_dernier_don', 'Ecart_dernier_don',"Duration", 'Taux_hémoglobine_(g/dl)', 'Eligibilite', 'Nb_Raison',
       'Raisons', 'Autre_Raisons', 'geometry']
data=data[['ID', 'Age', 'Classe_age','Date_remplissage',
       'Niveau_etude', 'Genre', 'Taille', 'Poids', 'Situation_Mat',
       'Profession', 'Good_profession', 'Region',
        'Good_Qrt', 'Good_Arrondissement', 'Lat',
       'Long', 'Nationalité', 'Good_Religion', 'Don_pass',
       'Date_last_don', 'Duration', 'Duration period', 'Tx_hémoglobine',
       'Eligibilité', 'Nb_Raison', 'Raisons', 'Autre_Raison']]

data.columns=['ID', 'Age', 'Classe_age', 'Date_remplissage', 'Niveau_etude', 'Genre',
       'Taille', 'Poids', 'Situation_Mat', 'Profession', 'Good_profession',
       'Region','Quartier', 'Arrondissement', 'Lat', 'Long', 'Nationalité',
       'Religion', 'Don_pass', 'Date_last_don', 'Duration', 'Duration_period', 'Tx_hémoglobine',
       'Eligibilité', 'Nb_Raison', 'Raisons', 'Autre_Raison']

var_qual=['Classe_age','Niveau_etude', 'Genre','Situation_Mat','Good_profession','Region','Quartier', 'Don_pass','Eligibilité']
var_quant=['Age','Tx_hémoglobine','Taille', 'Poids',]
#=======================Variables==================================
sequence_couleur=['reds', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'aggrnyl', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd']


#==========================================================
#==== FONCTION USUELLES ===================================
#==========================================================

#1. Fonction de production de carte
def make_chlorophet_map(df,style_carte="carto-positron",palet_color="Blues",opacity=0.8,width=500, height=300):
    geo_data_El=df[df["Eligibilite"]=="Eligible"]
    geo_data_TNE=df[df["Eligibilite"]=="Temporairement Non-eligible"]
    geo_data_NE=df[df["Eligibilite"]=="Définitivement non-eligible"]

    df_pts_El=geo_data_El.groupby("Quartier").agg({
    'Quartier': 'size',
    'Lat': 'first',
    'Long':'first'
    }).rename(columns={'Quartier': 'nb_donateur'})
    df_pts_El["Qrt"]=df_pts_El.index

    df_pts_TNE=geo_data_TNE.groupby("Quartier").agg({
        'Quartier': 'size',
        'Lat': 'first',
        'Long':'first'
    }).rename(columns={'Quartier': 'nb_donateur'})
    df_pts_TNE["Qrt"]=df_pts_TNE.index

    df_pts_NE=geo_data_NE.groupby("Quartier").agg({
        'Quartier': 'size',
        'Lat': 'first',
        'Long':'first'
    }).rename(columns={'Quartier': 'nb_donateur'})
    df_pts_NE["Qrt"]=df_pts_NE.index

    df_chlph=df.groupby("Arrondissement").agg({
        'Arrondissement': 'size',
        'geometry': 'first',
        'Long':'first',
        'Lat':'first'
    }).rename(columns={'Arrondissement': 'nb_donateur'})
    df_chlph["Arr"]=df_chlph.index
    df_chlph = gpd.GeoDataFrame(df_chlph, geometry='geometry')


    df_pts=df.groupby("Quartier").agg({
        'Quartier': 'size',
        'Lat': 'first',
        'Long':'first'
    }).rename(columns={'Quartier': 'nb_donateur'})
    df_pts["Qrt"]=df_pts.index
    
    fig = go.Figure(go.Choroplethmapbox(
        geojson=df_chlph.geometry.__geo_interface__,  # Géométries des arrondissements
        locations=df_chlph.index,  # Indices des polygones
        z=df_chlph["nb_donateur"],  # Variable à visualiser (nombre de donateurs)
        colorscale=palet_color,  # Échelle de couleurs
        marker_opacity=opacity,  # Opacité des polygones
        marker_line_width=0.5,  # Épaisseur des bordures
        colorbar_title="Nombre de Candidat",  # Titre de la barre de couleur
        hovertext=df_chlph['Arr'],
        hovertemplate=" %{hovertext}  <br>Nombre de candidat : %{z}<extra></extra>",
    ))
    
    fig.add_trace(go.Scattermapbox(
        lat=df_chlph["Lat"],
        lon=df_chlph["Long"],
        mode='text',
        text=df_chlph["Arr"],  # Nom des arrondissements
        textfont=dict(size=12, color="black"),
        hoverinfo='none'
    ))

    # Ajout des points pour représenter le nombre de donateurs
    fig.add_trace(go.Scattermapbox(
        lat=df_pts["Lat"],  # Colonne des latitudes des points
        lon=df_pts["Long"],  # Colonne des longitudes des points
        mode='markers',  # Mode de dispersion (points)
        name="Total candidat",
        marker=dict(
            size=df_pts["nb_donateur"],  # Taille des points basée sur 'nb_donateur'
            sizemode='area',  # La taille est proportionnelle à la surface
            sizeref=2. * max(df_pts["nb_donateur"]) / (45.**2),  # Ajustement de la taille
            color='#003F80',  # Couleur des points
            opacity=0.8  # Opacité des points
        ),
        hovertemplate=(
            "<b>Quartier :</b> %{text}<br>"
            "<b>Total candidat :</b> %{marker.size}<extra></extra>"
        ),  # Format de l'infobulle
        text=df_pts["Qrt"]
    ))
    # Ajout des points pour représenter le nombre de donateurs eligibles
    fig.add_trace(go.Scattermapbox(
        lat=df_pts_El["Lat"],  
        lon=df_pts_El["Long"],  
        mode='markers',  
        name="Eligibles",
        marker=dict(
            size=df_pts_El["nb_donateur"],  
            sizemode='area',  
            sizeref=2. * max(df_pts_El["nb_donateur"]) / (27.**2),  
            color='#0073E6',  
            opacity=0.75  
        ),
        hovertemplate=(
            "<b>Quartier :</b> %{text}<br>"
            "<b> candidat Eligibles :</b> %{marker.size}<extra></extra>"
        ),
        text=df_pts_El["Qrt"]
    ))
    # Ajout des points pour représenter le nombre de donateurs temporairement non-eligibles
    fig.add_trace(go.Scattermapbox(
        lat=df_pts_NE["Lat"],  
        lon=df_pts_NE["Long"],  
        mode='markers',  
        name="Temporairement Non-eligibles",
        marker=dict(
            size=df_pts_NE["nb_donateur"],  
            sizemode='area',  
            sizeref=2. * max(df_pts_NE["nb_donateur"]) / (17.**2),  
            color='#B3D9FF',  
            opacity=0.7  
        ),
        hovertemplate=(
            "<b>Quartier :</b> %{text}<br>"
            "<b>candidat Temporairement Non-eligibles :</b> %{marker.size}<extra></extra>"
        ), 
        text=df_pts_NE["Qrt"]
    ))
    # Ajout des points pour représenter le nombre de donateurs Non eligibles
    fig.add_trace(go.Scattermapbox(
        lat=df_pts_NE["Lat"],  
        lon=df_pts_NE["Long"],  
        mode='markers',  
        name="Non Eligible",
        marker=dict(
            size=df_pts_NE["nb_donateur"],  
            sizemode='area',  
            sizeref=2. * max(df_pts_NE["nb_donateur"]) / (10.**2),  
            color='white',  
            opacity=0.7  
        ),
        hovertemplate=(
            "<b>Quartier :</b> %{text}<br>"
            "<b> candidat Non-eligibles :</b> %{marker.size}<extra></extra>"
        ), 
        text=df_pts_NE["Qrt"]
    ))

    # Mise à jour de la mise en page de la carte
    fig.update_layout(
        showlegend=True,
        legend=dict(
            x=0.0,  # Centré horizontalement
            y=0.1,  # Sous la carte
            orientation='h',  # Légende horizontale
        ),
        mapbox=dict(
            #style="open-street-map",
            style=style_carte,  # Style de la carte
            #style="carto-darkmatter",
            center=dict(lat=df_pts["Lat"].mean(), lon=df_pts["Long"].mean()),  # Centrer la carte
            zoom=10  # Niveau de zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=width, height=height,
    )

    st.plotly_chart(fig)

#2. Fonction de calibrage de la carte
def calculate_zoom(lon_diff, lat_diff, map_width=800, map_height=600):
            max_zoom = 18
            zoom_level = 0
            while (lon_diff * 2 ** zoom_level < map_width) and (lat_diff * 2 ** zoom_level < map_height):
                zoom_level += 1
                if zoom_level >= max_zoom:
                    break
            return min(zoom_level, max_zoom)

def telecharger_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    
    st.download_button(
        label="📥 Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name="Rapport_TDB.pdf",
        mime="application/pdf",
    )
#==========================================================
#==========================================================

# Onglets d'affichage des données
tabs = st.tabs([
    translations[lang]["raw_data"], 
    translations[lang]["visualization"],
    translations[lang]["rapport"],
    translations[lang]["form"],
    translations[lang]["data_issues"], 
    translations[lang]["processed_data"],
])

# ONGLET 1: BASES DE DONNEES
with tabs[0]:
    st.markdown(f"**{translations[lang]['raw_data_desc']}**")
    st.dataframe(data)
    st.write("Données géospatialisée")
    st.dataframe(geo_data)
    st.write("Données sur les donneurs")
    st.dataframe(data_don)
    

# ONGLET 2: TABLEAU DE BORD PROPREMENT DIT
with tabs[1]:
    a1, a2, a3 = st.columns(3) #définition du nombre de colonne
    
    #Visualisation des métriques
    with a1:
        plot_metric("Total Individu",data.shape[0],prefix="",suffix="",show_graph=True,color_graph="rgba(0, 104, 201, 0.1)",)
    with a2:
        plot_metric_2("Age moyen des donneurs",data,"Age",prefix="",suffix=" ans",show_graph=True,color_graph="rgba(175, 32, 201, 0.2)",val_bin=45)
    with a3:
        plot_metric_2("Taux Moyen d'hémoglobine", data, "Tx_hémoglobine", suffix=" g/dl",show_graph=True,color_graph="rgba(1, 230, 15, 0.7)",val_bin=300)

    
    st.write(" ")
    # SECTION 1: DESCRIPTION GENERALE
    with st.expander("1. Description Générale des candidats", expanded=False,icon="🩸"):
        d1,d2= st.columns([7,3])
        with d1:
            da1,da2=st.columns(2)
            with da1:
                make_bar(data,var="Religion",titre="Répartition par réligion",ordre=1,width=500,height=350,sens='h')
            with da2:
                make_heat_map_2(data,vars=['Region', 'Arrondissement','Quartier'],order_var="ID",label_var='Quartier',titre="Répartition des candidats")
            
            dd1, dd2,dd3 =st.columns([2,4.5,3.5])
            with dd1:
                make_donutchart(data,var="Genre",titre="Genre des candidats")
            with dd2:
                make_cross_hist_b(data,var2="Niveau_etude",var1="Eligibilité",titre="Niveau d'étude",sens='v',typ_bar=1)
            with dd3:
                make_cross_hist_b(data,var2="Situation_Mat",var1="Eligibilité",titre="Statut Matrimonial",sens='v',typ_bar=0,width=650)
        with d2:
            make_cross_hist_b(data[data["Region"]!="Litoral"],"Eligibilité","Region",titre="Autre Région",width=600,height=400,typ_bar=1)
            make_donutchart(data,var="Eligibilité",titre="Statut des candidats",part=True)
            
 #SECTION 2: ANALYSE GEOGRAPHIQUE DANS DOUALA   
    with st.expander("Analyse géographique dans Douala", expanded=False,icon="🩸"):  
        cc1,cc2,cc3,cc4,cc5=st.columns([2, 1,1.5,2,3.5])
        with cc1:
            opacity=st.slider("Transparence Carte",min_value=0.0,max_value=1.0,value=0.8,step=0.01)
        with cc2:
            couleur=st.selectbox("Couleur carte",sequence_couleur)
        with cc3:
            style=st.selectbox("Type de carte",options=["open-street-map","carto-positron","carto-darkmatter"])
        with cc4:
            genre=st.multiselect("Filtre: Genre",options=data["Genre"].unique(),default=data["Genre"].unique())
        with cc5:
            Statut_Mat=st.multiselect("Filtre: Statut Marital",options=data["Situation_Mat"].unique(),default=data["Situation_Mat"].unique())
               
        col1, col2 = st.columns([5, 3.4])
        geo_data_dla=geo_data[geo_data["Genre"].isin(genre)] if len(genre)!=0 else geo_data 
        geo_data_dla=geo_data[geo_data["Situation_Matrimoniale"].isin(Statut_Mat)] if len(Statut_Mat)!=0 else geo_data_dla 
        with col1:       
            make_chlorophet_map_folium_2(geo_data_dla,style_carte=style,palet_color=couleur,opacity=opacity,width=1000,height=650)
        with col2:
            geo_data_dla["Categorie_profession"]=geo_data_dla["Categorie_profession"].replace("Personnel des services directs aux particuliers, commercants vendeurs","commercants vendeurs")
            make_bar(geo_data_dla,"Categorie_profession",titre="Categorie Professionnelle",ordre=1,sens='h',height=400,width=600) 
            make_area_chart(data,var="Date_remplissage",titre="Evolution du nombre de candidat",color=1)

        
        cb1, cb2, cb3=st.columns(3)
        with cb1:
            make_cross_hist_b(geo_data_dla,"Eligibilite","Arrondissement",titre="",width=400,height=450,typ_bar=0)
        with cb2:
            make_donutchart(geo_data_dla,var="ancien_don_sang",titre="ancien donateur")
        with cb3:
            #make_cross_hist_3(geo_data_dla,"Niveau_etude","Age",titre="",agregation="avg",width=400,height=300)
            make_bar(geo_data_dla,var="Classe_age",titre="Répartition ages",width=700,height=450,)
            
#SECTION 3 : ANALYSE PAR ARRONDISSEMENT
    with st.expander("Analyse par arrondissement",expanded=False,icon="🩸"):    
        b11, b12, b13, b14,b15, b16 =st.columns([1,1.3,1.5,4.2,1.7,2.1])
        with b11:
            couleur_2=st.selectbox("Couleur",sequence_couleur)
        with b12:
            opacity_2=st.slider("Transparence",min_value=0.0,max_value=1.0,value=0.8,step=0.01)
        with b13:
            style_2=st.selectbox("Thme carte",options=["open-street-map","carto-positron","carto-darkmatter"])
        with b14:
            arrondissement=st.multiselect("Arrondissement",options=["Douala 1er","Douala 2e","Douala 3e","Douala 4e", "Douala 5e"],default=["Douala 1er"])
        with b15:
            last_don=st.multiselect("Filtre: Ancien donateur",options=data["Don_pass"].unique(),default=data["Don_pass"].unique())
        with b16:    
            genre2=st.multiselect(" Filtre: Genre",options=data["Genre"].unique(),default=data["Genre"].unique())
            
        geo_data_arr=geo_data[geo_data["Arrondissement"].isin(arrondissement)] if len(arrondissement)!=0 else geo_data
        geo_data_arr=geo_data_arr[geo_data_arr["ancien_don_sang"].isin(last_don)] if len(last_don)!=0 else geo_data_arr
        geo_data_arr=geo_data_arr[geo_data_arr["Genre"].isin(genre2)] if len(genre2)!=0 else geo_data_arr
        
        b1, b2 =st.columns([6.5,2.5])
        with b1:
            make_chlorophet_map_folium_2(geo_data_arr,style_carte=style_2,palet_color=couleur_2,opacity=opacity_2,width=1000,height=500)
        with b2:     
            geo_data_arr_for_table=geo_data_arr.groupby("Quartier").agg({
                "ID":"size"
            })
            geo_data_arr_for_table=geo_data_arr_for_table.sort_values("ID",ascending=False)
            geo_data_arr_for_table=geo_data_arr_for_table.rename(columns={ "ID": "Nb_Candidats"})
            geo_data_arr_for_table["Quartier"]=geo_data_arr_for_table.index
            make_dataframe(geo_data_arr_for_table,col_alpha="Quartier",col_num="Nb_Candidats",hide_index=True)
    
#SECTION 4 :  CONDITION DE SANTE ET ELIGIBILITE
    with st.expander("4. Conditions de Santé & Éligibilité", expanded=False,icon="❤️"):
        c41, c42 ,c43=st.columns(3)
        with c41:
            data_el=data.groupby("Eligibilité").agg({
                "ID":"size"
            })
            data_el=data_el.rename(columns={"ID":"Nb_Candidats"})
            data_el["Proportion"]=data_el["Nb_Candidats"]/float(data_el["Nb_Candidats"].sum())
            #st.dataframe(data_el[["Nb_Candidats"]])
            make_progress_char(data_el["Proportion"][1],couleur="rgba(" + str(255*(1-data_el["Proportion"][1])) + "," + str(255*data_el["Proportion"][1]) +",0,1)",titre="Taux d'éligibilité")
        with c42:
            statut_el=st.multiselect("Statut des candidats",options=data["Eligibilité"].unique(),default=data["Eligibilité"].unique())
            data_nl=data_el[data_el.index!="Eligible"]
            make_multi_progress_bar(labels=data_nl.index,values=data_nl["Proportion"],colors=["red","orange"],titre="Candidats Non éligible",width=500,height=300)
            #st.dataframe(data_nl)
        with c43:
            data_raison=data[data["Raisons"].notna()]
            data_raison = data_raison[data_raison["Eligibilité"].isin(statut_el)] if len(statut_el) != 0 else data_raison
            #st.dataframe(data_raison)
            raison=",".join(data_raison["Raisons"])
            raison=raison.split(",")
            r_ID=["ID"+str(i) for i in range(len(raison))]
            raison=pd.DataFrame({"ID":r_ID,"Raisons":raison})
            group_raison=raison.groupby("Raisons").agg({"ID":"size"})
            group_raison=group_raison.rename(columns={"ID":"Effectif"})
            group_raison=group_raison.sort_values("Effectif",ascending=False)
            make_dataframe(group_raison,col_alpha="Raisons",col_num="Effectif")
            #make_bar(raison,var="Raisons",color=px.colors.colorbrewer.BrBG_r,titre="Raison évoquées")
        c4b1, c4b2 ,c4b3=st.columns(3)
        with c4b1:
            make_hist_box(data_raison,var1="Tx_hémoglobine",var2="Eligibilité",height=400)
        with c4b2:
            data_mot=data[data["Autre_Raison"].notna()]
            mot=" ".join(data_mot["Autre_Raison"])
            #make_wordcloud(mot,titre="Autre raison",width=600,height=400)
            st.dataframe(data_el[["Nb_Candidats"]]) 
            make_distribution_2(data_raison,var_alpha="Genre",var_num="Tx_hémoglobine",add_vline=12,add_vline2=13,titre="Distribution du taux d'hémoglobine")
            #resized_image = image.resize((626, 200))  # Ajuster à 400x300 pixels
            #make_distribution(data_raison,var_alpha="Genre",var_num="Tx_hémoglobine",add_vline=12,add_vline2=13,titre="Distribution du taux d'hémoglobine")
            #make_relative_bar(data,"Genre","Eligibilité",titre="Repartition selon le genre et le statut d'éligibilité",width=600,height=400)
        with c4b3:
            make_cross_hist_b(data,"Eligibilité","Classe_age",titre="Statut par classe d'age",typ_bar=1)
        
#SECTION 5:   PROFILAGE DES DONNEURS IDEAUX
    with st.expander("5. Profilage des Donneurs Idéaux", expanded=False,icon="❤️"):
            c5a1,c5a2,c5a3=st.columns(3)
            with c5a1:
                make_relative_bar(data,var1="Eligibilité",var2="Don_pass",width=500,height=400,titre="Proportion des anciens donneurs",)
                data_SM=data.groupby("Situation_Mat").agg({"ID":"size"})
                data_SM["Proportion"]=data_SM["ID"]/float(data_SM["ID"].sum())
                make_multi_progress_bar(labels=data_SM.index, values=data_SM["Proportion"],colors=px.colors.qualitative.Vivid_r,width=500,height=400,titre="Taux d'éligibilité par statut Marital")
            with c5a2:
                tx_el_F=data[(data["Genre"]=="Femme") & (data["Eligibilité"]=="Eligible")].shape[0]/data[data["Genre"]=="Femme"].shape[0]
                tx_el_M=data[(data["Genre"]=="Homme") & (data["Eligibilité"]=="Eligible")].shape[0]/data[data["Genre"]=="Homme"].shape[0]
                make_dbl_progress_char(vars=[tx_el_M,tx_el_F],labels=["Homme","Femme"],titre="Taux d'éligibilité",colors=["green","orange"])
                data_Met=pd.crosstab(data["Good_profession"],data["Eligibilité"])
                data_Met["Total"]=data_Met.sum(axis=1)
                data_Met["Taux Eligibilité (%)"]=round((data_Met["Eligible"]/data_Met["Total"])*100,2) 
                data_Met=data_Met[["Taux Eligibilité (%)","Eligible","Temporairement Non-eligible","Définitivement non-eligible","Total"]]
                st.write("Eligibilité selon le groupe de métier")
                st.dataframe(data_Met)
            with c5a3:
                data_prof=data.groupby("Niveau_etude").agg({"ID":"size"})
                data_prof["Proportion"]=data_prof["ID"]/float(data_prof["ID"].sum())
                make_multi_progress_bar(labels=data_prof.index, values=data_prof["Proportion"],colors=px.colors.qualitative.Vivid,width=500,height=400,titre="Taux d'éligibilité par niveaux d'éducation")
                make_cross_hist_b(data,var2="Religion",var1="Eligibilité",width=500,height=550,titre="Réligion",typ_bar=1)
    
#SECTION 6:   ANALYSE DE L'EFFICACITE DE LA CAMPAGNE
    with st.expander("6. Analyse de l’Efficacité des Campagnes", expanded=True,icon="❤️"):
        c61,c62,c63=st.columns(3)
        data_don["ID"]="Don_" + (data_don.index+1).astype(str)
        data_don=data_don.rename(columns={"Groupe Sanguin ABO / Rhesus ":"Gpr_sang"})
        with c61:
            make_progress_char(data_don.shape[0]/data.shape[0],"green",titre="Efficacité de la compagne")
        with c62:
            
            data_don=data_don.set_index("ID",drop=True)
            data_don["ID"]=data_don.index
            data_don["Date"]=data_don["Horodateur"].dt.date
            data_don["Heure"]=data_don["Horodateur"].dt.hour
            trend_don=pd.crosstab(data_don["Date"],data_don["Sexe"])
            make_bar(data_don,var="Date",color=0,titre="Répartition des dons dans le temps",height=400)
            
            #st.dataframe(data_don)
        with c63:
            make_area_chart(data_don,var="Heure",titre="Heure d'affluence",height=400)
            #make_cross_hist_2(data_don,var1="Date",var2="Sexe",titre="Répartition des doneurs")
        c6a,c6b,c6c=st.columns(3)
        
        with c6a:
            #make_cross_hist_2(data_don,var1="Sexe",var2="Type de donation",titre="Profil des donateurs",typ_bar=1)
           make_relative_bar(data_don,var1="Gpr_sang",var2="Type de donation",titre="Répartition des donneurs par groupe sanguin ")
        with c6b:
            make_cross_hist_b(data_don,var1="Type de donation",var2="Classe_age",typ_bar=0,titre="répartition des type de don par classe d'age",bordure=7 )
        
        with c6c:
            make_donutchart(data_don,var="Phenotype ", titre="Différent type de phénotype des donneurs")
            #fig_pie=px.pie(data_don,"Type de donation", hole=0.5)  
            #st.plotly_chart(fig_pie)

        st.write("Taux de participation par classe d'age")
        st.write("Type de Donation")
        st.write("Liaison entre type de donnation et groupe sanguing puis phénotype")
        st.write("Evolution temporelle")
        st.write("Analyse des donateurs")

#ONGLET 3: TEST STATISTIQUES
with tabs[2]:
    st.write("Faite vos test statistiques")
    cc1,cc2=st.columns([3,7])
    with cc1:
        Test=st.selectbox("Choisissez le test à effectuer",["Test d'indépendance","Test de comparaison de la moyenne", "Test de comparaison de la dispersion"])
        var1=st.selectbox("Variable1 de test",options=var_qual )
        var2=st.selectbox("Variable2 de test",options=var_qual if Test=="Test d'indépendance" else var_quant)
    with cc2:
        if st.button("Lancer le test"): 
            if var1=="" or var2=="":
                st.write("Veuillez sélectionner des variables de test")
            else:
                if Test=="Test d'indépendance":
                    conclusion, table_cross,chi2, p,dof=test_independance_khi2(data,var1,var2)
                    st.write(conclusion)
                    st.dataframe(table_cross)
                    table_cross = table_cross.reset_index().melt(id_vars='index', var_name=var2, value_name='Effectif')
                    table_cross.rename(columns={'index': var1}, inplace=True)
                    #fig = px.bar(table_cross, x=var2, y='Effectif', color=var1, barmode='group', 
                                #title='Graphique à barres du tableau de contingence')
                    make_cross_hist_b(data,var2,var1,typ_bar=0)
                    make_relative_bar(data,var2,var1,height=600)
                    #st.plotly_chart(fig)
                elif Test=="Test de comparaison de la moyenne":
                    conclusion, graph= test_comparaison_moyenne(data, var1, var2)
                    st.write(conclusion)
                    st.plotly_chart(graph)
                else:
                    pass
    
    st.write("Afficher votre rapport ")
    telecharger_pdf("Challenge_Proposal_2.pdf")

# ONGLET 4: FORMULAIRE
with tabs[3]:
    #Création des dictionnaire de localisation et de métier:
    df_new = pd.read_excel("Infos.xlsx",sheet_name="New_Base") #chargement de la base des nouveaux candidats
    df_ctrl = pd.read_excel("Infos.xlsx",sheet_name="Info") #chargement des informations de controle
    
    quartier_dict = {}
    for index, row in df_info_loc.iterrows():
        quartier = row['Quartier_de_Résidence']
        good_qrt=row["Good_Quartier"]
        arrondissement = row['Good_Arrondissement']
        lat = row['Lat']
        long = row['Long']
        values = [arrondissement, lat, long,good_qrt]
        quartier_dict[quartier] = values
        
    profession_dict = {}
    profession_dict = dict(zip(df_info_pof['Profession'], df_info_pof['Good_profession']))
    #Formulaire   
    with st.form(key="formulaire_eligibilite"):
        # Informations de base (obligatoires)
        st.subheader("Informations générales")
        col1, col2 = st.columns(2)
        
        with col1:
            sexe = st.radio("Sexe", options=["M", "F"], index=0, 
                            format_func=lambda x: "Masculin" if x == "M" else "Féminin")
            age = st.number_input("Âge (années)", min_value=16, max_value=80, value=30, step=1)
        
        with col2:
            poids = st.number_input("Poids (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.5)
            derniere_donation_options = ["Jamais donné", "Plus de 3 mois", "Plus de 2 mois", "Dans les 2 derniers mois"]
            derniere_donation_choix = st.selectbox("Dernière donation", options=derniere_donation_options)
            
            # Conversion du choix en jours
            if derniere_donation_choix == "Jamais donné":
                derniere_donation = 1000  # Valeur arbitraire élevée
            elif derniere_donation_choix == "Plus de 3 mois":
                derniere_donation = 91
            elif derniere_donation_choix == "Plus de 2 mois":
                derniere_donation = 61
            else:
                derniere_donation = 30
        
        # Informations socio-démographiques (nouvelles)
        st.subheader("Informations socio-démographiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Niveau d'étude (bouton radio)
            niveau_etude = st.radio(
                "Niveau d'étude", 
                options=["Aucun","Primaire", "Secondaire", "Universitaire 1er cycle", "Universitaire 2e cycle", "Universitaire 3e cycle"],
                index=2
            )
            
            # Statut matrimonial (liste de choix)
            statut_matrimonial = st.selectbox("Statut matrimonial",
                options=["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/Veuve", "Union libre"])
            
            # Religion
            religion = st.selectbox("Religion",options=All_religion)
        
        with col2:
            # Profession
            profession = st.selectbox("Profession",options=metier,)
            # Quartier de résidence (liste déroulante)
            quartier = st.selectbox("Quartier de résidence",options=all_Quartier)
            
            # Nationalité (pays africain)
            nationalite_options = [
                "Cameroun", "Nigeria", "Sénégal", "Côte d'Ivoire", "Ghana", "Bénin", 
                "Tchad", "République Centrafricaine", "Gabon", "Congo", "RDC", 
                "Autre pays africain", "Autre pays hors Afrique"
            ]
            nationalite = st.selectbox(
                "Nationalité",
                options=nationalite_options,
                index=0
            )
        
        # Critères spécifiques aux femmes
        if sexe == "F":
            st.subheader("Informations spécifiques (femmes)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                grossesse_recente = st.checkbox("Grossesse récente")
                if grossesse_recente:
                    temps_depuis_grossesse = st.number_input("Temps depuis l'accouchement (mois)", 
                                                             min_value=0, max_value=24, value=3)
                else:
                    temps_depuis_grossesse = None
                
                allaitement = st.checkbox("Allaitement en cours")
                
            with col2:
                en_periode_menstruelle = st.checkbox("Actuellement en période menstruelle")
                cycle_menstruel_irregulier = st.checkbox("Cycle menstruel irrégulier")
                saignements_anormaux = st.checkbox("Saignements anormaux")
        else:
            # Valeurs par défaut pour les hommes
            grossesse_recente = None
            temps_depuis_grossesse = None
            allaitement = None
            en_periode_menstruelle = None
            cycle_menstruel_irregulier = None
            saignements_anormaux = None
        
        # Critères médicaux
        st.subheader("Informations médicales")
        
        # Maladies chroniques
        maladies_selections = st.multiselect(
            "Sélectionnez vos conditions médicales",
            options=list(df_ctrl["Maladie"].dropna())
        )
        
        # Si "Aucune maladie" est sélectionné et d'autres options aussi, on enlève "Aucune maladie"
        if "Aucune maladie chronique" in maladies_selections and len(maladies_selections) > 1:
            maladies_selections.remove("Aucune maladie chronique")
        
        # Conversion pour la fonction
        if "Aucune maladie chronique" in maladies_selections or not maladies_selections:
            maladies_chroniques = None
        else:
            maladies_chroniques = maladies_selections
        
        # Médicaments
        medicaments_selections = st.multiselect(
            "Sélectionnez les médicaments que vous prenez actuellement",
            options=list(df_ctrl["Traitement"].dropna())
        )
        
        # Si "Aucun médicament" est sélectionné et d'autres options aussi, on enlève "Aucun médicament"
        if "Aucun médicament" in medicaments_selections and len(medicaments_selections) > 1:
            medicaments_selections.remove("Aucun médicament")
        
        # Conversion pour la fonction
        if "Aucun médicament" in medicaments_selections or not medicaments_selections:
            medicaments = None
        else:
            medicaments = medicaments_selections
        
        # Autres critères
        col1, col2 = st.columns(2)
        
        with col1:
            interventions_recentes = st.checkbox("Intervention chirurgicale récente")
            if interventions_recentes:
                temps_depuis_intervention = st.number_input("Temps depuis l'intervention (jours)", 
                                                           min_value=0, max_value=365, value=30)
            else:
                temps_depuis_intervention = None
        
        with col2:
            tatouages_recents = st.checkbox("Tatouage ou piercing récent (moins de 4 mois)")
        
        # Bouton de soumission
        submit_button = st.form_submit_button(label="Évaluer mon éligibilité")
    
    # Traitement des données après soumission
    if submit_button:
        # Appel de la fonction d'évaluation
        resultat = verifier_eligibilite_don_sang(
            sexe=sexe,
            age=age,
            poids=poids,
            derniere_donation=derniere_donation,
            grossesse_recente=grossesse_recente,
            temps_depuis_grossesse=temps_depuis_grossesse,
            allaitement=allaitement,
            en_periode_menstruelle=en_periode_menstruelle,
            cycle_menstruel_irregulier=cycle_menstruel_irregulier,
            saignements_anormaux=saignements_anormaux,
            maladies_chroniques=maladies_chroniques,
            medicaments=medicaments,
            interventions_recentes=interventions_recentes,
            temps_depuis_intervention=temps_depuis_intervention,
            tatouages_recents=tatouages_recents,
            # Ajout des nouvelles informations socio-démographiques
            #niveau_etude=niveau_etude,
            #statut_matrimonial=statut_matrimonial,
            #profession=profession,
            #quartier=quartier,
            #religion=religion,
            #nationalite=nationalite
        )
       
        # Affichage des résultats
        st.subheader("Résultat de l'évaluation")
        
        if resultat["eligible"]:
            st.success("✅ Vous êtes éligible pour passer aux examens approfondis pour le don de sang. Veuillez prendre un rendez-vous")
            statut="Temporairement éligible"
        elif (medicaments!=None) | (maladies_chroniques!=None) | tatouages_recents :
            st.error("❌ Vous n'êtes pas éligible pour le don de sang. Merci pour l'élant de coeur dont vous avez fait preuve")
            statut="Non-éligible"
        else:
            st.error("❌ Vous n'êtes pas éligible pour le don de sang actuellement.")
            statut="Temporairement non-éligible"
        
        # Affichage des raisons
        st.subheader("Détails:")
        for raison in resultat["raisons"]:
            st.write(f"- {raison}")
        
        # Affichage des recommandations si présentes
        if resultat["recommandations"]:
            st.subheader("Recommandations:")
            for recommandation in resultat["recommandations"]:
                st.write(f"- {recommandation}")
        
        # Avertissement 
        st.info("⚠️ Cette évaluation est indicative et ne remplace pas l'avis d'un professionnel de santé. Veuillez consulter le personnel médical du centre de don pour une évaluation définitive.")
        
        #Fonction d'enregistrement d'un nouveau formulaire.
        def save(df):
                nouveau_id="CDT_" + str(df.shape[0]+1)
                stdcd=",".join(resultat["raisons"])
                nouvelle_ligne = {
                        "ID": nouveau_id,
                        "Date_remplissage": datetime.datetime.now(),
                        "sexe": sexe,
                        "age": age,
                        "poids": poids,
                        "statut_matrimonial": statut_matrimonial,
                        "niveau_etude": niveau_etude,
                        "profession": profession,
                        "Catégorie_Professionnelle":profession_dict[profession],
                        "quartier": quartier_dict[quartier][3],
                        "Arrondissement":quartier_dict[quartier][0] ,
                        "Lat":quartier_dict[quartier][1] ,
                        "Long":quartier_dict[quartier][2] ,
                        "religion": religion,
                        "nationalite": nationalite,
                        "derniere_donation": derniere_donation,
                        "Statut":statut ,
                        "Raison":stdcd if statut=="Temporairement non-éligible" or statut=="Non-éligible"  else None,
                    }
                    
                        # Ajouter les données au DataFrame
                df = pd.concat([df, pd.DataFrame([nouvelle_ligne])], ignore_index=True)         
                try:
                    # Sauvegarder dans Excel
                    with pd.ExcelWriter("Infos.xlsx", engine='openpyxl', mode='a', 
                                                if_sheet_exists='replace') as writer:
                            df.to_excel(writer, sheet_name="New_Base", index=False)
                    st.success("Les informations ont été enregistrées avec succès!")
                except Exception as e:
                        st.error(f"Erreur lors de l'enregistrement: {e}")

        save(df_new)    
 
# ONGLET 5: TABLEAU DE BORD DE LA NOUVELLE CAMPAGNE
with tabs[4]:
    #==================================================================================================================
    mise_a_ajour=st.button("Mettre à Jour le Tableau de bord")
    if mise_a_ajour:
        forme_dla=geo_data.groupby("Arrondissement").agg({"geometry":"first"}) #Récupération des formes des arrondissemnt de Douala
        New_geo_data=pd.merge(forme_dla,df_new,on="Arrondissement", how="inner") #Jointure pour obtenir des données spatialisée des nouveau candidats
        New_geo_data=New_geo_data.set_index("ID")
        New_geo_data=gpd.GeoDataFrame(New_geo_data, geometry='geometry') # converssion du nouveau fichier en geodataframe, pour des analyse spatiale
        
        ab1,ab2,ab3=st.columns(3)
        
        with ab1:
            plot_metric("Total Individu",df_new.shape[0],prefix="",suffix="",show_graph=True,color_graph="rgba(0, 104, 201, 0.1)",)
        with ab2:
            plot_metric_2("Age moyen des donneurs",df_new,"age",prefix="",suffix=" ans",show_graph=True,color_graph="rgba(175, 32, 201, 0.2)",val_bin=45)
        
        with st.expander("Information Globales sur les Nouvea candidat", expanded=True):
            c1, c2=st.columns([4,2])
            with c1:
                make_map_folium(New_geo_data, style_carte="OpenStreetMap", palet_color="reds", opacity=0.8, width=900, height=600)
            with c2:
                make_cross_hist_b(df_new,var2="Arrondissement",var1="Statut",titre="Répartition des candidats par arrondissement",bordure=9,width=600)
            ca1,ca2,ca3,ca4=st.columns(4)
            with ca1:
                make_donutchart(df_new,var="sexe",titre="Répartiton des candidats par sexe")
            with ca2:
                make_cross_hist_b(df_new,var2="statut_matrimonial",var1="Statut",titre="Statut matrimonial des candidats",typ_bar=0,bordure=7)
            with ca3:
                make_cross_hist_b(df_new,var2="religion",var1="Statut",bordure=12)
            with ca4:
                make_donutchart(df_new,var="Statut")
# ONGLET 3:
with tabs[5]:
    pass
        

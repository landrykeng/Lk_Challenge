
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


#================Cnfiguration des styles de la page ===========================
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

lang = set_language()

#Dictionnaire de traduction
translations = {
    "Français": {
        "title": "Tableau de bord de la campagne de don de sang",
        "sub_title": "Exploiter les données pour une meilleure gestion et planification des dons de sang",
        "teacher": "🎓 Enseignant: Mr Serge Ndoumin",
        "analysis": "📊 Analyse des données du e-commerce au Pakistan",
        "raw_data": "📂 Données Brutes",
        "raw_data_desc": "📌 Base brute sans traitement incluant les valeurs atypiques",
        "data_issues": "🔍 Description des imperfections des données",
        "stats_desc": "📈 Statistiques descriptives de la base",
        "processed_data": "⚙️ Données Traitées",
        "visualization": "📊 Visualisation des Indicateurs",
        "modeling": "🤖 Modélisation en Bonus",
        "group_members": "👥 Membres du Groupe",
        "rapport":"Produire un rapport",
    },
    "English": {
        "title": "Blood Donation Campaign Dashboard",
        "sub_title": "Harnessing data for better management and planning of blood donations",
        "teacher": "🎓 Instructor: Mr. Serge Ndoumin",
        "analysis": "📊 Analysis of E-commerce Data in Pakistan",
        "raw_data": "📂 Raw Data",
        "raw_data_desc": "📌 Raw dataset without processing, including outliers",
        "data_issues": "🔍 Description of imperfections in the raw data",
        "stats_desc": "📈 Descriptive statistics of the dataset",
        "processed_data": "⚙️ Processed Data",
        "visualization": "📊 Indicator Visualization",
        "modeling": "🤖 Bonus Modeling",
        "group_members": "👥 Group Members",
        "rapport":"Produce repport",
    }
}

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

st.image("Image.png", use_container_width=True)  

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


@st.cache_data(persist="disk") #pour garder le cache entre les sessions
def load_data(excel_path, geodata_path):
    try:
        data = pd.read_excel(excel_path)
        geo_data = gpd.read_file(geodata_path)
        return data, geo_data
    except FileNotFoundError as e:
        st.error(f"Erreur : Fichier introuvable - {e}")
        return None, None
    except Exception as e:
        st.exception(f"Une erreur s'est produite : {e}")
        return None, None


data , geo_data = load_data("Challenge dataset_Alpha.xlsx","GeoData.shp")
geo_data = gpd.GeoDataFrame(geo_data, geometry='geometry')
geo_data.columns=['Arrondissement', 'ADM3_PCODE', 'Date_remplissage', 'Date_naissance', 'Niveau_etude',
       'Age', 'Genre', 'Taille', 'Poids', 'Situation_Matrimoniale', 'Profession', 'Ville',
       'Quartier', 'Lat', 'Long', 'Nationalité', 'Religion', 'ancien_don_sang',
       'date_dernier_don', 'Ecart_dernier_don', 'Taux_hémoglobine_(g/dl)', 'Eligibilite', 'Nb_Raison',
       'Raisons', 'Autre_Raisons', 'geometry']

#geo_data=geo_data.rename(columns={'ÉLIGIBILITÉ AU DON.': 'Eligibilite'})
#==========================================================
#====================== EN TETE ===========================
#==========================================================
#geo_data["valeur"]=[15,36,75,10,25,52]
#geo_data.set_index('ADM3_FR')
#Contenu de la barre latérale
st.sidebar.image("Logo.png", use_container_width=True)
st.sidebar.title(translations[lang]["group_members"])
members = [
    "ASSADICK IBNI Oumar Ali", "ATANGANA TSIMI Arsène Joël", "HUSKEN TIAWE Alphonse",
    "KENGNE Bienvenu Landry", "MAGUETSWET Rivalien", "MIKOUIZA BOUNGOUDI Jeanstel Hurvinel",
    "NOFOZO YIMFOU Sylvain", "YAKAWOU Komlanvi Eyram", "YALIGAZA Edson Belkrys De-Valor"
]
for member in members:
    st.sidebar.markdown(f"✅ {member}")

#==========================================================
#==== FONCTION USUELLES ===================================
#==========================================================
#1. Fonction d'affichage des métriques
def display_single_metric_advanced(label, value, delta, unit="", caption="", color_scheme="blue"):
    """Affiche une seule métrique avec un style avancé et personnalisable."""

    color = {
        "blue": {"bg": "#e6f2ff", "text": "#336699", "delta_pos": "#007bff", "delta_neg": "#dc3545"},
        "green": {"bg": "#e6ffe6", "text": "#28a745", "delta_pos": "#28a745", "delta_neg": "#dc3545"},
        "red": {"bg": "#ffe6e6", "text": "#dc3545", "delta_pos": "#28a745", "delta_neg": "#dc3545"},
    }.get(color_scheme, {"bg": "#f0f0f0", "text": "#333", "delta_pos": "#28a745", "delta_neg": "#dc3545"})

    delta_color = "green" if delta >= 0 else "red"

    st.markdown(
        f"""
        <div style="background-color: {color['bg']}; padding: 2px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color: {color['text']}; margin-bottom: 1.5px;">{label}</h4>
            <div style="font-size: 1.5em; font-weight: bold; color: {color['text']};">{value} {unit}</div>
            <div style="font-size: 1em; color: {delta_color};">{'▲' if delta >= 0 else '▼'} {abs(delta)}  {"-----"}  {'▲' if delta >= 0 else '▼'} {abs(delta)}</div>
            <p style="font-size: 1em; color: {color['text']};">{caption}{"-----"}{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

#2. Fonction de test d'indépendance de Khi 2
def test_independance_khi2(df, var1, var2):
    # Création de la table de contingence
    contingency_table = pd.crosstab(df[var1], df[var2])
    index_labels = list(df[var1].unique())
    table_cross = pd.DataFrame(contingency_table, index=index_labels)
    # Application du test Khi-2
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Conclusion
    if p < 0.05:
        conclusion = "Il y a une association significative entre les variables."
    else:
        conclusion = "Les variables sont indépendantes."
    
    
    
    # Retour des résultats
    return  conclusion, table_cross,chi2, p,dof
  
#3. Fonction de production de carte
def make_chlorophet_map(df,style_carte="carto-positron",palet_color="Blues",opacity=0.8):
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
        'geometry': 'first'
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
        colorbar_title="Nombre de donateurs",  # Titre de la barre de couleur
        hovertext=df_chlph['Arr'],
        hovertemplate=" %{hovertext}  <br>Nombre de donateurs : %{z}<extra></extra>",
    ))

    # Ajout des points pour représenter le nombre de donateurs
    fig.add_trace(go.Scattermapbox(
        lat=df_pts["Lat"],  # Colonne des latitudes des points
        lon=df_pts["Long"],  # Colonne des longitudes des points
        mode='markers',  # Mode de dispersion (points)
        name="Total donateurs",
        marker=dict(
            size=df_pts["nb_donateur"],  # Taille des points basée sur 'nb_donateur'
            sizemode='area',  # La taille est proportionnelle à la surface
            sizeref=2. * max(df_pts["nb_donateur"]) / (45.**2),  # Ajustement de la taille
            color='#003F80',  # Couleur des points
            opacity=0.8  # Opacité des points
        ),
        hovertemplate=(
            "<b>Quartier :</b> %{text}<br>"
            "<b>Total donateurs :</b> %{marker.size}<extra></extra>"
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
            "<b> Donateurs Eligibles :</b> %{marker.size}<extra></extra>"
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
            "<b>donateurs Temporairement Non-eligibles :</b> %{marker.size}<extra></extra>"
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
            "<b> Donateur sNon-eligibles :</b> %{marker.size}<extra></extra>"
        ), 
        text=df_pts_NE["Qrt"]
    ))

    # Mise à jour de la mise en page de la carte
    fig.update_layout(
        showlegend=True,
        legend=dict(
            x=0.2,  # Centré horizontalement
            y=-0.1,  # Sous la carte
            orientation='h',  # Légende horizontale
        ),
        mapbox=dict(
            #style="open-street-map",
            style=style_carte,  # Style de la carte
            #style="carto-darkmatter",
            center=dict(lat=df_pts["Lat"].mean(), lon=df_pts["Long"].mean()),  # Centrer la carte
            zoom=10  # Niveau de zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0)  # Marges de la carte
    )

    st.plotly_chart(fig)

#3. Fonction de test de comparaison de la moyenne
def test_comparaison_moyenne(df, var1, var2):
    # Séparation des groupes
    groupe1 = df[df[var1] == 1]  
    groupe2 = df[df[var1] == 0]
    # Test de Student pour comparer les moyennes
    t_stat, p_value = ttest_ind(groupe1[var2], groupe2[var2])
    # Affichage des résultats
    #print(f"Statistique t : {t_stat}")
    #print(f"Valeur p : {p_value}")
    
    # Conclusion
    if p_value < 0.05:
        result="Les moyennes des deux groupes sont significativement différentes."
    else:
        result="Les moyennes des deux groupes ne sont pas significativement différentes."
    return result

#3. Fonction de calibrage de la carte
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
    translations[lang]["data_issues"], 
    translations[lang]["processed_data"],
    translations[lang]["visualization"],
    translations[lang]["rapport"]
])

with tabs[0]:
    st.markdown(f"**{translations[lang]['raw_data_desc']}**")
    st.dataframe(data)
    st.write("Données géospatialisée")
    st.dataframe(geo_data)
with tabs[1]:
    pass

with tabs[2]:
    pass

with tabs[3]:
    a1, a2, a3 = st.columns(3)
    
    with a1:
        display_single_metric_advanced("Taux de chômage Moyen", 15, 3, unit="%", caption="Maximun", color_scheme="red")
    with a2:
        display_single_metric_advanced("Taux de chômage Moyen", 45.6, -1.5, unit="%", caption="Maximun", color_scheme="blue")
    with a3:
        display_single_metric_advanced("Taux de chômage Moyen", 125, 15, unit="t", caption="Maximun", color_scheme="green")
    col1, col2 = st.columns([5, 2])
    
    
    with col2:
        #with st.expander("Tableau", expanded=True):
        #with st.expander("Carte de la ville de Douala", expanded=True):
        data_ex = {'Category': ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L'], 'Value': [25, 15, 30, 30,25.6,48,69,45,78,10,13,35]}
        df_ex = pd.DataFrame(data_ex)
        opacity=st.slider("Ajuster la transparence de votre carte chlorophete",min_value=0.0,max_value=1.0,value=0.8,step=0.01)
        style=st.selectbox("Type de carte",options=["open-street-map","carto-positron","carto-darkmatter"])
        st.dataframe(df_ex, hide_index=True)
    
    with col1:       
        # Charger les données géographiques
        make_chlorophet_map(geo_data,style_carte=style,palet_color="Greens",opacity=opacity)
        
        
 
    cb1, cb2, cb3=st.columns(3)
    with cb1:
        with   st.expander("Graph", expanded=True):
            st.write("Graphique type histogramme")
    with cb2:
        with st.expander("Graph", expanded=True):
            st.write("Graphique type Time Series")
    with cb3:
        with st.expander("Graph", expanded=True):
            st.write("Graphique type Area")       
            
    with st.expander("Indicateur pertinent", expanded=True):
        st.write("ecrire ici quelques résultats pertinents")
        
    c1, c2, c3=st.columns(3)
    with c1:
        with   st.expander("Graph", expanded=True):
            st.write("graphique")
    with c2:
        with st.expander("Graph", expanded=True):
            st.write("graphique")
    with c3:
        with st.expander("Graph", expanded=True):
            st.write("graphique")
    
    ca1, ca2=st.columns([2,1])
    with ca1:
        with st.expander("Analyse Par arrondissement", expanded=True):
            st.write("Carte")
    with ca2:
        with st.expander("Tableau relatif à la carte", expanded=True):
            st.write("graphique")

with tabs[4]:
    st.write("Faite vos test statistiques")
    cc1,cc2=st.columns([3,7])
    with cc1:
        Test=st.selectbox("Choisissez le test à effectuer",["Test d'indépendance","Test de comparaison de la moyenne", "Test de comparaison de la dispersion"])
        var1=st.selectbox("Variable1 de test",options=["Niveau d'etude", 'Genre', 'Taille', 'Poids',
        'Situation Matrimoniale (SM)', 'Profession',
        'Arrondissement de résidence', 'Ville', 'Quartier de Résidence',
        'Nationalité', 'Religion', 'A-t-il (elle) déjà donné le sang',
        'Si oui preciser la date du dernier don.', "Taux d’hémoglobine",
        'ÉLIGIBILITÉ AU DON.',])
        
        var2=st.selectbox("Variable2 de test",options=["Niveau d'etude", 'Genre', 'Taille', 'Poids',
        'Situation Matrimoniale (SM)', 'Profession',
        'Arrondissement de résidence', 'Ville', 'Quartier de Résidence',
        'Nationalité', 'Religion', 'A-t-il (elle) déjà donné le sang',
        'Si oui preciser la date du dernier don.', "Taux d’hémoglobine",
        'ÉLIGIBILITÉ AU DON.',])
    with cc2:
        if st.button("Lancer le test"): 
            if var1=="" or var2=="":
                st.write("Veuillez sélectionner des variables de test")
            else:
                conclusion, table_cross,chi2, p,dof=test_independance_khi2(data,var1,var2)
                st.write(conclusion)
                st.dataframe(table_cross)
                
            table_cross = table_cross.reset_index().melt(id_vars='index', var_name=var2, value_name='Effectif')
            table_cross.rename(columns={'index': var1}, inplace=True)
            fig = px.bar(table_cross, x=var2, y='Effectif', color=var1, barmode='group', 
                        title='Graphique à barres du tableau de contingence')
            st.plotly_chart(fig)
    
    st.write("Afficher votre rapport ")
    telecharger_pdf("Challenge_Proposal_2.pdf")

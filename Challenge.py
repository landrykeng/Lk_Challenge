#==========IMPORTATION DES BIBLIOTHEQUES NECESSAIRES===================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import traceback
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
from Fonction_Classement import *
from PIL import Image
from pathlib import Path
import json
from streamlit_echarts import st_echarts
#==================================================================================================

st.set_page_config(layout="wide")

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

# Pour l'affichage
st.markdown("""
<style>
    .main .block-container {
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Pour les formulaires et éléments de connexion */
    input[type="text"], input[type="password"] {
        width: 100%;
    }
    
    /* Si vous utilisez des colonnes ou des conteneurs personnalisés */
    .custom-container, .stTabs {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .auth-container {
        width: 100% !important;  /* ou 100% selon vos besoins */
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

#===================================

st.markdown("""
            <style>
            /* Quand la sidebar est fermée */
            [data-testid="stSidebar"][aria-expanded="false"] {
                width: 0;
                min-width: 0;
                overflow: hidden;
                transition: width 0.3s ease;
            }
            
            /* Extension complète du contenu principal quand sidebar fermée */
            [data-testid="stSidebar"][aria-expanded="false"] + div [data-testid="stAppViewContainer"] {
                max-width: 100% !important;
                padding: 0 !important;
            }
            
            /* Graphiques en plein écran */
            [data-testid="stSidebar"][aria-expanded="false"] + div .stPlotlyChart {
                width: 100% !important;
                max-width: 100% !important;
            }
            
            /* Conteneurs étendus */
            [data-testid="stSidebar"][aria-expanded="false"] + div [data-testid="stBlock"] {
                width: 100% !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            
            /* Style de la sidebar quand ouverte */
            [data-testid="stSidebar"][aria-expanded="true"] {
                width: 250px !important;
                min-width: 250px !important;
                transition: width 0.3s ease;
            }
            
            /* Ajustements généraux */
            .stPlotlyChart {
                width: 100%;
                max-width: 100%;
            }
            </style>
            """, unsafe_allow_html=True)

        #pour les conteneurs de graphiques

st.markdown("""
                <style>
                /* Styles de base pour tous les thèmes */
                .stContainer {
                    border-radius: 10px;  /* Coins arrondis */
                    border: 2px solid transparent;  /* Bordure transparente par défaut */
                    padding: 20px;  /* Espacement intérieur */
                    margin-bottom: 20px;  /* Espace entre les conteneurs */
                    transition: all 0.3s ease;  /* Animation douce */
                }

                /* Mode Clair (par défaut) */
                body:not(.dark) .stContainer {
                    background-color: rgba(255, 255, 255, 0.9);  /* Fond blanc légèrement transparent */
                    border-color: rgba(224, 224, 224, 0.7);  /* Bordure grise légère */
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);  /* Ombre douce */
                }

                /* Mode Sombre */
                body.dark .stContainer {
                    background-color: rgba(30, 30, 40, 0.9);  /* Fond sombre légèrement transparent */
                    border-color: rgba(60, 60, 70, 0.7);  /* Bordure sombre légère */
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);  /* Ombre plus marquée */
                }

                /* Effet de survol - Mode Clair */
                body:not(.dark) .stContainer:hover {
                    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.5);  /* Ombre plus prononcée */
                    transform: translateY(-5px);  /* Léger soulèvement */
                    border-color: rgba(200, 200, 200, 0.9);  /* Bordure plus visible */
                }

                /* Effet de survol - Mode Sombre */
                body.dark .stContainer:hover {
                    box-shadow: 0 8px 12px rgba(255, 255, 255, 0.5);  /* Ombre claire */
                    transform: translateY(-5px);  /* Léger soulèvement */
                    border-color: rgba(100, 100, 110, 0.9);  /* Bordure plus visible */
                }

                /* Style spécifique pour les graphiques - Mode Clair */
                body:not(.dark) .stPlotlyChart {
                    background-color: rgba(250, 250, 250, 0.95);  /* Fond très légèrement gris */
                    border-radius: 8px;  /* Coins légèrement arrondis */
                    padding: 10px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);  /* Ombre très légère */
                }

                /* Style spécifique pour les graphiques - Mode Sombre */
                body.dark .stPlotlyChart {
                    background-color: rgba(40, 40, 50, 0.95);  /* Fond sombre légèrement transparent */
                    border-radius: 8px;  /* Coins légèrement arrondis */
                    padding: 10px;
                    box-shadow: 0 2px 4px rgba(255, 255, 255, 0.05);  /* Ombre très légère */
                }
                </style>
                """, unsafe_allow_html=True)

useless_style="""
        <style>
        .sidebar-link {
            display: block;
            margin-bottom: 15px;
            padding: 10px 15px;
            text-decoration: none;
            color: #333;
            background-color: #f8f9fa;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .sidebar-link-right {
            display: block;
            margin-bottom: 15px;
            padding: 10px 15px;
            text-decoration: none;
            color: #333;
            background-color: #f8f9fa;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
            text-align: right;
        }

        .sidebar-link-center {
            display: block;
            margin-bottom: 15px;
            padding: 10px 15px;
            text-decoration: none;
            color: #333;
            background-color: #f8f9fa;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
            text-align: center;
        }
        </style>
        """

sidebar_css = """
        <style>
        .sidebar-link {
            display: block;
            margin-bottom: 15px;
            padding: 10px 15px;
            text-decoration: none;
            color: #333;
            background-color: #f8f9fa;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .sidebar-link:hover {
            background-color: #e9ecef;
            color: #007bff;
            transform: translateX(5px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .sidebar-link-icon {
            margin-right: 10px;
        }
        </style>
        """
        
table_css = """
        <style>
        /* Style général des tableaux */
        .stDataFrame {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }

        /* En-tête du tableau */
        .stDataFrame thead {
            background-color: #4b8bff;
            color: white;
            font-weight: bold;
        }

        /* Lignes du tableau */
        .stDataFrame tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        .stDataFrame tbody tr:nth-child(odd) {
            background-color: #ffffff;
        }

        /* Effet de survol */
        .stDataFrame tbody tr:hover {
            background-color: #e9ecef;
            transition: background-color 0.3s ease;
        }

        /* Cellules */
        .stDataFrame th, .stDataFrame td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }

        /* Style des colonnes */
        .stDataFrame th {
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
        }
        </style>
        """

title_css = """
        <style>
        .dashboard-title-container {
            background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
            color: white;
            padding: 30px 20px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }

        .dashboard-title-container:hover {
            transform: scale(1.02);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }

        .dashboard-main-title {
            font-size: 2.5em;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .dashboard-subtitle {
            font-size: 1.2em;
            font-weight: 300;
            color: rgba(255,255,255,0.9);
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
        }

        .title-icon {
            margin: 0 15px;
            opacity: 0.8;
        }
        </style>
        """

header_css = """
        <style>
        .header-container {
            background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .header-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255,255,255,0.1);
            transform: skew(-15deg) rotate(-15deg);
            z-index: 1;
        }

        .header-title {
            color: white;
            font-size: 2.5em;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 2;
        }

        .header-subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1.2em;
            font-weight: 300;
            max-width: 800px;
            margin: 0 auto;
            text-align: center;
            line-height: 1.6;
            position: relative;
            z-index: 2;
        }

        .image-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }

        .image-wrapper {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .image-wrapper:hover {
            transform: scale(1.03);
        }
        </style>
        """

tabs_css="""
<style>
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: 
#f0f2f6;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 15px;
    margin: 0 5px;
    border-radius: 10px;
    transition: all 0.3s ease;
    font-weight: 500;
    color: 
#4a4a4a;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(75, 139, 255, 0.1);
    color: 
#4b8bff;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: 
#4b8bff;
    color: white;
    box-shadow: 0 4px 6px rgba(75, 139, 255, 0.3);
}

.stTabs [data-baseweb="tab"] svg {
    margin-right: 8px;
}
</style>
"""

global_font_css = """
        <style>
        /* Définit la taille de police par défaut pour toute la page */
        body, .stMarkdown, .stTextInput>div>div>input, .stSelectbox>div>div>select, 
        .stMultiSelect>div>div>div, .stDateInput>div>div>input, 
        .stNumberInput>div>div>input, .stTextArea>div>div>textarea {
            font-size: 19px !important; /* Taille de police de base */
        }

        /* Styles pour différents types de texte */
        h1 { font-size: 2.5em !important; }  /* Titres principaux */
        h2 { font-size: 2em !important; }    /* Sous-titres */
        h3 { font-size: 1.5em !important; }  /* Titres de section */
        p, div, span { font-size: 19px !important; } /* Texte de paragraphe */

        /* Option pour ajuster la taille de police de manière responsive */
        @media (max-width: 600px) {
            body, .stMarkdown {
                font-size: 14px !important;
            }
        }
        </style>
        """

profile_css = """
        <style>
        .profile-container {
            background-color: #1e2736;
            border-radius: 15px;
            padding: 20px;
            color: white;
            display: flex;
            align-items: center;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            max-width: 600px;
            margin: 20px auto;
        }

        .profile-image {
            width: 150px;
            height: 150px;
            object-fit: cover;
            margin-right: 20px;
            border-radius: 10px; /* Légèrement arrondi si souhaité */
        }

        .profile-content {
            flex-grow: 1;
        }

        .profile-name {
            font-size: 1.8em;
            color: #4b8bff;
            margin-bottom: 5px;
        }

        .profile-title {
            font-size: 1em;
            color: #a0a0a0;
            margin-bottom: 10px;
        }
        </style>
        """

button_style = """
            <style>
            div[data-baseweb="segmented-control"] > div {
                background-color: #f0f2f6;  /* Couleur de fond */
                border-radius: 10px;  /* Coins arrondis */
                padding: 5px;
            }
            
            div[data-baseweb="segmented-control"] button {
                color: white !important;  /* Couleur du texte */
                background-color: #4CAF50 !important;  /* Couleur de fond des boutons */
                border-radius: 8px !important;  /* Arrondi des boutons */
                padding: 10px 20px !important;  /* Espacement interne */
                font-weight: bold !important;
            }

            div[data-baseweb="segmented-control"] button:hover {
                background-color: #45a049 !important;  /* Couleur au survol */
            }
            </style>
            """

st.markdown(global_font_css, unsafe_allow_html=True)
        #=======================================================================
        #================== Sélecteur de langue ================================


def set_language():
    return st.sidebar.selectbox("🌍 Choisissez la langue / Choose the language", ["", "Français", "English"])

# Exemple d'utilisation dans votre application principale
def main():
    # Appliquer le CSS personnalisé pour les onglets
    
    # Vérifier l'authentification
    st.write("Compte par défaut")
    st.write("USER: IndabaX user")
    st.write("PASSWORD: IndabaX 2025")
    st.markdown(tabs_css, unsafe_allow_html=True)
    is_authenticated = authentication_system()
    
    # Si l'utilisateur est authentifié, afficher le contenu de l'application
    if is_authenticated:
        #================Configuration des styles de la page ===========================
        #
        #=========== Dictionnaire de traduction =========================
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
                "group_members": "👥 Profil de l'Equipe",
                "rapport":"Produire un rapport",
                "form":"Formulaire",
                "toc":"Tables des matières",
                "section1":"Description Générale",
                "section2":"Analyse géographique dans Douala",
                "section3":"Analyse par arrondissement",
                "section4":"Conditions de Santé & Éligibilité",
                "section5":"Profilage des Donneurs Idéaux",
                "section6":"Analyse de l’Efficacité des Campagnes",
                "metric_text1":"Total candidat",
                "metric_text2":"Age moyen des candidats",
                "metric_text3":"taux moyen d'hémoglobine",
                "metric_text4":"Poids moyen des individus",
                "tilte_A_1_1":"Répartition par réligion",
                "tilte_A_1_2":"Répartition des candidats",
                "tilte_A_1_3":"",
                "tilte_A_1_4":"",
                "tilte_A_1_5":"",
                "tilte_A_1_6":"",
                "tilte_A_1_7":"",
                "tilte_A_1_8":"",
                "tilte_A_1_9":"",
                "tilte_A_1_10":"",
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
                "group_members": "👥 Team Profil",
                "rapport":"Produce repport",
                "form":"Forms",
                "toc":"Tables of content",
                "section1":"General Description",
                "section2":"Geographical Analysis in Douala",
                "section3":"Analysis by District",
                "section4":"Health Conditions & Eligibility",
                "section5":"Profiling of Ideal Donors",
                "section6":"Analysis of Campaign Effectiveness",
                "metric_text1":"Total Candidates",
                "metric_text2":"Average Age of Candidates",
                "metric_text3":"Average Hemoglobin Rate",
                "metric_text4":"Average Weight of Individuals",
            }
        }
        #=======================================================================
        #================= Contenu de la barre latérale =========================
        st.sidebar.image("Logo.png", use_container_width=True)
        lang = set_language()
        lang1="Français" if lang=="" else lang
        with st.sidebar.expander(traduire_texte("QR Code & Lien", lang), expanded=True):
            st.image("QR_code.jpg", use_container_width=True)

        #them=set_custom_theme()
        #__________Table des matières ________________________
        st.sidebar.markdown(sidebar_css, unsafe_allow_html=True)
        st.sidebar.title(translations[lang1]["toc"])
        st.sidebar.markdown(f"""
            <a href="#section1" class="sidebar-link">
                <span class="sidebar-link-icon">🩸</span> {translations[lang1]["section1"]}
            </a>
        """, unsafe_allow_html=True)

        st.sidebar.markdown(f"""
            <a href="#section2" class="sidebar-link">
                <span class="sidebar-link-icon">🩸🩸</span> {translations[lang1]["section2"]}
            </a>
        """, unsafe_allow_html=True)

        st.sidebar.markdown(f"""
            <a href="#section3" class="sidebar-link">
                <span class="sidebar-link-icon">🩸🩸</span> {translations[lang1]["section3"]}
            </a>
        """, unsafe_allow_html=True)

        st.sidebar.markdown(f"""
            <a href="#section4" class="sidebar-link">
                <span class="sidebar-link-icon">❤️</span> {translations[lang1]["section4"]}
            </a>
        """, unsafe_allow_html=True)

        st.sidebar.markdown(f"""
            <a href="#section5" class="sidebar-link">
                <span class="sidebar-link-icon">❤️❤️</span> {translations[lang1]["section5"]}
            </a>
        """, unsafe_allow_html=True)

        st.sidebar.markdown(f"""
            <a href="#section6" class="sidebar-link">
                <span class="sidebar-link-icon">❤️❤️❤️</span> {translations[lang1]["section6"]}
            </a>
        """, unsafe_allow_html=True)

        #_____________________________________________________________________________
        #______________Membre du groupe_______________________________________________
        #==========================================================

        #====================== EN TETE ===========================
        #----------Ajout des images d'en tête ----------------------
        st.markdown(header_css, unsafe_allow_html=True)
        st.markdown(table_css, unsafe_allow_html=True)

        # Image columns with improved styling
        """

        """

        # Enhanced Header
        st.markdown(
            f"""
            <div class="header-container">
                <h1 class="header-title">
                    🩸 {traduire_texte("Tableau de bord de la campagne de don de sang", lang)} 🩸
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        #==========================================================

        #-------------Chargement des bases de données et dictionnaires utiles----------------------------
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
        #------- Formatage des colones des différentes bases--------------------------
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


        #=========== Définition des onglets d'affichage des données===================


        st.markdown(tabs_css, unsafe_allow_html=True)

        # Define tabs with improved design
        tabs = st.tabs([
            f"📂 {traduire_texte(' Feuille des Données', lang)}", 
            f"📊 {traduire_texte('Tableau de Bord', lang)}",
            f"📄 {traduire_texte('Autres graphiques et rapport', lang)}",
            f"📝 {traduire_texte('Formulaire', lang)}",
            f"🖥️ {traduire_texte('Nouveau Tableau de Bord', lang)}",
            f"🤖 {traduire_texte('Chat Bot',lang)}", 
            f"👥 {traduire_texte('Profil du groupe', lang)}"
            
        ])

        #----ONGLET 1: BASES DE DONNEES
        with tabs[0]:
            
            Make_Global_DataFrame(data, title=traduire_texte("Données",lang),cle="tab1")
            #st.dataframe(geo_data)
            Make_Global_DataFrame(geo_data, title=traduire_texte("Données géospatialisée",lang))
            #st.dataframe(data_don)
            Make_Global_DataFrame(data_don, title=traduire_texte("Données sur les donneurs",lang))
            
        #----ONGLET 2: TABLEAU DE BORD PROPREMENT DIT
        with tabs[1]:
            a1, a2, a3 = st.columns(3) #définition du nombre de colonne
            
            #Visualisation des métriques
            with a1:
                metric_total_candidat=plot_metric(traduire_texte("Total candidat",lang),data.shape[0],prefix="",suffix="",show_graph=True,color_graph="rgba(0, 104, 201, 0.1)",)
            with a2:
                metric_age_moyen=plot_metric_2(traduire_texte("Age moyen des candidats",lang),data,"Age",prefix="",suffix=" ans",show_graph=True,color_graph="rgba(175, 32, 201, 0.2)",val_bin=45)
            with a3:
                metric_tx_hm_moyen=plot_metric_2(traduire_texte("taux moyen d'hémoglobine",lang), data, "Tx_hémoglobine", suffix=" g/dl",show_graph=True,color_graph="rgba(1, 230, 15, 0.7)",val_bin=300)

            
            st.write(" ")
            # SECTION 1: DESCRIPTION GENERALE
            st.markdown('<div id="section1"></div>', unsafe_allow_html=True)
            with st.expander(traduire_texte("Description Générale des candidats",lang), expanded=True,icon="🩸"):
                d1,d2= st.columns([7,3])
                with d1:
                    da1,da2=st.columns(2)
                    with da1:
                        bar_religion=make_bar(data,var="Religion",titre=traduire_texte("Répartition par réligion",lang),ordre=1,width=500,height=350,sens='h',bordure=10)
                    with da2:
                        #liquidfill_option = {
                        #     "series": [{"type": "liquidFill", "data": [0.7, 0.8, 0.4, 0.3]}]
                        # }
                        #st_echarts(liquidfill_option)
                        heat_region=make_heat_map(data,vars=['Region', 'Arrondissement','Quartier'],oder_var="ID",label_var='Quartier',titre=traduire_texte("Répartition des candidats",lang))
                    
                    dd1, dd2,dd3 =st.columns([2,4.5,3.5])
                    with dd1:
                        st.markdown(profile_css, unsafe_allow_html=True)
                        rep_genre=make_donutchart(data,var="Genre",titre=traduire_texte("Genre des candidats",lang))
                    with dd2:
                        school_level=make_cross_hist_b(data,var2="Niveau_etude",var1="Eligibilité",titre=traduire_texte("Niveau d'étude",lang),sens='v',typ_bar=1)
                    with dd3:
                        civility=make_cross_hist_b(data,var2="Situation_Mat",var1="Eligibilité",titre=traduire_texte("Statut Matrimonial",lang),sens='v',typ_bar=0,width=650,bordure=10)
                with d2:
                    rep_par_region=make_cross_hist_b(data[data["Region"]!="Litoral"],"Eligibilité","Region",titre=traduire_texte("Autre Région",lang),width=600,height=400,typ_bar=1)
                    eligibility_rep=make_donutchart(data,var="Eligibilité",titre=traduire_texte("Statut des candidats",lang),part=True)
                    
            #SECTION 2: ANALYSE GEOGRAPHIQUE DANS DOUALA 
            st.markdown('<div id="section2"></div>', unsafe_allow_html=True)  
            st.markdown(button_style, unsafe_allow_html=True)
            with st.expander(traduire_texte("Analyse géographique dans Douala",lang), expanded=True,icon="🩸"):  
                cc1,cc2,cc3,cc4,cc5=st.columns([2, 1,1.5,2,3.5])
                with cc1:
                    opacity=st.slider(traduire_texte("Transparence Carte",lang),min_value=0.0,max_value=1.0,value=0.8,step=0.01)
                with cc2:
                    couleur=st.selectbox(traduire_texte("Couleur carte",lang),sequence_couleur)
                with cc3:
                    style=st.selectbox(traduire_texte("Type de carte",lang),options=["open-street-map","carto-positron","carto-darkmatter"])
                with cc4:
                    genre=st.segmented_control(traduire_texte("Filtre: Genre",lang),options=data["Genre"].unique(),selection_mode="multi",default=data["Genre"].unique(),key="Genre1")
                with cc5:
                    Statut_Mat=st.segmented_control(traduire_texte("Filtre: Statut Marital",lang),options=data["Situation_Mat"].unique(),selection_mode="multi",default=data["Situation_Mat"].unique())   
                col1, col2 = st.columns([5, 3.4])
                geo_data_dla=geo_data[geo_data["Genre"].isin(genre)] if len(genre)!=0 else geo_data 
                geo_data_dla=geo_data[geo_data["Situation_Matrimoniale"].isin(Statut_Mat)] if len(Statut_Mat)!=0 else geo_data_dla 
                with col1:       
                    cartographie_all=make_chlorophet_map_folium_2(geo_data_dla,style_carte=style,palet_color=couleur,opacity=opacity,width=1000,height=650)
                with col2:
                    geo_data_dla["Categorie_profession"]=geo_data_dla["Categorie_profession"].replace("Personnel des services directs aux particuliers, commercants vendeurs","commercants vendeurs")
                    make_bar(geo_data_dla,"Categorie_profession",titre=traduire_texte("Categorie Professionnelle",lang),ordre=1,sens='h',height=400,width=600,bordure=10) 
                    make_area_chart(data,var="Date_remplissage",titre=traduire_texte("Evolution du nombre de candidat",lang),color=1)

                
                cb1, cb2, cb3=st.columns(3)
                with cb1:
                    make_cross_hist_b(geo_data_dla,"Eligibilite","Arrondissement",titre=traduire_texte("Répartition par arrondissement",lang),width=400,height=450,typ_bar=0,bordure=10)
                with cb2:
                    make_donutchart(geo_data_dla,var="ancien_don_sang",titre=traduire_texte("ancien donateur",lang))
                with cb3:
                    #make_cross_hist_3(geo_data_dla,"Niveau_etude","Age",titre="",agregation="avg",width=400,height=300)
                    make_bar(geo_data_dla,var="Classe_age",titre=traduire_texte("Répartition des ages",lang),width=700,height=450,bordure=10)
                    
            #SECTION 3 : ANALYSE PAR ARRONDISSEMENT
            st.markdown('<div id="section3"></div>', unsafe_allow_html=True)
            with st.expander(translations[lang1]["section3"],expanded=True,icon="🩸"):    
                b11, b12, b13, b14,b15, b16 =st.columns([1,1.3,1.5,4.2,1.7,2.1])
                with b11:
                    couleur_2=st.selectbox(traduire_texte("Couleur",lang),sequence_couleur)
                with b12:
                    opacity_2=st.slider(traduire_texte("Transparence",lang),min_value=0.0,max_value=1.0,value=0.8,step=0.01)
                with b13:
                    style_2=st.selectbox(traduire_texte("Thme carte",lang),options=["open-street-map","carto-positron","carto-darkmatter"])
                with b14:
                    arrondissement=st.multiselect(traduire_texte("Arrondissement",lang),options=["Douala 1er","Douala 2e","Douala 3e","Douala 4e", "Douala 5e"],default=["Douala 1er"])
                with b15:
                    last_don=st.segmented_control(traduire_texte("Filtre: Ancien donateur",lang),options=data["Don_pass"].unique(),selection_mode="multi",default=data["Don_pass"].unique())
                with b16:    
                    genre2=st.segmented_control(traduire_texte("Genre",lang),options=data["Genre"].unique(),selection_mode="multi", default=data["Genre"].unique(),key="genre_selection")
                    
                geo_data_arr=geo_data[geo_data["Arrondissement"].isin(arrondissement)] if len(arrondissement)!=0 else geo_data
                data_arr=data[data["Arrondissement"].isin(arrondissement)] if len(arrondissement)!=0 else data
                geo_data_arr=geo_data_arr[geo_data_arr["ancien_don_sang"].isin(last_don)] if len(last_don)!=0 else geo_data_arr
                geo_data_arr=geo_data_arr[geo_data_arr["Genre"].isin(genre2)] if len(genre2)!=0 else geo_data_arr
                
                b1, b2 =st.columns([6.5,2.5])
                with b1:
                    make_chlorophet_map_folium_2(geo_data_arr,style_carte=style_2,palet_color=couleur_2,opacity=opacity_2,width=1200,height=500)
                with b2:     
                    geo_data_arr_for_table=geo_data_arr.groupby("Quartier").agg({
                        "ID":"size"
                    })
                    geo_data_arr_for_table=geo_data_arr_for_table.sort_values("ID",ascending=False)
                    geo_data_arr_for_table=geo_data_arr_for_table.rename(columns={ "ID": "Nb_Candidats"})
                    geo_data_arr_for_table["Quartier"]=geo_data_arr_for_table.index
                    make_dataframe(geo_data_arr_for_table,col_alpha="Quartier",col_num="Nb_Candidats",hide_index=True)
                ba=st.columns(2)
                with ba[0]:
                    make_cross_hist_b_ech(geo_data_arr,"Eligibilite","Categorie_profession",titre=traduire_texte("Répartition par Catégory professionnelle",lang),width=800,height=750,typ_bar=2,bordure=10)
                    st.write("Bonjour")
                    #make_bar(geo_data_dla,"Categorie_profession",titre=traduire_texte("Categorie Professionnelle",lang),ordre=1,sens='h',height=400,width=600,bordure=10) 
                with ba[1]:
                    make_donutchart_2(geo_data_arr,var="Situation_Matrimoniale",titre=traduire_texte("Statut Matrimonial",lang))
            #SECTION 4 :  CONDITION DE SANTE ET ELIGIBILITE
            st.markdown('<div id="section4"></div>', unsafe_allow_html=True)
            with st.expander(translations[lang1]["section4"], expanded=True,icon="❤️"):
                c41, c42 ,c43=st.columns(3)
                with c41:
                    data_el=data.groupby("Eligibilité").agg({
                        "ID":"size"
                    })
                    data_el=data_el.rename(columns={"ID":"Nb_Candidats"})
                    data_el["Proportion"]=data_el["Nb_Candidats"]/float(data_el["Nb_Candidats"].sum())
                    make_progress_char(data_el["Proportion"][1],couleur="rgba(" + str(255*(1-data_el["Proportion"][1])) + "," + str(255*data_el["Proportion"][1]) +",0,1)",titre="Taux d'éligibilité")
                    data_mot=data[data["Autre_Raison"].notna()]
                    mot=" ".join(data_mot["Autre_Raison"])
                    #make_wordcloud(mot,titre="Autre raison",width=600,height=400)
                    st.dataframe(data_el[["Nb_Candidats"]])
                    
                with c42:
                    statut_el=st.segmented_control(traduire_texte("Statut des candidats",lang),options=data["Eligibilité"].unique(),selection_mode="multi",default=data["Eligibilité"].unique())
                    data_nl=data_el[data_el.index!="Eligible"]
                    #data_nl=data_el
                    make_multi_progress_bar(labels=data_nl.index,values=data_nl["Proportion"],colors=["red","orange"],titre="Candidats Non éligible",width=500,height=300)
                with c43:
                    data_raison=data[data["Raisons"].notna()]
                    data_raison = data_raison[data_raison["Eligibilité"].isin(statut_el)] if len(statut_el) != 0 else data_raison
                    raison=",".join(data_raison["Raisons"])
                    raison=raison.split(",")
                    r_ID=["ID"+str(i) for i in range(len(raison))]
                    raison=pd.DataFrame({"ID":r_ID,"Raisons":raison})
                    group_raison=raison.groupby("Raisons").agg({"ID":"size"})
                    group_raison=group_raison.rename(columns={"ID":"Effectif"})
                    group_raison=group_raison.sort_values("Effectif",ascending=False)
                    make_dataframe(group_raison,col_alpha="Raisons",col_num="Effectif")
                c4b1, c4b2 ,c4b3=st.columns(3)
                with c4b1:
                    data_raison = data[data["Eligibilité"].isin(statut_el)] if len(statut_el) != 0 else data
                    make_hist_box(data_raison,var1="Tx_hémoglobine",var2="Eligibilité",height=400)
                with c4b2: 
                    make_distribution_2(data_raison,var_alpha="Genre",var_num="Tx_hémoglobine",add_vline=12,add_vline2=13,titre=traduire_texte("Distribution du taux d'hémoglobine",lang))
                with c4b3:
                    make_cross_hist_b(data,"Eligibilité","Classe_age",titre=traduire_texte("Statut par classe d'age",lang),height=400,typ_bar=1)
                
            #SECTION 5:   PROFILAGE DES DONNEURS IDEAUX
            st.markdown('<div id="section5"></div>', unsafe_allow_html=True)
            with st.expander(translations[lang1]["section5"], expanded=True,icon="❤️"):
                    c5a1,c5a2,c5a3=st.columns(3)
                    with c5a1:
                        make_relative_bar(data,var1="Eligibilité",var2="Don_pass",width=500,height=400,titre=traduire_texte("Proportion des anciens donneurs",lang),)
                        data_SM=data.groupby("Situation_Mat").agg({"ID":"size"})
                        data_SM["Proportion"]=data_SM["ID"]/float(data_SM["ID"].sum())
                        make_multi_progress_bar(labels=data_SM.index, values=data_SM["Proportion"],colors=px.colors.qualitative.Vivid_r,width=500,height=400,titre=traduire_texte("Taux d'éligibilité par statut Marital",lang))
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
                        make_multi_progress_bar(labels=data_prof.index, values=data_prof["Proportion"],colors=px.colors.qualitative.Vivid,width=500,height=400,titre=traduire_texte("Taux d'éligibilité par niveaux d'éducation",lang))
                        make_cross_hist_b(data,var2="Religion",var1="Eligibilité",width=500,height=550,titre=traduire_texte("Réligion",lang),typ_bar=1)
            
            #SECTION 6:   ANALYSE DE L'EFFICACITE DE LA CAMPAGNE
            st.markdown('<div id="section6"></div>', unsafe_allow_html=True)
            with st.expander(translations[lang1]["section6"], expanded=True,icon="❤️"):
                c61,c62,c63=st.columns(3)
                data_don["ID"]="Don_" + (data_don.index+1).astype(str)
                data_don=data_don.rename(columns={"Groupe Sanguin ABO / Rhesus ":"Gpr_sang"})
                with c61:
                    make_progress_char(data_don.shape[0]/data.shape[0],"green",titre=traduire_texte("Efficacité de la compagne",lang))
                    data_don=data_don.set_index("ID",drop=True)
                    data_don["ID"]=data_don.index
                    data_don["Date"]=data_don["Horodateur"].dt.date
                    data_don["Heure"]=data_don['Horodateur'].dt.strftime('%d-%m-%Y %H')
                    trend_don=pd.crosstab(data_don["Date"],data_don["Sexe"])
                    make_bar(data_don,var="Date",color=0,titre=traduire_texte("Répartition des dons dans le temps",lang),height=400)
                    make_area_chart(data_don,var="Heure",titre="Heure d'affluence",height=300)
                with c62: 
                    st.write(traduire_texte("Répartition des groupes sanguin",lang))
                    make_blood_group(data_don,"Gpr_sang")
                    make_cross_hist_b(data_don,var1="Type de donation",var2="Classe_age",typ_bar=0,titre=traduire_texte("répartition des type de don par classe d'age",lang),bordure=7 )
                with c63:
                    make_relative_bar(data_don,var1="Gpr_sang",var2="Type de donation",titre=traduire_texte("Répartition des donneurs par groupe sanguin ", lang))
                    make_donutchart(data_don,var="Phenotype ", titre=traduire_texte("Différent type de phénotype des donneurs",lang))
                    #make_area_chart(data_don,var="Heure",titre="Heure d'affluence",height=400)

        #----ONGLET 3: TEST STATISTIQUES
        with tabs[2]:
            st.write(traduire_texte("Faite vos test statistiques",lang))
            cc1,cc2=st.columns([3,7])
            with cc1:
                Test=st.selectbox(traduire_texte("Choisissez le test à effectuer",lang),["Test d'indépendance","Test de comparaison de la moyenne", "Test de comparaison de la dispersion"])
                var1=st.selectbox(traduire_texte("Variable 1 de test",lang),options=var_qual )
                var2=st.selectbox(traduire_texte("Variable 2 de test",lang),options=var_qual if Test=="Test d'indépendance" else var_quant)
            with cc2:
                if st.button(traduire_texte("Lancer le test",lang)): 
                    if var1=="" or var2=="":
                        st.write(traduire_texte("Veuillez sélectionner des variables de test",lang))
                    else:
                        if Test==traduire_texte("Test d'indépendance",lang):
                            conclusion, table_cross,chi2, p,dof=test_independance_khi2(data,var1,var2)
                            st.write(conclusion)
                            st.dataframe(table_cross)
                            table_cross = table_cross.reset_index().melt(id_vars='index', var_name=var2, value_name='Effectif')
                            table_cross.rename(columns={'index': var1}, inplace=True)
                            make_cross_hist_b(data,var2,var1,typ_bar=0)
                            make_relative_bar(data,var2,var1,height=600)
                        elif Test==traduire_texte("Test de comparaison de la moyenne",lang):
                            conclusion, graph= test_comparaison_moyenne(data, var1, var2)
                            st.write(conclusion)
                            st.plotly_chart(graph)
                        else:
                            pass
            
            #st.write("Afficher votre rapport ")
            telecharger_pdf("Challenge_Proposal_2.pdf", lang=lang)

        #----ONGLET 4: FORMULAIRE
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
                st.subheader(traduire_texte("Informations générales",lang))
                col1, col2 = st.columns(2)
                
                with col1:
                    sexe = st.radio("Sexe", options=["M", "F"], index=0, 
                                    format_func=lambda x: "Masculin" if x == "M" else "Féminin")
                    age = st.number_input(traduire_texte("Âge (années)", lang), min_value=16, max_value=80, value=30, step=1)
                
                with col2:
                    poids = st.number_input(traduire_texte("Poids (kg)", lang), min_value=40.0, max_value=150.0, value=70.0, step=0.5)
                    derniere_donation_options = ["Jamais donné", "Plus de 3 mois", "Plus de 2 mois", "Dans les 2 derniers mois"]
                    derniere_donation_choix = st.selectbox(traduire_texte("Dernière donation",lang), options=derniere_donation_options)
                    
                    # Conversion du choix en jours
                    if derniere_donation_choix == "Jamais donné":
                        derniere_donation = 1000  # Valeur arbitraire élevée
                    elif derniere_donation_choix == "Plus de 3 mois":
                        derniere_donation = 91
                    elif derniere_donation_choix == "Plus de 2 mois":
                        derniere_donation = 61
                    else:
                        derniere_donation = 30
                
                # Informations socio-démographiques
                st.subheader(traduire_texte("Informations socio-démographiques", lang))
                col1, col2 = st.columns(2)
                
                with col1:
                    # Niveau d'étude (bouton radio)
                    niveau_etude = st.radio(
                        traduire_texte("Niveau d'étude",lang), 
                        options=["Aucun","Primaire", "Secondaire", "Universitaire 1er cycle", "Universitaire 2e cycle", "Universitaire 3e cycle"],
                        index=2
                    )
                    
                    # Statut matrimonial (liste de choix)
                    statut_matrimonial = st.selectbox(traduire_texte("Statut matrimonial", lang),
                        options=["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/Veuve", "Union libre"])
                    
                    # Religion
                    religion = st.selectbox(traduire_texte("Religion",lang),options=All_religion)
                
                with col2:
                    # Profession
                    profession = st.selectbox(traduire_texte("Profession",lang),options=metier)
                    # Quartier de résidence 
                    quartier = st.selectbox(traduire_texte("Quartier de résidence",lang),options=all_Quartier)
                    
                    # Nationalité 
                    nationalite_options = [
                        "Cameroun", "Nigeria", "Sénégal", "Côte d'Ivoire", "Ghana", "Bénin", 
                        "Tchad", "République Centrafricaine", "Gabon", "Congo", "RDC", 
                        "Autre pays africain", "Autre pays hors Afrique"
                    ]
                    nationalite = st.selectbox(
                        traduire_texte("Nationalité",lang),
                        options=nationalite_options,
                        index=0
                    )
                
                # Critères spécifiques aux femmes
                if sexe == "F":
                    st.subheader(traduire_texte("Informations spécifiques (femmes)",lang))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        grossesse_recente = st.checkbox(traduire_texte("Grossesse récente", lang))
                        if grossesse_recente:
                            temps_depuis_grossesse = st.number_input(traduire_texte("Temps depuis l'accouchement (mois)",lang), 
                                                                    min_value=0, max_value=24, value=3)
                        else:
                            temps_depuis_grossesse = None
                        
                        allaitement = st.checkbox(traduire_texte("Allaitement en cours",lang))
                        
                    with col2:
                        en_periode_menstruelle = st.checkbox(traduire_texte("Actuellement en période menstruelle",lang))
                        cycle_menstruel_irregulier = st.checkbox(traduire_texte("Cycle menstruel irrégulier",lang))
                        saignements_anormaux = st.checkbox(traduire_texte("Saignements anormaux",lang))
                else:
                    # Valeurs par défaut pour les hommes
                    grossesse_recente = None
                    temps_depuis_grossesse = None
                    allaitement = None
                    en_periode_menstruelle = None
                    cycle_menstruel_irregulier = None
                    saignements_anormaux = None
                
                # Critères médicaux
                st.subheader(traduire_texte("Informations médicales",lang))
                
                # Maladies chroniques
                maladies_selections = st.multiselect(
                    traduire_texte("Sélectionnez vos conditions médicales",lang),
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
                    traduire_texte("Sélectionnez les médicaments que vous prenez actuellement",lang),
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
                    interventions_recentes = st.checkbox(traduire_texte("Intervention chirurgicale récente",lang))
                    if interventions_recentes:
                        temps_depuis_intervention = st.number_input(traduire_texte("Temps depuis l'intervention (jours)",lang), 
                                                                min_value=0, max_value=365, value=30)
                    else:
                        temps_depuis_intervention = None
                
                with col2:
                    tatouages_recents = st.checkbox(traduire_texte("Tatouage ou piercing récent (moins de 4 mois)",lang))
                
                # Bouton de soumission
                submit_button = st.form_submit_button(label=traduire_texte("Évaluer mon éligibilité",lang))
            
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
                )
            
                # Affichage des résultats
                st.subheader(traduire_texte("Résultat de l'évaluation",lang))
                
                if resultat["eligible"]:
                    st.success(traduire_texte("✅ Vous êtes éligible pour passer aux examens approfondis pour le don de sang. Veuillez prendre un rendez-vous",lang))
                    statut="Temporairement éligible"
                elif (medicaments!=None) | (maladies_chroniques!=None) | tatouages_recents :
                    st.error(traduire_texte("❌ Vous n'êtes pas éligible pour le don de sang. Merci pour l'élant de coeur dont vous avez fait preuve",lang))
                    statut="Non-éligible"
                else:
                    st.error(traduire_texte("❌ Vous n'êtes pas éligible pour le don de sang actuellement.",lang))
                    statut="Temporairement non-éligible"
                
                # Affichage des raisons
                st.subheader(traduire_texte("Détails:",lang))
                for raison in resultat["raisons"]:
                    st.write(f"- {traduire_texte(raison,lang)}")
                
                # Affichage des recommandations si présentes
                if resultat["recommandations"]:
                    st.subheader("Recommandations:")
                    for recommandation in resultat["recommandations"]:
                        st.write(f"- {traduire_texte(recommandation,lang)}")
                
                # Avertissement 
                st.info(traduire_texte("⚠️ Cette évaluation est indicative et ne remplace pas l'avis d'un professionnel de santé. Veuillez consulter le personnel médical du centre de don pour une évaluation définitive.",lang))
                
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
                                "Catégorie_Professionnelle":profession_dict[profession] if profession in profession_dict else profession,
                                "quartier": quartier_dict[quartier][3] if quartier in quartier_dict else quartier,
                                "Arrondissement":quartier_dict[quartier][0] if quartier in quartier_dict else None,
                                "Lat":quartier_dict[quartier][1] if quartier in quartier_dict else None ,
                                "Long":quartier_dict[quartier][2] if quartier in quartier_dict else None ,
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
                            st.success(traduire_texte("Les informations ont été enregistrées avec succès!",lang))
                        except Exception as e:
                                st.error(traduire_texte(f"Erreur lors de l'enregistrement: {e}",lang))

                save(df_new)    
        
        #----ONGLET 5: TABLEAU DE BORD DE LA NOUVELLE CAMPAGNE
        with tabs[4]:
            #==================================================================================================================
            mise_a_ajour=st.button(traduire_texte("Mettre à Jour le Tableau de bord",lang))
            if mise_a_ajour:
                pass
                #st.rerun()
            forme_dla=geo_data.groupby("Arrondissement").agg({"geometry":"first"}) #Récupération des formes des arrondissemnt de Douala
            New_geo_data=pd.merge(forme_dla,df_new,on="Arrondissement", how="inner") #Jointure pour obtenir des données spatialisée des nouveau candidats
            New_geo_data=New_geo_data.set_index("ID")
            New_geo_data=gpd.GeoDataFrame(New_geo_data, geometry='geometry') # converssion du nouveau fichier en geodataframe, pour des analyse spatiale
            df_new["Date"]=df_new["Date_remplissage"].dt.date #calcul de la colonne Date
            df_new['date_heure'] = df_new['Date_remplissage'].dt.strftime('%d-%m-%Y %H') # extraction de la date heure
            df_new["Classe_age"] = df_new["age"].apply(class_age) #calcul de la colonne classe d'age
                #st.dataframe(df_new)
            ab1,ab2,ab3=st.columns(3)
                
            with ab1:
                plot_metric(translations[lang1]["metric_text1"],df_new.shape[0],prefix="",suffix="",show_graph=True,color_graph="rgba(0, 104, 201, 0.1)",)
            with ab2:
                plot_metric_2(translations[lang1]["metric_text2"],df_new,"age",prefix="",suffix=" ans",show_graph=True,color_graph="rgba(175, 32, 201, 0.2)",val_bin=45)
            with ab3:
                plot_metric_2(translations[lang1]["metric_text4"],df_new,"poids",prefix="",suffix=" kg",show_graph=True,color_graph="rgba(10, 242, 20, 0.2)",val_bin=45)
                
            with st.expander(traduire_texte("Information Globales sur les Nouvea candidat",lang), expanded=True,icon="❤️"):
                f11, f12, f13, f14,f15 =st.columns([1,1.3,1.5,4.2,1.7])
                with f11:
                    couleur_3=st.selectbox(traduire_texte("Couleur de fond",lang),sequence_couleur)
                with f12:
                    opacity_3=st.slider(traduire_texte("Transparence sur les arrondissement",lang),min_value=0.0,max_value=1.0,value=0.8,step=0.01)
                with f13:
                    style_3=st.selectbox(traduire_texte("Theme du fond de carte",lang),options=["open-street-map","carto-positron","carto-darkmatter"])
                with f14:
                    arrondissement_2=st.multiselect(traduire_texte("Selectionner des Arrondissement",lang),options=["Douala 1er","Douala 2e","Douala 3e","Douala 4e", "Douala 5e"],default=["Douala 1er","Douala 2e","Douala 3e","Douala 4e", "Douala 5e"])
                with f15:
                    genre3=st.multiselect(traduire_texte(" Filtre: Sexe",lang),options=df_new["sexe"].dropna().unique(),default=df_new["sexe"].dropna().unique())
                    
                df_new_arr_geo=New_geo_data[New_geo_data["Arrondissement"].isin(arrondissement_2)] if len(arrondissement_2)!=0 else New_geo_data
                df_new_arr_geo=df_new_arr_geo[df_new_arr_geo["sexe"].isin(genre3)] if len(genre3)!=0 else df_new_arr_geo
                    
                df_new_arr=df_new[df_new["Arrondissement"].isin(arrondissement_2)] if len(arrondissement_2)!=0 else df_new
                df_new_arr=df_new_arr[df_new_arr["sexe"].isin(genre3)] if len(genre3)!=0 else df_new_arr
                    
                c1, c2=st.columns([3.8,2.2])
                with c1:
                    make_map_folium(df_new_arr_geo, style_carte=style_3, palet_color=couleur_3, opacity=opacity_3, width=900, height=600)
                with c2:
                    eff_qrt=df_new_arr.groupby("quartier").agg({"ID":"size"})
                    eff_qrt=eff_qrt.rename(columns={"ID":"Effectif"})
                    eff_qrt=eff_qrt.sort_values("Effectif",ascending=False)
                    eff_qrt["Quartier"]=eff_qrt.index
                    make_dataframe(eff_qrt,col_alpha="Quartier",col_num="Effectif",hide_index=True)
                    #make_cross_hist_b(df_new,var2="Arrondissement",var1="Statut",titre="Répartition des candidats par arrondissement",bordure=9,width=600)
                ca1,ca2,ca3,ca4=st.columns(4)
                with ca1:
                    make_donutchart(df_new_arr,var="sexe",titre=traduire_texte("Répartiton des candidats par sexe",lang))
                with ca2:
                    make_cross_hist_b(df_new_arr,var2="statut_matrimonial",var1="Statut",titre=traduire_texte("Statut matrimonial des candidats",lang),typ_bar=0,bordure=7)
                with ca3:
                    make_cross_hist_b(df_new_arr,var2="religion",var1="Statut",bordure=12)
                with ca4:
                    make_donutchart(df_new_arr,var="Statut")
            with st.expander("2",expanded=True):
                caa1,caa2,caa3=st.columns(3)
                with caa1:
                    make_cross_hist_b(df_new,var2="Arrondissement",var1="Statut",titre=traduire_texte("Répartition des candidats par arrondissement",lang),bordure=9,width=600, height=400)
                with caa2:
                    make_area_chart(df_new,var="date_heure",titre=traduire_texte("Evolution des inscriptions",lang),width=500,height=400)
                with caa3:
                    make_bar(df_new,var="Classe_age",color=2,width=500,height=400, titre=traduire_texte("Classe d'age des candidats",lang),bordure=12)
                cd1,cd2,cd3=st.columns(3)
                with cd1:
                    df_new["Catégorie_Professionnelle"]=df_new["Catégorie_Professionnelle"].replace("Personnel des services directs aux particuliers, commercants vendeurs","commercants")
                    make_bar(df_new,var="Catégorie_Professionnelle",titre=traduire_texte("Categorie professionnelle",lang), sens='h',height=400,bordure=10)
                with cd2:
                    tx_el_F_2=df_new[(df_new["sexe"]=="F") & (df_new["Statut"]=="Temporairement éligible")].shape[0]/df_new[df_new["sexe"]=="F"].shape[0]
                    tx_el_M_2=df_new[(df_new["sexe"]=="M") & (df_new["Statut"]=="Temporairement éligible")].shape[0]/df_new[df_new["sexe"]=="M"].shape[0]
                    make_dbl_progress_char(vars=[tx_el_M_2,tx_el_F_2],labels=["Homme","Femme"],titre="Taux d'éligibilité",colors=["green","orange"])
                with cd3:
                    data_mot=df_new[df_new["Raison"].notna()]
                    mot=" ".join(data_mot["Raison"])
                    make_wordcloud(mot,titre=traduire_texte("Raison de Non éléigibilité",lang),width=600,height=400)
            Make_Global_DataFrame(df_new,title=traduire_texte("Nouvelle Base de donnée",lang))
        
            #st.image("QR_code.jpg", use_container_width=False,width=700)
        
        #----ONGLET 5:  Chat Bot
        with tabs[5]:
            def main():
                if "authenticated" in st.session_state and st.session_state["authenticated"]:
                    blood_donation_chatbot() 
                
            if __name__ == "__main__":
                main()   
                   
        #----ONGLET 6:  Profile
        with tabs[6]:
            st.markdown(profile_css, unsafe_allow_html=True)
            display_team_profiles2()
            
if __name__ == "__main__":
    main()        
        
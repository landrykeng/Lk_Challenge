import streamlit as st
import joblib
import numpy as np
from Good_KNN import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from Challenge import *
from Functions_RF import *

# Custom styles

st.markdown(global_font_css, unsafe_allow_html=True)

st.markdown(sidebar_css, unsafe_allow_html=True)

st.markdown(table_css, unsafe_allow_html=True)

st.markdown(title_css, unsafe_allow_html=True)

st.markdown(header_css, unsafe_allow_html=True)

st.markdown(global_font_css, unsafe_allow_html=True)


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
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.3);  /* Ombre plus prononcée */
            transform: translateY(-5px);  /* Léger soulèvement */
            border-color: rgba(200, 200, 200, 0.9);  /* Bordure plus visible */
        }

        /* Effet de survol - Mode Sombre */
        body.dark .stContainer:hover {
            box-shadow: 0 8px 12px rgba(255, 255, 255, 0.3);  /* Ombre claire */
            transform: translateY(-5px);  /* Léger soulèvement */
            border-color: rgba(100, 100, 110, 0.9);  /* Bordure plus visible */
        }

        /* Style spécifique pour les graphiques - Mode Clair */
        body:not(.dark) .stPlotlyChart {
            background-color: rgba(250, 250, 250, 0.95);  /* Fond très légèrement gris */
            border-radius: 8px;  /* Coins légèrement arrondis */
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);  /* Ombre très légère */
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

tabs_css = """
<style>
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #f0f2f6;
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
    color: #4a4a4a;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(75, 139, 255, 0.1);
    color: #4b8bff;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #4b8bff;
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

lang1=st.sidebar.selectbox("🌍 Choisissez la langue / Choose the language", ["", "Français", "English"])

st.markdown(tabs_css, unsafe_allow_html=True)
tb = st.tabs([
    f" {traduire_texte('Modèles de prédiction  KNN & Random Forest', lang)}", 
    f" {traduire_texte('à propos des modèles de prédiction', lang)}",
])

with tb[0]:
    choix_model=st.selectbox(traduire_texte("Choisir un model pour la prédiction",lang1), options=["KNN","Random Forest"])
    col=st.columns(2)
    with col[0]:
        data=pd.read_excel("Base-KNN.xlsx") if choix_model=="KNN" else pd.read_excel("Base-RF.xlsx")
        data.rename(columns={"Taux _hémoglobine_(g/dl)":"Tx_hemo"})
        X_train, X_val, X_test, y_train, y_val, y_test=split_data(data)
        best_model, best_params=optimize_knn(X_train, y_train, X_val, y_val) if choix_model=="KNN" else optimize_rforest(X_train, y_train, X_val, y_val)

        accuracy_train, accuracy_val, accuracy_test, train_df, val_df, test_df=evaluate_model(best_model, X_train, X_val, X_test, y_train, y_val, y_test)

        importance_df, fig=get_feature_importance(best_model, X_test, y_test)

        display_feature_importance(best_model, X_test, y_test)
    with col[1]:
        def main():
            st.title(traduire_texte("Prédiction d'Éligibilité au Don de Sang",lang1))
            
            # Définition des catégories professionnelles
            profession_categories = {
            1: "artisants et ouvriers d'industrie",
            2: "Employes de type administratif",
            3: "Personnel des services directs aux particuliers, commercants vendeurs",
            4: "dirigeants, cadre de direction et gerants",
            5: "professions intermediaires",
            6: "Intellectuels et scientifiques",
            7: "Élève",
            8: "Chomeurs",
            9: "Non précisée",
            10: "forces de defense et securité personnel",
            11: "Agriculture, elevage, peche et foret"
        }
            
            # Formulaire de saisie
            with st.form(key='prediction_form'):
                st.header(traduire_texte("Informations Personnelles",lang1))
                
                # Colonnes pour une meilleure disposition
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.number_input("Âge", min_value=18, max_value=65, value=30, step=1)
                    genre = st.radio("Genre", options=["Femme", "Homme"], index=0)
                
                with col2:
                    taux_hemoglobine = st.number_input(traduire_texte("Taux d'Hémoglobine (g/dl)",lang1), min_value=8.0, max_value=20.0, value=14.0, step=0.1)
                    ancien_don_sang = st.checkbox(traduire_texte("A déjà donné du sang",lang1))
                
                profession = st.selectbox(
                    traduire_texte("Catégorie Professionnelle",lang1), 
                    options=list(profession_categories.keys()),
                    format_func=lambda x: profession_categories[x]
                )
                
                # Bouton de prédiction
                submit_button = st.form_submit_button(label=traduire_texte('Prédire Éligibilité',lang1))
            
            # Traitement de la prédiction
            if submit_button:
                # Préparer les données pour la prédiction
                new_individual = {
                    'Age': age,
                    'Tx_hemo': taux_hemoglobine,
                    'Genre': 1 if genre == "Homme" else 0,
                    'Good_profession': profession,
                    'ancien_don_sang': 1 if ancien_don_sang else 0
                }
                if (genre=="Femme" and taux_hemoglobine<13) or (genre=="Homme" and taux_hemoglobine<12):
                    st.warning(traduire_texte(f"🔴 Non éligible au don de sang",lang1))
                else:
                    # Faire la prédiction
                    try:
                        prediction_result = predict_new_individual(new_individual) if choix_model=="KNN" else predict_new_individual_RF(new_individual,choice=6)
                        
                        # Afficher le résultat
                        if 'error' in prediction_result:
                            st.error(traduire_texte(f"Erreur de prédiction : {prediction_result['error']}", lang1))
                        else:
                            # Mise en forme du résultat
                            if prediction_result['prediction'] == 1:
                                st.success(traduire_texte(f"🟢 Éligible au don de sang",lang1))
                            else:
                                st.warning(traduire_texte(f"🔴 Non éligible au don de sang",lang1))
                            
                            # Afficher la probabilité si disponible
                            if prediction_result['probability'] is not None:
                                st.info(traduire_texte(f"Probabilité d'éligibilité : {prediction_result['probability']*100:.2f}%",lang1))
                            
                            # Informations détaillées
                            with st.expander(traduire_texte("Détails de la Prédiction",lang1)):
                                st.write(traduire_texte("Informations saisies :", lang1))
                                st.table(pd.DataFrame.from_dict(new_individual, orient='index', columns=['Valeur']))
                    
                    except Exception as e:
                        st.error(f"Une erreur s'est produite : {e}")

        if __name__ == "__main__":
            main()

with tb[1]:
    pass
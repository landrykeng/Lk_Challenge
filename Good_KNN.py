
# # Packages
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# # Importation et nettoyage des données
def load_and_preprocess_data(file_path='Data_Model.xlsm', sheet='2019'):
    # Importation de la base
    data = pd.read_excel(file_path, sheet_name=sheet)
    
    # Calcul de l'âge par rapport à 2019
    data['Age'] = data['Age'] - 6
    
    # Récupérations des colonnes utiles
    cols = ['ÉLIGIBILITÉ AU DON.', 'Niveau_etude', 'Age', 'Genre', 'Situation_Matrimoniale',
            'Good_profession', 'Godd_Religion', 'ancien_don_sang', 'Taux_hémoglobine_(g/dl)']
    
    # Création d'un dataframe plus filtré
    df = data[cols]
    
    # Suppression des valeurs manquantes de la colonne Taux _hémoglobine_(g/dl)
    df = df.dropna(subset=['Taux_hémoglobine_(g/dl)'])
    
    # Suppression des individus avec des âges non prévus par la réglementation
    df = df[~df['Age'].isin([12.0, 17.0, 119.0])]
    
    # On remplace les âges manquants par la médiane des âges
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # On se ramène à 2 modalités pour la variable cible
    df['ÉLIGIBILITÉ AU DON.'].replace({'Eligible': 1,
                                      'Temporairement Non-eligible': 0,
                                      'Définitivement non-eligible': 0},
                                     inplace=True)
    
    # Sélection des variables pour le modèle
    var_model = ['ÉLIGIBILITÉ AU DON.', 'Age', 'Taux_hémoglobine_(g/dl)', 
                 'Genre', 'Good_profession', 'ancien_don_sang']
    datamodel = df[var_model]
    
    # Correction des outliers
    data_quanti = datamodel.select_dtypes(include=['int64', 'float64'])
    
    for var in data_quanti.columns:
        if var != 'ÉLIGIBILITÉ AU DON.':  # Ne pas traiter la variable cible
            IQR = datamodel[var].quantile(0.75) - datamodel[var].quantile(0.25)
            lower = datamodel[var].quantile(0.25) - (1.5 * IQR)
            upper = datamodel[var].quantile(0.75) + (1.5 * IQR)
            datamodel[var] = np.where(datamodel[var] > upper, upper,
                                    np.where(datamodel[var] < lower, lower, datamodel[var]))
    
    # One-hot encoding des variables catégorielles
    datamodel = pd.get_dummies(datamodel, drop_first=True, dtype='int')
    
    # Standardisation des variables quantitatives
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(datamodel[['Age', 'Taux_hémoglobine_(g/dl)']])
    datamodel[['Age', 'Taux_hémoglobine_(g/dl)']] = X_scaled
    
    # Sauvegarde du scaler pour réutilisation future
    joblib.dump(scaler, 'scaler_knn.pkl')
    
    return datamodel


def split_data(datamodel):
    # 'ÉLIGIBILITÉ AU DON.' est la variable à prédire
    X = datamodel.drop('ÉLIGIBILITÉ AU DON.', axis=1)  # Features
    y = datamodel['ÉLIGIBILITÉ AU DON.']  # Target
    
    # On sépare notre jeu de données en 3: entraînement, validation, test (80% training, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def optimize_knn(X_train, y_train, X_val, y_val):
      
    # Meilleurs paramètres 
    best_params = {'leaf_size': 20, 'n_neighbors': 19, 'p': 2, 'weights': 'distance'}
      
    print(f"Meilleurs paramètres: {best_params}")
   
    # Initialisation du modèle avec les meilleurs paramètres
    best_model = KNeighborsClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    # Évaluation sur l'ensemble de validation
    val_accuracy = best_model.score(X_val, y_val)
    print(f"Précision sur l'ensemble de validation: {val_accuracy:.4f}")
    # Sauvegarde du modèle
    joblib.dump(best_model, 'knn_model.pkl')
    
    return best_model, best_params


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Probabilités pour ROC AUC (si disponible)
    try:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcul des scores AUC
        train_auc = roc_auc_score(y_train, y_train_proba)
        val_auc = roc_auc_score(y_val, y_val_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
    except:
        #print("Impossible de calculer l'AUC (le modèle ne donne pas de probabilités)")
        pass 
    
    # Précision
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
        
    # Rapports de classification
    report_train = classification_report(y_train, y_train_pred, output_dict=True)
    report_val = classification_report(y_val, y_val_pred, output_dict=True)
    report_test = classification_report(y_test, y_test_pred, output_dict=True)
    
    train_df = pd.DataFrame(report_train).transpose()
    val_df = pd.DataFrame(report_val).transpose()
    test_df = pd.DataFrame(report_test).transpose()
    
    # Importance des caractéristiques (pour KNN, nous allons utiliser les poids de distance)
    #feature_importance = get_feature_importance(model, X_test, y_test)
    
    return accuracy_train, accuracy_val, accuracy_test, train_df, val_df, test_df #feature_importance



def get_feature_importance(model, X_test, y_test):
    """
    Calcule l'importance des caractéristiques en utilisant une méthode basée sur permutation
    et renvoie un graphique Plotly interactif
    
    Paramètres:
    - model: Modèle entraîné avec une méthode .score()
    - X_test: DataFrame des caractéristiques de test
    - y_test: Série des étiquettes de test
    
    Retourne:
    - DataFrame avec l'importance des caractéristiques
    - Figure Plotly
    """
    # Calculer l'importance des caractéristiques par permutation
    feature_importance = {}
    baseline_score = model.score(X_test, y_test)
    
    for col in X_test.columns:
        X_test_permuted = X_test.copy()
        X_test_permuted[col] = np.random.permutation(X_test_permuted[col])
        permuted_score = model.score(X_test_permuted, y_test)
        feature_importance[col] = baseline_score - permuted_score
    
    # Convertir en DataFrame et trier par importance
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=True)
    
    # Créer un graphique Plotly horizontal bar chart
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='rgba(50, 171, 96, 0.6)',
        marker_line_color='rgba(50, 171, 96, 1.0)',
        marker_line_width=1.5
    ))
    
    # Personnaliser la mise en page
    fig.update_layout(
        title='Importance des caractéristiques (Méthode de permutation)',
        xaxis_title='Diminution de la précision après permutation',
        yaxis_title='Caractéristique',
        height=600,
        width=800,
        template='plotly_white'
    )
    
    return importance_df, fig

# Exemple d'utilisation dans Streamlit
def display_feature_importance(model, X_test, y_test):
    """
    Fonction pour afficher l'importance des caractéristiques dans Streamlit
    """
    
    # Calculer l'importance des caractéristiques
    importance_df, fig = get_feature_importance(model, X_test, y_test)
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau d'importance
    st.dataframe(importance_df)


def predict_new_individual(new_data):
    """
    Prédire l'éligibilité au don de sang pour un nouvel individu
    
    Paramètres:
    new_data : dict avec les caractéristiques suivantes :
               - 'Age': âge en années
               - 'Taux_hemoglobine': taux d'hémoglobine en g/dl
               - 'Genre': 0 pour femme, 1 pour homme
               - 'Good_profession': catégorie professionnelle (utiliser les mêmes codes que dans les données d'entraînement)
               - 'ancien_don_sang': 0 pour non, 1 pour oui
    
    Retourne:
    - La prédiction (0: Non éligible, 1: Éligible)
    - La probabilité d'être éligible
    """
    try:
        # Charger le modèle et le scaler
        model = joblib.load('knn_model.pkl')
        scaler = joblib.load('scaler_knn.pkl')
        
        # Préparation des données
        df = pd.DataFrame([new_data])
        
        # Vérifier si les colonnes correspondent à celles du modèle
        expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        if expected_columns is not None:
            # Créer un DataFrame avec toutes les colonnes nécessaires, remplies de 0
            data_preprocessed = pd.DataFrame(0, index=[0], columns=expected_columns)
            
            # Mettre à jour les colonnes disponibles
            if 'Age' in new_data:
                data_preprocessed['Age'] = new_data['Age']
            if 'Tx_hemo' in new_data:
                data_preprocessed['Tx_hemo'] = new_data['Tx_hemo']
            
            # Colonnes catégorielles
            if 'Genre' in new_data and new_data['Genre'] == 1:
                if 'Genre_1' in expected_columns:
                    data_preprocessed['Genre_1'] = 1
                
            if 'Good_profession' in new_data:
                prof_col = f"Good_profession_{new_data['Good_profession']}"
                if prof_col in expected_columns:
                    data_preprocessed[prof_col] = 1
            
            if 'ancien_don_sang' in new_data and new_data['ancien_don_sang'] == 1:
                if 'ancien_don_sang_1' in expected_columns:
                    data_preprocessed['ancien_don_sang_1'] = 1
            
            # Standardiser les variables quantitatives
            if 'Age' in new_data and 'Tx_hemo' in new_data:
                # Identifier les colonnes quantitatives dans le DataFrame
                quant_cols = ['Age', 'Tx_hemo']
                quant_cols_present = [col for col in quant_cols if col in data_preprocessed.columns]
                
                if quant_cols_present:
                    # Appliquer la standardisation aux colonnes quantitatives présentes
                    data_preprocessed[quant_cols_present] = scaler.transform(data_preprocessed[quant_cols_present])
        else:
            # Si nous n'avons pas les noms des colonnes, créer un DataFrame simple
            data_preprocessed = pd.DataFrame({
                'Age': [new_data.get('Age', 0)],
                'Tx_hemo': [new_data.get('Tx_hemo', 0)],
                'Genre_1': [1 if new_data.get('Genre', 0) == 1 else 0],
                'ancien_don_sang_1': [1 if new_data.get('ancien_don_sang', 0) == 1 else 0]
            })
            
            # Ajouter les colonnes pour Good_profession
            if 'Good_profession' in new_data:
                prof_col = f"Good_profession_{new_data['Good_profession']}"
                data_preprocessed[prof_col] = 1
            
            # Standardiser les variables quantitatives
            quant_cols = ['Age', 'Tx_hemo']
            data_preprocessed[quant_cols] = scaler.transform(data_preprocessed[quant_cols])
        
        # Faire la prédiction
        prediction = model.predict(data_preprocessed)[0]
        
        # Obtenir la probabilité si disponible
        try:
            probability = model.predict_proba(data_preprocessed)[0][1]
        except:
            probability = None
        
        result = {
            'prediction': int(prediction),
            'probability': probability,
            'interpretation': 'Éligible' if prediction == 1 else 'Non éligible'
        }
        
        return result
    
    except Exception as e:
        return {
            'error': str(e),
            'status': 'échec'
        }




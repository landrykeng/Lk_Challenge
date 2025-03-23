#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# # Importation et nettoyage des données
def load_and_preprocess_data(file_path='Good_Data.xlsm', sheet='2019'):
    # Importation de la base
    data = pd.read_excel(file_path, sheet_name=sheet)
    
    # Calcul de l'âge par rapport à 2019
    data['Age'] = data['Age'] - 6
    
    # Récupérations des colonnes utiles
    cols = ['ÉLIGIBILITÉ AU DON.', 'Niveau_etude', 'Age', 'Genre', 'Situation_Matrimoniale',
            'Good_profession', 'Godd_Religion', 'ancien_don_sang', 'Taux _hémoglobine_(g/dl)']
    
    # Création d'un dataframe plus filtré
    df = data[cols]
    
    # Suppression des valeurs manquantes de la colonne Taux _hémoglobine_(g/dl)
    df = df.dropna(subset=['Taux _hémoglobine_(g/dl)'])
    
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
    var_model = ['ÉLIGIBILITÉ AU DON.', 'Age', 'Taux _hémoglobine_(g/dl)', 
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
    X_scaled = scaler.fit_transform(datamodel[['Age', 'Taux _hémoglobine_(g/dl)']])
    datamodel[['Age', 'Taux _hémoglobine_(g/dl)']] = X_scaled
    
    # Sauvegarde du scaler pour réutilisation future
    joblib.dump(scaler, 'scaler.pkl')
    
    return datamodel


# In[3]:


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


# In[4]:


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


# In[9]:


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


# In[6]:


def get_feature_importance(model, X_test, y_test):
    """
    Calcule l'importance des caractéristiques en utilisant une méthode basée sur permutation
    """
    feature_importance = {}
    baseline_score = model.score(X_test, y_test)
    
    for col in X_test.columns:
        X_test_permuted = X_test.copy()
        X_test_permuted[col] = np.random.permutation(X_test_permuted[col])
        permuted_score = model.score(X_test_permuted, y_test)
        feature_importance[col] = baseline_score - permuted_score
    
    # Convertir en DataFrame et trier par importance
    importance_df = pd.DataFrame({
        'Feature': feature_importance.keys(),
        'Importance': feature_importance.values()
    }).sort_values('Importance', ascending=False)
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Diminution de la précision après permutation')
    plt.ylabel('Caractéristique')
    plt.title('Importance des caractéristiques')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return importance_df


# In[7]:


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
        scaler = joblib.load('scaler.pkl')
        
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
            if 'Taux_hemoglobine' in new_data:
                data_preprocessed['Taux _hémoglobine_(g/dl)'] = new_data['Taux_hemoglobine']
            
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
            if 'Age' in new_data and 'Taux_hemoglobine' in new_data:
                # Identifier les colonnes quantitatives dans le DataFrame
                quant_cols = ['Age', 'Taux _hémoglobine_(g/dl)']
                quant_cols_present = [col for col in quant_cols if col in data_preprocessed.columns]
                
                if quant_cols_present:
                    # Appliquer la standardisation aux colonnes quantitatives présentes
                    data_preprocessed[quant_cols_present] = scaler.transform(data_preprocessed[quant_cols_present])
        else:
            # Si nous n'avons pas les noms des colonnes, créer un DataFrame simple
            data_preprocessed = pd.DataFrame({
                'Age': [new_data.get('Age', 0)],
                'Taux _hémoglobine_(g/dl)': [new_data.get('Taux_hemoglobine', 0)],
                'Genre_1': [1 if new_data.get('Genre', 0) == 1 else 0],
                'ancien_don_sang_1': [1 if new_data.get('ancien_don_sang', 0) == 1 else 0]
            })
            
            # Ajouter les colonnes pour Good_profession
            if 'Good_profession' in new_data:
                prof_col = f"Good_profession_{new_data['Good_profession']}"
                data_preprocessed[prof_col] = 1
            
            # Standardiser les variables quantitatives
            quant_cols = ['Age', 'Taux _hémoglobine_(g/dl)']
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


# In[13]:


def main():
    # Chargement et prétraitement des données
    print("Chargement et prétraitement des données...")
    datamodel = load_and_preprocess_data()
    
    # Division des données
    print("Division des données en ensembles d'entraînement, validation et test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(datamodel)
    
    # Optimisation du modèle KNN
    print("Optimisation des hyperparamètres du modèle KNN...")
    best_model, best_params = optimize_knn(X_train, y_train, X_val, y_val)
    
    # Évaluation du modèle
    print("Évaluation du modèle...")
    results = evaluate_model(best_model, X_train, X_val, X_test, y_train, y_val, y_test)
    return best_model, results


# In[14]:


main()


# In[15]:


# Exemple de prédiction pour un nouvel individu
print("\nExemple de prédiction pour un nouvel individu:")
new_individual = {
    'Age': 35,
    'Taux_hemoglobine': 14.2,
    'Genre': 1,  # Homme
    'Good_profession': 2,  # Catégorie professionnelle 2
    'ancien_don_sang': 1  # A déjà donné du sang
}

prediction_result = predict_new_individual(new_individual)
print(f"Prédiction: {prediction_result}")


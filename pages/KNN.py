#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# # Importation et nettoyage des données

# In[2]:


#Impotation de la base
data = pd.read_excel('Good_Data.xlsm', sheet_name='2019')
# Calcul de l'âge par rapport à 2019
data['Age'] = data['Age']-6
# Récuoérations des colonnes utiles
cols = ['ÉLIGIBILITÉ AU DON.','Niveau_etude', 'Age', 'Genre', 'Situation_Matrimoniale',
        'Good_profession', 'Godd_Religion', 'ancien_don_sang','Taux _hémoglobine_(g/dl)']
# création d'un dataframe plus filtré
df = data[cols]
#Suppression des valeurs manquantes de la colonne Taux _hémoglobine_(g/dl)
df1 = df.dropna(subset=['Taux _hémoglobine_(g/dl)'])
#Suppression des individus avec des âges non prévus par la reglémentation
df1 = df1[~df1['Age'].isin([12.0, 17.0, 119.0])]
# On remplace les âges manquants par la médiane des âges
df1['Age'] = df1['Age'].fillna(df1['Age'].median())
#On se ramène à 2 modalité pour la variable cible
df1['ÉLIGIBILITÉ AU DON.'].replace({'Eligible':1,
                                    'Temporairement Non-eligible':0,
                                    'Définitivement non-eligible':0},
                                    inplace=True)


# In[3]:


Quanti = [var for var in df1.columns if df1[var].dtype in['int64','float64']]
data_quanti = df1[Quanti]
var_model = ['ÉLIGIBILITÉ AU DON.','Age', 'Taux _hémoglobine_(g/dl)', 'Genre', 'Good_profession', 'ancien_don_sang']
datamodel = df1[var_model]


# In[5]:


def Correct_outliers(data):
    for var in data_quanti.columns[1:]:
        IQR = data[var].quantile(0.75) - data[var].quantile(0.25)
        lower = data[var].quantile(0.25) - (1.5*IQR)
        upper = data[var].quantile(0.75) + (1.5*IQR)
        data[var] = np.where(data[var]>upper,upper,
                           np.where(data[var]<lower,lower,data[var]))

Correct_outliers(datamodel)


# In[6]:


# On définit la cible (y) et les features (X)
X = datamodel.drop(columns=['ÉLIGIBILITÉ AU DON.'])
y = datamodel['ÉLIGIBILITÉ AU DON.']


# In[7]:


# Recodage des variables
datamodel['Genre'].replace({'Homme':1, 'Femme':0}, inplace=True)
datamodel['Good_profession'].replace({
    'professions intermediaires':0,
    'Intellectuels et scientifiques':1,
    'Personnel des services directs aux particuliers, commercants vendeurs':2,
       "artisants et ouvriers d'industrie":3,
       'Élève':4,
       'Employes de type administratif':5,
       'Chomeurs':5,
       'dirigeants,cadre de direction et gerants':6,
       'forces de defense et securité personnel':7,
       'Agriculture, elevage, peche et foret':8,
       'Élève ':9,
       'Non précisée':10 }, inplace=True)
datamodel['ancien_don_sang'].replace({'Non':1, 'Oui':0}, inplace=True)


# In[8]:


# Créer une instance de StandardScaler
scaler = StandardScaler()

# Ajuster le scaler et transformer les données
X_scaled = scaler.fit_transform(datamodel[['Age', 'Taux _hémoglobine_(g/dl)']])
datamodel[['Age', 'Taux _hémoglobine_(g/dl)']] = X_scaled


# In[9]:


# 'ÉLIGIBILITÉ AU DON.' est la variable à prédire
X = datamodel.drop('ÉLIGIBILITÉ AU DON.', axis=1)  # Features
y = datamodel['ÉLIGIBILITÉ AU DON.']  # Target

# On sépare notre jeu de donnée 3: entraînement, validation, test (80% training, 20% temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Nous avons maintenant:
# - X_train, y_train: Training data
# - X_val, y_val: Validation data
# - X_test, y_test: Test data



# In[10]:


# S'assurer que y_train est une variable binaire
unique, counts = np.unique(y_train, return_counts=True)

# Dictionnaire pour compter les modalités
modality_counts = dict(zip(unique, counts))




# # KNN

# In[11]:


# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Liste des valeurs de k à tester
k_values = list(range(1, 30))


# In[12]:


# Création de l'instance du classificateur KNN
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score


leaf_size =  20
n_neighbors =  15, 
weights =  'distance'


n_neighbors = 14 # Obtenu par gridsearch
best_model= KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, weights= weights)
best_model.fit(X_train, y_train)
# Prédiction sur l'ensemble d'entraînement

def result():

    y_train_pred = best_model.predict(X_train)
    y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    report_test = classification_report(y_test,best_model.predict(X_test), output_dict=True)
    test_df = pd.DataFrame(report_test ).transpose()
    report_train = classification_report(y_train,best_model.predict(X_train), output_dict=True)
# Convertir le dictionnaire en DataFrame
    train_df = pd.DataFrame(report_train ).transpose()

    return accuracy_train,accuracy_test, train_df, test_df

#print(result())

a, b, c,d = result()
print(a)
print(b)
print(c)
print(d)


#
# Nouvelle fonction pour prédire une personne
def predict_new_person(age, hemoglobin, gender, profession, previous_donation, scaler=scaler, model=best_model):
    """
    Prédit l'éligibilité au don de sang pour une nouvelle personne.
    
    Paramètres :
    - age (float) : Âge de la personne
    - hemoglobin (float) : Taux d'hémoglobine en g/dl
    - gender (str) : 'Homme' ou 'Femme'
    - profession (str) : Profession selon les catégories définies
    - previous_donation (str) : 'Oui' ou 'Non'
    - scaler : Objet StandardScaler entraîné (par défaut, celui du code)
    - model : Modèle KNN entraîné (par défaut, best_model)
    
    Retourne :
    - prediction (int) : 1 (Eligible) ou 0 (Non-eligible)
    - proba (float) : Probabilité d'être éligible
    """
    # Créer un DataFrame avec les données de la nouvelle personne
    new_data = pd.DataFrame({
        'Age': [age],
        'Taux _hémoglobine_(g/dl)': [hemoglobin],
        'Genre': [gender],
        'Good_profession': [profession],
        'ancien_don_sang': [previous_donation]
    })

    # Recodage des variables catégoriques
    new_data['Genre'].replace({'Homme': 1, 'Femme': 0}, inplace=True)
    new_data['Good_profession'].replace({
        'professions intermediaires': 0,
        'Intellectuels et scientifiques': 1,
        'Personnel des services directs aux particuliers, commercants vendeurs': 2,
        "artisants et ouvriers d'industrie": 3,
        'Élève': 4,
        'Employes de type administratif': 5,
        'Chomeurs': 5,
        'dirigeants,cadre de direction et gerants': 6,
        'forces de defense et securité personnel': 7,
        'Agriculture, elevage, peche et foret': 8,
        'Élève ': 9,
        'Non précisée': 10}, inplace=True)
    new_data['ancien_don_sang'].replace({'Non': 1, 'Oui': 0}, inplace=True)

    # Standardisation des variables quantitatives
    new_data[['Age', 'Taux _hémoglobine_(g/dl)']] = scaler.transform(new_data[['Age', 'Taux _hémoglobine_(g/dl)']])

    # Prédiction
    prediction = model.predict(new_data)
    proba = model.predict_proba(new_data)[:, 1]  # Probabilité d'être éligible (classe 1)

    return prediction[0], proba[0]

# Exemple d'utilisation de la fonction
age = 25
hemoglobin = 12
gender = 'Homme'
profession = 'Chomeurs'
previous_donation = 'Oui'

prediction, proba = predict_new_person(age, hemoglobin, gender, profession, previous_donation)
print(f"Prédiction pour la nouvelle personne : {'Eligible' if prediction == 1 else 'Non-eligible'}")
print(f"Probabilité d'être éligible : {proba:.2f}")
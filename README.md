# Lk_Challenge
Concours de data vizualisation
# Fonctions Utilitaires pour l'Analyse et la Visualisation de Données

Ce module Python contient une collection de fonctions utilitaires pour l'analyse et la visualisation de données, spécialement conçues pour des applications Streamlit. Il inclut des outils pour la traduction, la création de graphiques, des tests statistiques, et des visualisations géospatiales.

## Fonctionnalités Principales

### 1. Traduction
- **traduire_texte(texte, langue)**: Traduit un texte donné vers une langue cible (anglais ou français par défaut).

### 2. Visualisations Statistiques
- **class_age(age)**: Classe les âges en catégories prédéfinies.
- **make_cross_hist(df, var1, var2, ...)**: Crée un histogramme croisé entre deux variables.
- **make_donutchart(df, var, ...)**: Génère un diagramme en anneau (donut chart) avec des pourcentages.
- **make_bar(df, var, ...)**: Produit un graphique à barres simple ou empilé.
- **make_relative_bar(df, var1, var2, ...)**: Affiche un graphique à barres empilées en proportions relatives.
- **make_distribution_2(df, var_alpha, var_num, ...)**: Visualise la distribution d'une variable quantitative avec des options avancées.
- **make_hist_box(df, var1, var2, ...)**: Combine un histogramme et un box plot pour comparer des distributions.

### 3. Tests Statistiques
- **test_independance_khi2(df, var1, var2)**: Effectue un test d'indépendance du Khi-deux entre deux variables catégorielles.
- **test_comparaison_moyenne(df, var1, var2)**: Compare les moyennes de deux groupes à l'aide d'un test de Student.

### 4. Visualisations Géospatiales
- **make_chlorophet_map(df, ...)**: Crée une carte choroplèthe interactive avec Plotly.
- **make_chlorophet_map_folium_2(df, ...)**: Génère une carte choroplèthe avancée avec Folium.
- **make_map_folium(df, ...)**: Produit une carte dynamique avec des marqueurs clusterisés.
- **calculate_zoom(lon_diff, lat_diff, ...)**: Calcule le niveau de zoom optimal pour une carte.

### 5. Métriques et Indicateurs
- **display_single_metric_advanced(label, value, delta, ...)**: Affiche une métrique avec un style personnalisé.
- **plot_metric(label, value, ...)**: Affiche une métrique sous forme d'indicateur visuel.
- **plot_gauge(indicator_number, ...)**: Crée un indicateur de type jauge (gauge chart).

### 6. Autres Fonctions Utilitaires
- **make_wordcloud(texte, ...)**: Génère un nuage de mots à partir d'un texte.
- **set_custom_theme()**: Applique un thème personnalisé (clair ou sombre) à l'application Streamlit.
- **telecharger_pdf(file_path)**: Permet de télécharger un fichier PDF depuis l'interface Streamlit.

## Utilisation

1. **Importation**:
   ```python
   from my_fonction import *

  # Créer un histogramme croisé
make_cross_hist(df, "Genre", "Statut", titre="Répartition par Genre et Statut")

# Afficher une métrique
plot_metric("Taux de réussite", 85, suffix="%")

# Générer une carte choroplèthe
make_chlorophet_map(df, style_carte="carto-positron", palet_color="Reds") 
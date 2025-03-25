#!/usr/bin/env python
# coding: utf-8


def verifier_eligibilite_don_sang(
    sexe: str,  
    age: int,
    poids: float,
    derniere_donation: int,
    grossesse_recente: bool = None,
    temps_depuis_grossesse: int = None,
    allaitement: bool = None,
    en_periode_menstruelle: bool = None,
    cycle_menstruel_irregulier: bool = None,
    saignements_anormaux: bool = None,
    maladies_chroniques: list = None,
    medicaments: list = None,
    interventions_recentes: bool = False,
    temps_depuis_intervention: int = None,
    tatouages_recents: bool = False
) -> dict:
    """
    Évalue l'éligibilité d'une personne pour passer aux examens approfondis pour le don de sang.
    
    Args:
        sexe (str): Sexe du donneur ("M" pour masculin, "F" pour féminin)
        age (int): Âge du donneur potentiel en années
        poids (float): Poids du donneur potentiel en kg
        derniere_donation (int): Nombre de jours depuis la dernière donation
        grossesse_recente (bool, optional): Si la personne a été enceinte récemment (pour femmes uniquement)
        temps_depuis_grossesse (int, optional): Nombre de mois depuis l'accouchement
        allaitement (bool, optional): Si la personne allaite actuellement (pour femmes uniquement)
        en_periode_menstruelle (bool, optional): Si la personne est en période de menstruation (pour femmes uniquement)
        cycle_menstruel_irregulier (bool, optional): Si la personne a un cycle menstruel irrégulier (pour femmes uniquement)
        saignements_anormaux (bool, optional): Si la personne a des saignements anormaux (pour femmes uniquement)
        maladies_chroniques (list, optional): Liste des maladies chroniques du donneur
        medicaments (list, optional): Liste des médicaments pris actuellement
        interventions_recentes (bool, optional): Si la personne a subi une intervention chirurgicale récemment
        temps_depuis_intervention (int, optional): Nombre de jours depuis l'intervention
        tatouages_recents (bool, optional): Si la personne s'est fait tatouer récemment
        
    Returns:
        dict: Dictionnaire contenant l'éligibilité (bool), les raisons détaillées (list) et recommandations (list)
    """
    # Validation des paramètres d'entrée
    if sexe not in ["M", "F"]:
        raise ValueError("Le sexe doit être 'M' pour masculin ou 'F' pour féminin")
    
    # Initialisation des variables
    eligible = True
    raisons = []
    
    # Liste des médicaments incompatibles avec le don de sang
    medicaments_incompatibles = [
        "isotretinoine", "finasteride", "dutasteride", "acitretine",
        "anticoagulants", "immunosuppresseurs", "antibiotiques"
    ]
    
    # Liste des maladies chroniques incompatibles avec le don de sang
    maladies_incompatibles = [
        "hepatite", "vih", "sida", "malaria", "brucellose", "tuberculose active",
        "cancer", "maladie cardiovasculaire severe", "diabete non controle", "anemie"
    ]
    
    # Vérification de l'âge (18 à 70 ans généralement)
    if age < 18:
        eligible = False
        raisons.append(f"Âge insuffisant: {age} ans (minimum 18 ans)")
    elif age > 70:
        eligible = False
        raisons.append(f"Âge dépassé: {age} ans (maximum 70 ans)")
        
    # Vérification du poids (minimum 50 kg généralement)
    if poids < 50:
        eligible = False
        raisons.append(f"Poids insuffisant: {poids} kg (minimum 50 kg)")
        
    # Vérification du délai depuis la dernière donation (différent selon le sexe)
    if sexe == "M" and derniere_donation < 56:
        eligible = False
        raisons.append(f"Délai depuis la dernière donation insuffisant: {derniere_donation} jours (minimum 56 jours pour les hommes)")
    elif sexe == "F" and derniere_donation < 84:
        eligible = False
        raisons.append(f"Délai depuis la dernière donation insuffisant: {derniere_donation} jours (minimum 84 jours pour les femmes)")
        
    # Vérifications spécifiques aux femmes
    if sexe == "F":
        # Vérification grossesse récente
        if grossesse_recente is not None and grossesse_recente:
            if temps_depuis_grossesse is None or temps_depuis_grossesse < 6:
                eligible = False
                raisons.append("Grossesse récente (moins de 6 mois)")
        
        # Vérification allaitement
        if allaitement is not None and allaitement:
            eligible = False
            raisons.append("Allaitement en cours")
        
        # Vérification période menstruelle
        if en_periode_menstruelle is not None and en_periode_menstruelle:
            eligible = False
            raisons.append("Don non recommandé pendant la période menstruelle")
        
        # Vérification cycle menstruel irrégulier
        if cycle_menstruel_irregulier is not None and cycle_menstruel_irregulier:
            eligible = False
            raisons.append("Cycle menstruel irrégulier (nécessite évaluation médicale)")
        
        # Vérification saignements anormaux
        if saignements_anormaux is not None and saignements_anormaux:
            eligible = False
            raisons.append("Saignements anormaux (nécessite évaluation médicale)")
    
    # Vérification des maladies chroniques
    if maladies_chroniques:
        maladies_detectees = []
        for maladie in maladies_chroniques:
            if any(m in maladie.lower() for m in maladies_incompatibles):
                maladies_detectees.append(maladie)
        
        if maladies_detectees:
            eligible = False
            raisons.append(f"Condition(s) médicale(s) incompatible(s): {', '.join(maladies_detectees)}")
    
    # Vérification des médicaments
    if medicaments:
        medicaments_problematiques = []
        for medicament in medicaments:
            if any(m in medicament.lower() for m in medicaments_incompatibles):
                medicaments_problematiques.append(medicament)
        
        if medicaments_problematiques:
            eligible = False
            raisons.append(f"Médicament(s) incompatible(s): {', '.join(medicaments_problematiques)}")
    
    # Vérification des interventions chirurgicales récentes
    if interventions_recentes:
        if temps_depuis_intervention is None or temps_depuis_intervention < 120:
            eligible = False
            raisons.append("Intervention chirurgicale récente (moins de 4 mois)")
    
    # Vérification des tatouages récents
    if tatouages_recents:
        eligible = False
        raisons.append("Tatouage récent (moins de 4 mois)")
    
    # Résultat
    resultat = {
        "eligible": eligible,
        "raisons": raisons if not eligible else ["Éligible pour passer aux examens approfondis; notamment l'examen de l'hemoglobine, le TDR et d'autres examens que le medecin jugera necessaire"],
        "recommandations": []
    }
    
    # Ajouter des recommandations si nécessaire
    if not eligible:
        if any("poids" in raison for raison in raisons):
            resultat["recommandations"].append("Consulter un professionnel de santé concernant le poids")
            
        if any("délai" in raison for raison in raisons):
            if sexe == "M":
                resultat["recommandations"].append("Attendre au moins 56 jours entre deux dons")
            else:
                resultat["recommandations"].append("Attendre au moins 84 jours entre deux dons")
        
        if sexe == "F" and any(("période menstruelle" in raison or "cycle" in raison or "saignements" in raison) for raison in raisons):
            resultat["recommandations"].append("Envisager un don après la fin du cycle menstruel actuel et consulter un médecin pour tout saignement anormal")
    
    return resultat


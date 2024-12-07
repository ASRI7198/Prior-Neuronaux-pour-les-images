# TP - Priors Neuronaux pour les Images


Ce TP explore l'utilisation des priors neuronaux pour résoudre des problèmes de reconstruction, d'inpainting et de débruitage d'images en s'appuyant sur un réseau de type encodeur-décodeur.

---

## Objectifs du TP

1. Construire un réseau neuronal pour reconstruire des images à partir de bruit.  
2. Utiliser ce réseau pour effectuer de l'inpainting (compléter des zones manquantes).  
3. Exploiter les priors neuronaux pour débruiter des images corrompues par du bruit gaussien.

---

## Librairies et Logiciels

La programmation est réalisée en Python à l'aide des librairies suivantes :

- **Numpy** : Calculs numériques.
- **Matplotlib** : Visualisation de l'évolution des fonctions de coût et des résultats.
- **PyTorch** : Modélisation et entraînement des réseaux neuronaux.
- **PIL (Pillow)** : Lecture et sauvegarde des images.

Un environnement Conda, défini par le fichier `tp_ml.yml`, est fourni sur la page du cours pour configurer toutes les dépendances nécessaires.

### Installation de l'environnement Conda

1. Créez l'environnement en exécutant :
   ```bash
   conda env create -f tp_ml.yml

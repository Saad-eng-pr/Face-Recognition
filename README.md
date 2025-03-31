# Face-Recognition
# Vérification d'Identité Faciale avec OpenCV et Dlib

## Description
Ce projet est un programme Python permettant de comparer deux images afin de déterminer si elles contiennent la même personne. Il utilise OpenCV et Dlib pour la détection et l'encodage des visages, puis applique une distance euclidienne pour comparer les encodages.

## Fonctionnalités
- Détection des visages dans les images
- Extraction des caractéristiques faciales
- Comparaison des visages avec un seuil configurable
- Affichage des images avec les points de repère faciaux (optionnel)

## Prérequis
Avant d'exécuter le projet, assurez-vous d'avoir installé les dépendances suivantes :

```bash
pip install opencv-python dlib numpy scipy

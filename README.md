# MedFusionNet — Détection Multi-Échelle de la Pneumonie

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ce dépôt contient l'intégralité des travaux du projet **DPR_PFA4IADO** : une solution d'IA hybride pour la détection de pneumonie sur radiographies thoraciques, ainsi que ses supports de communication techniques.

---

## 📂 Structure du Projet

L'organisation du dépôt est modulaire pour séparer le développement scientifique, la documentation et la visualisation :

- 🧠 **`DPR_MedFusionNet/`** : Cœur algorithmique (PyTorch).
    - Architecture hybride **DenseNet-121** (Local) + **Swin Transformer** (Global).
    - Système de fusion adaptative et estimation d'incertitude (MC-Dropout).
- 📊 **`DPR_tex/`** : Code source LaTeX de la présentation académique (Beamer).
    - Plus de 50 diapositives détaillant la méthodologie, l'architecture TikZ et les résultats.
- 🎬 **`DPR_Manim/`** : Scripts d'animations mathématiques et explicatives générées avec Manim.
- 🏗️ **`gcp/`** : (Ignoré) Scripts de déploiement et configuration Cloud.

---

## 🚀 MedFusionNet — Guide de Démarrage

### 1. Installation
Il est recommandé d'utiliser un environnement virtuel :

```bash
cd DPR_MedFusionNet/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Données (Dataset)
Nous utilisons le dataset **Chest X-Ray Images (Pneumonia)** disponible sur [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
Placez les données dans le dossier `DPR_MedFusionNet/data/` (non synchronisé via Git).

### 3. Entraînement sur Google Colab
En raison de la haute consommation mémoire du Swin Transformer, nous utilisons **Google Colab** pour l'accélération GPU :
- Ouvrez le carnet `MedFusionNet_Colab.ipynb`.
- Connectez votre Drive pour accéder aux scripts.
- Lancez l'entraînement via `train.py` avec l'optimisation **AMP** (Automatic Mixed Precision).

---

## 🧠 Spécifications Techniques

### Architecture Hybride
Le modèle MedFusionNet fusionne deux visions complémentaires :
1. **Branche CNN (Local)** : Capture les textures fines et les opacités alvéolaires.
2. **Branche Swin (Global)** : Analyse les dépendances à longue distance et la symétrie pulmonaire.
3. **Gated Fusion** : Une porte apprenante décide dynamiquement du poids à accorder à chaque branche.

### Sortie Clinique
Pour chaque prédiction $p$, le système fournit :
- ✅ **Confiance** : Calculée via 20 passes de MC-Dropout.
- 🔍 **Explicabilité** : Heatmap Grad-CAM localisant les zones suspectes.

---

## 📖 Présentation Beamer
Pour compiler la présentation technique :
```bash
cd DPR_tex/
pdflatex beamer.tex
```

---

## ⚖️ Licence
Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---
**Auteurs** : HANFAOUI Karim & KAFIF Imane (EMSI)

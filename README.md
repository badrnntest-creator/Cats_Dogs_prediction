
# Classification d'Images : Dogs vs Cats — CNN & Transfer Learning

Projet Final — Module 06 Deep Learning | EHTP 2024
Auteur : **Badr ben el ahmar **

---

## Description

Classification binaire d'images (chiens vs chats) par reseaux de neurones convolutionnels.
Trois approches comparees : CNN from scratch, Data Augmentation, Transfer Learning (MobileNetV2).
Meilleure accuracy obtenue : **~92%** avec MobileNetV2 pre-entraine sur ImageNet.

---

## Structure du projet

```
Projet - M6 Deep Learning/
|
+-- notebook/
|   +-- COLAB_dogs_vs_cats.ipynb    # Notebook complet avec outputs (Google Colab)
|
+-- streamlit_app/
|   +-- app.py                      # Application web Streamlit
|   +-- requirements.txt            # Dependances Python
|   +-- README.md                   # Instructions de deploiement
|
+-- model/
|   +-- cats_dogs_model.keras       # Modele final sauvegarde (MobileNetV2)
|
+-- rapport/
|   +-- rapport_classification_CNN.pdf  # Rapport academique PDF
|
+-- data_cats_and_dogs/             # Dataset local (non versionne sur GitHub)
|   +-- train/  (cats/ + dogs/)
|   +-- validation/  (cats/ + dogs/)
|
+-- README.md                       # Ce fichier
```

---

## Lancer l'application en local

```bash
# 1. Installer les dependances
pip install -r streamlit_app/requirements.txt

# 2. Lancer Streamlit depuis la racine du projet
streamlit run streamlit_app/app.py
```

Application disponible sur : `http://localhost:8501`

---

## Deployer sur Streamlit Cloud (5 etapes)

### 1. Preparer le depot GitHub

Creer un depot public avec la structure suivante (le dataset est exclu) :

```
mon-repo/
+-- streamlit_app/
|   +-- app.py
|   +-- requirements.txt
+-- model/
    +-- cats_dogs_model.keras
```

### 2. Pousser sur GitHub

```bash
git init
git add streamlit_app/ model/ README.md
git commit -m "Dogs vs Cats classifier - EHTP M06"
git remote add origin https://github.com/TON_USERNAME/TON_REPO.git
git push -u origin main
```

> Si `cats_dogs_model.keras` depasse 100 MB :
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes
> git add model/cats_dogs_model.keras
> ```

### 3. Se connecter sur Streamlit Cloud

Aller sur [share.streamlit.io](https://share.streamlit.io) et se connecter avec GitHub.

### 4. Creer une nouvelle application

- Cliquer **"New app"**
- Selectionner le depot GitHub
- Branch : `main`
- **Main file path** : `streamlit_app/app.py`

### 5. Deployer et obtenir l'URL

Cliquer **"Deploy!"** — apres quelques minutes, l'URL publique est disponible.

---

## Resultats

| Modele | Val Accuracy | Val Loss |
|---|---|---|
| CNN from scratch | ~75-80% | ~0.45 |
| CNN + Data Augmentation | ~78-83% | ~0.40 |
| MobileNetV2 Transfer Learning | **~88-93%** | **~0.25** |

---

## Livrables

- **Notebook** : `notebook/COLAB_dogs_vs_cats.ipynb` (outputs visibles — Google Colab GPU T4)
- **URL application** : [A completer apres deploiement]
- **Rapport PDF** : `rapport/rapport_classification_CNN.pdf`

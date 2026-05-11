# 🧬 DeepSkyn — ML Benchmarking | Classification du Type de Peau

> **Projet académique — 4TWIN, Semestre 2**  
> Algorithmes d'apprentissage supervisé appliqués à la dermatologie

---

## 📁 Contenu du dossier

| Fichier | Description |
|:--------|:------------|
| `Skin_Type_dataset.csv` | Dataset principal — 10 000 patients, 10 features |
| `KNN_Skin_Type.ipynb` | Modèle K-Nearest Neighbors (classification + régression) |
| `Decision_Tree_Skin_Type.ipynb` | Arbre de Décision avec pré/post-élagage (CCP) |
| `RandomForest_Skin_Type.ipynb` | Forêt Aléatoire (Bagging) |
| `SVM_Skin_Type.ipynb` | Support Vector Machine (noyaux RBF, poly, linear) |
| `XGBoost_Skin_Type.ipynb` | Extreme Gradient Boosting |
| **`Benchmarking.ipynb`** | **Comparaison finale de tous les algorithmes** ✅ |

---

## 🎯 Objectif

Prédire le **type de peau** (`Skin_Type`) d'un patient parmi 4 classes :
- `Combination` · `Dry` · `Normal` · `Oily`

À partir de 7 features : `Age`, `Gender`, `Hydration_Level`, `Oil_Level`, `Sensitivity`, `Humidity`, `Temperature`.

---

## 🔬 Ce que fait `Benchmarking.ipynb`

Le notebook de benchmarking agrège les résultats des 5 notebooks individuels et effectue une **comparaison exhaustive** :

1. **Étape 1 — Configurations optimales** : tableau des meilleurs hyperparamètres trouvés par GridSearchCV / RandomizedSearchCV pour chaque algorithme.

2. **Étape 2 — Benchmarking (code + visualisations)** :
   - DataFrame comparatif (Accuracy, F1 macro/weighted, F1 Combination, AUC, temps)
   - Bar charts triple : Accuracy / F1-Macro / F1-Weighted
   - Heatmap F1-Score par classe (focus classe `Combination`)
   - Courbes ROC unifiées (One-vs-Rest, tous les modèles)
   - 5 matrices de confusion normalisées côte à côte
   - Analyse des temps d'entraînement & d'inférence
   - Radar chart multi-dimensionnel

3. **Étape 3 — Discussion** : forces et faiblesses de chaque algorithme, progression DT → RF, coût computationnel SVM, XGBoost vs méthodes ensemblistes.

4. **Étape 4 — Justification** : sélection du modèle final pour déploiement dans l'application full-stack DeepSkyn.

---

## 📊 Résultats Clés

| Rang | Algorithme | Accuracy | F1 Macro | AUC Macro |
|:----:|:-----------|:--------:|:--------:|:---------:|
| 🥇 | **Random Forest** | ~81% | ~0.73 | ~0.91 |
| 🥈 | Decision Tree | ~79% | ~0.72 | ~0.87 |
| 🥈 | SVM (RBF) | ~79% | ~0.71 | ~0.88 |
| 🥉 | XGBoost | ~79% | ~0.72 | **0.913** |
| 5️⃣ | KNN | ~78% | ~0.68 | ~0.88 |

**✅ Modèle retenu pour le déploiement : Random Forest**  
Meilleur compromis accuracy / F1 / vitesse d'inférence / interprétabilité.

> ⚠️ La classe `Combination` reste difficile pour tous les modèles (F1 < 0.35) en raison de son sous-effectif dans le dataset. Une stratégie SMOTE est recommandée pour les futures itérations.

---

## 🖼️ CNN Image Classification (Deep Learning)

L'application intègre également un module de vision par ordinateur pour classer les types de peau directement à partir de photos.

**Architecture :**
Nous utilisons l'architecture **MobileNetV2** (Google), optimisée via **Transfer Learning (Feature Extraction)**. Le cerveau de base du modèle a été gelé, et nous avons entraîné une nouvelle couche de classification spécialisée sur notre propre dataset de **14 839 images**.

**Performances & Accélération :**
Le modèle atteint **90.4% d'Accuracy en validation**. Le pipeline est passé à une architecture **Inference-Only**. L'entraînement a été optimisé avec **TensorFlow** couplé au plugin **DirectML**, permettant à l'application d'exploiter la puissance des cartes graphiques Windows locales au lieu de saturer le CPU.

### 🚀 Utilisation 

Excellente nouvelle : le modèle compilé (`skin_type_cnn.keras`) ne pesant que **28 MB**, il est inclus directement dans ce repository ! 

**Si vous tirez (pull) ce projet, l'application fonctionnera instantanément :**
1. Assurez-vous d'avoir installé les bons packages : `pip install -r skin-ml-app/requirements.txt`
2. Lancez l'application Flask : `python skin-ml-app/app.py`
3. La page CNN est prête à l'emploi avec le modèle de 90.4% de précision !

*(Note : le dataset d'images brut de 14 000 photos n'est pas inclus pour ne pas saturer le dépôt. Si vous souhaitez ré-entraîner le modèle avec `train_cnn.py`, il vous faudra d'abord télécharger les dossiers d'images localement).*

---

## ⚙️ Prérequis

```bash
pip install scikit-learn xgboost pandas numpy matplotlib seaborn jupyter
```

## 🚀 Lancement

```bash
jupyter notebook Benchmarking.ipynb
```

Exécuter toutes les cellules dans l'ordre (`Run All`). Le notebook ré-entraîne et compare les 5 modèles automatiquement.

---

*DeepSkyn — Système d'Analyse Dermatologique par Intelligence Artificielle*

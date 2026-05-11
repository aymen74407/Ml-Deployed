# 🔧 Hyperparamètres Utilisés — Guide Simplifié

> Pour chaque algorithme : la valeur choisie ✅ et ce qu'elle contrôle en une phrase simple.

---

## 🔵 KNN — K-Nearest Neighbors

| Hyperparamètre | Valeur ✅ | Rôle |
|:---------------|:---------|:-----|
| `n_neighbors` | `17` | Combien de voisins consulter pour voter. Trop peu → instable, trop → trop générique. |
| `weights` | `'distance'` | Les voisins proches pèsent plus que les lointains dans le vote. |
| `metric` | `'euclidean'` | Comment mesurer la "distance" entre deux patients (formule de Pythagore généralisée). |

---

## 🌸 Decision Tree — Arbre de Décision

| Hyperparamètre | Valeur ✅ | Rôle |
|:---------------|:---------|:-----|
| `ccp_alpha` | `0.000944` | Force l'élagage de l'arbre après construction. Sans ça : 100% train / 68% test. Avec : 79% test. |
| `max_depth` | `7` | Limite à 7 niveaux de questions maximum → évite de mémoriser les données. |
| `min_samples_split` | `10` | Un groupe doit contenir au moins 10 patients pour être subdivisé. |
| `min_samples_leaf` | `5` | Chaque décision finale doit s'appuyer sur au moins 5 cas. |

---

## 🌲 Random Forest — Forêt Aléatoire

| Hyperparamètre | Valeur ✅ | Rôle |
|:---------------|:---------|:-----|
| `n_estimators` | `100` | Nombre d'arbres dans la forêt. 100 arbres votent ensemble → décision robuste. |
| `max_features` | `'sqrt'` | Chaque arbre ne voit que √7 ≈ 3 features aléatoires → diversité entre arbres. |
| `max_depth` | `None` | Pas de limite de profondeur : chaque arbre peut s'y spécialiser librement. Le vote collectif corrige les excès. |

---

## 🟠 SVM — Support Vector Machine

| Hyperparamètre | Valeur ✅ | Rôle |
|:---------------|:---------|:-----|
| `kernel` | `'rbf'` | Type de frontière de séparation. RBF = courbe gaussienne, idéal pour données non-linéaires. |
| `C` | `1` | Tolérance aux erreurs. `C` élevé → cherche à tout classer parfaitement → sur-apprentissage. |
| `gamma` | `0.1` | Rayon d'influence de chaque point. Petit = influence large et lisse, grand = trop localisé. |

---

## 🟢 XGBoost — Extreme Gradient Boosting

| Hyperparamètre | Valeur ✅ | Rôle |
|:---------------|:---------|:-----|
| `max_depth` | `7` | Profondeur de chaque arbre dans la séquence. |
| `learning_rate` | `0.3` | Taille du pas de correction à chaque étape. Grand = apprend vite, petit = apprend prudemment. |
| `n_estimators` | `100` | Nombre de rounds de boosting (arbres construits en séquence). |
| `subsample` | `0.8` | Chaque arbre utilise 80% des données → évite le sur-apprentissage. |
| `colsample_bytree` | `0.5` | Chaque arbre utilise 50% des features → diversité entre arbres. |
| `gamma` | `1.0` | Une division n'est faite que si elle améliore vraiment la précision. Filtre anti-bruit. |
| `reg_lambda` | `10` | Régularisation L2 : force les prédictions à rester modérées (évite les valeurs extrêmes). |
| `reg_alpha` | `0.1` | Régularisation L1 : peut annuler les contributions inutiles. |
| `min_child_weight` | `3` | Une feuille doit représenter au moins 3 unités de données pour exister. |

---

## 💡 À retenir en une phrase

> Tous ces paramètres servent un seul but : **ne pas apprendre par cœur les données d'entraînement**, pour mieux prédire de nouveaux patients.

---

*DeepSkyn — Projet Académique 4TWIN, Semestre 2*

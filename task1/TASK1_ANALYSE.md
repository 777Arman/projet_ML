# Task 1 - Analyse des Résultats : Classification Baseline (SVM et Decision Tree)

## Résumé de l'implémentation

### Objectif
Entraîner deux classifieurs multiclasses sur les features ResNet-50 pré-extraites et évaluer leurs performances sur l'ensemble de validation pour la détection d'objets (background, person, car, truck).

### Données utilisées
- **Features d'entraînement** : 161,752 régions avec vecteurs de 2048 dimensions
- **Features de validation** : 56,764 régions avec vecteurs de 2048 dimensions
- **Classes** : 4 classes {0: background, 1: person, 2: car, 3: truck}

### Distribution des classes

#### Ensemble d'entraînement
- **background** : 121,314 régions (75.0%)
- **person** : 26,704 régions (16.5%)
- **car** : 6,864 régions (4.2%)
- **truck** : 6,870 régions (4.2%)

#### Ensemble de validation
- **background** : 42,573 régions (75.0%)
- **person** : 9,173 régions (16.2%)
- **car** : 2,423 régions (4.3%)
- **truck** : 2,595 régions (4.6%)

---

## Modèles entraînés

### 1. Support Vector Machine (SVM)
- **Algorithme** : LinearSVC (scikit-learn)
- **Kernel** : Linéaire
- **Stratégie** : One-vs-rest (multiclasse)
- **Paramètres** : max_iter=5000, random_state=42
- **Temps d'entraînement** : Plusieurs minutes (normal pour 161k échantillons)

### 2. Decision Tree
- **Algorithme** : DecisionTreeClassifier (scikit-learn)
- **Profondeur maximale** : 10 (max_depth=10)
- **Paramètres** : random_state=42
- **Temps d'entraînement** : Rapide (quelques secondes)

---

## Résultats détaillés

### Support Vector Machine (SVM)

#### Métriques de classification (avant NMS)
| Classe      | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| background  | 0.925     | 0.954  | 0.939    | 42,573  |
| person      | 0.833     | 0.805  | 0.819    | 9,173   |
| car         | 0.713     | 0.581  | 0.640    | 2,423   |
| truck       | 0.706     | 0.545  | 0.615    | 2,595   |

- **Accuracy globale** : 89.5%
- **Macro F1 (toutes classes)** : 0.753
- **Foreground-only Macro F1** : **0.745**

#### Métriques de détection (après NMS, IoU=0.5)
- **mAP@0.5** : **0.1885**
- **AP par classe** :
  - person : 0.2129
  - car : 0.1598
  - truck : 0.1927

---

### Decision Tree (max_depth=10)

#### Métriques de classification (avant NMS)
| Classe      | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| background  | 0.877     | 0.925  | 0.900    | 42,573  |
| person      | 0.709     | 0.643  | 0.675    | 9,173   |
| car         | 0.569     | 0.419  | 0.483    | 2,423   |
| truck       | 0.555     | 0.379  | 0.450    | 2,595   |

- **Accuracy globale** : 83.3%
- **Macro F1 (toutes classes)** : 0.627
- **Foreground-only Macro F1** : **0.611**

#### Métriques de détection (après NMS, IoU=0.5)
- **mAP@0.5** : **0.1029**
- **AP par classe** :
  - person : 0.1223
  - car : 0.0834
  - truck : 0.1030

---

## Comparaison des modèles

| Métrique                  | SVM    | Decision Tree | Gagnant |
|---------------------------|--------|---------------|---------|
| Foreground Macro F1       | 0.745  | 0.611         | **SVM** |
| mAP@0.5                   | 0.1885 | 0.1029        | **SVM** |
| Accuracy                  | 89.5%  | 83.3%         | **SVM** |
| Vitesse d'entraînement    | Lente  | Rapide        | Tree    |

**→ Le SVM linéaire surpasse significativement le Decision Tree sur toutes les métriques de performance.**

---

## Discussion : Déséquilibre des classes et impact

### 1. Déséquilibre observé dans les données

Le dataset présente un **déséquilibre de classes très marqué** :
- La classe **background** représente **75%** des données
- Les trois classes d'avant-plan (foreground) ne représentent que 25% :
  - person : ~16%
  - car et truck : ~4% chacun

### 2. Classes les plus affectées par le déséquilibre

D'après les résultats, les classes **"car"** et **"truck"** sont les plus affectées :

#### Pour le SVM :
- **car** : F1=0.640, Recall=0.581 (taux de détection le plus faible)
- **truck** : F1=0.615, Recall=0.545 (pire recall de toutes les classes)
- **person** : F1=0.819, Recall=0.805 (meilleures performances parmi foreground)

#### Pour le Decision Tree :
- **truck** : F1=0.450, Recall=0.379 (performance catastrophique)
- **car** : F1=0.483, Recall=0.419 (très faible également)
- **person** : F1=0.675, Recall=0.643 (meilleur mais toujours affecté)

### 3. Pourquoi ces classes souffrent-elles le plus ?

#### A. Manque de données d'entraînement
- **truck** et **car** ont seulement ~6,800 exemples chacun (vs 121,314 pour background)
- Les modèles ont **18 fois moins d'exemples** pour apprendre ces classes
- Cela limite leur capacité à capturer la variabilité intra-classe (différents angles, éclairages, tailles)

#### B. Biais vers la classe majoritaire
- Les deux modèles tendent naturellement à prédire la classe **background** plus souvent
- Le SVM optimise la marge globale, ce qui favorise la classe majoritaire
- Le Decision Tree, même limité en profondeur, privilégie les splits qui réduisent l'impureté globale, favorisant background

#### C. Similarité visuelle potentielle
- Les features ResNet-50 pour **car** et **truck** peuvent être similaires (véhicules)
- Avec peu d'exemples, le modèle peine à discriminer entre ces deux classes proches
- On observe d'ailleurs des confusions probables entre car/truck dans les prédictions

#### D. Profondeur limitée du Decision Tree
- Un Decision Tree avec max_depth=10 a une capacité représentationnelle limitée
- Il ne peut créer que 2^10 = 1024 feuilles maximum
- Avec 75% de background, la majorité des feuilles seront dédiées à cette classe
- Les classes minoritaires disposent de moins de subdivisions pour capturer leur complexité

### 4. Impact sur les métriques

#### Recall faible pour car et truck
- Le modèle rate beaucoup de vrais positifs (faux négatifs nombreux)
- Beaucoup de vraies instances de car/truck sont classées comme background ou confondues entre elles

#### Precision raisonnable mais inférieure
- Quand le modèle prédit car ou truck, il se trompe relativement souvent
- Cela suggère des faux positifs provenant du background ou de confusions inter-classes

#### mAP faible
- La détection après NMS monneun mAP@0.5 de seulement 0.19 pour SVM et 0.10 pour Decision Tree
- Cela indique que beaucoup d'objets ne sont pas détectés correctement
- Les classes minoritaires contribuent négativement au mAP moyen

### 5. Observations spécifiques au Decision Tree

Le Decision Tree performe **encore plus mal** sur les classes minoritaires :
- F1 pour truck = 0.450 (vs 0.615 pour SVM)
- F1 pour car = 0.483 (vs 0.640 pour SVM)

**Raisons** :
1. **Overfitting sur background** : Même avec max_depth=10, l'arbre se concentre sur bien classer la classe majoritaire
2. **Manque de généralisation** : Les arbres de décision ont tendance à mémoriser les patterns d'entraînement
3. **Frontières de décision rigides** : Contrairement au SVM qui trouve des marges optimales, l'arbre crée des frontières rectangulaires qui peuvent mal capturer la distribution des classes minoritaires

### 6. Avantage relatif du SVM

Le SVM linéaire gère mieux le déséquilibre car :
- Il maximise la marge entre classes au lieu de simplement minimiser l'impureté
- La stratégie one-vs-rest crée un classifieur binaire pour chaque classe, donnant plus d'attention aux minoritaires
- C'est un modèle plus robuste pour les données haute dimension (2048 features)

---

## Recommandations pour améliorer les performances

1. **Gestion du déséquilibre** (Task 2) :
   - Utiliser des techniques de resampling (SMOTE, undersampling)
   - Ajuster les poids de classes (class_weight='balanced')
   - Augmentation de données pour car et truck

2. **Amélioration des modèles** :
   - Tester d'autres kernels SVM (RBF, polynomial) - Task 3
   - Optimiser la profondeur du Decision Tree - Task 4
   - Essayer des ensembles (Random Forest, Gradient Boosting)

3. **Optimisation des hyperparamètres** :
   - Grid search sur C (régularisation SVM)
   - Variation de max_depth, min_samples_split pour Decision Tree

4. **Feature engineering** :
   - Normalisation/standardisation des features
   - Réduction de dimensionnalité (PCA, t-SNE)
   - Fine-tuning de ResNet-50 sur ce dataset spécifique

---

## Fichiers générés

- `cm_svm.png` : Matrice de confusion pour SVM
- `cm_decision_tree.png` : Matrice de confusion pour Decision Tree
- `predictions_svm.csv` : Prédictions détaillées du SVM
- `predictions_decision_tree.csv` : Prédictions détaillées du Decision Tree
- `results.json` : Résumé des métriques pour comparaison

---

## Conclusion

La **Task 1** démontre clairement :

1. ✅ Le **SVM linéaire** est supérieur au Decision Tree sur ce problème (F1=0.745 vs 0.611)
2. ⚠️ Le **déséquilibre de classes** affecte gravement les performances, particulièrement pour **car** et **truck**
3. 📉 Les classes minoritaires (car, truck) avec seulement 4% des données chacune ont des performances médiocres
4. 🎯 La classe **person** (16% des données) s'en sort mieux mais reste affectée
5. 🔄 Des stratégies de gestion du déséquilibre sont **essentielles** pour améliorer les résultats (voir Tasks 2-4)

Le SVM capture mieux les patterns des classes minoritaires grâce à son approche par maximisation de marge, mais il reste clairement limité par le manque de données d'entraînement pour car et truck.

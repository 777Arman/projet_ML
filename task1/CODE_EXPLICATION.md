# Explication du Code - Task 1

## Vue d'ensemble du pipeline

Le script `task1.py` implémente un pipeline complet de classification pour la détection d'objets basée sur des régions (Region-based detection). Voici comment cela fonctionne étape par étape :

---

## 1. Chargement des données

```python
X_train, y_train, _, _ = load_features(data_dir / "features_train.npz")
X_val, y_val, boxes_val, ids_val = load_features(data_dir / "features_val.npz")
```

### Que contiennent ces fichiers ?

Les fichiers `.npz` sont des archives NumPy compressées contenant :
- **features** : Vecteurs de 2048 dimensions extraits par ResNet-50 (couche avant-dernière)
- **labels** : Étiquettes de classe (0=background, 1=person, 2=car, 3=truck)
- **boxes** : Coordonnées des boîtes englobantes [x1, y1, x2, y2]
- **image_ids** : Identifiants des images COCO dont proviennent les régions

### Pourquoi ResNet-50 ?

ResNet-50 est un CNN pré-entraîné sur ImageNet qui :
- Extrait des représentations visuelles riches
- Capture des patterns de haut niveau (formes, textures, contextes)
- Transforme chaque région d'image (crop) en un vecteur de 2048 nombres
- Ces vecteurs servent ensuite de features pour nos classifieurs simples (SVM, Decision Tree)

**Important** : Les features sont **pré-calculées**, donc on n'a pas besoin de recharger les images ni de réentraîner ResNet-50. C'est beaucoup plus rapide !

---

## 2. Entraînement du SVM

```python
from sklearn.svm import LinearSVC
svm = LinearSVC(max_iter=5000, random_state=42)
svm.fit(X_train, y_train)
```

### Comment fonctionne LinearSVC ?

**LinearSVC** est une implémentation optimisée de SVM avec kernel linéaire :

1. **Stratégie multiclasse** : One-vs-Rest (OvR)
   - Entraîne 4 classifieurs binaires (un par classe)
   - Classifieur 1 : background vs (person, car, truck)
   - Classifieur 2 : person vs (background, car, truck)
   - Classifieur 3 : car vs (background, person, truck)
   - Classifieur 4 : truck vs (background, person, car)

2. **Fonction de décision** : 
   - Chaque classifieur calcule un score de confiance
   - La classe avec le score le plus élevé est choisie
   - Mathématiquement : `score = w · x + b` (produit scalaire + biais)

3. **Optimisation** :
   - Cherche l'hyperplan optimal qui sépare les classes
   - Maximise la marge entre les classes (distance aux points les plus proches)
   - Utilise l'algorithme LIBLINEAR (très efficace pour grands datasets)

4. **Pourquoi max_iter=5000 ?**
   - Avec 161k échantillons, l'algorithme a besoin de beaucoup d'itérations pour converger
   - Si max_iter est trop faible, on obtient un warning de non-convergence
   - 5000 itérations permettent d'atteindre la solution optimale

### Prédictions et scores de confiance

```python
y_pred_svm = svm.predict(X_val)
scores_svm = svm.decision_function(X_val)   # (N, 4) - un score par classe
conf_svm = scores_svm[np.arange(len(X_val)), y_pred_svm]  # Score de la classe prédite
```

- `decision_function` retourne les distances signées à l'hyperplan pour chaque classe
- Plus le score est élevé, plus le modèle est confiant dans sa prédiction
- On extrait le score de la classe prédite pour l'utiliser comme mesure de confiance

---

## 3. Entraînement du Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=10, random_state=42)
tree.fit(X_train, y_train)
```

### Comment fonctionne le Decision Tree ?

Un arbre de décision construit une hiérarchie de règles de type if-else :

1. **Construction de l'arbre** :
   - À chaque nœud, choisit la meilleure feature et le meilleur seuil pour séparer les données
   - "Meilleur" = qui minimise l'impureté (Gini par défaut)
   - Exemple de règle : "Si feature[742] > 0.35, aller à gauche, sinon droite"

2. **Critère de split (Gini impurity)** :
   - Mesure à quel point un nœud est "pur" (contient une seule classe)
   - Gini = 0 → nœud parfaitement pur (une seule classe)
   - Gini élevé → mélange de plusieurs classes
   - L'algorithme choisit le split qui réduit le plus l'impureté

3. **max_depth=10** :
   - Limite la profondeur de l'arbre à 10 niveaux
   - Empêche l'overfitting (mémorisation du training set)
   - Avec 2048 features, un arbre non limité pourrait devenir énorme et sur-apprendre

4. **Structure résultante** :
   - Jusqu'à 2^10 = 1024 feuilles (nœuds terminaux)
   - Chaque feuille contient une distribution de classes
   - Prédiction = classe majoritaire dans la feuille

### Prédictions et probabilités

```python
y_pred_tree = tree.predict(X_val)
probas_tree = tree.predict_proba(X_val)  # (N, 4) - une probabilité par classe
conf_tree = probas_tree[np.arange(len(X_val)), y_pred_tree]  # Proba de la classe prédite
```

- `predict_proba` retourne la proportion de chaque classe dans la feuille terminale
- Exemple : feuille avec [80 background, 15 person, 3 car, 2 truck]
  - Probas = [0.8, 0.15, 0.03, 0.02]
  - Prédiction = background (classe majoritaire)

---

## 4. Évaluation : Métriques de classification

Le script calcule plusieurs métriques pour évaluer les performances :

### A. Confusion Matrix

```python
cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])
```

Matrice 4x4 montrant :
- Lignes = classes réelles
- Colonnes = classes prédites
- Diagonale = prédictions correctes
- Hors-diagonale = erreurs de classification

Exemple :
```
                 Prédit
              bg   per  car  trk
Réel  bg    [40k   500  200  100]
      per   [1000  7k   100  50 ]
      car   [800   50   1.4k 30 ]
      trk   [900   100  50   1.5k]
```

### B. Precision, Recall, F1-Score

Pour chaque classe :

**Precision** = TP / (TP + FP)
- Sur toutes les prédictions "car", quelle proportion sont vraiment des cars ?
- Mesure la fiabilité des prédictions positives

**Recall** = TP / (TP + FN)
- Sur tous les vrais cars, quelle proportion a été détectée ?
- Mesure la capacité à trouver tous les exemplaires

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Moyenne harmonique de precision et recall
- Meilleure métrique globale quand on veut équilibrer les deux

### C. Macro F1 (foreground-only)

```python
fg_mask = np.isin(y_true, [1, 2, 3])  # Seulement person, car, truck
fg_macro_f1 = moyenne(F1_person, F1_car, F1_truck)
```

- Ignore la classe background (trop facile, biaise les résultats)
- Moyenne simple des F1 des 3 classes d'intérêt
- **Métrique principale pour comparer les modèles**

---

## 5. Évaluation : mAP (mean Average Precision)

### Qu'est-ce que le mAP ?

Le mAP est la métrique standard pour évaluer les détecteurs d'objets :

1. **Transforme les régions classifiées en détections** :
   - Chaque région devient une boîte englobante avec classe et score de confiance
   - Exemple : [x1=100, y1=150, x2=400, y2=500, classe="car", confiance=0.87]

2. **Applique Non-Maximum Suppression (NMS)** :
   - Supprime les boîtes redondantes (même objet détecté plusieurs fois)
   - Si IoU > 0.5 et même classe prédite, garde seulement la boîte avec le score le plus élevé

3. **Calcule l'Average Precision (AP) par classe** :
   - Trie les détections par score de confiance (décroissant)
   - Pour chaque seuil, calcule precision et recall
   - Intègre la courbe precision-recall
   - AP = aire sous la courbe

4. **IoU (Intersection over Union)** :
   - Mesure le chevauchement entre boîte prédite et ground truth
   - IoU = Aire(intersection) / Aire(union)
   - Une détection est considérée correcte si IoU ≥ 0.5

5. **mAP@0.5** :
   - Moyenne des AP de toutes les classes (person, car, truck)
   - @0.5 signifie qu'on utilise IoU ≥ 0.5 comme seuil

### Pourquoi mAP est important ?

- Reflète la performance réelle pour la détection d'objets
- Prend en compte la localisation (IoU) et la classification
- Évalue sur l'ensemble de l'image, pas juste région par région
- Standard dans la communauté (COCO, Pascal VOC)

---

## 6. Structure du code : pourquoi cette architecture ?

### Séparation des responsabilités

```
load_features()                    → Chargement datos
train classifiers                  → Apprentissage
predict()                          → Inférence
classification_report()            → Métriques de classification
build_predictions_df()             → Formatage pour détection
evaluate_detection()               → mAP avec NMS
plot_confusion_matrix()            → Visualisation
```

### Avantages :
- **Modulaire** : Chaque fonction a un rôle précis
- **Réutilisable** : Les fonctions peuvent être utilisées dans les autres tasks
- **Testable** : Facile de débugger étape par étape
- **Lisible** : Le flux principal est clair dans `main()`

---

## 7. Comparaison SVM vs Decision Tree

### SVM Linéaire

**Avantages** :
- ✅ Meilleure généralisation (maximise la marge)
- ✅ Robuste en haute dimension (2048 features)
- ✅ Gère mieux le déséquilibre (one-vs-rest)
- ✅ Frontières de décision optimales

**Inconvénients** :
- ❌ Ententraînement lent (plusieurs minutes)
- ❌ Complexité O(n² × d) où n=échantillons, d=dimensions
- ❌ Nécessite beaucoup de mémoire pour grands datasets

### Decision Tree (max_depth=10)

**Avantages** :
- ✅ Entraînement très rapide (quelques secondes)
- ✅ Interprétable (peut visualiser l'arbre)
- ✅ Pas besoin de normaliser les features
- ✅ Gère naturellement le multiclasse

**Inconvénients** :
- ❌ Performances limitées par max_depth
- ❌ Biaisé vers classe majoritaire (background)
- ❌ Frontières rectangulaires (moins flexible)
- ❌ Moins bon pour haute dimension

### Résultat :
SVM gagne largement (F1=0.745 vs 0.611, mAP=0.19 vs 0.10)

---

## 8. Points techniques importants

### A. Pourquoi LinearSVC et pas SVC(kernel='linear') ?

`LinearSVC` utilise liblinear (optimisé pour problèmes linéaires) :
- Complexité O(n × d) au lieu de O(n² × d)
- Beaucoup plus rapide pour grands datasets
- Peut gérer 100k+ échantillons en temps raisonnable

`SVC(kernel='linear')` utilise libsvm :
- Plus flexible (supporte tous types de kernels)
- Mais beaucoup trop lent pour 161k échantillons
- Ne terminerait pas en temps raisonnable ici

### B. random_state=42

- Assure la reproductibilité des résultats
- Sans random_state, Decision Tree donnera des résultats différents à chaque run
- Important pour pouvoir comparer les expériences

### C. Extraction des scores de confiance

```python
scores_svm[np.arange(len(X_val)), y_pred_svm]
```

Cette notation NumPy avancée :
- `np.arange(len(X_val))` = [0, 1, 2, ..., N-1] (indices des lignes)
- `y_pred_svm` = [2, 0, 1, ...] (indices des colonnes = classes prédites)
- Résultat = extrait un élément par ligne (le score de la classe prédite)
- Équivalent à une boucle mais beaucoup plus rapide (vectorisé)

---

## Conclusion : Flux complet

1. 📂 **Charger features pré-extraites** (ResNet-50) → X_train (161k × 2048), y_train
2. 🎓 **Entraîner SVM** LinearSVC → Trouve hyperplans optimaux (one-vs-rest)
3. 🌳 **Entraîner Decision Tree** max_depth=10 → Construit arbre de règles
4. 🔮 **Prédire sur validation** → y_pred + scores de confiance
5. 📊 **Évaluer classification** → Confusion matrix, F1, precision, recall
6. 🎯 **Évaluer détection** → mAP@0.5 avec NMS
7. 💾 **Sauvegarder résultats** → JSON + CSV + PNG (matrices)
8. 📈 **Analyser** → Discussion sur déséquilibre de classes

C'est un pipeline classique de machine learning pour la détection d'objets basée sur régions (R-CNN approach).

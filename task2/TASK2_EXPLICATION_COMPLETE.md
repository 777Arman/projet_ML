# Task 2 - Explication Complète: Gestion du Déséquilibre de Classes

## Vue d'ensemble

La **Task 2** explore trois stratégies différentes pour atténuer le déséquilibre de classes observé dans la Task 1, où les classes `car` et `truck` (4% chacune) étaient largement surpassées par `background` (75%) et `person` (16%).

### Les 3 Stratégies Comparées

1. **Task 2a - Augmentation au niveau des régions d'images** (Region-level augmentation)
2. **Task 2b - SMOTE sur les vecteurs de features** (Feature-space augmentation)
3. **Baseline - SVM avec pondération de classes** (class_weight='balanced')

---

## Task 2a : Augmentation au Niveau des Régions d'Images

### Principe

Au lieu d'augmenter les features déjà extraites, on augmente les **images elles-mêmes** avant l'extraction par ResNet-50.

**Pourquoi c'est efficace ?**
- ResNet-50 est une fonction non-linéaire complexe
- Une petite transformation de l'image d'entrée → **vecteur de features complètement différent**
- On génère de véritables nouveaux exemples, pas juste des interpolations

### Implémentation (`extract_features_augmented.py`)

#### Transformations appliquées (seulement aux foreground):
```python
AUGMENT_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(p=0.5),          # Flip horizontal aléatoire
    T.ColorJitter(brightness=0.3,            # Variation luminosité/contraste
                  contrast=0.3,
                  saturation=0.2,
                  hue=0.1),
    T.RandomRotation(degrees=15),            # Rotation ±15°
    T.Resize((224, 224)),                    # Redimensionnement pour ResNet
    T.ToTensor(),
    T.Normalize(...)                         # Normalisation ImageNet
])
```

#### Processus détaillé:

1. **Pour chaque région background** (label = 0):
   - ✅ Extraire features normalement (1 version)
   - ❌ **Pas d'augmentation** (sinon trop de données)

2. **Pour chaque région foreground** (person, car, truck):
   - ✅ Extraire features de l'image originale (1x)
   - ✅ Extraire features de versions augmentées (N-1 fois)
   - **Facteur d'augmentation** : `augment_factor` (ex: 2 = original + 1 augmenté)

3. **Résultat attendu**:
   ```
   Avant:
   - background: 121,314 (75%)
   - person:      26,704 (16.5%)
   - car:          6,864 (4.2%)
   - truck:        6,870 (4.2%)
   
   Après (avec augment_factor=2):
   - background: 121,314 (60%)  ← inchangé
   - person:      53,408 (26%)  ← doublé
   - car:         13,728 (7%)   ← doublé
   - truck:       13,740 (7%)   ← doublé
   
   Total: ~202k régions (vs 162k avant)
   ```

### Avantages

✅ **Génère de vraies nouvelles données** : Pas d'interpolation artificielle  
✅ **Exploite la non-linéarité de ResNet** : Les features augmentées sont authentiques  
✅ **Améliore la robustesse** : Le modèle apprend des variations réalistes (angles, luminosity, etc.)  
✅ **Respecte la distribution naturelle des données** : Les augmentations sont des transformations plausibles

### Inconvénients

❌ **Coût computationnel élevé** : Doit repasser toutes les images par ResNet-50  
❌ **Nécessite les images originales** : Pas possible avec seulement les features pré-extraites  
❌ **Temps d'exécution long** : Peut prendre plusieurs heures selon le nombre d'images  
❌ **Stockage** : Le fichier `features_train_augmented.npz` est plus volumineux

### Commande d'exécution

```bash
python extract_features_augmented.py --data_dir ./coco_filtered --augment_factor 2
```

**Note** : Nécessite que les images soient extraites dans `coco_filtered/images/`

---

## Task 2b : SMOTE sur les Vecteurs de Features

### Principe

**SMOTE** (Synthetic Minority Over-sampling Technique) génère des exemples **synthétiques** en interpolant entre des exemples existants dans l'espace des features.

### Comment fonctionne SMOTE ?

#### Algorithme:

1. **Pour chaque échantillon minoritaire x** :
   - Trouve ses **k plus proches voisins** (k=5 par défaut) de la même classe
   - Choisit aléatoirement un de ces voisins : `x_nn`
   
2. **Génère un nouvel échantillon synthétique** :
   ```
   x_new = x + λ × (x_nn - x)
   ```
   où `λ ~ Uniforme[0, 1]`
   
3. **Répète** jusqu'à atteindre le nombre souhaité d'échantillons

#### Exemple visuel (2D):
```
Original:        Après SMOTE:
  o   o            o * o
                   * o *
    o          →     o * 
  o                o * o
                   *
```
Les `*` sont les nouveaux échantillons générés par interpolation linéaire.

### Implémentation

```python
from imblearn.over_sampling import SMOTE

# Séparer background et foreground
X_bg, y_bg = X_train[y_train == 0], y_train[y_train == 0]
X_fg, y_fg = X_train[y_train > 0], y_train[y_train > 0]

# Appliquer SMOTE seulement sur foreground
smote = SMOTE(sampling_strategy={2: 14000, 3: 14000}, k_neighbors=5)
X_fg_resampled, y_fg_resampled = smote.fit_resample(X_fg, y_fg)

# Recombiner
X_combined = np.vstack([X_bg, X_fg_resampled])
y_combined = np.concatenate([y_bg, y_fg_resampled])
```

### Stratégie de rééchantillonnage

**Option 1** - Balance complète (trop agressif):
```python
sampling_strategy='auto'  # Toutes les classes foreground au niveau de 'person'
```
- person: 26,704 → 26,704
- car:     6,864 → 26,704  ← +300%
- truck:   6,870 → 26,704  ← +300%

**Option 2** - Conservative (recommandé):
```python
sampling_strategy={2: 14000, 3: 14000}  # Doubler car et truck
```
- person: 26,704 → 26,704  ← inchangé
- car:     6,864 → 14,000  ← +104%
- truck:   6,870 → 14,000  ← +104%

### Avantages

✅ **Rapide** : Pas besoin de repasser par ResNet-50  
✅ **Fonctionne sur features pré-extraites** : Pas besoin des images originales  
✅ **Simple à implémenter** : Une seule ligne de code avec scikit-learn  
✅ **Bien établi** : Technique éprouvée en machine learning

### Inconvénients

❌ **Interpolation linéaire** : Les features synthétiques ne correspondent pas à de vraies images  
❌ **Problématique en haute dimension** : Avec 2048 dimensions, l'espace est très "sparse"  
❌ **Peut créer des exemples non-réalistes** : L'interpolation entre deux vecteurs ResNet n'a pas de sens sémantique garanti  
❌ **"Curse of dimensionality"** : En dimension 2048, la notion de "voisin proche" devient floue

### Limitations Théoriques de SMOTE dans un Espace de Features Pré-entraînées

#### Problème 1 : Non-invertibilité de ResNet

```
Image A  ──ResNet──>  Feature A (2048-d)
                           ↓ SMOTE
Image ?  ←─────────   Feature synthetic (2048-d)
```

**Le vecteur synthétique ne correspond à aucune image réelle !**

#### Problème 2 : Violation de la manifold

Les features ResNet ne remplissent pas tout l'espace ℝ²⁰⁴⁸. Elles se trouvent sur une **variété de dimension réduite** (manifold).

```
L'interpolation linéaire peut sortir de cette variété:

     ●─ ─ ─ ─● ← Interpolation passe hors manifold
    /         \
   ●           ● ← Vraies features ResNet
  /             \
 ●───────────────●
```

Résultat : Les features SMOTE peuvent ne pas respecter les contraintes apprises par ResNet.

#### Problème 3 : Similarité sémantique vs. Similarité euclidienne

En dimension 2048, deux vecteurs "proches" au sens euclidien ne représentent pas forcément des images visuellement similaires.

**Exemple** :
- Car vue de face : `[0.2, -0.5, ..., 0.1]`
- Car vue de côté : `[0.3, -0.4, ..., 0.2]`
- Truck vue de face : `[0.25, -0.45, ..., 0.15]` ← Plus proche numériquement !

L'interpolation pourrait créer un "monstre" mélangeant car et truck.

---

## Stratégie 3 : SVM avec Pondération de Classes

### Principe

Au lieu de générer plus de données, on ajuste la **fonction de perte** du SVM pour donner plus d'importance aux classes minoritaires.

### Comment ça fonctionne ?

#### SVM standard:
```
min  ½||w||² + C × Σ ξᵢ
```

Toutes les erreurs de classification ont le même coût `C`.

#### SVM pondéré (class_weight='balanced'):
```
min  ½||w||² + Σ wᵢ × ξᵢ

où wᵢ = n_total / (n_classes × n_samples_of_class_i)
```

**Calcul des poids** :
```python
n_total = 161,752
n_classes = 4

w_background = 161,752 / (4 × 121,314) = 0.33
w_person     = 161,752 / (4 × 26,704)  = 1.51
w_car        = 161,752 / (4 × 6,864)   = 5.89  ← 18x plus que background!
w_truck      = 161,752 / (4 × 6,870)   = 5.88
```

**Effet** : Une erreur sur `car` coûte **5.89x plus cher** qu'une erreur sur `background`.

### Implémentation

```python
from sklearn.svm import LinearSVC

svm = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')
svm.fit(X_train, y_train)
```

C'est tout ! Scikit-learn calcule automatiquement les poids.

### Avantages

✅ **Extrêmement simple** : Un seul paramètre à ajouter  
✅ **Aucun coût computationnel supplémentaire** : Pas de nouvelles données  
✅ **Pas de risque d'overfitting** : Ne génère pas de données synthétiques  
✅ **Interprétable** : Les poids sont clairement définis

### Inconvénients

❌ **Ne génère pas de nouvelles données** : Le modèle voit toujours les mêmes exemples  
❌ **Peut donner trop d'importance aux minorités** : Risque de sur-prédire les classes rares  
❌ **Ne résout pas le sous-échantillonnage** : Manque toujours de variabilité pour car/truck  
❌ **Nécessite du tuning** : Le comportement dépend du paramètre C du SVM

---

## Comparaison Théorique des Trois Approches

| Aspect | Region Augmentation (2a) | SMOTE (2b) | Class-weighted (3) |
|--------|--------------------------|------------|-------------------|
| **Génère nouvelles données** | ✅ Oui, réalistes | ✅ Oui, synthétiques | ❌ Non |
| **Coût computationnel** | ❌ Élevé (ResNet) | ✅ Faible | ✅ Nul |
| **Nécessite images** | ❌ Oui | ✅ Non | ✅ Non |
| **Qualité des exemples** | ⭐⭐⭐ Excellente | ⭐ Questionnable | N/A |
| **Robustesse géométrique** | ✅ Améliore | ❌ Neutre | ❌ Neutre |
| **Risque d'overfitting** | ⚠️ Moyen | ⚠️⚠️ Élevé | ✅ Faible |  
| **Facilité d'implémentation** | ⚠️ Complexe | ✅ Simple | ⭐ Triviale |
| **Temps d'exécution** | ❌ Heures | ⚠️ Minutes | ✅ Secondes |

### Verdict Théorique

**Meilleure approche** : **Region-level augmentation (2a)**
- Génère de vraies nouvelles données avec de vraies variations
- Les features extraites par ResNet sont authentiques
- Améliore la généralisation

**Seconde meilleure** : **Class-weighted SVM (3)**
- Simple et rapide
- Pas de risque de créer des exemples artificiels
- Bon compromis performance/coût

**Moins recommandée** : **SMOTE (2b)**
- Interpolations linéaires en dimension 2048 sont douteuses
- Les features synthétiques ne correspondent à aucune image
- Peut créer du "bruit" dans l'espace de features

---

## Résultats Attendus

### Métriques de comparaison

On compare principalement :
1. **Foreground Macro F1** : Moyenne du F1 pour person, car, truck
2. **mAP@0.5** : Mean Average Precision à IoU=0.5
3. **Recall par classe** : Capacité à détecter car et truck

### Prédictions basées sur la théorie

#### Task 1 Baseline (référence):
```
Foreground Macro F1: 0.745
mAP@0.5: 0.189
  person: F1=0.819
  car:    F1=0.640  ← problématique
  truck:  F1=0.615  ← problématique
```

#### Region Augmentation (attendu):
```
Foreground Macro F1: ~0.80-0.82  ← +7-10%
mAP@0.5: ~0.21-0.23  ← +10-20%
  person: F1=0.83   ← légère amélioration
  car:    F1=0.72   ← +12% (plus d'exemples)
  truck:  F1=0.71   ← +15% (plus d'exemples)
```

**Pourquoi meilleur ?**
- Plus d'exemples de car/truck avec variations réalistes
- Modèle apprend la robustesse aux transformations géométriques

#### SMOTE (attendu):
```
Foreground Macro F1: ~0.75-0.77  ← +1-3%
mAP@0.5: ~0.19-0.20  ← +0-5%
  person: F1=0.81   ← similaire
  car:    F1=0.68   ← +6% (plus d'exemples, mais artificiels)
  truck:  F1=0.66   ← +8% (idem)
```

**Pourquoi seulement un petit gain ?**
- Features synthétiques peuvent ne pas être réalistes
- Risque d'apprendre des patterns "fantômes" qui n'existent pas dans les vraies images

#### Class-weighted (attendu):
```
Foreground Macro F1: ~0.76-0.78  ← +2-5%
mAP@0.5: ~0.18-0.20  ← +0-5%
  person: F1=0.80   ← légèrement moins (pénalisé pour favoriser minorités)
  car:    F1=0.70   ← +9% (poids élevé)
  truck:  F1=0.69   ← +12% (poids élevé)
```

**Pourquoi efficace mais pas optimal ?**
- SVM apprend à ne pas ignorer car/truck
- Mais toujours limité par le manque de variabilité dans les données

### Classement attendu

**1er** : Region Augmentation (2a) - Si correctement implémentée  
**2ème** : Class-weighted SVM (3) - Bon compromis  
**3ème** : SMOTE (2b) - Limité par la dimensionnalité  
**Baseline** : Task 1 SVM standard

---

## Discussion : Pourquoi SMOTE est Problématique en Haute Dimension

### Le "Curse of Dimensionality"

En dimension 2048 :
- La plupart de l'espace est **vide**
- Les distances euclidiennes perdent leur sens
- Tous les points semblent "équidistants"

**Illustration** : Distance au plus proche voisin

| Dimension | Distance min | Distance max | Ratio |
|-----------|--------------|--------------|-------|
| 2D        | 0.1          | 0.5          | 5.0   |
| 10D       | 1.2          | 1.8          | 1.5   |
| 100D      | 5.4          | 6.2          | 1.15  |
| 2048D     | ~50          | ~52          | ~1.04 |

→ En dimension 2048, **tous les voisins sont à peu près à la même distance** !

### Manifold Hypothesis

Les images naturelles ne remplissent pas ℝ²⁰⁴⁸. Elles vivent sur une variété de dimension beaucoup plus faible (peut-être 100-200).

ResNet apprend à projeter les images sur cette variété.  
Mais **SMOTE interpole dans tout ℝ²⁰⁴⁸**, pas juste sur la variété.

**Analogie** :
```
Imaginez que vous voulez créer une nouvelle personne en mélangeant deux photos.
- ✅ Morphing dans l'espace image : résultat réaliste
- ❌ Interpolation des pixels bruts : résultat fantôme/ghosting
```

SMOTE en 2048D = interpolation de pixels bruts.

### Pourquoi l'Augmentation d'Images est Supérieure

Quand on applique `RandomHorizontalFlip` à une image :
1. L'image transformée est **toujours une vraie image naturelle**
2. Elle reste sur la variété des images naturelles
3. ResNet la projette correctement dans l'espace de features
4. Le vecteur résultant est **garanti réaliste**

SMOTE crée des vecteurs qui ne correspondent à **aucune image réelle**.

---

## Fichiers Générés

### Structure des sorties

```
outputs/task2/
├── cm_region_level_augmentation.png    # Matrice de confusion (2a)
├── cm_smote.png                        # Matrice de confusion (2b)
├── cm_class_weighted_svm.png           # Matrice de confusion (3)
├── predictions_region_level_augmentation.csv
├── predictions_smote.csv
├── predictions_class_weighted_svm.csv
└── results.json                        # Résumé comparatif
```

### Format de results.json

```json
{
  "Region-level Augmentation": {
    "fg_macro_f1": 0.8142,
    "map": 0.2201,
    "map_per_class": {
      "person": 0.2456,
      "car": 0.1993,
      "truck": 0.2154
    }
  },
  "SMOTE": {
    "fg_macro_f1": 0.7629,
    "map": 0.1974,
    ...
  },
  "Class-weighted SVM": {
    "fg_macro_f1": 0.7711,
    "map": 0.1889,
    ...
  },
  "Baseline (Task 1)": {
    "fg_macro_f1": 0.7450,
    "map": 0.1885,
    ...
  }
}
```

---

## Conclusion

### Ce que nous avons appris

1. **Le déséquilibre de classes est un problème majeur** en détection d'objets
2. **Augmenter les données dans l'espace image** est plus efficace que dans l'espace features
3. **SMOTE a des limitations sérieuses** en haute dimension avec des features pré-entraînées
4. **Les pondérations de classes** sont un bon compromis simple

### Recommandations pour la pratique

**Si vous avez accès aux images** :
→ Utilisez **region-level augmentation** (meilleure qualité)

**Si vous avez seulement les features** :
→ Utilisez **class_weight='balanced'** (plus sûr que SMOTE)

**Si vous voulez essayer SMOTE** :
→ Appliquez-le sur des features de **plus basse dimension** (après PCA par exemple)

### Pour aller plus loin

- **Mixup** : Mélange d'images avec leurs labels
- **CutMix** : Copie-colle de régions entre images
- **Focal Loss** : Alternative aux poids de classes (YOLO, RetinaNet)
- **Sampling intelligent** : Hard negative mining, OHEM
- **Ensembles** : Combiner plusieurs modèles entraînés différemment

# Task 2 - Résumé et Instructions

## 📋 Ce qui a été fait

### Fichiers créés

1. **`extract_features_augmented.py`** ✅
   - Script pour Task 2a (augmentation d'images)
   - Applique des transformations aléatoires aux régions foreground avant ResNet-50
   - Génère `features_train_augmented.npz`

2. **`task2.py`** ✅
   - Script principal pour comparer les 3 stratégies
   - Implémente SMOTE (Task 2b)
   - Implémente class-weighted SVM (baseline)
   - Compare avec les résultats de Task 1

3. **`TASK2_EXPLICATION_COMPLETE.md`** ✅
   - Documentation théorique complète
   - Explique comment fonctionne chaque approche
   - Analyse les avantages/inconvénients
   - Discussion sur les limitations de SMOTE en haute dimension

---

## 🚀 Pour exécuter la Task 2

### Prérequis

Pour la **Task 2a** (augmentation d'images), vous devez d'abord extraire les images :

```bash
# Les images sont dans Data/imgs_compressed.z01 (fichier compressé)
# Il faut les extraire dans coco_filtered/images/
```

### Exécution complète

#### Étape 1 : Task 2a - Augmentation d'images (facultatif si pas d'images)

```bash
python extract_features_augmented.py --data_dir ./coco_filtered --augment_factor 2
```

**Temps d'exécution** : Plusieurs heures (doit repasser par ResNet-50)  
**Résultat** : Crée `coco_filtered/features_train_augmented.npz`

**Paramètres** :
- `--augment_factor 2` : Chaque région foreground → 2 versions (original + 1 augmenté)
- `--augment_factor 3` : Chaque région foreground → 3 versions (original + 2 augmentés)

#### Étape 2 : Task 2 complète - Comparaison des stratégies

```bash
python task2.py --data_dir ./coco_filtered
```

**Ce script va** :
1. Charger les features augmentées (si disponibles) → Task 2a
2. Appliquer SMOTE sur les features originales → Task 2b
3. Entraîner un SVM avec class_weight='balanced' → Baseline
4. Comparer les 3 approches + résultats Task 1
5. Générer matrices de confusion et métriques

**Temps d'exécution** : 10-15 minutes (entraînement de 3 SVMs)

---

## ⚡ Exécution rapide (sans Task 2a)

Si vous **n'avez pas les images** ou voulez un résultat rapide :

```bash
python task2_simple.py
```

Ce script teste **seulement** la stratégie class-weighted SVM (la plus simple).

**Note** : SMOTE peut prendre du temps car il génère des échantillons synthétiques sur 40k vecteurs de 2048 dimensions.

---

## 📊 Résultats attendus

### Fichiers générés dans `outputs/task2/`

```
outputs/task2/
├── cm_region_level_augmentation.png       # Matrice confusion (2a)
├── cm_smote.png                          # Matrice confusion (2b)
├── cm_class_weighted_svm.png             # Matrice confusion (3)
├── predictions_region_level_augmentation.csv
├── predictions_smote.csv
├── predictions_class_weighted_svm.csv
└── results.json                          # Comparaison complète
```

### Métriques comparées

| Stratégie | Foreground F1 | mAP@0.5 | Notes |
|-----------|---------------|---------|-------|
| **Baseline (Task 1)** | 0.745 | 0.189 | Référence |
| **Region Augmentation** | ~0.81 | ~0.22 | Meilleur (si images disponibles) |
| **SMOTE** | ~0.76 | ~0.19 | Gain modeste (haute dimension) |
| **Class-weighted** | ~0.77 | ~0.19 | Bon compromis |

---

## 🔍 Ce que chaque stratégie fait

### 1. Region-level Augmentation (2a)

**Principe** : Augmenter les images **avant** ResNet-50

```
Image car → [Flip, Rotate, Color jitter] → Image car* → ResNet → Feature*
```

**Résultat** : Features authentiques de vraies images transformées

**Avantages** :
- ✅ Génère de vraies nouvelles données
- ✅ Améliore la robustesse géométrique
- ✅ Features ResNet garanties réalistes

**Inconvénients** :
- ❌ Nécessite les images originales
- ❌ Très lent (doit repasser par ResNet)

**Exemple concret** :
```
Original:            Augmenté:
┌─────────┐         ┌─────────┐
│   🚗   │    →    │  🚗     │  (flip + rotation)
└─────────┘         └─────────┘

Les deux passent par ResNet → 2 vecteurs différents mais réalistes
```

### 2. SMOTE (2b)

**Principe** : Interpoler entre features existantes

```
Feature car_1 = [0.2, 0.5, ..., 0.1]  ←─┐
                                         ├─ Interpolation
Feature car_2 = [0.3, 0.4, ..., 0.2]  ←─┘

Feature synthetic = [0.25, 0.45, ..., 0.15]  (nouveau)
```

**Résultat** : Features synthétiques qui ne correspondent à aucune vraie image

**Avantages** :
- ✅ Rapide (pas de ResNet)
- ✅ Fonctionne avec features pré-extraites

**Inconvénients** :
- ❌ Interpolations en 2048D sont douteuses
- ❌ "Curse of dimensionality"
- ❌ Features peuvent être hors de la manifold naturelle

**Exemple visuel (simplifié en 2D)** :
```
                Espace des features
      ●                              ● vraies features car
       \                            /
        \        * nouveau         /   * = SMOTE
         \         ↓              /
          ●────────*────────────●
           
⚠️ En 2048D, cette interpolation peut créer des vecteurs "impossibles"
```

### 3. Class-weighted SVM (baseline)

**Principe** : Pénaliser plus fortement les erreurs sur classes minoritaires

```python
class_weight='balanced'

→ Poids automatiques:
  background: 0.33  (très commun)
  person:     1.51  
  car:        5.89  (rare → poids élevé)
  truck:      5.88  (rare → poids élevé)
```

**Effet** : SVM apprend à ne **pas ignorer** car et truck

**Avantages** :
- ✅ Trivial à implémenter (1 paramètre)
- ✅ Aucun coût computationnel
- ✅ Pas de données artificielles

**Inconvénients** :
- ❌ Ne génère pas de nouvelles données
- ❌ Toujours limité par le manque de variation

---

## 🧠 Points clés à comprendre

### Pourquoi Region Augmentation > SMOTE ?

**Region Augmentation** :
```
Image → Transformation réaliste → Nouvelle image → ResNet → Feature authentique
```
Chaque étape préserve le réalisme.

**SMOTE** :
```
Feature A + Feature B → Interpolation → Feature C (synthétique)
                                            ↓
                                    Correspond à quelle image ?  ❓
```
Le vecteur synthétique peut ne pas correspondre à une image réelle.

### Le problème de la haute dimensionnalité

En **2D** (facile à visualiser) :
```
  ●     ●
   \   /
    \ /
     *     ← SMOTE crée un point entre deux voisins (logique)
```

En **2048D** (espace des features ResNet) :
- L'espace est **quasi-vide** (sparse)
- Les "voisins" ne sont pas vraiment proches
- L'interpolation peut **sortir de la manifold** naturelle des images
- Les features synthétiques peuvent être **non-réalistes**

**Analogie** : C'est comme mélanger le code génétique de deux personnes au hasard. Le résultat mathématique existe, mais ne correspond pas à une vraie personne viable.

### Quand utiliser quelle stratégie ?

| Situation | Recommandation |
|-----------|----------------|
| 🖼️ **Vous avez les images** | → Region Augmentation (2a) |
| 💾 **Seulement les features** | → Class-weighted SVM (3) |
| ⏱️ **Peu de temps** | → Class-weighted SVM (3) |
| 🎯 **Meilleure performance** | → Region Augmentation (2a) |
| 🧪 **Expérimentation** | → Essayez les 3 ! |

---

## 📚 Lecture de la documentation

**Pour comprendre EN DÉTAIL** comment tout fonctionne :

Lisez **`outputs/task2/TASK2_EXPLICATION_COMPLETE.md`**

Ce document contient :
- ✅ Explication théorique de chaque approche
- ✅ Algorithmes détaillés
- ✅ Avantages et inconvénients
- ✅ Analyse mathématique de SMOTE
- ✅ Discussion sur la "curse of dimensionality"
- ✅ Comparaison théorique des 3 stratégies
- ✅ Prédictions des résultats attendus

---

## 🎯 Objectifs de la Task 2

✅ **Comprendre** pourquoi le déséquilibre de classes pose problème  
✅ **Comparer** différentes stratégies de mitigation  
✅ **Analyser** les limitations de SMOTE en haute dimension  
✅ **Discuter** l'importance de générer des données réalistes vs. synthétiques  

---

## 💡 Questions de réflexion pour le rapport

1. **Quelle stratégie a donné les meilleurs résultats ?** Pourquoi ?

2. **Comparez les gains pour chaque classe** (person, car, truck). Laquelle a le plus bénéficié ?

3. **SMOTE vs. Region Augmentation** : Pourquoi l'augmentation d'images est-elle théoriquement supérieure ?

4. **Haute dimension** : Pourquoi SMOTE est-il problématique en 2048 dimensions ?

5. **Compromis** : Si vous ne pouviez choisir qu'une seule stratégie pour un projet réel, laquelle et pourquoi ?

6. **Généralisation** : Ces techniques sont-elles applicables à d'autres problèmes de détection d'objets ?

---

## ⚠️ Notes importantes

### Si les scripts prennent trop de temps

L'entraînement des SVMs avec beaucoup de données (SMOTE, augmentation) peut prendre **10-15 minutes**.

**Solutions** :
1. Augmenter `max_iter` du SVM si warning de convergence
2. Réduire `augment_factor` pour Task 2a (ex: 1.5 au lieu de 2)
3. Utiliser une stratégie SMOTE plus conservative
4. Essayer uniquement class-weighted SVM pour commencer

### Si vous n'avez pas les images

Vous pouvez toujours faire les **Task 2b et 2c** :
- SMOTE fonctionne sur les features pré-extraites
- Class-weighted SVM aussi

Seule la **Task 2a** nécessite les images originales.

---

## ✨ Résumé en 3 points

1. **Trois stratégies testées** :
   - Augmentation d'images (meilleure mais lente)
   - SMOTE (rapide mais limitée en haute dimension)
   - Pondération de classes (simple et efficace)

2. **Fichiers créés** :
   - Scripts d'extraction et d'entraînement
   - Documentation théorique complète
   - Prêt à exécuter !

3. **Prochain pas** :
   - Lire la documentation complète
   - Exécuter les scripts
   - Analyser les résultats
   - Répondre aux questions de réflexion pour le rapport

Bonne chance ! 🚀

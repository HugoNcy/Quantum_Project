# Projet 2 : Modèles génératifs photoniques hybrides
## Plan d'implémentation détaillé

Ce document est écrit pour être exploitable par un membre de l'équipe ou par une IA assistante sans contexte préalable. Il contient l'état validé de l'environnement, les pièges connus de l'API, l'architecture cible et le découpage en tâches avec critères d'acceptation.

---

## 0. Contexte et objectif

Projet étudiant en équipe de 3 (Hugo, Niels, Tony). Objectif : construire un modèle génératif hybride classique-quantique où un réseau PyTorch classique est couplé à une couche quantique photonique différentiable via MerLin (framework QML de Quandela, au-dessus de Perceval). Entraînement par loss MMD. Question de recherche : comparer l'expressivité générative d'un mesh MZI (Clements) et d'un mesh à tritters (3x3), et déterminer si le classement entre les deux tient sous bruit photonique réaliste (photons distinguables, pertes, mismatch entre bruit d'entraînement et bruit de déploiement).

Ce n'est pas une recherche d'avantage quantique. Le livrable est un dépôt GitHub reproductible avec une démonstration sur QPU réel (Quandela Cloud) si l'accès le permet.

Répartition : Niels et Tony sur le backend photonique (circuits Perceval, construction du tritter, exécution cloud). Hugo sur l'architecture générative, la boucle d'entraînement, la loss MMD, l'évaluation et toute la caractérisation de robustesse au bruit.

---

## 1. Environnement validé

Versions testées et fonctionnelles :

```
pip install perceval-quandela merlinquantum
```

Donne Perceval 1.2.3 et MerLin 0.4.0 (PyTorch tiré en dépendance). Python 3.12.

### 1.1 Ce qui a été validé en pratique

Un `QuantumLayer` construit via `CircuitBuilder` fait un forward et un backward corrects. Exemple minimal validé :

```python
import torch
import merlin

builder = merlin.CircuitBuilder(n_modes=4)
builder.add_angle_encoding(name="x")
builder.add_entangling_layer(model="mzi", trainable=True)

layer = merlin.QuantumLayer(
    input_size=4,
    builder=builder,
    n_photons=2,
)

x = torch.rand(8, 4) * torch.pi
out = layer(x)
# out shape: (8, 6) with 4 modes / 2 photons in no-bunching space
# rows sum to 1, gradients flow to layer.thetas
```

### 1.2 Pièges API connus (importants, découverts par test)

**Piège 1 : passer `builder=` et non `circuit=`.** `CircuitBuilder.build()` retourne un type interne MerLin incompatible avec le paramètre `circuit=` de `QuantumLayer` (qui attend un `pcvl.Circuit`). Passer directement le builder évite le problème et laisse MerLin inférer les specs des paramètres d'entrée et entraînables. Si on doit passer un circuit Perceval brut (cas du tritter, voir section 4), il faut déclarer manuellement `input_parameters` et `trainable_parameters` par préfixe de nom.

**Piège 2 : le bruit change la dimension de sortie.** Avec `indistinguishability=1.0`, MerLin restreint le calcul au sous-espace no-bunching (C(m,n) états). Dès que `indistinguishability < 1`, il bascule en espace de Fock complet (C(n+m-1,n) états). Deux layers avec des noise models différents produisent donc des vecteurs de tailles différentes et toute comparaison directe est invalide. Solution obligatoire : forcer l'espace de Fock complet sur TOUS les layers dès le départ, y compris le cas sans bruit :

```python
strategy = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)
layer = merlin.QuantumLayer(..., measurement_strategy=strategy)
```

Cette convention est non négociable pour tout le projet. Tout script de comparaison clean vs noisy doit l'appliquer.

**Piège 3 : `transmittance < 1` change la structure interne du layer.** Le noise model avec pertes ajoute des modes de pertes internes, donc un `state_dict` d'un layer sans pertes ne se charge pas dans un layer avec pertes (clé manquante `_photon_loss_transform._matrix`). Conséquence pratique : pour transférer des poids entraînés entre profils de bruit, copier uniquement les paramètres du circuit (les phases entraînables, par nom), pas le state_dict complet. Écrire un utilitaire `copy_circuit_params(src_layer, dst_layer)` dès la phase 1.

**Piège 4 : pas de tritter natif.** `add_entangling_layer` ne supporte que `model="mzi"` et `model="bell"`. Le mesh de tritters doit être construit à la main en Perceval pur (section 4).

### 1.3 Briques MerLin utiles identifiées

- `merlin.PhotonicGenerator` : wrapper génératif prêt à l'emploi. Prend un ou plusieurs `QuantumLayer`, un `output_adapter` (nn.Module) et un `latent` (par défaut `NormalLatent`).
- `merlin.models.VectorAdapter` : adapte la distribution de sortie du circuit vers un vecteur de dimension cible (pour données tabulaires 2D).
- `merlin.models.ImageAdapter` : équivalent pour images (utile si extension MNIST réduit).
- `pcvl.NoiseModel(brightness, indistinguishability, g2, transmittance, phase_imprecision, phase_error)` : le modèle de bruit, passé au paramètre `noise=` de QuantumLayer.

---

## 2. Architecture cible

```
z ~ N(0, I)  (latent classique, dim L)
   |
[optionnel : MLP classique d'adaptation, L -> input_size]
   |
QuantumLayer (angle encoding + mesh entraînable, m modes, n photons)
   |
distribution de probabilité sur les états de Fock (dim C(n+m-1, n), FOCK forcé)
   |
OutputAdapter (VectorAdapter, nn.Module classique)
   |
échantillon généré x_gen (dim 2 pour les données synthétiques)
```

Loss : MMD avec kernel gaussien multi-bandwidth entre batch généré et batch réel.

```python
def mmd_loss(x, y, bandwidths=(0.1, 0.5, 1.0, 2.0, 5.0)):
    # x: generated samples (B, D), y: target samples (B, D)
    def kernel(a, b):
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (2 * s ** 2)) for s in bandwidths) / len(bandwidths)
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
```

Tailles de départ recommandées : m=6 modes, n=3 photons (espace de Fock complet de dimension 56, gérable), latent L=4 à 8. Ne pas dépasser 8 modes / 4 photons en simulation exacte sans mesurer d'abord le coût.

---

## 3. Découpage en phases avec tâches et critères d'acceptation

### Phase 1 : Générateur minimal qui apprend (owner : Hugo)

Objectif : prouver que la pipeline complète apprend une distribution cible simple.

Tâches :
1. `data.py` : générateurs de datasets synthétiques 2D. Cibles : gaussienne simple, mixture de 2 gaussiennes, anneau. Chaque fonction retourne un tenseur (N, 2).
2. `model.py` : construction du générateur (QuantumLayer via CircuitBuilder mzi + VectorAdapter, ou PhotonicGenerator si l'API se prête bien au contrôle fin du noise). FOCK forcé.
3. `losses.py` : la MMD multi-bandwidth ci-dessus.
4. `train.py` : boucle d'entraînement standard (Adam, lr 1e-2 comme point de départ, 500 à 2000 steps), logging de la MMD par step dans un CSV ou un dict sauvegardé en JSON. Pas de print décoratifs.
5. `eval.py` : scatter plot cible vs généré, courbe de MMD, sauvegarde des figures en PNG.

Critères d'acceptation : la MMD décroît de façon monotone (au bruit d'optimisation près) sur la gaussienne simple, et le scatter généré recouvre visuellement la mixture de 2 gaussiennes. Seed fixée, résultat reproductible.

### Phase 1bis : Cible financière (owner : Hugo)

Remplacer la cible synthétique par une distribution empirique de log-returns (1D d'abord, puis 2D avec deux actifs corrélés). Source de données au choix : returns journaliers d'un actif liquide, ou réutilisation des données du projet Bitcoin existant.

Critères d'acceptation : comparaison histogramme réel vs généré, avec attention aux queues (QQ-plot ou comparaison des quantiles extrêmes). Remarque honnête attendue dans le rapport : un générateur à si petite échelle ne capturera probablement pas les queues épaisses correctement, et c'est un résultat en soi.

### Phase 2 : Pipeline de bruit et matrice de mismatch (owner : Hugo)

Tâches :
1. `noise.py` : définition d'une grille de profils de bruit nommés. Proposition de grille initiale :
   - P0 : indistinguishability=1.0 (référence propre)
   - P1 : 0.95, P2 : 0.90, P3 : 0.85
   - P4 : 0.95 + transmittance=0.9 (introduit les pertes, attention au piège 3)
2. Utilitaire `copy_circuit_params(src, dst)` : copie des phases entraînables par nom de paramètre entre deux layers de profils différents.
3. `train_noise_grid.py` : entraîner le générateur sous chaque profil, sauvegarder poids et MMD finale.
4. `mismatch_matrix.py` : pour chaque couple (profil d'entraînement A, profil d'évaluation B), charger les poids de A dans un modèle configuré avec B et mesurer la MMD contre la cible. Produit une matrice K x K.
5. Figures : heatmap de la matrice de mismatch, courbes MMD finale vs niveau d'indistinguishability.

Critères d'acceptation : la diagonale de la matrice (A=B) doit être meilleure ou égale au hors-diagonale en moyenne, sinon suspecter un bug. Résultats commentés honnêtement, y compris s'ils sont plats.

### Phase 3 : MZI vs tritters (owners : Niels et Tony pour le circuit, Hugo pour la comparaison)

Interface contractuelle entre le backend et la boucle d'entraînement : le backend fournit une fonction

```python
def build_circuit(mesh_type: str, n_modes: int) -> pcvl.Circuit:
    # mesh_type in {"mzi", "tritter"}
    # input encoding parameters named "x0", "x1", ...
    # trainable parameters named "theta0", "theta1", ...
```

Les conventions de nommage des paramètres sont l'interface. La boucle d'entraînement de Hugo déclare `input_parameters=["x"]` et `trainable_parameters=["theta"]` par préfixe et ne doit rien savoir de la topologie interne.

Construction du tritter (côté backend) : un tritter idéal est l'unitaire DFT 3x3 (coefficients en racines cubiques de l'unité). Il se décompose en beam splitters et phase shifters par Reck ou Clements sur 3 modes. Le mesh complet alterne des tritters sur les triplets de modes (0,1,2), (3,4,5) puis (1,2,3), (4,5,...) en couches décalées, avec des phase shifters entraînables entre les couches. Comparaison à budget équitable obligatoire : même n_modes, même n_photons, nombre de paramètres entraînables comparable entre les deux meshes (ajuster le nombre de couches).

Tâches Hugo : rejouer les phases 1 et 2 avec les deux meshes, produire le tableau final MZI vs tritter x profils de bruit.

Critères d'acceptation : tableau avec MMD finale par (mesh, profil), et réponse explicite à la question : le classement entre les deux meshes est-il stable sous bruit croissant et sous mismatch.

### Phase 4 : Exécution hardware Quandela (owners : Niels et Tony, analyse Hugo)

Objectif minimal : inférence seule (pas l'entraînement) du meilleur modèle sur Belenos ou Lucy via Quandela Cloud. Comparer la distribution mesurée sur hardware à la distribution simulée sous les différents profils de bruit.

Angle d'analyse : le hardware réel est un profil de bruit inconnu. L'écart simulation vs hardware est une mesure de noise mismatch en conditions réelles. Identifier quel profil simulé de la grille se rapproche le plus de la distribution hardware.

Critères d'acceptation : au moins une exécution réussie avec statistiques suffisantes, une figure distribution simulée vs mesurée, et un paragraphe d'analyse du mismatch.

### Phase 5 : Artefact final

Dépôt GitHub : README avec question de recherche et résultats clés, structure `src/` + `notebooks/` + `figures/`, instructions de reproduction, contributions statement explicite par membre. Formulation de la contribution de Hugo : conception et entraînement du modèle génératif hybride, loss MMD, évaluation, caractérisation complète de la robustesse au bruit et au noise mismatch, analyse simulation vs hardware.

---

## 4. Conventions de code (à respecter par tous, humains et IA)

- Commentaires en anglais.
- Pas de print décoratifs ni de prints de suivi d'exécution. Le logging passe par des fichiers (CSV/JSON) ou un logger si nécessaire.
- Le code existant fourni par un membre ne se modifie pas, il se complète.
- Code simple et concis, pas de sur-ingénierie (pas de framework de config, pas de classes inutiles).
- Seeds fixées dans tout script produisant un résultat.
- Toute comparaison clean vs noisy force ComputationSpace.FOCK partout.
- Les explications et interprétations vont dans des cellules Markdown des notebooks, pas dans des commentaires de code.

---

## 5. Risques et parades

1. Explosion combinatoire de la simulation : rester à 6 modes / 3 photons par défaut, mesurer le temps d'un forward avant toute montée en taille.
2. Tritter plus long que prévu : prototyper la construction du circuit dès la phase 1, en parallèle, même sans l'entraîner.
3. Accès Quandela Cloud incertain : les phases 1 à 3 constituent un artefact défendable sans hardware.
4. Résultats plats (classement stable sous bruit) : résultat négatif à assumer et documenter honnêtement, pas à survendre.
5. Instabilité de l'entraînement MMD : jouer sur les bandwidths avant de toucher à l'architecture, et vérifier la baseline (MMD entre deux batchs réels) comme plancher de référence.

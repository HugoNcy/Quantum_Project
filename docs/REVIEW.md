# Revue adversariale : src/, tests/ et briques MMD

Revue du 2026-07-05, ciblée sur les bugs physiques et méthodologiques silencieux, du type du bug de phase globale documenté dans CLAUDE.md. Aucun code n'a été modifié. Chaque affirmation chiffrée ci-dessous a été mesurée par un script jetable sur l'environnement du projet (Perceval 1.2.4, MerLin 0.4.0, seed torch 0).

Résumé des findings, du plus grave au moins grave :

| # | Sévérité | Sujet |
|---|----------|-------|
| F1 | Haute | Le budget "60 = 60" est nominal : 50 directions effectives pour mzi contre 32 pour tritter |
| F2 | Moyenne | Couverture des tritters à n_modes=6 : le triplet (4,5,6) est silencieusement droppé, modes 0 et 5 sous-mélangés |
| F3 | Moyenne | FOCK forcé supprime le garde-fou dimensionnel : eval.py évaluera silencieusement en propre un modèle entraîné bruité |
| F4 | Moyenne | Le plancher MMD est estimé sur une seule paire de batchs, variance mesurée d'un facteur 2.6 |
| F5 | Moyenne | Bandwidths MMD inadaptées à l'échelle des log-returns de la phase 1bis (signal atténué ~1000x) |
| F6 | Basse | Estimateur MMD biaisé (V-statistique) : valeurs finales non comparables entre batch sizes différents |
| F7 | Basse | Décomposition du tritter : déterminisme empirique mais non garanti par l'API |
| F8 | Basse | Seuil du test de sensibilité (1e-3) plus laxiste que le contrat CLAUDE.md (ordre 1e-2) |
| F9 | Basse | Le mapping input vers paramètres x repose sur l'ordre d'insertion, pas sur les noms |
| F10 | Info | L'état d'entrée par défaut est |1,0,1,0,1,0>, implicite et asymétrique vis-à-vis des blocs de tritters |

---

## F1. Budget de paramètres "comparable" : vrai nominalement, faux effectivement (HAUTE)

C'est le finding le plus sérieux parce qu'il touche directement la question de recherche (comparaison d'expressivité à budget égal).

Constat. `build_circuit("mzi", 6)` et `build_circuit("tritter", 6)` exposent chacun 60 paramètres theta, et le check 3 de tests/test_circuits.py passe. Mais une partie des thetas du mesh tritter sont des directions plates : deux phase shifters consécutifs sur un mode que rien ne mélange entre les deux se composent, PS(a) puis PS(b) vaut PS(a+b), donc un seul paramètre effectif. Mesure directe par rang de la jacobienne des 56 probabilités de sortie par rapport aux 60 thetas, empilée sur 4 inputs aléatoires (224 lignes, tolérance 1e-7 relative sur les valeurs singulières) :

- mzi : rang 50 sur 60
- tritter : rang 32 sur 60

Le mzi perd 10 directions (phases de sortie pures avant mesure et jauges globales, c'est attendu pour un mesh Clements). Le tritter en perd 28, presque moitié. La comparaison actuelle donne donc au mzi environ 56 pour cent de capacité effective en plus, tout en affichant des budgets égaux. Un résultat "le mzi est plus expressif" serait en partie un artefact de comptage.

Origine, trois mécanismes qui se cumulent :
1. Les couches de PS trainables couvrent les 6 modes, mais les couches de tritters aux offsets 1 et 2 ne mélangent que 3 modes (voir F2). Sur un mode non mélangé pendant k couches consécutives, les k phases collapsent en une seule. Exemple concret, mode 0 : mélangé uniquement aux couches 0 et 3, donc les phases des couches PS 1, 2, 3 collapsent (2 paramètres perdus), et les phases des couches PS 4, 5 collapsent aussi (1 perdu).
2. Dans le second mesh du sandwich, les phases traînantes d'un mode qui n'est plus jamais mélangé avant la mesure sont de la jauge pure : la mesure en nombre de photons est insensible aux phases par mode.
3. Une phase commune à tous les modes dans une couche PS est une phase globale, inobservable.

Correctif proposé (par ordre de préférence) :
1. Dans `_tritter_mesh`, ne placer les PS trainables que sur les modes couverts par la couche de tritters suivante. Cela élimine mécaniquement les collapses et la jauge traînante. Recompter ensuite et réajuster `n_layers` pour retrouver la parité, mais cette fois entre comptes utiles.
2. Quel que soit le choix, remplacer le critère du check 3 : comparer le rang de jacobienne (le code de mesure tient en 15 lignes, voir ci-dessus) plutôt que le nombre de noms de paramètres. C'est le seul comptage qui protège la conclusion scientifique.
3. À défaut, documenter honnêtement dans le rapport que le budget égal est nominal et donner les deux rangs.

Ne pas corriger en copiant l'astuce du mzi (compter large des deux côtés) : les deux architectures ne gaspillent pas dans les mêmes proportions, c'est justement le problème.

## F2. Couverture des modes par les couches de tritters à n_modes=6 (MOYENNE)

Réponse à la question posée : le triplet à l'offset 1 ne déborde ni ne wrappe, il est silencieusement droppé. `range(offset, n_modes - 2, 3)` avec offset 1 et n=6 donne uniquement start=1, donc le seul tritter placé est (1,2,3). Le triplet (4,5,6) n'existe pas et aucun wraparound (4,5,0) n'est tenté. Couverture mesurée par couche (mesh de 6 couches, offsets cycliques 0,1,2) :

| Couche | Offset | Tritters places | Modes non mélangés |
|--------|--------|-----------------|--------------------|
| 0 | 0 | (0,1,2), (3,4,5) | aucun |
| 1 | 1 | (1,2,3) | 0, 4, 5 |
| 2 | 2 | (2,3,4) | 0, 1, 5 |
| 3 | 0 | (0,1,2), (3,4,5) | aucun |
| 4 | 1 | (1,2,3) | 0, 4, 5 |
| 5 | 2 | (2,3,4) | 0, 1, 5 |

Les modes 0 et 5 ne sont mélangés que dans 2 couches sur 6, les modes 2 et 3 dans les 6. La connectivité complète existe (0 atteint 5 via les couches 0 et 3) mais le mesh est fortement biaisé vers le centre. C'est un choix de topologie défendable pour un chip planaire sans croisements, pas un bug en soi, mais il est invisible dans le code et alimente directement F1.

Correctif proposé : garder le drop (le wraparound n'est pas physique sur un chip planaire sans permutations), mais l'expliciter dans la docstring de `_tritter_mesh` avec le tableau ci-dessus, et le mentionner dans le rapport comme propriété de l'architecture tritter comparée. Si on veut une variante plus homogène en simulation, une couche sur deux peut utiliser un placement aligné en haut (dernier triplet flush sur le mode n-1), à tester avec le même test de sensibilité.

## F3. FOCK forcé partout supprime le canari dimensionnel du mismatch de bruit (MOYENNE)

Les deux seuls sites d'instanciation de QuantumLayer (model.py:28 et tests/test_circuits.py:47) forcent bien `MeasurementStrategy.probs(ComputationSpace.FOCK)`. Aucune fuite aujourd'hui. Mais la convention a un effet secondaire méthodologique : puisque clean et noisy vivent désormais dans la même base de dimension 56, charger des poids dans le mauvais profil de bruit ne peut plus jamais planter. Le piège 2 du plan est éliminé au prix de la disparition du symptôme.

Le point de fuite concret est eval.py : il reconstruit `PhotonicGenerator(...)` sans argument noise (donc propre) et log.json n'enregistre pas le profil de bruit du run. Dès la phase 2, évaluer un run bruité via eval.py produira des figures propres, sans erreur, sans avertissement. C'est exactement le genre de bug silencieux que cette revue cherche.

Correctif proposé : ne pas modifier eval.py (règle du projet), mais dans les scripts de phase 2 à venir, (1) enregistrer le nom du profil de bruit dans le log JSON, (2) écrire un eval_noise.py qui lit ce champ et reconstruit le NoiseModel avant de charger les poids, (3) refuser d'évaluer un log sans champ de profil. Ajouter au passage un assert sur la dimension 56 dans tout script de comparaison, non pour le mismatch (il ne le détecte plus) mais comme garde-fou FOCK.

## F4. Plancher MMD estimé sur une seule paire de batchs (MOYENNE)

train.py:33 calcule le plancher réel-contre-réel sur une unique paire de tranches de 256 points. Mesuré sur 8 paires disjointes de two_gaussians : moyenne 0.0030, min 0.0018, max 0.0048. Un facteur 2.6 entre paires. La ligne de plancher des figures d'eval.py peut donc être décalée du simple au double selon la paire tirée, ce qui change visuellement la conclusion "le modèle atteint le plancher ou pas".

Correctif proposé : moyenner le plancher sur 16 paires (une boucle de 3 lignes) dans les futurs scripts de comparaison, et reporter moyenne et écart-type. train.py existant ne se modifie pas, mais tout script de phase 2 ou 3 qui reporte une MMD finale doit utiliser ce plancher moyenné.

## F5. Bandwidths MMD face à l'échelle des données (MOYENNE, concerne la phase 1bis)

Pour les cibles synthétiques actuelles, rien à signaler : distance médiane inter-points mesurée à 1.17 sur two_gaussians, bien encadrée par les bandwidths (0.1, 0.5, 1, 2, 5). Le terme 0.1 est surtout du bruit de diagonale mais la mixture le dilue.

Pour la phase 1bis en revanche, les log-returns journaliers vivent à l'échelle 0.01. La plus petite bandwidth est alors 10 fois le diamètre des données et tous les noyaux valent quasiment 1 partout. Mesure : MMD entre N(0, 0.01) et N(0, 0.02), deux distributions très différentes (facteur 2 sur la vol), vaut 1.6e-4, contre 3.7e-6 entre deux batchs de même loi. Le signal existe encore (ratio 44) mais il est atténué d'un facteur environ 1000 par rapport aux cibles synthétiques : gradients minuscules, entraînement au ras du bruit d'optimisation, et courbes MMD illisibles.

Correctif proposé : standardiser les returns (diviser par leur écart-type empirique) avant la MMD et regénérer à cette échelle, ou ajouter des bandwidths issues de l'heuristique de la médiane calculée sur la cible. La première option est plus simple et garde DEFAULT_BANDWIDTHS inchangé.

## F6. Estimateur MMD biaisé, valeurs non comparables entre batch sizes (BASSE)

losses.py implémente la V-statistique (les termes diagonaux k(x,x)=1 sont inclus dans les .mean()). Espérance non nulle sous P=Q, environ (2/B)(1 - E[k]) soit ~3e-3 à B=256 sur two_gaussians, ce qui colle au plancher mesuré en F4. Ce n'est pas un problème pour l'entraînement (la diagonale est constante, gradient nul) et la comparaison au plancher compense le biais. Le vrai risque est ailleurs : le biais dépend de B, donc deux MMD finales obtenues avec des batch_size différents ne sont pas comparables. batch_size est un paramètre libre de train().

Correctif proposé : figer B=256 dans tous les scripts de comparaison mesh x profil de bruit de la phase 3, ou reporter systématiquement MMD moins plancher (même B, même estimateur). Une ligne de doc dans le README du dépôt final suffit.

## F7. Reproductibilité de la décomposition du tritter (BASSE)

Réponse à la question posée : mesurée reproductible, mais pas garantie. `pcvl.Circuit.decomposition` est un solveur itératif sans seed exposée. Tests effectués : deux appels de `tritter()` dans le même process diffèrent de 3.7e-9 au max sur l'unitaire, deux process séparés de 3.7e-9 aussi, et la phase globale du bloc converge vers zéro (l'unitaire obtenu est la DFT elle-même, pas une version déphasée). Les deux meshes d'un même `build_circuit` utilisent deux appels distincts de `tritter()` et diffèrent donc de ~1e-9, ce qui est physiquement négligeable (des ordres de grandeur sous toute imprecision de phase matérielle).

Nuance importante quand même : une phase de bloc non nulle ne serait PAS une phase globale du circuit complet. Un e^(i alpha) sur un tritter (1,2,3) est une phase relative physique par rapport aux modes 0, 4, 5. Aujourd'hui alpha vaut ~1e-9 donc le point est théorique, mais si une future version de Perceval change le comportement du solveur, le circuit changerait silencieusement.

Correctif proposé : figer les phases une fois pour toutes. Soit dumper les valeurs numériques du circuit décomposé dans src/ (un tuple de constantes et une reconstruction explicite), soit ajouter au test un check inter-appels : deux `tritter()` successifs égaux élément par élément à 1e-8. La deuxième option est 3 lignes et suffit.

## F8. Seuil du test de sensibilité plus laxiste que le contrat (BASSE)

CLAUDE.md exige une sensibilité d'ordre 1e-2, tests/test_circuits.py:59 asserte `std > 1e-3`. Les valeurs mesurées (5.2e-2 pour mzi, 4.2e-2 pour tritter) passent le seuil du contrat avec de la marge. Par ailleurs le test prend le max de la std sur les 56 dimensions de sortie, critère généreux : une seule composante sensible suffirait à faire passer un circuit globalement mort.

Correctif proposé : remonter le seuil à 1e-2 sur le max, et ajouter un second assert à 1e-3 sur la moyenne des std, qui vérifie que la sensibilité est distribuée et pas concentrée sur une composante.

## F9. Le mapping input vers x repose sur l'ordre d'insertion, pas sur les noms (BASSE)

Vérifié dans le source MerLin (pcvl_pytorch/locirc_to_tensor.py:329-339) : les paramètres d'un préfixe sont collectés par `startswith` dans l'ordre de `circuit.get_parameters()`, c'est-à-dire l'ordre d'insertion dans le circuit, sans tri des noms. Deux conséquences :
1. build_circuit insère x0..x{n-1} dans l'ordre des modes, donc le mapping composante d'input vers mode est correct à tout n_modes, y compris au-delà de 9 où un tri lexicographique aurait permuté x10 avant x2. Mais cette correction vient de l'ordre d'insertion, pas des noms. Un futur circuit qui insérerait les x dans le désordre aurait des noms mensongers et aucun test ne le détecterait.
2. Le matching par `startswith` capturerait tout paramètre dont le nom commence par x ou theta (un "xi" ou un "theta_bs" resté libre serait silencieusement absorbé). Les phases internes du tritter sont figées après décomposition donc invisibles, c'est vérifié par le check "aucun paramètre libre hors x et theta" du test 3.

Correctif proposé : ajouter au test un check d'alignement, par exemple `[p.name for p in circuit.get_parameters() if p.name.startswith("x")] == [f"x{i}" for i in range(n_modes)]`, qui fige le contrat ordre-egale-nom.

## F10. État d'entrée implicite |1,0,1,0,1,0> (INFO)

Avec `n_photons=3` et sans `input_state` explicite, MerLin place les photons en |1,0,1,0,1,0>. C'est asymétrique vis-à-vis des blocs de tritters de la couche 0 : le bloc (0,1,2) reçoit 2 photons, le bloc (3,4,5) en reçoit 1. Pas un bug, mais un degré de liberté caché de la comparaison mzi contre tritter, et un point de reproductibilité si une version de MerLin change ce défaut.

Correctif proposé : passer `input_state` explicitement dans les scripts de comparaison de la phase 3 et l'enregistrer dans les logs JSON.

---

## Ce qui a été vérifié et n'a rien donné

Pour éviter de refaire le travail : l'unitaire du tritter isolé égale la DFT 3x3 à 6e-9 près (phase de bloc ~0). Le sandwich est respecté dans les deux meshes, aucun encodage avant le premier mesh. Les deux circuits passent le test de sensibilité avec un ordre de grandeur de marge. Les sorties somment à 1 en base de Fock complète de dimension 56. mmd_loss(x, x) vaut exactement 0. Le mesh mzi de GenericInterferometer est bien un Clements rectangulaire à 15 MZIs de 2 phases. Aucune instanciation de QuantumLayer sans measurement_strategy dans le dépôt.

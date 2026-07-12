# Kit de Survie Hors-Ligne — Perceval 1.2.3 / MerLin 0.4.0

> **Comment ce kit a été fabriqué.** Chaque snippet et chaque exercice de ce document a été
> **réellement exécuté** le 2026-07-05 sur la stack épinglée du projet
> (`perceval-quandela==1.2.3`, `merlinquantum==0.4.0`), dans DEUX environnements : le venv
> local (Python 3.11.6, torch 2.12.1+cpu) et le conteneur officiel du projet
> (Python 3.12.13, torch 2.5.1+cpu) — les 6 exercices passent dans les deux. Les sorties
> affichées sont les sorties réelles, pas des sorties supposées. Les signatures ont été
> extraites par `inspect.signature` sur le package installé. Quand une croyance de l'équipe
> s'est révélée incomplète, ce kit documente le comportement observé.
>
> Les exercices existent aussi en scripts exécutables dans `docs/exercises/ex1_*.py` à `ex6_*.py`.

---

## Sommaire

- [Partie 0 — Aide-mémoire express](#partie-0--aide-mémoire-express)
- [Partie 1 — Exercices de crash-test](#partie-1--exercices-de-crash-test)
  - [Exercice 1 : Perceval pur (BS + PS)](#exercice-1)
  - [Exercice 2 : MerLin CircuitBuilder](#exercice-2)
  - [Exercice 3 : QuantumLayer et l'espace de Fock](#exercice-3)
  - [Exercice 4 : Rétropropagation](#exercice-4)
  - [Exercice 5 : Le bruit et le piège des poids](#exercice-5)
  - [Exercice 6 (bonus) : Le tritter à la main](#exercice-6)
- [Partie 2 — Documentation sur-mesure](#partie-2--documentation-sur-mesure)
  - [A. Écosystème Perceval](#a-écosystème-perceval-pcvl)
  - [B. Écosystème MerLin](#b-écosystème-merlin)
- [Partie 3 — Annexes](#partie-3--annexes)
  - [Le tritter : recette validée](#annexe-1--le-tritter-recette-validée)
  - [copy_circuit_params canonique](#annexe-2--copy_circuit_params-canonique)
  - [MMD multi-bandwidth de référence](#annexe-3--mmd-multi-bandwidth-de-référence)
  - [Troubleshooting : erreurs réellement rencontrées](#annexe-4--troubleshooting)

---

# Partie 0 — Aide-mémoire express

## Imports standards du projet

```python
import numpy as np
import torch
import perceval as pcvl
import merlin

FOCK = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)  # everywhere, always
```

## Dimensions de sortie (vérifiées par exécution)

La sortie d'une `QuantumLayer` est un tenseur `(batch, D)` de probabilités (chaque ligne somme à 1).
`D` dépend de l'espace de calcul **et du contenu du NoiseModel** :

| Configuration | Formule | m=4, n=2 | m=6, n=3 |
|---|---|---|---|
| no-bunching (`UNBUNCHED`, défaut) | $C(m,n)$ | 6 | 20 |
| Fock complet (`FOCK`), sans perte | $C(n+m-1,n)$ | **10** | **56** |
| + `transmittance<1` ou `brightness<1` | $\sum_{k=0}^{n} C(k+m-1,k)$ | 15 | 84 |
| + `g2>0` (sans perte) | $\sum_{k=n}^{2n} C(k+m-1,k)$ | 65 | — |
| NoiseModel complet (pertes **et** g2) | $\sum_{k=0}^{2n} C(k+m-1,k)$ | 70 | — |

**Conséquence pratique :** forcer `FOCK` rend les comparaisons *clean vs indistinguishability<1*
directement compatibles (10 = 10), mais **les pertes et g2 étendent encore la dimension**
(10 → 15 → 70). Deux modèles ne sont dimensionnellement compatibles que si leurs NoiseModels
impliquent le même nombre de photons possible. Pour la grille P0–P4 du projet : P0–P3
(indistinguishability seule) partagent D=56 à 6 modes / 3 photons ; **P4 (transmittance=0.9) a D=84**.
L'`OutputAdapter` de P4 a donc une entrée différente — c'est une raison de plus pour laquelle
`copy_circuit_params` ne copie QUE les phases du circuit.

## Les 3 pièges en bref

1. **`QuantumLayer(builder=...)`, jamais `circuit=builder.build()`.** `build()` retourne un
   type interne MerLin ; `circuit=` exige un vrai `pcvl.Circuit`. Exception tritter :
   `circuit=<pcvl.Circuit>` + `input_parameters=["x"]` + `trainable_parameters=["theta"]`.
2. **FOCK forcé partout.** Sans ça, le passage au bruit change silencieusement D (6→10).
   Validé : MerLin émet même un `UserWarning` disant que le bruit de source force FOCK.
3. **Jamais de `load_state_dict` entre profils de bruit différents.** `transmittance<1`
   ajoute la clé `_photon_loss_transform._matrix`. Utiliser `copy_circuit_params` (annexe 2).

---

# Partie 1 — Exercices de crash-test

<a id="exercice-1"></a>
## Exercice 1 : Perceval pur — circuit à base de BS et PS

**Énoncé.** Construire un circuit de 3 modes : un beam splitter sur les modes (0,1), un phase
shifter *paramétré symboliquement* (nommé `phi0`) sur le mode 1, un beam splitter sur (1,2).
Lister les paramètres libres, assigner `phi0 = π/4`, calculer l'unitaire 3×3 et vérifier son
unitarité.

**Ce qu'on attend de vous.** Savoir instancier `pcvl.Circuit`, ajouter des composants sur des
modes précis, créer un paramètre nommé avec `pcvl.P`, et extraire l'unitaire numérique. C'est
exactement la gymnastique requise pour construire le tritter en Phase 3.

**Solution.**

```python
"""Exercise 1 -- Pure Perceval: build a small circuit from BS and PS."""

import numpy as np
import perceval as pcvl

# A 3-mode circuit: BS on modes (0,1), a parametrized PS on mode 1,
# then BS on modes (1,2).
circuit = pcvl.Circuit(3, name="warmup")
circuit.add((0, 1), pcvl.BS())
circuit.add(1, pcvl.PS(phi=pcvl.P("phi0")))
circuit.add((1, 2), pcvl.BS())

# Named parameters are the interface used later by MerLin (prefix matching).
params = circuit.get_parameters()
print("parameter names:", [p.name for p in params])

# Assign a value to the symbolic parameter, then compute the 3x3 unitary.
params[0].set_value(np.pi / 4)
u = circuit.compute_unitary()
print("unitary:\n", np.round(u, 3))
print("is unitary:", np.allclose(u @ np.conjugate(u.T), np.eye(3)))
```

**Sortie réelle observée.**

```text
parameter names: ['phi0']
unitary:
 [[ 0.707+0.j     0.   +0.707j  0.   +0.j   ]
 [-0.354+0.354j  0.354+0.354j  0.   +0.707j]
 [-0.354-0.354j -0.354+0.354j  0.707+0.j   ]]
is unitary: True
```

**À retenir.**
- `circuit.add(mode, composant)` : `mode` est un entier (composant à 1 mode) ou un tuple
  d'entiers *contigus* (composant multi-modes). Le composant est placé sur ces modes.
- `pcvl.P("nom")` crée un paramètre symbolique ; `pcvl.P` est un alias de
  `perceval.utils.parameter.Parameter`. Tant qu'un paramètre n'a pas de valeur,
  `compute_unitary()` échoue avec `AssertionError: All parameters must be defined`.
- `p.set_value(v)` assigne ; `circuit.get_parameters()` retourne les paramètres **libres**.
- Le BS par défaut est 50:50 (convention `Rx`, `theta=π/2`) : la phase `i` sur la branche
  réfléchie est visible dans l'unitaire (`0.707j`).

<a id="exercice-2"></a>
## Exercice 2 : MerLin CircuitBuilder — construction fluide

**Énoncé.** Construire avec `merlin.CircuitBuilder` un circuit de 4 modes comportant un
encodage en angles (préfixe `x`) et une couche d'intrication MZI entraînable. Inspecter les
préfixes de paramètres d'entrée déclarés par le builder.

**Ce qu'on attend de vous.** Le réflexe « builder » : c'est la voie standard du projet pour le
mesh MZI (le tritter est l'exception, exercice 6).

**Solution.**

```python
"""Exercise 2 -- MerLin CircuitBuilder: fluent circuit construction."""

import merlin

builder = merlin.CircuitBuilder(n_modes=4)

# Input encoding: classical features become phase-shifter angles.
# The name defines the parameter prefix ("x0", "x1", ...).
builder.add_angle_encoding(name="x")

# Trainable universal mesh: Clements grid of MZIs.
builder.add_entangling_layer(model="mzi", trainable=True)

# Inspect what the builder produced (this is what QuantumLayer will infer).
print("input parameter prefixes:", builder.input_parameter_prefixes)
```

**Sortie réelle observée.**

```text
input parameter prefixes: ['x']
```

**À retenir.**
- Les méthodes retournent le builder (chaînables).
- `add_angle_encoding(modes=None, name=None, *, scale=1.0, subset_combinations=False,
  max_order=None)` — sans `modes`, tous les modes sont encodés. `scale` multiplie les features
  avant de les injecter comme phases (utile : `scale=np.pi` pour des features dans [0,1]).
- `add_entangling_layer(modes=None, *, trainable=True, model="mzi", name=None, ...)` —
  seuls `model="mzi"` et `model="bell"` existent. **Pas de tritter natif.**
- Autres méthodes disponibles (validé par introspection) : `add_rotations`,
  `add_superpositions`, `add_memristive_ps`, `from_circuit`, `to_pcvl_circuit`, `build`.

<a id="exercice-3"></a>
## Exercice 3 : QuantumLayer et l'espace de Fock (LA règle du projet)

**Énoncé.** Instancier deux `QuantumLayer` identiques (4 modes, 2 photons, builder de
l'exercice 2) : l'une avec la mesure par défaut, l'autre en forçant
`MeasurementStrategy.probs(ComputationSpace.FOCK)`. Faire passer un batch `(8, 4)` et imprimer
les shapes de sortie. Vérifier que chaque ligne somme à 1.

**Ce qu'on attend de vous.** Voir de vos yeux la différence 6 vs 10 — la source du piège le
plus coûteux du projet.

**Solution.**

```python
"""Exercise 3 -- QuantumLayer and the Fock space (THE critical project rule)."""

import torch
import merlin

def make_builder():
    builder = merlin.CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(name="x")
    builder.add_entangling_layer(model="mzi", trainable=True)
    return builder

x = torch.rand(8, 4) * torch.pi

# --- Layer A: default measurement (no-bunching subspace) ---
layer_default = merlin.QuantumLayer(
    input_size=4,
    builder=make_builder(),
    n_photons=2,
)
out_default = layer_default(x)
print("default output shape:", tuple(out_default.shape), "-> C(4,2) = 6")

# --- Layer B: full Fock space, forced (PROJECT RULE, non-negotiable) ---
strategy = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)
layer_fock = merlin.QuantumLayer(
    input_size=4,
    builder=make_builder(),
    n_photons=2,
    measurement_strategy=strategy,
)
out_fock = layer_fock(x)
print("FOCK output shape:", tuple(out_fock.shape), "-> C(5,2) = 10")

# Both are probability distributions: rows sum to 1.
print("rows sum to 1 (default):", torch.allclose(out_default.sum(dim=1), torch.ones(8)))
print("rows sum to 1 (FOCK):   ", torch.allclose(out_fock.sum(dim=1), torch.ones(8)))
```

**Sortie réelle observée.**

```text
default output shape: (8, 6) -> C(4,2) = 6
FOCK output shape: (8, 10) -> C(5,2) = 10
rows sum to 1 (default): True
rows sum to 1 (FOCK):    True
```

**Preuve du basculement silencieux sous bruit** (validé séparément) : la même layer par défaut,
avec `noise=pcvl.NoiseModel(indistinguishability=0.9)`, sort `(8, 10)` au lieu de `(8, 6)` —
sans erreur, sans avertissement bloquant. C'est pour cela que la convention FOCK est
non négociable :

```text
clean, default strategy : (2, 6)   output_size: 6
noisy, default strategy : (2, 10)  output_size: 10   <-- silent switch!
clean, forced FOCK      : (2, 10)
noisy, forced FOCK      : (2, 10)  <-- stable
```

**À retenir.**
- Le défaut de `MeasurementStrategy.probs()` est `ComputationSpace.UNBUNCHED` (vérifié par
  signature). Ne jamais s'y fier dans ce projet.
- `layer.output_size` donne D sans faire de forward ; `layer.output_keys` liste les états de
  Fock dans l'ordre des colonnes (tuples d'occupation, ex. `(2,0,0,0)`, `(1,1,0,0)`, ...).

<a id="exercice-4"></a>
## Exercice 4 : rétropropagation à travers la couche quantique

**Énoncé.** Faire passer un batch aléatoire `(8, 4)` dans la `QuantumLayer` FOCK, calculer une
loss factice, appeler `.backward()` et inspecter les gradients des paramètres quantiques via
`named_parameters()`.

**Ce qu'on attend de vous.** Constater que la couche se comporte comme n'importe quel
`nn.Module` — et découvrir un piège de loss factice au passage.

**Solution.**

```python
"""Exercise 4 -- PyTorch autograd through the quantum layer."""

import torch
import merlin

builder = merlin.CircuitBuilder(n_modes=4)
builder.add_angle_encoding(name="x")
builder.add_entangling_layer(model="mzi", trainable=True)

layer = merlin.QuantumLayer(
    input_size=4,
    builder=builder,
    n_photons=2,
    measurement_strategy=merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK),
)

x = torch.rand(8, 4) * torch.pi   # batch of 8 samples, 4 features
out = layer(x)                    # (8, 10) probability rows

# --- Degenerate loss: rows sum to 1, so out.mean() == 1/10 is constant ---
loss_bad = out.mean()
loss_bad.backward(retain_graph=True)
grads_bad = {n: p.grad.norm().item() for n, p in layer.named_parameters()}
print("grad norms with out.mean() (constant loss):", grads_bad)  # ~0.0

# --- Proper dummy loss: probability of the first Fock state ---
layer.zero_grad()
loss = out[:, 0].mean()
loss.backward()
for name, p in layer.named_parameters():
    print(f"{name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}, "
          f"grad_norm={round(p.grad.norm().item(), 6)}")
```

**Sortie réelle observée.**

```text
grad norms with out.mean() (constant loss): {'el': 9.704632653040335e-09}
el: shape=(12,), requires_grad=True, grad_norm=0.002574
```

**À retenir.**
- **Piège découvert en validant cet exercice** : `out.mean()` est une loss factice DÉGÉNÉRÉE.
  Les lignes sont des probabilités qui somment à 1, donc la moyenne vaut `1/D` constante et son
  gradient est nul (le `9.7e-09` observé est du bruit numérique). Toute loss de test doit
  dépendre de la *forme* de la distribution, ex. `out[:, 0].mean()`.
- Les phases entraînables d'un layer construit par builder apparaissent comme **un seul tenseur
  nommé `el`** (entangling layer) — ici shape `(12,)` pour le mesh MZI 4 modes. Avec un circuit
  brut (exercice 6), le nom vient du préfixe déclaré (`theta`).
- `layer.zero_grad()`, `retain_graph=True`, optimiseurs Adam/SGD : tout PyTorch standard
  fonctionne tel quel.

<a id="exercice-5"></a>
## Exercice 5 : le bruit et le piège des poids

**Énoncé.** Créer deux layers identiques, l'une sans bruit, l'autre avec
`pcvl.NoiseModel(transmittance=0.9)`. Comparer leurs `state_dict`. Montrer que
`lossy.load_state_dict(clean.state_dict())` échoue, expliquer pourquoi, puis écrire et tester
`copy_circuit_params(src_layer, dst_layer)`.

**Ce qu'on attend de vous.** Reproduire le piège 3 en conditions réelles et disposer de
l'utilitaire canonique dont toute la Phase 2 (matrice de mismatch) dépend.

**Solution.**

```python
"""Exercise 5 -- Noise with losses and the state_dict trap."""

import torch
import perceval as pcvl
import merlin

FOCK = merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK)

def make_layer(noise=None):
    builder = merlin.CircuitBuilder(n_modes=4)
    builder.add_angle_encoding(name="x")
    builder.add_entangling_layer(model="mzi", trainable=True)
    return merlin.QuantumLayer(
        input_size=4,
        builder=builder,
        n_photons=2,
        measurement_strategy=FOCK,
        noise=noise,
    )

clean = make_layer()
lossy = make_layer(noise=pcvl.NoiseModel(transmittance=0.9))

print("clean state_dict keys:", sorted(clean.state_dict().keys()))
print("lossy state_dict keys:", sorted(lossy.state_dict().keys()))

# --- The trap: full state_dict transfer fails across noise profiles ---
try:
    lossy.load_state_dict(clean.state_dict())
    print("load_state_dict: unexpectedly succeeded")
except RuntimeError as e:
    print("load_state_dict FAILED as expected:\n", str(e)[:300])

# --- The fix: copy only trainable circuit phases, by name ---
def copy_circuit_params(src_layer, dst_layer):
    """Copy trainable circuit phases between layers with different noise.

    Never copy the full state_dict across noise profiles: lossy layers
    contain extra internal tensors (loss modes) that clean layers lack.
    """
    src = dict(src_layer.named_parameters())
    with torch.no_grad():
        for name, p in dst_layer.named_parameters():
            if name in src and src[name].shape == p.shape:
                p.copy_(src[name])

copy_circuit_params(clean, lossy)

# Verify: phases are now identical.
src = dict(clean.named_parameters())
ok = all(torch.equal(p, src[n]) for n, p in lossy.named_parameters() if n in src)
print("phases identical after copy_circuit_params:", ok)
```

**Sortie réelle observée.**

```text
clean state_dict keys: ['el']
lossy state_dict keys: ['_photon_loss_transform._matrix', 'el']
load_state_dict FAILED as expected:
 Error(s) in loading state_dict for QuantumLayer:
	Missing key(s) in state_dict: "_photon_loss_transform._matrix".
phases identical after copy_circuit_params: True
```

**À retenir.**
- Le message d'erreur documenté par l'équipe est reproduit **mot pour mot** : la clé
  `_photon_loss_transform._matrix` (un buffer interne, PAS un paramètre entraînable) n'existe
  que dans le layer avec pertes.
- `copy_circuit_params` copie par nom via `named_parameters()` : seuls les vrais paramètres
  entraînables (les phases) transitent, les buffers de structure restent en place.
- Sens du transfert indifférent : clean→lossy et lossy→clean fonctionnent tous deux (le buffer
  n'apparaît pas dans `named_parameters()`).
- Rappel dimension : avec `transmittance=0.9`, `output_size` passe de 10 à **15**
  (états à 0, 1 ou 2 photons). L'`OutputAdapter` branché derrière doit donc être
  dimensionné par profil — copier ses poids entre P0–P3 et P4 n'a pas de sens.

<a id="exercice-6"></a>
## Exercice 6 (bonus) : le tritter à la main — l'exception au piège 1

**Énoncé.** Construire en Perceval pur un circuit de 3 modes : encodage `x0..x2` (PS), tritter
fixe (DFT 3×3 décomposée en BS+PS), phases entraînables `theta0..theta2` (PS), second tritter.
L'envelopper dans une `QuantumLayer` en déclarant **manuellement** `input_parameters` et
`trainable_parameters` par préfixe. Vérifier forward et backward.

**Ce qu'on attend de vous.** La répétition générale du point de friction API n°1 de la Phase 3.

**Solution.** (extraits clés — script complet dans `docs/exercises/ex6_tritter_manual.py`)

```python
import numpy as np
import torch
import perceval as pcvl
import merlin

def dft3_unitary():
    """The ideal tritter: 3x3 DFT matrix, coefficients in cube roots of unity."""
    w = np.exp(2j * np.pi / 3)
    u = np.array([[1, 1, 1],
                  [1, w, w**2],
                  [1, w**2, w]]) / np.sqrt(3)
    return pcvl.MatrixN(u)

def tritter_circuit():
    """Fixed DFT3 decomposed into BS + PS (hardware-realistic layout)."""
    mzi = pcvl.BS() // pcvl.PS(pcvl.P("t")) // pcvl.BS() // pcvl.PS(pcvl.P("p"))
    decomposed = pcvl.Circuit.decomposition(
        dft3_unitary(),
        mzi,
        phase_shifter_fn=lambda phi: pcvl.PS(phi),  # phi is the SOLVED VALUE
        shape="triangle",
    )
    assert decomposed is not None, "decomposition failed"
    assert not decomposed.get_parameters(), "tritter must have no free parameters"
    return decomposed

# Assemble: encoding PS -> tritter -> trainable PS -> tritter
circuit = pcvl.Circuit(3, name="tritter_block")
for i in range(3):
    circuit.add(i, pcvl.PS(phi=pcvl.P(f"x{i}")))
circuit.add(0, tritter_circuit(), merge=True)
for i in range(3):
    circuit.add(i, pcvl.PS(phi=pcvl.P(f"theta{i}")))
circuit.add(0, tritter_circuit(), merge=True)

# Manual declaration: this is the tritter exception to trap 1.
layer = merlin.QuantumLayer(
    circuit=circuit,                    # raw pcvl.Circuit
    input_size=3,
    input_parameters=["x"],             # prefix -> x0, x1, x2
    trainable_parameters=["theta"],     # prefix -> theta0, theta1, theta2
    n_photons=2,
    measurement_strategy=merlin.MeasurementStrategy.probs(merlin.ComputationSpace.FOCK),
)

x = torch.rand(8, 3) * torch.pi
out = layer(x)
out[:, 0].mean().backward()
```

**Sortie réelle observée.**

```text
free parameters: ['x0', 'x1', 'x2', 'theta0', 'theta1', 'theta2']
output shape: (8, 6) -> C(4,2) = 6 states (3 modes, 2 photons, FOCK)
params with grads: [('theta', 0.296971)]
```

**À retenir — trois faits validés qui ne sont écrits nulle part ailleurs :**
1. `Circuit.decomposition` avec un template `pcvl.BS()` nu **échoue et retourne `None`**
   (silencieusement !). Il faut un template MZI complet :
   `pcvl.BS() // pcvl.PS(pcvl.P("t")) // pcvl.BS() // pcvl.PS(pcvl.P("p"))`.
2. `phase_shifter_fn` reçoit **la valeur de phase résolue** (un float), pas un indice —
   malgré l'annotation de type `Callable[[int], ACircuit]`. `lambda phi: pcvl.PS(phi)` donne
   un circuit entièrement numérique, sans paramètre libre parasite.
3. La décomposition (triangle) du DFT3 donne 15 composants et reproduit la matrice à `1e-5`
   près (vérifié par `np.allclose`).
- Raccourci simulation : `pcvl.Unitary(dft3_unitary())` insère la matrice telle quelle
  (1 composant, pas de layout physique). Suffisant pour simuler, insuffisant pour raisonner
  hardware.
- Les phases entraînables apparaissent groupées sous le nom du préfixe : ici un tenseur
  `theta` de shape `(3,)`.

---

# Partie 2 — Documentation sur-mesure

## A. Écosystème Perceval (`pcvl`)

### `pcvl.Circuit`

**Rôle.** Le circuit photonique linéaire bas niveau : une séquence ordonnée de composants
posés sur des modes. C'est un objet Perceval (PAS un `nn.Module`) ; il ne connaît ni les
tenseurs ni le gradient.

**Signature validée.** `Circuit(m: int, name: str = None)` — `m` = nombre de modes.

**Ajouter des composants.**

```python
import numpy as np
import perceval as pcvl

c = pcvl.Circuit(3, name="demo")
c.add(0, pcvl.PS(phi=1.2))               # int: single-mode component on mode 0
c.add((0, 1), pcvl.BS())                 # tuple: multi-mode component on modes 0-1
c.add(0, sub_circuit, merge=True)        # merge a sub-circuit's components in place
c2 = pcvl.BS() // pcvl.PS(pcvl.P("a"))   # // composes on modes starting at 0
```

- `add` retourne le circuit (chaînable). `merge=True` fusionne les composants d'un
  sous-circuit (nécessaire pour insérer un circuit décomposé).
- **Erreur réelle rencontrée** : `AssertionError: Only unitary components can compose a linear
  optics circuit, use Experiment for non-unitary` — déclenchée notamment si on `add(...)` un
  objet `None` (cas typique : une `decomposition` qui a échoué en silence).

**Paramètres variables.**

```python
p = pcvl.P("theta0")          # pcvl.P is perceval.utils.parameter.Parameter
c.add(1, pcvl.PS(phi=p))
free = c.get_parameters()     # list of FREE (unassigned) Parameters
free[0].set_value(np.pi / 3)  # assign
u = c.compute_unitary()       # numeric m x m matrix; fails if any param undefined
```

Les **noms** des paramètres sont l'interface avec MerLin (matching par préfixe). Convention du
projet : entrées `x0, x1, ...`, entraînables `theta0, theta1, ...`.

**Décomposition d'une unitaire arbitraire** (validée sur le tritter, voir exercice 6) :

```python
mzi = pcvl.BS() // pcvl.PS(pcvl.P("t")) // pcvl.BS() // pcvl.PS(pcvl.P("p"))
decomposed = pcvl.Circuit.decomposition(
    target_matrix,                        # pcvl.MatrixN (wrap a numpy complex array)
    mzi,                                  # FULL MZI template (bare BS fails -> None)
    phase_shifter_fn=lambda phi: pcvl.PS(phi),   # receives the solved float value
    shape="triangle",                     # or "rectangle" (Clements)
)
# Returns None on failure -- ALWAYS check.
```

**Composants utilisés dans le projet.**

| Composant | Signature validée | Notes |
|---|---|---|
| `pcvl.PS` | `PS(phi, max_error=0)` | 1 mode ; `phi` float ou `pcvl.P` |
| `pcvl.BS` | `BS(theta=π/2, phi_tl=0, phi_bl=0, phi_tr=0, phi_br=0, convention=Rx)` | 2 modes ; défaut = 50:50 ; `BS.r_to_theta(R)` convertit une réflectivité |
| `pcvl.Unitary` | `Unitary(MatrixN)` | insère une matrice fixe telle quelle |

### `pcvl.NoiseModel`

**Rôle.** Agrège toutes les imperfections physiques. Par défaut (tout à `None`), la simulation
est parfaite. Se passe à `QuantumLayer(noise=...)` (validé) ou s'attache à un
`pcvl.Experiment` (`experiment.noise = ...`, voie alternative documentée par MerLin).

**Signature validée (1.2.3).**

```python
pcvl.NoiseModel(
    brightness: float = None,           # P(photon emitted when requested), source
    indistinguishability: float = None, # HOM visibility between photons; 1.0 = perfect
    g2: float = None,                   # multi-photon emission probability (2nd-order corr.)
    g2_distinguishable: bool = None,    # whether the parasitic 2nd photon is distinguishable
    transmittance: float = None,        # P(photon survives propagation); losses
    phase_imprecision: float = None,    # phase-shifter resolution limit
    phase_error: float = None,          # random phase offset (calibration error)
)
```

**Sémantique et impact, paramètre par paramètre.**

| Paramètre | Physique | Effet sur le calcul (validé) |
|---|---|---|
| `indistinguishability < 1` | photons partiellement discernables (spectre/timing) | interférence HOM partielle ; **D passe de C(m,n) à C(n+m-1,n)** si non forcé |
| `transmittance < 1` | pertes en propagation | photons perdus → états à k ≤ n photons ; **D=15** (4m/2p) ; **ajoute le buffer `_photon_loss_transform._matrix`** |
| `brightness < 1` | la source n'émet pas toujours | même extension d'espace que les pertes (**D=15** observé) ; MerLin combine `brightness × transmittance` en probabilité de survie globale |
| `g2 > 0` | émission parasite de 2 photons | états jusqu'à 2n photons ; **D=65** (4m/2p, g2 seul), **70** avec pertes |
| `phase_error` | offset aléatoire des phases | le circuit exécuté ≠ circuit programmé ; échantillonné (`n_phase_error_samples` sur la layer) |
| `phase_imprecision` | résolution finie des shifters | quantification des phases |

**Snippet minimal (exécuté).**

```python
noise = pcvl.NoiseModel(brightness=0.9, indistinguishability=0.95, g2=0.01,
                        transmittance=0.9, phase_imprecision=0.001, phase_error=0.01)
layer = merlin.QuantumLayer(..., noise=noise,
                            measurement_strategy=FOCK)
out = layer(x)     # rows still sum to 1 (loss states carry the missing mass)
```

**Grille du projet (rappel).** P0 : tout parfait ; P1/P2/P3 : indistinguishability
0.95/0.90/0.85 ; P4 : indistinguishability 0.95 + transmittance 0.9.

## B. Écosystème MerLin (`merlin`)

### `merlin.CircuitBuilder`

**Rôle.** Construction fluide du circuit paramétré, avec déclaration automatique des specs
d'entrée/entraînables que `QuantumLayer` saura inférer. C'est la voie standard du projet pour
le mesh MZI.

**Signatures validées.**

```python
CircuitBuilder(n_modes: int)

add_angle_encoding(modes: list[int] | None = None, name: str | None = None, *,
                   scale: float = 1.0, subset_combinations: bool = False,
                   max_order: int | None = None) -> CircuitBuilder

add_entangling_layer(modes: list[int] | None = None, *, trainable: bool = True,
                     model: str = "mzi", name: str | None = None,
                     trainable_inner: bool | None = None,
                     trainable_outer: bool | None = None) -> CircuitBuilder
```

**Snippet minimal (exécuté).**

```python
builder = merlin.CircuitBuilder(n_modes=4)
builder.add_angle_encoding(name="x")                    # all modes, prefix "x"
builder.add_entangling_layer(model="mzi", trainable=True)
print(builder.input_parameter_prefixes)                 # ['x']
```

**Le piège `.build()` (reproduit, message exact).**

```python
built = builder.build()
type(built)                      # merlin.core.circuit.Circuit  <-- INTERNAL type
builder.to_pcvl_circuit()        # perceval...Circuit           <-- real pcvl.Circuit

merlin.QuantumLayer(circuit=builder.build(), ...)
# ValueError: The number of modes should be a strictly positive integer
#             (got Circuit(n_modes=4, components=5))
```

- `build()` retourne un type **interne MerLin**, incompatible avec `circuit=` qui attend un
  `pcvl.Circuit`. → **Toujours passer `builder=builder`** et laisser MerLin inférer.
- **Découverte utile (non documentée par l'équipe)** : `builder.to_pcvl_circuit()` exporte un
  vrai `pcvl.Circuit` — pratique pour inspecter/afficher le circuit que le builder a construit,
  ou pour l'hybrider avec du Perceval pur.

### `merlin.QuantumLayer`

**Rôle.** Le `nn.Module` différentiable : entrée tenseur `(batch, input_size)` de features,
sortie tenseur `(batch, D)` de probabilités sur les états de Fock. Les phases du circuit sont
des `nn.Parameter` optimisables par n'importe quel optimiseur PyTorch.

**Signature complète validée (0.4.0).**

```python
QuantumLayer(
    input_size: int | None = None,
    builder: CircuitBuilder | None = None,       # route 1 (standard)
    circuit: pcvl.Circuit | None = None,         # route 2 (tritter/custom)
    experiment: pcvl.Experiment | None = None,   # route 3 (detectors/noise via Experiment)
    input_state: StateVector | pcvl.BasicState | list | tuple | None = None,
    n_photons: int | None = None,                # alternative to input_state
    trainable_parameters: list[str] | None = None,   # name prefixes (route 2)
    input_parameters: list[str] | None = None,       # name prefixes (route 2)
    amplitude_encoding: bool = False,
    measurement_strategy: MeasurementStrategyLike | None = None,
    return_object: bool = False,
    noise: pcvl.NoiseModel | None = None,        # direct noise attachment (validated)
    n_phase_error_samples: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
)
```

**Les trois routes de construction.**

```python
# Route 1 -- builder (standard, MZI mesh): parameters inferred automatically
layer = merlin.QuantumLayer(input_size=4, builder=builder, n_photons=2,
                            measurement_strategy=FOCK)

# Route 2 -- raw pcvl.Circuit (tritter): declare parameter prefixes MANUALLY
layer = merlin.QuantumLayer(circuit=my_pcvl_circuit, input_size=3,
                            input_parameters=["x"], trainable_parameters=["theta"],
                            n_photons=2, measurement_strategy=FOCK)

# Route 3 -- pcvl.Experiment (noise + detectors as single source of truth)
exp = pcvl.Experiment(my_pcvl_circuit)
exp.noise = pcvl.NoiseModel(transmittance=0.9)
layer = merlin.QuantumLayer(input_size=3, experiment=exp,
                            input_parameters=["x"], input_state=[1, 1, 0])
```

**Attributs utiles (validés).**
- `layer.output_size` : D, sans forward.
- `layer.output_keys` : liste ordonnée des états de Fock (tuples d'occupation) — l'index i de
  la sortie correspond à `output_keys[i]`. Ex. lossy : `(2,0,0,0), (1,0,0,0), (0,0,0,0), ...`.
- `layer.named_parameters()` : phases entraînables. Nom `el` (builder) ou préfixe déclaré
  (circuit brut).

**Notes.**
- `n_photons=n` place par défaut les photons un par mode sur les n premiers modes ;
  `input_state=[1,0,1,0]` (liste d'occupations) donne le contrôle exact.
- `n_phase_error_samples` : nombre d'échantillons tirés quand `phase_error > 0`.
- `return_object=True` retourne un objet typé au lieu du tenseur brut — non utilisé dans le
  projet.

### `merlin.MeasurementStrategy` et `merlin.ComputationSpace`

**Rôle.** Contrôlent ce que la layer mesure et dans quel espace d'états elle exprime sa sortie.

**Signature validée.**

```python
MeasurementStrategy.probs(
    computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,  # <-- default!
    grouping: LexGrouping | ModGrouping | None = None,
    *, occupancy_readout: bool = False,
) -> MeasurementStrategy

list(merlin.ComputationSpace)
# [ComputationSpace.FOCK, ComputationSpace.UNBUNCHED, ComputationSpace.DUAL_RAIL]
```

**UNBUNCHED vs FOCK.**
- `UNBUNCHED` (défaut) : sous-espace « au plus 1 photon par mode », dimension $C(m,n)$.
  Physiquement : post-sélection sur les événements sans groupement.
- `FOCK` : tous les états à n photons (groupements inclus), dimension $C(n+m-1,n)$.
- `DUAL_RAIL` : encodage qubit par paires de modes (hors périmètre du projet).

**Pourquoi forcer FOCK est vital (démontré, exercice 3).** Dès que le NoiseModel rend les
photons partiellement discernables, MerLin bascule en FOCK de lui-même — le warning réel émis :

```text
UserWarning: Noisy simulations with source noise currently use ComputationSpace.FOCK.
Other computation spaces are not yet supported for noise models.
```

Un modèle entraîné en UNBUNCHED (D=6) et évalué sous bruit (D=10) casse ou compare
l'incomparable. **La convention du projet** : le module `FOCK` défini une fois, importé partout.

**Nuance validée qui manquait aux notes d'équipe** : forcer FOCK ne fige PAS D une fois pour
toutes — les pertes et g2 étendent l'espace aux nombres de photons ≠ n (tableau Partie 0).
FOCK garantit la *cohérence de convention*, pas l'égalité dimensionnelle entre tous les profils.

### `merlin.PhotonicGenerator`

**Rôle.** Le wrapper génératif prêt à l'emploi : latent → layer(s) quantique(s) → mesures →
adapter → échantillons. C'est un `nn.Module` (entraînable de bout en bout).

**Signature et méthodes validées.**

```python
PhotonicGenerator(
    layers: QuantumLayer | Sequence[QuantumLayer],
    output_adapter: nn.Module,
    latent: LatentDistribution | None = None,     # default: NormalLatent
    *, count: int | None = None,
)

gen.generate(batch_size, *, device=None, dtype=None) -> torch.Tensor  # end-to-end sampling
gen.forward(z: torch.Tensor) -> torch.Tensor                          # explicit latent
gen.sample_latent(n) -> torch.Tensor
gen.latent_dim                                                        # property

NormalLatent(dim: int, mean: float = 0.0, std: float = 1.0)
```

**Snippet minimal (exécuté, shapes réelles).**

```python
layer = merlin.QuantumLayer(input_size=4, builder=builder, n_photons=2,
                            measurement_strategy=FOCK)
gen = merlin.PhotonicGenerator(
    layers=layer,
    output_adapter=merlin.VectorAdapter(2),   # target dimension: 2 (our 2D toys)
    latent=merlin.NormalLatent(dim=4),        # z ~ N(0, I_4); dim must match input_size
)

samples = gen.generate(16)          # (16, 2)  -- one call does latent->quantum->adapter
z = gen.sample_latent(8)            # (8, 4)
x_gen = gen(z)                      # (8, 2)   -- differentiable w.r.t. gen.parameters()
list(gen.named_parameters())        # [('layers.0.el', (12,)), ...adapter params if any]
```

**Boucle d'entraînement type (MMD).**

```python
opt = torch.optim.Adam(gen.parameters(), lr=1e-2)
for step in range(n_steps):
    x_gen = gen(gen.sample_latent(batch_size))
    x_real = sample_target(batch_size)
    loss = mmd_loss(x_gen, x_real)            # annexe 3
    opt.zero_grad(); loss.backward(); opt.step()
```

### `merlin.models.VectorAdapter`

**Rôle.** Adapte la distribution de sortie quantique (dimension D, dépendante du circuit et du
bruit) vers un vecteur de dimension cible fixe (2 pour nos jouets synthétiques). Disponible en
`merlin.VectorAdapter` et `merlin.models.VectorAdapter` (même classe).

**Signature validée.** `VectorAdapter(size: int)` — `size` = dimension de sortie cible.

**Point de typage important (validé par signature).**
`VectorAdapter.forward(measurements: GeneratorMeasurements) -> torch.Tensor` : l'adapter ne
consomme PAS un tenseur brut mais un objet `GeneratorMeasurements` produit par le
`PhotonicGenerator`. **Conséquence : utiliser `VectorAdapter` à l'intérieur d'un
`PhotonicGenerator`**, qui fait la plomberie. Pour brancher un circuit dans un
`nn.Sequential` fait main, utiliser plutôt un `nn.Linear(layer.output_size, target_dim)`
classique (ou `merlin.LexGrouping`/`ModGrouping` pour réduire D par regroupement d'états).

**Variantes.** `merlin.ImageAdapter` : équivalent pour des sorties image (extension MNIST
réduite éventuelle). `merlin.OutputAdapter` : classe de base pour écrire un adapter custom.

---

# Partie 3 — Annexes

## Annexe 1 — Le tritter : recette validée

Le tritter idéal est l'unitaire DFT 3×3 :

$$U_{\text{tritter}} = \frac{1}{\sqrt{3}}\begin{pmatrix}1&1&1\\1&\omega&\omega^2\\1&\omega^2&\omega\end{pmatrix},\qquad \omega=e^{2i\pi/3}$$

Recette Perceval complète et validée (reproduit la matrice à 1e-5, 15 composants) :

```python
def dft3_unitary():
    w = np.exp(2j * np.pi / 3)
    u = np.array([[1, 1, 1], [1, w, w**2], [1, w**2, w]]) / np.sqrt(3)
    return pcvl.MatrixN(u)

def tritter_circuit():
    mzi = pcvl.BS() // pcvl.PS(pcvl.P("t")) // pcvl.BS() // pcvl.PS(pcvl.P("p"))
    decomposed = pcvl.Circuit.decomposition(
        dft3_unitary(), mzi,
        phase_shifter_fn=lambda phi: pcvl.PS(phi),
        shape="triangle")
    assert decomposed is not None and not decomposed.get_parameters()
    return decomposed
```

Pour le mesh complet de la Phase 3 (m=6) : couches de tritters sur (0,1,2) et (3,4,5), puis
couche décalée sur (1,2,3), avec des PS entraînables `theta{k}` entre les couches, assemblées
par `circuit.add(offset, tritter_circuit(), merge=True)`. Budget équitable vs MZI : compter
`len(circuit.get_parameters())` côté tritter et `layer_mzi.state_dict()['el'].numel()` côté
builder, et ajuster le nombre de couches.

## Annexe 2 — `copy_circuit_params` canonique

```python
def copy_circuit_params(src_layer, dst_layer):
    """Copy trainable circuit phases between layers with different noise models.

    Full state_dict transfer fails across noise profiles (lossy layers own an
    extra '_photon_loss_transform._matrix' buffer). Parameters are matched by
    name and shape; buffers are left untouched.
    """
    src = dict(src_layer.named_parameters())
    with torch.no_grad():
        for name, p in dst_layer.named_parameters():
            if name in src and src[name].shape == p.shape:
                p.copy_(src[name])
```

Testé dans les deux sens (clean↔lossy). À étendre avec un compteur de paramètres copiés si un
jour deux architectures différentes sont mélangées par erreur (un retour silencieux « 0 copié »
serait un bug vicieux).

## Annexe 3 — MMD multi-bandwidth de référence

```python
def mmd_loss(x, y, bandwidths=(0.1, 0.5, 1.0, 2.0, 5.0)):
    # x: generated samples (B, D), y: target samples (B, D)
    def kernel(a, b):
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (2 * s ** 2)) for s in bandwidths) / len(bandwidths)
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
```

Rappels : (1) la MMD entre deux batchs *réels* donne le plancher de référence — la loss ne
descendra pas en dessous ; (2) piège de l'exercice 4 : ne jamais utiliser `out.mean()` comme
loss de debug sur des probabilités.

## Annexe 4 — Troubleshooting

Erreurs **réellement rencontrées** pendant la validation de ce kit, avec leur cause :

| Message | Cause réelle | Correctif |
|---|---|---|
| `ValueError: The number of modes should be a strictly positive integer (got Circuit(...))` | `QuantumLayer(circuit=builder.build())` — type interne MerLin | passer `builder=` ; ou `builder.to_pcvl_circuit()` |
| `Missing key(s) in state_dict: "_photon_loss_transform._matrix"` | `load_state_dict` entre profils avec/sans pertes | `copy_circuit_params` (annexe 2) |
| `AssertionError: Only unitary components can compose a linear optics circuit, use Experiment for non-unitary` | `circuit.add(x, None)` — souvent une `decomposition` qui a retourné `None` | template MZI complet dans `decomposition` ; toujours tester `is not None` |
| `AssertionError: All parameters must be defined to compute numeric unitary matrix` | `compute_unitary()` avec des `pcvl.P` non assignés | `p.set_value(...)` ou `phase_shifter_fn=lambda phi: pcvl.PS(phi)` |
| Gradient ≈ 0 (1e-8) sur une loss factice | `out.mean()` est constant (probabilités sommant à 1) | loss dépendant de la forme : `out[:, 0].mean()`, MMD, ... |
| `UserWarning: Noisy simulations with source noise currently use ComputationSpace.FOCK...` | rappel MerLin du piège 2 — bénin si FOCK déjà forcé | convention FOCK partout |
| `pip` : `SSLError(SSLCertVerificationError...)` vers PyPI | interception TLS locale (antivirus/proxy) — rencontré sur la machine de Tony | `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ...` ou passer par le conteneur |

**Réflexes de débogage hors-ligne.**
- `inspect.signature(obj.__init__)` : la signature exacte, sans internet.
- `layer.output_keys` : quand une dimension surprend, lister les états pour comprendre
  *quels* états sont apparus (pertes ? g2 ?).
- `[p.name for p in circuit.get_parameters()]` : vérifier les préfixes avant de déclarer
  `input_parameters`/`trainable_parameters`.
- `help(merlin.CircuitBuilder.add_angle_encoding)` : les docstrings sont installées avec le
  package — la doc de secours est déjà sur votre machine.

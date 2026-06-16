# Quantum Reservoir Computing under NISQ Constraints

## Research Question
“Under realistic NISQ noise, does the most expressive data encoding still yield the best quantum reservoir computing performance, or does a depth-induced noise penalty reverse the ranking in favor of shallower encodings?”

## Hypothesis
Below a certain noise threshold, expressive encodings (amplitude, entanglement-based) outperform shallow ones (angle). Above it, the ranking inverts: the noise penalty from added circuit depth outweighs their expressivity advantage.

## Project Structure

- `src/`: Code source principal
  - `data_generation/`: Génération de séries temporelles (ex: NARMA).
  - `encodings/`: Stratégies d'encodage (Angle, Amplitude, Feature Maps).
  - `noise/`: Modèles de bruit réalistes (Qiskit Aer, calibrations IBM).
  - `reservoir/`: Implémentation du Quantum Reservoir (QRC).
  - `experiments/`: Scripts pour lancer les grilles d'expériences.
- `notebooks/`: Carnets Jupyter pour l'exploration, l'analyse et la génération des figures.
- `data/`: Dossier pour les jeux de données (ignoré par git).
- `results/`: Dossier pour les métriques et figures générées (ignoré par git).

## Setup & Installation

1. Créer un environnement virtuel :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Configuration :
   Copiez `.env.example` en `.env` et ajoutez votre token IBM Quantum si nécessaire.

## Contributions Prévues
- **Hugo** : Pipeline de bruit réaliste, exécution de la grille expérimentale complète (encodage × intensité de bruit), analyse des résultats et validation de l'hypothèse de croisement. Potentiel run sur backend réel.
- **Tony & Niels** : Infrastructure de bout en bout (génération NARMA, implémentation du Réservoir, readout, implémentation des circuits d'encodage).

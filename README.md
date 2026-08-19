# Gradual-Pattern-Guided-Correlation

Projet de mémoire de recherche (Master 2 Data Science Engineering) comparant deux
approches d'extraction de motifs graduels fréquents : **GGC** (Correlation-Based
Gradual Extractor) et **GRITE**, sur plusieurs jeux de données réels.

## Contexte

Les motifs graduels décrivent des tendances du type *"plus A augmente, plus B
augmente"* (`A+,B+`) ou *"plus A augmente, plus B diminue"* (`A+,B-`). Ce projet
implémente et compare deux algorithmes d'extraction de ces motifs :

- **GGC** — filtre d'abord les attributs par corrélation significative avant de
  générer les itemsets, ce qui réduit l'espace de recherche.
- **GRITE** — génère les itemsets par extension progressive (Apriori), avec
  élagage basé sur les matrices binaires et le AND.

L'objectif est d'évaluer leurs performances respectives (temps d'exécution,
mémoire utilisée, nombre de motifs extraits) selon la taille du dataset et les
seuils de support/corrélation.

## Structure du projet

```
Gradual-Pattern-Guided-Correlation/
├── data/
│   ├── raw/                          # datasets bruts (non versionnés)
│   └── processed/                    # datasets nettoyés (générés, non versionnés)
├── src/
│   ├── config.py                     # chemins, datasets, grille de test
│   ├── utils.py                      # PerformanceMonitor, gestion du timeout
│   ├── algorithms/
│   │   ├── common.py                 # fonctions partagées (matrices binaires)
│   │   ├── ggc.py                    # algorithme GGC
│   │   ├── grite.py                  # algorithme GRITE
│   │   └── analyse_comparative.py    # orchestration de la comparaison
│   ├── data/
│   │   ├── load_dataset.py           # chargement CSV/Excel
│   │   └── processing.py             # nettoyage (valeurs manquantes, doublons)
│   └── visualisation/
│       └── comparative_visualisation.py  # graphiques et rapport texte
├── scripts/
│   └── prepare_data.py               # prépare data/processed/ à partir de data/raw/
├── tests/                            # tests unitaires (pytest)
├── notebooks/                        # exploration ponctuelle
├── main.py                           # point d'entrée principal
└── requirements.txt
```

## Installation

```bash
git clone git@github.com:Marie-IPK/Gradual-Pattern-Guided-Correlation.git
cd Gradual-Pattern-Guided-Correlation
pip install -r requirements.txt
```

Dépendances principales : `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `psutil`, `pytest`.

## Préparation des données

Placer les 4 fichiers bruts dans `data/raw/`, puis générer les versions nettoyées :

```bash
python3 -m scripts.prepare_data
```

Ce script charge chaque dataset défini dans `src/config.py::DATASETS`, applique
le pipeline de nettoyage (`select_numeric_column`, `remove_duplicates`,
`handle_missing_values`), et sauvegarde le résultat dans `data/processed/`.

## Utilisation

Toutes les commandes s'exécutent **depuis la racine du projet**.

```bash
# Lancer l'analyse comparative sur un seul dataset
python3 main.py --dataset air_quality

# Lancer l'analyse sur tout les datasets à la suite (comportement par défaut)
python3 main.py
```

Les noms disponibles pour `--dataset` correspondent aux clés définies dans
`src/config.py::DATASETS`.

### Sorties générées

Pour chaque dataset, un dossier `GGC_GRITE_Analysis_<nom_dataset>/` est créé, contenant :

- `results.json` — résultats bruts de toutes les configurations testées
- `01_ggc_execution_time.png` à `03_ggc_total_patterns.png` — GGC : métriques vs seuil de corrélation
- `04_grite_execution_time.png` à `06_grite_total_patterns.png` — GRITE : métriques vs seuil de support
- `07_algorithm_comparison.png` — comparaison directe GGC vs GRITE
- `08_status_distribution.png` — répartition des statuts d'exécution (succès/timeout/erreur)
- `analysis_report.txt` — rapport texte récapitulatif

## Configuration de la grille de test

Modifiable dans `src/config.py` :

```python
DATA_SIZES = [20, 30, 50, 500, 3000]
SUPPORT_RATIOS = [0.03, 0.05, 0.07]
CORRELATION_THRESHOLDS = [0.25, 0.50, 0.75]
ANALYSIS_TIMEOUT_SECONDS = 350
```

> GRITE peut être coûteux sur de grands datasets ; un timeout est appliqué par
> configuration, avec récupération des motifs partiellement extraits.

## Tests

```bash
pytest tests/ -v
```

encore en construction ...

## Auteur

**Ipouk Marie Victoire D.** (Mary_Data) — Master 2 Data Science Engineering, ENSPY Yaoundé
Contact : ipoukmarievictoire1@gmail.com

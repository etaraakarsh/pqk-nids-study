# Entangling structure shapes projected quantum kernels but rarely their accuracy

Code, frozen protocol files and results for the paper

> A. Etar, J. Soni and H. Upadhyay, *Entangling structure shapes projected quantum kernels but rarely their accuracy*, submitted to *Quantum Machine Intelligence* (2026).

The study asks which structural property of a quantum feature map controls how much each qubit of a projected quantum kernel still responds to the input, and whether that property predicts classification accuracy. It was run as a data audit and four preregistered stages on network intrusion detection data (NF-UNSW-NB15-v2, UNSW-NB15), with Fashion-MNIST as a control. Every hypothesis and threshold was frozen and hashed before the corresponding data were seen.

All quantum results come from exact (noiseless) statevector simulation. No quantum hardware is needed.

## Main results

| Question | Outcome |
|---|---|
| Does causal-cone size govern per-qubit concentration? (Stage 1, H1) | Rejected. Cone size explains at most 37% of the variation |
| Does coupling degree govern it? (Stage 2, H2) | Mixed. More couplings mean more concentration in 90% of within-graph tests, but the regression criteria fail |
| Does concentration predict accuracy? (Stage 3) | No, for the ZZ map |
| Do these results hold for ten circuit families? (Stage 4) | Cone effect small in 7 of 9 families, degree effect in 5 of 9. Concentration predicts accuracy in 1 of 10 families. Two-qubit gates add more than 0.02 macro-F1 in 3 of 81 comparisons |
| Quantum kernel against classical models | Non-inferior to an RBF support vector machine in 87 of 90 settings, 0.02 to 0.05 behind tree ensembles on UNSW-NB15 |

## Repository layout

```
pqk-nids-study/
├── README.md                              this file
├── DATA.md                                where to download the datasets and how to check them
├── requirements.txt                       Python packages
├── LICENSE
├── CITATION.cff
│
├── phase0_data_and_baselines.ipynb        Data audit, MODE = "quick" (3 seeds). Its splits are the ones frozen in Stage 1
├── phase0_full_data_and_baselines.ipynb   Same notebook, MODE = "full" (10 seeds, 17,220 classical runs reported in the paper)
├── phase1_causal_cone_theory.ipynb        Stage 1, causal-cone hypothesis
├── phase1b_degree_test.ipynb              Stage 2, coupling-degree hypothesis
├── phase2_task_performance.ipynb          Stage 3, task performance of the ZZ map
├── stage4_generalisation.ipynb            Stage 4, ten circuit families (parts A and B)
├── pqk_circuits.py                        Circuit module used by Stage 4 (frozen, SHA-256 cc1c57ff…)
├── freeze_phase1_inputs.py                Writes the Stage 1 freeze
├── freeze_phase1b.py                      Writes the Stage 2 freeze
├── freeze_phase2.py                       Writes the Stage 3 freeze
├── freeze_stage4.py                       Writes the Stage 4 freeze
│
├── figures/
│   ├── paper_figures.ipynb                Redraws Figs. 3 and 6 to 11 from the result files
│   ├── circuit_diagrams.ipynb             Redraws Figs. 4 and 5 from the simulated gate lists
│   └── tikz/                              LaTeX sources of Figs. 1 and 2
│
└── pqk-nids/                              Project root that the notebooks read and write
    ├── phase1_freeze/                     Frozen protocol files and their .sha256 hashes
    ├── phase1b_freeze/
    ├── phase2_freeze/
    ├── stage4_freeze/
    ├── results/                           Every result file of every stage
    │   ├── phase0/{quick,full}/
    │   ├── phase1/{smoke,run}/
    │   ├── phase1b/{smoke,run}/
    │   ├── phase2/{smoke,run}/
    │   └── stage4/{smoke,run}/
    └── data/                              Not included, see DATA.md
```

File names use the working names of the stages: `phase0` is the data audit, `phase1` is Stage 1, `phase1b` is Stage 2, `phase2` is Stage 3 and `stage4` is Stage 4. In every results folder, `smoke` holds the end-to-end test on synthetic data and `run` holds the frozen run reported in the paper.

## The freeze protocol

Each stage was designed after the previous one finished and was frozen before it ran. A freeze is a JSON file that records the hypotheses, decision thresholds, parameter grid, data subsets and the SHA-256 hashes of the input files. From Stage 2 onwards it also records a hash of the analysis notebook's code. Each freeze records the hashes of all earlier freezes, so the four files form a chain.

| Freeze | File | SHA-256 (first 16 characters) | Created (EDT) |
|---|---|---|---|
| Stage 1 | `pqk-nids/phase1_freeze/phase1_frozen_inputs.json` | `93ce9a06922e9ab0` | 6 Oct 2026, 17:55 |
| Stage 2 | `pqk-nids/phase1b_freeze/phase1b_frozen.json` | `647737d289a22896` | 6 Oct 2026, 19:12 |
| Stage 3 | `pqk-nids/phase2_freeze/phase2_frozen.json` | `2d0fa5b9a2124e57` | 6 Oct 2026, 22:28 |
| Stage 4 | `pqk-nids/stage4_freeze/stage4_frozen.json` | `ecc96083e90cc668` | 7 Oct 2026, 09:59 |

### Check the hashes yourself

From the repository root:

```bash
cd pqk-nids
for f in */*_frozen*.json; do echo "$f"; shasum -a 256 "$f"; cat "$f.sha256"; done
```

The two hashes printed for each file should be identical. Each notebook also checks its freeze, the input hashes and (from Stage 2 on) its own code hash before it touches real data, and it stops if any check fails.

## Reproducing the results

### 1. Install

Tested with Python 3.14 on macOS (Apple silicon). Python 3.12 or newer should work.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

Download the three datasets as described in [DATA.md](DATA.md) and place them under `pqk-nids/data/raw/`. The audit notebook checks every file against the SHA-256 hashes listed there.

### 3. Run the notebooks in order

Open Jupyter in the repository root (`jupyter lab`) and run the notebooks in this order. Each notebook has one `MODE` line at the top. Every other line is frozen.

| Order | Notebook | `MODE` |
|---|---|---|
| 1 | `phase0_data_and_baselines.ipynb` | `"quick"` |
| 2 | `phase0_full_data_and_baselines.ipynb` | `"full"` |
| 3 | `phase1_causal_cone_theory.ipynb` | `"run"` |
| 4 | `phase1b_degree_test.ipynb` | `"run"` |
| 5 | `phase2_task_performance.ipynb` | `"run"` |
| 6 | `stage4_generalisation.ipynb` | `"run"` |

Setting `MODE = "smoke"` instead runs a short end-to-end test on synthetic data. Set the environment variable `PQK_N_JOBS` to limit the number of CPU cores used. The full audit and Stages 3 and 4 take several hours each on a laptop.

The notebooks write to `pqk-nids/results/`. The results already in this repository are the ones reported in the paper, so a rerun overwrites them. Work on a copy if you want to compare.

The freeze files are included, so you do not need to run the `freeze_*.py` scripts. They are provided to show how each freeze was made, and they refuse to overwrite an existing freeze.

### 4. Redraw the figures

The figure notebooks need only the result files in this repository, not the raw data.

```bash
cd figures
jupyter nbconvert --to notebook --execute paper_figures.ipynb
jupyter nbconvert --to notebook --execute circuit_diagrams.ipynb
```

The figures are written to `figures/figs/` and `figures/circuit_figures/`. Figs. 1 and 2 are compiled from `figures/tikz/` with `pdflatex`.

## Where each number in the paper comes from

| Paper | Result file |
|---|---|
| Data audit (Sect. 5.1, Fig. 6, Appendices A and B) | `pqk-nids/results/phase0/full/` |
| Stage 1 (Sect. 5.2, Fig. 7, Table E5) | `pqk-nids/results/phase1/run/criterion_A.csv`, `criterion_B.csv`, `verdict.json` |
| Stage 2 (Sect. 5.3, Fig. 8, Table E6) | `pqk-nids/results/phase1b/run/C1_…` to `C4_…`, `verdict.json` |
| Stage 3 (Sect. 5.4) | `pqk-nids/results/phase2/run/P1_…`, `P2_…`, `P3_…`, `verdict.json` |
| Stage 4A (Sect. 5.5, Table 6, Fig. 9, Tables C3, D4, F7) | `pqk-nids/results/stage4/run/partA/`, `circuit_checks.json` |
| Stage 4B (Sect. 5.6, Tables 7 and 8, Figs. 10 and 11) | `pqk-nids/results/stage4/run/partB/` |
| Overall verdicts | `pqk-nids/results/stage4/run/verdict.json` |

## Use of AI assistance

A large language model (Claude, Anthropic) was used to help write and test the analysis code and to draft parts of the manuscript, as stated in Section 4.10 of the paper. The authors designed the study, ran every experiment and checked every reported number against the result files in this repository.

## Citation

If you use this code or these results, please cite the paper. The details are in [CITATION.cff](CITATION.cff), and GitHub shows a "Cite this repository" button for it.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The datasets are not redistributed here and remain under the terms of their original providers.

## Contact

Aakarsh Etar, Department of Electrical and Computer Engineering, Florida International University, aetar@fiu.edu

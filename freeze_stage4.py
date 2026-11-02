"""Freeze Stage 4 (generalisation across ten feature-map families) before it is run.

Run once, from the folder that contains pqk-nids/, pqk_circuits.py and stage4_generalisation.ipynb:
    python freeze_stage4.py
Writes pqk-nids/stage4_freeze/stage4_frozen.json (+ .sha256) and refuses to overwrite. It records the SHA-256 of
pqk_circuits.py and a hash of the notebook's code (MODE line excluded), so neither can change after freezing.
Commit both files and tag the commit 'stage4-freeze' immediately afterwards.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="pqk-nids")
ap.add_argument("--notebook", default="stage4_generalisation.ipynb")
ap.add_argument("--module", default="pqk_circuits.py")
ap.add_argument("--force", action="store_true", help="overwrite an existing freeze (never after Stage 4 results exist)")
a = ap.parse_args(); ROOT = Path(a.root); NB = Path(a.notebook); MOD = Path(a.module)
OUT = ROOT / "stage4_freeze"; OUT.mkdir(parents=True, exist_ok=True); F = OUT / "stage4_frozen.json"
if F.exists() and not a.force:
    sys.exit(f"{F} already exists. A freeze is written once; refusing to overwrite.")

def text_sha(p):
    return hashlib.sha256(Path(p).read_text().encode()).hexdigest()
def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
need = {"phase1": ROOT / "phase1_freeze" / "phase1_frozen_inputs.json", "phase1b": ROOT / "phase1b_freeze" / "phase1b_frozen.json",
        "phase2": ROOT / "phase2_freeze" / "phase2_frozen.json"}
for k, p in need.items():
    if not p.exists():
        sys.exit(f"{k} freeze not found at {p}")
    if text_sha(p) != (p.parent / (p.name + ".sha256")).read_text().split()[0]:
        sys.exit(f"{k} freeze file was modified; stop.")
if not (ROOT / "results" / "phase2" / "run" / "verdict.json").exists():
    sys.exit("Phase 2 run results not found: Stage 4 is designed after Phase 2.")
for p in [NB, MOD]:
    if not p.exists():
        sys.exit(f"{p} not found next to this script.")
smoke = ROOT / "results" / "stage4" / "smoke" / "circuit_checks.json"
if not smoke.exists():
    sys.exit("Run stage4_generalisation.ipynb with MODE = 'smoke' first: its circuit checks must pass before freezing.")
chk = json.loads(smoke.read_text())
if not all(v.get("pass_") for v in chk.values()):
    sys.exit("Smoke-mode circuit checks did not all pass; do not freeze.")

def code_hash(path):
    nb = json.loads(Path(path).read_text()); src = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            src.append("\n".join(l for l in s.splitlines() if not l.strip().startswith("MODE =")))
    return hashlib.sha256("\n#----\n".join(src).encode()).hexdigest()

sys.path.insert(0, str(MOD.resolve().parent))
import pqk_circuits as pc

FAM = pc.FAMILIES; ENT = pc.ENTANGLING
freeze = dict(
    created=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    builds_on=dict(phase1_freeze_sha256=text_sha(need["phase1"]), phase1b_freeze_sha256=text_sha(need["phase1b"]),
                   phase2_freeze_sha256=text_sha(need["phase2"]),
                   verdicts_so_far=dict(stage1="H1 (causal cone) REJECTED", stage2="H2 (coupling degree) MIXED: C2, C3 pass; C1, C4 fail",
                                        stage3="P1 FAIL, P2 FAIL; P3 PQK non-inferior to RBF-SVM in 9/9 task x N cells"),
                   design_note="Stage 4 was designed after Stages 1-3 to test whether their conclusions are specific to the ZZ "
                               "re-uploading map. Stage 1-3 verdicts are not revisited. All Stage 4 hypotheses, thresholds and "
                               "the analysis code are fixed by this file before any Stage 4 result on real data exists."),
    families={f: pc.DESCRIPTION[f] for f in FAM},
    entangler_classes=dict(commuting=pc.COMMUTING_ENTANGLER, non_commuting=pc.NON_COMMUTING_ENTANGLER, control=["separable"]),
    circuit_conventions="L layers; each layer = single-qubit encoding gates then two-qubit gates on the sorted edge list; "
                        "R_a(t) = exp(-i t A/2); qubit q receives feature q (importance order); exact statevector simulation; "
                        "Bloch vectors (X, Y, Z expectations) of every qubit are the projected-kernel features",
    cone_definition="light cone measured from the circuit: D(k) = {j : shifting x_j by 1.0 changes qubit k's Bloch vector by "
                    "more than 1e-13 (max over 24 Gaussian inputs, seed 7, c = 1.0)}; computed per family, graph, n and L. Checks: "
                    "inside the backward gate-walk bound; equal to the graph-distance-L cone for zz_reupload; no sensitivity value "
                    "in the ambiguous band [1e-14, 1e-12).",
    concentration_measure="V_k = trace of the covariance (across inputs) of qubit k's Bloch vector",
    partA=dict(
        qubits=[8, 12], layers=[1, 2, 3], bandwidths=[0.3, 0.6, 1.0], samples_per_config=200,
        graphs=["empty", "matching", "chain", "ring", "cycles4", "triangles", "two_cycles", "k4s", "regular3", "star", "wheel",
                "double_star", "erdos_renyi_a", "random_tree_a", "complete"],
        graphs_note="the Stage 2 graph generator with seed 20261001; graphs that do not exist at a given n, or duplicate "
                    "another graph's edge set, are skipped",
        inputs=["gaussian (seed 20261001+4)", "fashion_mnist pca16 (first n)", "nf_unsw_nb15_v2 mi16 (first n)", "unsw_nb15 mi16 (first n)"],
        fresh_rows="200 training rows per dataset drawn with seed 20261001+4, excluding every row used in Stages 1 and 2"),
    partB=dict(
        tasks={"fashion_mnist:binary": "pca8", "nf_unsw_nb15_v2:binary": "mi8", "unsw_nb15:binary": "mi8"},
        qubits=8, training_sizes=[250, 1000, 4000], seeds=[0, 1, 2],
        graphs=["empty", "chain", "ring", "star", "regular3", "complete"], layers=[1, 2], bandwidths=[0.3, 0.6, 1.0],
        separable_family="graph ignored: run once per (L, c) as graph 'empty'",
        validation_subset="as Stage 3: stratified 2000 rows of the validation split, seed 0",
        test_subset="as Stage 3: stratified min(4000, all) rows of the clean test split, seed 20261001; the same rows for every model",
        quantum_kernel="as Stage 3: projected 1-RDM kernel, gamma = multiplier / median distance (500 training points), multipliers "
                       "{0.5, 1, 2}, SVM C in {1, 10, 100}, class_weight balanced, tuned on validation with the seed-0 sample, refit per seed",
        comparators="as Stage 3: RBF-SVM and the Phase 0 best classical model on identical training samples and test rows",
        selection="'validation-selected' = highest validation macro-F1 among the stated configurations; ties broken by the "
                  "order graph list -> L -> c",
        primary_metric="macro-F1 on the test subset"),
    hypotheses=dict(
        A1="Degree mechanism generalises. For each entangling family, in every cell (inputs, c, n, heterogeneous graph, L in {2,3}) "
           "compute the Spearman correlation between coupling degree d_k and V_k across qubits; the family passes if >= 80% of its "
           "cells are negative (undefined correlations count as not negative). GENERALISES if >= 7 of the 9 entangling families pass; "
           "FAMILY-SPECIFIC if 3-6 pass; DOES NOT GENERALISE if <= 2 pass.",
        A2="Cone irrelevance at equal degree generalises. For each entangling family, compare graphs with equal uniform degree (same "
           "n, L, c, inputs) whose mean light cone differs by >= 0.5: median |log(V_larger_cone / V_smaller_cone)| (V averaged "
           "over qubits). The family passes if it has >= 10 such pairs, the median is <= 0.05, and the median is <= one third of the "
           "family's median one-step degree effect (matching -> ring, ring -> regular3, same n, L, c, inputs). Same 7 / 3-6 / <= 2 rule.",
        A3="Descriptive (no pass/fail): per family, R2 of log V_k on [d_k, L, d_k*L] versus [|D(k)|, L, |D(k)|*L], per (inputs, c).",
        B1="Concentration does not predict accuracy (replicates Stage 3 P1 failure). For each family and task at N = 1000, Spearman "
           "correlation across that family's configurations (graph, L, c) between mean V (seed-0 training sample) and seed-averaged "
           "test macro-F1. Concentration is predictive for a family if the correlation is positive with p < 0.05 in >= 2 of 3 tasks. "
           "B1 REPLICATES if concentration is not predictive in >= 8 of the 10 families.",
        B2="Entanglement adds no meaningful accuracy (replicates Stage 3 entangled-vs-uncoupled). For each entangling family, task "
           "and N: delta = seed-averaged test macro-F1 of the family's validation-selected coupled configuration (graph != empty) "
           "minus that of its validation-selected uncoupled configuration (graph = empty; same gates without two-qubit gates). "
           "B2 REPLICATES if delta <= +0.02 in >= 80% of the 81 cells.",
        B3="Non-inferiority to RBF-SVM (replicates Stage 3 P3). For each family, task and N: the family's validation-selected "
           "configuration versus RBF-SVM, paired bootstrap 95% CI (1000 resamples of test rows, seed-0 predictions) of the macro-F1 "
           "difference; non-inferior if the lower bound is > -0.01. B3 REPLICATES if non-inferior in >= 80% of the 90 cells.",
        B4="Bandwidth matters more than graph (replicates Stage 3). For each entangling family and task at N = 1000: bandwidth "
           "effect = mean over (graph, L) of the range of seed-averaged macro-F1 across c; graph effect = mean over (L, c) of the "
           "range across graphs. B4 REPLICATES if the bandwidth effect exceeds the graph effect in >= 18 of the 27 cells."),
    decision=dict(stage3_generalises="B1, B2 and B3 all replicate -> 'Stage 3 conclusions GENERALISE across the ten families'; "
                                     "one or two replicate -> 'PARTLY GENERALISE'; none -> 'DO NOT GENERALISE'. B4 reported alongside.",
                  concentration_mechanism="A1 and A2 reported with their three-level outcome",
                  verification="if any circuit check fails, nothing is interpreted",
                  reproduction="zz_reupload configurations that also appear in Stage 3 are compared with the Stage 3 results "
                               "(reported, not a hypothesis)"),
    module=str(MOD), module_sha256=file_sha(MOD),
    notebook=str(NB), notebook_code_sha256=code_hash(NB),
)
text = json.dumps(freeze, indent=2); F.write_text(text); d = hashlib.sha256(text.encode()).hexdigest()
(OUT / "stage4_frozen.json.sha256").write_text(d + "  stage4_frozen.json\n")
print(f"Frozen: {F}\nSHA-256: {d}\nModule hash: {freeze['module_sha256']}\nNotebook code hash: {freeze['notebook_code_sha256']}")
print("Next:\n  git add freeze_stage4.py pqk_circuits.py stage4_generalisation.ipynb pqk-nids/stage4_freeze\n"
      "  git commit -m 'Freeze Stage 4 (generalisation across circuit families)'\n  git tag stage4-freeze")

"""Freeze Phase 2 (task performance of projected quantum kernels) before it is run.

Run once, from the folder that contains pqk-nids/ and phase2_task_performance.ipynb:
    python freeze_phase2.py
Writes pqk-nids/phase2_freeze/phase2_frozen.json (+ .sha256) and refuses to overwrite. Commit and tag 'phase2-freeze'.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="pqk-nids"); ap.add_argument("--notebook", default="phase2_task_performance.ipynb")
ap.add_argument("--force", action="store_true")
a = ap.parse_args(); ROOT = Path(a.root); NB = Path(a.notebook)
OUT = ROOT / "phase2_freeze"; OUT.mkdir(parents=True, exist_ok=True); F = OUT / "phase2_frozen.json"
if F.exists() and not a.force:
    sys.exit(f"{F} already exists. A freeze is written once; refusing to overwrite.")

def text_sha(p):
    return hashlib.sha256(Path(p).read_text().encode()).hexdigest()
need = {"phase1": ROOT / "phase1_freeze" / "phase1_frozen_inputs.json", "phase1b": ROOT / "phase1b_freeze" / "phase1b_frozen.json"}
for k, p in need.items():
    if not p.exists():
        sys.exit(f"{k} freeze not found at {p}")
    if text_sha(p) != (p.parent / (p.name + ".sha256")).read_text().split()[0]:
        sys.exit(f"{k} freeze file was modified; stop.")
if not (ROOT / "results" / "phase1b" / "run" / "verdict.json").exists():
    sys.exit("Phase 1b run results not found: Phase 2 is designed after Phase 1b.")
if not NB.exists():
    sys.exit(f"{NB} not found next to this script.")

def code_hash(path):
    nb = json.loads(Path(path).read_text()); src = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            src.append("\n".join(l for l in s.splitlines() if not l.strip().startswith("MODE =")))
    return hashlib.sha256("\n#----\n".join(src).encode()).hexdigest()

freeze = dict(
    created=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    builds_on=dict(phase1_freeze_sha256=text_sha(need["phase1"]), phase1b_freeze_sha256=text_sha(need["phase1b"]),
                   phase1_verdict="H1 (causal cone) REJECTED", phase1b_verdict="H2 (coupling degree) MIXED: C2, C3 pass; C1, C4 fail"),
    design=dict(
        tasks={"fashion_mnist:binary": "pca8", "nf_unsw_nb15_v2:binary": "mi8", "unsw_nb15:binary": "mi8"},
        qubits=8, training_sizes=[250, 1000, 4000], seeds=[0, 1, 2],
        validation_subset="stratified, 2000 rows of the validation split, seed 0",
        test_subset="stratified, min(4000, all) rows of the clean test split, seed 20261001; the same rows for every model",
        graphs=["empty", "matching", "chain", "ring", "cycles4", "star", "wheel", "k4s", "regular3", "erdos_renyi_a", "complete"],
        layers=[1, 2], bandwidths=[0.3, 0.6, 1.0],
        assignments=dict(rank="qubit q receives feature q (features in importance order: PCA order or MI rank)",
                         top_to_low_degree="most important features on lowest-degree qubits (heterogeneous graphs only)",
                         top_to_high_degree="most important features on highest-degree qubits (heterogeneous graphs only)"),
        heterogeneous_graphs=["star", "wheel", "erdos_renyi_a"],
        quantum_kernel="projected 1-RDM kernel k(x,x') = exp(-gamma * sum_k ||rho_k(x)-rho_k(x')||_F^2); gamma = multiplier * 1/median distance on 500 training points; multipliers {0.5,1,2}; SVM C in {1,10,100}; class_weight balanced",
        tuning="hyperparameters chosen on the validation subset with the seed-0 training sample (macro-F1), then refit for every seed",
        comparators="on the identical training samples and test subset: RBF-SVM (Phase 0 grid) on the same 8 features, and the Phase 0 best classical model for that task, view and N (Phase 0 grid)",
        primary_metric="macro-F1 on the test subset"),
    hypotheses=dict(
        P1="Per-qubit concentration predicts usefulness: across feature-map configurations (assignment = rank) at N = 1000, the "
           "Spearman correlation between mean V (on the seed-0 training sample) and seed-averaged test macro-F1 is positive with p < 0.05 "
           "in at least 2 of the 3 tasks. Reported for N = 250 and 4000 as secondary.",
        P2="Feature-to-qubit assignment matters as predicted by Phase 1b's exploratory refinement: in paired comparisons (same task, "
           "heterogeneous graph, L, c, N; seed-averaged), top_to_low_degree beats top_to_high_degree in >= 70% of pairs with a positive median difference.",
        P3_comparison="Pre-specified, descriptive: for each task and N, the projected-kernel configuration with the best validation macro-F1 "
                      "is compared on test with RBF-SVM and the best classical model (difference and bootstrap 95% CI over test rows, seed 0). "
                      "Non-inferiority to RBF-SVM is declared if the lower CI bound of (PQK - RBF) is above -0.01."),
    decision=dict(P1="pass/fail as stated", P2="pass/fail as stated", P3="reported per task and N; no pass/fail on the paper's claims"),
    notebook=str(NB), notebook_code_sha256=code_hash(NB),
)
text = json.dumps(freeze, indent=2); F.write_text(text); d = hashlib.sha256(text.encode()).hexdigest()
(OUT / "phase2_frozen.json.sha256").write_text(d + "  phase2_frozen.json\n")
print(f"Frozen: {F}\nSHA-256: {d}\nNotebook code hash: {freeze['notebook_code_sha256']}")
print("Next:\n  git add freeze_phase2.py phase2_task_performance.ipynb pqk-nids/phase2_freeze\n"
      "  git commit -m 'Freeze Phase 2 (task performance)'\n  git tag phase2-freeze")

"""Freeze the Phase 1 inputs before any quantum result exists.

Run once, from the folder that contains pqk-nids/:
    python freeze_phase1_inputs.py

It writes pqk-nids/phase1_freeze/phase1_frozen_inputs.json plus a .sha256 file, and refuses to overwrite an
existing freeze. Commit both files to git (or upload them to OSF) immediately afterwards so the timestamp is public.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="pqk-nids", help="project root that contains data/ and results/")
ap.add_argument("--phase0-mode", default="quick", help="Phase 0 run whose splits are frozen (quick and full produce identical splits)")
ap.add_argument("--force", action="store_true", help="overwrite an existing freeze (do NOT use after Phase 1 results exist)")
a = ap.parse_args()

ROOT = Path(a.root); M = a.phase0_mode
RES = ROOT / "results" / "phase0" / M; PROC = ROOT / "data" / "processed"
OUT = ROOT / "phase1_freeze"; OUT.mkdir(parents=True, exist_ok=True)
FREEZE = OUT / "phase1_frozen_inputs.json"
if FREEZE.exists() and not a.force:
    sys.exit(f"{FREEZE} already exists. A freeze is written once; refusing to overwrite.")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# ---------------------------------------------------------------- frozen design
TASKS = {
    "fashion_mnist:binary":   {"dataset": "fashion_mnist",   "views": ["pca8", "pca16"]},
    "nf_unsw_nb15_v2:binary": {"dataset": "nf_unsw_nb15_v2", "views": ["mi8", "mi16"]},
    "unsw_nb15:binary":       {"dataset": "unsw_nb15",       "views": ["mi8", "mi16"]},
}
N_LIST = [250, 500, 1000, 2000, 4000]

# ---------------------------------------------------------------- check inputs exist
need = [RES / "baselines.csv", RES / "splits.csv", RES / "feature_rankings.json", RES / "phase0_gate.json"]
for t in TASKS.values():
    d = PROC / t["dataset"]
    need += [d / f"train_{M}.parquet", d / f"val_{M}.parquet", d / f"test_{M}.parquet", d / f"preprocessor_{M}.joblib"]
    if t["dataset"] == "fashion_mnist":
        need.append(d / f"pca_{M}.joblib")
missing = [str(p) for p in need if not p.exists()]
if missing:
    sys.exit("Missing Phase 0 outputs:\n  " + "\n  ".join(missing))

gate = json.load(open(RES / "phase0_gate.json"))
if not gate.get("gate_pass"):
    sys.exit("Phase 0 gate did not pass; fix Phase 0 before freezing Phase 1.")

# ---------------------------------------------------------------- reference baselines
B = pd.read_csv(RES / "baselines.csv")
B = B[B["error"].fillna("").astype(str).str.len() == 0]
B = B[(B.regime == "matched")]
ref, missing_views = {}, []
for task, t in TASKS.items():
    ref[task] = {}
    for v in t["views"]:
        x = B[(B.task == task) & (B.features == v)]
        if x.empty:
            missing_views.append(f"{task}/{v}"); continue
        g = x.groupby(["n_target", "model"]).macro_f1.agg(["mean", "std", "count"]).reset_index()
        ref[task][v] = {}
        for n in N_LIST:
            gn = g[g.n_target == n]
            if gn.empty:
                missing_views.append(f"{task}/{v}/N={n}"); continue
            svm = gn[gn.model == "svm"].iloc[0]
            best = gn.sort_values("mean", ascending=False).iloc[0]
            ref[task][v][str(n)] = dict(rbf_svm_macro_f1=round(float(svm["mean"]), 4), rbf_svm_sd=round(float(svm["std"]), 4),
                                        best_model=str(best["model"]), best_macro_f1=round(float(best["mean"]), 4),
                                        best_sd=round(float(best["std"]), 4), seeds=int(best["count"]))
if missing_views:
    sys.exit("Phase 0 baselines are missing for: " + ", ".join(missing_views))

# ---------------------------------------------------------------- feature definitions
ranks = json.load(open(RES / "feature_rankings.json"))
features = {}
for task, t in TASKS.items():
    ds = t["dataset"]
    for v in t["views"]:
        k = int(v.lstrip("abcdefghijklmnopqrstuvwxyz")); meth = v[: -len(str(k))]
        if meth == "pca":
            features[f"{ds}/{v}"] = f"first {k} principal components (PCA fitted on the training split, each standardised on the training split)"
        else:
            features[f"{ds}/{v}"] = ranks[ds][meth][:k]

# ---------------------------------------------------------------- hashes of everything Phase 1 reads
hashes = {str(p.relative_to(ROOT)): sha(p) for p in need}
splits = pd.read_csv(RES / "splits.csv")
split_hashes = {f"{r.dataset}/{r.split}": r.row_id_sha256 for r in splits.itertuples()}

freeze = dict(
    created=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    phase0_mode_of_frozen_splits=M,
    tasks=TASKS, training_sizes=N_LIST,
    comparators=dict(like_for_like="RBF-SVM (classical kernel, same features, same N)",
                     ceiling="best classical model of LR, RBF-SVM, RFF, RF, XGBoost, LightGBM, MLP (same features, same N)"),
    primary_metric="macro-F1 on the clean test split", secondary_metrics=["MCC", "ROC-AUC", "minority-class PR-AUC"],
    reference_baselines_phase0=ref,
    feature_definitions=features,
    hypotheses=dict(
        H1="At fixed bandwidth, concentration of projected-kernel features is governed by causal-cone size, not interaction density.",
        H0="Interaction density explains concentration at least as well as causal-cone size.",
        concentration_measure="V = mean over qubits of the trace of the covariance (across inputs) of single-qubit Bloch vectors",
        cone_definition="C_L(k) = qubits within graph distance L of qubit k, for an L-layer feature map; reported as the mean |C_L(k)| over qubits",
        criterion_A="In each (dataset, bandwidth) cell: R^2 of log V on mean cone size >= 0.70, and adding density to that model is not significant (partial F-test, p > 0.05). H1 supported on A if at least 75% of cells pass.",
        criterion_B="Across all pairs of graphs with equal n and equal edge count (equal density) but different mean cone size, at equal L and bandwidth: the graph with the larger cone has the lower V in at least 80% of pairs.",
        decision="H1 is supported if both A and B pass; rejected if both fail; mixed otherwise (reported as such).",
        small_n_test=dict(qubits=[6, 8, 9, 10], layers=[1, 2, 3], bandwidths=[0.3, 0.6, 1.0], samples_per_config=200,
                          inputs=["Gaussian control", "fashion_mnist pca16 (first n)", "nf_unsw_nb15_v2 mi16 (first n)", "unsw_nb15 mi16 (first n)"])),
    phase0_split_row_id_sha256=split_hashes,
    file_sha256=hashes,
)
text = json.dumps(freeze, indent=2, sort_keys=False)
FREEZE.write_text(text)
digest = hashlib.sha256(text.encode()).hexdigest()
(OUT / "phase1_frozen_inputs.json.sha256").write_text(digest + "  phase1_frozen_inputs.json\n")
print(f"Frozen: {FREEZE}")
print(f"SHA-256 of the freeze file: {digest}")
print("Reference baselines (macro-F1, RBF-SVM / best classical):")
for task, vv in ref.items():
    for v, nn in vv.items():
        print(f"  {task:24s} {v:6s} " + "  ".join(f"N={n}: {d['rbf_svm_macro_f1']:.3f}/{d['best_macro_f1']:.3f}" for n, d in nn.items()))
print("\nNext: commit both files now, e.g.\n  git add pqk-nids/phase1_freeze && git commit -m 'Freeze Phase 1 inputs' && git tag phase1-freeze")

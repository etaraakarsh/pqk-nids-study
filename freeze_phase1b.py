"""Freeze Phase 1b (coupling-degree hypothesis) before it is run.

Run once, from the folder that contains pqk-nids/ and phase1b_degree_test.ipynb:
    python freeze_phase1b.py

Writes pqk-nids/phase1b_freeze/phase1b_frozen.json and a .sha256 file, and refuses to overwrite. The freeze also
records a hash of the notebook's code (the MODE line excluded), so the analysis cannot change after freezing.
Commit both files and tag the commit 'phase1b-freeze' immediately afterwards.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="pqk-nids")
ap.add_argument("--notebook", default="phase1b_degree_test.ipynb")
ap.add_argument("--force", action="store_true", help="overwrite an existing freeze (never after Phase 1b results exist)")
a = ap.parse_args()
ROOT = Path(a.root); NB = Path(a.notebook)
OUT = ROOT / "phase1b_freeze"; OUT.mkdir(parents=True, exist_ok=True); F = OUT / "phase1b_frozen.json"
if F.exists() and not a.force:
    sys.exit(f"{F} already exists. A freeze is written once; refusing to overwrite.")

P1 = ROOT / "phase1_freeze" / "phase1_frozen_inputs.json"
if not P1.exists():
    sys.exit("Phase 1 freeze not found: Phase 1b reuses its frozen inputs.")
p1_text = P1.read_text(); p1_sha = hashlib.sha256(p1_text.encode()).hexdigest()
rec = (P1.parent / "phase1_frozen_inputs.json.sha256").read_text().split()[0]
if p1_sha != rec:
    sys.exit("Phase 1 freeze file has been modified; stop.")
if not (ROOT / "results" / "phase1" / "run" / "verdict.json").exists():
    sys.exit("Phase 1 run results not found: Phase 1b is designed from them and must come after.")
if not NB.exists():
    sys.exit(f"Notebook {NB} not found next to this script.")

def code_hash(path):
    nb = json.loads(Path(path).read_text())
    src = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            src.append("\n".join(l for l in s.splitlines() if not l.strip().startswith("MODE =")))
    return hashlib.sha256("\n#----\n".join(src).encode()).hexdigest()

freeze = dict(
    created=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    builds_on=dict(phase1_freeze_sha256=p1_sha, phase1_verdict="H1 REJECTED (causal-cone hypothesis)",
                   design_note="H2 and the margins below were chosen after inspecting Phase 1 results (exploratory). "
                               "Phase 1b tests them on new graphs, larger n and data rows not used in Phase 1."),
    hypothesis=dict(
        H2="For the ZZ re-uploading feature map, the Bloch-vector variance V_k of qubit k is governed by its own coupling "
           "degree d_k together with the layer count L, not by its causal-cone size |C_L(k)| and not by global graph density.",
        H0="Causal-cone size or global density explains V_k at least as well as coupling degree."),
    grid=dict(qubits=[8, 10, 12, 14], layers=[1, 2, 3], bandwidths=[0.3, 0.6, 1.0], samples_per_config=200,
              inputs=["gaussian (fresh seed)", "fashion_mnist pca16 (first n)", "nf_unsw_nb15_v2 mi16 (first n)", "unsw_nb15 mi16 (first n)"],
              fresh_rows="200 training rows per dataset drawn with seed 20261001+1, excluding every row used in Phase 1",
              graph_families=["empty", "matching", "chain", "ring", "cycles4", "triangles", "two_cycles", "k4s", "regular3",
                              "star", "wheel", "double_star", "erdos_renyi_p0.3 (seeds a,b)", "random_tree (seeds a,b)", "complete"],
              graph_seed=20261001),
    criteria=dict(
        C1="Per (inputs, bandwidth) cell, per-qubit regression of log V_k: degree model [d_k, L, d_k*L] vs cone model "
           "[|C_L(k)|, L, |C_L(k)|*L]. Cell passes if R2_degree >= 0.70, R2_degree - R2_cone >= 0.10, and adding graph density "
           "to the degree model raises R2 by <= 0.05. C1 passes if >= 75% of the 12 cells pass.",
        C2="Within each graph whose qubits have unequal degrees, for L in {2,3}, every bandwidth and input set: Spearman "
           "correlation between d_k and V_k is negative. C2 passes if >= 80% of these cells are negative.",
        C3="Equal-degree, different-cone comparisons (regular graphs with the same n and degree, mean cone differing by >= 0.5, "
           "same L, bandwidth, inputs): median |log(V_large_cone / V_small_cone)| <= 0.05 AND <= one third of the median "
           "|log V ratio| for a one-step degree change (matching->ring, ring->regular3) in the same run.",
        C4="Out of sample: per cell, fit the degree model on n in {8,10} and predict log V_k for n in {12,14} without refitting. "
           "C4 passes if R2 >= 0.60 in >= 75% of the 12 cells.",
        C5="Verification (must pass): for L = 1, the closed form r_k = (sin t_k Re P_k, sin t_k Im P_k, cos t_k), "
           "P_k = prod_{j in N(k)} (cos p_kj + i sin p_kj cos t_j), t_k = c x_k, p_kj = c^2 x_k x_j, matches the simulator to 1e-10.",
        calibration="Margins from Phase 1 (exploratory): same-degree different-cone median |log V ratio| = 0.017; "
                    "one-step degree change = 0.085-0.127."),
    decision=dict(supported="C1 and C2 pass (strongly supported if C3 and C4 also pass)",
                  rejected="C1 and C2 both fail", mixed="any other combination; reported as such",
                  verification="if C5 fails, results are invalid and nothing is interpreted"),
    notebook=str(NB), notebook_code_sha256=code_hash(NB),
)
text = json.dumps(freeze, indent=2)
F.write_text(text); d = hashlib.sha256(text.encode()).hexdigest()
(OUT / "phase1b_frozen.json.sha256").write_text(d + "  phase1b_frozen.json\n")
print(f"Frozen: {F}\nSHA-256: {d}\nNotebook code hash: {freeze['notebook_code_sha256']}")
print("Next:\n  git add freeze_phase1b.py phase1b_degree_test.ipynb pqk-nids/phase1b_freeze\n"
      "  git commit -m 'Freeze Phase 1b (coupling-degree hypothesis)'\n  git tag phase1b-freeze")

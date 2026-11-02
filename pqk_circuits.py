"""Stage 4 circuit module: ten feature-map families for projected quantum kernels.

Every family maps an input x (n features, one per qubit) to U(x)|0...0> through L repeated layers of
single-qubit encoding gates followed by two-qubit gates on the edges of an interaction graph. Edges are applied
in sorted order (i < j); for non-commuting entanglers this order is part of the circuit definition.

Conventions
    qubit 0 is the most significant bit of the basis index; R_a(t) = exp(-i t A / 2);
    two-qubit gates act on (first, second) = (i, j) of the edge; c is the bandwidth (input scaling).

Families (per layer)
    zz_reupload   R_y(c x_k); ZZ(c^2 x_i x_j)                         identical to the Phase 1-3 map
    havlicek_zz   H; R_z(2 c x_k); ZZ(2 (pi - c x_i)(pi - c x_j))       Havlicek et al. (2019) style
    iqp           H; R_z(c x_k); ZZ(c^2 x_i x_j)                        IQP-type, product-angle couplings
    xy            R_y(c x_k); XY(c^2 x_i x_j) = exp(-i t (XX+YY)/2)     non-commuting, excitation preserving
    heisenberg    R_y(c x_k); exp(-i t (XX+YY+ZZ)/2), t = c^2 x_i x_j   non-commuting
    ms_xx         R_y(c x_k); XX(c^2 x_i x_j)  (Molmer-Sorensen type)   commuting, non-diagonal
    hea_cnot      R_y(c x_k) R_z(c x_k); CNOT(i -> j)                   hardware-efficient, fixed entangler, non-commuting
    cz_layers     R_y(c x_k) R_z(c x_k); CZ(i, j)                       hardware-efficient, fixed entangler, commuting
    crx           R_y(c x_k); CRX(c x_i) control i, target j            data-controlled rotation, non-commuting
    separable     R_y(c x_k) R_z(c x_k); no two-qubit gates             product-state control (graph ignored)

The module also provides an independent dense-matrix implementation (for verification only) and each qubit's light
cone measured from the circuit itself: the set of inputs x_j whose change alters qubit k's reduced state
(dependence_cones). It is checked against an upper bound obtained by walking the gate list backwards (gate_walk_cones).
"""
import numpy as np

FAMILIES = ["zz_reupload", "havlicek_zz", "iqp", "xy", "heisenberg", "ms_xx", "hea_cnot", "cz_layers", "crx", "separable"]
COMMUTING_ENTANGLER = ["zz_reupload", "havlicek_zz", "iqp", "cz_layers", "ms_xx"]    # two-qubit gates mutually commute
NON_COMMUTING_ENTANGLER = ["xy", "heisenberg", "hea_cnot", "crx"]
ENTANGLING = COMMUTING_ENTANGLER + NON_COMMUTING_ENTANGLER
DESCRIPTION = {
    "zz_reupload": "R_y(c x_k); ZZ(c^2 x_i x_j)",
    "havlicek_zz": "H; R_z(2 c x_k); ZZ(2 (pi - c x_i)(pi - c x_j))",
    "iqp": "H; R_z(c x_k); ZZ(c^2 x_i x_j)",
    "xy": "R_y(c x_k); exp(-i t (XX+YY)/2), t = c^2 x_i x_j",
    "heisenberg": "R_y(c x_k); exp(-i t (XX+YY+ZZ)/2), t = c^2 x_i x_j",
    "ms_xx": "R_y(c x_k); exp(-i t XX/2), t = c^2 x_i x_j",
    "hea_cnot": "R_y(c x_k) R_z(c x_k); CNOT(i -> j)",
    "cz_layers": "R_y(c x_k) R_z(c x_k); CZ(i, j)",
    "crx": "R_y(c x_k); CRX(c x_i), control i, target j",
    "separable": "R_y(c x_k) R_z(c x_k); no two-qubit gates",
}

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=complex)
_I4 = np.eye(4, dtype=complex)


# ------------------------------------------------------------------ gate matrices (batched over inputs)
def _single(name, t):
    """(N, 2, 2) matrices for a batch of angles t (shape (N,)), or (2, 2) for fixed gates."""
    if name == "h":
        return _H
    t = np.asarray(t, dtype=float); c, s = np.cos(t / 2), np.sin(t / 2); z = np.zeros_like(t)
    if name == "ry":
        return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)
    if name == "rz":
        return np.stack([np.stack([np.exp(-0.5j * t), z], -1), np.stack([z, np.exp(0.5j * t)], -1)], -2)
    if name == "rx":
        return np.stack([np.stack([c + 0j, -1j * s], -1), np.stack([-1j * s, c + 0j], -1)], -2)
    raise ValueError(name)


def _two(name, t):
    """(N, 4, 4) matrices (basis |q1 q2>, q1 = first qubit of the gate) or (4, 4) for fixed gates."""
    if name == "cnot":
        return _CNOT
    if name == "cz":
        return _CZ
    t = np.asarray(t, dtype=float)[:, None, None]
    if name == "zz":
        d = np.exp(-0.5j * t[:, 0, 0])[:, None] ** np.array([1, -1, -1, 1])[None, :]
        return d[:, :, None] * _I4[None]
    if name == "xx":
        return np.cos(t / 2) * _I4 - 1j * np.sin(t / 2) * _XX
    if name == "xy":     # exp(-i t (XX+YY)/2): identity on |00>,|11>; cos t, -i sin t on |01>,|10>
        U = np.zeros((t.shape[0], 4, 4), dtype=complex); c, s = np.cos(t[:, 0, 0]), np.sin(t[:, 0, 0])
        U[:, 0, 0] = U[:, 3, 3] = 1; U[:, 1, 1] = U[:, 2, 2] = c; U[:, 1, 2] = U[:, 2, 1] = -1j * s
        return U
    if name == "heis":   # XX+YY+ZZ = 2 SWAP - I: eigenvalue +1 on the triplet, -3 on the singlet
        Pt, Ps = (_I4 + _SWAP) / 2, (_I4 - _SWAP) / 2
        return np.exp(-0.5j * t) * Pt + np.exp(1.5j * t) * Ps
    if name == "crx":    # control = first qubit
        R = _single("rx", t[:, 0, 0]); U = np.zeros((t.shape[0], 4, 4), dtype=complex)
        U[:, 0, 0] = U[:, 1, 1] = 1; U[:, 2:, 2:] = R
        return U
    raise ValueError(name)


# ------------------------------------------------------------------ circuits
def build_circuit(family, X, edges, L, c):
    """Gate list [(kind, name, qubits, angles)] for a batch X (N, n). kind is 1 or 2 (number of qubits)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family}")
    X = np.asarray(X, dtype=np.float64); n = X.shape[1]
    E = [] if family == "separable" else sorted(tuple(sorted(e)) for e in edges)
    g = []
    for _ in range(L):
        if family in ("havlicek_zz", "iqp"):
            g += [(1, "h", (k,), None) for k in range(n)]
        if family == "havlicek_zz":
            g += [(1, "rz", (k,), 2 * c * X[:, k]) for k in range(n)]
            g += [(2, "zz", (i, j), 2 * (np.pi - c * X[:, i]) * (np.pi - c * X[:, j])) for i, j in E]
        elif family == "iqp":
            g += [(1, "rz", (k,), c * X[:, k]) for k in range(n)]
            g += [(2, "zz", (i, j), c ** 2 * X[:, i] * X[:, j]) for i, j in E]
        elif family in ("hea_cnot", "cz_layers", "separable"):
            for k in range(n):
                g += [(1, "ry", (k,), c * X[:, k]), (1, "rz", (k,), c * X[:, k])]
            if family == "hea_cnot":
                g += [(2, "cnot", (i, j), None) for i, j in E]
            elif family == "cz_layers":
                g += [(2, "cz", (i, j), None) for i, j in E]
        else:
            g += [(1, "ry", (k,), c * X[:, k]) for k in range(n)]
            name = {"zz_reupload": "zz", "xy": "xy", "heisenberg": "heis", "ms_xx": "xx", "crx": "crx"}[family]
            if family == "crx":
                g += [(2, "crx", (i, j), c * X[:, i]) for i, j in E]
            else:
                g += [(2, name, (i, j), c ** 2 * X[:, i] * X[:, j]) for i, j in E]
    return g


def _apply1(psi, n, k, U):
    """Apply a 2x2 gate (fixed (2, 2) or batched (N, 2, 2)) to qubit k, term by term (exact, no approximation)."""
    N = psi.shape[0]; p = psi.reshape(N, 2 ** k, 2, 2 ** (n - k - 1)); out = np.zeros_like(p)
    nz = (U != 0) if U.ndim == 2 else (U != 0).any(0)
    for w in range(2):
        for j in range(2):
            if nz[w, j]:
                cf = U[w, j] if U.ndim == 2 else U[:, w, j][:, None, None]
                out[:, :, w, :] += cf * p[:, :, j, :]
    return out.reshape(N, -1)


def _apply2(psi, n, q1, q2, U):
    """Apply a 4x4 gate U (basis |q1 q2>, fixed or batched) to qubits q1 != q2 in any order, term by term."""
    N = psi.shape[0]; a, b = min(q1, q2), max(q1, q2)
    U5 = U.reshape(U.shape[:-2] + (2, 2, 2, 2))            # (..., out1, out2, in1, in2)
    if q1 > q2:                                             # reorder to (out_a, out_b, in_a, in_b)
        U5 = U5.swapaxes(-4, -3).swapaxes(-2, -1)
    p = psi.reshape(N, 2 ** a, 2, 2 ** (b - a - 1), 2, 2 ** (n - b - 1)); out = np.zeros_like(p)
    batched = U5.ndim == 5; nz = (U5 != 0).any(0) if batched else (U5 != 0)
    for w in range(2):
        for x in range(2):
            for j in range(2):
                for l in range(2):
                    if nz[w, x, j, l]:
                        cf = U5[:, w, x, j, l][:, None, None, None] if batched else U5[w, x, j, l]
                        out[:, :, w, :, x, :] += cf * p[:, :, j, :, l, :]
    return out.reshape(N, -1)


def simulate(gates, N, n):
    psi = np.zeros((N, 2 ** n), dtype=complex); psi[:, 0] = 1.0
    for kind, name, q, t in gates:
        if kind == 1:
            psi = _apply1(psi, n, q[0], _single(name, t))
        else:
            psi = _apply2(psi, n, q[0], q[1], _two(name, t))
    return psi


def bloch(psi, n):
    """(N, n, 3) single-qubit Bloch vectors (X, Y, Z expectations) of each qubit, computed exactly."""
    N = psi.shape[0]; R = np.empty((N, n, 3))
    for k in range(n):
        p = psi.reshape(N, 2 ** k, 2, 2 ** (n - k - 1)); a0, a1 = p[:, :, 0, :], p[:, :, 1, :]
        r01 = (a0 * a1.conj()).sum((1, 2))
        R[:, k] = np.stack([2 * r01.real, -2 * r01.imag, (abs(a0) ** 2).sum((1, 2)) - (abs(a1) ** 2).sum((1, 2))], 1)
    return R


def bloch_vectors(family, X, edges, L, c, batch=2000):
    X = np.asarray(X, dtype=np.float64); n = X.shape[1]; out = []
    for i in range(0, len(X), batch):
        Xb = X[i:i + batch]; out.append(bloch(simulate(build_circuit(family, Xb, edges, L, c), len(Xb), n), n))
    return np.concatenate(out, 0) if out else np.empty((0, n, 3))


def bloch_features(family, X, edges, L, c, batch=2000):
    """(N, 3n) flattened Bloch vectors: the features of the projected (1-RDM) kernel."""
    R = bloch_vectors(family, X, edges, L, c, batch); return R.reshape(len(R), -1)


# ------------------------------------------------------------------ independent dense implementation (verification only)
def _dense_gate(n, qubits, M):
    """Full 2^n x 2^n matrix of a 1- or 2-qubit gate by explicit index mapping (no reshapes, no einsum)."""
    D = 2 ** n; G = np.zeros((D, D), dtype=complex)
    bits = lambda i, q: (i >> (n - 1 - q)) & 1
    for out in range(D):
        for inn in range(D):
            if any(bits(out, q) != bits(inn, q) for q in range(n) if q not in qubits):
                continue
            ro = 0; ri = 0
            for q in qubits:
                ro = 2 * ro + bits(out, q); ri = 2 * ri + bits(inn, q)
            G[out, inn] = M[ro, ri]
    return G


def dense_state(gates, n, idx):
    """State of input number idx of the batch, built from dense matrices."""
    U = np.eye(2 ** n, dtype=complex)
    for kind, name, q, t in gates:
        tt = None if t is None else np.asarray([t[idx]])
        M = _single(name, tt) if kind == 1 else _two(name, tt)
        M = M if M.ndim == 2 else M[0]
        U = _dense_gate(n, q, M) @ U
    v = np.zeros(2 ** n, dtype=complex); v[0] = 1
    return U @ v


def dense_bloch(state, n):
    """Bloch vectors from explicit partial traces of |psi><psi| (verification only)."""
    rho = np.outer(state, state.conj()); out = np.empty((n, 3))
    X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
    for k in range(n):
        others = [q for q in range(n) if q != k]; rk = np.zeros((2, 2), dtype=complex)   # explicit partial trace
        for i in range(2 ** n):
            for j in range(2 ** n):
                bi = [(i >> (n - 1 - q)) & 1 for q in range(n)]; bj = [(j >> (n - 1 - q)) & 1 for q in range(n)]
                if all(bi[q] == bj[q] for q in others):
                    rk[bi[k], bj[k]] += rho[i, j]
        out[k] = [np.trace(rk @ P).real for P in (X, Y, Z)]
    return out


# ------------------------------------------------------------------ light cones
def gate_walk_cones(family, n, edges, L):
    """Upper bound: walk the gate list backwards from qubit k, adding both qubits of every two-qubit gate that
    touches the current set (an n x n boolean matrix, row k = cone of qubit k)."""
    g = build_circuit(family, np.zeros((1, n)), edges, L, 1.0); M = np.zeros((n, n), bool)
    for k in range(n):
        S = {k}
        for kind, _, q, _ in reversed(g):
            if kind == 2 and (q[0] in S or q[1] in S):
                S |= set(q)
        M[k, list(S)] = True
    return M


def sensitivity(family, n, edges, L, c=1.0, reps=24, delta=1.0, seed=7):
    """sens[k, j] = largest change of qubit k's Bloch vector when input x_j is shifted by delta (random Gaussian inputs)."""
    X0 = np.random.default_rng(seed).standard_normal((reps, n))
    Xs = [X0] + [X0 + delta * np.eye(n)[j][None, :] for j in range(n)]
    R = bloch_vectors(family, np.vstack(Xs), edges, L, c).reshape(n + 1, reps, n, 3)
    return np.abs(R[1:] - R[0][None]).max(axis=(1, 3)).T


CONE_THRESHOLD = 1e-13     # numerical zero is < 1e-14; the weakest genuine dependence found is > 1e-12 (checked in the notebook)


def dependence_cones(family, n, edges, L, **kw):
    """The light cone used in Stage 4: dep[k, j] = True if input x_j changes qubit k's Bloch vector (sensitivity above
    numerical zero). Returns (dep, sens)."""
    sens = sensitivity(family, n, edges, L, **kw)
    return sens > CONE_THRESHOLD, sens

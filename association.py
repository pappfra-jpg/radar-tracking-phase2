"""
association.py - ASSOCIAZIONE DEI DATI tracce <-> detection (modulo indipendente).

Per ogni coppia (traccia i, detection j) calcola la distanza di Mahalanobis al quadrato

        d2[i, j] = y^T  S_i^-1  y        con   y = detection_j - misura_predetta_i

dove S_i e' l'incertezza dello scarto (predizione + rumore del sensore) della traccia i.
Le coppie con d2 oltre il GATE vengono escluse; sulle altre si risolve l'assegnazione
globale ottima con l'algoritmo ungherese: prima si massimizza il numero di accoppiamenti,
poi si minimizza la somma dei d2.

Il modulo non conosce il filtro di Kalman ne' il radar: riceve solo numeri. Funziona con
misure a 1 dimensione (distanza) oggi, e a 2 o piu' (distanza + azimut) domani.

Uso:
    res = associate(pred, S, dets, gate_d2)
        pred : (n_tracce, d)    misura predetta di ogni traccia      [1D: anche lista di scalari]
        S    : (n_tracce, d, d) incertezza dello scarto di ogni traccia [1D: anche lista di varianze]
        dets : (n_det, d)       detection di questo frame            [1D: anche lista di scalari]
    res.matches           [(i_traccia, j_detection), ...]
    res.unmatched_tracks  [i_traccia, ...]   tracce senza detection (buco)
    res.unmatched_dets    [j_detection, ...] detection senza traccia (candidate a nuova traccia)
    res.d2                matrice completa delle distanze (per debug)

Gate: gate_d2 = 25 equivale a 5 sigma in 1D; oppure gate_from_prob(dim, 0.9999).
      gate_d2 puo' essere un numero (uguale per tutte le tracce) o una lista con un gate per traccia.

Richiede: numpy, scipy
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


@dataclass
class AssociationResult:
    matches: list = field(default_factory=list)
    unmatched_tracks: list = field(default_factory=list)
    unmatched_dets: list = field(default_factory=list)
    d2: np.ndarray = None


def gate_from_prob(dim, prob=0.9999):
    """Soglia su d2 che include una misura corretta con probabilita' 'prob' (chi-quadro, 'dim' gradi di liberta')."""
    return float(chi2.ppf(prob, df=dim))


def _prepare(pred, S, dets):
    """Porta gli ingressi a forma (n,d), (n,d,d), (m,d); accetta scalari per il caso 1D."""
    pred = np.asarray(pred, dtype=float)
    S = np.asarray(S, dtype=float)
    dets = np.asarray(dets, dtype=float)

    if pred.ndim <= 1:
        pred = pred.reshape(-1, 1)
    n, d = pred.shape
    if S.ndim <= 1:
        S = S.reshape(n, 1, 1)
    if S.shape != (n, d, d):
        raise ValueError(f"S deve avere forma ({n}, {d}, {d}), ricevuto {S.shape}")
    dets = dets.reshape(-1, d) if dets.size else dets.reshape(0, d)
    return pred, S, dets


def mahalanobis_d2(pred, S, dets):
    """Matrice (n_tracce, n_det) delle distanze di Mahalanobis al quadrato."""
    pred, S, dets = _prepare(pred, S, dets)
    n, m = len(pred), len(dets)
    d2 = np.empty((n, m))
    for i in range(n):
        y = dets - pred[i]                         # (m, d)
        x = np.linalg.solve(S[i], y.T)             # S^-1 y, (d, m)
        d2[i] = np.sum(y.T * x, axis=0)
    return d2


def associate(pred, S, dets, gate_d2=25.0):
    pred_a, S_a, dets_a = _prepare(pred, S, dets)
    n, m = len(pred_a), len(dets_a)

    if n == 0 or m == 0:
        return AssociationResult([], list(range(n)), list(range(m)), np.zeros((n, m)))

    d2 = mahalanobis_d2(pred_a, S_a, dets_a)

    # gate_d2: un numero uguale per tutte le tracce, oppure una lista con un gate per traccia
    gate = np.broadcast_to(np.asarray(gate_d2, dtype=float), (n,))
    ok = d2 <= gate[:, None]                       # coppie dentro il gate

    # Le coppie fuori gate costano BIG. BIG e' piu' grande di qualunque somma di costi
    # validi, quindi il risolutore evita le coppie fuori gate finche' puo' (massimizza
    # il numero di accoppiamenti validi) e solo dopo minimizza la somma dei d2.
    big = (min(n, m) + 1) * float(gate.max()) + 1.0
    cost = np.where(ok, d2, big)
    rows, cols = linear_sum_assignment(cost)

    matches = [(int(i), int(j)) for i, j in zip(rows, cols) if ok[i, j]]
    used_t = {i for i, _ in matches}
    used_d = {j for _, j in matches}
    return AssociationResult(
        matches=sorted(matches),
        unmatched_tracks=[i for i in range(n) if i not in used_t],
        unmatched_dets=[j for j in range(m) if j not in used_d],
        d2=d2,
    )


# ===========================================================================
# AUTOTEST: python association.py
# ===========================================================================
def _greedy(d2, gate):
    """Per confronto: la strategia 'coppia piu' vicina per prima', usata prima."""
    pairs = sorted((d2[i, j], i, j) for i in range(d2.shape[0]) for j in range(d2.shape[1]) if d2[i, j] <= gate)
    ti, dj, out = set(), set(), []
    for _, i, j in pairs:
        if i not in ti and j not in dj:
            ti.add(i); dj.add(j); out.append((i, j))
    return sorted(out)


def _brute(d2, gate):
    """Soluzione esatta per matrici piccole: (max accoppiamenti, minima somma di d2)."""
    n, m = d2.shape
    gate = np.broadcast_to(np.asarray(gate, dtype=float), (n,))
    best = (0, 0.0)

    def rec(i, used, cnt, tot):
        nonlocal best
        if i == n:
            if cnt > best[0] or (cnt == best[0] and tot < best[1] - 1e-12) or (best == (0, 0.0) and cnt == 0):
                best = (cnt, tot)
            return
        rec(i + 1, used, cnt, tot)                                  # traccia i senza detection
        for j in range(m):
            if j not in used and d2[i, j] <= gate[i]:
                rec(i + 1, used | {j}, cnt + 1, tot + d2[i, j])

    rec(0, frozenset(), 0, 0.0)
    return best


def _selftest():
    # 1) caso semplice
    r = associate([0.0, 1.0], [0.01, 0.01], [1.02, -0.03], gate_d2=25)
    assert r.matches == [(0, 1), (1, 0)], r.matches
    print("1) caso semplice: ok", r.matches)

    # 2) Hungarian contro 'coppia piu' vicina per prima': tracce a 0 e 1, detection a 0.6 e 1.9
    sig2 = 0.5 ** 2
    r = associate([0.0, 1.0], [sig2, sig2], [0.6, 1.9], gate_d2=100)
    g = _greedy(r.d2, 100)
    print(f"2) tracce a 0 e 1, detection a 0.6 e 1.9:  ungherese {r.matches}   greedy {g}")
    assert r.matches == [(0, 0), (1, 1)] and g == [(0, 1), (1, 0)]

    # 3) gating e casi vuoti: detection lontana senza traccia, traccia senza detection
    r = associate([1.0, 2.0], [0.0004, 0.0004], [1.01, 5.0], gate_d2=25)
    assert r.matches == [(0, 0)] and r.unmatched_tracks == [1] and r.unmatched_dets == [1], r
    assert associate([], [], [1.0]).unmatched_dets == [0]
    assert associate([1.0], [0.01], []).unmatched_tracks == [0]
    print("3) gating e casi vuoti: ok")

    # 4) misure 2D (distanza, azimut) con covarianze diverse: la Mahalanobis pesa le direzioni
    pred = [[1.0, 0.0], [1.0, 0.3]]
    S = [np.diag([0.01 ** 2, 0.2 ** 2]), np.diag([0.01 ** 2, 0.2 ** 2])]
    r = associate(pred, S, [[1.005, 0.28], [0.995, 0.05]], gate_d2=gate_from_prob(2, 0.999))
    assert r.matches == [(0, 1), (1, 0)], r.matches
    print("4) misure 2D: ok", r.matches)

    # 4b) gate diverso per traccia: due tracce nello stesso punto, una sola puo' prendere la detection a 3.0
    r = associate([0.0, 0.0], [1.0, 1.0], [3.0], gate_d2=[4.0, 100.0])
    assert r.matches == [(1, 0)] and r.unmatched_tracks == [0], r
    print("4b) gate per traccia: ok", r.matches)

    # 5) confronto con la soluzione esatta (forza bruta) su 400 casi casuali, 1D e 2D
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(400):
        d = int(rng.integers(1, 3))
        n, m = int(rng.integers(0, 5)), int(rng.integers(0, 5))
        pred = rng.uniform(0, 3, (n, d))
        S = np.array([np.diag(rng.uniform(0.05, 0.4, d) ** 2) for _ in range(n)]).reshape(n, d, d)
        dets = rng.uniform(0, 3, (m, d))
        gate = gate_from_prob(d, 0.99) * rng.uniform(0.2, 2.0, n)      # un gate diverso per traccia
        r = associate(pred, S, dets, gate)
        if n == 0 or m == 0:
            continue
        cnt, tot = _brute(r.d2, gate)
        got = (len(r.matches), sum(r.d2[i, j] for i, j in r.matches))
        if got[0] != cnt or abs(got[1] - tot) > 1e-9:
            bad += 1
    assert bad == 0, f"{bad} casi diversi dalla soluzione esatta"
    print("5) 400 casi casuali (gate diverso per traccia) = soluzione esatta (forza bruta): ok")


if __name__ == "__main__":
    _selftest()
    print("\ntutto ok")

"""
Tracker MULTI-bersaglio 1D (solo range) costruito sopra Kalman1D.

Ogni traccia ha il suo Kalman1D e UNO STATO:

    PROVVISORIA --(3 detection su 5)--> CONFERMATA --(0.5 s senza detection)--> PERSA
         |                                  ^                                     |
         | (2 buchi di fila: muore)         |                                     | (riaggancio)
                                            +---(3 detection su 5)--- RIAGGANCIATA <+
    RIAGGANCIATA --(2 buchi di fila: falso riaggancio)--> torna PERSA
    PERSA --(max_lost_s senza detection)--> cancellata

Ad ogni frame:
  1) predict delle tracce con filtro attivo (le PERSE restano ferme sull'ultima posizione)
  2) associazione detection <-> tracce attive (distanza di Mahalanobis + gate + ungherese)
  3) update delle tracce associate, "buco" per quelle senza detection
  4) RIAGGANCIO: le detection ancora libere possono riprendere una traccia PERSA se cadono
     entro   reacq_base + reacq_speed * (tempo dall'ultima detection)   dall'ultima posizione;
     la traccia mantiene lo stesso ID ma va riconfermata (cosi' un falso allarme non la resuscita)
  5) le detection ancora libere diventano tracce PROVVISORIE (se non sono a meno di
     min_separation da una traccia attiva: il radar non risolve bersagli piu' vicini)
  6) transizioni di stato e cancellazioni

Non dipende dall'ordine delle distanze nella riga del radar.

Uso:   tracker.step(zs)   -> tracce CONFERMATE   [{id, range, vel, coasting}]
       tracker.lost       -> tracce PERSE        [{id, range (ultima posizione), lost_s}]

Eseguito da solo: simula una scena e valuta il tracker (nessun hardware).
Richiede: numpy, scipy, matplotlib; association.py e kalman_phase1.py nella stessa cartella.
"""

from collections import Counter, deque

import numpy as np

from association import associate
from kalman_phase1 import Kalman1D


class Track:
    def __init__(self, z, meas_std, sigma_a, window):
        self.id = None                            # l'ID si assegna solo alla PRIMA conferma: i falsi allarmi non consumano numeri
        self.meas_std, self.sigma_a, self.window = meas_std, sigma_a, window
        self.kf = Kalman1D(z, meas_std, sigma_a)
        self.recent = deque([1], maxlen=window)   # 1 = detection associata, 0 = buco
        self.missed_run = 0                       # frame consecutivi senza detection
        self.state = "provvisoria"                # provvisoria | confermata | persa | riagganciata
        self.last_x = z                           # ultima posizione AGGIORNATA da una detection
        self.was_lost = False                     # True se ha gia' passato lo stato "persa"
        self.lost_frames = 0                      # frame dall'ultima detection (solo se was_lost)

    @property
    def confirmed(self):
        return self.state == "confermata"

    def restart(self, z):
        """Riaggancio: STESSO ID, filtro riavviato sulla nuova detection (la velocita' vecchia non vale piu')."""
        self.kf = Kalman1D(z, self.meas_std, self.sigma_a)
        self.recent = deque([1], maxlen=self.window)
        self.missed_run = 0
        self.state = "riagganciata"
        # last_x NON cambia: finche' il riaggancio non e' confermato resta l'ultima posizione
        # in cui la traccia e' stata vista davvero (un falso allarme non deve cancellarla).


class MultiTargetTracker:
    def __init__(self, dt, meas_std=0.015, sigma_a=2.0, gate_sigma=5.0,
                 confirm_m=3, confirm_n=5, max_coast_s=0.5, min_separation=0.08,
                 max_lost_s=5.0, reacq_speed=1.0, reacq_base=0.08):
        self.dt = dt
        self.meas_std = meas_std
        self.sigma_a = sigma_a
        self.gate_sigma = gate_sigma
        self.confirm_m = confirm_m            # detection necessarie per confermare...
        self.confirm_n = confirm_n            # ...entro gli ultimi N frame
        self.max_coast_s = max_coast_s        # senza detection per piu' di cosi' la traccia diventa PERSA
        self.min_separation = min_separation  # m: il radar non separa bersagli piu' vicini di cosi'
        self.max_lost_s = max_lost_s          # s dall'ultima detection dopo cui una traccia persa e' dimenticata
        self.reacq_speed = reacq_speed        # m/s: velocita' massima plausibile mentre non lo si vede
        self.reacq_base = reacq_base          # m: raggio di riaggancio subito dopo la perdita
        self.tracks = []
        self._next_id = 1                     # contatore degli ID (per tracker, assegnati alla prima conferma)

    # ------------------------------------------------------------------
    @property
    def lost(self):
        """Tracce perse (o in verifica dopo un riaggancio): ultima posizione nota e da quanto sono perse."""
        return [dict(id=tr.id, range=tr.last_x, lost_s=tr.lost_frames * self.dt)
                for tr in self.tracks if tr.state in ("persa", "riagganciata")]

    @staticmethod
    def _pred_and_S(kf):
        return float((kf.H @ kf.x)[0]), float((kf.H @ kf.P @ kf.H.T)[0, 0] + kf.R)

    # ------------------------------------------------------------------
    def step(self, zs):
        """zs = lista delle distanze di questo frame (anche vuota).
        Ritorna le tracce confermate come lista di dict."""
        dt = self.dt

        # 1) predict (le PERSE non si muovono) e orologio delle tracce perse
        active = [tr for tr in self.tracks if tr.state != "persa"]
        for tr in active:
            tr.kf.predict(dt)
        for tr in self.tracks:
            if tr.was_lost:
                tr.lost_frames += 1

        # 2) associazione con le tracce attive: Mahalanobis + gate + assegnazione ottima
        chosen = {}                           # indice in 'active' -> indice detection
        if active and zs:
            ps = [self._pred_and_S(tr.kf) for tr in active]
            res = associate([p for p, _ in ps], [S for _, S in ps], zs, gate_d2=self.gate_sigma ** 2)
            chosen = dict(res.matches)

        # 3) update / buco
        for i, tr in enumerate(active):
            if i in chosen:
                tr.kf.update(zs[chosen[i]])
                if tr.state != "riagganciata":
                    tr.last_x = float(tr.kf.x[0])
                tr.recent.append(1)
                tr.missed_run = 0
            else:
                tr.recent.append(0)
                tr.missed_run += 1
        matched = set(chosen.values())

        # 4) riaggancio: detection libere <-> tracce perse. Il raggio cresce col tempo perso;
        #    il costo e' la distanza assoluta, quindi vince la traccia persa piu' vicina.
        lost = [tr for tr in self.tracks if tr.state == "persa"]
        free = [j for j in range(len(zs)) if j not in matched]
        if lost and free:
            radius = [self.reacq_base + self.reacq_speed * tr.lost_frames * dt for tr in lost]
            res = associate([tr.last_x for tr in lost], [self.reacq_base ** 2] * len(lost),
                            [zs[j] for j in free],
                            gate_d2=[(r / self.reacq_base) ** 2 for r in radius])
            for i, jj in res.matches:
                lost[i].restart(zs[free[jj]])
                matched.add(free[jj])

        # 5) detection ancora libere -> nuove tracce provvisorie, ma non a meno di
        #    min_separation da una traccia attiva (sarebbe un duplicato o rumore)
        for j, z in enumerate(zs):
            if j in matched:
                continue
            if any(abs(z - tr.kf.x[0]) < self.min_separation for tr in self.tracks if tr.state != "persa"):
                continue
            self.tracks.append(Track(z, self.meas_std, self.sigma_a, self.confirm_n))

        # 6) transizioni di stato e cancellazioni
        alive = []
        for tr in self.tracks:
            if tr.state in ("provvisoria", "riagganciata") and sum(tr.recent) >= self.confirm_m:
                tr.state = "confermata"
                tr.was_lost, tr.lost_frames = False, 0
                tr.last_x = float(tr.kf.x[0])
                if tr.id is None:                            # prima conferma: nasce l'ID
                    tr.id = self._next_id
                    self._next_id += 1
            if tr.state == "confermata":
                if tr.missed_run * dt > self.max_coast_s:
                    tr.state = "persa"                       # non e' piu' mostrata come attiva
                    tr.was_lost = True
                    tr.lost_frames = tr.missed_run           # il tempo si conta dall'ultima detection
            elif tr.state == "provvisoria":
                if tr.missed_run >= 2:
                    continue                                 # falso allarme: muore senza lasciare traccia
            elif tr.state == "riagganciata":
                if tr.missed_run >= 2:
                    tr.state = "persa"                       # falso riaggancio: torna persa, il timer continua
            if tr.state == "persa" and tr.lost_frames * dt > self.max_lost_s:
                continue                                     # dimenticata
            alive.append(tr)
        self.tracks = alive

        return [dict(id=tr.id, range=tr.kf.x[0], vel=tr.kf.x[1], coasting=tr.missed_run > 0)
                for tr in self.tracks if tr.state == "confermata"]


# ===========================================================================
# SCENA SIMULATA PER VALUTARE IL TRACKER (nessun hardware)
# ===========================================================================
MIN_SEP = 0.08      # ASSUNZIONE: sotto 8 cm il detector fonde due bersagli in uno solo


def simulate_scene(dt, duration=10.0, seed=0, p_detect=0.95, p_false=0.015):
    rng = np.random.default_rng(seed)
    n = int(duration / dt)
    t = np.arange(n) * dt
    truths = {
        "A": 0.5 + 0.2 * t,                                            # si allontana
        "B": 2.5 - 0.2 * t,                                            # si avvicina, incrocia A a 5 s
        "C": np.where((t >= 6) & (t < 9), 0.9, np.nan),                # compare a 6 s, ferma, sparisce a 9 s
    }
    frames = []
    for k in range(n):
        zs = []
        for tr in truths.values():
            if not np.isnan(tr[k]) and rng.random() < p_detect:
                zs.append(tr[k] + rng.normal(0, 0.015))
        zs.sort()
        merged = []                                  # fusione se troppo vicini
        for z in zs:
            if merged and z - merged[-1][-1] < MIN_SEP:
                merged[-1].append(z)
            else:
                merged.append([z])
        zs = [float(np.mean(g)) for g in merged]
        if rng.random() < p_false:
            zs.append(float(rng.uniform(0.3, 2.6)))
        frames.append(sorted(zs))
    return t, truths, frames


def id_near(t, truth, history, t0, w=0.1, near=0.15):
    """ID della traccia piu' vicina al bersaglio vero attorno a t0 (quello piu' frequente)."""
    ids = []
    for k in np.where((t >= t0 - w) & (t <= t0 + w))[0]:
        if np.isnan(truth[k]):
            continue
        best = min(((abs(trk["range"] - truth[k]), trk["id"]) for trk in history[k]), default=None)
        if best is not None and best[0] < near:
            ids.append(best[1])
    return Counter(ids).most_common(1)[0][0] if ids else None


# incroci della scena: (nome, bersaglio, istante prima, istante dopo)
CROSSINGS = [("A-B a 5.0 s", "A", 2.0, 7.0), ("A-B a 5.0 s", "B", 2.0, 7.0),
             ("B-C a 8.0 s", "B", 7.0, 9.5), ("B-C a 8.0 s", "C", 6.5, 8.7)]


def evaluate(t, truths, history, max_coast_s=0.5, near=0.15, verbose=True):
    """Ritorna dict con copertura, RMS, identita' conservate dopo gli incroci e frame fantasma."""
    out = {"identity": [], "cov": {}, "rms": {}}
    for name, tr in truths.items():
        present = ~np.isnan(tr)
        errs, covered = [], 0
        for k in np.where(present)[0]:
            best = min((abs(trk["range"] - tr[k]) for trk in history[k]), default=None)
            if best is not None and best < near:
                covered += 1
                errs.append(best)
        out["cov"][name] = covered / present.sum() * 100
        out["rms"][name] = np.sqrt(np.mean(np.square(errs))) * 100 if errs else float("nan")

    for label, name, t_before, t_after in CROSSINGS:
        i0 = id_near(t, truths[name], history, t_before)
        i1 = id_near(t, truths[name], history, t_after)
        out["identity"].append((label, name, i0 is not None and i0 == i1))

    # frame fantasma = traccia confermata lontana da ogni bersaglio vero, ESCLUSI i
    # max_coast_s secondi dopo la sparizione di un bersaglio (li' e' voluto che coasti)
    vanish = []
    for tr in truths.values():
        pres = ~np.isnan(tr)
        vanish += [t[k] for k in range(1, len(t)) if pres[k - 1] and not pres[k]]
    ghost = 0
    for k in range(len(t)):
        if any(0 <= t[k] - tv <= max_coast_s + 0.05 for tv in vanish):
            continue
        if any(all(np.isnan(v[k]) or abs(trk["range"] - v[k]) >= near for v in truths.values())
               for trk in history[k]):
            ghost += 1
    out["ghost_frames"] = ghost

    if verbose:
        print(f"{'bersaglio':<10}{'copertura':>11}{'errore RMS':>13}")
        for name in truths:
            print(f"{name:<10}{out['cov'][name]:>10.1f}%{out['rms'][name]:>10.2f} cm")
        print()
        for label, name, ok in out["identity"]:
            print(f"identita' di {name} dopo incrocio {label}: {'conservata' if ok else 'SCAMBIATA'}")
        print(f"\nframe con traccia fantasma (esclusa la coda voluta): {ghost} su {len(t)}")
    return out


def summary_over_seeds(n_seeds=10, dt=0.00757):
    """Ripete la scena con rumori diversi: quante volte le identita' si conservano."""
    ok = {}
    ghosts = []
    for seed in range(n_seeds):
        t, truths, frames = simulate_scene(dt, seed=seed)
        trk = MultiTargetTracker(dt)
        res = evaluate(t, truths, [trk.step(z) for z in frames], verbose=False)
        for label, name, good in res["identity"]:
            ok.setdefault((label, name), []).append(good)
        ghosts.append(res["ghost_frames"])
    print(f"\n--- {n_seeds} scene con rumore diverso ---")
    for (label, name), v in ok.items():
        print(f"identita' di {name} dopo incrocio {label}: conservata {sum(v)}/{len(v)} volte")
    print(f"frame fantasma per scena: {ghosts}")


def main():
    import matplotlib.pyplot as plt

    dt = 0.00757
    t, truths, frames = simulate_scene(dt)
    tracker = MultiTargetTracker(dt)
    history = [tracker.step(zs) for zs in frames]

    print(f"frame: {len(frames)}   detection medie per frame: {np.mean([len(z) for z in frames]):.2f}\n")
    evaluate(t, truths, history)
    summary_over_seeds()

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, zs in enumerate(frames):
        ax.plot([t[k]] * len(zs), zs, ".", color="lightgray", markersize=2)
    for name, tr in truths.items():
        ax.plot(t, tr, "--", color="black", linewidth=1, label=f"verita' {name}" if name == "A" else None)
    by_id = {}
    for k, trs in enumerate(history):
        for trk in trs:
            by_id.setdefault(trk["id"], ([], []))
            by_id[trk["id"]][0].append(t[k])
            by_id[trk["id"]][1].append(trk["range"])
    cmap = plt.get_cmap("tab10")
    for tid, (xs, ys) in by_id.items():
        ax.plot(xs, ys, "-", color=cmap(tid % 10), linewidth=2, label=f"traccia #{tid}")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Range (m)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("multitarget_test.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()

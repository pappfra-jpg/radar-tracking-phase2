"""
FASE 2 - Kalman 1D in tempo reale sulle detection del radar via seriale.

Un solo bersaglio (il multi-target viene dopo).

Struttura:
    [thread lettore]  -- coda -->  [timer del grafico: filtro + plot]

Il thread fa SOLO readline() e parsing, cosi' il disegno del grafico non puo'
mai far perdere o ritardare i byte in arrivo (il problema dei buchi seriali).

Richiede: numpy, matplotlib, pyserial (e kalman_phase1.py nella stessa cartella)
Prova senza radar: metti USE_SIMULATOR = True
"""

import queue
import re
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kalman_phase1 import Kalman1D          # lo STESSO filtro della fase 1

# ---------------------------------------------------------------------------
# PARAMETRI
# ---------------------------------------------------------------------------
USE_SIMULATOR = False        # True = finto radar, per provare senza hardware
PORT = "COM6"
BAUDRATE = 921600

DT = 0.00757                 # s tra due frame (dal tuo CSV: durata/intervalli = 7.56 ms)
MEAS_STD = 0.015             # m, rumore del sensore (misurato con bersaglio fermo)
SIGMA_A = 2.0                # m/s^2, vedi fase 1 (2-5 andava bene nei test)
GATE_SIGMA = 5.0             # scarta misure oltre N sigma dalla predizione
MAX_GATED_RUN = 15           # frame consecutivi scartati -> traccia persa, si riparte
MAX_COAST_S = 0.5            # secondi senza detection -> la traccia viene eliminata

WINDOW_S = 10.0              # secondi mostrati nel grafico

# Il tempo scorre con l'OROLOGIO, non con le righe. Se mancano piu' di TOLL_FRAMES frame
# rispetto al tempo reale, il firmware sta tacendo (nessuna detection) e il filtro fa
# passi "a vuoto" (solo predizione). Sotto questa soglia i ritardi sono jitter della
# seriale (nel tuo CSV buchi fino a ~4 frame) e vengono ignorati.
TOLL_FRAMES = 12             # circa 90 ms


# ---------------------------------------------------------------------------
# PARSING DELLE RIGHE (identico allo script di acquisizione)
# ---------------------------------------------------------------------------
# Il firmware stampa "N detected distances: d1 m d2 m" se N>=1 e "0 detected distances" (SENZA i due
# punti) quando non c'e' nessuna detection: verificato con raw_dump.py. I due punti sono opzionali.
# (Con i due punti obbligatori le righe a zero detection venivano scartate in silenzio.)
FRAME_RE = re.compile(r"^(\d+)\s+detected distances:?(.*)")
DIST_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m")


def parse_line(line):
    """Ritorna la lista delle distanze, oppure None se la riga e' troncata/corrotta."""
    m = FRAME_RE.match(line.strip())
    if not m:
        return None
    count = int(m.group(1))
    distances = [float(x) for x in DIST_RE.findall(m.group(2))]
    if len(distances) != count:
        return None
    return distances


# ---------------------------------------------------------------------------
# FINTO RADAR (solo per provare senza hardware)
# ---------------------------------------------------------------------------
class FakeSerial:
    """Una pallina che va avanti e indietro tra 0.5 e 2.0 m, con frame vuoti e falsi allarmi."""

    def __init__(self, dt, silent=False, hide=None):
        self.dt = dt
        self.silent = silent          # True: senza detection il firmware NON stampa nulla
        self.hide = hide              # (t0, t1) in secondi: bersaglio assente in questo intervallo
        self.k = 0
        self.t_next = time.perf_counter()
        self.rng = np.random.default_rng()

    def reset_input_buffer(self):
        pass

    def close(self):
        pass

    def readline(self):
        self.t_next += self.dt                       # rispetta il ritmo di 132 Hz
        wait = self.t_next - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        t = self.k * self.dt
        self.k += 1
        phase = (t % 8.0) / 8.0                      # onda triangolare, periodo 8 s
        truth = 0.5 + 1.5 * (1 - abs(2 * phase - 1))
        r = self.rng.random()
        absent = self.hide is not None and self.hide[0] <= t < self.hide[1]
        if absent or r < 0.04:                       # frame senza detection
            return b"" if self.silent else b"0 detected distances\n"
        d = truth + self.rng.normal(0, 0.015)
        if r > 0.98:                                 # falso allarme
            return f"2 detected distances: {d:.6f} m {self.rng.uniform(0.3, 2.5):.6f} m \n".encode()
        return f"1 detected distances: {d:.6f} m \n".encode()


# ---------------------------------------------------------------------------
# THREAD LETTORE: solo readline + parsing, mette le distanze in coda
# ---------------------------------------------------------------------------
def reader_thread(ser, q, stop, stats=None):
    """Legge la seriale e mette in coda le detection di ogni riga riconosciuta.
    Le righe NON riconosciute vengono contate in stats["rejected"] (la prima, spesso troncata, no)
    cosi' un errore di formato non resta invisibile."""
    first = True
    while not stop.is_set():
        line = ser.readline().decode("ascii", errors="ignore")
        zs = parse_line(line)
        if zs is not None:
            q.put(zs)
        elif stats is not None and line.strip() and not first:
            stats["rejected"] = stats.get("rejected", 0) + 1
            stats["last_bad"] = line.strip()[:60]
        first = False


# ---------------------------------------------------------------------------
# TRACKER A BERSAGLIO SINGOLO: un frame per volta
# ---------------------------------------------------------------------------
class SingleTargetTracker:
    def __init__(self):
        self.kf = None
        self.gated_run = 0       # frame consecutivi scartati dal gate
        self.missed_run = 0      # frame consecutivi senza detection
        self.gated = 0           # contatori totali, solo per il display
        self.reinit = 0
        self.dropped = 0

    def step(self, zs, dt):
        """Ritorna (z, range, velocita', innovazione, sigma innovazione, accettata).
        Tutti NaN se non c'e' una traccia; innovazione NaN se nel frame non c'e' una misura."""
        nan = float("nan")

        if self.kf is None:                          # in attesa della prima detection
            if zs:
                self.kf = Kalman1D(zs[0], MEAS_STD, SIGMA_A)
                self.missed_run = 0
                return zs[0], zs[0], 0.0, nan, nan, False
            return nan, nan, nan, nan, nan, False

        self.kf.predict(dt)                          # sempre

        z = nan
        y_out = s_out = nan
        used = False
        if zs:
            self.missed_run = 0
            z = min(zs, key=lambda d: abs(d - self.kf.x[0]))   # la piu' vicina alla predizione
            y, S = self.kf.innovation(z)
            y_out, s_out = y, float(np.sqrt(S))
            if abs(y) <= GATE_SIGMA * np.sqrt(S):
                self.kf.update(z)
                used = True
                self.gated_run = 0
            else:
                self.gated += 1
                self.gated_run += 1
                if self.gated_run >= MAX_GATED_RUN:  # traccia persa: riparto da questa misura
                    self.kf = Kalman1D(z, MEAS_STD, SIGMA_A)
                    self.reinit += 1
                    self.gated_run = 0
        else:
            self.missed_run += 1
            if self.missed_run * dt > MAX_COAST_S:   # bersaglio sparito: elimino la traccia
                self.kf = None
                self.dropped += 1
                return nan, nan, nan, nan, nan, False

        return z, self.kf.x[0], self.kf.x[1], y_out, s_out, used


# ---------------------------------------------------------------------------
# APP: grafico + funzione di aggiornamento
# ---------------------------------------------------------------------------
def make_app(ser):
    q = queue.Queue()
    stop = threading.Event()
    stats = {"rejected": 0, "last_bad": ""}                       # righe non riconosciute dal parser
    threading.Thread(target=reader_thread, args=(ser, q, stop, stats), daemon=True).start()

    tracker = SingleTargetTracker()
    n_keep = int(WINDOW_S / DT) + 20
    T = deque(maxlen=n_keep)         # tempo ricostruito
    Z = deque(maxlen=n_keep)         # detection scelte
    R = deque(maxlen=n_keep)         # range stimato
    V = deque(maxlen=n_keep)         # velocita' stimata
    Y = deque(maxlen=n_keep)         # innovazione (misura - predizione)
    SG = deque(maxlen=n_keep)        # sigma attesa dell'innovazione
    U = deque(maxlen=n_keep)         # True se la misura e' passata dal gate
    hist = deque()                   # (tempo, n. righe) per misurare le righe/s
    frame_idx = [0]                  # frame elaborati = tempo virtuale / DT
    lines_total = [0]                # righe realmente ricevute dalla seriale
    clock = {"t0": time.perf_counter(), "silent": 0}   # orologio e frame inseriti a vuoto
    reported = [0]                                    # righe non riconosciute gia' segnalate in console

    fig, (a1, a2, a3) = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    (l_z,) = a1.plot([], [], ".", markersize=3, color="tab:orange", label="detection")
    (l_r,) = a1.plot([], [], "-", linewidth=1.8, color="tab:blue", label="Kalman")
    a1.set_ylabel("Range (m)")
    a1.legend(loc="upper right")
    a1.grid(alpha=0.3)

    (l_v,) = a2.plot([], [], color="tab:green")
    a2.axhline(0, color="gray", linestyle="--")
    a2.set_ylabel("Velocita' stimata (m/s)")
    a2.grid(alpha=0.3)

    # terzo pannello: innovazione = misura - predizione. Se il filtro e' ben tarato
    # sta quasi tutta dentro la fascia +-2 sigma (circa 95%).
    (l_ok,) = a3.plot([], [], ".", markersize=3, color="tab:purple", label="innovazione (accettata)")
    (l_bad,) = a3.plot([], [], "x", markersize=5, color="red", label="scartata dal gate")
    (l_hi,) = a3.plot([], [], "--", color="gray", linewidth=1, label="+-2 sigma attesi")
    (l_lo,) = a3.plot([], [], "--", color="gray", linewidth=1)
    a3.axhline(0, color="gray", linewidth=0.5)
    a3.set_ylabel("Innovazione (cm)")
    a3.set_xlabel("Tempo (s)")
    a3.legend(loc="upper right", fontsize=8, ncol=3)
    a3.grid(alpha=0.3)

    a1.set_title("titolo")                      # segnaposto: lo spazio per i titoli
    a3.set_title("titolo", fontsize=10)         # va riservato PRIMA di tight_layout
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    def push(zs):
        """Un passo del filtro = un frame. zs vuoto = frame senza detection."""
        frame_idx[0] += 1
        t = frame_idx[0] * DT
        z, r, v, y, sg, used = tracker.step(zs, DT)
        T.append(t); Z.append(z); R.append(r); V.append(v)
        Y.append(y); SG.append(sg); U.append(used)

    def update(_frame):
        # 1) righe arrivate: ogni riga e' un frame (con le sue detection)
        while True:
            try:
                zs = q.get_nowait()
            except queue.Empty:
                break
            lines_total[0] += 1
            push(zs)

        # 2) il tempo scorre anche senza righe: confronto i frame elaborati con quelli
        #    attesi dall'orologio. Un ritardo piccolo e' jitter; uno grande e' silenzio
        #    del firmware -> inserisco frame vuoti (solo predizione, "buco" per il tracker).
        gap = (time.perf_counter() - clock["t0"]) / DT - frame_idx[0]
        if gap > TOLL_FRAMES:
            for _ in range(int(gap - TOLL_FRAMES / 2)):
                push([])
                clock["silent"] += 1
        elif gap < -2 * TOLL_FRAMES:                 # siamo in anticipo (righe piu' veloci di DT):
            clock["t0"] += gap * DT                  # riallineo l'orologio invece di accumulare

        if not T:
            return []

        # 2) righe al secondo, misurate su circa 1 s (se e' molto sotto 1/DT, qualcosa non torna)
        now = time.perf_counter()
        hist.append((now, lines_total[0]))
        while len(hist) > 1 and now - hist[1][0] >= 1.0:
            hist.popleft()
        t_old, n_old = hist[0]
        rate = (lines_total[0] - n_old) / (now - t_old) if now > t_old else 0.0

        # 3) aggiorno le linee
        t_arr = np.array(T)
        z_arr, r_arr, v_arr = np.array(Z), np.array(R), np.array(V)
        l_z.set_data(t_arr, z_arr)
        l_r.set_data(t_arr, r_arr)
        l_v.set_data(t_arr, v_arr)

        a1.set_xlim(max(0, t_arr[-1] - WINDOW_S), max(WINDOW_S, t_arr[-1]))
        vals = z_arr[np.isfinite(z_arr)]
        if vals.size:
            lo, hi = vals.min(), vals.max()
            pad = max(0.15, (hi - lo) * 0.1)
            a1.set_ylim(lo - pad, hi + pad)
        vmax = np.nanmax(np.abs(v_arr)) if np.isfinite(v_arr).any() else 0.5
        a2.set_ylim(-max(0.5, vmax * 1.2), max(0.5, vmax * 1.2))

        # innovazione: punti accettati, scartati (rossi, tenuti sul bordo se fuori scala), fascia +-2 sigma
        y_arr = np.array(Y) * 100
        band = 2 * np.array(SG) * 100
        u_arr = np.array(U, dtype=bool)
        cons = np.isfinite(y_arr)                    # frame in cui e' arrivata una misura
        good = cons & u_arr
        bad = cons & ~u_arr
        lim = 6.0
        if good.any():
            lim = max(lim, 1.3 * np.abs(y_arr[good]).max())
        if np.isfinite(band).any():
            lim = max(lim, 1.3 * np.nanmax(band))
        l_ok.set_data(t_arr[good], y_arr[good])
        l_bad.set_data(t_arr[bad], np.clip(y_arr[bad], -lim, lim))
        l_hi.set_data(t_arr, band)
        l_lo.set_data(t_arr, -band)
        a3.set_ylim(-lim, lim)
        if cons.sum() >= 30:                         # percentuale dentro +-2 sigma (misure scartate incluse)
            inside = np.mean(np.abs(y_arr[cons]) <= band[cons]) * 100
            a3.set_title(f"Innovazione: dentro +-2 sigma {inside:.0f}%  (ideale ~95%)", fontsize=10)

        stato = "TRACCIA ATTIVA" if tracker.kf is not None else "nessuna traccia"
        if stats["rejected"] > reported[0] and reported[0] < 5:      # avviso in console
            reported[0] = stats["rejected"]
            print(f"[parser] riga non riconosciuta (n.{stats['rejected']}): {stats['last_bad']!r}")
        a1.set_title(f"righe/s: {rate:.0f} (attese ~{1/DT:.0f})  non lette: {stats['rejected']}  "
                     f"frame a vuoto: {clock['silent']}  |  {stato}  |  "
                     f"scartate: {tracker.gated}  reinit: {tracker.reinit}  eliminate: {tracker.dropped}",
                     fontsize=10)
        return [l_z, l_r, l_v, l_ok, l_bad]

    return fig, update, stop


def main():
    if USE_SIMULATOR:
        ser = FakeSerial(DT)
    else:
        import serial
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        ser.reset_input_buffer()                     # scarta dati vecchi nel buffer

    fig, update, stop = make_app(ser)
    fig.canvas.mpl_connect("close_event", lambda _e: stop.set())

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    stop.set()
    ser.close()
    return ani


if __name__ == "__main__":
    main()

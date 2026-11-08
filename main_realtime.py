"""
main_realtime.py  -  il MAIN: collega i moduli gia' fatti, non ricalcola niente.

    kalman_realtime.py : parse_line(), reader_thread()   lettura seriale in thread
    multitarget.py     : MultiTargetTracker              associazione + tracce
    kalman_phase1.py   : Kalman1D                        (usato dal tracker)

Flusso:
    [thread: seriale -> parse] --coda--> [timer grafico: tracker -> plot]

Qui dentro ci sono solo: configurazione, collegamento e grafico.
Richiede: numpy, matplotlib, pyserial.  Tutti i file nella stessa cartella.
Prova senza radar: USE_SIMULATOR = True
"""

import queue
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kalman_realtime import parse_line, reader_thread
from multitarget import MultiTargetTracker, simulate_scene

# ---------------------------------------------------------------------------
# CONFIGURAZIONE (tutto in un posto)
# ---------------------------------------------------------------------------
USE_SIMULATOR = False        # True = finto radar con 3 bersagli, per provare senza hardware
PORT = "COM6"
BAUDRATE = 921600

DT = 0.00757                 # s tra due frame (dal tuo CSV: 7.56 ms)
MEAS_STD = 0.015             # m, rumore del sensore (misurato con bersaglio fermo)
SIGMA_A = 2.0                # m/s^2, quanto puo' "sorprendere" un bersaglio
GATE_SIGMA = 5.0             # associazione: scarta misure oltre N sigma dalla predizione
CONFIRM_M, CONFIRM_N = 3, 5  # una traccia e' vera dopo M detection su N frame
MAX_COAST_S = 0.5            # una traccia senza detection muore dopo questi secondi
MIN_SEPARATION = 0.08        # m: sotto questa distanza due bersagli non si distinguono

# Tracce PERSE ("blinking"): dopo MAX_COAST_S senza detection la traccia non e' piu' attiva ma
# resta in memoria per MAX_LOST_S. Una detection libera entro  REACQ_BASE + REACQ_SPEED * t
# dall'ultima posizione (t = tempo dall'ultima detection) la riaggancia con lo STESSO ID.
MAX_LOST_S = 5.0             # s
REACQ_SPEED = 1.0            # m/s: velocita' massima plausibile mentre non lo si vede
REACQ_BASE = 0.08            # m: raggio di riaggancio subito dopo la perdita

WINDOW_S = 10.0              # secondi mostrati nel grafico

# Il tempo scorre con l'OROLOGIO, non con le righe: se mancano piu' di TOLL_FRAMES frame
# rispetto al tempo reale il firmware sta tacendo (nessuna detection) e il tracker fa
# passi "a vuoto" (solo predizione). Sotto la soglia sono ritardi di seriale e si ignorano.
TOLL_FRAMES = 12             # circa 90 ms


# ---------------------------------------------------------------------------
# FINTO RADAR MULTI-BERSAGLIO (solo per provare senza hardware)
# Riusa la scena di multitarget.simulate_scene e la "stampa" nel formato del radar.
# La scena dura 10 s e poi ricomincia (al riavvio i bersagli "saltano").
# ---------------------------------------------------------------------------
class ReplaySerial:
    def __init__(self, dt, silent=False, hide=None):
        _, _, self.frames = simulate_scene(dt, duration=10.0, seed=int(time.time()) % 1000)
        self.silent = silent          # True: senza detection il firmware NON stampa nulla
        self.hide = hide              # (t0, t1) in secondi: nessun bersaglio in questo intervallo
        self.dt = dt
        self.k = 0
        self.t_next = time.perf_counter()

    def reset_input_buffer(self):
        pass

    def close(self):
        pass

    def readline(self):
        self.t_next += self.dt                       # rispetta il ritmo di ~132 Hz
        wait = self.t_next - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        zs = self.frames[self.k % len(self.frames)]
        t = self.k * self.dt
        self.k += 1
        if self.hide is not None and self.hide[0] <= t < self.hide[1]:
            zs = []
        if not zs and self.silent:
            return b""
        # come il firmware vero: con zero detection niente due punti
        text = (f"{len(zs)} detected distances:" + "".join(f" {z:.6f} m" for z in zs) + " \n") if zs \
            else "0 detected distances\n"
        return text.encode()


# ---------------------------------------------------------------------------
# COLLEGAMENTO + GRAFICO
# ---------------------------------------------------------------------------
def make_app(ser):
    q = queue.Queue()
    stop = threading.Event()
    stats = {"rejected": 0, "last_bad": ""}                       # righe non riconosciute dal parser
    threading.Thread(target=reader_thread, args=(ser, q, stop, stats), daemon=True).start()

    tracker = MultiTargetTracker(
        dt=DT, meas_std=MEAS_STD, sigma_a=SIGMA_A, gate_sigma=GATE_SIGMA,
        confirm_m=CONFIRM_M, confirm_n=CONFIRM_N, max_coast_s=MAX_COAST_S,
        min_separation=MIN_SEPARATION,
        max_lost_s=MAX_LOST_S, reacq_speed=REACQ_SPEED, reacq_base=REACQ_BASE,
    )

    n_keep = int(WINDOW_S / DT) + 20
    dets = deque(maxlen=n_keep * 3)      # (t, distanza): tutte le detection
    hist = {}                            # id traccia -> deque di (t, range, velocita')
    lines = {}                           # id traccia -> (linea range, linea velocita')
    hist_lost = {}                       # id traccia persa -> deque di (t, ultima posizione)
    lines_lost = {}                      # id traccia persa -> linea tratteggiata
    rate_hist = deque()                  # (tempo reale, n. righe), per le righe/s
    state = {"frame": 0, "t": 0.0, "n_active": 0}   # frame elaborati (tempo virtuale = frame * DT)
    lines_total = [0]                                # righe realmente ricevute dalla seriale
    clock = {"t0": time.perf_counter(), "silent": 0}  # orologio e frame inseriti a vuoto
    reported = [0]                                   # righe non riconosciute gia' segnalate in console
    cmap = plt.get_cmap("tab10")

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    (l_det,) = a1.plot([], [], ".", markersize=3, color="lightgray", label="detection")
    a1.set_ylabel("Range (m)")
    a1.grid(alpha=0.3)
    a2.axhline(0, color="gray", linestyle="--")
    a2.set_ylabel("Velocita' stimata (m/s)")
    a2.set_xlabel("Tempo (s)")
    a2.grid(alpha=0.3)
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    def push(zs):
        """Un passo del tracker = un frame. zs vuoto = frame senza detection."""
        state["frame"] += 1
        state["t"] = t = state["frame"] * DT
        for z in zs:
            dets.append((t, z))
        confirmed = tracker.step(zs)
        state["n_active"] = len(confirmed)           # solo le tracce vive in questo istante
        for l in tracker.lost:                       # tracce perse: ultima posizione nota
            hist_lost.setdefault(l["id"], deque(maxlen=n_keep)).append((t, l["range"]))
        for trk in confirmed:
            h = hist.setdefault(trk["id"], deque(maxlen=n_keep))
            if h and t - h[-1][0] > 1.5 * DT:        # buco (traccia persa e riagganciata): interrompo la linea
                h.append((h[-1][0] + DT, float("nan"), float("nan")))
            h.append((t, trk["range"], trk["vel"]))

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
        #    attesi dall'orologio. Ritardo piccolo = jitter; grande = silenzio del firmware
        #    -> frame vuoti (solo predizione, "buco" per il tracker).
        gap = (time.perf_counter() - clock["t0"]) / DT - state["frame"]
        if gap > TOLL_FRAMES:
            for _ in range(int(gap - TOLL_FRAMES / 2)):
                push([])
                clock["silent"] += 1
        elif gap < -2 * TOLL_FRAMES:                 # in anticipo (righe piu' veloci di DT):
            clock["t0"] += gap * DT                  # riallineo l'orologio

        if state["frame"] == 0:
            return []
        t_now = state["t"]

        for tid in [i for i, h in hist_lost.items() if h[-1][0] < t_now - WINDOW_S]:
            del hist_lost[tid]
            lines_lost.pop(tid).remove()

        # 2) elimino le tracce uscite dalla finestra (e le loro linee)
        for tid in [i for i, h in hist.items() if h[-1][0] < t_now - WINDOW_S]:
            del hist[tid]
            for ln in lines.pop(tid, ()):
                ln.remove()

        # 3) righe/s misurate su ~1 s (se e' molto sotto 1/DT, qualcosa non torna)
        now = time.perf_counter()
        rate_hist.append((now, lines_total[0]))
        while len(rate_hist) > 1 and now - rate_hist[1][0] >= 1.0:
            rate_hist.popleft()
        t_old, n_old = rate_hist[0]
        rate = (lines_total[0] - n_old) / (now - t_old) if now > t_old else 0.0

        # 4) aggiorno le linee
        d = np.array(dets) if dets else np.empty((0, 2))
        l_det.set_data(d[:, 0], d[:, 1])
        for tid, h in hist_lost.items():              # persa = linea tratteggiata ferma sull'ultima posizione
            arr = np.array(h)
            if tid not in lines_lost:
                (ll,) = a1.plot([], [], ":", linewidth=2.5, color=cmap(tid % 10))
                lines_lost[tid] = ll
            lines_lost[tid].set_data(arr[:, 0], arr[:, 1])
        for tid, h in hist.items():
            arr = np.array(h)
            if tid not in lines:
                col = cmap(tid % 10)
                (lr,) = a1.plot([], [], "-", linewidth=1.8, color=col, label=f"traccia #{tid}")
                (lv,) = a2.plot([], [], "-", linewidth=1.5, color=col)
                lines[tid] = (lr, lv)
            lines[tid][0].set_data(arr[:, 0], arr[:, 1])
            lines[tid][1].set_data(arr[:, 0], arr[:, 2])

        a1.set_xlim(max(0, t_now - WINDOW_S), max(WINDOW_S, t_now))
        vis = d[d[:, 0] >= t_now - WINDOW_S, 1]
        if vis.size:
            pad = max(0.15, (vis.max() - vis.min()) * 0.1)
            a1.set_ylim(vis.min() - pad, vis.max() + pad)
        def _vmax(h):                                # ignora i NaN dei buchi
            v = np.abs(np.array(h)[:, 2])
            v = v[np.isfinite(v)]
            return v.max() if v.size else 0.0
        vmax = max([_vmax(h) for h in hist.values()], default=0.5)
        a2.set_ylim(-max(0.5, vmax * 1.2), max(0.5, vmax * 1.2))

        if stats["rejected"] > reported[0] and reported[0] < 5:      # avviso in console
            reported[0] = stats["rejected"]
            print(f"[parser] riga non riconosciuta (n.{stats['rejected']}): {stats['last_bad']!r}")
        a1.set_title(f"righe/s: {rate:.0f} (attese ~{1/DT:.0f})  non lette: {stats['rejected']}  "
                     f"frame a vuoto: {clock['silent']}  |  "
                     f"attive: {state['n_active']}  perse: {len(tracker.lost)}", fontsize=10)
        a1.legend(loc="upper right", fontsize=8, ncol=2)
        return []

    return fig, update, stop, tracker


def main():
    if USE_SIMULATOR:
        ser = ReplaySerial(DT)
    else:
        import serial
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        ser.reset_input_buffer()                     # scarta dati vecchi nel buffer

    fig, update, stop, _tracker = make_app(ser)
    fig.canvas.mpl_connect("close_event", lambda _e: stop.set())

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
    stop.set()
    ser.close()
    return ani


if __name__ == "__main__":
    main()

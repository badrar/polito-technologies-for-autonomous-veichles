# ==================== LIBRERIE ====================
import sys

import cv2                                  # OpenCV per processare immagini/video
import mediapipe as mp                      # MediaPipe per rilevamento facce
import numpy as np                          # NumPy per operazioni numeriche
import time                                 # Modulo per misurare il tempo
import statistics as st                     # Statistiche (non utilizzato)
import os                                   # Modulo per percorsi file
import urllib.request                       # Per scaricare il modello

from mediapipe.tasks import python
from mediapipe.tasks.python import vision   # Task di visione di MediaPipe

import numpy as np
from scipy.signal import butter, filtfilt, detrend, welch
from sklearn.decomposition import FastICA
from collections import deque



#? Kinds of distractions:
#? - Owl long distraction: drivers gaze away from the road for 5s
#? - - Shall be reported to the driver until the gaze returns to the road
#? - Owl short distraction: drivers gaze away from the road and back to the 
#?      road for a total of 10s within 30s
#? - - Shall be reported to the driver until the gaze returns to the road for 2s
#? - Lizard long distraction: same as owl, but the head remains in
#?     forward gaze position, while the eyes gaze at a different location
#? - Lizard short distraction: same as owl, but the head remains in
#?     forward gaze position, while the eyes gaze at a different location
#? - Microsleep: warning shall be provided if the driver keeps both eyes
#?     closed for at least 4 seconds, until the drives keeps the eyes open for >= 2s
#? - warning shall be provided if the driver keeps both eyes
#?      closed for at least 7 seconds, until the drives keeps the eyes open for at
#?      least 2 seconds


# ==================== CARICAMENTO MODELLO ====================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    # Scarica il modello pre-addestrato se non è presente localmente
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print("Downloading face_landmarker.task...")
    urllib.request.urlretrieve(url, MODEL_PATH)

# ==================== CONFIGURAZIONE FACELANDMARKER ====================
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,  # Modalità video per frame consecutivi
    num_faces=1,                            # Rileva solo 1 volto
    min_face_detection_confidence=0.5,      # Soglia di confidenza rilevamento
    min_face_presence_confidence=0.5,       # Soglia di presenza viso
    min_tracking_confidence=0.5,            # Soglia di tracciamento
)

# Crea il rilevatore di punti di riferimento facciali
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ==================== DEFINIZIONE PUNTI DI RIFERIMENTO FACCIALI ====================
# 468 punti totali di riferimento sulla faccia secondo MediaPipe
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]  # Indici occhio sinistro
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]    # Indici occhio destro
LEFT_IRIS  = [473, 474, 475, 476, 477]   # Indici iride sinistra
RIGHT_IRIS = [468, 469, 470, 471, 472]   # Indici iride destra
NOSE_TIP   = [45, 4, 275]                 # Indici punta naso

# Angoli occhio (outer, inner) e contorni palpebre (da outer a inner)
RIGHT_OUTER, RIGHT_INNER = 33, 133
RIGHT_UPPER_LID = [246, 161, 160, 159, 158, 157]
RIGHT_LOWER_LID = [7,   163, 144, 145, 153, 154]

LEFT_OUTER, LEFT_INNER = 362, 263
LEFT_UPPER_LID = [466, 388, 387, 386, 385, 384]
LEFT_LOWER_LID = [249, 390, 373, 374, 380, 381]


# rppg

FOREHEAD_ROI = [109, 10, 338, 337, 336, 9, 107, 108]  # forehad polygon
BPM_WINDOW_S       = 8.0   # seconds for BPM estimation window
BPM_UPDATE_EVERY_N = 60     # recompute BPM every N frames (for efficiency)

from enum import Enum

class MicrosleepState(Enum):
    NORMAL    = 0
    WARNING_4 = 1   # occhi chiusi >= 4s
    WARNING_7 = 2   # occhi chiusi >= 7s
    RECOVERY  = 3   # occhi aperti, in attesa dei 2s


class MicrosleepDetector:
    CLOSED_4S   = 4.0
    CLOSED_7S   = 7.0
    RECOVERY_2S = 2.0

    def __init__(self):
        self.state          = MicrosleepState.NORMAL
        self.closed_since   = None   # timestamp inizio chiusura corrente
        self.opened_since   = None   # timestamp inizio apertura (recovery)
        self._recovery_from = None   # stato da cui si sta recuperando

    def update(self, is_closed: bool, now: float) -> MicrosleepState:
        if is_closed:
            self.opened_since = None          # interrompe recovery
            if self.closed_since is None:
                self.closed_since = now
            closed_dur = now - self.closed_since
            if closed_dur >= self.CLOSED_7S:
                self.state = MicrosleepState.WARNING_7
            elif closed_dur >= self.CLOSED_4S:
                self.state = MicrosleepState.WARNING_4
            # se già in WARNING/RECOVERY e chiusura < 4s: stato invariato
        else:
            self.closed_since = None
            if self.state == MicrosleepState.NORMAL:
                self.opened_since = None
            else:
                if self.opened_since is None:
                    self.opened_since   = now
                    self._recovery_from = self.state
                if now - self.opened_since >= self.RECOVERY_2S:
                    self.state          = MicrosleepState.NORMAL
                    self.opened_since   = None
                    self._recovery_from = None
                else:
                    self.state = MicrosleepState.RECOVERY
        return self.state


class DistractionState(Enum):
    NORMAL        = 0
    WARNING_LONG  = 1   # singolo episodio >= 5s, recovery immediata
    WARNING_SHORT = 2   # accumulato >= 10s in 30s, recovery 2s
    RECOVERY      = 3   # attesa 2s prima di tornare NORMAL


class DistractionDetector:
    LONG_THRESHOLD   = 5.0    # s: episodio singolo
    SHORT_ACCUMULATE = 10.0   # s: totale nella finestra
    SHORT_WINDOW     = 30.0   # s: ampiezza finestra scorrevole
    SHORT_RECOVERY   = 2.0    # s: sul strada per uscire dal warning short

    def __init__(self):
        self.state         = DistractionState.NORMAL
        self.away_since    = None   # inizio episodio corrente di distrazione
        self.on_road_since = None   # inizio periodo di ritorno (recovery)
        self.away_log      = []     # lista (start, end) episodi completati

    def _accumulated_away(self, now: float) -> float:
        window_start = now - self.SHORT_WINDOW
        self.away_log = [(s, e) for s, e in self.away_log if e > window_start]
        total = sum(min(e, now) - max(s, window_start) for s, e in self.away_log)
        if self.away_since is not None:
            total += now - max(self.away_since, window_start)
        return total

    def update(self, is_distracted: bool, now: float) -> DistractionState:
        if is_distracted:
            self.on_road_since = None
            if self.away_since is None:
                self.away_since = now
            away_dur    = now - self.away_since
            accumulated = self._accumulated_away(now)
            if away_dur >= self.LONG_THRESHOLD:
                self.state = DistractionState.WARNING_LONG
            elif accumulated >= self.SHORT_ACCUMULATE and self.state == DistractionState.NORMAL:
                self.state = DistractionState.WARNING_SHORT
        else:
            if self.away_since is not None:
                self.away_log.append((self.away_since, now))
                self.away_since = None
            if self.state == DistractionState.WARNING_LONG:
                self.state = DistractionState.NORMAL        # recovery immediata
            elif self.state in (DistractionState.WARNING_SHORT, DistractionState.RECOVERY):
                if self.on_road_since is None:
                    self.on_road_since = now
                self.state = DistractionState.RECOVERY
                if now - self.on_road_since >= self.SHORT_RECOVERY:
                    self.state = DistractionState.NORMAL
                    self.on_road_since = None
        return self.state


def owl_yaw(face_landmarks) -> float:
    """Deviazione del naso dal centro [0..0.5]. 0 = centrato, 0.5 = profilo."""
    nose   = face_landmarks[4].x
    l_corn = face_landmarks[33].x
    r_corn = face_landmarks[263].x
    total_w = abs(r_corn - l_corn)
    if total_w < 1e-6:
        return 0.0
    return abs((nose - l_corn) / total_w - 0.5)


def lizard_gaze(face_landmarks) -> float:
    """Offset medio dell'iride rispetto al centro dell'occhio [0..0.5]. 0 = centrato."""
    def offset(outer_idx, inner_idx, iris_idx):
        outer = face_landmarks[outer_idx].x
        inner = face_landmarks[inner_idx].x
        iris  = face_landmarks[iris_idx].x
        w = abs(inner - outer)
        if w < 1e-6:
            return 0.0
        center = (outer + inner) / 2
        return abs(iris - center) / w   # simmetrico, indipendente dall'ordine outer/inner

    return (offset(33, 133, 468) + offset(362, 263, 473)) / 2


def is_owl_distracted(face_landmarks, yaw_threshold=0.25) -> bool:
    return owl_yaw(face_landmarks) > yaw_threshold


def is_lizard_distracted(face_landmarks, iris_threshold=0.15) -> bool:
    return lizard_gaze(face_landmarks) > iris_threshold


# Colori BGR
_GREEN  = (0, 200, 0)
_ORANGE = (0, 165, 255)
_RED    = (0, 0, 255)

def driver_state_label(ms_state, ms_recovery_from, owl_state, lizard_state):
    """Restituisce (testo, colore_BGR) con priorità: Sleep > Microsleep > Distracted > Focused."""
    if ms_state == MicrosleepState.WARNING_7:
        return "Sleep", _RED
    if ms_state == MicrosleepState.RECOVERY and ms_recovery_from == MicrosleepState.WARNING_7:
        return "Sleep", _RED
    if ms_state in (MicrosleepState.WARNING_4, MicrosleepState.RECOVERY):
        return "Microsleep", _RED
    if owl_state == DistractionState.WARNING_LONG or lizard_state == DistractionState.WARNING_LONG:
        return "Distracted (long)", _RED
    if owl_state in (DistractionState.WARNING_SHORT, DistractionState.RECOVERY) or \
       lizard_state in (DistractionState.WARNING_SHORT, DistractionState.RECOVERY):
        return "Distracted (short)", _ORANGE
    return "Focused on the road", _GREEN


def draw_debug_timers(image, ms_det, owl_det, liz_det, now):
    """Pannello debug in alto a destra con i timer attivi dei tre detector."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
    WHITE  = (255, 255, 255)
    YELLOW = (0, 220, 220)
    GRAY   = (160, 160, 160)

    def fmt(label, val, target, active):
        color = YELLOW if active else GRAY
        text  = f"{label}: {val:.1f}s / {target}s" if active else f"{label}: --"
        return text, color

    rows = []
    rows.append(("[ MICROSLEEP ]", WHITE))
    active = ms_det.closed_since is not None
    rows.append(fmt("closed", (now - ms_det.closed_since) if active else 0, "4/7", active))
    active = ms_det.opened_since is not None
    rows.append(fmt("recov ", (now - ms_det.opened_since) if active else 0, 2, active))

    rows.append(("[ OWL ]", WHITE))
    active = owl_det.away_since is not None
    rows.append(fmt("episode", (now - owl_det.away_since) if active else 0, 5, active))
    acc = owl_det._accumulated_away(now)
    rows.append((f"accum : {acc:.1f}s / 10s", YELLOW if acc > 0 else GRAY))
    active = owl_det.on_road_since is not None
    rows.append(fmt("recov ", (now - owl_det.on_road_since) if active else 0, 2, active))

    rows.append(("[ LIZARD ]", WHITE))
    active = liz_det.away_since is not None
    rows.append(fmt("episode", (now - liz_det.away_since) if active else 0, 5, active))
    acc_l = liz_det._accumulated_away(now)
    rows.append((f"accum : {acc_l:.1f}s / 10s", YELLOW if acc_l > 0 else GRAY))
    active = liz_det.on_road_since is not None
    rows.append(fmt("recov ", (now - liz_det.on_road_since) if active else 0, 2, active))

    line_h = 18
    pad    = 7
    panel_w = 220
    panel_h = len(rows) * line_h + pad * 2

    h, w = image.shape[:2]
    x1 = w - panel_w - pad
    y1 = pad
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for i, (text, color) in enumerate(rows):
        y = y1 + pad + (i + 1) * line_h
        cv2.putText(image, text, (x1 + pad, y), font, scale, color, thick)


def draw_status(image, label, color):
    """Disegna il label di stato in basso a destra con sfondo semitrasparente."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    h, w = image.shape[:2]
    pad = 8
    x1, y1 = w - tw - pad * 2, h - th - baseline - pad * 2
    x2, y2 = w, h
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
    cv2.putText(image, label, (x1 + pad, y2 - baseline - pad), font, scale, color, thickness)


class EyeClosureDetector:
    def __init__(self, perclos_threshold=0.60, history_size=5):
        self.baseline_aperture = None
        self.perclos_threshold = perclos_threshold  # PERCLOS-80
        self.history = []
        self.history_size = history_size

    def calibrate(self, apertures_awake):
        """Chiamato durante i primi N secondi quando il driver è attento."""
        # Mediana invece di media: robusta a blink occasionali e outlier
        self.baseline_aperture = np.median(apertures_awake)

    def is_closed(self, aperture):
        if aperture is None:
            return True  # apertura non misurabile = occhio probabilmente chiuso
        if self.baseline_aperture is None:
            return False  # ancora in calibrazione

        # Percentuale di chiusura rispetto al baseline
        closure_pct = 1.0 - (aperture / self.baseline_aperture)
        closure_pct = np.clip(closure_pct, 0.0, 1.0)

        # Smoothing temporale: media mobile per ridurre il jitter dei landmark
        self.history.append(closure_pct)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        smoothed = np.mean(self.history)

        return smoothed >= self.perclos_threshold

def _line_polyline_intersection(p0, dir_vec, polyline):
    """
    Trova l'intersezione della retta {p0 + t*dir_vec} con una polilinea.
    Ritorna il punto di intersezione, o None se non c'è.
    """
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        # Risolve: p0 + t*dir = a + s*(b-a),  0 <= s <= 1
        seg = b - a
        M = np.array([[dir_vec[0], -seg[0]],
                      [dir_vec[1], -seg[1]]])
        rhs = a - p0
        if abs(np.linalg.det(M)) < 1e-9:
            continue  # paralleli
        t, s = np.linalg.solve(M, rhs)
        if 0.0 <= s <= 1.0:
            return p0 + t * dir_vec
    return None


def eyelid_aperture(landmarks, outer_idx, inner_idx, upper_lid_idx, lower_lid_idx,
                    img_w, img_h):
    """
    Apertura palpebrale come da definizione del driver monitoring:
    distanza fra palpebra superiore e inferiore lungo la perpendicolare
    al segmento angolo-angolo, passante per il suo midpoint.
    """
    to_px = lambda i: np.array([landmarks[i].x * img_w, landmarks[i].y * img_h])

    p_outer = to_px(outer_idx)
    p_inner = to_px(inner_idx)
    upper = np.array([to_px(i) for i in upper_lid_idx])
    lower = np.array([to_px(i) for i in lower_lid_idx])

    # Midpoint e direzione perpendicolare al segmento angolo-angolo
    midpoint = 0.5 * (p_outer + p_inner)
    d = p_inner - p_outer
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])  # normale 2D ruotando d di 90°

    p_up = _line_polyline_intersection(midpoint, n, upper)
    p_low = _line_polyline_intersection(midpoint, n, lower)

    if p_up is None or p_low is None:
        return None  # caso degenerato: occhio molto chiuso, fallback

    return np.linalg.norm(p_up - p_low)


CALIBRATION_SECONDS = 3  # secondi iniziali usati per misurare la baseline

def build_mask(img_shape, landmarks, roi_indices):
    """Maschera binaria del poligono ROI definito dai landmark."""
    h, w = img_shape[:2]
    pts = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in roi_indices],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

# --- livello 1: chiamato ad ogni frame -------------------------------
def extract_roi_rgb(frame_rgb, landmarks, roi_indices, img_shape):
    """Media RGB dei pixel dentro la ROI poligonale dei landmark."""
    mask = build_mask(img_shape, landmarks, roi_indices)  # cv2.fillPoly
    pixels = frame_rgb[mask > 0]
    if len(pixels) < 100:   # ROI troppo piccola / occlusione
        return None
    return pixels.mean(axis=0)   # shape (3,)

# --- livello 2: chiamato su una finestra di ~10 s --------------------
def estimate_bpm(rgb_window, fps, hr_band=(0.8, 3.0)):
    """rgb_window: array (N, 3) di medie RGB consecutive."""
    # detrend + bandpass + zscore per canale
    sig = detrend(rgb_window, axis=0)
    lo, hi = np.array(hr_band) / (fps / 2)
    b, a = butter(4, [lo, hi], btype='band')
    sig = filtfilt(b, a, sig, axis=0)
    sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-8)

    # ICA su 3 canali
    S = FastICA(n_components=3, max_iter=500, random_state=0,
            whiten='unit-variance').fit_transform(sig)
    # per ogni componente: Welch PSD e picco nella banda HR
    best_bpm, best_power = None, -np.inf
    for k in range(3):
        f, P = welch(S[:, k], fs=fps, nperseg=min(1024, len(S)))
        band = (f >= hr_band[0]) & (f <= hr_band[1])
        idx = np.argmax(P[band])
        if P[band][idx] > best_power:
            best_power = P[band][idx]
            best_bpm = f[band][idx] * 60
    return best_bpm

def main():
    # ==================== INIZIALIZZAZIONE WEBCAM ====================
    cap = cv2.VideoCapture(0)
    capture_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rgb_buffer  = deque(maxlen=int(BPM_WINDOW_S * capture_fps))
    frame_count = 0
    last_bpm    = None

    startup_time = time.time()

    detector = EyeClosureDetector()
    microsleep = MicrosleepDetector()
    owl_detector    = DistractionDetector()
    lizard_detector = DistractionDetector()
    calibration_samples  = []
    owl_yaw_samples      = []
    lizard_gaze_samples  = []
    owl_threshold        = 0.25   # default, sovrascritto dopo calibrazione
    lizard_threshold     = 0.15   # default, sovrascritto dopo calibrazione

    debug_mode = "--debug" in sys.argv
    if debug_mode:
        print("Debug mode enabled")

    video_writer = None
    _frame_times: list[float] = []   # per misurare FPS reale prima di aprire il writer
    _WARMUP = 30                     # frame di warmup per stima FPS

    # ==================== LOOP PRINCIPALE ====================
    while cap.isOpened():

        success, image = cap.read()  # Legge un frame dalla webcam

        start = time.time()  # Avvia cronometro per FPS

        if image is None:
            break  # Esce se non c'è un frame

        # Prepara l'immagine per il rilevamento
        image = cv2.flip(image, 1)                          # Capovolgimento orizzontale (specchio)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converte da BGR a RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)  # Converte in formato MediaPipe

        # Esegue il rilevamento dei punti di riferimento facciali
        timestamp_ms = int(time.time() * 1000)  # Timestamp in millisecondi
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)  # Rileva punti di riferimento

        img_h, img_w, _ = image.shape

        _frame_times.append(time.time())
        if video_writer is None and len(_frame_times) >= _WARMUP:
            real_fps = (_WARMUP - 1) / (_frame_times[-1] - _frame_times[0])
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter("dms_output.avi", fourcc, real_fps, (img_w, img_h))

        # ==================== DISEGNO PUNTI DI RIFERIMENTO ====================
        if results.face_landmarks:  # Se è stato rilevato un volto
            for face_landmarks in results.face_landmarks:  # Per ogni volto rilevato
                for idx, lm in enumerate(face_landmarks):  # Per ogni punto di riferimento
                    # Converte coordinate normalizzate (0-1) in pixel
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Disegna cerchi rossi per i punti degli occhi
                    if idx in LEFT_EYE or idx in RIGHT_EYE:
                        cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

                    # Disegna cerchi verdi per le iridi
                    if idx in LEFT_IRIS or idx in RIGHT_IRIS:
                        cv2.circle(image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

                    # Disegna cerchi blu per la punta del naso
                    if idx in NOSE_TIP:
                        cv2.circle(image, (x, y), radius=2, color=(255, 0, 0), thickness=-1)
                
                # Calcola apertura palpebrale (media occhio sinistro e destro)
                r_ap = eyelid_aperture(face_landmarks, RIGHT_OUTER, RIGHT_INNER,
                                       RIGHT_UPPER_LID, RIGHT_LOWER_LID, img_w, img_h)
                l_ap = eyelid_aperture(face_landmarks, LEFT_OUTER, LEFT_INNER,
                                       LEFT_UPPER_LID, LEFT_LOWER_LID, img_w, img_h)
                aperture = np.mean([a for a in [r_ap, l_ap] if a is not None]) if any(
                    a is not None for a in [r_ap, l_ap]) else None

                # Fase di calibrazione: raccoglie campioni nei primi N secondi
                elapsed = time.time() - startup_time
                calibrate_eye_detector(detector, calibration_samples, aperture, elapsed)

                if elapsed < CALIBRATION_SECONDS:
                    owl_yaw_samples.append(owl_yaw(face_landmarks))
                    lizard_gaze_samples.append(lizard_gaze(face_landmarks))
                elif owl_yaw_samples:
                    # mediana + 3σ: robusto a movimenti occasionali durante calibrazione
                    owl_threshold    = float(np.median(owl_yaw_samples)    + 3 * np.std(owl_yaw_samples))
                    lizard_threshold = float(np.median(lizard_gaze_samples) + 3 * np.std(lizard_gaze_samples))
                    owl_yaw_samples.clear()
                    lizard_gaze_samples.clear()
                    if debug_mode:
                        print(f"Calibrazione gaze: owl_thr={owl_threshold:.3f}  lizard_thr={lizard_threshold:.3f}")

                eyes_closed = bool(detector.is_closed(aperture))
                ms_state = microsleep.update(eyes_closed, time.time())

                now = time.time()
                owl_dist    = is_owl_distracted(face_landmarks, owl_threshold)
                # Lizard solo se la testa è dritta (altrimenti è Owl)
                lizard_dist = not owl_dist and is_lizard_distracted(face_landmarks, lizard_threshold)

                owl_state    = owl_detector.update(owl_dist, now)
                lizard_state = lizard_detector.update(lizard_dist, now)

                # RPPG: estrazione media RGB da ROI fronte e stima BPM ogni N frame
                rgb_mean = extract_roi_rgb(image_rgb, face_landmarks, FOREHEAD_ROI, (img_h, img_w))
                if rgb_mean is not None:
                    rgb_buffer.append(rgb_mean)

                frame_count += 1
                if len(rgb_buffer) == rgb_buffer.maxlen and frame_count % BPM_UPDATE_EVERY_N == 0:
                    last_bpm = estimate_bpm(np.array(rgb_buffer), capture_fps)

                label, color = driver_state_label(ms_state, microsleep._recovery_from, owl_state, lizard_state)
                draw_status(image, label, color)
                if debug_mode:
                    if aperture is not None and detector.baseline_aperture is not None:
                        closure_pct = 1.0 - (aperture / detector.baseline_aperture)
                        print(f"aperture={aperture:.1f}px  baseline={detector.baseline_aperture:.1f}px  closure={closure_pct:.2f}")
                    if detector.baseline_aperture is None:
                        cv2.putText(image, f'Calibrating... {elapsed:.1f}s', (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    draw_debug_timers(image, microsleep, owl_detector, lizard_detector, now)
                
                if last_bpm is not None:
                    _bpm_label = f'BPM: {last_bpm:.1f}'
                    _bpm_font, _bpm_scale, _bpm_thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                    _bpm_color = (0, 255, 255)
                else:
                    progress = 100.0 * len(rgb_buffer) / rgb_buffer.maxlen
                    _bpm_label = f'Calibrating HR... {progress:.0f}%'
                    _bpm_font, _bpm_scale, _bpm_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    _bpm_color = (200, 200, 0)
                (bpm_tw, bpm_th), bpm_bl = cv2.getTextSize(_bpm_label, _bpm_font, _bpm_scale, _bpm_thick)
                bpm_pad = 8
                bpm_x1, bpm_y1 = 0, img_h - bpm_th - bpm_bl - bpm_pad * 2
                bpm_x2, bpm_y2 = bpm_tw + bpm_pad * 2, img_h
                bpm_overlay = image.copy()
                cv2.rectangle(bpm_overlay, (bpm_x1, bpm_y1), (bpm_x2, bpm_y2), (80, 80, 80), -1)
                cv2.addWeighted(bpm_overlay, 0.5, image, 0.5, 0, image)
                cv2.putText(image, _bpm_label, (bpm_x1 + bpm_pad, bpm_y2 - bpm_bl - bpm_pad),
                            _bpm_font, _bpm_scale, _bpm_color, _bpm_thick)


            end = time.time()
            totalTime = end - start  # Calcola tempo di elaborazione
            #show FPS on the image
            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


            #show total run time on the image
            cv2.putText(image, f'Time: {(time.time()-startup_time):.2f}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('output window', image)

        if video_writer is not None:
            video_writer.write(image)

        # Premi ESC (tasto 27) per uscire dal programma
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # ==================== PULIZIA RISORSE ====================
    cap.release()
    if video_writer is not None:
        video_writer.release()
    face_landmarker.close()

def calibrate_eye_detector(detector, calibration_samples, aperture, elapsed):
    if elapsed < CALIBRATION_SECONDS:
        if aperture is not None:
            calibration_samples.append(aperture)
    elif detector.baseline_aperture is None and calibration_samples:
        detector.calibrate(calibration_samples)
        print(f"Calibrazione completata: baseline={detector.baseline_aperture:.1f}px")         # Libera il rilevatore di punti di riferimento


if __name__ == "__main__":
    main()

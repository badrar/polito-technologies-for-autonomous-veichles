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


# def _ear(landmarks, indices, img_w, img_h):
#     pts = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h]) for i in indices]
#     vertical = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
#     horizontal = np.linalg.norm(pts[0] - pts[3])
#     return vertical / (2.0 * horizontal)


# def are_eyes_closed(face_landmarks, img_w, img_h, threshold=EAR_THRESHOLD):
#     right = _ear(face_landmarks, RIGHT_EYE_EAR, img_w, img_h)
#     left  = _ear(face_landmarks, LEFT_EYE_EAR,  img_w, img_h)
#     return right < threshold and left < threshold

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
        self.state        = MicrosleepState.NORMAL
        self.closed_since = None   # timestamp inizio chiusura corrente
        self.opened_since = None   # timestamp inizio apertura (recovery)

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
                    self.opened_since = now
                if now - self.opened_since >= self.RECOVERY_2S:
                    self.state = MicrosleepState.NORMAL
                    self.opened_since = None
                else:
                    self.state = MicrosleepState.RECOVERY
        return self.state


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


def main():
    # ==================== INIZIALIZZAZIONE WEBCAM ====================
    cap = cv2.VideoCapture(0)
    startup_time = time.time()

    detector = EyeClosureDetector()
    microsleep = MicrosleepDetector()
    calibration_samples = []

    # check args for debug mode
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        print("Debug mode enabled")
        debug_mode = True
    else:
        debug_mode = False

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

        img_h, img_w, _ = image.shape  # Dimensioni dell'immagine

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

                eyes_closed = bool(detector.is_closed(aperture))
                ms_state = microsleep.update(eyes_closed, time.time())

                if eyes_closed:
                    cv2.putText(image, 'Eyes closed', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(image, 'Eyes open', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if ms_state == MicrosleepState.WARNING_7:
                    cv2.putText(image, '! MICROSLEEP (7s) !', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                elif ms_state == MicrosleepState.WARNING_4:
                    cv2.putText(image, 'MICROSLEEP (4s)', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                elif ms_state == MicrosleepState.RECOVERY:
                    cv2.putText(image, 'Recovery...', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if debug_mode:
                    if aperture is not None and detector.baseline_aperture is not None:
                        closure_pct = 1.0 - (aperture / detector.baseline_aperture)
                        print(f"aperture={aperture:.1f}px  baseline={detector.baseline_aperture:.1f}px  closure={closure_pct:.2f}")
                    if detector.baseline_aperture is None:
                        cv2.putText(image, f'Calibrating... {elapsed:.1f}s', (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


            end = time.time()
            totalTime = end - start  # Calcola tempo di elaborazione
            #show FPS on the image
            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


            #show total run time on the image
            cv2.putText(image, f'Time: {(time.time()-startup_time):.2f}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('output window', image)  # Mostra il frame con i punti disegnati

        # Premi ESC (tasto 27) per uscire dal programma
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # ==================== PULIZIA RISORSE ====================
    cap.release()                    # Chiude la webcam
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

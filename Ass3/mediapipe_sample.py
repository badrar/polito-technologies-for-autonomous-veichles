# ==================== LIBRERIE ====================
import cv2                                  # OpenCV per processare immagini/video
import mediapipe as mp                      # MediaPipe per rilevamento facce
import numpy as np                          # NumPy per operazioni numeriche
import time                                 # Modulo per misurare il tempo
import statistics as st                     # Statistiche (non utilizzato)
import os                                   # Modulo per percorsi file
import urllib.request                       # Per scaricare il modello

from mediapipe.tasks import python
from mediapipe.tasks.python import vision   # Task di visione di MediaPipe

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

def main():
    # ==================== INIZIALIZZAZIONE WEBCAM ====================
    cap = cv2.VideoCapture(0)

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

            end = time.time()
            totalTime = end - start  # Calcola tempo di elaborazione

            cv2.imshow('output window', image)  # Mostra il frame con i punti disegnati

        # Premi ESC (tasto 27) per uscire dal programma
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # ==================== PULIZIA RISORSE ====================
    cap.release()                    # Chiude la webcam
    face_landmarker.close()          # Libera il rilevatore di punti di riferimento


if __name__ == "__main__":
    main()

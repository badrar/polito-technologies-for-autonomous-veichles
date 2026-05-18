import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# --- Indici FaceMesh (con refine_landmarks=True) ---
# Occhio sinistro (dal punto di vista del soggetto)
L_OUTER, L_INNER = 33, 133
L_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133]
L_LOWER = [33,   7, 163, 144, 145, 153, 154, 155, 133]
# Occhio destro
R_OUTER, R_INNER = 263, 362
R_UPPER = [263, 466, 388, 387, 386, 385, 384, 398, 362]
R_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]

# ... (incolla qui eyelid_aperture e _line_polyline_intersection del turno precedente) ...


def both_eyes_aperture(lms, w, h):
    """Media dell'apertura dei due occhi, in pixel. None se entrambi falliscono."""
    left  = eyelid_aperture(lms, L_OUTER, L_INNER, L_UPPER, L_LOWER, w, h)
    right = eyelid_aperture(lms, R_OUTER, R_INNER, R_UPPER, R_LOWER, w, h)
    vals = [v for v in (left, right) if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    mp_fm = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,        # FONDAMENTALE per gli occhi
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window_seconds = 60
    perclos_buffer = deque(maxlen=int(fps * window_seconds))

    detector = EyeClosureDetector(perclos_threshold=0.80, history_size=5)

    # --- Fase 1: Calibrazione (5 secondi a occhi aperti) ---
    calib_apertures = []
    t0 = time.time()
    print("Calibrazione: guarda la camera con occhi aperti per 5s...")
    while time.time() - t0 < 5.0:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)  # selfie view
        h, w = frame.shape[:2]
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            a = both_eyes_aperture(res.multi_face_landmarks[0].landmark, w, h)
            if a is not None:
                calib_apertures.append(a)
        cv2.putText(frame, "CALIBRATING...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("eye", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return

    if len(calib_apertures) < 10:
        print("Calibrazione fallita: troppo pochi campioni.")
        return
    detector.calibrate(calib_apertures)
    print(f"Baseline: {detector.baseline_aperture:.1f}px")

    # --- Fase 2: Loop di rilevamento ---
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        closed = False
        aperture = None
        if res.multi_face_landmarks:
            aperture = both_eyes_aperture(res.multi_face_landmarks[0].landmark, w, h)
            closed = detector.is_closed(aperture)
        else:
            # nessun volto: NON pushare nulla, evita di inquinare PERCLOS
            pass

        if res.multi_face_landmarks:
            perclos_buffer.append(1 if closed else 0)

        perclos = np.mean(perclos_buffer) if perclos_buffer else 0.0
        drowsy = perclos > 0.15  # >15% del tempo a occhi chiusi in 60s = soglia NHTSA

        color = (0, 0, 255) if drowsy else (0, 255, 0)
        cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"eye: {'CLOSED' if closed else 'OPEN'}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if drowsy:
            cv2.putText(frame, "DROWSY!", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("eye", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
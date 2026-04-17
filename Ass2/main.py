import os
import cv2
import numpy as np
from scipy.signal import find_peaks
from lane_detection import calculate_threshold
from ultralytics import YOLO

from collections import deque

DEBUG = True

# frame buffer for temporal consistency in lane seed tracking
# this is implemented to avoid high variance in high noise frames, 
#   by keeping a history of recent seeds and checking if the current
#   seed is consistent with the recent history. If a seed deviates too 
#   much from the average of the history, it can be considered a false
#   positive and ignored. Additionally, if no seeds are detected for 
#   several consecutive frames, we can declare that no lanes are 
#   detected, which can help to avoid false positives in frames where 
#   the lane markings are not visible.

HISTORY_LEN = 6          # quanti frame tenere
MAX_DEVIATION = 5       # pixel: se il seed attuale dista più di così dalla media, è sospetto
MAX_LOST_FRAMES = 4     # dopo tot frame senza detection → "no lanes detected"
MAX_SLOPE_DEVIATION = 0.10  # per la coerenza della pendenza del fit (es. 0.15 per strade quasi dritte)

left_history  = deque(maxlen=HISTORY_LEN)
right_history = deque(maxlen=HISTORY_LEN)
left_lost  = 0
right_lost = 0


# Intrinsic camera parameters 
FOCAL_LENGTH = (1970.0, 1970.0)        # fx, fy in pixels
PRINCIPAL_POINT = (970.0, 483.0)       # cx, cy in pixels
CAMERA_HEIGHT = 1.66                   # meters above ground
CAMERA_X_OFFSET = 1.875               # meters forward offset
PITCH = 0.0                           # degrees

# BEV configuration TEST PARAMS
BEV_WIDTH = 600                        # pixels
BEV_HEIGHT = 600                       # pixels
X_MIN = 6                              # nearest ground distance (meters)
X_MAX = 30.0                           # farthest ground distance (meters)
Y_MIN = -8.0                           # right boundary (meters)
Y_MAX = 8.0                            # left boundary (meters)

# lane column thresholding
PEAK_MIN_DISTANCE = 30  # minimum pixel distance between peaks for NMS

# ── Pipeline tunables ──────────────────────────────────────────────────────
# Preprocessing
BLUR_KSIZE = (3, 3)

# GOLD feature extraction
GOLD_M = 10
GOLD_LOW_THRESHOLD = 30
GOLD_SIDE_MARGIN_RATIO = 0.3     # fraction of width zeroed on each side

# Adaptive max thresholding
ADAPTIVE_NEIGHBORHOOD = 8
ADAPTIVE_K = 2
ADAPTIVE_NOISE_FLOOR = 40

# Morphological cleanup
GEODESIC_ITERATIONS = 8
VERTICAL_OPEN_KERNEL = (1, 15)

# Crosswalk removal
CROSSWALK_MARGIN_RATIO = 0.2
CROSSWALK_DENSITY_THRESHOLD = 0.10

# Connected-component filtering (two passes)
CC_FIRST_MIN_AREA = 20
CC_SECOND_MIN_AREA = 50
CC_MIN_ASPECT = 2.0              # height/width ≥ this → vertical blob

# Lane seed search
LANE_MARGIN_RATIO = 0.15
LANE_PEAK_HEIGHT_RATIO = 0.10    # peaks below this fraction of max are dropped
LANE_MAX_MARKING_WIDTH = 15
LANE_N = 2

# Lane polynomial fit
LANE_STRIP_HALF_WIDTH = 25
LANE_FIT_DEG = 1
LANE_MIN_POINTS = 10

# Lane solid/dashed classification
CLASSIFY_BAND = 10
CLASSIFY_GAP_THRESHOLD = 0.5

# Lane drawing
LANE_N_DASHES = 6
LANE_DASH_DUTY = 0.6
LANE_THICKNESS_BEV = 3
LANE_THICKNESS_FRAME = 4
LEFT_LANE_COLOR = (0, 255, 255)
RIGHT_LANE_COLOR = (255, 100, 0)

# Display
WAIT_KEY_MS = 300

# YOLO object detection
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.35
# COCO classes relevant to driving scenes
YOLO_KEEP_CLASSES = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck
YOLO_BOX_COLOR = (0, 255, 0)
YOLO_TEXT_COLOR = (0, 255, 255)

yolo_model = YOLO(YOLO_MODEL_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# IPM - Inverse Perspective Mapping
# ═══════════════════════════════════════════════════════════════════════════════
 
def compute_homography_bev_to_img():
    """
    Compute the 3x3 homography that maps BEV pixel coordinates
    to original image coordinates.
 
    The derivation:
    1. Camera intrinsic matrix K
    2. Rotation R from world frame to camera frame
       World: X-forward, Y-left, Z-up
       Camera: x-right, y-down, z-forward
    3. Ground plane Z=0 → homography H = K @ [r1 | r2 | t]
    4. BEV pixel-to-world affine transform T
    5. Final: H_bev_to_img = H_ground_to_img @ T_bev_to_world
    """
    fx, fy = FOCAL_LENGTH
    cx, cy = PRINCIPAL_POINT
    h = CAMERA_HEIGHT
    x_off = CAMERA_X_OFFSET
    theta = np.radians(PITCH)
 
    # Intrinsic matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
 
    # Rotation: world → camera
    # Base rotation (no pitch): maps (X-fwd, Y-left, Z-up) → (x-right, y-down, z-fwd)
    R_base = np.array([[0, -1,  0],
                       [0,  0, -1],
                       [1,  0,  0]], dtype=np.float64)
 
    # Pitch rotation around camera x-axis
    R_pitch = np.array([[1, 0,            0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta),  np.cos(theta)]], dtype=np.float64)
 
    R = R_pitch @ R_base
 
    # Translation: t = -R @ C, where C is camera position in world
    C_world = np.array([x_off, 0.0, h])
    t = -R @ C_world
 
    # Homography from ground plane (Z=0) to image: [X, Y, 1] → [u, v, 1]
    # H = K @ [r1 | r2 | t]  (columns 0,1 of R and translation)
    H_ground_to_img = K @ np.column_stack([R[:, 0], R[:, 1], t])
 
    # Affine transform: BEV pixel → world ground coordinates
    # u_bev ∈ [0, W) → Y ∈ [Y_MAX, Y_MIN]  (left-to-right in BEV = positive-to-negative Y)
    # v_bev ∈ [0, H) → X ∈ [X_MAX, X_MIN]  (top-to-bottom in BEV = far-to-near)
    dx = X_MAX - X_MIN
    dy = Y_MAX - Y_MIN
 
    T_bev_to_world = np.array([
        [0,           -dx / BEV_HEIGHT, X_MAX],   # X = X_MAX - dx/H * v
        [-dy / BEV_WIDTH, 0,            Y_MAX],   # Y = Y_MAX - dy/W * u
        [0,            0,               1    ]
    ], dtype=np.float64)
 
    H_bev_to_img = H_ground_to_img @ T_bev_to_world
    return H_bev_to_img
 
 
def compute_image_to_ground():
    """
    Direct 3x3 homography mapping image pixels to world ground-plane (Z=0)
    coordinates (X forward, Y left), expressed in the same world frame used
    by compute_homography_bev_to_img (camera placed at (CAMERA_X_OFFSET, 0,
    CAMERA_HEIGHT)). Derived from the pinhole model:
        [u, v, 1]^T ∝ K · [r1 | r2 | t] · [X, Y, 1]^T
    so the inverse of [K · [r1|r2|t]] maps pixels to ground points.
    """
    fx, fy = FOCAL_LENGTH
    cx, cy = PRINCIPAL_POINT
    theta = np.radians(PITCH)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    R_base = np.array([[0, -1,  0],
                       [0,  0, -1],
                       [1,  0,  0]], dtype=np.float64)
    R_pitch = np.array([[1, 0,            0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta),  np.cos(theta)]], dtype=np.float64)
    R = R_pitch @ R_base

    C_world = np.array([CAMERA_X_OFFSET, 0.0, CAMERA_HEIGHT])
    t = -R @ C_world

    H_ground_to_img = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return np.linalg.inv(H_ground_to_img)


def image_point_to_ground(u, v, H_img_to_ground):
    """
    Back-project an image pixel to the ground plane. Returns forward distance
    from the camera (X_rel, meters), lateral offset (Y_rel, meters, left +),
    and Euclidean range. Returns None if the pixel lies on/above the horizon.
    """
    p = H_img_to_ground @ np.array([u, v, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-9:
        return None
    X_world = p[0] / p[2]
    Y_world = p[1] / p[2]
    X_rel = X_world - CAMERA_X_OFFSET  # distance forward from camera
    if X_rel <= 0:
        return None  # point behind the camera / above horizon
    distance = float(np.hypot(X_rel, Y_world))
    return X_rel, Y_world, distance


def compute_bev(image, H_bev_to_img):
    """Warp the perspective image into a bird's eye view using IPM."""
    bev = cv2.warpPerspective(
        image, H_bev_to_img, (BEV_WIDTH, BEV_HEIGHT),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    )
    return bev

def adaptive_max_threshold(e, c=7, k=2):
    """
    Implementa la binarizzazione adattiva basata sul massimo locale (Equazione 9),
    con l'aggiunta vitale di un noise_floor per evitare il rumore sull'asfalto.
    
    e: immagine enhanced in ingresso (uint8)
    c: dimensione del vicinato (c x c)
    k: costante di divisione (dal paper k=2)
    """
    # Assicuriamoci che l'immagine sia in float per la divisione
    e_float = e.astype(np.float32)
    
    # 1. Calcoliamo m(x,y): il valore massimo in un intorno c x c
    kernel = np.ones((c, c), np.uint8)
    m = cv2.dilate(e_float, kernel)
    
    # 2. Calcoliamo la soglia locale adattiva (Equazione 9)
    local_threshold = m / k
    
    # 3. FIX: Calcoliamo la soglia finale combinando local e noise_floor.
    # Usiamo il valore MASSIMO tra la soglia adattiva e il noise_floor.
    final_threshold = np.maximum(local_threshold, ADAPTIVE_NOISE_FLOOR)
    
    # 4. Applichiamo la condizione finale
    t = (e_float >= final_threshold).astype(np.uint8) * 255
    
    return t

def gold_feature_extraction(pct_mask, m=8, low_threshold=30):
    """
    GOLD paper feature extraction (Eq. 5-6).
    For each pixel b(x,y), computes the sum of horizontal gradients toward
    its neighbours at distance m. A pixel is a lane candidate only if both
    gradients are positive (i.e. it is brighter than both sides).

    Args:
        pct_mask:      binary/grayscale BEV mask (uint8 H×W)
        m:             half-width of the parallel filter (default 4)
        low_threshold: responses below this value are zeroed out (default 30)

    Returns:
        uint8 response map, same shape as pct_mask
    """
    r = np.zeros_like(pct_mask, dtype=np.float32)
    pct_mask_float = pct_mask.astype(np.float32)

    # Eq. 6 — shifted views: centre b(x,y), left b(x, y-m), right b(x, y+m)
    centro   = pct_mask_float[:, m:-m]
    sinistra = pct_mask_float[:, :-2*m]
    destra   = pct_mask_float[:, 2*m:]

    d_plus_m  = centro - destra    # gradient toward right neighbour
    d_minus_m = centro - sinistra  # gradient toward left neighbour

    # Eq. 5 — keep only pixels brighter than both neighbours
    condizione = (d_plus_m > 0) & (d_minus_m > 0)
    r[:, m:-m][condizione] = (d_plus_m + d_minus_m)[condizione]

    # Clip to uint8 and suppress weak responses
    r = np.clip(r, 0, 255).astype(np.uint8)
    r[r < low_threshold] = 0
    return r


def geodesic_dilation(r, num_iterations=3):
    """
    Migliora l'immagine filtrata applicando la dilatazione morfologica geodetica.
    
    Parametri:
    r: numpy array (l'immagine in uscita dal passaggio precedente)
    num_iterations: quante volte ripetere il processo ("a few iterations")
    """
    # 1. Immagine di controllo c(x,y) - Equazione (7)
    # Creiamo una maschera che vale 1 dove r non è 0, e 0 altrimenti.
    # Usiamo lo stesso tipo di dato di r per poter fare la moltiplicazione dopo.
    c = (r != 0).astype(r.dtype)
    
    # 2. Elemento Strutturante - La griglia 3x3
    # OpenCV ha una costante MORPH_CROSS che crea esattamente la griglia
    # con i punti cardinali descritta nella tua immagine.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # Creiamo una copia di r su cui lavorare
    enhanced_r = r.copy()
    
    # 3. Applichiamo le iterazioni
    for _ in range(num_iterations):
        # Passo A: Calcola il "valore massimo nell'intorno" descritto dal kernel
        # La funzione dilate fa esattamente questo.
        dilated = cv2.dilate(enhanced_r, kernel)
        
        # Passo B: Moltiplica per l'immagine di controllo
        # Questo assicura che le zone a 0 nell'immagine originale rimangano a 0
        enhanced_r = dilated * c
        
    return enhanced_r

def feature_identification(binary_bev, expected_width=None, width_tol=0.4):
    """
    GOLD Feature Identification su run contigui anziché pixel singoli.
    
    Per ogni riga trova i segmenti contigui di pixel bianchi,
    ne prende il centroide, e considera solo coppie di run ADIACENTI.
    """
    H, N = binary_bev.shape
    observations = []

    for i in range(H):
        # ── Trova i "run" contigui nella riga ──
        row = binary_bev[i]
        runs = []
        in_run = False
        for x in range(N):
            if row[x] > 0 and not in_run:
                start = x
                in_run = True
            elif row[x] == 0 and in_run:
                runs.append((start, x - 1))
                in_run = False
        if in_run:
            runs.append((start, N - 1))

        if len(runs) < 2:
            continue

        # ── Considera solo coppie ADIACENTI di run ──
        for j in range(len(runs) - 1):
            p = (runs[j][0] + runs[j][1]) / 2.0    # centroide run sinistro
            q = (runs[j+1][0] + runs[j+1][1]) / 2.0  # centroide run destro

            candidates = [
                ((p + q) / 2.0, (q - p) / 2.0, 'A'),
                (q,             q - p,          'B'),
                (p,             q - p,          'C'),
            ]

            for c, w, cfg in candidates:
                # Vincoli geometrici GOLD
                if not (0 <= c <= N and w < N / 3
                        and c - w <= 0.75 * N
                        and c + w >= N / 4):
                    continue
                # Filtro opzionale sulla larghezza attesa
                if expected_width is not None:
                    if abs(w - expected_width) > width_tol * expected_width:
                        continue
                observations.append({
                    'row': i, 'c': c, 'w': w, 'config': cfg
                })

    return observations

def find_lane_seeds(binary_bev, margin_ratio=0.15, min_peak_distance=40, n_lanes=2):
    """
    Column-projection histogram on the lower half of the binary BEV image.
    Peaks in the histogram give the x-positions (seeds) of each lane marking.
    NMS is applied through the `distance` parameter of find_peaks.
    """
    h, w   = binary_bev.shape
    margin = int(w * margin_ratio)

    lower = binary_bev[h // 2:, margin: w - margin].astype(np.float64)
    hist  = np.zeros(w, dtype=np.float64)
    hist[margin: w - margin] = lower.sum(axis=0)

    if hist.max() == 0:
        return [], hist

    peaks, props = find_peaks(hist, distance=min_peak_distance, height=hist.max() * 0.10)
    if len(peaks) == 0:
        return [], hist

    order = np.argsort(props['peak_heights'])[::-1]
    seeds = sorted(peaks[order[:n_lanes]].tolist())
    return seeds, hist

def find_lane_seeds_improved(binary_bev, margin_ratio=0.15, min_peak_distance=40,
                    n_lanes=2, max_marking_width=15):
    h, w = binary_bev.shape
    margin = int(w * margin_ratio)

    roi = binary_bev[h // 2:, margin:w - margin]

    # Per ogni colonna conta in quante RIGHE c'è almeno un pixel bianco,
    # ma solo se il run orizzontale in quella riga è stretto (≤ max_marking_width).
    # Questo elimina i picchi da oggetti larghi.
    row_count = np.zeros(roi.shape[1], dtype=np.float64)
    for y in range(roi.shape[0]):
        row = roi[y]
        in_run = False
        start = 0
        for x in range(len(row)):
            if row[x] > 0 and not in_run:
                start = x
                in_run = True
            elif (row[x] == 0 or x == len(row) - 1) and in_run:
                end = x if row[x] == 0 else x + 1
                if (end - start) <= max_marking_width:
                    row_count[start:end] += 1
                in_run = False

    hist = np.zeros(w, dtype=np.float64)
    hist[margin:w - margin] = row_count

    if hist.max() == 0:
        return [], hist

    peaks, props = find_peaks(hist, distance=min_peak_distance,
                              height=hist.max() * 0.10)
    if len(peaks) == 0:
        return [], hist

    order = np.argsort(props['peak_heights'])[::-1]
    seeds = sorted(peaks[order[:n_lanes]].tolist())
    return seeds, hist

def fit_lane_from_seed(binary_bev, seed, strip_half_width=30, deg=1, min_points=10):
    """
    Collect all white pixels inside a vertical strip centred on `seed` and
    fit the polynomial  x = poly(y)  via least squares (Eq. 4.A of GOLD).

    deg=1  -> straight road (motorway default)
    deg=2  -> curved road
    """
    h, w = binary_bev.shape
    x_lo = max(0, seed - strip_half_width)
    x_hi = min(w, seed + strip_half_width)

    ys_rel, xs_rel = np.nonzero(binary_bev[:, x_lo:x_hi])
    if len(ys_rel) < min_points:
        if DEBUG: print(f"Seed {seed}: not enough points ({len(ys_rel)}) for fitting.")
        return None

    # Quante righe distinte hanno pixel? Se troppo poche, è rumore
    row_coverage = len(np.unique(ys_rel)) / h
    if row_coverage < 0.15:  # meno del 15% delle righe ha pixel → probabilmente rumore
        if DEBUG: print(f"Seed {seed}: low row coverage ({row_coverage:.2f}), likely noise.")
        return None

    # Dopo il check row_coverage, prima del polyfit
    xs = (xs_rel + x_lo).astype(np.float32)
    ys = ys_rel.astype(np.float32)

    # Fit e poi controlla il residuo
    try:
        fit = np.polyfit(ys, xs, deg)
    except np.linalg.LinAlgError:
        if DEBUG: print(f"Seed {seed}: polyfit failed, likely degenerate points.")
        return None

    residuals = xs - np.polyval(fit, ys)
    rmse = np.sqrt(np.mean(residuals ** 2))
    if rmse > 3:  # pixel — una lane vera ha residuo bassissimo --> 6 andava bene per il dastaset borderline
        if DEBUG: print(f"Seed {seed}: high RMSE ({rmse:.2f}), likely bad fit.")
        return None

    return fit

def lane_segments(fit, bev_h, lane_type, n_dashes=LANE_N_DASHES, duty=LANE_DASH_DUTY):
    """Return a list of BEV endpoint pairs for a straight lane fit.

    solid  -> one segment spanning the full image height.
    dashed -> n_dashes segments, each covering `duty` fraction of its period.
    """
    if fit is None:
        return []
    H = bev_h - 1
    if lane_type == 'solid':
        ts = [(0.0, 1.0)]
    else:
        ts = [(k / n_dashes, (k + duty) / n_dashes) for k in range(n_dashes)]
    segs = []
    for t0, t1 in ts:
        y0, y1 = t0 * H, t1 * H
        x0 = float(np.polyval(fit, y0))
        x1 = float(np.polyval(fit, y1))
        segs.append(((x0, y0), (x1, y1)))
    return segs


def draw_lane_bev(img, fit, lane_type, color, thickness=3):
    """Draw a single lane (solid or dashed) on the BEV image."""
    for (p0, p1) in lane_segments(fit, img.shape[0], lane_type):
        cv2.line(img, (int(p0[0]), int(p0[1])),
                      (int(p1[0]), int(p1[1])), color, thickness)


def draw_lane_frame(img, fit, lane_type, H_bev_to_img, bev_h,
                    color, thickness=4):
    """Draw a single lane on the original frame by reprojecting BEV segments."""
    segs = lane_segments(fit, bev_h, lane_type)
    if not segs:
        return
    pts = np.float32([p for seg in segs for p in seg]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H_bev_to_img).reshape(-1, 2).astype(int)
    for k in range(0, len(proj), 2):
        cv2.line(img, tuple(proj[k]), tuple(proj[k + 1]), color, thickness)

def calcola_soglia_iterativa(immagine):
    """
    Calcola la soglia ottimale di un'immagine in scala di grigi
    utilizzando l'algoritmo iterativo descritto.
    
    Parametri:
    immagine (numpy.ndarray): L'immagine di input in scala di grigi (matrice 2D).
    
    Ritorna:
    float: Il valore della soglia finale calcolata.
    """
    # Assicuriamoci che l'immagine sia un array numpy (usiamo float per evitare overflow)
    img = immagine    
    # --- Step 1 ---
    # Calcolo di g_max e g_min nell'immagine originale
    g_max = np.max(img)
    g_min = np.min(img)
    
    # Inizializzazione della soglia (Th_0)
    th_corrente = (g_max + g_min) / 2.0
    
    iterazione = 0
    
    while True:
        # --- Step 2 ---
        # Divisione in due regioni:
        # Regione A: pixel >= Th_i
        # Regione B: pixel < Th_i
        regione_a = img[img >= th_corrente]
        regione_b = img[img < th_corrente]
        
        # Calcolo dei valori medi g_A e g_B
        # np.mean fa esattamente la somma dei valori divisa per il numero di elementi.
        # Aggiungiamo un controllo per evitare divisioni per zero nel caso una regione sia vuota.
        g_a = np.mean(regione_a) if len(regione_a) > 0 else 0
        g_b = np.mean(regione_b) if len(regione_b) > 0 else 0
        
        # --- Step 3 ---
        # Aggiornamento della soglia (Th_i+1)
        th_successivo = (g_a + g_b) / 2.0
        
        # --- Step 4 ---
        # Il processo si ripete finché Th_i+1 non è uguale a Th_i
        # Nota: usiamo una tolleranza minima (1e-5) al posto dell'uguaglianza stretta (==) 
        # perché stiamo lavorando con numeri in virgola mobile (float).
        if abs(th_successivo - th_corrente) < 1e-5:
            break
            
        th_corrente = th_successivo
        iterazione += 1
        
    return th_corrente

def create_lane_kernels(sigma_x, sigma_y, kernel_size):
    """Crea i kernel 1D g(x) e g(y) basati sulle formule della slide."""
    
    # Creiamo un array di coordinate centrate sullo zero (es. da -2 a +2)
    half_size = kernel_size // 2
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    
    # Calcoliamo g(x): Derivata seconda della gaussiana (orizzontale)
    # Questa formula esalta i picchi di luce stretti (la linea della corsia)
    g_x = (1 / (sigma_x**2)) * np.exp(-(x**2) / (2 * sigma_x**2)) * (1 - (x**2) / (sigma_x**2))
    
    # Calcoliamo g(y): Gaussiana standard (verticale)
    # Questa formula "spalma" il segnale verticalmente
    g_y = np.exp(-(y**2) / (2 * sigma_y**2))
    
    return g_x, g_y

def enhance_lanes(image):
    # 1. Imposta i parametri (da tarare in base alla risoluzione della telecamera)
    sigma_x = 2.0  # Legato alla larghezza in pixel della corsia
    sigma_y = 5.0  # Quanto vogliamo lisciare verticalmente
    k_size = 11    # Dimensione del kernel (deve essere dispari)
    
    # 2. Genera i kernel 1D
    g_x, g_y = create_lane_kernels(sigma_x, sigma_y, k_size)
    
    # 3. Applica il filtro separabile in modo super efficiente
    # cv2.CV_32F indica che vogliamo il risultato in float (i valori possono essere negativi)
    enhanced_img = cv2.sepFilter2D(image, cv2.CV_32F, g_x, g_y)
    
    return enhanced_img

def classify_lane(binary_bev, seed_x, band=CLASSIFY_BAND, gap_threshold=CLASSIFY_GAP_THRESHOLD):
    """
    Classify a lane as 'solid' or 'dashed' based on the presence 
    of pixels in a vertical band around the seed.
    """
    _, w = binary_bev.shape
    x_lo = max(0, seed_x - band)
    x_hi = min(w, seed_x + band)

    # Per ogni riga: c'è almeno un pixel acceso nella banda?
    presence = np.any(binary_bev[:, x_lo:x_hi] > 0, axis=1).astype(np.float64)

    coverage = presence.mean()

    return 'solid' if coverage > gap_threshold else 'dashed'

if __name__ == "__main__":
    video_id="044" #base test
    #video_id="040" #grande beccheggio
    #video_id="008" #continua e non
    #video_id="019" #leggera curva
    #video_id="029"
    #video_id="043"
    datset_path = f"./archive/{video_id}/camera/front_camera/"

    # images id are only jpg files from dataset_path
    images_id = [file for file in os.listdir(datset_path) if file.endswith(".jpg")]

    print(f"Found {len(images_id)} images in the dataset.")

    #vidcap alike to read images in sequence
    class ImageSequence:
        def __init__(self, path):
            self.path = path
            self.images_id = [file.replace(".jpg","") for file in os.listdir(path) if file.endswith(".jpg")]

            #since dtasae is based on sequential images, probably from a video, sorting can smooth out the showcase
            self.images_id.sort() 
            self.index = 0

        def read(self, loop=False):
            if self.index < len(self.images_id):
                img_path = os.path.join(self.path, f"{self.images_id[self.index]}.jpg")
                frame = cv2.imread(img_path)
                self.index += 1
                return True, (frame, self.images_id[self.index - 1])
            else:
                if loop:
                    self.index = 0
                    return self.read(loop)
                else:
                    return True, (None, 0)

    image_sequence = ImageSequence(datset_path)
    H_img_to_ground = compute_image_to_ground()
    H_bev_to_img_static = compute_homography_bev_to_img()
    H_img_to_bev_static = np.linalg.inv(H_bev_to_img_static)

    while True:
        ret, (frame, id) = image_sequence.read(loop=True)
        if not ret or frame is None:
            break

        warped_image = frame
        H_bev_to_img = compute_homography_bev_to_img()
        warped_image = compute_bev(warped_image, H_bev_to_img)

        ######################! STANDARD DISPLAY #####################

        # Display original
        cv2.putText(frame, f"Image ID: {id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame - q to exit", frame)

        ######################! BEV DISPLAY #####################

        bev_image = warped_image.copy()
        cv2.putText(bev_image, f"BEV - Image ID: {id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.imshow("BEV - q to exit", bev_image)

        ######################! LANE MANAGEMENT DISPLAY #####################


        gray_bev = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        pct_mask = cv2.GaussianBlur(gray_bev, BLUR_KSIZE, 0)


        #? GOLD Implementation
        r = gold_feature_extraction(pct_mask, m=GOLD_M, low_threshold=GOLD_LOW_THRESHOLD)
        margin = int(BEV_WIDTH * GOLD_SIDE_MARGIN_RATIO)
        r[:, :margin] = 0
        r[:, -margin:] = 0


        binary = adaptive_max_threshold(r, c=ADAPTIVE_NEIGHBORHOOD, k=ADAPTIVE_K)
        binary = cv2.GaussianBlur(binary, BLUR_KSIZE, 0)

        binary = geodesic_dilation(binary, num_iterations=GEODESIC_ITERATIONS)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, VERTICAL_OPEN_KERNEL)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

        # Crosswalk removal
        margin = int(binary.shape[1] * CROSSWALK_MARGIN_RATIO)
        active = binary[:, margin:-margin]
        row_density = np.count_nonzero(active, axis=1) / active.shape[1]
        binary[row_density > CROSSWALK_DENSITY_THRESHOLD] = 0

        # Connected component filtering: tieni solo blob verticali e abbastanza grandi
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for lab in range(1, num_labels):
            w = stats[lab, cv2.CC_STAT_WIDTH]
            h = stats[lab, cv2.CC_STAT_HEIGHT]
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= CC_FIRST_MIN_AREA and (w == 0 or h / w >= CC_MIN_ASPECT):
                filtered[labels == lab] = 255

        binary = filtered

        cv2.imshow("GOLD feature extraction", binary)

        # Second-pass CC filtering with a larger min area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for lab in range(1, num_labels):
            w = stats[lab, cv2.CC_STAT_WIDTH]
            h = stats[lab, cv2.CC_STAT_HEIGHT]
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= CC_SECOND_MIN_AREA and (w == 0 or h / w >= CC_MIN_ASPECT):
                filtered[labels == lab] = 255

        binary = filtered
    
        observations = feature_identification(binary)

        #histogram over w_i value from observations, to find the most common lane width in the image, which can be used as a prior for lane detection and filtering. This can help to identify lanes that are more likely to be valid based on their width, and to filter out noise or false positives that do not match the expected lane width distribution.
        w_histogram = {}
        for obs in observations:
            w = obs['w']
            w_histogram[w] = w_histogram.get(w, 0) + 1
        
        #lowpass filter on histogram
        w_histogram_filtered = {}
        for w in w_histogram:
            w_histogram_filtered[w] = w_histogram.get(w-1, 0) + w_histogram[w] + w_histogram.get(w+1, 0)


        # ── Step 5: column-projection histogram → lane seeds (GOLD Sec. 4.A) ──
        seeds, hist = find_lane_seeds_improved(binary,
                                      margin_ratio=LANE_MARGIN_RATIO,
                                      min_peak_distance=PEAK_MIN_DISTANCE,
                                      n_lanes=LANE_N,
                                      max_marking_width=LANE_MAX_MARKING_WIDTH)

        ###? impelemntazione per riconoscimento corsia dx o sx
        mid = binary.shape[1] // 2

        left  = [s for s in seeds if s < mid]
        right = [s for s in seeds if s >= mid]

        # Prendi il più forte per lato
        left_seed  = left[-1]  if left  else None   # più vicino al centro
        right_seed = right[0]  if right else None   # più vicino al centro

        print(f"Left seed {left_seed} –– Right seed {right_seed}")
        # ── Per-lane processing: fit + classify solid/dashed, one side at a time ──
        lanes = []

        for side, seed, history, lost, lane_colour in [
            ('left',  left_seed,  left_history,  left_lost,  LEFT_LANE_COLOR),
            ('right', right_seed, right_history, right_lost, RIGHT_LANE_COLOR),
        ]:
            fit, lane_type = None, 'solid'

            if seed is not None:
                fit = fit_lane_from_seed(binary, seed,
                                         strip_half_width=LANE_STRIP_HALF_WIDTH,
                                         deg=LANE_FIT_DEG,
                                         min_points=LANE_MIN_POINTS)


            if fit is not None:
                # Controlla coerenza con la storia
                if len(history) > 0:
                    avg_seed = np.mean([h[0] for h in history])
                    avg_slope = np.mean([h[1][0] for h in history])  # h[1] è il fit, [0] è 'a'
                    
                    seed_ok = abs(seed - avg_seed) < MAX_DEVIATION
                    slope_ok = abs(fit[0] - avg_slope) < MAX_SLOPE_DEVIATION  # es. 0.15
                    
                    if not (seed_ok and slope_ok):
                        fit = history[-1][1]
                        lane_type = history[-1][2]
                        seed = history[-1][0]
                    else:
                        lane_type = classify_lane(binary, seed)

                history.append((seed, fit, lane_type))
                lost = 0
            else:
                lost += 1
                if lost <= MAX_LOST_FRAMES and len(history) > 0:
                    # Fallback: usa l'ultimo fit buono
                    seed, fit, lane_type = history[-1]
                # else: fit resta None → no lane
            if lost > MAX_LOST_FRAMES:
                history.clear()
            # Aggiorna i contatori (deque si aggiorna in-place, lost no)
            if side == 'left':
                left_lost = lost
            else:
                right_lost = lost

            if fit is not None:
                lanes.append((seed, fit, lane_type, lane_colour))
        # ── Step 7: draw each lane on BEV with histogram overlay ──
        bev_lanes = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for _, fit, lane_type, color in lanes:
            draw_lane_bev(bev_lanes, fit, lane_type, color, thickness=LANE_THICKNESS_BEV)
        h_bev, max_hist = bev_lanes.shape[0], max(hist.max(), 1.0)
        for x_col in range(len(hist)):
            bar = int(hist[x_col] / max_hist * (h_bev // 4))
            if bar > 0:
                cv2.line(bev_lanes, (x_col, h_bev), (x_col, h_bev - bar), (0, 180, 0), 1)
        for seed, _, _, _ in lanes:
            cv2.line(bev_lanes, (seed, 0), (seed, h_bev), (0, 0, 255), 1)
        cv2.imshow("BEV lanes", bev_lanes)

        # ── Step 8: reproject each lane onto original camera frame ──
        final = frame.copy()
        if not lanes:
            # show no lanes detected message on the frame center
            cv2.putText(final, "No lanes detected", (final.shape[1] // 2 - 100, final.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for _, fit, lane_type, color in lanes:
            draw_lane_frame(final, fit, lane_type, H_bev_to_img,
                            BEV_HEIGHT, color, thickness=LANE_THICKNESS_FRAME)
        
        #show masked image over original frame
        color_mask_bev = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
        color_mask_bev[binary > 0] = [255, 0, 255]  # Colored mask (Magenta)
        inv_warped_mask = cv2.warpPerspective(color_mask_bev, H_bev_to_img, (frame.shape[1], frame.shape[0]))
        cv2.addWeighted(inv_warped_mask, 0.4, final, 1.0, 0, final)

        # ── Step 9: YOLO object detection + distance via BEV projection ──
        yolo_results = yolo_model(frame, verbose=False)[0]
        for box in yolo_results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf < YOLO_CONF_THRESHOLD or cls_id not in YOLO_KEEP_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = yolo_model.names[cls_id]

            # Ground contact = bottom-center of the bbox
            u_ground = (x1 + x2) // 2
            v_ground = y2
            ground = image_point_to_ground(u_ground, v_ground, H_img_to_ground)
            if ground is None:
                continue
            X_rel, Y_rel, distance = ground

            # Project the same point into BEV pixels for the side-view marker
            pb = H_img_to_bev_static @ np.array([u_ground, v_ground, 1.0])
            u_bev = pb[0] / pb[2]
            v_bev = pb[1] / pb[2]

            cv2.rectangle(final, (x1, y1), (x2, y2), YOLO_BOX_COLOR, 2)
            text = f"{label} {distance:.1f}m"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(final, (x1, y1 - th - 8), (x1 + tw + 4, y1), YOLO_BOX_COLOR, -1)
            cv2.putText(final, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Mirror marker on the BEV visualization if inside the ROI
            ub, vb = int(round(u_bev)), int(round(v_bev))
            if 0 <= ub < BEV_WIDTH and 0 <= vb < BEV_HEIGHT:
                cv2.circle(bev_lanes, (ub, vb), 6, YOLO_TEXT_COLOR, -1)
                cv2.putText(bev_lanes, f"{distance:.1f}m", (ub + 6, vb - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, YOLO_TEXT_COLOR, 1)
        cv2.imshow("BEV lanes", bev_lanes)

        cv2.putText(final, f"Image ID: {id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("[GOLD] final", final)

        # reset before next video loop
        if id == "00":
            left_lost = 0
            left_history.clear()
            right_lost = 0
            right_history.clear()

        
        if cv2.waitKey(WAIT_KEY_MS) & 0xFF == ord('q'):
            break

    
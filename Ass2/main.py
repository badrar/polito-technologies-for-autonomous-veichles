import os
import cv2
import numpy as np
from scipy.signal import find_peaks
from lane_detection import calculate_threshold

DEBUG = True

# Intrinsic camera parameters 
FOCAL_LENGTH = (1970.0, 1970.0)        # fx, fy in pixels
PRINCIPAL_POINT = (970.0, 483.0)       # cx, cy in pixels
CAMERA_HEIGHT = 1.66                   # meters above ground
CAMERA_X_OFFSET = 1.875               # meters forward offset
PITCH = 0.0                           # degrees
 
# lane column thresholding
SUM_THRESHOLD = 1000  # minimum sum of pixel values in a column to consider it as lane marking
PEAK_MIN_DISTANCE = 30  # minimum pixel distance between peaks for NMS

# BEV configuration TEST PARAMS
BEV_WIDTH = 600                        # pixels
BEV_HEIGHT = 600                       # pixels
X_MIN = 6                              # nearest ground distance (meters)
X_MAX = 30.0                           # farthest ground distance (meters)
Y_MIN = -8.0                           # right boundary (meters)
Y_MAX = 8.0                            # left boundary (meters)

 
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
    # Se local_threshold è 5 ma noise_floor è 40, useremo 40.
    final_threshold = np.maximum(local_threshold, 40)
    
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
    _, w = binary_bev.shape
    x_lo = max(0, seed - strip_half_width)
    x_hi = min(w, seed + strip_half_width)

    ys_rel, xs_rel = np.nonzero(binary_bev[:, x_lo:x_hi])
    if len(ys_rel) < min_points:
        return None

    ys = ys_rel.astype(np.float32)
    xs = (xs_rel + x_lo).astype(np.float32)
    try:
        return np.polyfit(ys, xs, deg)
    except np.linalg.LinAlgError:
        return None

def draw_lanes_on_bev(bev_bgr, fits, colors=None):
    """Draw polynomial lane curves on a colour BEV image."""
    if colors is None:
        colors = [(0, 255, 255), (255, 100, 0)]
    h, result = bev_bgr.shape[0], bev_bgr.copy()
    y_rng = np.arange(h, dtype=np.float32)
    for i, fit in enumerate(fits):
        if fit is None:
            continue
        x_vals = np.polyval(fit, y_rng).astype(int)
        pts = np.array([(x, y) for y, x in zip(y_rng.astype(int), x_vals)
                        if 0 <= x < result.shape[1]], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(result, [pts.reshape(-1, 1, 2)], False,
                          colors[i % len(colors)], 3)
    return result

def reproject_lanes_to_frame(frame, fits, H_bev_to_img, bev_shape, colors=None):
    """Reproject fitted BEV lane curves to the original camera frame."""
    if colors is None:
        colors = [(0, 255, 255), (255, 100, 0)]
    bev_h, bev_w = bev_shape
    ih, iw = frame.shape[:2]
    result = frame.copy()
    y_rng  = np.arange(bev_h, dtype=np.float32)
    for i, fit in enumerate(fits):
        if fit is None:
            continue
        x_vals  = np.polyval(fit, y_rng)
        valid   = (x_vals >= 0) & (x_vals < bev_w)
        pts_bev = np.column_stack([x_vals[valid], y_rng[valid]]) \
                    .reshape(-1, 1, 2).astype(np.float32)
        if pts_bev.shape[0] < 2:
            continue
        pts_img  = cv2.perspectiveTransform(pts_bev, H_bev_to_img).astype(np.int32)
        in_frame = ((pts_img[:, 0, 0] >= 0) & (pts_img[:, 0, 0] < iw) &
                    (pts_img[:, 0, 1] >= 0) & (pts_img[:, 0, 1] < ih))
        pts_img  = pts_img[in_frame]
        if len(pts_img) > 1:
            cv2.polylines(result, [pts_img], False, colors[i % len(colors)], 4)
    return result

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

def classify_lane(binary_bev, seed_x, band=10, gap_threshold=0.5):
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
    video_id="008"
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

    while True:
        ret, (frame, id) = image_sequence.read(loop=True)
        if not ret:
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
        pct_mask = cv2.GaussianBlur(gray_bev, (3, 3), 0)

        
        #? GOLD Implementation
        r = gold_feature_extraction(pct_mask, m=10, low_threshold=30)
        margin = int(BEV_WIDTH * 0.3)
        r[:, :margin] = 0
        r[:, -margin:] = 0


        binary = adaptive_max_threshold(r, c=8, k=2)
        binary = cv2.GaussianBlur(binary, (3, 3), 0)

        binary = geodesic_dilation(binary, num_iterations=8)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

        # Crosswalk removal
        margin = int(binary.shape[1] * 0.2)
        active = binary[:, margin:-margin]
        row_density = np.count_nonzero(active, axis=1) / active.shape[1]
        binary[row_density > 0.10] = 0

        # Connected component filtering: tieni solo blob verticali e abbastanza grandi
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = 20
        min_aspect = 2.0  # altezza/larghezza >= 2 → struttura verticale

        filtered = np.zeros_like(binary)
        for lab in range(1, num_labels):
            w = stats[lab, cv2.CC_STAT_WIDTH]
            h = stats[lab, cv2.CC_STAT_HEIGHT]
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_area and (w == 0 or h / w >= min_aspect):
                filtered[labels == lab] = 255

        binary = filtered

        cv2.imshow("GOLD feature extraction", binary)

        # Connected component filtering: tieni solo blob verticali e abbastanza grandi
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = 50
        min_aspect = 2.0  # altezza/larghezza >= 2 → struttura verticale

        filtered = np.zeros_like(binary)
        for lab in range(1, num_labels):
            w = stats[lab, cv2.CC_STAT_WIDTH]
            h = stats[lab, cv2.CC_STAT_HEIGHT]
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_area and (w == 0 or h / w >= min_aspect):
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
                                      margin_ratio=0.15,
                                      min_peak_distance=PEAK_MIN_DISTANCE,
                                      n_lanes=2)

        ### ? impelemntazione per riconoscimento corsia dx o sx
        mid = binary.shape[1] // 2

        left  = [s for s in seeds if s < mid]
        right = [s for s in seeds if s >= mid]

        # Prendi il più forte per lato
        left_seed  = left[-1]  if left  else None   # più vicino al centro
        right_seed = right[0]  if right else None   # più vicino al centro

        if left_seed is not None:
            lane_type = classify_lane(binary, left_seed)
            print(f"Left lane seed at x={left_seed} classified as {lane_type}")
        if right_seed is not None:
            lane_type = classify_lane(binary, right_seed)
            print(f"Right lane seed at x={right_seed} classified as {lane_type}")

        seeds = []
        if left_seed is not None:
            seeds.append(left_seed)
        if right_seed is not None:
            seeds.append(right_seed)

        # ── Step 6: strip pixel collection + polynomial fit (deg=1 = straight road) ──
        fits = [fit_lane_from_seed(binary, s, strip_half_width=25, deg=1)
                for s in seeds]

        # ── Step 7: draw fitted curves on BEV with histogram overlay ──
        bev_lanes = draw_lanes_on_bev(
            cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), fits)
        h_bev, max_hist = bev_lanes.shape[0], max(hist.max(), 1.0)
        for x_col in range(len(hist)):
            bar = int(hist[x_col] / max_hist * (h_bev // 4))
            if bar > 0:
                cv2.line(bev_lanes, (x_col, h_bev), (x_col, h_bev - bar), (0, 180, 0), 1)
        for s in seeds:
            cv2.line(bev_lanes, (s, 0), (s, h_bev), (0, 0, 255), 1)
        cv2.imshow("BEV lanes", bev_lanes)

        # ── Step 8: reproject lanes onto original camera frame ──
        final = reproject_lanes_to_frame(frame, fits, H_bev_to_img,
                                         (BEV_HEIGHT, BEV_WIDTH))
        
        #show masked image over original frame
        color_mask_bev = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
        color_mask_bev[binary > 0] = [255, 0, 255]  # Colored mask (Magenta)
        inv_warped_mask = cv2.warpPerspective(color_mask_bev, H_bev_to_img, (frame.shape[1], frame.shape[0]))
        cv2.addWeighted(inv_warped_mask, 0.4, final, 1.0, 0, final)

        cv2.putText(final, f"Image ID: {id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("[GOLD] final", final)

        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

    
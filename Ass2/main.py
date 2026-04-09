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

# BEV configuration SAFE PARAMS
# BEV_WIDTH = 300                        # pixels
# BEV_HEIGHT = 600                       # pixels
# X_MIN = 6                            # nearest ground distance (meters)
# X_MAX = 30.0                           # farthest ground distance (meters)
# Y_MIN = -8.0                           # right boundary (meters)
# Y_MAX = 8.0                            # left boundary (meters)

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

import numpy as np
import cv2

import numpy as np
import cv2

def adaptive_max_threshold_FIXED(e, c=7, k=2, noise_floor=40):
    """
    Implementa la binarizzazione adattiva basata sul massimo locale (Equazione 9),
    con l'aggiunta vitale di un noise_floor per evitare il rumore sull'asfalto.
    
    e: immagine enhanced in ingresso (uint8)
    c: dimensione del vicinato (c x c)
    k: costante di divisione (dal paper k=2)
    noise_floor: valore minimo assoluto sotto il quale ignoriamo l'adattività (es. 40-60)
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
    final_threshold = np.maximum(local_threshold, 90)
    
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

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IDENTIFICATION  –  GOLD paper Section 3.B
# ═══════════════════════════════════════════════════════════════════════════════

def feature_identification(binary_bev):
    """
    GOLD paper – Feature Identification (Section 3.B).

    Scans the binary BEV image row by row.  For each row, nonzero pixels are
    considered in pairs.  A pair (p, q) with p < q can represent one of three
    road configurations:
        A)  p = left edge,   q = right edge   → c = (p+q)/2,  w = (q-p)/2
        B)  p = left edge,   q = centre line  → c = q,        w = q-p
        C)  p = centre line, q = right edge   → c = p,        w = q-p

    Only pairs satisfying all four GOLD constraints are kept:
        0  ≤ c  ≤ N
        w  < N/3
        c  - w  ≤ (3/4) N
        c  + w  ≥ N/4

    Args:
        binary_bev: uint8 binary image (H × W), nonzero pixels are candidates.

    Returns:
        List of dicts, one per valid observation:
            {'row': i, 'c': c_i, 'w': w_i, 'config': 'A'|'B'|'C'}
    """
    H, N = binary_bev.shape
    observations = []

    for i in range(H):
        cols = np.nonzero(binary_bev[i])[0]
        if len(cols) < 2:
            continue

        # Consider every pair (p, q) with p < q
        for idx_p in range(len(cols)):
            for idx_q in range(idx_p + 1, len(cols)):
                p, q = int(cols[idx_p]), int(cols[idx_q])

                candidates = [
                    ((p + q) / 2.0, (q - p) / 2.0, 'A'),  # left + right edge
                    (float(q),       float(q - p),   'B'),  # left edge + centre
                    (float(p),       float(q - p),   'C'),  # centre + right edge
                ]

                for c, w, cfg in candidates:
                    if (0 <= c <= N and
                            w < N / 3 and
                            c - w <= 0.75 * N and
                            c + w >= N / 4):
                        observations.append({'row': i, 'c': c, 'w': w, 'config': cfg})

    return observations


def build_road_chain(observations, W, max_col_gap=20, max_row_gap=15):
    """
    GOLD paper – road-centre chain building (Section 3.B, continuation).

    Steps:
      1. Filter observations to those with  W - W/4 < w_i < W + W/4.
      2. Scan bottom-to-top.  For each candidate in a row, the best
         predecessor is the closest (in column) observation in any of the
         `max_row_gap` rows directly below it (vertical correlation).
      3. Return the longest such chain (DP), plus the filtered observations.

    Args:
        observations: output of feature_identification()
        W:            dominant lane width (from histogram)
        max_col_gap:  max |c_i - c_{i-1}| allowed between consecutive links
        max_row_gap:  max row distance to search for a predecessor

    Returns:
        chain    – list of {'row','c','w'} (longest chain, bottom-to-top order)
        filtered – observations that passed the width filter
    """
    from collections import defaultdict

    if W == 0:
        return [], observations

    w_lo = W - W / 4.0
    w_hi = W + W / 4.0
    filtered = [o for o in observations if w_lo < o['w'] < w_hi]

    if not filtered:
        return [], filtered

    by_row = defaultdict(list)
    for o in filtered:
        by_row[o['row']].append(o)

    rows_sorted = sorted(by_row.keys(), reverse=True)   # highest row first (bottom of image)
    row_set = set(rows_sorted)

    # dp[(row, j)] = (chain_length, prev_key | None)
    dp = {}

    for row in rows_sorted:
        for j, obs in enumerate(by_row[row]):
            best_len, best_prev = 1, None
            # Search predecessor rows: any row in (row, row+max_row_gap]
            for delta in range(1, max_row_gap + 1):
                prev_row = row + delta
                if prev_row not in row_set:
                    continue
                for k, prev_obs in enumerate(by_row[prev_row]):
                    if abs(obs['c'] - prev_obs['c']) <= max_col_gap:
                        prev_len = dp.get((prev_row, k), (1, None))[0]
                        if prev_len + 1 > best_len:
                            best_len = prev_len + 1
                            best_prev = (prev_row, k)
            dp[(row, j)] = (best_len, best_prev)

    # Trace back the longest chain
    best_key = max(dp, key=lambda k: dp[k][0])
    chain = []
    key = best_key
    while key is not None:
        r, j = key
        chain.append(by_row[r][j])
        key = dp[key][1]

    chain.reverse()   # now bottom-to-top
    return chain, filtered


# ═══════════════════════════════════════════════════════════════════════════════
# LANE MODEL FITTING  –  GOLD paper Section 4.A (post-binarisation steps)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_chain(chain, deg=1, min_points=5):
    """
    Fit a polynomial  x = poly(y)  through the road-centre chain.

    Args:
        chain:      list of {'row', 'c', 'w'} from build_road_chain()
        deg:        polynomial degree (1 = line, 2 = parabola)
        min_points: minimum chain length to attempt a fit

    Returns:
        np.poly1d coefficients (highest degree first), or None
    """
    if len(chain) < min_points:
        return None
    ys = np.array([o['row'] for o in chain], dtype=np.float32)
    xs = np.array([o['c']   for o in chain], dtype=np.float32)
    try:
        return np.polyfit(ys, xs, deg)
    except np.linalg.LinAlgError:
        return None


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


def fit_lane_from_seed(binary_bev, seed, strip_half_width=30, deg=1, min_points=10):
    """
    Collect all white pixels inside a vertical strip centred on `seed` and
    fit the polynomial  x = poly(y)  via least squares (Eq. 4.A of GOLD).

    deg=1  → straight road (motorway default)
    deg=2  → curved road
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

if __name__ == "__main__":
    video_id="044"
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

    # Trackbar for percentile threshold (50-100, mapped to 50.0-100.0)
    cv2.namedWindow("Percentile BEV - q to exit")
    cv2.createTrackbar("Percentile", "Percentile BEV - q to exit", 97, 100, lambda x: None)

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

        ######## ! PROF IMPLEMENTATION DISPLAY ########

        gray_bev = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        pct_mask_float = gray_bev.astype(np.float32)

        th = calcola_soglia_iterativa(gray_bev)
        _, binary_bev = cv2.threshold(gray_bev, th, 255, cv2.THRESH_BINARY)
        cv2.putText(binary_bev, f"Iterative Th={th:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("Iterative Thresholding - q to exit", binary_bev)

        ######################! LANE MANAGEMENT DISPLAY #####################


        # Percentile threshold with interactive slider
        gray_bev = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        pct = cv2.getTrackbarPos("Percentile", "Percentile BEV - q to exit")
        pct = max(pct, 1) # avoid 0 percentile

        #filtered_bev = lane_filter(gray_bev, narrow=3, wide=10)
        filtered_bev = gray_bev.copy()
        th_pct = calculate_threshold(filtered_bev, percentile=float(pct))
        #pct_mask = binarize(filtered_bev, th_pct)
        pct_mask = gray_bev.copy()
        #pct_mask = cv2.GaussianBlur(pct_mask, (5, 5), 0)
        # Kernel verticale: pesa i vicini lungo la direzione delle lane
        #density = cv2.blur(pct_mask.astype(np.float32), (2, 21))
        # conv
        #pct_mask = (density > density.mean()).astype(np.uint8) * 255
        #pct_mask = binarize(pct_mask, th_pct)
        

        # removes the left and right 15% of the BEV to avoid noise from borders

        print(f"Matrix size: {filtered_bev.shape}, Non-zero pixels: {np.count_nonzero(filtered_bev)}, Threshold: {th_pct:.2f}")
        
        #? GOLD Implementation
        r = gold_feature_extraction(pct_mask,m=10)

        imporoved = geodesic_dilation(r, num_iterations=8)
        
        margin = int(BEV_WIDTH * 0.2)
        imporoved[:, :margin] = 0
        imporoved[:, -margin:] = 0

        filtered_improved = adaptive_max_threshold_FIXED(imporoved, c=33, k=2, noise_floor=80)
        
        observations = feature_identification(filtered_improved)

        #histogram over w_i value from observations, to find the most common lane width in the image, which can be used as a prior for lane detection and filtering. This can help to identify lanes that are more likely to be valid based on their width, and to filter out noise or false positives that do not match the expected lane width distribution.
        w_histogram = {}
        for obs in observations:
            w = obs['w']
            w_histogram[w] = w_histogram.get(w, 0) + 1
        
        #lowpass filter on histogram
        w_histogram_filtered = {}
        for w in w_histogram:
            w_histogram_filtered[w] = w_histogram.get(w-1, 0) + w_histogram[w] + w_histogram.get(w+1, 0)

        #W max
        w_max = max(w_histogram_filtered, key=w_histogram_filtered.get) if w_histogram_filtered else 0

        chain, filtered_obs = build_road_chain(observations, W=w_max)
        center_fit = fit_chain(chain, deg=1)

        # show bev with center_fit and centerfit + w_max/2 and centerfit - w_max/2 as lane boundaries (if center_fit is not None)
        if center_fit is not None and len(chain) >= 2:
            bev_lanes = cv2.cvtColor(filtered_improved, cv2.COLOR_GRAY2BGR)
            y_min = min(o['row'] for o in chain)
            y_max = max(o['row'] for o in chain)
            y_rng = np.arange(y_min, y_max + 1, dtype=np.float32)
            x_center = np.polyval(center_fit, y_rng)
            half_w = w_max / 2.0

            for y, xc in zip(y_rng, x_center):
                yy = int(y)
                # left boundary
                xl = int(xc - half_w)
                if 0 <= xl < bev_lanes.shape[1]:
                    bev_lanes[yy, xl] = (255, 0, 0)
                # center line
                xci = int(xc)
                if 0 <= xci < bev_lanes.shape[1]:
                    bev_lanes[yy, xci] = (0, 255, 0)
                # right boundary
                xr = int(xc + half_w)
                if 0 <= xr < bev_lanes.shape[1]:
                    bev_lanes[yy, xr] = (0, 0, 255)

            cv2.imshow("BEV Lane Boundaries", bev_lanes)

        # Visualise chain + fitted centre line on the BEV image
        chain_vis = cv2.cvtColor(filtered_improved, cv2.COLOR_GRAY2BGR)
        for obs in filtered_obs:
            cv2.circle(chain_vis, (int(obs['c']), obs['row']), 1, (0, 80, 200), -1)
        for obs in chain:
            cv2.circle(chain_vis, (int(obs['c']), obs['row']), 2, (0, 255, 0), -1)
        # Draw polynomial only within the row range covered by the chain
        if center_fit is not None and len(chain) >= 2:
            y_min = min(o['row'] for o in chain)
            y_max = max(o['row'] for o in chain)
            y_rng = np.arange(y_min, y_max + 1, dtype=np.float32)
            x_vals = np.polyval(center_fit, y_rng).astype(int)
            pts = np.array([(x, int(y)) for y, x in zip(y_rng, x_vals)
                            if 0 <= x < chain_vis.shape[1]], dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(chain_vis, [pts.reshape(-1, 1, 2)], False, (0, 255, 0), 2)
        cv2.imshow("Road chain", chain_vis)

        #cv2.imshow("Risultato Ridge Detection", r)
        #cv2.imshow("Improved", imporoved)
        #cv2.imshow("Filtered improved", filtered_improved)

        # #! histogram generation
        # # consider full BEV vertical data
        # #y_values = pct_mask.sum(axis=0).astype(np.float64)
        # # Consider only lower half of BEV to reduce noise (expecially in cases where the upper part is very noisy due to a bump or a shadow ie 043))
        # y_values = np.sum(filtered_improved[filtered_improved.shape[0]//2:,:], axis=0).astype(np.float64)

        # #? force y values to be between 0 and 1 -> helps managing data
        # y_values = y_values / (y_values.max() + 1e-5)

        # # find peaks in y_values (local maxima)
        # from scipy.signal import find_peaks
        # peaks, _ = find_peaks(y_values)  # > 50%


        # print(f"Y values: mean={y_values.mean():.2f}, std={y_values.std():.2f}, max={y_values.max():.2f}")
        # # Tieni solo i picchi significativi (sopra media + 1 std)

        # #? Istogramma colonne come overlay
        # overlay = cv2.cvtColor(filtered_improved, cv2.COLOR_GRAY2BGR)
        # h_img = overlay.shape[0]
        # max_val = max(y_values.max(), 1)
        # # Sfondo scuro semitrasparente
        # dark = overlay.copy()
        # dark[:] = (0, 0, 0)
        # overlay = cv2.addWeighted(overlay, 0.5, dark, 0.5, 0)

        # # non maximum suppression: disegna solo i picchi in un certo raggio (es. 30 pixel) per evitare sovrapposizioni
        # peaks_by_value = sorted(peaks, key=lambda p: y_values[p], reverse=True)
        # nms_peaks = []
        # for p in peaks_by_value:
        #     if y_values[p] > 0 and all(abs(p - k) >= PEAK_MIN_DISTANCE for k in nms_peaks):
        #         nms_peaks.append(p)
        # for p in nms_peaks:
        #     cv2.line(overlay, (p, 0), (p, h_img), (0, 0, 255), 1)


        # for x in range(len(y_values)):
        #     bar_h = int(y_values[x] / max_val * (h_img//3))
        #     cv2.line(overlay, (x, h_img), (x, h_img - bar_h), (0, 255, 0), 1)

        # cv2.putText(overlay, f"Percentile={pct} th={th_pct:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # cv2.imshow("Percentile BEV - q to exit", overlay)

        # if DEBUG:
        #     print(f"[INFO] Peaks found at columns: {peaks}")
        #     for x in nms_peaks:
        #         pct_mask[:, x] = np.uint8(255)

        # # Riproietta maschera BEV → immagine originale
        # lane_on_img = cv2.warpPerspective(pct_mask, H_bev_to_img, (frame.shape[1], frame.shape[0]))
        # result = frame.copy()
        # result[lane_on_img > 0] = (0, 255, 0)
        # cv2.imshow("Lane Overlay - q to exit", result)

        # ── Step 5: column-projection histogram → lane seeds (GOLD Sec. 4.A) ──
        seeds, hist = find_lane_seeds(filtered_improved,
                                      margin_ratio=0.15,
                                      min_peak_distance=PEAK_MIN_DISTANCE,
                                      n_lanes=2)

        # ── Step 6: strip pixel collection + polynomial fit (deg=1 = straight road) ──
        fits = [fit_lane_from_seed(filtered_improved, s, strip_half_width=25, deg=1)
                for s in seeds]

        if DEBUG:
            print(f"[GOLD] seeds={seeds}  fits_ok={[f is not None for f in fits]}")

        # ── Step 7: draw fitted curves on BEV with histogram overlay ──
        bev_lanes = draw_lanes_on_bev(
            cv2.cvtColor(filtered_improved, cv2.COLOR_GRAY2BGR), fits)
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
        cv2.putText(final, f"Image ID: {id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("[GOLD] final", final)

        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

    
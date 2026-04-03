import os
import cv2
import numpy as np

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



def calculate_threshold(img_gray: np.ndarray, percentile: float = 97.0) -> float:
    """Soglia basata su percentile: tiene solo i pixel più luminosi,
    cioè le lane markings."""
    return np.percentile(img_gray[img_gray > 0], percentile)

def lane_filter(img_gray: np.ndarray,
                narrow: int = 2, wide: int = 5) -> np.ndarray:
    """Filtro a doppio top-hat per lane markings.
    
    - Top-hat con kernel WIDE: evidenzia tutto ciò che è più stretto
      di 'wide' pixel (corsie + rumore)
    - Top-hat con kernel NARROW: evidenzia ciò che è più stretto
      di 'narrow' pixel (solo rumore, crepe, ecc.)
    - La differenza tiene solo le strutture tra narrow e wide pixel
      di larghezza, cioè le lane markings.
    """
    kern_w = cv2.getStructuringElement(cv2.MORPH_RECT, (wide, 1))
    kern_n = cv2.getStructuringElement(cv2.MORPH_RECT, (narrow, 1))

    tophat_wide = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kern_w)
    tophat_narrow = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kern_n)

    filtered = cv2.subtract(tophat_wide, tophat_narrow)
    return filtered


def binarize(img_gray: np.ndarray, th: float) -> np.ndarray:
    """Binarizzazione: pixel > soglia → 255, altrimenti 0."""
    return np.where(img_gray > th, np.uint8(255), np.uint8(0))


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

        ######################! LANE MANAGEMENT DISPLAY #####################


        # Percentile threshold with interactive slider
        gray_bev = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        pct = cv2.getTrackbarPos("Percentile", "Percentile BEV - q to exit")
        pct = max(pct, 1) # avoid 0 percentile

        filtered_bev = lane_filter(gray_bev, narrow=3, wide=10)
        th_pct = calculate_threshold(filtered_bev, percentile=float(pct))
        pct_mask = binarize(filtered_bev, th_pct)
        # Kernel verticale: pesa i vicini lungo la direzione delle lane
        density = cv2.blur(pct_mask.astype(np.float32), (2, 21))
        # conv
        #pct_mask = (density > density.mean()).astype(np.uint8) * 255
        pct_mask = binarize(pct_mask, th_pct)

        # removes the left and right 15% of the BEV to avoid noise from borders
        margin = int(BEV_WIDTH * 0.3)
        pct_mask[:, :margin] = 0
        pct_mask[:, -margin:] = 0
        print(f"Matrix size: {filtered_bev.shape}, Non-zero pixels: {np.count_nonzero(filtered_bev)}, Threshold: {th_pct:.2f}")
        

        #! histogram generation
        # consider full BEV vertical data
        #y_values = pct_mask.sum(axis=0).astype(np.float64)
        # Consider only lower half of BEV to reduce noise (expecially in cases where the upper part is very noisy due to a bump or a shadow ie 043))
        y_values = np.sum(pct_mask[pct_mask.shape[0]//2:,:], axis=0).astype(np.float64)

        #? force y values to be between 0 and 1 -> helps managing data
        y_values = y_values / (y_values.max() + 1e-5)

        # find peaks in y_values (local maxima)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(y_values)  # > 50%


        print(f"Y values: mean={y_values.mean():.2f}, std={y_values.std():.2f}, max={y_values.max():.2f}")
        # Tieni solo i picchi significativi (sopra media + 1 std)
        y_values = np.where(y_values > 0.5, y_values, 0)

        #? Istogramma colonne come overlay
        overlay = cv2.cvtColor(pct_mask, cv2.COLOR_GRAY2BGR)
        h_img = overlay.shape[0]
        max_val = max(y_values.max(), 1)
        # Sfondo scuro semitrasparente
        dark = overlay.copy()
        dark[:] = (0, 0, 0)
        overlay = cv2.addWeighted(overlay, 0.5, dark, 0.5, 0)

        # non maximum suppression: disegna solo i picchi in un certo raggio (es. 30 pixel) per evitare sovrapposizioni
        peaks_by_value = sorted(peaks, key=lambda p: y_values[p], reverse=True)
        nms_peaks = []
        for p in peaks_by_value:
            if y_values[p] > 0 and all(abs(p - k) >= PEAK_MIN_DISTANCE for k in nms_peaks):
                nms_peaks.append(p)
        for p in nms_peaks:
            cv2.line(overlay, (p, 0), (p, h_img), (0, 255, 0), 1)


        for x in range(len(y_values)):
            if(x in nms_peaks):
                bar_h = int(y_values[x] / max_val * h_img)
                cv2.line(overlay, (x, h_img), (x, h_img - bar_h), (0, 255, 0), 1)

        cv2.putText(overlay, f"Percentile={pct} th={th_pct:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("Percentile BEV - q to exit", overlay)

        if DEBUG:
            print(f"[INFO] Peaks found at columns: {peaks}")
            for x in nms_peaks:
                pct_mask[:, x] = np.uint8(255)

        # Riproietta maschera BEV → immagine originale
        lane_on_img = cv2.warpPerspective(pct_mask, H_bev_to_img, (frame.shape[1], frame.shape[0]))
        result = frame.copy()
        result[lane_on_img > 0] = (0, 255, 0)
        cv2.imshow("Lane Overlay - q to exit", result)

        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

    
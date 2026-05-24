# ==================== IMPORTS ====================
import sys

import cv2                                  # OpenCV for image/video processing
import mediapipe as mp                      # MediaPipe for face detection
import numpy as np                          # NumPy for numerical operations
import time                                 # Time measurement
import statistics as st                     # Statistics (unused)
import os                                   # File path utilities
import urllib.request                       # Model download

from mediapipe.tasks import python
from mediapipe.tasks.python import vision   # MediaPipe vision tasks

import numpy as np
from scipy.signal import butter, filtfilt, detrend, welch
from sklearn.decomposition import FastICA
from collections import deque

# ==================== MODEL LOADING ====================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    # Download the pre-trained model if not present locally
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print("Downloading face_landmarker.task...")
    urllib.request.urlretrieve(url, MODEL_PATH)

# ==================== FACE LANDMARKER CONFIGURATION ====================
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,  # Video mode for consecutive frames
    num_faces=1,                            # Track a single face
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ==================== FACE LANDMARK INDICES ====================
# MediaPipe Face Landmarker produces 478 landmarks (468 face + 10 iris)
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]  # left eye contour
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]    # right eye contour
LEFT_IRIS  = [473, 474, 475, 476, 477]   # left iris ring
RIGHT_IRIS = [468, 469, 470, 471, 472]   # right iris ring
NOSE_TIP   = [45, 4, 275]               # nose tip landmarks

# Eye corners and eyelid contours (outer → inner)
RIGHT_OUTER, RIGHT_INNER = 33, 133
RIGHT_UPPER_LID = [246, 161, 160, 159, 158, 157]
RIGHT_LOWER_LID = [7,   163, 144, 145, 153, 154]

LEFT_OUTER, LEFT_INNER = 362, 263
LEFT_UPPER_LID = [466, 388, 387, 386, 385, 384]
LEFT_LOWER_LID = [249, 390, 373, 374, 380, 381]

# rPPG — forehead ROI polygon (landmark indices)
FOREHEAD_ROI = [109, 10, 338, 337, 336, 9, 107, 108]
BPM_WINDOW_S       = 10.0   # seconds for BPM estimation window
BPM_UPDATE_EVERY_N = 60     # recompute BPM every N frames (for efficiency)

from enum import Enum

class MicrosleepState(Enum):
    NORMAL    = 0
    WARNING_4 = 1   # eyes closed >= 4 s
    WARNING_7 = 2   # eyes closed >= 7 s
    RECOVERY  = 3   # eyes open, waiting for the 2 s recovery window


class MicrosleepDetector:
    """Detects microsleep episodes based on continuous eye-closure duration.

    Raises WARNING_4 after 4 s of continuous closure, WARNING_7 after 7 s.
    Returns to NORMAL only after the eyes have been open for at least 2 s
    (RECOVERY state), preventing instant state resets on brief blinks.
    """

    CLOSED_4S   = 4.0
    CLOSED_7S   = 7.0
    RECOVERY_2S = 2.0

    def __init__(self):
        self.state          = MicrosleepState.NORMAL
        self.closed_since   = None   # timestamp of the start of the current closure
        self.opened_since   = None   # timestamp of the start of the recovery window
        self._recovery_from = None   # state from which recovery began

    def update(self, is_closed: bool, now: float) -> MicrosleepState:
        """Update the detector with the current eye-closure status.

        Args:
            is_closed: True if both eyes are considered closed this frame.
            now: Current wall-clock time in seconds.

        Returns:
            The updated MicrosleepState.
        """
        if is_closed:
            self.opened_since = None          # interrupt any ongoing recovery
            if self.closed_since is None:
                self.closed_since = now
            closed_dur = now - self.closed_since
            if closed_dur >= self.CLOSED_7S:
                self.state = MicrosleepState.WARNING_7
            elif closed_dur >= self.CLOSED_4S:
                self.state = MicrosleepState.WARNING_4
            # state unchanged if already in WARNING/RECOVERY and closure < 4 s
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
    WARNING_LONG  = 1   # single episode >= 5 s; immediate recovery when gaze returns
    WARNING_SHORT = 2   # accumulated >= 10 s within a 30 s sliding window; 2 s recovery
    RECOVERY      = 3   # waiting for the 2 s on-road window before returning to NORMAL


class DistractionDetector:
    """Tracks gaze-away episodes and raises warnings based on ISO/SAE DMS criteria.

    Two warning types are distinguished:
    - WARNING_LONG: a single uninterrupted distraction episode lasting >= 5 s.
    - WARNING_SHORT: cumulative gaze-away time >= 10 s within the last 30 s.

    Recovery from WARNING_LONG is immediate; recovery from WARNING_SHORT requires
    the driver to look at the road continuously for at least 2 s (RECOVERY state).
    """

    LONG_THRESHOLD   = 5.0    # s: single-episode threshold
    SHORT_ACCUMULATE = 10.0   # s: cumulative threshold within the sliding window
    SHORT_WINDOW     = 30.0   # s: width of the sliding accumulation window
    SHORT_RECOVERY   = 2.0    # s: on-road time required to exit WARNING_SHORT

    def __init__(self):
        self.state         = DistractionState.NORMAL
        self.away_since    = None   # start timestamp of the current distraction episode
        self.on_road_since = None   # start timestamp of the current recovery window
        self.away_log      = []     # list of completed (start, end) episode pairs

    def _accumulated_away(self, now: float) -> float:
        """Return total gaze-away time (seconds) within the sliding window ending at now."""
        window_start = now - self.SHORT_WINDOW
        self.away_log = [(s, e) for s, e in self.away_log if e > window_start]
        total = sum(min(e, now) - max(s, window_start) for s, e in self.away_log)
        if self.away_since is not None:
            total += now - max(self.away_since, window_start)
        return total

    def update(self, is_distracted: bool, now: float) -> DistractionState:
        """Update the detector with the current distraction flag.

        Args:
            is_distracted: True if the driver is currently looking away.
            now: Current wall-clock time in seconds.

        Returns:
            The updated DistractionState.
        """
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
                self.state = DistractionState.NORMAL        # immediate recovery for long warning
            elif self.state in (DistractionState.WARNING_SHORT, DistractionState.RECOVERY):
                if self.on_road_since is None:
                    self.on_road_since = now
                self.state = DistractionState.RECOVERY
                if now - self.on_road_since >= self.SHORT_RECOVERY:
                    self.state = DistractionState.NORMAL
                    self.on_road_since = None
        return self.state


def owl_yaw(face_landmarks) -> float:
    """Measure horizontal head yaw as nose deviation from the inter-eye midpoint.

    Returns a value in [0, 0.5]: 0 means the nose is perfectly centred between
    the eye corners (frontal gaze), 0.5 means the face is in full profile.
    """
    nose   = face_landmarks[4].x
    l_corn = face_landmarks[33].x
    r_corn = face_landmarks[263].x
    total_w = abs(r_corn - l_corn)
    if total_w < 1e-6:
        return 0.0
    return abs((nose - l_corn) / total_w - 0.5)


def lizard_gaze(face_landmarks) -> float:
    """Measure lateral iris deviation relative to the eye width.

    Computes, for each eye, the absolute offset of the iris centre from the
    horizontal midpoint of the eye fissure, normalised by the eye width.
    Returns the average over both eyes in [0, 0.5]: 0 means both irises are
    centred, larger values indicate sideways gaze with the head still forward.
    """
    def offset(outer_idx, inner_idx, iris_idx):
        outer = face_landmarks[outer_idx].x
        inner = face_landmarks[inner_idx].x
        iris  = face_landmarks[iris_idx].x
        w = abs(inner - outer)
        if w < 1e-6:
            return 0.0
        center = (outer + inner) / 2
        return abs(iris - center) / w   # symmetric: direction does not matter

    return (offset(33, 133, 468) + offset(362, 263, 473)) / 2


def is_owl_distracted(face_landmarks, yaw_threshold=0.25) -> bool:
    return owl_yaw(face_landmarks) > yaw_threshold


def is_lizard_distracted(face_landmarks, iris_threshold=0.15) -> bool:
    return lizard_gaze(face_landmarks) > iris_threshold


# BGR colours used for on-screen status labels
_GREEN  = (0, 200, 0)
_ORANGE = (0, 165, 255)
_RED    = (0, 0, 255)

def driver_state_label(ms_state, ms_recovery_from, owl_state, lizard_state):
    """Return the highest-priority driver state as a (text, BGR colour) pair.

    Priority order (highest first): Sleep > Microsleep > Distracted > Focused.
    """
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
    """Overlay a debug panel in the top-right corner showing live timer values for all three detectors."""
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
    """Draw the driver state label in the bottom-right corner with a semi-transparent background."""
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
    """Determines whether the eyes are closed using a PERCLOS-80 criterion.

    During a calibration phase the open-eye aperture baseline is established.
    On each subsequent frame the eyelid aperture is expressed as a closure
    percentage relative to that baseline.  A short moving-average filter is
    applied to reduce landmark jitter, and the eye is considered closed when
    the smoothed closure reaches the PERCLOS threshold (default 60 %).
    """

    def __init__(self, perclos_threshold=0.60, history_size=5):
        self.baseline_aperture = None
        self.perclos_threshold = perclos_threshold  # PERCLOS-80: closed if aperture <= 20% of baseline
        self.history = []
        self.history_size = history_size

    def calibrate(self, apertures_awake):
        """Set the open-eye baseline from samples collected while the driver was alert.

        Uses the median instead of the mean to be robust against occasional blinks
        and landmark outliers during the calibration window.
        """
        self.baseline_aperture = np.median(apertures_awake)

    def is_closed(self, aperture):
        """Return True if the smoothed closure percentage meets the PERCLOS threshold.

        Args:
            aperture: Current eyelid aperture in pixels, or None if unmeasurable.

        Returns:
            True if the eye is considered closed, False otherwise.
        """
        if aperture is None:
            return True  # unmeasurable aperture treated as closed (e.g. heavy occlusion)
        if self.baseline_aperture is None:
            return False  # still in calibration phase

        closure_pct = 1.0 - (aperture / self.baseline_aperture)
        closure_pct = np.clip(closure_pct, 0.0, 1.0)

        # moving-average smoothing to reduce per-frame landmark jitter
        self.history.append(closure_pct)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        smoothed = np.mean(self.history)

        return smoothed >= self.perclos_threshold

def _line_polyline_intersection(p0, dir_vec, polyline):
    """Find the first intersection of an infinite line with a polyline.

    The line is parametrised as p0 + t * dir_vec.  For each segment of the
    polyline the system p0 + t*dir = a + s*(b-a) is solved; the intersection
    is returned for the first segment where 0 <= s <= 1.

    Returns the 2-D intersection point, or None if no intersection is found.
    """
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        # solve: p0 + t*dir = a + s*(b-a),  0 <= s <= 1
        seg = b - a
        M = np.array([[dir_vec[0], -seg[0]],
                      [dir_vec[1], -seg[1]]])
        rhs = a - p0
        if abs(np.linalg.det(M)) < 1e-9:
            continue  # segments are parallel
        t, s = np.linalg.solve(M, rhs)
        if 0.0 <= s <= 1.0:
            return p0 + t * dir_vec
    return None


def eyelid_aperture(landmarks, outer_idx, inner_idx, upper_lid_idx, lower_lid_idx,
                    img_w, img_h):
    """Compute the eyelid aperture in pixels using the driver-monitoring standard definition.

    The aperture is measured as the distance between the upper and lower eyelid
    along the line perpendicular to the eye-corner segment, passing through its
    midpoint.  This is robust to head roll because the measurement direction
    always adapts to the eye orientation.

    Returns the aperture in pixels, or None if the eye is too closed for the
    perpendicular line to intersect both eyelid polylines.
    """
    to_px = lambda i: np.array([landmarks[i].x * img_w, landmarks[i].y * img_h])

    p_outer = to_px(outer_idx)
    p_inner = to_px(inner_idx)
    upper = np.array([to_px(i) for i in upper_lid_idx])
    lower = np.array([to_px(i) for i in lower_lid_idx])

    # perpendicular to the outer-inner segment, anchored at its midpoint
    midpoint = 0.5 * (p_outer + p_inner)
    d = p_inner - p_outer
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])  # 90° rotation of d

    p_up = _line_polyline_intersection(midpoint, n, upper)
    p_low = _line_polyline_intersection(midpoint, n, lower)

    if p_up is None or p_low is None:
        return None  # degenerate case: eye almost fully closed

    return np.linalg.norm(p_up - p_low)


CALIBRATION_SECONDS = 3  # seconds at startup used to establish the open-eye baseline

def build_mask(img_shape, landmarks, roi_indices):
    """Build a binary mask for the polygonal ROI defined by the given landmark indices."""
    h, w = img_shape[:2]
    pts = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in roi_indices],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def extract_roi_rgb(frame_rgb, landmarks, roi_indices, img_shape):
    """Extract the mean RGB value of all skin pixels inside the landmark-defined ROI polygon.

    Returns a (3,) array [R, G, B], or None if the ROI contains fewer than
    100 pixels (e.g. due to occlusion or the face being too far from the camera).
    """
    mask = build_mask(img_shape, landmarks, roi_indices)
    pixels = frame_rgb[mask > 0]
    if len(pixels) < 100:
        return None
    return pixels.mean(axis=0)


def estimate_bpm(rgb_window, fps, hr_band=(0.8, 3.0)):
    """Estimate heart rate in BPM from a window of mean RGB traces using FastICA.

    Pipeline:
      1. Per-channel normalisation by the temporal mean (removes DC offset differences).
      2. Linear detrend to suppress slow illumination drift.
      3. 4th-order Butterworth bandpass filter in the cardiac band (default 0.8–3.0 Hz).
      4. Z-score normalisation so all channels have equal variance before ICA.
      5. FastICA with the green channel as the initial weight direction (strongest PPG signal).
      6. Welch PSD on each independent component; the component whose in-band peak
         has the highest power is selected.

    Args:
        rgb_window: Array of shape (N, 3) containing consecutive per-frame mean RGB values.
        fps: Actual capture frame rate in Hz.
        hr_band: Frequency band (Hz) corresponding to valid heart rates (default 48–180 BPM).

    Returns:
        Estimated heart rate in BPM, or None if estimation fails.
    """
    # per-channel normalisation: make channels comparable by removing DC differences
    sig = rgb_window / (rgb_window.mean(axis=0) + 1e-8)

    sig = detrend(sig, axis=0)
    lo, hi = np.array(hr_band) / (fps / 2)
    b, a = butter(4, [lo, hi], btype='band')
    sig = filtfilt(b, a, sig, axis=0)
    sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-8)

    # initialise ICA with the green channel first: converges faster toward the PPG component
    w_init = np.eye(3)[[1, 0, 2]]   # shape (3, 3): green, red, blue
    S = FastICA(n_components=3, max_iter=500, w_init=w_init,
                whiten='unit-variance').fit_transform(sig)

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
    # ==================== INITIALISATION ====================
    cap = cv2.VideoCapture(0)
    capture_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rgb_buffer  = deque(maxlen=int(BPM_WINDOW_S * capture_fps))
    frame_count = 0
    last_bpm    = None
    bpm_history = deque(maxlen=5)   # recent BPM estimates used to reject spurious spikes

    startup_time = time.time()

    detector = EyeClosureDetector()
    microsleep = MicrosleepDetector()
    owl_detector    = DistractionDetector()
    lizard_detector = DistractionDetector()
    calibration_samples  = []
    owl_yaw_samples      = []
    lizard_gaze_samples  = []
    owl_threshold        = 0.25   # overwritten after calibration
    lizard_threshold     = 0.15   # overwritten after calibration

    debug_mode = "--debug" in sys.argv
    if debug_mode:
        print("Debug mode enabled")

    video_writer = None
    _frame_times: list[float] = []   # timestamps used to measure actual processing FPS
    _WARMUP = 30                     # number of frames to collect before opening the writer

    # ==================== MAIN LOOP ====================
    while cap.isOpened():

        success, image = cap.read()

        start = time.time()

        if image is None:
            break

        image = cv2.flip(image, 1)                           # mirror horizontally
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.time() * 1000)
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        img_h, img_w, _ = image.shape

        _frame_times.append(time.time())
        if video_writer is None and len(_frame_times) >= _WARMUP:
            real_fps = (_WARMUP - 1) / (_frame_times[-1] - _frame_times[0])
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter("dms_output.avi", fourcc, real_fps, (img_w, img_h))

        # ==================== LANDMARK OVERLAY ====================
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                for idx, lm in enumerate(face_landmarks):
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    if idx in LEFT_EYE or idx in RIGHT_EYE:
                        cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

                    if idx in LEFT_IRIS or idx in RIGHT_IRIS:
                        cv2.circle(image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

                    if idx in NOSE_TIP:
                        cv2.circle(image, (x, y), radius=2, color=(255, 0, 0), thickness=-1)

                r_ap = eyelid_aperture(face_landmarks, RIGHT_OUTER, RIGHT_INNER,
                                       RIGHT_UPPER_LID, RIGHT_LOWER_LID, img_w, img_h)
                l_ap = eyelid_aperture(face_landmarks, LEFT_OUTER, LEFT_INNER,
                                       LEFT_UPPER_LID, LEFT_LOWER_LID, img_w, img_h)
                aperture = np.mean([a for a in [r_ap, l_ap] if a is not None]) if any(
                    a is not None for a in [r_ap, l_ap]) else None

                elapsed = time.time() - startup_time
                calibrate_eye_detector(detector, calibration_samples, aperture, elapsed)

                if elapsed < CALIBRATION_SECONDS:
                    owl_yaw_samples.append(owl_yaw(face_landmarks))
                    lizard_gaze_samples.append(lizard_gaze(face_landmarks))
                elif owl_yaw_samples:
                    # threshold = median + max(k*std, fixed_floor) — robust to small head movements during calibration
                    owl_threshold    = float(np.median(owl_yaw_samples)    + max(3 * np.std(owl_yaw_samples), 0.12))
                    lizard_threshold = float(np.median(lizard_gaze_samples) + max(np.std(lizard_gaze_samples), 0.03))
                    owl_yaw_samples.clear()
                    lizard_gaze_samples.clear()
                    if debug_mode:
                        print(f"Gaze calibration: owl_thr={owl_threshold:.3f}  lizard_thr={lizard_threshold:.3f}")

                eyes_closed = bool(detector.is_closed(aperture))
                ms_state = microsleep.update(eyes_closed, time.time())

                now = time.time()
                owl_dist    = is_owl_distracted(face_landmarks, owl_threshold)
                # lizard is only evaluated when the head is not already turned (owl takes priority)
                lizard_dist = not owl_dist and is_lizard_distracted(face_landmarks, lizard_threshold)

                owl_state    = owl_detector.update(owl_dist, now)
                lizard_state = lizard_detector.update(lizard_dist, now)

                # rPPG: accumulate forehead RGB mean and recompute BPM every N frames
                rgb_mean = extract_roi_rgb(image_rgb, face_landmarks, FOREHEAD_ROI, (img_h, img_w))
                if rgb_mean is not None:
                    rgb_buffer.append(rgb_mean)

                frame_count += 1
                if len(rgb_buffer) == rgb_buffer.maxlen and frame_count % BPM_UPDATE_EVERY_N == 0:
                    raw_bpm = estimate_bpm(np.array(rgb_buffer), capture_fps)
                    if raw_bpm is not None:
                        med = np.median(bpm_history) if bpm_history else raw_bpm
                        if not bpm_history or abs(raw_bpm - med) <= 20:
                            bpm_history.append(raw_bpm)
                            last_bpm = np.mean(bpm_history)

                label, color = driver_state_label(ms_state, microsleep._recovery_from, owl_state, lizard_state)
                draw_status(image, label, color)
                if debug_mode:
                    pts = np.array([[int(face_landmarks[i].x * img_w),
                                     int(face_landmarks[i].y * img_h)]
                                    for i in FOREHEAD_ROI], dtype=np.int32)
                    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=1)

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
            totalTime = end - start
            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f'Time: {(time.time()-startup_time):.2f}s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('output window', image)

        if video_writer is not None:
            video_writer.write(image)

        if cv2.waitKey(5) & 0xFF == 27:   # ESC to quit
            break

    # ==================== CLEANUP ====================
    cap.release()
    if video_writer is not None:
        video_writer.release()
    face_landmarker.close()

def calibrate_eye_detector(detector, calibration_samples, aperture, elapsed):
    """Collect aperture samples during the calibration window and trigger baseline fitting.

    Samples are accumulated while elapsed < CALIBRATION_SECONDS.  On the first
    call after the window closes, EyeClosureDetector.calibrate() is invoked with
    all collected samples.
    """
    if elapsed < CALIBRATION_SECONDS:
        if aperture is not None:
            calibration_samples.append(aperture)
    elif detector.baseline_aperture is None and calibration_samples:
        detector.calibrate(calibration_samples)
        print(f"Calibration complete: baseline={detector.baseline_aperture:.1f}px")


if __name__ == "__main__":
    main()

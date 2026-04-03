import os
import cv2
import numpy as np

def calculate_threshold(img_gray: np.ndarray, percentile: float = 97.0) -> float:
    """Soglia basata su percentile: tiene solo i pixel più luminosi,
    cioè le lane markings."""
    return np.percentile(img_gray[img_gray > 0], percentile)


def binarize(img_gray: np.ndarray, th: float) -> np.ndarray:
    """Binarizzazione: pixel > soglia → 255, altrimenti 0."""
    return np.where(img_gray > th, np.uint8(255), np.uint8(0))

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


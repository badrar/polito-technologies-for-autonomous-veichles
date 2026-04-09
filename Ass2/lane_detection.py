import numpy as np

def calculate_threshold(img_gray: np.ndarray, percentile: float = 97.0) -> float:
    """Soglia basata su percentile: tiene solo i pixel più luminosi,
    cioè le lane markings."""
    return np.percentile(img_gray[img_gray > 0], percentile)


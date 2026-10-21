import cv2
import numpy as np
from typing import Tuple

def validate_radiograph(image_np: np.ndarray) -> Tuple[bool, str]:
    """
    Performs input validation guardrails:
    1. Sharpness via Laplacian Variance (detects extreme blur).
    2. Color profile check (ensures grayscale/X-ray profile, rejecting standard RGB photos).
    """
    # 1. Blur Detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 40.0:
        return False, f"Image quality check failed: High blur detected (Laplacian Variance: {lap_var:.1f}). Please upload a clear radiograph."

    # 2. Color Divergence Check (Chest X-rays are monochrome)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
        color_diff = np.mean(np.abs(r.astype(int) - g.astype(int))) + np.mean(np.abs(g.astype(int) - b.astype(int)))
        if color_diff > 35.0:
            return False, "Input validation error: Non-monochrome image detected. Please upload a valid Chest X-Ray radiograph."

    return True, "Valid Radiograph"
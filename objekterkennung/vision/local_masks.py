import cv2
import numpy as np


def component_mask(mask: np.ndarray, contour: np.ndarray) -> np.ndarray:
    point = np.rint(contour.reshape(-1, 2)[0]).astype(int)
    if not (0 <= point[0] < mask.shape[1] and 0 <= point[1] < mask.shape[0]):
        return np.zeros_like(mask)
    # Select actual connected pixels instead of filling the contour and its holes.
    _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    label = labels[point[1], point[0]]
    return (labels == label).astype(np.uint8) * 255 if label else np.zeros_like(mask)


def replace_component(mask: np.ndarray, contour: np.ndarray,
                      foreground: np.ndarray, origin_xy: tuple[int, int]) -> None:
    x, y, width, height = cv2.boundingRect(contour)
    old = mask[y:y + height, x:x + width]
    support = component_mask(old, contour - (x, y))
    old[support > 0] = 0
    left, top = origin_xy
    mask[top:top + foreground.shape[0], left:left + foreground.shape[1]] |= foreground

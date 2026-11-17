from dataclasses import dataclass, field

import cv2
import numpy as np

from objekterkennung.vision.local_masks import component_mask


@dataclass(frozen=True)
class SurfaceRecovery:
    contour: np.ndarray
    accepted: bool
    reason: str
    foreground_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    origin_xy: tuple[int, int] | None = None


def recover_surface(frame: np.ndarray, contour: np.ndarray, color: str,
                    masks: dict[str, np.ndarray], hue_ranges, *,
                    blocked_mask: np.ndarray | None = None) -> SurfaceRecovery:
    def unchanged(reason):
        return SurfaceRecovery(contour, False, reason)

    x, y, width, height = cv2.boundingRect(contour)
    if min(width, height) < 20:
        return unchanged("Zu kleine Farbstützung für die Seitenergänzung.")
    padding = round(1.5 * max(width, height))
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(frame.shape[1], x + width + padding), min(frame.shape[0], y + height + padding)
    if (right - left) * (bottom - top) > 250_000:
        return unchanged("Suchbereich für die Seitenergänzung zu groß.")
    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    offset = np.array([left, top], np.int32)
    source = masks[color][top:bottom, left:right]
    seed = component_mask(source, contour - offset)
    core = seed > 0
    hard_background = (source > 0) & ~core
    for other, mask in masks.items():
        if other != color:
            hard_background |= mask[top:bottom, left:right] > 0
    if blocked_mask is not None:
        hard_background |= blocked_mask[top:bottom, left:right] > 0
    core &= ~hard_background
    if np.count_nonzero(core) < 30:
        return unchanged("Zu wenige Farbpixel für die Seitenergänzung.")
    hue = hsv[:, :, 0].astype(float)
    peak = int(np.argmax(np.bincount(hue[core].astype(int), minlength=180)))
    center = (peak + float(np.median((hue[core] - peak + 90) % 180 - 90))) % 180
    distance = np.abs((hue - center + 90) % 180 - 90)
    tolerance = float(np.clip(np.quantile(distance[core], .95) + 3, 20, 25))
    hue_allowed = np.zeros(hue.shape, bool)
    for lower, upper in hue_ranges:
        hue_allowed |= (hue >= lower[0]) & (hue <= upper[0])
    allowed = (hue_allowed & (distance <= tolerance)
               & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 25))
    allowed[core] = True
    allowed[hard_background] = False
    # Grow only through compatible pixels connected to the surviving object core.
    _, components = cv2.connectedComponents(allowed.astype(np.uint8))
    labels = np.unique(components[core])
    grown = np.isin(components, labels[labels > 0]).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, kernel)
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, kernel)
    # Preserve hard exclusions even when morphological closing bridges gaps.
    grown[hard_background] = 0
    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return unchanged("Keine zusammenhängende Seitenregion gefunden.")
    selected = max(contours, key=lambda candidate: np.count_nonzero(
        (component_mask(grown, candidate) > 0) & core))
    a, b, w, h = cv2.boundingRect(selected)
    if a <= 0 or b <= 0 or a + w >= grown.shape[1] or b + h >= grown.shape[0]:
        return unchanged("Seitenergänzung läuft an die Suchgrenze.")
    area = cv2.contourArea(selected)
    hull_area = cv2.contourArea(cv2.convexHull(selected))
    old_area = cv2.contourArea(contour)
    if (not 1.1 * old_area <= area <= 3.5 * old_area or w > 1.35 * width or h > 3 * height
            or hull_area <= 0 or area / hull_area < .93):
        return unchanged("Seitenergänzung passt nicht zu einer begrenzten, geschlossenen Körperkontur.")
    selected_mask = component_mask(grown, selected)
    if np.count_nonzero((selected_mask > 0) & core) < .95 * np.count_nonzero(core):
        return unchanged("Seitenergänzung verliert die ursprüngliche Farbstützung.")
    silhouette = np.zeros_like(selected_mask)
    cv2.drawContours(silhouette, [selected], -1, 255, -1)
    neighbor = cv2.dilate(selected_mask, np.ones((3, 3), np.uint8)) > 0
    if np.any(((silhouette > 0) | neighbor) & (source > 0) & (seed == 0)):
        return unchanged("Seitenergänzung würde einen anderen gleichfarbigen Kandidaten übernehmen.")
    return SurfaceRecovery(selected + offset, True,
        "Schwächere Körperseite durch lokales, farbgestütztes Regionenwachstum ergänzt.",
        selected_mask, (left, top))

from dataclasses import dataclass, field

import cv2
import numpy as np

from objekterkennung.vision.local_masks import component_mask


@dataclass(frozen=True)
class RegionRefinement:
    contour: np.ndarray
    accepted: bool
    reason: str
    foreground_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    origin_xy: tuple[int, int] | None = None


def refine_region(frame: np.ndarray, hsv: np.ndarray, contour: np.ndarray,
                  seed_masks: dict[str, np.ndarray], color: str, hue_ranges,
                  *, blocked_mask: np.ndarray | None = None) -> RegionRefinement:
    def unchanged(reason):
        return RegionRefinement(contour, False, reason)

    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0
    x, y, width, height = cv2.boundingRect(contour)
    if solidity >= 0.98:
        return unchanged("HSV-Kontur ausreichend geschlossen; keine Ergänzung nötig.")
    if min(width, height) < 20 or area < 100:
        return unchanged("Zu wenig Farbstützung für eine Konturergänzung.")
    image_height, image_width = frame.shape[:2]
    if x <= 0 or y <= 0 or x + width >= image_width or y + height >= image_height:
        return unchanged("Am Bildrand wird keine vollständige Kontur ergänzt.")
    padding = max(12, round(0.30 * max(width, height)))
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(image_width, x + width + padding), min(image_height, y + height + padding)
    if (right - left) * (bottom - top) > 250_000:
        return unchanged("Suchbereich für die lokale Ergänzung zu groß.")
    roi = frame[top:bottom, left:right]
    local_hsv = hsv[top:bottom, left:right]
    local_contour = contour - np.array([left, top], np.int32)
    source_seed = seed_masks[color][top:bottom, left:right]
    original = component_mask(source_seed, local_contour)
    other_seed = (source_seed > 0) & (original == 0)


    closed_seed = solidity >= 0.95
    core_radius = max(1, round(0.075 * min(width, height))) if closed_seed else 1
    core = cv2.erode(original, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * core_radius + 1, 2 * core_radius + 1)))
    chroma = local_hsv[:, :, 1].astype(float) * local_hsv[:, :, 2] / 255
    if blocked_mask is not None:


        core[(local_hsv[:, :, 1] < 150) | (chroma < 25)] = 0
    if np.count_nonzero(core) < 30:
        return unchanged("Zu wenige sichere Farbpixel für eine Konturergänzung.")

    labels = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
    labels[y - top:y - top + height, x - left:x - left + width] = cv2.GC_PR_FGD


    margin_x, margin_y = max(3, round(0.10 * width)), max(3, round(0.10 * height))
    allowed = np.zeros(roi.shape[:2], bool)
    allowed[max(0, y - top - margin_y):min(roi.shape[0], y - top + height + margin_y),
            max(0, x - left - margin_x):min(roi.shape[1], x - left + width + margin_x)] = True
    labels[~allowed] = cv2.GC_BGD


    hue = local_hsv[:, :, 0].astype(int)
    compatible = np.zeros(hue.shape, bool)
    for lower, upper in hue_ranges:
        compatible |= (hue - (lower[0] - 5)) % 180 <= upper[0] - lower[0] + 10


    incompatible = (~compatible) & (local_hsv[:, :, 1] >= 60) & (chroma >= 15)
    # Protect other instances and explicit exclusions as definite GrabCut background.
    hard_background = incompatible | other_seed | ~allowed
    for other_color, mask in seed_masks.items():
        if other_color != color:
            hard_background |= mask[top:bottom, left:right] > 0
    labels[core > 0] = cv2.GC_FGD
    if blocked_mask is not None:
        hard_background |= blocked_mask[top:bottom, left:right] > 0
    hard_background[0, :] = hard_background[-1, :] = True
    hard_background[:, 0] = hard_background[:, -1] = True
    labels[hard_background] = cv2.GC_BGD
    try:


        cv2.setRNGSeed(0)
        cv2.grabCut(roi, labels, None, np.zeros((1, 65)), np.zeros((1, 65)),
                    3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return unchanged("Lokale Vordergrundtrennung fehlgeschlagen; HSV-Kontur bleibt erhalten.")
    foreground = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    # Reapply exclusions after morphology, which can fill previously excluded pixels.
    foreground[hard_background] = 0
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return unchanged("Keine gestützte ergänzte Kontur gefunden.")

    def seed_overlap(candidate):
        mask = component_mask(foreground, candidate)
        return np.count_nonzero((mask > 0) & (core > 0))

    # Follow the original color core rather than the largest nearby foreground region.
    selected = max(contours, key=seed_overlap)
    if seed_overlap(selected) < 0.90 * np.count_nonzero(core):
        return unchanged("Ergänzung enthält zu wenig der sicheren Farbstützung.")
    a, b, w, h = cv2.boundingRect(selected)
    if a <= 0 or b <= 0 or a + w >= roi.shape[1] or b + h >= roi.shape[0]:
        return unchanged("Ergänzung erreicht die Suchgrenze und ist nicht verlässlich begrenzt.")
    out_area = cv2.contourArea(selected)
    out_hull = cv2.contourArea(cv2.convexHull(selected))
    out_solidity = out_area / out_hull if out_hull > 0 else 0


    circularity = 4 * np.pi * area / cv2.arcLength(contour, True) ** 2
    out_circularity = 4 * np.pi * out_area / cv2.arcLength(selected, True) ** 2
    if closed_seed:
        quality_ok = (out_solidity >= max(0.95, solidity - 0.015)
                      and (out_solidity >= solidity + 0.005 or out_circularity >= circularity + 0.03))
    else:
        quality_ok = out_solidity >= max(0.93, solidity + 0.03)
    max_area_factor = 1.25 if closed_seed else 2.7
    if (not 0.8 * area <= out_area <= max_area_factor * area or w > 1.25 * width or h > 1.25 * height
            or not quality_ok):
        return unchanged(f"Ergänzung verworfen: Solidity {solidity:.3f} auf {out_solidity:.3f}, "
                         f"Flächenfaktor {out_area / area:.2f}, Rahmenfaktoren {w / width:.2f}/{h / height:.2f}.")
    selected_mask = component_mask(foreground, selected)
    silhouette = np.zeros_like(selected_mask)
    cv2.drawContours(silhouette, [selected], -1, 255, -1)
    neighbor = cv2.dilate(selected_mask, np.ones((3, 3), np.uint8)) > 0
    if np.any(((silhouette > 0) | neighbor) & other_seed):
        return unchanged("Ergänzung würde einen anderen gleichfarbigen Kandidaten übernehmen.")
    result = selected + np.array([left, top], np.int32)
    action = "Farbrand lokal verfeinert" if closed_seed else "Dunkle Bereiche lokal ergänzt"
    return RegionRefinement(result, True,
        f"{action} (Solidity {solidity:.3f} auf {out_solidity:.3f}; "
        f"Rundheit {circularity:.3f} auf {out_circularity:.3f}).", selected_mask, (left, top))

import cv2
import numpy as np

from objekterkennung.vision.local_masks import component_mask
from objekterkennung.vision.region_refinement import refine_region
from objekterkennung.vision.shapes import classify_shape
from objekterkennung.vision.touching_separation import SeparationResult


def _contact_lines(gray: np.ndarray) -> list[tuple[float, float]]:
    height, width = gray.shape
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return []
    hypotheses = []
    for a, b, aa, bb in lines.reshape(-1, 4):
        if abs(bb - b) < .55 * height or abs(aa - a) > .15 * abs(bb - b):
            continue
        slope = float((aa - a) / (bb - b))
        center = float(a + slope * (height / 2 - b))
        if .15 * width < center < .85 * width:
            hypotheses.append((center, slope))
    if len(hypotheses) > 24:
        return []
    groups = []
    for center, slope in sorted(hypotheses):
        if groups and abs(center - np.mean([p[0] for p in groups[-1]])) <= max(3, .04 * height):
            groups[-1].append((center, slope))
        else:
            groups.append([(center, slope)])
    if len(groups) > 4:
        return []
    return [(float(np.mean([p[0] for p in group])), float(np.mean([p[1] for p in group])))
            for group in groups]


def _seam_evidence(value: np.ndarray, support: np.ndarray, center: float, slope: float) -> dict:
    height, width = value.shape
    rows = np.arange(round(.1 * height), round(.9 * height))
    columns = np.rint(center + slope * (rows - height / 2)).astype(int)
    band = max(2, round(.025 * height))
    flank = max(band + 2, round(.065 * height))
    if np.any(columns - flank - 1 < 0) or np.any(columns + flank + 1 >= width):
        return {"accepted": False, "reason": "Kontaktprofil zu nahe am Außenrand."}
    middle = np.min([value[rows, columns + k] for k in range(-band, band + 1)], axis=0)
    left = np.median([value[rows, columns - flank + k] for k in (-1, 0, 1)], axis=0)
    right = np.median([value[rows, columns + flank + k] for k in (-1, 0, 1)], axis=0)
    supported = (support[rows, columns - flank] > 0) & (support[rows, columns + flank] > 0)
    contrast = np.minimum(left, right).astype(float) - middle
    dark = supported & (contrast >= np.maximum(8., .05 * np.minimum(left, right)))
    blocks = [float(np.mean(block)) for block in np.array_split(dark, 4)]
    accepted = np.mean(supported) >= .9 and np.mean(dark) >= .7 and sum(v >= .6 for v in blocks) >= 3
    return {"accepted": bool(accepted), "bilateral_support_fraction": float(np.mean(supported)),
            "dark_valley_fraction": float(np.mean(dark)), "dark_valley_blocks": blocks,
            "median_valley_contrast": float(np.median(contrast))}


def split_contact(frame: np.ndarray, contour: np.ndarray, masks: dict[str, np.ndarray],
                  color: str, hue_ranges, min_area_px: float, *, refine: bool = False,
                  blocked_mask: np.ndarray | None = None) -> SeparationResult:
    x, y, width, height = cv2.boundingRect(contour)
    report = {"bbox_xywh": [x, y, width, height], "accepted": False,
              "method": "contact_seam", "markers_px": [], "parts": [],
              "retained_fraction": None, "hypotheses": []}

    def unchanged(reason):
        report["reason"] = reason
        return SeparationResult([contour], report)

    if classify_shape(contour).label != "unknown":
        return unchanged("Bereits erkannter Einzelkörper bleibt ungeteilt.")
    if min(width, height) < 30 or not 1.2 <= width / height <= 3 or cv2.contourArea(contour) < 2 * min_area_px:
        return unchanged("Keine breite, ausreichend große Gruppe aufrechter Körper.")
    if x <= 0 or y <= 0 or x + width >= frame.shape[1] or y + height >= frame.shape[0]:
        return unchanged("Angeschnittene Region bleibt unverändert.")
    padding = max(12, round(.3 * max(width, height)))
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(frame.shape[1], x + width + padding), min(frame.shape[0], y + height + padding)
    if (right - left) * (bottom - top) > 250_000:
        return unchanged("Kontaktprüfung überschreitet das lokale Rechenbudget.")
    roi = frame[top:bottom, left:right]
    local_contour = contour - (left, top)
    local_masks = {k: mask[top:bottom, left:right].copy() for k, mask in masks.items()}
    source = component_mask(local_masks[color], local_contour)
    local_x, local_y = x - left, y - top
    box = np.s_[local_y:local_y + height, local_x:local_x + width]
    source_pixels = np.count_nonzero(source)
    if not source_pixels:
        return unchanged("Keine zugehörige Farbstützung.")
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lines = _contact_lines(cv2.cvtColor(roi[box], cv2.COLOR_BGR2GRAY))
    silhouette = np.zeros_like(source)
    cv2.drawContours(silhouette, [local_contour], -1, 255, -1)
    hard = (silhouette > 0) & (source == 0)
    hard |= (local_masks[color] > 0) & (source == 0)
    for other, mask in local_masks.items():
        if other != color:
            hard |= mask > 0
    if blocked_mask is not None:
        hard |= blocked_mask[top:bottom, left:right] > 0
    allowed_box = np.zeros_like(source, bool)
    allowed_box[box] = True
    hard |= ~allowed_box
    yy, xx = np.indices(source.shape)
    accepted = []
    for center, slope in lines:
        evidence = _seam_evidence(hsv[box][:, :, 2], source[box], center, slope)
        attempt = {"line_midpoint_px": [float(x + center), float(y + height / 2)],
                   "slope": slope, "evidence": evidence, "accepted": False}
        report["hypotheses"].append(attempt)
        if not evidence["accepted"]:
            continue
        signed = xx - (local_x + center + slope * (yy - local_y - height / 2))
        sides = [signed < -1, signed > 1]
        seeds = [((source > 0) & side).astype(np.uint8) * 255 for side in sides]
        seed_masks = {**local_masks, color: seeds[0] | seeds[1]}
        parts, outlines, measurements = [], [], []
        for side, seed in zip(sides, seeds):
            found = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if not found:
                break
            outline = max(found, key=cv2.contourArea)
            part = component_mask(seed, outline)
            if np.count_nonzero(part) < .95 * np.count_nonzero(seed):
                break
            blocked = (hard | ~side).astype(np.uint8) * 255
            if refine:
                correction = refine_region(roi, hsv, outline, seed_masks, color, hue_ranges,
                                           blocked_mask=blocked)
                if correction.accepted:
                    part = np.zeros_like(source)
                    a, b = correction.origin_xy
                    part[b:b + correction.foreground_mask.shape[0], a:a + correction.foreground_mask.shape[1]] = correction.foreground_mask
                    outline = correction.contour
            prediction = classify_shape(outline)
            fraction = np.count_nonzero(part & source) / source_pixels
            area = cv2.contourArea(outline)
            if (prediction.label == "unknown" or area < min_area_px or not .12 <= fraction <= .85
                    or prediction.features.solidity < .93 or np.any(part & blocked)):
                break
            parts.append(part)
            outlines.append(outline + (left, top))
            measurements.append({"shape": prediction.label, "area_px": area,
                                 "solidity": prediction.features.solidity,
                                 "source_fraction": fraction})
        if len(parts) != 2:
            attempt["reason"] = "Keine zwei vollständigen, plausiblen Körperkonturen."
            continue
        union = parts[0] | parts[1]
        retained = np.count_nonzero(union & source) / source_pixels
        if retained < .88 or cv2.connectedComponents(union)[0] != 3:
            attempt["reason"] = "Zu viel Stützung verloren oder Instanzen nicht getrennt."
            continue
        attempt.update(accepted=True, retained_fraction=float(retained))
        accepted.append((outlines, parts, measurements, float(retained)))
    if len(accepted) != 1:
        return unchanged("Keine eindeutige, beidseitig gestützte Kontaktfuge mit plausiblen Körpern.")
    outlines, parts, measurements, retained = accepted[0]
    report.update(accepted=True, parts=measurements, retained_fraction=retained,
                  reason="Körper durch geprüfte dunkle Kontaktfuge und getrennt geschützte Konturen getrennt.")
    return SeparationResult(outlines, report, source[box], [part[box] for part in parts])

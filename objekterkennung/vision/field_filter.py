from dataclasses import replace

import cv2
import numpy as np

from objekterkennung.geometry.calibration import PlaneCalibration
from objekterkennung.vision.detection import DetectionResult
from objekterkennung.vision.segmentation import Candidate


def _touches_field(candidate: Candidate, calibration: PlaneCalibration) -> bool:
    polygon = np.rint(calibration.corners_px).astype(np.int32)
    x, y, width, height = candidate.bbox_xywh
    fx, fy, fw, fh = cv2.boundingRect(polygon)
    left, top = max(x, fx), max(y, fy)
    right, bottom = min(x + width, fx + fw), min(y + height, fy + fh)
    if right <= left or bottom <= top:
        return False
    offset = np.array([left, top], np.int32)
    field = np.zeros((bottom - top, right - left), np.uint8)
    silhouette = np.zeros_like(field)
    cv2.fillPoly(field, [polygon - offset], 255)
    cv2.drawContours(silhouette, [np.rint(candidate.contour).astype(np.int32) - offset], -1, 255, -1)
    return bool(np.any((field > 0) & (silhouette > 0)))


def filter_to_field(result: DetectionResult, *, enabled: bool = True,
                    calibration: PlaneCalibration | None = None) -> DetectionResult:
    if not enabled or result.positions is None:
        return result
    if not len(result.segmentation.candidates) == len(result.shapes) == len(result.positions):
        raise ValueError("Flächenfilter benötigt zueinander passende Kandidaten, Formen und Positionen.")
    # Prefer ground positions; silhouette contact is a fallback when position is unavailable.
    indices = [i for i, (candidate, position) in enumerate(zip(result.segmentation.candidates, result.positions))
               if position.status != "outside" and
               (position.position_norm is not None or calibration is None or _touches_field(candidate, calibration))]
    if len(indices) == len(result.positions):
        return result
    return replace(result,
        segmentation=replace(result.segmentation,
            candidates=[result.segmentation.candidates[i] for i in indices]),
        shapes=[result.shapes[i] for i in indices],
        positions=[result.positions[i] for i in indices])


def field_filter_summary(result: DetectionResult, *, enabled: bool = True,
                         calibration: PlaneCalibration | None = None,
                         visible_result: DetectionResult | None = None) -> str:
    if not enabled:
        return "Flächenfilter aus: alle Kandidaten sichtbar."
    if result.positions is None:
        return "Flächenfilter inaktiv: keine Kalibrierung."
    view = visible_result if visible_result is not None else filter_to_field(result, calibration=calibration)
    outside = sum(position.status == "outside" for position in result.positions)
    no_contact = len(result.shapes) - len(view.shapes) - outside
    unknown = sum(position.position_norm is None for position in view.positions)
    return (f"Flächenfilter an: {outside} außerhalb ausgeblendet; "
            f"{no_contact} ohne Flächenkontakt ausgeblendet; "
            f"{unknown} ohne sichere Flächenzuordnung weiterhin sichtbar.")

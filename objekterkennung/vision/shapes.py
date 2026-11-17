from dataclasses import dataclass
import math

import cv2
import numpy as np


SHAPE_NAMES = ("cube", "cylinder", "cone", "hemisphere")
RULES_VERSION = "2.2"
APPROX_EPSILON_RATIO = 0.02
MIN_SIDE_PX = 20
MIN_SOLIDITY = 0.93


@dataclass(frozen=True)
class ShapeFeatures:

    width_px: int
    height_px: int
    height_width_ratio: float
    extent: float
    circularity: float
    solidity: float
    polygon_vertices: int
    upper_width_ratio: float
    polygon_area_loss: float = 0.0


@dataclass(frozen=True)
class ShapePrediction:

    label: str
    reason: str
    features: ShapeFeatures | None
    temporally_confirmed: bool = False


def extract_features(contour: np.ndarray) -> ShapeFeatures:
    if (not isinstance(contour, np.ndarray) or contour.ndim not in (2, 3)
            or contour.shape[-1] != 2 or (contour.ndim == 3 and contour.shape[1] != 1)
            or contour.shape[0] < 3 or contour.dtype.kind not in "iuf"
            or not np.isfinite(contour).all()):
        raise ValueError("Eine endliche Kontur mit mindestens drei 2D-Punkten ist erforderlich.")
    points = contour.astype(np.float32).reshape(-1, 1, 2)
    area = cv2.contourArea(points)
    perimeter = cv2.arcLength(points, True)
    if area <= 0 or perimeter <= 0:
        raise ValueError("Die Kontur hat keine messbare Fläche.")
    x, y, width, height = cv2.boundingRect(points)
    hull_area = cv2.contourArea(cv2.convexHull(points))
    polygon = cv2.approxPolyDP(points, APPROX_EPSILON_RATIO * perimeter, True)


    local = np.rint(points - np.array([x, y], dtype=np.float32)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [local], -1, 255, -1)
    row_widths = np.count_nonzero(mask, axis=1)
    start = int(0.10 * height)
    stop = max(start + 1, int(0.30 * height))
    upper_width = float(np.mean(row_widths[start:stop])) / width
    return ShapeFeatures(width, height, height / width, area / (width * height),
                         4 * math.pi * area / (perimeter * perimeter), area / hull_area,
                         len(polygon), upper_width,
                         max(0.0, 1 - cv2.contourArea(polygon) / area))


def classify_shape(contour: np.ndarray) -> ShapePrediction:
    try:
        features = extract_features(contour)
    except ValueError as error:
        return ShapePrediction("unknown", str(error), None)
    f = features
    if min(f.width_px, f.height_px) < MIN_SIDE_PX:
        return ShapePrediction("unknown", "Zu wenig Bildauflösung für die Formregeln.", f)
    if f.solidity < MIN_SOLIDITY:
        return ShapePrediction("unknown", "Stark eingezogene Kontur; Berührung oder Segmentierungsfehler möglich.", f)


    # Rounded silhouettes can have six polygon vertices; area loss helps distinguish cylinders.
    rounded_six = f.polygon_vertices == 6 and f.polygon_area_loss >= .06
    cube_corners = ((4 <= f.polygon_vertices <= 6 and not rounded_six)
                    or (f.polygon_vertices == 7 and f.upper_width_ratio <= 0.90))
    rules = [
        ("hemisphere", "Breite, runde Silhouette ohne starke obere Verjüngung.", f.polygon_vertices >= 7,
         {"height_width_ratio": (.75, 1.0), "circularity": (.80, math.inf),
          "extent": (.68, .84), "upper_width_ratio": (.60, .85)}),
        ("cone", "Obere Silhouette deutlich schmaler als die Gesamtbreite.", f.polygon_vertices >= 4,
         {"height_width_ratio": (.90, 1.30), "extent": (.50, .74), "upper_width_ratio": (.20, .58)}),
        ("cube", "Wenige ausgeprägte Konturecken und keine kegelförmige Verjüngung.", cube_corners,
         {"height_width_ratio": (.95, 1.60), "extent": (.55, 1.0), "upper_width_ratio": (.55, math.inf)}),
        ("cylinder", "Hohe, oben breite Silhouette mit gekrümmten Konturabschnitten.",
         7 <= f.polygon_vertices <= 12 or rounded_six,
         {"height_width_ratio": (1.12, 1.55), "extent": (.70, .90),
          "upper_width_ratio": (.75, math.inf), "circularity": (.75, math.inf)}),
    ]


    # Allow one narrowly missed feature to account for pixel-grid quantization.
    tolerance = {"height_width_ratio": (1 + f.height_width_ratio) / max(1, f.width_px - 1),
                 "extent": min(.03, 1 / f.width_px + 1 / f.height_px),
                 "upper_width_ratio": 1 / f.width_px, "circularity": .02}
    matches, near = [], []
    for label, reason, corners_ok, intervals in rules:
        if not corners_ok:
            continue
        costs = [max(low - getattr(f, name), getattr(f, name) - high, 0) / tolerance[name]
                 for name, (low, high) in intervals.items()]
        if not any(costs):
            matches.append((label, reason))
        elif sum(cost > 0 for cost in costs) == 1 and max(costs) <= 1:
            near.append((label, reason))
    if not matches and len(near) == 1 and f.solidity >= .97:
        label, reason = near[0]
        return ShapePrediction(label, reason + " Ein Grenzmerkmal innerhalb begrenzter Rastertoleranz.", f)
    # Keep ambiguous classifications unknown instead of choosing the first matching rule.
    if len(matches) != 1:
        reason = "Keine Formregel passt." if not matches else "Mehrere Formregeln passen; Ergebnis mehrdeutig."
        return ShapePrediction("unknown", reason, f)
    label, reason = matches[0]
    return ShapePrediction(label, reason, f)

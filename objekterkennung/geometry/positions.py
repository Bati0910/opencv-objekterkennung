from objekterkennung.configuration import SETUP

from dataclasses import dataclass
import math

import cv2
import numpy as np

from objekterkennung.geometry.calibration import PlaneCalibration, SURFACE_SIZE_M


@dataclass(frozen=True)
class ObjectGeometry:

    cube_side_m: float
    cylinder_radius_m: float
    cone_radius_m: float
    hemisphere_radius_m: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            maximum = SURFACE_SIZE_M[0] if name == "cube_side_m" else SURFACE_SIZE_M[0] / 2
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not 0 < value <= maximum or not math.isfinite(value)):
                raise ValueError(f"Ungültiges Objektmaß {name}; positive Meterangabe innerhalb des Feldmaßes erforderlich.")


def load_geometry() -> ObjectGeometry:
    if set(SETUP.geometry) != set(ObjectGeometry.__dataclass_fields__):
        raise ValueError("Die Körpermaße enthalten fehlende oder unbekannte Felder.")
    return ObjectGeometry(**SETUP.geometry)


@dataclass(frozen=True)
class PositionEstimate:

    position_norm: tuple[float, float] | None
    ground_point_px: tuple[float, float] | None
    status: str
    method: str
    reason: str


def _away_from_camera(calibration: PlaneCalibration, ground: np.ndarray) -> np.ndarray:
    step = 1e-4
    # Estimate image-height changes along each ground-plane axis using central differences.
    derivatives = []
    for shift in (np.array([step, 0]), np.array([0, step])):
        before = calibration.normalized_to_pixel(tuple(ground - shift))[1]
        after = calibration.normalized_to_pixel(tuple(ground + shift))[1]
        derivatives.append((after - before) / (2 * step))
    gradient = np.array(derivatives)
    magnitude = np.linalg.norm(gradient)
    if not np.isfinite(magnitude) or magnitude < 1e-8:
        raise ValueError("Die Bildhöhenrichtung auf der Bodenebene ist nicht bestimmbar.")
    return -gradient / magnitude


def _round_center(contour: np.ndarray, radius_m: float, calibration: PlaneCalibration) -> np.ndarray:
    points = contour.reshape(-1, 2)
    bottom = points[points[:, 1] == points[:, 1].max()]

    pixel = (float(bottom[:, 0].min() + bottom[:, 0].max()) / 2, float(bottom[0, 1]))
    rim = np.array(calibration.pixel_to_normalized(pixel))
    radius_norm = radius_m / SURFACE_SIZE_M[0]
    # Move from the visible front rim toward the base center using the known radius.
    return rim + radius_norm * _away_from_camera(calibration, rim)


def _cube_center(contour: np.ndarray, side_m: float, calibration: PlaneCalibration) -> tuple[np.ndarray, int]:
    _, y, _, height = cv2.boundingRect(contour)
    polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    side_norm = side_m / SURFACE_SIZE_M[0]
    centers = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        if min(first[1], second[1]) < y + 0.45 * (height - 1):
            continue
        image_edge = second - first
        if abs(image_edge[0]) < 0.25 * np.linalg.norm(image_edge):
            continue
        a = np.array(calibration.pixel_to_normalized(tuple(first)))
        b = np.array(calibration.pixel_to_normalized(tuple(second)))
        edge = b - a
        length = np.linalg.norm(edge)
        if not 0.8 * side_norm <= length <= 1.2 * side_norm:
            continue
        midpoint = (a + b) / 2
        normal = np.array([-edge[1], edge[0]]) / length
        options = (midpoint + normal * side_norm / 2, midpoint - normal * side_norm / 2)
        image_heights = [calibration.normalized_to_pixel(tuple(point))[1] for point in options]
        if abs(image_heights[0] - image_heights[1]) < 1:
            continue

        # For this camera view, the inward offset projects higher in the image.
        centers.append(options[int(np.argmin(image_heights))])
    if not centers:
        raise ValueError("Keine ausreichend lange, sichtbare Grundkante des Würfels gefunden.")
    center = np.mean(centers, axis=0)
    disagreement_mm = max(np.linalg.norm(point - center) for point in centers) * (SURFACE_SIZE_M[0] * 1000)
    # Reject inconsistent center estimates from different visible base edges.
    if disagreement_mm > 3:
        raise ValueError("Die aus den Grundkanten geschätzten Mittelpunkte widersprechen sich.")
    return center, len(centers)


def estimate_position(contour: np.ndarray, shape: str, calibration: PlaneCalibration,
                      geometry: ObjectGeometry) -> PositionEstimate:
    method = "unavailable"
    try:
        if shape not in ("cube", "cylinder", "cone", "hemisphere"):
            raise ValueError("Ohne eindeutige Form ist kein Standflächenmodell verfügbar.")
        if (not isinstance(contour, np.ndarray) or contour.ndim not in (2, 3)
                or contour.shape[-1] != 2 or (contour.ndim == 3 and contour.shape[1] != 1)
                or contour.shape[0] < 3 or contour.dtype.kind not in "iuf"
                or not np.isfinite(contour).all()):
            raise ValueError("Keine gültige Kontur für die Positionsschätzung.")
        points = contour.astype(np.float32).reshape(-1, 1, 2)
        if cv2.contourArea(points) <= 0:
            raise ValueError("Keine messbare Standkontur.")
        width, height = calibration.image_size_px
        flat = points.reshape(-1, 2)
        if (flat <= 0).any() or (flat >= np.array([width - 1, height - 1])).any():
            raise ValueError("Die Kontur berührt den Bildrand; ihr Bodenbezug ist unvollständig.")
        if shape == "cube":
            center, edges = _cube_center(points, geometry.cube_side_m, calibration)
            method = "square_base_edges"
            reason = f"Standflächenmitte aus {edges} sichtbaren Grundkanten geschätzt."
        else:
            radius = getattr(geometry, f"{shape}_radius_m")
            center = _round_center(points, radius, calibration)
            method = "circular_base_rim"
            reason = "Standflächenmitte aus vorderem Bodenrand und bekanntem Radius geschätzt."
        normalized = tuple(float(value) for value in center)
        pixel = calibration.normalized_to_pixel(normalized)
        inside = all(-1e-6 <= value <= 1 + 1e-6 for value in normalized)
        return PositionEstimate(normalized, pixel, "estimated" if inside else "outside", method, reason)
    except ValueError as error:
        return PositionEstimate(None, None, "unavailable", method, str(error))

from objekterkennung.configuration import SURFACE_SIZE_M

from dataclasses import dataclass, field
import json
from pathlib import Path

import cv2
import numpy as np


CORNER_ORDER = ("front_left", "front_right", "back_right", "back_left")
UNIT_CORNERS = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)


def _transform_point(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    coordinates = np.asarray(point, dtype=np.float64)
    if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
        raise ValueError("Ein Punkt muss aus zwei endlichen Zahlen bestehen.")
    homogeneous = np.array([*coordinates, 1.0])
    projected = matrix @ homogeneous


    denominator_scale = np.abs(matrix[2]) @ np.abs(homogeneous)
    if (not np.isfinite(projected).all()
            or abs(projected[2]) <= 1e-6 * max(denominator_scale, 1e-12)):
        raise ValueError("Dieser Punkt lässt sich nicht endlich transformieren.")
    x, y = projected[:2] / projected[2]
    if not np.isfinite([x, y]).all():
        raise ValueError("Dieser Punkt lässt sich nicht endlich transformieren.")
    return float(x), float(y)


@dataclass(frozen=True)
class PlaneCalibration:

    corners_px: tuple[tuple[float, float], ...]
    image_size_px: tuple[int, int]
    matrix: np.ndarray = field(init=False, repr=False, compare=False)
    inverse_matrix: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        size = self.image_size_px
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in size)):
            raise ValueError("Die Bildauflösung muss Breite und Höhe als Ganzzahlen ab 2 enthalten.")
        try:
            raw = np.asarray(self.corners_px)
            if raw.shape != (4, 2) or raw.dtype.kind not in "iuf":
                raise ValueError
            points = raw.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Genau vier Eckpunkte mit je zwei Zahlen sind erforderlich.") from error
        if not np.isfinite(points).all():
            raise ValueError("Die Eckpunkte müssen endlich sein.")
        if (points < 0).any() or (points > np.array(size) - 1).any():
            raise ValueError("Alle Eckpunkte müssen innerhalb des Bildes liegen.")

        contour = points.astype(np.float32)

        if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 16:
            raise ValueError("Die Ecken müssen ein konvexes Viereck in Umlaufreihenfolge bilden (mindestens 16 Pixel²).")
        matrix = cv2.getPerspectiveTransform(contour, UNIT_CORNERS)
        if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e12:
            raise ValueError("Die Eckpunkte ergeben keine ausreichend stabile Transformation.")
        matrix.setflags(write=False)
        inverse_matrix = np.linalg.inv(matrix)
        inverse_matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "inverse_matrix", inverse_matrix)
        object.__setattr__(self, "corners_px", tuple(tuple(float(v) for v in p) for p in points))
        object.__setattr__(self, "image_size_px", tuple(size))

    def pixel_to_normalized(self, point: tuple[float, float]) -> tuple[float, float]:
        return _transform_point(self.matrix, point)

    def normalized_to_pixel(self, point: tuple[float, float]) -> tuple[float, float]:
        return _transform_point(self.inverse_matrix, point)

    def validate_image_size(self, size: tuple[int, int]) -> None:
        if tuple(size) != self.image_size_px:
            raise ValueError(f"Kalibrierung gilt für {self.image_size_px}, das Bild hat aber {tuple(size)} Pixel.")

    def save(self, path: Path, *, overwrite: bool = False) -> None:
        if path.suffix.lower() != ".json":
            raise ValueError("Die Kalibrierdatei muss die Endung .json besitzen.")
        payload = {
            "schema_version": 1,
            "image_size_px": self.image_size_px,
            "surface_size_m": SURFACE_SIZE_M,
            "corner_order": CORNER_ORDER,
            "corners_px": self.corners_px,
        }
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
            handle.write(text)

    @classmethod
    def load(cls, path: Path) -> "PlaneCalibration":
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"Ungültige Kalibrierdatei: {path}") from error
        expected = {"schema_version", "image_size_px", "surface_size_m", "corner_order", "corners_px"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Die Kalibrierdatei enthält fehlende oder unbekannte Felder.")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("Die Kalibrierdateiversion wird nicht unterstützt.")
        if payload["surface_size_m"] != list(SURFACE_SIZE_M):
            raise ValueError(f"Die Kalibrierung muss für die konfigurierte Fläche {SURFACE_SIZE_M} m gelten.")
        if payload["corner_order"] != list(CORNER_ORDER):
            raise ValueError("Die Reihenfolge der Kalibrierecken ist nicht kompatibel.")
        return cls(payload["corners_px"], payload["image_size_px"])

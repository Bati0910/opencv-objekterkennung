from pathlib import Path
from dataclasses import replace
from time import monotonic

import numpy as np

from objekterkennung.geometry.calibration import PlaneCalibration
from objekterkennung.geometry.automatic_calibration import AutomaticCalibration
from objekterkennung.camera import validate_camera_frame
from objekterkennung.vision.detection import DetectionResult, detect_frame
import cv2
from objekterkennung.geometry.positions import ObjectGeometry, estimate_position
from objekterkennung.vision.segmentation import SegmentationConfig
from objekterkennung.tracking import ShapeConfirmation
from objekterkennung.tracking import DetectionPacing
from objekterkennung.adaptive_detection import AdaptiveDetection


def load_image(path: Path) -> np.ndarray:
    if path.is_dir():
        raise ValueError(f"Der Bildpfad ist ein Ordner: {path}")


    image_bytes = path.read_bytes()
    if not image_bytes:
        raise ValueError(f"Die Bilddatei ist leer: {path}")

    encoded_image = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        frame = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(f"Die Bilddatei konnte nicht dekodiert werden: {path}") from error

    if frame is None:
        raise ValueError(f"Keine lesbare oder unterstützte Bilddatei: {path}")
    return frame


def display_to_image(x: float, y: float,
                     mapping: tuple[int, int, int, int, int, int]) -> tuple[float, float] | None:
    left, top, shown_width, shown_height, width, height = mapping
    if not left <= x < left + shown_width or not top <= y < top + shown_height:
        return None
    return (min(width - 1, (x - left) * width / shown_width),
            min(height - 1, (y - top) * height / shown_height))


class ImageSession:

    def __init__(self) -> None:
        self.frame: np.ndarray | None = None
        self.image_path: Path | None = None
        self.corners: list[tuple[float, float]] = []
        self.calibration: PlaneCalibration | None = None
        self.result: DetectionResult | None = None
        self.stage = "empty"
        self.confirmation = ShapeConfirmation()
        self.auto_calibration = AutomaticCalibration()
        self.detection_pacing = DetectionPacing()
        self.adaptive_detection = AdaptiveDetection()
        self.result_frame = None
        self.region_filter_overrides: dict[str, bool] = {}

    def _reset_confirmation(self):
        self.confirmation.clear()
        self.adaptive_detection.reset()

    def open_image(self, path: Path) -> None:
        # Validate the new image before replacing the current session state.
        frame = load_image(path)
        self.frame = frame
        self.image_path = path
        self.auto_calibration.reset()
        self.detection_pacing.reset()
        self._reset_confirmation()
        self.corners = []
        self.calibration = None
        self.result = None
        self.stage = "preview"

    def clear(self) -> None:
        self.frame = None
        self.image_path = None
        self.corners = []
        self.calibration = None
        self.result = None
        self.stage = "empty"
        self._reset_confirmation()
        self.auto_calibration.reset()
        self.detection_pacing.reset()
        self.result_frame = None

    def update_live_frame(self, frame: np.ndarray, *, keep_result: bool = False) -> bool:
        validate_camera_frame(frame)
        changed_size = self.frame is not None and self.frame.shape != frame.shape
        if changed_size:
            self.clear()
        # Freeze the image during manual selection so all corner clicks refer to one frame.
        if self.stage != "calibrating":
            self.frame = frame.copy()
            self.image_path = None

            if not keep_result or changed_size:
                self.result = None
                self.result_frame = None
            if self.stage == "empty":
                self.stage = "preview"
        return changed_size

    def _image_size(self) -> tuple[int, int]:
        if self.frame is None:
            raise ValueError("Zuerst ein Bild öffnen.")
        return self.frame.shape[1], self.frame.shape[0]

    def begin_calibration(self) -> None:
        self._image_size()
        self.auto_calibration.set_enabled(False)
        self.corners = []
        self.calibration = None
        self.result = None
        self.stage = "calibrating"
        self._reset_confirmation()

    def select_corner(self, point: tuple[float, float]) -> None:
        if self.stage != "calibrating":
            raise ValueError("Zuerst Kalibrieren wählen.")
        width, height = self._image_size()
        if (len(point) != 2 or not np.isfinite(point).all()
                or not 0 <= point[0] <= width - 1 or not 0 <= point[1] <= height - 1):
            raise ValueError("Der Eckpunkt muss innerhalb des Bildes liegen.")
        proposed = [*self.corners, tuple(point)]
        calibration = PlaneCalibration(tuple(proposed), (width, height)) if len(proposed) == 4 else None
        self.corners = proposed
        if calibration is not None:
            self.calibration = calibration
            self.stage = "calibrated"

    def undo_corner(self) -> None:
        if not self.corners:
            return
        self.auto_calibration.set_enabled(False)
        self.corners.pop()
        self.calibration = None
        self.result = None
        self.stage = "calibrating"
        self._reset_confirmation()

    def load_calibration(self, path: Path) -> None:
        size = self._image_size()
        calibration = PlaneCalibration.load(path)
        calibration.validate_image_size(size)
        self.auto_calibration.set_enabled(False)
        self._adopt_calibration(calibration)

    def _adopt_calibration(self, calibration: PlaneCalibration) -> None:
        self.calibration = calibration
        self.corners = list(calibration.corners_px)
        self.result = None
        self.stage = "calibrated"
        self._reset_confirmation()
        self.detection_pacing.reset()
        self.result_frame = None

    def request_single_calibration(self) -> None:
        self._image_size()
        if self.stage == "calibrating":
            self.corners = []
            self.stage = "preview"
        self.auto_calibration.request_once()

    def set_automatic_calibration(self, enabled: bool) -> None:
        if enabled and self.stage == "calibrating":
            self.corners = []
            self.stage = "preview"
        if not enabled and self.auto_calibration.blocked:
            self.calibration = None
            self.corners = []
            self.result = None
            self.stage = "preview" if self.frame is not None else "empty"
            self._reset_confirmation()
        self.auto_calibration.set_enabled(enabled)

    def check_automatic_calibration(self, now: float, *, single_frame: bool = False) -> bool:
        if self.frame is None or self.stage == "calibrating":
            return False
        proposed = self.auto_calibration.check(self.frame, self.calibration, now, single_frame=single_frame)
        if self.auto_calibration.blocked:
            self.result = None
            self._reset_confirmation()
        if proposed is None:
            return False
        self._adopt_calibration(proposed)
        return True

    def detect(self, config: SegmentationConfig, geometry: ObjectGeometry, *, live: bool = False) -> bool:
        if self.calibration is None:
            raise ValueError("Zuerst vier gültige Feldecken wählen oder eine passende Kalibrierung laden.")
        if self.auto_calibration.blocked:
            raise ValueError("Flächenbewegung noch ungeklärt; erst neu kalibrieren.")
        adaptive = live and self.adaptive_detection.enabled
        if adaptive:
            raw = self.adaptive_detection.step(self.frame, config, self.calibration, geometry,
                                               monotonic(), detector=detect_frame)
            if raw is None:
                return False
            self.adaptive_detection.completed(monotonic())
        else:
            raw = detect_frame(self.frame, config, self.calibration, geometry)
        self.result = raw
        # Keep the analyzed image paired with its result during pauses and adaptive reuse.
        self.result_frame = self.frame.copy()
        if adaptive:
            self.confirmation.clear()
        elif live and config.stabilize_live_shapes:
            shown = self.confirmation.apply(raw.segmentation.candidates, raw.shapes, monotonic())
            positions = [estimate_position(candidate.contour, prediction.label, self.calibration, geometry)
                         if prediction.label != original.label else position
                         for candidate, prediction, original, position in
                         zip(raw.segmentation.candidates, shown, raw.shapes, raw.positions)]
            self.result = replace(raw, shapes=shown, positions=positions)

        else:
            self._reset_confirmation()
        self.stage = "detection"
        return True

    def show_preview(self) -> None:
        self.result = None
        self._reset_confirmation()
        if self.calibration is not None:
            self.stage = "calibrated"

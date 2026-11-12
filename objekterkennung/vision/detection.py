from dataclasses import dataclass

import numpy as np
import cv2

from objekterkennung.vision.segmentation import SegmentationConfig, SegmentationResult, process_frame
from objekterkennung.vision.shapes import ShapePrediction, classify_shape
from objekterkennung.geometry.calibration import PlaneCalibration
from objekterkennung.geometry.positions import ObjectGeometry, PositionEstimate, estimate_position
from objekterkennung.vision.region_plausibility import check_width
from objekterkennung.vision.surface_recovery import recover_surface
from objekterkennung.vision.region_plausibility import check_region, connected_support
from objekterkennung.vision.local_masks import component_mask, replace_component


@dataclass
class DetectionResult:

    segmentation: SegmentationResult
    shapes: list[ShapePrediction]
    positions: list[PositionEstimate] | None = None


def detect_frame(frame: np.ndarray, config: SegmentationConfig,
                 calibration: PlaneCalibration | None = None,
                 geometry: ObjectGeometry | None = None, *,
                 reusable: SegmentationResult | None = None) -> DetectionResult:
    regions = process_frame(frame, config) if reusable is None else process_frame(frame, config, reusable=reusable)
    if calibration is not None:
        calibration.validate_image_size((frame.shape[1], frame.shape[0]))
        if geometry is None:
            raise ValueError("Für die Positionsschätzung sind bekannte Objektmaße erforderlich.")
    if config.reject_thin_regions or config.reject_field_filling_regions:
        regions.plausibility_report = []
        kept = []
        masks = {color: mask.copy() for color, mask in regions.masks.items()}
        for candidate in regions.candidates:
            decision = check_region(candidate, regions.masks[candidate.color],
                calibration if config.reject_field_filling_regions else None, geometry,
                reject_thin=config.reject_thin_regions)
            regions.plausibility_report.append(decision)
            if decision["accepted"]:
                kept.append(candidate)
            else:
                support = connected_support(candidate, regions.masks[candidate.color])
                x, y, width, height = candidate.bbox_xywh
                masks[candidate.color][y:y + height, x:x + width][support[1:-1, 1:-1] > 0] = 0
        regions.candidates, regions.masks = kept, masks
    if config.filter_small_regions and calibration is not None:
        regions.size_report = [{"color": candidate.color, **check_width(candidate.bbox_xywh, calibration, geometry)}
                               for candidate in regions.candidates]
        kept = []
        regions.masks = {color: mask.copy() for color, mask in regions.masks.items()}
        for candidate, check in zip(regions.candidates, regions.size_report):
            if check["accepted"]:
                kept.append(candidate)
            else:
                x, y, width, height = candidate.bbox_xywh
                local = regions.masks[candidate.color][y:y + height, x:x + width]
                support = component_mask(local, candidate.contour - (x, y))
                local[support > 0] = 0
        regions.candidates = kept
    shapes = [classify_shape(candidate.contour) for candidate in regions.candidates]
    if config.recover_weak_surfaces:
        for index, (candidate, shape) in enumerate(zip(regions.candidates, shapes)):


            if shape.label != "unknown" or candidate.separation_reason is not None:
                continue
            recovered = recover_surface(frame, candidate.contour, candidate.color, regions.masks,
                                        config.hsv_ranges[candidate.color],
                                        blocked_mask=regions.shadow_rejections[candidate.color]
                                        if regions.shadow_rejections is not None else None)
            if recovered.accepted:
                prediction = classify_shape(recovered.contour)
                if prediction.label != "unknown":
                    regions.masks[candidate.color] = regions.masks[candidate.color].copy()
                    replace_component(regions.masks[candidate.color], candidate.contour,
                                      recovered.foreground_mask, recovered.origin_xy)
                    candidate.contour = recovered.contour
                    candidate.bbox_xywh = cv2.boundingRect(recovered.contour)
                    candidate.area_px = cv2.contourArea(recovered.contour)
                    candidate.refinement_reason = recovered.reason
                    shapes[index] = prediction
                else:
                    candidate.refinement_reason = "Seitenergänzung verworfen: Form weiterhin uneindeutig."
            else:
                candidate.refinement_reason = (candidate.refinement_reason or "") + "\nSeitenergänzung: " + recovered.reason
    positions = None
    if calibration is not None:
        positions = [estimate_position(candidate.contour, shape.label, calibration, geometry)
                     for candidate, shape in zip(regions.candidates, shapes)]
    return DetectionResult(regions, shapes, positions)

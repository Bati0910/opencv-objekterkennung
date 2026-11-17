from objekterkennung.configuration import SETUP

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from objekterkennung.vision.region_refinement import refine_region
from objekterkennung.vision.shadow_suppression import suppress_shadow_colors
from objekterkennung.vision.touching_separation import split_touching
from objekterkennung.vision.contact_separation import split_contact
from objekterkennung.vision.local_masks import component_mask, replace_component


COLOR_NAMES = ("red", "green", "blue", "yellow")
HSVBound = tuple[int, int, int]
HSVRange = tuple[HSVBound, HSVBound]


@dataclass(frozen=True)
class SegmentationConfig:

    hsv_ranges: dict[str, tuple[HSVRange, ...]]
    min_area_px: float
    kernel_size: int
    refine_dark_regions: bool = False
    suppress_shadows: bool = False
    recover_weak_surfaces: bool = False
    filter_small_regions: bool = False
    stabilize_live_shapes: bool = False
    reject_thin_regions: bool = False
    reject_field_filling_regions: bool = False
    separate_touching_regions: bool = False

    def __post_init__(self) -> None:
        if type(self.refine_dark_regions) is not bool:
            raise ValueError("refine_dark_regions muss true oder false sein.")
        if type(self.suppress_shadows) is not bool:
            raise ValueError("suppress_shadows muss true oder false sein.")
        for name in ("recover_weak_surfaces", "filter_small_regions", "stabilize_live_shapes",
                     "reject_thin_regions", "reject_field_filling_regions", "separate_touching_regions"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} muss true oder false sein.")
        if (isinstance(self.min_area_px, bool) or not isinstance(self.min_area_px, (int, float))
                or (isinstance(self.min_area_px, float) and not math.isfinite(self.min_area_px))
                or self.min_area_px <= 0):
            raise ValueError("min_area_px muss eine endliche positive Zahl sein.")
        if type(self.kernel_size) is not int or not 1 <= self.kernel_size <= 31 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size muss eine ungerade Ganzzahl zwischen 1 und 31 sein.")
        if not isinstance(self.hsv_ranges, dict) or set(self.hsv_ranges) != set(COLOR_NAMES):
            raise ValueError("HSV-Bereiche für red, green, blue und yellow sind erforderlich.")
        canonical = {}
        for color in COLOR_NAMES:
            intervals = self.hsv_ranges[color]
            if not isinstance(intervals, (tuple, list)) or not intervals:
                raise ValueError(f"Mindestens ein HSV-Bereich für {color} ist erforderlich.")
            validated = []
            for interval in intervals:
                try:
                    values = np.asarray(interval, dtype=object)
                except ValueError as error:
                    raise ValueError(f"Ungültiger HSV-Bereich für {color}.") from error
                if values.shape != (2, 3) or any(type(v) is not int for v in values.flat):
                    raise ValueError(f"HSV-Bereiche für {color} brauchen zwei Listen mit je drei Ganzzahlen.")
                lower, upper = values
                if (lower < 0).any() or (upper > [179, 255, 255]).any() or (lower > upper).any():
                    raise ValueError(f"Ungültige HSV-Grenzen für {color}; H: 0–179, S/V: 0–255.")
                validated.append((tuple(int(v) for v in lower), tuple(int(v) for v in upper)))
            canonical[color] = tuple(validated)

        for index, first in enumerate(COLOR_NAMES):
            for second in COLOR_NAMES[index + 1:]:
                for lower_a, upper_a in canonical[first]:
                    for lower_b, upper_b in canonical[second]:
                        if np.all(np.maximum(lower_a, lower_b) <= np.minimum(upper_a, upper_b)):
                            raise ValueError(f"HSV-Bereiche von {first} und {second} überlappen.")
        object.__setattr__(self, "hsv_ranges", canonical)


def load_config() -> SegmentationConfig:
    fields = set(SegmentationConfig.__dataclass_fields__)
    required = {"hsv_ranges", "min_area_px", "kernel_size"}
    if not required <= set(SETUP.segmentation) or set(SETUP.segmentation) - fields:
        raise ValueError("Die Farbparameter enthalten fehlende oder unbekannte Felder.")
    return SegmentationConfig(**SETUP.segmentation)


@dataclass
class Candidate:

    color: str
    bbox_xywh: tuple[int, int, int, int]
    area_px: float
    contour: np.ndarray = field(repr=False, compare=False)
    refinement_reason: str | None = None
    separation_reason: str | None = None
    reuse_age_s: float | None = None


@dataclass
class SegmentationResult:

    candidates: list[Candidate]
    masks: dict[str, np.ndarray] = field(repr=False)
    seed_masks: dict[str, np.ndarray] | None = field(default=None, repr=False)
    hsv_masks: dict[str, np.ndarray] | None = field(default=None, repr=False)
    shadow_rejections: dict[str, np.ndarray] | None = field(default=None, repr=False)
    shadow_report: dict | None = None
    size_report: list[dict] | None = None
    plausibility_report: list[dict] | None = None
    separation_report: list[dict] | None = None


def _reuse_unchanged(candidates, masks, reusable):
    if reusable is None or not reusable.candidates:
        return candidates, masks
    covers = {color: np.zeros_like(mask) for color, mask in masks.items()}
    cached_support = {color: np.zeros_like(mask) for color, mask in masks.items()}
    for cached in reusable.candidates:
        x, y, width, height = cached.bbox_xywh
        padding = max(3, round(.15 * max(width, height)))
        cv2.rectangle(covers[cached.color], (max(0, x - padding), max(0, y - padding)),
                      (x + width + padding, y + height + padding), 255, -1)
        old_mask = reusable.masks[cached.color][y:y + height, x:x + width]
        old_support = component_mask(old_mask, cached.contour - (x, y)) > 0
        current = masks[cached.color][y:y + height, x:x + width]
        if (not np.any(old_support) or
                np.count_nonzero(old_support & (current > 0)) < .2 * np.count_nonzero(old_support)):
            return candidates, masks
        cached_support[cached.color][y:y + height, x:x + width][old_support] = 255
    replaced, remaining = [], []
    for candidate in candidates:
        x, y, width, height = candidate.bbox_xywh
        local = masks[candidate.color][y:y + height, x:x + width]
        support = component_mask(local, candidate.contour - (x, y)) > 0
        overlap = support & (covers[candidate.color][y:y + height, x:x + width] > 0)
        count, inside = np.count_nonzero(support), np.count_nonzero(overlap)
        if inside and inside < .98 * count:

            return candidates, masks
        # A cache rectangle must not absorb a new component without matching old pixel support.
        if inside and not np.any(support & (cached_support[candidate.color][y:y + height, x:x + width] > 0)):
            return candidates, masks
        (replaced if inside else remaining).append(candidate)
    masks = {color: mask.copy() for color, mask in masks.items()}
    for candidate in replaced:
        x, y, width, height = candidate.bbox_xywh
        local = masks[candidate.color][y:y + height, x:x + width]
        support = component_mask(local, candidate.contour - (x, y))
        local[support > 0] = 0
    for cached in reusable.candidates:
        x, y, width, height = cached.bbox_xywh
        old_mask = reusable.masks[cached.color][y:y + height, x:x + width]
        masks[cached.color][y:y + height, x:x + width] |= component_mask(old_mask, cached.contour - (x, y))
        remaining.append(cached)
    return remaining, masks


def process_frame(frame: np.ndarray, config: SegmentationConfig, *,
                  reusable: SegmentationResult | None = None) -> SegmentationResult:
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
        raise ValueError("Ein nicht leeres uint8-BGR-Bild mit drei Kanälen ist erforderlich.")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.kernel_size, config.kernel_size))
    masks = {}
    for color in COLOR_NAMES:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in config.hsv_ranges[color]:
            part = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            mask = cv2.bitwise_or(mask, part)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks[color] = mask
    hsv_masks = shadow_rejections = shadow_report = None
    if config.suppress_shadows:
        hsv_masks = masks
        supported = suppress_shadow_colors(frame, hsv, masks, config.kernel_size)
        masks, shadow_rejections, shadow_report = supported.masks, supported.rejected_masks, supported.report
    candidates = []
    for color, mask in masks.items():
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours):


            # Hole contours are not separate objects; the pixel masks still retain their holes.
            if hierarchy[0, index, 3] != -1:
                continue
            area = cv2.contourArea(contour)
            if area >= config.min_area_px:
                candidates.append(Candidate(color, cv2.boundingRect(contour), area, contour))
    candidates, masks = _reuse_unchanged(candidates, masks, reusable)
    seed_masks = None
    if config.refine_dark_regions:
        seed_masks = masks
        masks = {color: mask.copy() for color, mask in seed_masks.items()}
        for candidate in candidates:
            if candidate.reuse_age_s is not None:
                continue
            result = refine_region(frame, hsv, candidate.contour, seed_masks,
                                   candidate.color, config.hsv_ranges[candidate.color],
                                   blocked_mask=shadow_rejections[candidate.color] if shadow_rejections is not None else None)
            candidate.refinement_reason = result.reason
            if result.accepted:
                replace_component(masks[candidate.color], candidate.contour, result.foreground_mask, result.origin_xy)
                candidate.contour = result.contour
                candidate.area_px = cv2.contourArea(result.contour)
                candidate.bbox_xywh = cv2.boundingRect(result.contour)
    separation_report = None
    if config.separate_touching_regions:
        separation_report = []
        separate = []

        masks = {color: mask.copy() for color, mask in masks.items()}
        for candidate in candidates:
            if candidate.reuse_age_s is not None:
                separate.append(candidate)
                continue
            result = split_touching(frame, candidate.contour, masks[candidate.color], config.min_area_px)
            # Try contact-seam evidence after marker-based separation fails.
            if not result.report["accepted"]:
                contact = split_contact(frame, candidate.contour, masks, candidate.color,
                                        config.hsv_ranges[candidate.color], config.min_area_px,
                                        refine=config.refine_dark_regions,
                                        blocked_mask=shadow_rejections[candidate.color] if shadow_rejections is not None else None)
                if contact.report["accepted"]:
                    contact.report["primary_attempt"] = result.report
                    result = contact
                else:
                    result.report["contact_fallback"] = contact.report
            separation_report.append({"color": candidate.color, **result.report})
            if not result.report["accepted"]:
                separate.append(candidate)
                continue
            x, y, width, height = candidate.bbox_xywh
            target = masks[candidate.color][y:y + height, x:x + width]


            target[result.source_support > 0] = 0
            for contour, part in zip(result.contours, result.part_supports):
                target |= part
                separate.append(Candidate(candidate.color, cv2.boundingRect(contour), cv2.contourArea(contour),
                                          contour, candidate.refinement_reason, result.report["reason"]))
        candidates = separate

    candidates.sort(key=lambda candidate: (candidate.bbox_xywh[0], candidate.bbox_xywh[1], candidate.color))
    return SegmentationResult(candidates, masks, seed_masks, hsv_masks, shadow_rejections, shadow_report,
                              separation_report=separation_report)

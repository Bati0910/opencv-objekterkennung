from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ColorSupportResult:
    masks: dict[str, np.ndarray]
    rejected_masks: dict[str, np.ndarray]
    report: dict


def invariant_colors(bgr: np.ndarray) -> np.ndarray:
    # Channel-ratio angles reduce sensitivity to uniform intensity scaling.
    b, g, r = cv2.split(bgr.astype(np.float32))
    return np.stack((np.arctan2(r, np.maximum(g, b)),
                     np.arctan2(g, np.maximum(r, b)),
                     np.arctan2(b, np.maximum(r, g))), axis=-1)


def suppress_shadow_colors(frame: np.ndarray, hsv: np.ndarray,
                           masks: dict[str, np.ndarray], kernel_size: int) -> ColorSupportResult:
    output, rejected, report = {}, {}, {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for color, source in masks.items():
        cleaned = np.zeros_like(source)
        decisions = []
        count, labels, stats, _ = cv2.connectedComponentsWithStats(source, connectivity=8)
        for label in range(1, count):
            x, y, width, height = (int(value) for value in stats[label, :4])
            roi = frame[y:y + height, x:x + width]
            local_hsv = hsv[y:y + height, x:x + width]


            member = labels[y:y + height, x:x + width] == label
            member_count = int(stats[label, cv2.CC_STAT_AREA])
            chroma = local_hsv[:, :, 1].astype(np.float32) * local_hsv[:, :, 2] / 255


            core = member & (local_hsv[:, :, 1] >= 150) & (chroma >= 25)
            core_count = int(np.count_nonzero(core))
            decision = {"bbox_xywh": [x, y, width, height], "source_pixels": member_count,
                        "core_pixels": core_count, "kept_pixels": 0, "tolerance_rad": None}
            decisions.append(decision)
            if core_count < max(30, 0.05 * member_count):
                decision["reason"] = "Keine ausreichende kräftige Körperfarbstützung."
                continue
            features = invariant_colors(roi)
            center = np.median(features[core], axis=0)
            distance = np.linalg.norm(features - center, axis=2)
            tolerance = float(np.clip(np.quantile(distance[core], 0.95) + 0.08, 0.18, 0.40))


            hues = local_hsv[:, :, 0].astype(np.float32)
            histogram = np.bincount(hues[core].astype(int), minlength=180)
            peak = int(np.argmax(histogram))
            # Hue wraps around: values near 0 and 179 can represent the same red color.
            offsets = (hues - peak + 90) % 180 - 90
            hue_center = (peak + float(np.median(offsets[core]))) % 180
            hue_distance = np.abs((hues - hue_center + 90) % 180 - 90)
            hue_tolerance = float(np.clip(np.quantile(hue_distance[core], 0.95) + 2, 8, 20))
            keep = member & ((distance <= tolerance) | (hue_distance <= hue_tolerance) | core)
            target = cleaned[y:y + height, x:x + width]
            target[keep] = 255
            decision.update(kept_pixels=int(np.count_nonzero(keep)), tolerance_rad=tolerance,
                            reason="Farbähnlichkeit zum robusten Kern im aktuellen Bild geprüft.",
                            hue_center=hue_center, hue_tolerance=hue_tolerance,
                            core_center_c1c2c3=[float(v) for v in center])
        if not np.array_equal(cleaned, source):
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # Morphology must not introduce pixels outside the original color support.
        cleaned &= source
        output[color] = cleaned
        rejected[color] = cv2.bitwise_and(source, cv2.bitwise_not(cleaned))
        report[color] = {"source_pixels": int(np.count_nonzero(source)),
                         "kept_pixels": int(np.count_nonzero(cleaned)), "regions": decisions}
    return ColorSupportResult(output, rejected, report)

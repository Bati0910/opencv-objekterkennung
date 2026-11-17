from dataclasses import dataclass
from heapq import heappop, heappush

import cv2
import numpy as np

from objekterkennung.vision.local_masks import component_mask


@dataclass
class SeparationResult:
    contours: list[np.ndarray]
    report: dict
    source_support: np.ndarray | None = None
    part_supports: list[np.ndarray] | None = None


def _distinct_cores(smooth: np.ndarray, first: tuple, second: tuple) -> bool:
    px, py, pr = first
    qx, qy, qr = second
    # A saddle between distance peaks supports distinct cores even when they are close.
    level = min(smooth[py, px], smooth[qy, qx]) - max(2., .1 * min(pr, qr))
    if level <= 0:
        return False
    _, labels = cv2.connectedComponents((smooth >= level).astype(np.uint8))
    return bool(labels[py, px] and labels[qy, qx] and labels[py, px] != labels[qy, qx])


def _masked_distance_partition(source: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    owners = np.where(source > 0, seeds, 0).astype(np.int32)
    # Flood higher distance values first; insertion order breaks ties deterministically.
    queue = []
    age = 0
    for y, x in zip(*np.nonzero(owners >= 2)):
        heappush(queue, (-float(distance[y, x]), age, int(y), int(x)))
        age += 1

    neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    while queue:
        level, _, y, x = heappop(queue)
        for dy, dx in neighbors:
            yy, xx = y + dy, x + dx
            if source[yy, xx] and owners[yy, xx] == 0:
                owners[yy, xx] = owners[y, x]
                heappush(queue, (max(level, -float(distance[yy, xx])), age, yy, xx))
                age += 1
    return _separate_labels(owners)


def _separate_labels(owners: np.ndarray) -> np.ndarray:


    boundary = np.zeros(owners.shape, bool)
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]:
        other = np.roll(owners, (dy, dx), axis=(0, 1))
        boundary |= (owners > other) & (other >= 2)
    owners[boundary] = -1
    return owners


def split_touching(frame: np.ndarray, contour: np.ndarray, mask: np.ndarray,
                   min_area_px: float) -> SeparationResult:
    x, y, width, height = cv2.boundingRect(contour)
    report = {"bbox_xywh": [x, y, width, height], "accepted": False,
              "markers_px": [], "parts": [], "retained_fraction": None}

    def unchanged(reason):
        report["reason"] = reason
        return SeparationResult([contour], report)

    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area else 0.

    if solidity >= .97 or min(width, height) < 30 or area < 2 * min_area_px:
        return unchanged("Keine ausreichend große, eingezogene Region für eine sichere Trennung.")
    if area / (width * height) < .3 or solidity < .5:
        return unchanged("Zu wenig kompakte Farbstützung für eine Körpertrennung.")
    if x <= 0 or y <= 0 or x + width >= frame.shape[1] or y + height >= frame.shape[0]:
        return unchanged("Angeschnittene Region am Bildrand bleibt unverändert.")
    if width * height > 500_000:
        return unchanged("Region überschreitet das lokale Rechenbudget für die Trennung.")


    local = np.pad(mask[y:y + height, x:x + width], 1)
    _, labels = cv2.connectedComponents((local > 0).astype(np.uint8))
    first = contour.reshape(-1, 2)[0] - (x, y) + 1
    label = labels[first[1], first[0]]
    if label == 0:
        return unchanged("Ursprüngliche Farbregion nicht sicher zugeordnet.")
    support = (labels == label).astype(np.uint8) * 255
    distance = cv2.distanceTransform(support, cv2.DIST_L2, 5)
    radius = float(distance.max())
    if radius < 6:
        return unchanged("Keine ausreichend breiten Innenkerne.")


    smooth = cv2.GaussianBlur(distance, (0, 0), max(.6, .012 * radius))
    window = max(3, round(.48 * radius)) | 1
    maxima = ((smooth >= cv2.dilate(smooth, np.ones((window, window), np.uint8)) - 1e-4)
              & (smooth > max(3., .25 * radius)))
    count, plateaus = cv2.connectedComponents(maxima.astype(np.uint8))
    if count > 25:
        return unchanged("Zu viele uneindeutige Maximalbereiche; Region bleibt unverändert.")
    peaks = []
    for index in range(1, count):
        yy, xx = np.nonzero(plateaus == index)
        best = np.argmax(distance[yy, xx])
        peaks.append((int(xx[best]), int(yy[best]), float(distance[yy[best], xx[best]])))

    selected = []
    report["raw_markers_px"] = [[px + x - 1, py + y - 1] for px, py, _ in peaks]
    report["saddle_supported_markers_px"] = []
    for px, py, pr in sorted(peaks, key=lambda peak: (-peak[2], peak[1], peak[0])):
        nearby = [peak for peak in selected if np.hypot(px - peak[0], py - peak[1]) < .65 * (pr + peak[2])]
        if all(_distinct_cores(smooth, (px, py, pr), peak) for peak in nearby):
            selected.append((px, py, pr))
            if nearby:
                report["saddle_supported_markers_px"].append([px + x - 1, py + y - 1])
    report["markers_px"] = [[px + x - 1, py + y - 1] for px, py, _ in selected]
    if not 2 <= len(selected) <= 6:
        return unchanged("Keine eindeutige Gruppe aus zwei bis sechs getrennten Innenkernen.")

    padding = max(3, round(.1 * radius))
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(frame.shape[1], x + width + padding), min(frame.shape[0], y + height + padding)
    roi = frame[top:bottom, left:right]
    source = np.zeros(roi.shape[:2], np.uint8)
    source[y - top:y - top + height, x - left:x - left + width] = support[1:-1, 1:-1]
    markers = np.where(source > 0, 0, 1).astype(np.int32)
    for index, (px, py, pr) in enumerate(selected, 2):
        cv2.circle(markers, (px + x - 1 - left, py + y - 1 - top),
                   max(2, round(.2 * pr)), index, -1)
    markers[source == 0] = 1
    seeds = np.where(markers >= 2, markers, 0)
    cv2.watershed(roi, markers)
    source_pixels = int(np.count_nonzero(source))
    background_pixels = int(np.count_nonzero((source > 0) & (markers == 1)))
    image_retained = np.count_nonzero((source > 0) & (markers >= 2)) / source_pixels
    report["method"] = "image_watershed"
    report["image_watershed"] = {"background_pixels": background_pixels,
                                  "source_pixels": source_pixels,
                                  "retained_fraction": float(image_retained)}


    # Use masked distance flooding when image watershed loses too much support to background.
    if image_retained < .88 and background_pixels:
        markers = _masked_distance_partition(source, seeds)
        report["method"] = "masked_distance"
    else:
        markers = _separate_labels(markers)

    contours = []
    part_supports = []
    retained = np.zeros_like(source)
    for index in range(2, len(selected) + 2):
        part = ((markers == index) & (source > 0)).astype(np.uint8) * 255
        found, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            return unchanged("Ein Innenkern liefert keine erhaltene Teilregion.")
        outline = max(found, key=cv2.contourArea)
        part_area = cv2.contourArea(outline)
        part_hull = cv2.contourArea(cv2.convexHull(outline))
        part_solidity = part_area / part_hull if part_hull else 0.
        part_mask = np.zeros_like(source)
        cv2.drawContours(part_mask, [outline], -1, 255, -1)
        actual_part = component_mask(part, outline)
        part_pixels = int(np.count_nonzero(actual_part))
        fraction = part_pixels / source_pixels
        report["parts"].append({"area_px": part_area, "solidity": part_solidity,
                                "source_fraction": fraction})
        if (part_area < min_area_px or not .12 <= fraction <= .85
                or min(cv2.boundingRect(outline)[2:]) < 20 or part_solidity < .93
                or part_pixels < .95 * np.count_nonzero(part)):
            return unchanged("Teilregion zu klein, fragmentiert oder nicht ausreichend geschlossen.")

        if np.any((part_mask > 0) & (source > 0) & (markers >= 2) & (markers != index)):
            return unchanged("Teilkontur würde eine andere Instanz einschließen.")
        retained |= actual_part
        part_supports.append(actual_part[y - top:y - top + height, x - left:x - left + width])
        contours.append(outline + np.array([left, top], np.int32))
    fraction = np.count_nonzero(retained) / source_pixels
    report["retained_fraction"] = float(fraction)
    if fraction < .88:
        return unchanged("Trennung verliert zu viel der ursprünglichen Farbstützung.")
    # Require improved geometric plausibility, not just multiple output regions.
    weighted_solidity = sum(p["area_px"] * p["solidity"] for p in report["parts"]) / sum(
        p["area_px"] for p in report["parts"])
    if weighted_solidity < solidity + .025:
        return unchanged("Trennung verbessert die Geschlossenheit nicht ausreichend.")
    evidence = "Distanzkarte" if report["method"] == "masked_distance" else "Bildkanten"
    report.update(accepted=True, reason=f"Farbregion anhand von {len(contours)} Innenkernen und {evidence} getrennt.")
    return SeparationResult(contours, report, support[1:-1, 1:-1], part_supports)

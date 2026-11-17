import math

import cv2
import numpy as np

from objekterkennung.geometry.calibration import PlaneCalibration, SURFACE_SIZE_M
from objekterkennung.geometry.positions import ObjectGeometry


def connected_support(candidate, mask):
    x, y, width, height = candidate.bbox_xywh
    local = np.pad(mask[y:y + height, x:x + width], 1)
    _, labels = cv2.connectedComponents((local > 0).astype(np.uint8))
    first = candidate.contour.reshape(-1, 2)[0] - (x, y) + 1
    label = labels[first[1], first[0]]
    return (labels == label).astype(np.uint8) * 255 if label else None


def check_region(candidate, mask, calibration=None, geometry=None, *, reject_thin=True) -> dict:
    x, y, width, height = candidate.bbox_xywh
    support = connected_support(candidate, mask)
    report = {"color": candidate.color, "bbox_xywh": list(candidate.bbox_xywh),
              "accepted": True, "reason": "Keine extreme Hintergrundgeometrie.",
              "field_coverage": None, "field_core_diameter_fraction": None}
    if support is None:
        report["reason"] = "Farbstützung nicht sicher zugeordnet; Kandidat bleibt erhalten."
        return report
    hull = np.zeros_like(support)
    cv2.drawContours(hull, [cv2.convexHull(candidate.contour) - (x - 1, y - 1)], -1, 255, -1)
    count = int(np.count_nonzero(support))
    extent = count / (width * height)
    solidity = min(1., count / max(1, np.count_nonzero(hull)))
    diameter = 2 * float(cv2.distanceTransform(support, cv2.DIST_L2, 5).max())
    relative_core = diameter / min(width, height)
    report.update(extent=extent, solidity=float(solidity), inner_diameter_px=diameter,
                  inner_diameter_to_short_bbox=relative_core)
    if reject_thin and extent < .20 and solidity < .35 and relative_core < .15:
        report.update(accepted=False, reason="Extrem dünne, flächenarme Farbregion ohne breiten Innenbereich.")
        return report
    if calibration is None or geometry is None:
        return report
    largest = max(math.sqrt(2) * geometry.cube_side_m, 2 * geometry.cylinder_radius_m,
                  2 * geometry.cone_radius_m, 2 * geometry.hemisphere_radius_m)

    if largest > .25 * min(SURFACE_SIZE_M):
        return report
    side = 180
    offset = np.array([[1., 0, x - 1], [0, 1., y - 1], [0, 0, 1.]])
    # Measure field coverage in plane coordinates to account for perspective.
    transform = np.diag([side, side, 1.]) @ calibration.matrix @ offset
    field = cv2.warpPerspective(support, transform, (side, side), flags=cv2.INTER_NEAREST)
    coverage = float(np.count_nonzero(field) / field.size)
    field_core = 2 * float(cv2.distanceTransform(np.pad(field, 1), cv2.DIST_L2, 5).max()) / side
    minimum_core = max(.5, 3 * largest / min(SURFACE_SIZE_M))
    report.update(field_coverage=coverage, field_core_diameter_fraction=field_core,
                  minimum_field_core_fraction=minimum_core)
    if coverage >= .90 and field_core >= minimum_core:
        report.update(accepted=False,
            reason="Nahezu vollständige Feldbedeckung mit übergroßem zusammenhängendem Farbkern.")
    return report


def check_width(bbox, calibration: PlaneCalibration, geometry: ObjectGeometry) -> dict:
    x, y, width, height = bbox
    result = {"bbox_xywh": list(bbox), "accepted": True, "projected_width_px": None,
              "width_ratio": None, "minimum_ratio": .80}
    diameter = min(geometry.cube_side_m, 2 * geometry.cylinder_radius_m,
                   2 * geometry.cone_radius_m, 2 * geometry.hemisphere_radius_m)
    try:
        center = calibration.pixel_to_normalized((x + (width - 1) / 2, y + height - 1))
        radius = diameter / (2 * SURFACE_SIZE_M[0])
        # Project the smallest expected footprint locally instead of using a global pixel threshold.
        ring = [calibration.normalized_to_pixel((center[0] + radius * math.cos(t),
                                                center[1] + radius * math.sin(t)))
                for t in np.linspace(0, 2 * math.pi, 32, endpoint=False)]
        projected_width = max(p[0] for p in ring) - min(p[0] for p in ring)
        if not math.isfinite(projected_width) or projected_width < 5 or projected_width > calibration.image_size_px[0]:
            raise ValueError("Unplausible lokale Projektionsskala.")
        ratio = width / projected_width
        result.update(accepted=ratio >= .80, projected_width_px=projected_width, width_ratio=ratio)
        result["reason"] = ("Lokale Größenplausibilität erfüllt." if result["accepted"] else
                            "Region deutlich schmaler als der kleinste bekannte Körper an dieser Feldposition.")
    except ValueError:
        result["reason"] = "Lokale Größenprüfung nicht sicher möglich; Kandidat bleibt erhalten."
    return result

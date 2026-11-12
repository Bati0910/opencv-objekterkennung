from itertools import combinations

import cv2
import numpy as np


MAX_LINES_PER_DIRECTION = 10
MAX_QUADS = 48


def order_corners(points):
    by_y = points[np.argsort(points[:, 1])]
    back = by_y[:2][np.argsort(by_y[:2, 0])]
    front = by_y[2:][np.argsort(by_y[2:, 0])]
    if front[:, 1].min() - back[:, 1].max() < .15 * np.ptp(points[:, 1]):
        return None
    return np.array([front[0], front[1], back[1], back[0]], np.float32)


def _sample(image, points):
    return cv2.remap(image, points[..., 0].astype(np.float32),
                     points[..., 1].astype(np.float32), cv2.INTER_LINEAR)


def _lines(gray, lab):
    segments = []
    detector = cv2.createLineSegmentDetector()
    for channel in (gray, lab[:, :, 1], lab[:, :, 2]):
        found = detector.detect(channel)[0]
        if found is not None:
            pairs = found.reshape(-1, 2, 2)
            lengths = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=1)
            segments.extend(pairs[np.argsort(lengths)[-180:]])
    segments.sort(key=lambda pair: np.linalg.norm(pair[1] - pair[0]), reverse=True)
    groups = []
    for segment in segments[:180]:
        start, end = segment
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < max(30, .045 * max(gray.shape)):
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        base = start + np.linspace(.1, .9, 21)[:, None] * direction
        samples = np.stack([base - 6 * normal, base + 6 * normal], axis=1)
        colors = _sample(lab, samples).astype(np.float32)
        if np.median(np.linalg.norm(colors[:, 0] - colors[:, 1], axis=1)) < 6:
            continue
        equation = np.array([*normal, -np.dot(normal, start)])
        if equation[0] < 0 or (abs(equation[0]) < 1e-6 and equation[1] < 0):
            equation = -equation
        existing = next((group for group in groups
                         if np.dot(equation[:2], group["line"][:2]) > .995
                         and np.max(np.abs(segment @ group["line"][:2] + group["line"][2])) < 4), None)
        if existing is not None:
            existing["segments"].append(segment)


            vx, vy, px, py = cv2.fitLine(np.concatenate(existing["segments"]).astype(np.float32),
                                        cv2.DIST_HUBER, 0, .01, .01).ravel()
            fitted = np.array([-vy, vx, vy * px - vx * py])
            if fitted[0] < 0 or (abs(fitted[0]) < 1e-6 and fitted[1] < 0):
                fitted = -fitted
            existing["line"] = fitted
        else:
            groups.append({"line": equation, "segments": [segment], "length": length,
                           "horizontal": abs(direction[0]) >= abs(direction[1])})
    return ([group for group in groups if group["horizontal"]][:MAX_LINES_PER_DIRECTION],
            [group for group in groups if not group["horizontal"]][:MAX_LINES_PER_DIRECTION])


def _coverage(start, end, group):
    direction = end - start
    squared_length = np.dot(direction, direction)
    samples = (np.arange(32) + .5) / 32
    covered = np.zeros(32, bool)
    for segment in group["segments"]:
        low, high = sorted((segment - start) @ direction / squared_length)
        covered |= (samples >= low) & (samples <= high)
    return float(covered.mean())


def _quadrilaterals(horizontal, vertical, shape):
    if len(horizontal) < 2 or len(vertical) < 2:
        return []
    crossings = np.cross(np.array([g["line"] for g in horizontal])[:, None],
                         np.array([g["line"] for g in vertical])[None, :])
    valid = np.abs(crossings[:, :, 2]) > .05
    points = crossings[:, :, :2] / np.where(valid, crossings[:, :, 2], 1)[:, :, None]
    hypotheses = []
    for a, b in combinations(range(len(horizontal)), 2):
        for c, d in combinations(range(len(vertical)), 2):
            if not all((valid[a, c], valid[a, d], valid[b, d], valid[b, c])):
                continue
            quad = np.float32([points[a, c], points[a, d], points[b, d], points[b, c]])
            if (np.any(quad < 10) or np.any(quad > np.array(shape[::-1]) - 11)
                    or not cv2.isContourConvex(quad)):
                continue
            area = cv2.contourArea(quad)
            if not .01 * np.prod(shape) <= area <= .65 * np.prod(shape):
                continue
            edges = np.roll(quad, -1, axis=0) - quad
            if np.min(np.linalg.norm(edges, axis=1)) < 40:
                continue
            support = [_coverage(start, end, group) for start, end, group in zip(
                quad, np.roll(quad, -1, axis=0), (horizontal[a], vertical[d], horizontal[b], vertical[c]))]
            if min(support) < .55:
                continue
            ordered = order_corners(quad)
            if ordered is not None:
                hypotheses.append((float(np.mean(support)), ordered))
    hypotheses.sort(key=lambda entry: entry[0], reverse=True)
    distinct = []
    for support, quad in hypotheses:
        if not any(np.max(np.linalg.norm(quad - other, axis=1)) < 5 for _, other in distinct):
            distinct.append((support, quad))
        if len(distinct) >= MAX_QUADS:
            break
    return distinct


def _paper_color_model(lab, neutral):
    yy, xx = np.nonzero(neutral)
    if len(xx) < 30:
        return None
    indices = np.linspace(0, len(xx) - 1, min(8192, len(xx))).astype(int)
    xx, yy = xx[indices], yy[indices]
    design = np.column_stack([np.ones(len(xx)), xx / lab.shape[1], yy / lab.shape[0]])
    colors = lab[yy, xx].astype(np.float64)
    keep = np.ones(len(xx), bool)
    for _ in range(3):
        coefficients, _, rank, _ = np.linalg.lstsq(design[keep], colors[keep], rcond=None)
        if rank < 3:
            return None
        residual = np.linalg.norm(colors - design @ coefficients, axis=1)
        keep = residual <= max(1., float(np.quantile(residual, .8)))
    tolerance = float(np.clip(2 * np.quantile(residual, .7) + 6, 10, 24))
    return coefficients, tolerance, float(np.median(residual))


def _evaluate(frame, lab, quad):
    x, y, width, height = cv2.boundingRect(quad)
    interior = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(interior, np.rint(quad - (x, y)).astype(np.int32), 255)
    interior = cv2.erode(interior, np.ones((9, 9), np.uint8))
    local_hsv = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2HSV)
    neutral = (interior > 0) & (local_hsv[:, :, 1] <= 85) & (local_hsv[:, :, 2] >= 50)
    if np.count_nonzero(neutral) < .55 * np.count_nonzero(interior):
        return None
    model = _paper_color_model(lab[y:y + height, x:x + width], neutral)
    if model is None:
        return None
    coefficients, tolerance, model_residual = model
    lines, supports, contrasts = [], [], []
    center = quad.mean(axis=0)
    offsets = np.arange(-6, 7, dtype=np.float32)
    fractions = np.linspace(.035, .965, 65)
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        direction = end - start
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        if np.dot(normal, (start + end) / 2 - center) < 0:
            normal = -normal
        base = start + fractions[:, None] * direction
        profiles = _sample(lab, base[:, None] + offsets[None, :, None] * normal).astype(np.float32)
        profiles = cv2.GaussianBlur(profiles, (3, 1), 0)
        gradient = np.linalg.norm(np.diff(profiles, axis=1), axis=2)
        indices = gradient.argmax(axis=1)
        inside_outside = _sample(lab, np.stack([base - 8 * normal, base + 8 * normal], axis=1)).astype(np.float32)
        contrast = np.linalg.norm(inside_outside[:, 0] - inside_outside[:, 1], axis=1)
        inside_points = base - 8 * normal
        design = np.column_stack([np.ones(len(base)), (inside_points[:, 0] - x) / width,
                                   (inside_points[:, 1] - y) / height])
        expected_color = design @ coefficients
        paper_side = np.linalg.norm(inside_outside[:, 0] - expected_color, axis=1) <= tolerance
        usable = ((gradient[np.arange(len(indices)), indices] >= 2) & (indices > 0)
                  & (indices < len(offsets) - 2) & (contrast >= 6) & paper_side)


        if usable.mean() < .7 or usable[:6].sum() < 3 or usable[-6:].sum() < 3:
            return None
        boundary = base[usable] + (offsets[indices[usable]] + .5)[:, None] * normal
        vx, vy, px, py = cv2.fitLine(boundary.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        residuals = np.abs((boundary[:, 0] - px) * vy - (boundary[:, 1] - py) * vx)
        if np.quantile(residuals, .8) > 1.5:
            return None
        lines.append(np.array([-vy, vx, vy * px - vx * py]))
        supports.append(float(usable.mean()))
        contrasts.append(float(np.median(contrast[usable])))
    crossings = np.cross(np.roll(lines, 1, axis=0), lines)
    if np.any(np.abs(crossings[:, 2]) < .05):
        return None
    refined = (crossings[:, :2] / crossings[:, 2, None]).astype(np.float32)
    if (not cv2.isContourConvex(refined) or np.max(np.linalg.norm(refined - quad, axis=1)) > 10
            or np.any(refined < 5) or np.any(refined > np.array(frame.shape[1::-1]) - 6)):
        return None
    score = float(np.mean(supports) + .25 * np.mean(np.minimum(contrasts, 40)) / 40)
    return refined, {"score": score, "side_support": supports, "side_contrast": contrasts,
                     "paper_color_model": "affine_lab", "paper_model_residual": model_residual,
                     "paper_color_tolerance": tolerance}


def find_paper_boundary(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    horizontal, vertical = _lines(gray, lab)
    hypotheses = _quadrilaterals(horizontal, vertical, gray.shape)
    report = {"horizontal_lines": len(horizontal), "vertical_lines": len(vertical),
              "evaluated_quads": len(hypotheses), "accepted_quads": 0, "selected": None}
    groups = []
    for _, quad in hypotheses:
        result = _evaluate(frame, lab, quad)
        if result is None:
            continue
        points, measurements = result
        diagonal = .5 * (np.linalg.norm(points[0] - points[2]) + np.linalg.norm(points[1] - points[3]))
        duplicate = next((i for i, (other, _) in enumerate(groups)
                          if np.max(np.linalg.norm(points - other, axis=1)) < .025 * diagonal), None)
        if duplicate is None:
            groups.append(result)
        elif measurements["score"] > groups[duplicate][1]["score"]:
            groups[duplicate] = result
    report["accepted_quads"] = len(groups)
    groups.sort(key=lambda entry: entry[1]["score"], reverse=True)
    if not groups:
        report["reason"] = "Keine ausreichend gestützten Papiergrenzen gefunden."
        return None, report
    best, measurements = groups[0]
    if any(cv2.contourArea(other) >= .35 * cv2.contourArea(best)
           and info["score"] >= .8 * measurements["score"] for other, info in groups[1:]):
        report["reason"] = "Mehrere plausible Flächen; bitte manuell kalibrieren."
        return None, report
    report.update(selected=measurements, reason="Fläche anhand von Rand- und Farbkontrasten gefunden.")
    return best, report

from dataclasses import dataclass

import cv2
import numpy as np

from objekterkennung.geometry.calibration import PlaneCalibration
from objekterkennung.geometry.paper_boundary import find_paper_boundary, order_corners as _ordered


@dataclass(frozen=True)
class PaperSearch:
    calibration: PlaneCalibration | None
    reason: str
    method: str = "brightness"
    details: dict | None = None


def _refine_edges(gray, points):
    lines = []
    partial_edges = 0
    center = points.mean(axis=0)
    offsets = np.arange(-8, 9, dtype=np.float32)
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 30:
            return None
        normal = np.array([-direction[1], direction[0]]) / length
        if np.dot(normal, (start + end) / 2 - center) < 0:
            normal = -normal
        base = start + np.linspace(.12, .88, 65)[:, None] * direction
        samples = base[:, None, :] + offsets[None, :, None] * normal
        profiles = cv2.remap(gray, samples[:, :, 0].astype(np.float32),
                            samples[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
        profiles = cv2.GaussianBlur(profiles.astype(np.float32), (3, 1), 0)
        drops = profiles[:, :-1] - profiles[:, 1:]
        indices = drops.argmax(axis=1)
        strengths = drops[np.arange(len(indices)), indices]
        usable = (strengths >= 12) & (indices > 0) & (indices < len(offsets) - 2)
        if usable.sum() < .7 * len(base):


            support = np.flatnonzero(usable)
            if (len(support) < .5 * len(base) or np.ptp(support) < .5 * (len(base) - 1)
                    or partial_edges >= 1):
                return None
            partial_edges += 1
        boundary = base[usable] + (offsets[indices[usable]] + .5)[:, None] * normal
        vx, vy, x, y = cv2.fitLine(boundary.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        residuals = np.abs((boundary[:, 0] - x) * vy - (boundary[:, 1] - y) * vx)
        if np.quantile(residuals, .8) > 1.5:
            return None
        lines.append(np.array([-vy, vx, vy * x - vx * y]))
    refined = []
    for previous, current in zip(np.roll(lines, 1, axis=0), lines):
        intersection = np.cross(previous, current)
        if abs(intersection[2]) < 1e-4:
            return None
        refined.append(intersection[:2] / intersection[2])
    refined = np.array(refined, np.float32)
    if np.max(np.linalg.norm(refined - points, axis=1)) > 20:
        return None
    return refined


def _diagonal(points):
    p = np.asarray(points)
    return float((np.linalg.norm(p[0] - p[2]) + np.linalg.norm(p[1] - p[3])) / 2)


def _distance(first, second):
    return float(np.max(np.linalg.norm(np.asarray(first) - np.asarray(second), axis=1)))


def find_paper(frame: np.ndarray) -> PaperSearch:
    height, width = frame.shape[:2]
    scale = min(1., 1280 / max(width, height))
    small = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]

    neutral = gray[(gray > 30) & (saturation <= 85)]
    if neutral.size < 100:
        return PaperSearch(None, "Zu wenig helle, wenig gesättigte Bildfläche für die Blattsuche.")
    threshold, _ = cv2.threshold(neutral.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    center = int(np.clip(threshold, 60, 210))
    groups = []
    for threshold in (max(30, center - 40), center - 20, center, center + 20):
        mask = ((gray > threshold) & (saturation <= 85)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not .01 * gray.size <= area <= .65 * gray.size:
                continue


            hull = cv2.convexHull(contour)
            quad = cv2.approxPolyDP(hull, .02 * cv2.arcLength(hull, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad) or area / cv2.contourArea(quad) < .8:
                continue
            points = _ordered(quad.reshape(4, 2).astype(np.float32))
            if points is None or np.any(points < 10) or np.any(points > np.array(gray.shape[::-1]) - 11):
                continue
            refined = _refine_edges(gray, points)
            if refined is None or not cv2.isContourConvex(refined):
                continue
            match = next((group for group in groups if _distance(group[0][1], refined) < .025 * _diagonal(refined)), None)
            if match is None:
                groups.append([(threshold, refined)])
            elif threshold not in [entry[0] for entry in match]:
                match.append((threshold, refined))
    # Require agreement across brightness thresholds to reject unstable quadrilaterals.
    groups = [group for group in groups if len(group) >= 2]
    method, details = "brightness", None
    if groups:
        groups.sort(key=lambda group: cv2.contourArea(group[0][1]), reverse=True)
        if len(groups) > 1 and cv2.contourArea(groups[1][0][1]) >= .35 * cv2.contourArea(groups[0][0][1]):
            return PaperSearch(None, "Mehrere ähnlich große helle Flächen gefunden; bitte manuell kalibrieren.")
        points = np.median([entry[1] for entry in groups[0]], axis=0)
    else:
        # Fall back to boundary and color contrast if brightness thresholds find no stable plane.
        points, details = find_paper_boundary(small)
        method = "boundary_color"
        if points is None:
            return PaperSearch(None, details["reason"], method, details)

    points = (points + .5) * np.array([width / small.shape[1], height / small.shape[0]]) - .5
    try:
        calibration = PlaneCalibration(tuple(map(tuple, points)), (width, height))
    except ValueError:
        return PaperSearch(None, "Die gefundenen Blattränder ergeben keine gültige Kalibrierung.", method, details)
    reason = ("Helles Blatt mit übereinstimmenden Randgeraden gefunden." if details is None else details["reason"])
    return PaperSearch(calibration, reason, method, details)


class AutomaticCalibration:

    CHANGE_RATIO = .015
    STABLE_RATIO = .005

    def __init__(self):
        self.enabled = False
        self.one_shot = False
        self.interval_s = 10
        self.reset()

    def reset(self):
        if self.one_shot:
            self.enabled = False
        self.one_shot = False
        self.next_check = 0.
        self.pending = []
        self.attempts = 0
        self.deadline = 0.
        self.blocked = False
        self.last_search = None
        self.status = "Automatik bereit; wartet auf ein Bild." if self.enabled else "Automatische Kalibrierung aus."

    def set_enabled(self, enabled: bool):
        self.one_shot = False
        self.enabled = enabled
        self.reset()

    def set_interval(self, seconds: int):
        if seconds not in (5, 10, 15):
            raise ValueError("Prüfintervall muss 5, 10 oder 15 Sekunden betragen.")
        self.interval_s = seconds
        self.request_check()

    def request_check(self):
        self.next_check = 0.
        self.pending = []
        self.attempts = 0

    def request_once(self):
        blocked = self.blocked
        self.set_enabled(True)
        self.blocked = blocked
        self.one_shot = True
        self.status = "Einmalige Kalibrierung angefordert; wartet auf ein Bild."

    def check(self, frame, current, now: float, *, single_frame=False):
        if not self.enabled or (not self.attempts and now < self.next_check):
            return None
        if self.attempts and now > self.deadline:
            self._finish(now, "Keine zeitnahe Bestätigung; erneute Flächenprüfung folgt." +
                         (" Positionsausgabe pausiert." if self.blocked else ""))
            return None
        if not self.attempts:
            self.deadline = now + 3
        self.attempts += 1
        found = find_paper(frame)
        self.last_search = {"method": found.method, "reason": found.reason, "details": found.details}
        proposed = found.calibration
        if proposed is not None:
            points = proposed.corners_px
            if (not self.one_shot and current is not None
                    and _distance(points, current.corners_px) <= self.CHANGE_RATIO * _diagonal(current.corners_px)):
                self.blocked = False
                self._finish(now, "Fläche unverändert; bisherige Kalibrierung bleibt fest.")
                return None
            # Pause positions while a changed plane awaits confirmation across fresh frames.
            self.blocked = current is not None
            if self.pending and _distance(points, self.pending[0]) > self.STABLE_RATIO * _diagonal(points):
                self.pending = []
            self.pending.append(points)
            if single_frame or len(self.pending) >= 3:
                averaged = tuple(map(tuple, np.median(self.pending, axis=0)))
                result = PlaneCalibration(averaged, proposed.image_size_px)
                self.blocked = False
                self._finish(now, "Neue Fläche automatisch kalibriert; Raster bitte prüfen.")
                return result
            self.status = f"Fläche gefunden; stabile Ecken bestätigen ({len(self.pending)}/3)."
        else:
            self.pending = []
            self.status = found.reason
        if single_frame or self.attempts >= 6 or now >= self.deadline:
            note = (" Bewegung noch ungeklärt; Positionsausgabe pausiert." if self.blocked else
                    " Bisherige Kalibrierung bleibt erhalten." if current is not None else
                    " Keine Kalibrierung übernommen; manuelle Auswahl möglich.")
            self._finish(now, self.status + note)
        return None

    def _finish(self, now, status):
        self.next_check = now + self.interval_s
        self.pending = []
        self.attempts = 0
        self.status = status
        if self.one_shot:
            self.enabled = False
            self.one_shot = False
            self.status = "Einmalige Prüfung beendet: " + status

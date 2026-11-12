from dataclasses import dataclass

from objekterkennung.vision.shapes import ShapePrediction


def _iou(first, second):
    x, y, w, h = first
    a, b, c, d = second
    overlap = max(0, min(x + w, a + c) - max(x, a)) * max(0, min(y + h, b + d) - max(y, b))
    return overlap / (w * h + c * d - overlap)


@dataclass
class _Track:
    color: str
    bbox: tuple
    label: str
    streak: int
    confirmed: str | None
    confirmed_at: float
    last_seen: float
    gaps: int = 0


class ShapeConfirmation:

    def __init__(self):
        self.clear()

    def clear(self):
        self._tracks = []

    def apply(self, candidates, predictions, now: float):
        eligible = [track for track in self._tracks if 0 <= now - track.last_seen <= .8]

        edges = sorted((( _iou(candidate.bbox_xywh, track.bbox), i, j)
                        for i, candidate in enumerate(candidates) for j, track in enumerate(eligible)
                        if candidate.color == track.color), reverse=True)
        assigned, used = {}, set()
        for overlap, i, j in edges:
            if overlap >= .5 and i not in assigned and j not in used:
                assigned[i] = eligible[j]
                used.add(j)
        output, current = [], []
        for i, (candidate, prediction) in enumerate(zip(candidates, predictions)):
            previous = assigned.get(i)
            track = _Track(candidate.color, candidate.bbox_xywh, prediction.label, 1, None, now, now)
            shown = prediction
            if previous is not None:
                if prediction.label != "unknown":
                    track.streak = previous.streak + 1 if previous.label == prediction.label else 1
                    if track.streak >= 2:
                        track.confirmed, track.confirmed_at = prediction.label, now
                else:
                    track.streak = 0
                    track.confirmed, track.confirmed_at = previous.confirmed, previous.confirmed_at
                    track.gaps = previous.gaps + 1
                    features = prediction.features
                    if (track.confirmed is not None and track.gaps <= 1 and 0 <= now - track.confirmed_at <= .8
                            and features is not None and features.solidity >= .93
                            and min(features.width_px, features.height_px) >= 20):
                        shown = ShapePrediction(track.confirmed,
                            "Zeitlich bestätigt; aktuelle Einzelbildform unklar. " + prediction.reason, features, True)
            output.append(shown)
            current.append(track)

        self._tracks = current
        return output


class DetectionPacing:
    def __init__(self):
        self.enabled = False
        self.interval_s = 1.
        self.reset()

    def reset(self):
        self.last_finished = None

    def configure(self, enabled: bool, interval_s: float):
        if interval_s not in (.5, 1., 2.):
            raise ValueError("Erkennungspause muss 0,5, 1 oder 2 Sekunden betragen.")
        self.enabled, self.interval_s = enabled, interval_s
        self.reset()

    def due(self, now: float) -> bool:
        return not self.enabled or self.last_finished is None or now - self.last_finished >= self.interval_s

    def completed(self, now: float):
        self.last_finished = now

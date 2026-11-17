from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from objekterkennung.geometry.positions import estimate_position
from objekterkennung.tracking import _iou
from objekterkennung.vision.field_filter import filter_to_field
from objekterkennung.vision.segmentation import Candidate
from objekterkennung.vision.shapes import ShapePrediction


@dataclass
class _Known:
    candidate: Candidate
    prediction: ShapePrediction
    anchor: np.ndarray
    analyzed_at: float
    checked_at: float
    failures: int = 0


class AdaptiveDetection:

    DISCOVERY_INTERVAL = .2
    FULL_INTERVAL = 10.

    def __init__(self):
        self.enabled = False
        self.known_interval_s = 1.
        self.unknown_interval_s = .2
        self.field_filter_enabled = True
        self.reset()

    def configure(self, enabled, known_interval_s, unknown_interval_s):
        if (type(enabled) is not bool or type(known_interval_s) is bool
                or known_interval_s not in (.5, 1., 2.) or type(unknown_interval_s) is bool
                or unknown_interval_s not in (.2, .5, 1.) or unknown_interval_s > known_interval_s):
            raise ValueError("Bekannt: 0,5/1/2 s; unbekannt: 0,2/0,5/1 s und höchstens das bekannte Intervall.")
        self.enabled = enabled
        self.known_interval_s, self.unknown_interval_s = known_interval_s, unknown_interval_s
        self.reset()

    def reset(self):
        self.result = None
        self.known = []
        self.background = None
        self.setup = None
        self.last_probe = self.last_run = self.last_full = None
        self.has_unknown = False
        self.report = None
        self.full_runs = self.partial_runs = self.cheap_checks = 0
        self._fresh = []
        self._was_full = False

    @staticmethod
    def _probe(frame):
        scale = min(1., 480 / frame.shape[1])
        small = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)),
                                   max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (3, 3), .8)

    @staticmethod
    def _rect(candidate, frame_shape, probe_shape, padding=.15):
        x, y, width, height = candidate.bbox_xywh
        margin = max(4, padding * max(width, height))
        sx, sy = probe_shape[1] / frame_shape[1], probe_shape[0] / frame_shape[0]
        left, top = max(0, int((x - margin) * sx)), max(0, int((y - margin) * sy))
        right = min(probe_shape[1], max(left + 1, int(np.ceil((x + width + margin) * sx))))
        bottom = min(probe_shape[0], max(top + 1, int(np.ceil((y + height + margin) * sy))))
        return slice(top, bottom), slice(left, right)

    def _changed(self, probe, track, frame_shape, *, emergency=False):
        rows, cols = self._rect(track.candidate, frame_shape, probe.shape)
        delta = np.max(cv2.absdiff(probe[rows, cols], track.anchor[rows, cols]), axis=2)


        return np.mean(delta > (40 if emergency else 18)) > (.3 if emergency else .06)

    def _outside_changed(self, probe, frame_shape):
        outside = np.ones(probe.shape[:2], np.uint8)
        for candidate in self.result.segmentation.candidates:
            outside[self._rect(candidate, frame_shape, probe.shape)] = 0
        delta = np.max(cv2.absdiff(probe, self.background), axis=2)
        changed = ((delta > 18) & (outside > 0)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(changed)
        return bool(count > 1 and np.max(stats[1:, cv2.CC_STAT_AREA]) >= max(6, .00008 * changed.size))

    def step(self, frame, config, calibration, geometry, now, *, detector):
        setup = (config, calibration, geometry, frame.shape, self.field_filter_enabled)
        if self.setup != setup or (self.last_probe is not None and now < self.last_probe):
            self.reset()
            self.setup = setup
        # Periodic full analysis limits the lifetime of cached object evidence.
        scheduled = (self.last_full is None or now - self.last_full >= self.FULL_INTERVAL
                     or (self.has_unknown and self.last_run is not None
                         and now - self.last_run >= self.unknown_interval_s - 1e-9)
                     or any(now - track.checked_at >= self.known_interval_s - 1e-9 for track in self.known))
        if (not scheduled and self.last_probe is not None
                and now - self.last_probe < self.DISCOVERY_INTERVAL - 1e-9):
            return None
        probe = self._probe(frame)
        self.last_probe = now
        self.cheap_checks += 1
        changed = self.result is None
        if self.result is not None:
            changed = self._outside_changed(probe, frame.shape)
            for track in self.known:
                ordinary_due = now - track.checked_at >= self.known_interval_s - 1e-9
                changed |= self._changed(probe, track, frame.shape, emergency=not ordinary_due)
                if ordinary_due:
                    track.checked_at = now
        full = changed or self.last_full is None or now - self.last_full >= self.FULL_INTERVAL
        retry = self.has_unknown and (self.last_run is None or now - self.last_run >= self.unknown_interval_s - 1e-9)
        if not full and not retry:
            return None

        # Partial retries reuse stable known instances while unknown regions are analyzed again.
        reusable = None
        if not full:


            reusable = replace(self.result.segmentation, candidates=[
                replace(track.candidate, reuse_age_s=max(0., now - track.analyzed_at))
                for track in self.known if not track.failures
                and not self._changed(probe, track, frame.shape)
                and now - track.analyzed_at < self.FULL_INTERVAL])
        raw = (detector(frame, config, calibration, geometry) if full else
               detector(frame, config, calibration, geometry, reusable=reusable))
        visible = filter_to_field(raw, enabled=self.field_filter_enabled, calibration=calibration)
        self.has_unknown = any(prediction.label == "unknown" for prediction in visible.shapes)
        shapes, positions = list(raw.shapes), list(raw.positions)
        tracks, decisions = [], []
        available = list(self.known)
        self._fresh = []
        for index, (candidate, prediction) in enumerate(zip(raw.segmentation.candidates, raw.shapes)):
            matches = [track for track in available if track.candidate.color == candidate.color
                       and _iou(track.candidate.bbox_xywh, candidate.bbox_xywh) >= .7]
            previous = max(matches, key=lambda track: _iou(track.candidate.bbox_xywh, candidate.bbox_xywh)) if matches else None
            if previous is not None:
                available = [track for track in available if track is not previous]
            cached = candidate.reuse_age_s is not None
            pending = False
            if cached and previous is not None:
                track = replace(previous, candidate=candidate)
                tracks.append(track)
                shapes[index] = replace(prediction, reason="Aus unverändertem Bildbereich übernommen. " + prediction.reason)
            elif prediction.label != "unknown":
                track = _Known(candidate, prediction, probe, now, now)
                tracks.append(track)
                self._fresh.append(track)
            elif (previous is not None and not changed and previous.failures == 0
                  and prediction.features is not None and prediction.features.solidity >= .93
                  and not self._changed(probe, previous, frame.shape)):


                pending = True
                shapes[index] = replace(prediction, label=previous.prediction.label, temporally_confirmed=True,
                    reason="Bisherige Form kurz bestätigt; erneute Prüfung folgt. " + prediction.reason)
                positions[index] = estimate_position(candidate.contour, shapes[index].label, calibration, geometry)
                tracks.append(replace(previous, candidate=candidate, failures=1))
            decisions.append({"number": index + 1, "color": candidate.color,
                "bbox_xywh": list(candidate.bbox_xywh), "raw_shape": None if cached else prediction.label,
                "display_shape": shapes[index].label, "reused": cached,
                "analysis_age_s": candidate.reuse_age_s or 0., "verification_pending": pending,
                "display_position": asdict(positions[index])})
        self.known = tracks
        self.result = replace(raw, shapes=shapes, positions=positions)
        self.last_run = now
        self._was_full = full
        if full:
            self.last_full, self.background = now, probe
            self.full_runs += 1
        else:
            self.partial_runs += 1
        self.report = {"mode": "full" if full else "unknown_retry", "scene_changed": bool(changed),
                       "reused_count": sum(row["reused"] for row in decisions), "objects": decisions}
        return self.result

    def completed(self, now):
        self.last_run = now
        if self._was_full:
            self.last_full = now
        for track in self._fresh:
            track.analyzed_at = track.checked_at = now

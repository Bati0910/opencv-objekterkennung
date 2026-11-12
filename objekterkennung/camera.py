from objekterkennung.configuration import SETUP

from dataclasses import dataclass
import sys
from threading import Event, Lock, Thread
from time import monotonic

import cv2
import numpy as np


def validate_camera_frame(frame: np.ndarray) -> None:
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 2):
        raise ValueError("Die Kamera liefert kein gültiges 8-Bit-BGR-Bild.")


@dataclass(frozen=True)
class CameraSnapshot:

    status: str
    frame: np.ndarray | None
    sequence: int
    backend: str
    error: str | None


class CameraStream:

    def __init__(self, index: int, *, width: int = SETUP.camera.width, height: int = SETUP.camera.height,
                 fps: int = SETUP.camera.fps, capture_factory=None, backends: tuple[int, ...] | None = None) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Der Kameraindex muss eine ganze Zahl ab 0 sein.")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (width, height, fps)):
            raise ValueError("Auflösung und gewünschte Bildrate müssen positive Ganzzahlen sein.")
        self.index = index
        self.width, self.height, self.fps = width, height, fps
        self._factory = capture_factory or cv2.VideoCapture
        self._backends = backends if backends is not None else (
            (cv2.CAP_DSHOW, cv2.CAP_MSMF) if sys.platform == "win32" else (cv2.CAP_ANY,))
        if not self._backends:
            raise ValueError("Mindestens ein Kamera-Backend ist erforderlich.")
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._status = "idle"
        self._frame = None
        self._sequence = 0
        self._last_frame_time = None
        self._backend = ""
        self._error = None

    def start(self) -> None:
        with self._lock:
            if self._status != "idle":
                raise ValueError("Dieser Kamerastream wurde bereits gestartet.")
            self._status = "opening"
        self._thread = Thread(target=self._run, name=f"Camera-{self.index}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 1.0) -> bool:
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.is_alive

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> CameraSnapshot:
        with self._lock:
            status = self._status
            if status == "running" and monotonic() - self._last_frame_time > 2:
                status = "waiting"
            return CameraSnapshot(status, None if self._frame is None else self._frame.copy(),
                                  self._sequence, self._backend, self._error)

    def _run(self) -> None:
        capture = None
        try:
            for backend in self._backends:
                if self._stop.is_set():
                    return
                capture = self._factory(self.index, backend)
                if capture.isOpened():
                    break
                capture.release()
                capture = None
            if capture is None:
                raise ValueError(f"Kamera {self.index} konnte nicht geöffnet werden. Anschluss, Kameraindex und andere Kamera-Apps prüfen.")

            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            with self._lock:
                self._backend = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "AUTO"}.get(backend, str(backend))
            failures = 0
            while not self._stop.is_set():
                success, frame = capture.read()
                if self._stop.is_set():
                    break
                if not success or frame is None:
                    failures += 1
                    with self._lock:
                        self._status = "waiting"
                    if failures >= 5:
                        raise ValueError("Kein Kamerabild empfangen. Kamera getrennt, belegt oder nicht verfügbar.")
                    self._stop.wait(0.05)
                    continue
                validate_camera_frame(frame)
                failures = 0
                with self._lock:
                    self._frame = frame.copy()
                    self._sequence += 1
                    self._last_frame_time = monotonic()
                    self._status = "running"
        except (OSError, ValueError, cv2.error) as error:
            with self._lock:
                self._error = str(error)
        finally:
            if capture is not None:
                try:
                    capture.release()
                except (OSError, cv2.error) as error:
                    with self._lock:
                        self._error = str(error)
            with self._lock:
                self._frame = None
                self._status = "error" if self._error else "stopped"

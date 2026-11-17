from pathlib import Path
from dataclasses import replace
import sys
from time import monotonic
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2

from objekterkennung.configuration import CALIBRATION_DIRECTORY
from objekterkennung.session import ImageSession, display_to_image
from objekterkennung.camera import CameraStream
from objekterkennung.vision.field_filter import filter_to_field, field_filter_summary
from objekterkennung.geometry.positions import load_geometry
from objekterkennung.vision.segmentation import load_config
from objekterkennung.ui.view import DetectionView


class DetectionApp:
    def __init__(self, root):
        self.config = load_config()
        self.geometry = load_geometry()
        self.root = root
        self.session = ImageSession()
        self.camera = None
        self.camera_job = None
        self.retiring_camera = None
        self.camera_sequence = 0
        self.camera_ready = False
        self.camera_note = ""
        self.live_detection = False
        self.live_config = None
        self.live_geometry = None
        self.closing = False
        self.display_result = None
        self.filter_summary = ""
        self._display_source = None
        self._display_calibration = None
        self._display_filter = None
        self.view = DetectionView(root, self)
        self.sync_region_filter_controls(self.config)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()

    def report_error(self, error: Exception) -> None:
        messagebox.showerror("Aktion nicht möglich", str(error), parent=self.root)

    def open_image(self, path: Path) -> None:
        try:
            profile = self.config
            self.session.open_image(path)
            self.stop_camera(clear=False)
            self.sync_region_filter_controls(profile)
            if self.session.auto_calibration.enabled:
                self.session.check_automatic_calibration(monotonic(), single_frame=True)
        except (OSError, ValueError, cv2.error) as error:
            self.report_error(error)
        self.refresh()

    def choose_image(self) -> None:
        name = filedialog.askopenfilename(parent=self.root, title="Bild öffnen",
            filetypes=[("Bilder", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Alle Dateien", "*.*")])
        if name:
            self.open_image(Path(name))

    def begin_calibration(self) -> None:
        self.live_detection = False
        self.session.begin_calibration()
        self.refresh()

    def toggle_auto_calibration(self) -> None:
        self.session.set_automatic_calibration(self.view.auto_calibration_enabled.get())
        if self.session.auto_calibration.enabled:
            self.session.auto_calibration.request_check()

            if self.camera is None:
                self.session.check_automatic_calibration(monotonic(), single_frame=True)
        if self.session.calibration is None:
            self.live_detection = False
        self.refresh()

    def calibrate_once(self) -> None:
        self.session.request_single_calibration()
        if self.camera is None:
            self.session.check_automatic_calibration(monotonic(), single_frame=True)
        self.refresh()

    def change_detection_rate(self, event: tk.Event | None = None) -> None:
        seconds = float(self.view.detection_interval.get().split()[0].replace(",", "."))
        if self.view.reduce_rate.get():
            self.session.adaptive_detection.configure(False, self.session.adaptive_detection.known_interval_s,
                                                       self.session.adaptive_detection.unknown_interval_s)
            self.view.adaptive_enabled.set(False)
        self.session.detection_pacing.configure(self.view.reduce_rate.get(), seconds)
        self.session._reset_confirmation()
        self.refresh()

    def change_adaptive_detection(self, event: tk.Event | None = None) -> None:
        known = float(self.view.known_interval.get().split()[0].replace(",", "."))
        unknown = float(self.view.unknown_interval.get().split()[0].replace(",", "."))
        if unknown > known:
            unknown = known
            self.view.unknown_interval.set("0,5 s" if unknown == .5 else "1 s")
        self.session.adaptive_detection.configure(self.view.adaptive_enabled.get(), known, unknown)
        if self.view.adaptive_enabled.get():
            self.view.reduce_rate.set(False)
            self.session.detection_pacing.configure(False, self.session.detection_pacing.interval_s)
        self.session._reset_confirmation()
        self.session.detection_pacing.reset()
        self.refresh()

    def change_field_filter(self) -> None:
        self.session.adaptive_detection.field_filter_enabled = self.view.field_filter_enabled.get()
        self.session.adaptive_detection.reset()
        self.refresh()

    def effective_config(self, config):
        return replace(config, **self.session.region_filter_overrides)

    def sync_region_filter_controls(self, profile) -> None:
        config = self.effective_config(profile)
        self.view.thin_regions_enabled.set(config.reject_thin_regions)
        self.view.field_filling_enabled.set(config.reject_field_filling_regions)
        self.view.touching_enabled.set(config.separate_touching_regions)

    def change_region_filters(self) -> None:
        try:
            profile = self.live_config if self.live_detection else self.config
            flags = {"reject_thin_regions": self.view.thin_regions_enabled.get(),
                     "reject_field_filling_regions": self.view.field_filling_enabled.get(),
                     "separate_touching_regions": self.view.touching_enabled.get()}
            config = replace(profile, **flags)
            had_result = self.session.result is not None
            self.session.region_filter_overrides = flags
            self.live_config = config
            self.session._reset_confirmation()
            self.session.detection_pacing.reset()
            if had_result:
                self.session.show_preview()
                if self.camera is None and self.session.calibration is not None and not self.session.auto_calibration.blocked:
                    self.session.detect(config, self.geometry)


        except (OSError, ValueError, cv2.error) as error:
            self.report_error(error)
        self.refresh()

    def change_auto_interval(self, event: tk.Event | None = None) -> None:
        self.session.auto_calibration.set_interval(int(self.view.auto_interval.get()))
        self.refresh()

    def check_auto_now(self) -> None:
        self.session.auto_calibration.request_check()
        if self.camera is None:
            self.session.check_automatic_calibration(monotonic(), single_frame=True)
        self.refresh()

    def undo_corner(self) -> None:
        self.live_detection = False
        self.session.undo_corner()
        self.refresh()

    def load_calibration(self) -> None:
        name = filedialog.askopenfilename(parent=self.root, title="Passende Kalibrierung laden",
            initialdir=CALIBRATION_DIRECTORY, filetypes=[("Kalibrierung", "*.json")])
        if not name:
            return
        try:
            self.session.load_calibration(Path(name))
            self.live_detection = False
        except (OSError, ValueError) as error:
            self.report_error(error)
        self.refresh()

    def save_calibration(self) -> None:
        if self.session.calibration is None:
            return
        try:
            CALIBRATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
            name = filedialog.asksaveasfilename(parent=self.root, title="Kalibrierung speichern",
                initialdir=CALIBRATION_DIRECTORY, initialfile="calibration.json", defaultextension=".json",
                filetypes=[("Kalibrierung", "*.json")], confirmoverwrite=True)
            if not name:
                return
            self.session.calibration.save(Path(name), overwrite=Path(name).exists())
        except (OSError, ValueError) as error:
            self.report_error(error)

    def detect(self) -> None:
        try:
            config, geometry = self.effective_config(self.config), self.geometry
            self.session.adaptive_detection.reset()
            self.session.detect(config, geometry, live=self.camera is not None)
            self.live_config, self.live_geometry = config, geometry
            self.live_detection = self.camera is not None
            if self.live_detection:
                self.session.detection_pacing.completed(monotonic())
        except (OSError, ValueError, cv2.error) as error:
            self.report_error(error)
        self.refresh()

    def show_preview(self) -> None:
        self.live_detection = False
        self.session.show_preview()
        self.refresh()

    def start_camera(self) -> None:
        if self.camera is not None or self.retiring_camera is not None:
            return
        try:
            index = int(self.view.camera_index.get())
            profile = self.config
            stream = CameraStream(index)
            stream.start()
        except (OSError, ValueError, RuntimeError) as error:
            self.report_error(error)
            return
        # Switch sources only after validation and the immediate thread start succeed.
        self.session.clear()
        self.sync_region_filter_controls(profile)
        self.live_detection = False
        self.camera_sequence = 0
        self.camera_ready = False
        self.camera_note = f"Kamera {index} wird geöffnet …"
        self.camera = stream
        self.refresh()
        self.camera_job = self.root.after(50, self.poll_camera)

    def poll_camera(self) -> None:
        self.camera_job = None
        if self.camera is None or self.closing:
            return
        snapshot = self.camera.snapshot()
        if snapshot.status in ("error", "stopped"):
            error = snapshot.error or "Die Kameraaufnahme wurde beendet."
            self.stop_camera()
            self.camera_note = error
            self.refresh()
            self.report_error(ValueError(error))
            return
        try:
            self.camera_ready = snapshot.status == "running"
            if not self.camera_ready:
                self.session.result = None
                self.session.result_frame = None
                self.session._reset_confirmation()
                self.camera_note = (f"Kamera {self.camera.index} wird geöffnet …" if snapshot.status == "opening"
                                    else "Warte auf ein neues Kamerabild; keine aktuellen Erkennungsergebnisse.")
                self.refresh()
            elif snapshot.sequence != self.camera_sequence:
                pacing = self.session.detection_pacing
                adaptive = self.session.adaptive_detection.enabled
                hold = (pacing.enabled or adaptive) and self.live_detection
                previous_result = self.session.result
                previous_auto_status = self.session.auto_calibration.status
                self.camera_sequence = snapshot.sequence
                changed_size = self.session.update_live_frame(snapshot.frame, keep_result=hold)
                if changed_size:
                    self.live_detection = False
                now = monotonic()
                self.session.check_automatic_calibration(now)
                analyzed = False
                if (self.live_detection and not self.session.auto_calibration.blocked
                        and (adaptive or self.session.result is None or pacing.due(now))):
                    analyzed = self.session.detect(self.live_config, self.live_geometry, live=True)
                    if analyzed:
                        pacing.completed(monotonic() if pacing.enabled else now)
                height, width = snapshot.frame.shape[:2]
                self.camera_note = f"Kamera {self.camera.index} · {width} × {height}"
                if changed_size:
                    self.camera_note += " · Auflösung geändert: bitte neu kalibrieren."
                if (not hold or analyzed or changed_size or self.session.result is not previous_result
                        or self.session.auto_calibration.status != previous_auto_status):
                    self.refresh()
        except (OSError, ValueError, cv2.error) as error:
            self.live_detection = False
            self.session.result = None
            self.session.show_preview()
            self.refresh()
            self.report_error(error)
        if self.camera is not None and not self.closing:
            self.camera_job = self.root.after(80, self.poll_camera)

    def stop_camera(self, *, clear: bool = True) -> None:
        if self.camera_job is not None:
            self.root.after_cancel(self.camera_job)
            self.camera_job = None
        self.live_detection = False
        self.camera_ready = False
        self.camera_note = ""
        if self.camera is not None:
            self.camera.stop()
            self.retiring_camera = self.camera
            self.camera = None
            self.root.after(50, self.check_camera_stopped)
            if clear:
                self.session.clear()
        self.refresh()

    def check_camera_stopped(self) -> None:
        if self.closing:
            return
        if self.retiring_camera is not None and self.retiring_camera.is_alive:
            self.root.after(100, self.check_camera_stopped)
        else:
            self.retiring_camera = None
            self.refresh()

    def close(self) -> None:
        self.closing = True
        self.view.closed = True
        if self.camera_job is not None:
            self.root.after_cancel(self.camera_job)
            self.camera_job = None
        if self.view.redraw_job is not None:
            self.root.after_cancel(self.view.redraw_job)
            self.view.redraw_job = None
        stream = self.camera or self.retiring_camera
        if stream is not None:
            stream.stop()
        deadline = monotonic() + 2

        # Poll shutdown through Tk callbacks so a slow driver does not block the window.
        def finish() -> None:
            if stream is not None and stream.is_alive and monotonic() < deadline:
                self.root.after(50, finish)
            else:
                if stream is not None and stream.is_alive:
                    print("Kameratreiber reagiert nicht auf Stopp; Prozess wird beendet.", file=sys.stderr)
                self.root.destroy()
        finish()

    def click_image(self, event: tk.Event) -> None:
        if self.session.stage != "calibrating" or self.view.image_mapping is None:
            return
        point = display_to_image(event.x, event.y, self.view.image_mapping)
        if point is None:
            return
        try:
            self.session.select_corner(point)
        except ValueError as error:
            self.report_error(error)
        self.refresh()

    def refresh(self):
        result = self.session.result
        calibration = self.session.calibration
        enabled = self.view.field_filter_enabled.get()
        # Share one filtered result between the table, status and image overlay.
        if (result is not self._display_source or calibration is not self._display_calibration
                or enabled != self._display_filter):
            self._display_source = result
            self._display_calibration = calibration
            self._display_filter = enabled
            self.display_result = filter_to_field(result, enabled=enabled, calibration=calibration) if result is not None else None
            self.filter_summary = field_filter_summary(result, enabled=enabled, calibration=calibration,
                                                       visible_result=self.display_result) if result is not None else ""
        self.view.refresh(self)


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"Die Oberfläche konnte nicht gestartet werden: {error}", file=sys.stderr)
        return 1
    try:
        DetectionApp(root)
    except (OSError, ValueError, cv2.error) as error:
        messagebox.showerror("Start nicht möglich", str(error), parent=root)
        root.destroy()
        return 1
    root.mainloop()
    return 0

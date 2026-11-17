import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from objekterkennung.configuration import SETUP
from objekterkennung.vision.shapes import APPROX_EPSILON_RATIO
DISPLAY_COLORS = {"red": (0, 0, 255), "green": (0, 180, 0),
                  "blue": (255, 100, 0), "yellow": (0, 210, 230)}


COLOR_NAMES = {"red": "Rot", "green": "Grün", "blue": "Blau", "yellow": "Gelb"}
SHAPE_NAMES = {"cube": "Würfel", "cylinder": "Zylinder", "cone": "Kegel",
               "hemisphere": "Halbkugel", "unknown": "Unbekannt"}


class DetectionView:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.photo = None
        self.image_mapping = None
        self.redraw_job = None
        self.closed = False
        self.result = None
        root.title("Farben und Körper · Objekterkennung")
        root.geometry("1200x820")
        root.minsize(960, 680)
        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=(8, 5))
        panel = ttk.Frame(root, padding=14)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(3, weight=1)
        ttk.Label(panel, text="Farben und Körper", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, text="Quelle wählen · Arbeitsfläche kalibrieren · Erkennung starten").grid(
            row=1, column=0, sticky="w", pady=(2, 10))
        source = ttk.LabelFrame(panel, text="Quelle", padding=8)
        source.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(source, text="Kamera:").pack(side="left")
        self.camera_index = tk.StringVar(root, value=str(SETUP.camera.index))
        self.camera_selector = ttk.Spinbox(source, from_=0, to=99, width=4, textvariable=self.camera_index)
        self.camera_selector.pack(side="left", padx=(6, 10))
        self.camera_button = ttk.Button(source, text="Kamera starten", command=app.start_camera)
        self.camera_button.pack(side="left", padx=(0, 6))
        self.stop_button = ttk.Button(source, text="Stoppen", command=app.stop_camera)
        self.stop_button.pack(side="left")
        ttk.Button(source, text="Bild öffnen …", command=app.choose_image).pack(side="left", padx=(16, 0))
        self.source_text = tk.StringVar(root, value="Keine Quelle ausgewählt")
        ttk.Label(source, textvariable=self.source_text).pack(side="right", padx=8)
        body = ttk.Frame(panel)
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body, width=310)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        modes = ttk.Notebook(sidebar, width=300, height=310)
        modes.pack(fill="both", expand=True)
        calibration = ttk.Frame(modes, padding=10)
        operation = ttk.Frame(modes, padding=10)
        filters = ttk.Frame(modes, padding=10)
        modes.add(calibration, text="Kalibrierung")
        modes.add(operation, text="Betrieb")
        modes.add(filters, text="Filter")
        ttk.Label(calibration, text="Arbeitsfläche festlegen", style="Section.TLabel").pack(anchor="w")
        self.calibrate_button = ttk.Button(calibration, text="Vier Ecken auswählen", command=app.begin_calibration)
        self.calibrate_button.pack(fill="x", pady=(8, 4))
        self.undo_button = ttk.Button(calibration, text="Letzte Ecke zurück", command=app.undo_corner)
        self.undo_button.pack(fill="x")
        self.calibrate_once_button = ttk.Button(calibration, text="Blatt einmal erkennen", command=app.calibrate_once)
        self.calibrate_once_button.pack(fill="x", pady=(8, 4))
        saved = ttk.Frame(calibration)
        saved.pack(fill="x", pady=(0, 10))
        self.load_button = ttk.Button(saved, text="Laden …", command=app.load_calibration)
        self.load_button.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.save_button = ttk.Button(saved, text="Speichern …", command=app.save_calibration)
        self.save_button.pack(side="left", expand=True, fill="x")
        ttk.Separator(calibration).pack(fill="x", pady=6)
        self.auto_calibration_enabled = tk.BooleanVar(root, value=False)
        side_cm = SETUP.surface_size_m[0] * 100
        ttk.Checkbutton(calibration, text=f"Helles {side_cm:g} × {side_cm:g} cm-Blatt überwachen",
                        variable=self.auto_calibration_enabled, command=app.toggle_auto_calibration).pack(anchor="w")
        interval = ttk.Frame(calibration)
        interval.pack(fill="x", pady=6)
        ttk.Label(interval, text="Prüfung alle").pack(side="left")
        self.auto_interval = tk.StringVar(root, value="10")
        self.auto_interval_selector = ttk.Combobox(interval, textvariable=self.auto_interval,
                                                  values=("5", "10", "15"), state="readonly", width=3)
        self.auto_interval_selector.pack(side="left", padx=6)
        self.auto_interval_selector.bind("<<ComboboxSelected>>", app.change_auto_interval)
        ttk.Label(interval, text="Sekunden").pack(side="left")
        self.auto_check_button = ttk.Button(calibration, text="Jetzt prüfen", command=app.check_auto_now)
        self.auto_check_button.pack(fill="x")
        self.auto_status = tk.StringVar(root)
        ttk.Label(calibration, textvariable=self.auto_status, wraplength=275).pack(anchor="w", pady=(8, 0))
        ttk.Label(operation, text="Aktualisierung", style="Section.TLabel").pack(anchor="w")
        ttk.Label(operation, text="Standard: jedes ausgewertete Kamerabild.", wraplength=275).pack(anchor="w", pady=8)
        self.reduce_rate = tk.BooleanVar(root, value=False)
        ttk.Checkbutton(operation, text="Pause zwischen Auswertungen", variable=self.reduce_rate,
                        command=app.change_detection_rate).pack(anchor="w")
        self.detection_interval = tk.StringVar(root, value="1 s")
        self.detection_interval_selector = ttk.Combobox(operation, textvariable=self.detection_interval,
                                                       values=("0,5 s", "1 s", "2 s"), state="disabled", width=7)
        self.detection_interval_selector.pack(anchor="w", pady=(4, 12))
        self.detection_interval_selector.bind("<<ComboboxSelected>>", app.change_detection_rate)
        self.adaptive_enabled = tk.BooleanVar(root, value=False)
        ttk.Checkbutton(operation, text="Adaptiv aktualisieren", variable=self.adaptive_enabled,
                        command=app.change_adaptive_detection).pack(anchor="w")
        for name, label, initial, values in (
                ("known", "Bekannte Körper", "1 s", ("0,5 s", "1 s", "2 s")),
                ("unknown", "Unbekannte Körper", "0,2 s", ("0,2 s", "0,5 s", "1 s"))):
            row = ttk.Frame(operation)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label).pack(side="left")
            variable = tk.StringVar(root, value=initial)
            selector = ttk.Combobox(row, textvariable=variable, values=values, state="disabled", width=7)
            selector.pack(side="right")
            selector.bind("<<ComboboxSelected>>", app.change_adaptive_detection)
            setattr(self, name + "_interval", variable)
            setattr(self, name + "_interval_selector", selector)
        ttk.Label(operation, text="Bei Pausen bleiben Bild und Ergebnisse gemeinsam stehen.",
                  wraplength=275).pack(anchor="w", pady=12)
        self.field_filter_enabled = tk.BooleanVar(root, value=True)
        self.thin_regions_enabled = tk.BooleanVar(root)
        self.field_filling_enabled = tk.BooleanVar(root)
        self.touching_enabled = tk.BooleanVar(root)
        self.show_contours = tk.BooleanVar(root, value=False)
        self.show_polygon_points = tk.BooleanVar(root, value=False)
        ttk.Label(filters, text="Erkennung und Darstellung", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        for label, variable, command in (
                ("Auf Arbeitsfläche begrenzen", self.field_filter_enabled, app.change_field_filter),
                ("Dünne Farbsäume ausblenden", self.thin_regions_enabled, app.change_region_filters),
                ("Feldfüllende Farbregionen ausblenden", self.field_filling_enabled, app.change_region_filters),
                ("Berührende Körper trennen", self.touching_enabled, app.change_region_filters),
                ("Konturen anzeigen", self.show_contours, self.schedule_redraw),
                ("Konturpunkte anzeigen", self.show_polygon_points, self.schedule_redraw)):
            ttk.Checkbutton(filters, text=label, variable=variable, command=command).pack(anchor="w", pady=5)
        self.detect_button = ttk.Button(sidebar, text="Erkennung starten", command=app.detect)
        self.detect_button.pack(fill="x", pady=(10, 4))
        self.preview_button = ttk.Button(sidebar, text="Nur Vorschau", command=app.show_preview)
        self.preview_button.pack(fill="x")
        self.canvas = tk.Canvas(body, background="#202a35", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.canvas.bind("<Configure>", self.schedule_redraw)
        self.canvas.bind("<Button-1>", app.click_image)
        self.status = tk.StringVar(root)
        ttk.Label(panel, textvariable=self.status, wraplength=1120).grid(row=4, column=0, sticky="ew", pady=(8, 6))
        table = ttk.Frame(panel)
        table.grid(row=5, column=0, sticky="ew")
        self.results = ttk.Treeview(table, columns=("number", "color", "shape", "x", "y", "position"),
                                   show="headings", height=5, selectmode="browse")
        for key, title, width in (("number", "Nr.", 45), ("color", "Farbe", 90), ("shape", "Form", 170),
                                   ("x", "X (0–1)", 100), ("y", "Y (0–1)", 100), ("position", "Position", 250)):
            self.results.heading(key, text=title)
            self.results.column(key, width=width, minwidth=40, stretch=True)
        self.results.pack(side="left", fill="x", expand=True)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.results.yview)
        scroll.pack(side="right", fill="y")
        self.results.configure(yscrollcommand=scroll.set)
        self.results.bind("<<TreeviewSelect>>", self.show_selection)
        self.explanation = tk.StringVar(root)
        ttk.Label(panel, textvariable=self.explanation, wraplength=1120).grid(row=6, column=0, sticky="ew", pady=(5, 0))

    def refresh(self, app):
        session = app.session
        automatic = session.auto_calibration
        has_image = session.frame is not None and (app.camera is None or app.camera_ready)
        calibrated = session.calibration is not None
        continuous = automatic.enabled and not automatic.one_shot
        self.auto_calibration_enabled.set(continuous)
        self.auto_status.set(automatic.status)
        self.auto_check_button.state(["!disabled"] if continuous and has_image else ["disabled"])
        self.auto_interval_selector.configure(state="readonly" if continuous else "disabled")
        self.detection_interval_selector.configure(state="readonly" if session.detection_pacing.enabled else "disabled")
        adaptive = session.adaptive_detection
        self.adaptive_enabled.set(adaptive.enabled)
        self.known_interval_selector.configure(state="readonly" if adaptive.enabled else "disabled")
        self.unknown_interval_selector.configure(state="readonly" if adaptive.enabled else "disabled",
            values=tuple(value for value in ("0,2 s", "0,5 s", "1 s")
                         if float(value.split()[0].replace(",", ".")) <= adaptive.known_interval_s))
        for button, enabled in ((self.calibrate_button, has_image), (self.load_button, has_image),
                (self.calibrate_once_button, has_image and not automatic.one_shot),
                (self.detect_button, calibrated and has_image and not automatic.blocked),
                (self.save_button, calibrated and not automatic.blocked),
                (self.undo_button, bool(session.corners)), (self.preview_button, session.result is not None)):
            button.state(["!disabled"] if enabled else ["disabled"])
        busy = app.camera is not None or app.retiring_camera is not None
        self.camera_button.state(["disabled"] if busy else ["!disabled"])
        self.camera_selector.state(["disabled"] if busy else ["!disabled"])
        self.stop_button.state(["!disabled"] if app.camera is not None else ["disabled"])
        if automatic.blocked:
            status = "Flächenbewegung ungeklärt. Positionsausgabe pausiert; bitte kalibrieren."
        elif session.stage == "calibrating":
            names = ("vorne links (0,0)", "vorne rechts (1,0)", "hinten rechts (1,1)", "hinten links (0,1)")
            status = f"Ecke {len(session.corners) + 1}/4: {names[len(session.corners)]} anklicken."
        elif session.stage == "detection":
            mode = "Adaptiv" if adaptive.enabled else "Mit Pause" if session.detection_pacing.enabled else "Fortlaufend"
            status = f"{mode} · Standflächenmittelpunkte geschätzt. " + app.filter_summary
        elif calibrated:
            status = "Kalibrierung bereit. Raster prüfen und Erkennung starten."
        elif has_image:
            status = "Arbeitsfläche kalibrieren: vier Bodenecken wählen oder das helle Blatt erkennen lassen."
        else:
            status = "Kamera starten oder ein eigenes Bild öffnen."
        if app.camera_note:
            status = app.camera_note + " · " + status
        if app.retiring_camera is not None:
            status += " Kamera wird freigegeben …"
        self.status.set(status)
        self.source_text.set(f"Kamera {app.camera.index}" if app.camera is not None else
                             session.image_path.name if session.image_path else "Keine Quelle ausgewählt")
        selected = self.results.selection()
        self.results.delete(*self.results.get_children())
        self.result = app.display_result
        if self.result is not None:
            for index, (candidate, shape, position) in enumerate(zip(self.result.segmentation.candidates,
                                                                    self.result.shapes, self.result.positions)):
                x, y = (f"{value:.3f}" for value in position.position_norm) if position.position_norm is not None else ("—", "—")
                label = SHAPE_NAMES[shape.label]
                if candidate.reuse_age_s is not None:
                    label += " (übernommen)"
                elif shape.temporally_confirmed:
                    label += " (bestätigt)"
                status = {"estimated": "Geschätzt", "outside": "Außerhalb", "unavailable": "Nicht bestimmbar"}[position.status]
                self.results.insert("", "end", iid=str(index), values=(index + 1, COLOR_NAMES[candidate.color], label, x, y, status))
            if self.result.shapes:
                self.results.selection_set(selected[0] if selected and self.results.exists(selected[0]) else "0")
        self.show_selection()
        self.schedule_redraw()

    def show_selection(self, event=None):
        selected = self.results.selection()
        if not selected or self.result is None:
            self.explanation.set("")
            return
        index = int(selected[0])
        if index < len(self.result.positions):
            self.explanation.set(self.result.positions[index].reason)

    def schedule_redraw(self, event=None):
        if self.closed:
            return
        if self.redraw_job is not None:
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after_idle(self.redraw)

    def redraw(self):
        self.redraw_job = None
        self.canvas.delete("all")
        self.image_mapping = None
        session = self.app.session
        if session.frame is None:
            self.canvas.create_text(max(1, self.canvas.winfo_width()) // 2,
                                    max(1, self.canvas.winfo_height()) // 2,
                                    text="Kamera starten oder Bild öffnen", fill="#dce4ec", font=("Segoe UI", 15))
            return
        frame = session.result_frame if session.result is not None and session.result_frame is not None else session.frame
        view = frame.copy()
        if session.result is None:
            if session.calibration is not None and not session.auto_calibration.blocked:
                for t in (0.25, 0.5, 0.75):
                    for a, b in (((t, 0), (t, 1)), ((0, t), (1, t))):
                        first = tuple(round(v) for v in session.calibration.normalized_to_pixel(a))
                        second = tuple(round(v) for v in session.calibration.normalized_to_pixel(b))
                        cv2.line(view, first, second, (150, 150, 150), 1, cv2.LINE_AA)
            if session.corners and not session.auto_calibration.blocked:
                polygon = np.round(session.corners).astype(np.int32)
                cv2.polylines(view, [polygon], len(polygon) == 4, (0, 210, 0), 2, cv2.LINE_AA)
                for number, (x, y) in enumerate(polygon, 1):
                    cv2.circle(view, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(view, str(number), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        elif self.app.display_result is not None:
            for number, (candidate, position) in enumerate(zip(self.app.display_result.segmentation.candidates,
                                                               self.app.display_result.positions), 1):
                x, y, width, height = candidate.bbox_xywh
                cv2.rectangle(view, (x, y), (x + width - 1, y + height - 1), DISPLAY_COLORS[candidate.color], 2)
                if self.show_contours.get():
                    cv2.drawContours(view, [candidate.contour], -1, DISPLAY_COLORS[candidate.color], 2, cv2.LINE_AA)
                if self.show_polygon_points.get():
                    points = candidate.contour.astype(np.float32)
                    # Use the classifier approximation tolerance when displaying polygon corners.
                    polygon = cv2.approxPolyDP(points, APPROX_EPSILON_RATIO * cv2.arcLength(points, True), True)
                    polygon = np.rint(polygon).astype(np.int32)
                    cv2.polylines(view, [polygon], True, (255, 255, 255), 1, cv2.LINE_AA)
                    for px, py in polygon.reshape(-1, 2):
                        cv2.circle(view, (int(px), int(py)), 6, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(view, (int(px), int(py)), 4, (255, 255, 255), -1, cv2.LINE_AA)
                for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.putText(view, str(number), (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, thickness, cv2.LINE_AA)
                if position.ground_point_px is not None:
                    ground = tuple(round(value) for value in position.ground_point_px)
                    cv2.drawMarker(view, ground, (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        height, width = view.shape[:2]
        available_width, available_height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(available_width / width, available_height / height, 1.)
        shown_width, shown_height = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(view, (shown_width, shown_height), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".png", resized)
        if not success:
            self.app.report_error(ValueError("Die Bildvorschau konnte nicht erstellt werden."))
            return
        self.photo = tk.PhotoImage(master=self.root, data=encoded.tobytes(), format="png")
        left, top = (available_width - shown_width) // 2, (available_height - shown_height) // 2
        self.canvas.create_image(left, top, anchor="nw", image=self.photo)
        self.image_mapping = (left, top, shown_width, shown_height, width, height)

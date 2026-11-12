from dataclasses import dataclass
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
CALIBRATION_DIRECTORY = PROJECT_ROOT / "runtime" / "calibrations"


@dataclass(frozen=True)
class CameraSettings:
    index: int
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class Setup:
    camera: CameraSettings
    surface_size_m: tuple[float, float]
    segmentation: dict
    geometry: dict


def load_setup(path: Path = CONFIG_PATH) -> Setup:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Die Konfiguration ist nicht lesbar: {path}") from error
    required = {"schema_version", "camera", "surface_size_m", "segmentation", "geometry"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("Die Konfiguration enthält fehlende oder unbekannte Bereiche.")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Die Konfigurationsversion wird nicht unterstützt.")
    camera = data["camera"]
    if not isinstance(camera, dict) or set(camera) != {"index", "width", "height", "fps"}:
        raise ValueError("Kameraindex, Breite, Höhe und Bildrate sind erforderlich.")
    for name, value in camera.items():
        minimum = 0 if name == "index" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"Ungültige Kameraeinstellung: {name}.")
    size = data["surface_size_m"]
    if (not isinstance(size, list) or len(size) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in size)
            or size[0] != size[1]):
        raise ValueError("Die Arbeitsfläche muss quadratisch sein; zwei gleiche positive Meterangaben verwenden.")
    if not isinstance(data["segmentation"], dict) or not isinstance(data["geometry"], dict):
        raise ValueError("Farb- und Körperparameter müssen als Objekte angegeben werden.")
    return Setup(CameraSettings(**camera), tuple(size), data["segmentation"], data["geometry"])


SETUP = load_setup()
SURFACE_SIZE_M = SETUP.surface_size_m

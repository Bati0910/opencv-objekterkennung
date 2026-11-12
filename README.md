# Farben und Körper

Lokale Python-Anwendung zur Erkennung einfarbiger geometrischer Körper mit einer fest montierten, schräg auf die Arbeitsfläche gerichteten Kamera. Die Oberfläche zeigt Farbe, Form und den geschätzten Mittelpunkt der Standfläche als normierte Koordinaten. Eigene Bilder können ebenfalls geöffnet werden. Die Verarbeitung verwendet klassische OpenCV-Verfahren, keine LLMs, Cloud-Dienste oder Netzwerkquellen.

## Voraussetzungen und Installation

Geprüft unter **Windows 11 (Build 26200), Python 3.12.10, 64 Bit**, mit Tkinter/ttk, NumPy 2.5.4 und opencv-python 5.0.0.93. Andere Plattformen wurden nicht geprüft. Benötigt werden eine lokal angeschlossene Kamera, eine ebene 18 × 18 cm-Fläche und gleichmäßige Beleuchtung. Python einschließlich Tcl/Tk installieren; PyCharm ist optional.

PowerShell im Projektordner öffnen:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Der Download findet nur bei der Installation statt. Beim Programmstart werden keine Pakete installiert. Die Anwendung benötigt anschließend keine Netzwerkverbindung. Für die Vorführung Netzwerkverbindungen am Rechner deaktivieren; auch keine Netzwerkordner als Bild- oder Kalibrierungsquelle verwenden.

## Start und Bedienung

Der einzige vorgesehene Startweg ist:

```powershell
.\.venv\Scripts\python.exe main.py
```

1. **Quelle:** Kameraindex wählen und „Kamera starten“, alternativ „Bild öffnen“. Die Kamera wird erst auf Knopfdruck geöffnet. Bei Kamerafehlern Anschluss, Index und andere Kameraanwendungen prüfen.
2. **Kalibrierung:** „Vier Ecken auswählen“ und die Bodenecken in dieser Reihenfolge anklicken: vorne links `(0,0)`, vorne rechts `(1,0)`, hinten rechts `(1,1)`, hinten links `(0,1)`. Währenddessen bleibt das Kamerabild stehen. „Letzte Ecke zurück“ korrigiert die Auswahl.
3. Alternativ ein vollständig sichtbares helles, neutrales Blatt mit **18 × 18 cm** über „Blatt einmal erkennen“ kalibrieren. „Blatt überwachen“ prüft es regelmäßig und übernimmt Änderungen nach Bestätigung. Das eingeblendete Raster immer kontrollieren; bei ungeklärter Flächenbewegung pausieren die Positionen.
4. **Erkennung starten:** Kamerabilder werden fortlaufend ausgewertet. Die Tabelle enthält Farben, Formen und Positionen. Ein Kreuz markiert die geschätzte Standflächenmitte. „Nicht bestimmbar“ bedeutet, dass die Positionsprüfung kein tragfähiges Ergebnis liefert. Die Begründung erscheint unter der ausgewählten Zeile.
5. Unter **Betrieb** wahlweise Auswertungspausen oder adaptive Aktualisierung einschalten. Adaptive Prüfung übernimmt unveränderte bekannte Körper, prüft unbekannte häufiger und führt spätestens nach zehn Sekunden zuzüglich Verarbeitungszeit eine Vollprüfung aus. Bild und Ergebnisse bleiben zwischen Auswertungen gemeinsam stehen. **Filter** enthält Flächenfilter, Plausibilitätsfilter, Berührungstrennung und Konturanzeige. „Nur Vorschau“ beendet die Erkennung; „Stoppen“ gibt die Kamera frei.

Nach Änderung von Quelle, Auflösung oder Kameraaufbau neu kalibrieren. Kalibrierungen lassen sich standardmäßig unter `runtime/calibrations/` speichern; dieser persönliche Laufzeitordner wird von Git ignoriert. Es wird keine fertige Kamerakalibrierung mitgeliefert. Kontrollmarken auf der realen Fläche helfen bei der Positionsprüfung.

## Vorgesehener Aufbau und Konfiguration

Alle Aufbauparameter stehen in **`config.json`** und werden beim Start geladen: Kamera, quadratische Arbeitsfläche, HSV-Farbgrenzen, Verarbeitungsoptionen und Körpermaße. Änderungen erfordern einen Neustart. Standardkamera: Index 0, angeforderte 1920 × 1080 Pixel bei 30 Bildern/s; das Gerät kann andere Werte liefern.

Unterstützte Farben: **Rot, Grün, Blau und Gelb**. Jedes Objekt muss einfarbig sein.

| Form | Konfiguriertes Maß |
|---|---|
| Würfel | Kantenlänge 30 mm |
| Zylinder | Grundradius 15 mm |
| Kegel | Grundradius 15 mm |
| Halbkugel | Grundradius 15 mm |

Zylinder- und Kegelhöhen sind keine Konfigurationsparameter; die Silhouettenregeln setzen den vorgesehenen aufrechten Objektsatz und den passenden Kamerablick voraus. Es gibt keine erzwungene Objektzahl. Für die Hochschulvorführung muss der physische Satz zusätzlich jede Form in mindestens zwei Farben sowie mindestens vier doppelt vorhandene Objekte umfassen.

## Grenzen und Prüfstand

Reflexionen, ähnliche Hintergrundfarben, ungünstiges Licht, starke Verdeckung, angeschnittene Körper und nicht trennbare Kontakte können zu unbekannten Formen oder fehlenden Positionen führen. Positionen sind modellbasierte Schätzungen, keine nachgewiesenen Millimetermessungen. Die Homographie gilt für die Bodenebene; Linsenverzeichnung wird nicht kalibriert. Die spätere geometrische Fortführung verdeckter Würfelkanten und daraus abgeleitete Positionen sind bewusst nicht enthalten. Ältere farbgestützte Konturkorrektur und Seitenergänzung bleiben erhalten.


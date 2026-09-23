# Brainrot Prestige Reader

A passive, read-only computer-vision tool that watches a screen recording or
live screen, OCRs the nameplate above a "Brainrot" character, and sounds an
alarm when the detected Brainrot's tier is a known **Prestige** tier.

## Scope — read-only, no gameplay automation

This tool only:
- captures/reads pixels from a video file or the screen (`mss`)
- detects the nameplate region and OCRs it (`opencv-python`, `pytesseract`)
- classifies the tier text against a configurable Prestige list
- displays what it found and plays a sound

It **never** moves the mouse, sends keystrokes, clicks, aims, or touches any
other process/window/memory. `src/capture.py` only calls `cv2.VideoCapture`
(file decode) and `mss.grab` (screen pixel read) — nothing else.

---

## A. Project structure

```
brainrots-detector/
├── .venv/                        # Python 3.14 virtual environment
├── src/
│   ├── main.py                   # pipeline orchestration + debug overlay
│   ├── capture.py                # VideoSource (file) / ScreenSource (mss, read-only)
│   ├── detector.py               # dynamic nameplate region detection
│   ├── tracker.py                # multi-target IoU tracking across frames
│   ├── ocr.py                    # tiered OCR preprocessing + Tesseract
│   ├── classifier.py             # normalization, matching, temporal confirmation
│   ├── alarm.py                  # winsound alarm with per-Brainrot cooldown
│   ├── config.py                 # settings/prestige-list loading
│   └── models.py                 # shared dataclasses
├── config/
│   ├── settings.json             # all tunables (see below)
│   └── prestige_brainrots.json   # the Prestige tier/name list
├── tools/
│   └── calibrate_roi.py          # optional manual ROI calibration utility
├── recordings/                   # video1.mov and prestige.MP4 were later removed by the
│   ├── video2.mov                #   user; their findings (Elite/Nebula tiers, the literal
│   └── new-video.MP4             #   "Prestige" tier text) still shaped the design and are
│                                  #   documented below even though the files are gone.
│                                  # video2.mov (17.4s): 2x "Six-Seven" Prestige
│                                  # new-video.MP4 (47.2s): 2x "Six-Seven" Prestige, used to
│                                  #   optimize detector.min_region_y/max_region_y (see below)
├── debug/
│   ├── frames/                   # sample frames extracted during analysis
│   └── test_report.txt           # measured accuracy report (see section E)
├── tests/
│   └── test_pipeline.py          # runs the real pipeline against both recordings
├── requirements.txt
├── run.py                        # CLI entry point
└── README.md
```

---

## B. Setup commands (Windows)

Built and tested against **Python 3.14.7** (the only version available via
the `py` launcher on this machine). `opencv-python`, `numpy`, `pytesseract`,
`mss`, and `pillow` all had prebuilt wheels for 3.14 at time of writing; if
you're on an older Python 3.9–3.13 the same `requirements.txt` versions
should still resolve fine.

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Installing Tesseract OCR (the actual OCR engine, not just the Python wrapper)

`pytesseract` is only a wrapper — it shells out to the real `tesseract.exe`.
Install it with winget:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

This installs to `C:\Program Files\Tesseract-OCR\tesseract.exe`, which is
already set in `config/settings.json` (`"tesseract_cmd"`). If you installed
it elsewhere, update that path.

Verify everything works:

```powershell
python -c "import cv2, numpy, PIL, mss, pytesseract; print('OK')"
python -c "import pytesseract; pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'; print(pytesseract.get_tesseract_version())"
```

---

## C. Files created

- `requirements.txt` — opencv-python, numpy, pytesseract, mss, pillow
- `config/settings.json` — capture/detection/OCR/classification/alarm tunables
- `config/prestige_brainrots.json` — the Prestige list (see section "Adding Prestige Brainrots")
- `src/*.py` — the 8 pipeline modules listed above
- `run.py` — CLI entry point (`video` / `live` subcommands)
- `tools/calibrate_roi.py` — manual ROI calibration utility
- `tests/test_pipeline.py` — end-to-end accuracy test against the real recordings
- `debug/test_report.txt` — the actual measured output of that test

---

## D. How the recording was analyzed

The originally supplied recordings (`video1.mov`, 13.9s, and
`video2.mov`, 17.4s — `video1.mov` has since been removed from
`recordings/`) were inspected programmatically before any detection code
was written:

- **Resolution**: 2796×1290, landscape — this is a mobile screen recording
  (PUBG Mobile, with a "Brainrot" game mode) mirrored/recorded at native
  phone resolution, not a PC monitor resolution.
- **FPS**: ~59–60 fps, H.264.
- Representative frames were extracted at evenly-spaced intervals and
  visually inspected (`debug/frames/`).
- A programmatic scan (thresholding for the health-bar's characteristic
  bright green) located every frame in which a Brainrot nameplate was
  actually visible, rather than assuming it was always on screen.

A third recording, `recordings/prestige.MP4` (15.6s), was added partway
through and analyzed the same way.

**Key findings that shaped the design:**

1. When a Brainrot is on-screen and targeted, a 4-line stacked nameplate
   appears above it: **Name** (white) → **Tier** (color varies by tier) →
   **Rate** ($/s, green) → **Health bar** (green bar).
2. The tier line is the Prestige signal. Tiers observed across the three
   recordings: `Elite` (green), `Nebula` (purple), **`Six-Seven`** (pink —
   two different Brainrots, "Ballerina Cappuccina" and "La Vacca Saturno
   Saturnita"), and the literal **`Prestige`** (red, gradient font — two
   different Brainrots, "UChik Chik Chik Chik Chun" and one named "67").
   All confirmed tier/name strings are in `config/prestige_brainrots.json`.
3. **The nameplate position is not fixed.** It's anchored to the
   Brainrot's position in the 3D world, so it moves anywhere in the
   upper-middle region of the screen as the camera pans, and the plate
   simply isn't there at all when no Brainrot is visible/targeted. A fixed
   ROI was therefore ruled out; detection is dynamic (see below).
4. **The health bar is not always visible even when the nameplate is.**
   In `prestige.MP4`, several large Prestige Brainrots (robot-type models)
   permanently hide their own health bar behind their 3D model from the
   camera angles in that clip. A detector that only anchors off the health
   bar (the initial design, validated against video1/video2) silently
   missed every sighting in that recording. `src/detector.py` now runs a
   second, independent anchor strategy — a bright white text band (the
   name line), accepted only if a colored/saturated line of text is found
   directly beneath it (the tier line) — specifically to catch this case.
   Both strategies run every detection cycle and their results are merged.

---

## E. Detection results (measured, not assumed)

Ran `python tests/test_pipeline.py`, which decodes recordings frame by
frame through the real pipeline (detection → OCR → classification → temporal
confirmation → alarm) and writes `debug/test_report.txt`.

**Historical run** (video1.mov and prestige.MP4 have since been removed
from `recordings/` by the user, but their results are kept here since they
drove real design decisions — the dynamic name-text anchor strategy and the
tier-based temporal confirmation grouping both exist because of what these
two runs found):

```
--- video1.mov (Elite / Nebula tiers only — no Prestige) ---
Total frames: 822        Frames with a candidate region: 99
OCR attempts: 53         OCR success rate: 100.0%
OCR latency: avg 873ms, min 289ms, max 2068ms
Prestige alarms triggered: 0   <-- correct: no Prestige tier present

--- video2.mov (2 separate "Six-Seven" Prestige Brainrots) ---
Total frames: 1043       Frames with a candidate region: 53
OCR attempts: 27         OCR success rate: 100.0%
OCR latency: avg 544ms, min 343ms, max 694ms
Prestige alarms triggered: 2   <-- both real Prestige Brainrots caught, one alarm each

--- prestige.MP4 (2 separate literal-"Prestige"-tier Brainrots) ---
Total frames: 932        Frames with a candidate region: 126
OCR attempts: 65         OCR success rate: 100.0%
OCR latency: avg 667ms, min 295ms, max 2101ms
Prestige alarms triggered: 2   <-- "UChik Chik Chik Chik Chun" caught (2 separate sightings,
                                    each with its own cooldown-respecting alarm); the other
                                    Prestige Brainrot ("67") was only visible for a single
                                    sampled frame and correctly did NOT alarm — temporal
                                    confirmation requires 3 matching reads by design (Phase 6)
```

All three recordings are real footage with genuinely confirmed Prestige
Brainrots in two of them, so this **is** a real validation of the Prestige
classifier end-to-end, not a synthetic stand-in. Result across all three
clips: **4/4 sufficiently-visible Prestige sightings alarmed, 0 false
positives**, plus one correctly-conservative non-alarm on a Prestige
Brainrot that only appeared for a single frame.

OCR was noisy on individual frames (e.g. `YIX-SEVEN`, `SiX-SBVEN`,
`Ballerina Gappuccina`, and a red gradient-font "Prestige" that OCR'd as
`Drectine`/`Dractine`/`Brestine` depending on the frame) but the
classifier's normalized/fuzzy matching and tier-based temporal
confirmation absorbed that noise correctly — see "Known limitations" for
the tradeoffs this surfaced.

### Region-bound optimization from new-video.MP4 (47.2s, 2823 frames)

A 47-second recording was added later specifically to check whether the
detector's search region (`detector.min_region_x/max_region_x_margin/
min_region_y/max_region_y` in `config/settings.json`) was wide enough — the
nameplate in this clip moves across much more of the screen than the
earlier, shorter clips did.

**Method:** every single frame (not a sample) was run through the real
`LabelDetector`, with its region bounds temporarily opened to the full
frame, and every candidate region was then OCR'd and required to produce
at least 2 real alphabetic words that weren't a known HUD string (blocking
out "Rank", "Shop", "Gallery", etc.) before counting as a genuine
nameplate sighting. This mattered: pure shape/color heuristics alone
(no OCR check) kept matching false positives — the top-left menu icon row,
the minimap, the player's own character model, and background scenery all
briefly produce green/bright-white blobs that pass the raw geometric
filters. Requiring an actual multi-word OCR read eliminated those.

**Result:** 1290 genuine sightings out of 2823 frames. Their 1st/99th
percentile pixel extent, converted to the 2796×1290 reference coordinate
system `settings.json` uses:

```
min_region_y: 150 -> 84    (nameplates seen as high as ~6.5% down the frame)
max_region_y: 800 -> 1183  (nameplates seen as low as ~88.6% down the frame —
                             this was the real bottleneck: at 800 (62%),
                             anything lower was silently invisible to the
                             detector regardless of how clean the OCR would
                             have been)
min_region_x / max_region_x_margin: left at 150 each — this video's
                             horizontal extent (26.7%-89.5% at p1/p99) fit
                             comfortably inside the existing bound, so it
                             was left untouched rather than narrowed, since
                             one video isn't enough evidence to shrink a
                             bound that worked on three others.
```

Re-ran `tests/test_pipeline.py` against video2.mov and new-video.MP4 after
the change: video2.mov's results were byte-identical to before (its
nameplates never approached the old y-bound, so widening it changed
nothing there — good, no regression), and new-video.MP4 now correctly
catches both of its real Prestige Brainrots:

```
--- video2.mov ---
Prestige alarms triggered: 2   <-- unchanged from before

--- new-video.MP4 ---
Total frames: 2823       Frames with a candidate region: 103
OCR attempts: 62         OCR success rate: 100.0%
Prestige alarms triggered: 2   <-- "Goosini Planini" and "Whalino Helmetino",
                                    both "Six-Seven", 0 false positives
```

---

## F. How to run video mode

```powershell
python run.py video --path recordings/video2.mov
python run.py video --path recordings/new-video.MP4 --no-debug
```

With the debug overlay on (default), a window shows the detected region
(green box), the anchor box (blue), OCR'd text, match, Prestige status,
confidence, FPS, and OCR latency. Press `q` to quit.

## G. How to run live-screen mode

```powershell
python run.py live --monitor 1
python run.py live --monitor 1 --no-debug
```

This only **reads** pixels via `mss` — see the Scope section above. `--monitor`
selects which monitor to read (`mss`'s monitor index; `1` is the primary
monitor for the default multi-monitor layout). If you're playing through an
emulator or a mirrored phone screen on your PC, point it at whichever
monitor/window shows that.

### Single-monitor setups: use `snapshot`, not `live` with the debug window

If you only have one physical display (e.g. a phone screen-mirroring setup
with no second monitor), running `live` **with** its debug window causes a
recursive "hall of mirrors" effect: `mss` captures the whole screen, the
debug window is drawn on that same screen, and the next capture includes
last frame's debug window nested inside it, growing every frame. It's not a
bug in detection — there's just nowhere on-screen for the debug view to sit
that isn't also being captured.

Two ways around it:

```powershell
# 1. Run live mode with no visual output at all (just stats + the alarm sound)
python run.py live --monitor 1 --no-debug

# 2. Grab ONE frame, run detection once, save the result to a file — no loop, no recursion
python run.py snapshot --monitor 1
python run.py snapshot --monitor 1 --out debug/live_snapshot.png
```

Use `snapshot` first to sanity-check detection against your actual mirrored
resolution/scale before trusting `--no-debug` blind — it prints the same
info the debug overlay would show (regions found, OCR text/confidence,
matched name/tier, Prestige status) and saves an annotated image you can
open and inspect.

---

## Configuration (`config/settings.json`)

| Key | Meaning |
|---|---|
| `tesseract_cmd` | Path to `tesseract.exe` |
| `capture.monitor_index` | Default monitor for live mode |
| `capture.target_capture_fps` | Screen-grab rate in live mode |
| `sampling.detection_fps` | How often the nameplate detector runs |
| `sampling.ocr_fps` | How often OCR actually runs (≤ detection_fps) |
| `detector.*` | Nameplate-anchor search geometry (see `src/detector.py`) |
| `ocr.fixed_brightness_threshold` | Primary text-isolation threshold |
| `ocr.dim_brightness_threshold` | Fallback threshold for dimmer tier colors |
| `classification.fuzzy_match_threshold` | Minimum fuzzy-match ratio (0–1) to count as Prestige |
| `temporal_confirmation.required_matches` / `window_seconds` | How many matching tier reads, in how many seconds, before alarming |
| `alarm.cooldown_seconds` | Per-Brainrot-tier re-alarm cooldown |

## Target Categories (`config/prestige_brainrots.json`)

The nameplate consists of:
1. **Brainrot Name** (e.g. `Ballerina Cappuccina`, `67`, `La Vacca Saturno Saturnita`)
2. **Category / Tier** (e.g. `Prestige`, `Shining`, `Crystal`, `Six-Seven`, `Elite`, `Nebula`)
3. **Rate / Health Bar**

The detector **only targets the Category line**. A Brainrot's name (such as `67`) does not trigger an alarm by itself; alarms only trigger when the category line matches one of the target categories:
- `Prestige`
- `Shining` (and `Shinning`)
- `Crystal`

Edit `config/prestige_brainrots.json`:

```json
{
  "prestige": [
    "Prestige",
    "Shining",
    "Shinning",
    "Crystal"
  ],
  "known_non_prestige_tiers": [
    "Six-Seven",
    "Elite",
    "Nebula"
  ]
}
```

Categories like `Six-Seven`, `Elite`, and `Nebula` are recognized as non-prestige and will not sound an alarm.

## ROI calibration (optional)

Detection is dynamic by default (see section D). If you want to lock
detection to one fixed screen region instead (e.g. a static camera angle):

```powershell
python tools/calibrate_roi.py --video recordings/video2.mov --frame 100
```

Drag a box around a nameplate, press Enter to save (writes
`detector.manual_roi` into `settings.json`), Esc to cancel. To go back to
dynamic detection:

```powershell
python tools/calibrate_roi.py --clear
```

## Debug mode

On by default (`debug.enabled` / omit `--no-debug`). Shows a live OpenCV
window with:
- the detected anchor (blue box) and nameplate ROI (green box) drawn on the frame
- OCR'd text and confidence
- matched name/tier and Prestige status
- FPS and OCR latency
- a red "PRESTIGE DETECTED" banner on confirmation

## OCR troubleshooting

- **Whole tier line missing / only the name reads**: some tier colors
  (e.g. a dim purple like "Nebula") sit below the fast brightness
  threshold. The engine already retries at a lower threshold
  (`ocr.dim_brightness_threshold`) when fewer than 2 lines come back —
  lower it further if a specific color still doesn't show up.
- **Garbled text mixed with real text** (e.g. `SQ FF o- Ballerina...`):
  happens when the ROI overlaps busy background geometry or an adjacent
  player nametag. The classifier's fuzzy/normalized matching tolerates a
  fair amount of this for the tier field; if it's happening constantly,
  narrow `detector.label_band_side_margin_max` / `_ratio` in
  `settings.json`.
- **Nothing detected at all**: check `debug.show_window` output — if no
  green/blue boxes ever appear, the health-bar color mask
  (`detector.py`'s HSV green range) may not match your game's actual
  green; sample a pixel from your own health bar and adjust.

## Performance tuning

Measured on this machine (see `debug/test_report.txt`): sustained decode+
detect+OCR throughput was 13–29 fps depending on how often the fallback OCR
passes were needed. Levers, in order of impact:
1. `sampling.ocr_fps` — this is the biggest cost; lower it if CPU-bound.
2. `sampling.detection_fps` — cheap (just a color mask + contour scan), but
   still gates how often OCR can run.
3. `ocr.upscale_factor` — larger ROIs cost more per Tesseract call.
4. `capture.target_capture_fps` — only affects how smooth the debug
   preview is; detection/OCR are independently throttled.

## Limitations

- **Position, not identity**: detection finds "a nameplate-shaped region,"
  not "the specific Brainrot you're farming." If two Brainrots are on
  screen, the largest/clearest anchor wins each cycle.
- **Adjacent UI text can bleed into the OCR crop** (e.g. a nearby player's
  name tag), which is why classification checks every OCR'd line rather
  than assuming a fixed line index — but very close overlaps can still
  merge onto the same line as the tier text.
- **Temporal confirmation and the alarm cooldown key are based on whichever
  OCR line actually matched the Prestige list**, not a fixed "tier line"
  position — see `src/classifier.py` docstring. This was a deliberate fix:
  grouping strictly by name+tier together caused a single misread letter in
  a long name to silently break confirmation even when the tier read
  correctly three times in a row, and in `prestige.MP4` the *name*
  ("UChik Chik Chik Chik Chun") was actually the reliable, consistently-OCR'd
  field while the tier ("Prestige" in a red gradient font) read as a
  different garbled string almost every frame. Tradeoffs: (1) two
  *different* Prestige Brainrots that match on the *same* string within
  `alarm.cooldown_seconds` of each other will only alarm once; (2) when a
  specific Brainrot *name* (not a tier label) is what matched the Prestige
  list, the debug overlay's "Match" / "Tier" fields can appear swapped —
  Prestige detection itself is unaffected either way.
- **Detection needs either a visible health bar or a visible name+tier
  text pair.** Two independent anchor strategies run (health-bar-anchored
  and name-text-anchored, see `src/detector.py`), which recovered every
  sighting in testing, but a Brainrot whose *entire* nameplate — both the
  health bar and the name/tier text — is off-screen or fully occluded
  won't be detected at all.
- **OCR needs a real Tesseract install**; there's no bundled fallback OCR.
- Only tested against the two supplied recordings (2796×1290 mobile
  capture). A PC-native resolution or a very different HUD theme may need
  retuning of `detector.*` geometry and OCR thresholds.

## Implemented Improvements

- **Multi-target tracking across frames**: Replaced single "largest anchor wins" with `TargetTracker` (`src/tracker.py`), an IoU/distance-based spatial tracker. Each visible Brainrot maintains its own independent track, `TemporalConfirmer` history queue, and composite cooldown key (`f"{track_id}:{normalized_tier}"`), preventing multiple targets from starving or silencing each other.
- **50% OCR latency reduction**: Eliminated redundant `pytesseract.image_to_string` calls in `src/ocr.py` by parsing layout lines directly from `pytesseract.image_to_data` output dict.
- **Asynchronous non-blocking OCR in live mode**: Decoupled live screen capture (30-60 FPS) from OCR via background worker thread queue in `src/main.py`.
- **Search region slicing & cached morphology kernels**: Optimized `src/detector.py` to crop search bands before running HSV/grayscale conversions and cached structuring elements.
- **Unicode dash normalization & phrase containment matching**: Improved `normalize_text` and `_match_against_list` in `src/classifier.py` to handle em-dashes and merged lines.
- **Thread-safe audio alert playback**: Added thread serialization locks and optional WAV sound file playback in `src/alarm.py`.
- **Monitor enumeration CLI**: Added `python run.py list-monitors` to query available screen displays and bounding rectangles.
- **Added "67" to Prestige config**: Included numeric `"67"` alongside `"Six-Seven"` in `config/prestige_brainrots.json`.

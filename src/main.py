"""Pipeline orchestration: capture -> detect -> OCR -> classify -> temporal
confirmation -> alarm, with an optional OpenCV debug overlay window.

This module never sends input to any window or process. It only reads
frames (from a video file or from the screen via mss) and writes to its own
debug window / stdout / the system speaker.
"""
import os
import queue
import threading
import time

import cv2

from . import config as cfg
from .alarm import AlarmManager
from .capture import ScreenSource, VideoSource
from .classifier import PrestigeClassifier, TemporalConfirmer
from .detector import LabelDetector
from .ocr import OCREngine
from .tracker import TargetTracker


class Pipeline:
    def __init__(self, settings, prestige_data, async_ocr=False):
        self.settings = settings
        self.detector = LabelDetector(settings)
        self.ocr = OCREngine(settings)
        self.classifier = PrestigeClassifier(settings, prestige_data)
        self.confirmer = TemporalConfirmer(settings)
        self.tracker = TargetTracker(settings)
        self.alarm = AlarmManager(settings)

        self.detection_interval = 1.0 / settings["sampling"]["detection_fps"]
        self.ocr_interval = 1.0 / settings["sampling"]["ocr_fps"]
        self._last_detection_time = -1e9
        self._last_ocr_time = -1e9

        self.async_ocr = async_ocr
        self._ocr_queue = None
        self._ocr_thread = None
        self._ocr_result_lock = threading.Lock()
        self._latest_async_results = []
        if self.async_ocr:
            self._ocr_queue = queue.Queue(maxsize=2)
            self._ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
            self._ocr_thread.start()

        self.stats = {
            "frames_seen": 0,
            "frames_detected_region": 0,
            "ocr_attempts": 0,
            "ocr_successes": 0,
            "prestige_confirmed": 0,
            "last_ocr_ms": 0.0,
        }

    def _ocr_worker(self):
        while True:
            item = self._ocr_queue.get()
            if item is None:
                break
            roi, target_id, timestamp = item
            ocr_result = self.ocr.read(roi)
            classification = self.classifier.classify(ocr_result.lines) if ocr_result.lines else None
            with self._ocr_result_lock:
                self._latest_async_results.append((target_id, timestamp, ocr_result, classification))
            self._ocr_queue.task_done()

    def stop(self):
        if self.async_ocr and self._ocr_queue:
            self._ocr_queue.put(None)

    def process_frame(self, frame_index, timestamp, frame):
        self.stats["frames_seen"] += 1
        result = {"regions": [], "targets": [], "ocr": None, "classification": None, "confirmed": None}

        # 1. Ingest any completed background OCR results
        if self.async_ocr:
            with self._ocr_result_lock:
                completed = list(self._latest_async_results)
                self._latest_async_results.clear()
            for tid, t_stamp, ocr_res, classif in completed:
                self.stats["last_ocr_ms"] = ocr_res.processing_ms
                if ocr_res.lines:
                    self.stats["ocr_successes"] += 1
                target = self.tracker.tracks.get(tid)
                if target:
                    target.latest_ocr = ocr_res
                    target.latest_classification = classif
                    result["ocr"] = ocr_res
                    result["classification"] = classif
                    if classif:
                        confirmed = target.confirmer.observe(t_stamp, classif)
                        if confirmed is not None:
                            result["confirmed"] = confirmed
                            if confirmed.is_prestige:
                                target.is_confirmed_prestige = True
                                key = f"{target.track_id}:{confirmed.normalized_tier}"
                                if self.alarm.maybe_trigger(key, t_stamp):
                                    self.stats["prestige_confirmed"] += 1

        if timestamp - self._last_detection_time < self.detection_interval:
            return result
        self._last_detection_time = timestamp

        regions = self.detector.find_regions(frame)
        result["regions"] = regions
        active_targets = self.tracker.update(regions, timestamp)
        result["targets"] = active_targets
        if not regions:
            return result
        self.stats["frames_detected_region"] += 1

        if timestamp - self._last_ocr_time < self.ocr_interval:
            return result
        self._last_ocr_time = timestamp

        target = self.tracker.select_target_for_ocr(active_targets)
        if target is None:
            return result

        x, y, w, h = target.region.label_box
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return result

        if self.async_ocr:
            if not self._ocr_queue.full():
                self.stats["ocr_attempts"] += 1
                target.last_ocr_time = timestamp
                self._ocr_queue.put_nowait((roi.copy(), target.track_id, timestamp))
            return result

        # Synchronous OCR execution (video evaluation and tests)
        self.stats["ocr_attempts"] += 1
        target.last_ocr_time = timestamp
        ocr_result = self.ocr.read(roi)
        self.stats["last_ocr_ms"] = ocr_result.processing_ms
        target.latest_ocr = ocr_result
        result["ocr"] = ocr_result
        result["region"] = target.region
        result["target"] = target

        if not ocr_result.lines:
            return result
        self.stats["ocr_successes"] += 1

        classification = self.classifier.classify(ocr_result.lines)
        target.latest_classification = classification
        result["classification"] = classification

        confirmed = target.confirmer.observe(timestamp, classification)
        if confirmed is not None:
            result["confirmed"] = confirmed
            if confirmed.is_prestige:
                target.is_confirmed_prestige = True
                key = f"{target.track_id}:{confirmed.normalized_tier}"
                triggered = self.alarm.maybe_trigger(key, timestamp)
                if triggered:
                    self.stats["prestige_confirmed"] += 1

        return result


def draw_debug_overlay(frame, result, stats, fps):
    overlay = frame
    y0 = 40
    cv2.putText(overlay, "Brainrot Reader", (30, y0), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2)

    targets = result.get("targets", [])
    if targets:
        for t in targets:
            x, y, w, h = t.region.label_box
            color = (0, 0, 255) if t.is_confirmed_prestige else (0, 255, 0)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            ax, ay, aw, ah = t.region.anchor_box
            cv2.rectangle(overlay, (ax, ay), (ax + aw, ay + ah), (255, 0, 0), 2)
            label = f"#{t.track_id}"
            if t.latest_classification and t.latest_classification.brainrot_name:
                label += f": {t.latest_classification.brainrot_name[:12]}"
            cv2.putText(overlay, label, (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        for region in result.get("regions", []):
            x, y, w, h = region.label_box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ax, ay, aw, ah = region.anchor_box
            cv2.rectangle(overlay, (ax, ay), (ax + aw, ay + ah), (255, 0, 0), 2)

    lines = [f"FPS: {fps:.1f}", f"OCR latency: {stats['last_ocr_ms']:.0f}ms"]
    ocr = result.get("ocr")
    if ocr is not None:
        lines.append(f"OCR: {' | '.join(ocr.lines[:2]) if ocr.lines else '(empty)'}")
        lines.append(f"OCR confidence: {ocr.confidence:.0f}%")
    classification = result.get("classification")
    if classification is not None:
        lines.append(f"Match: {classification.brainrot_name}")
        lines.append(f"Tier: {classification.tier_text}")
        prestige_str = "YES" if classification.is_prestige else "no"
        lines.append(f"Prestige (this frame): {prestige_str}")
    confirmed = result.get("confirmed")
    if confirmed is not None and confirmed.is_prestige:
        cv2.putText(overlay, "PRESTIGE DETECTED", (30, y0 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    for i, line in enumerate(lines):
        cv2.putText(overlay, line, (30, y0 + 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    return overlay


def run_video(video_path, debug=True, settings=None, prestige_data=None):
    settings = settings or cfg.load_settings()
    prestige_data = prestige_data or cfg.load_prestige_list()
    pipeline = Pipeline(settings, prestige_data)
    source = VideoSource(video_path)

    fps_timer = time.time()
    frame_counter = 0
    display_fps = 0.0

    for frame_index, timestamp, frame in source.frames():
        result = pipeline.process_frame(frame_index, timestamp, frame)

        frame_counter += 1
        now = time.time()
        if now - fps_timer >= 0.5:
            display_fps = frame_counter / (now - fps_timer)
            frame_counter = 0
            fps_timer = now

        if debug:
            overlay = draw_debug_overlay(frame.copy(), result, pipeline.stats, display_fps)
            cv2.imshow("Brainrot Prestige Reader", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    source.release()
    if debug:
        cv2.destroyAllWindows()
    return pipeline.stats


def run_live(monitor_index=1, debug=True, settings=None, prestige_data=None):
    settings = settings or cfg.load_settings()
    prestige_data = prestige_data or cfg.load_prestige_list()
    pipeline = Pipeline(settings, prestige_data, async_ocr=True)
    target_fps = settings["capture"]["target_capture_fps"]
    source = ScreenSource(monitor_index=monitor_index, target_fps=target_fps)

    fps_timer = time.time()
    frame_counter = 0
    display_fps = 0.0

    try:
        for frame_index, timestamp, frame in source.frames():
            result = pipeline.process_frame(frame_index, timestamp, frame)

            frame_counter += 1
            now = time.time()
            if now - fps_timer >= 0.5:
                display_fps = frame_counter / (now - fps_timer)
                frame_counter = 0
                fps_timer = now

            if debug:
                overlay = draw_debug_overlay(frame.copy(), result, pipeline.stats, display_fps)
                cv2.imshow("Brainrot Prestige Reader (LIVE - read only)", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        source.release()
        if debug:
            cv2.destroyAllWindows()
    return pipeline.stats


def run_snapshot(monitor_index=1, out_path="debug/live_snapshot.png", settings=None, prestige_data=None):
    """Grabs a single live screen frame, runs detection/OCR/classification on
    it once, draws the debug overlay, and saves it to a file instead of
    opening a live preview window. This is the safe way to check detection
    on a single-monitor setup: a continuous debug window there ends up
    capturing itself frame over frame (mss reads the whole screen, and the
    debug window is drawn on that same screen), producing a recursive
    "hall of mirrors" effect. A one-shot grab-and-save has nothing left on
    screen to re-capture.
    """
    settings = settings or cfg.load_settings()
    prestige_data = prestige_data or cfg.load_prestige_list()
    pipeline = Pipeline(settings, prestige_data)
    source = ScreenSource(monitor_index=monitor_index, target_fps=1)

    frame_index, timestamp, frame = next(source.frames())
    result = pipeline.process_frame(frame_index, timestamp, frame)
    source.release()

    overlay = draw_debug_overlay(frame.copy(), result, pipeline.stats, 0.0)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, overlay)

    return result, out_path

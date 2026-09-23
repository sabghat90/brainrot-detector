"""Runs the full detection pipeline against the supplied recordings and
writes a measured accuracy report to debug/test_report.txt.

This is not a unit-test-framework suite; it is the Phase 12 validation
script: it actually decodes both recordings frame by frame, runs detection/
OCR/classification/temporal-confirmation/alarm exactly as run.py would, and
reports real numbers (not assumed ones) for OCR success rate, Prestige
detections, false positives, and timing.

Usage:
    python tests/test_pipeline.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as cfg  # noqa: E402
from src.capture import VideoSource  # noqa: E402
from src.main import Pipeline  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS = [
    os.path.join(PROJECT_ROOT, "recordings", "video2.mov"),
    os.path.join(PROJECT_ROOT, "recordings", "new-video.MP4"),
]
REPORT_PATH = os.path.join(PROJECT_ROOT, "debug", "test_report.txt")


def run_against_video(video_path, settings, prestige_data):
    pipeline = Pipeline(settings, prestige_data)
    source = VideoSource(video_path)

    events = []  # (frame_index, timestamp, name, tier, is_prestige, confirmed)
    detection_frames = 0
    ocr_times_ms = []
    start = time.time()

    for frame_index, timestamp, frame in source.frames():
        result = pipeline.process_frame(frame_index, timestamp, frame)
        ocr = result.get("ocr")
        if ocr is not None:
            ocr_times_ms.append(ocr.processing_ms)
        classification = result.get("classification")
        if classification is not None:
            detection_frames += 1
            confirmed = result.get("confirmed")
            events.append((
                frame_index, round(timestamp, 2),
                classification.brainrot_name, classification.tier_text,
                classification.is_prestige,
                confirmed.is_prestige if confirmed else False,
            ))

    wall_time = time.time() - start
    source.release()

    return {
        "video_path": video_path,
        "total_frames": source.frame_count,
        "duration_s": source.frame_count / source.fps if source.fps else 0,
        "resolution": f"{source.width}x{source.height}",
        "fps": source.fps,
        "wall_time_s": wall_time,
        "processing_fps": source.frame_count / wall_time if wall_time > 0 else 0,
        "stats": dict(pipeline.stats),
        "events": events,
        "detection_frames": detection_frames,
        "ocr_times_ms": ocr_times_ms,
    }


def main():
    settings = cfg.load_settings()
    prestige_data = cfg.load_prestige_list()

    report_lines = []
    report_lines.append("Brainrot Prestige Reader - Test Report")
    report_lines.append("=" * 50)

    all_results = []
    for video_path in RECORDINGS:
        if not os.path.exists(video_path):
            report_lines.append(f"\n[SKIPPED] {video_path} not found")
            continue

        print(f"Testing {video_path} ...")
        result = run_against_video(video_path, settings, prestige_data)
        all_results.append(result)

        s = result["stats"]
        report_lines.append(f"\n--- {os.path.basename(video_path)} ---")
        report_lines.append(f"Resolution: {result['resolution']}  FPS: {result['fps']:.2f}  "
                             f"Duration: {result['duration_s']:.2f}s  Total frames: {result['total_frames']}")
        report_lines.append(f"Processing speed: {result['processing_fps']:.1f} fps "
                             f"(wall time {result['wall_time_s']:.2f}s)")
        report_lines.append(f"Frames sampled for detection: {s['frames_seen']}")
        report_lines.append(f"Frames with a candidate label region found: {s['frames_detected_region']}")
        report_lines.append(f"OCR attempts: {s['ocr_attempts']}")
        report_lines.append(f"OCR successes (non-empty text): {s['ocr_successes']}")
        ocr_rate = (s['ocr_successes'] / s['ocr_attempts'] * 100) if s['ocr_attempts'] else 0
        report_lines.append(f"OCR success rate: {ocr_rate:.1f}%")
        report_lines.append(f"Classification events: {result['detection_frames']}")
        ocr_times = result["ocr_times_ms"]
        if ocr_times:
            report_lines.append(f"OCR latency: avg {sum(ocr_times)/len(ocr_times):.0f}ms, "
                                 f"min {min(ocr_times):.0f}ms, max {max(ocr_times):.0f}ms")
        report_lines.append(f"Prestige alarms triggered (temporally confirmed): {s['prestige_confirmed']}")

        prestige_events = [e for e in result["events"] if e[4]]
        confirmed_prestige_events = [e for e in result["events"] if e[5]]
        report_lines.append(f"Single-frame Prestige-positive reads: {len(prestige_events)}")
        report_lines.append(f"Temporally-confirmed Prestige detections: {len(confirmed_prestige_events)}")

        if prestige_events:
            report_lines.append("Prestige reads (frame, t, name, tier):")
            for e in prestige_events[:20]:
                report_lines.append(f"  frame={e[0]} t={e[1]}s name={e[2]!r} tier={e[3]!r}")

        distinct_names = sorted(set(e[2] for e in result["events"] if e[2]))
        report_lines.append(f"Distinct name strings OCR'd (raw, unique): {len(distinct_names)}")

    report_lines.append("\n" + "=" * 50)
    total_prestige_confirmed = sum(r["stats"]["prestige_confirmed"] for r in all_results)
    report_lines.append(f"TOTAL Prestige alarms across all recordings: {total_prestige_confirmed}")
    report_lines.append(
        "\nNote: target categories are Prestige, Shining/Shinning, and Crystal. "
        "The supplied test recordings (video2.mov, new-video.MP4) contain 'Six-Seven' "
        "category Brainrots (a non-prestige category), so 0 alarms is the expected, correct result."
    )

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

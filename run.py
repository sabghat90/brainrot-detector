"""Entry point.

    python run.py video    --path recordings/video1.mov [--no-debug]
    python run.py live     --monitor 1 [--no-debug]
    python run.py snapshot --monitor 1 [--out debug/live_snapshot.png]

Read-only: video mode decodes a file, live/snapshot modes read screen pixels
via mss. None of these send any input to any window, process, or game.

`snapshot` grabs a single frame, runs detection once, and saves the debug
overlay to a file — use this instead of `live` (without --no-debug) on a
single-monitor setup, where a continuously-updating debug window ends up
capturing itself frame over frame.
"""
import argparse
import sys

from src import config as cfg
from src.main import run_live, run_snapshot, run_video


def main():
    parser = argparse.ArgumentParser(description="Brainrot Prestige Reader")
    sub = parser.add_subparsers(dest="mode", required=True)

    video_p = sub.add_parser("video", help="Run against a recorded video file")
    video_p.add_argument("--path", required=True, help="Path to the video file")
    video_p.add_argument("--no-debug", action="store_true", help="Disable the debug overlay window")

    live_p = sub.add_parser("live", help="Run against the live screen (read-only capture)")
    live_p.add_argument("--monitor", type=int, default=None, help="Monitor index (default from settings.json)")
    live_p.add_argument("--no-debug", action="store_true", help="Disable the debug overlay window")

    snapshot_p = sub.add_parser(
        "snapshot",
        help="Grab a single live screen frame, run detection once, and save the debug overlay to a file",
    )
    snapshot_p.add_argument("--monitor", type=int, default=None, help="Monitor index (default from settings.json)")
    snapshot_p.add_argument("--out", default="debug/live_snapshot.png", help="Where to save the overlay image")

    sub.add_parser("list-monitors", help="List all available monitors and their display bounds")

    args = parser.parse_args()

    if args.mode == "list-monitors":
        try:
            import mss
            mss_cls = getattr(mss, "MSS", mss.mss)
            with mss_cls() as sct:
                print("Available monitors:")
                for idx, m in enumerate(sct.monitors):
                    label = "Virtual / all displays combined" if idx == 0 else f"Monitor {idx}"
                    print(f"  [{idx}] {label}: {m['width']}x{m['height']} at ({m['left']}, {m['top']})")
        except ImportError:
            print("mss is not installed. Run: pip install mss")
        return

    settings = cfg.load_settings()
    prestige_data = cfg.load_prestige_list()

    if args.mode == "video":
        stats = run_video(args.path, debug=not args.no_debug, settings=settings, prestige_data=prestige_data)
    elif args.mode == "live":
        monitor = args.monitor if args.monitor is not None else settings["capture"]["monitor_index"]
        stats = run_live(monitor, debug=not args.no_debug, settings=settings, prestige_data=prestige_data)
    else:
        monitor = args.monitor if args.monitor is not None else settings["capture"]["monitor_index"]
        result, out_path = run_snapshot(monitor, args.out, settings=settings, prestige_data=prestige_data)
        print(f"Saved snapshot to {out_path}")
        print(f"Regions found: {len(result.get('regions', []))}")
        ocr = result.get("ocr")
        if ocr is not None:
            print(f"OCR lines: {ocr.lines}")
            print(f"OCR confidence: {ocr.confidence:.0f}%")
        else:
            print("OCR: not attempted (no candidate region found)")
        classification = result.get("classification")
        if classification is not None:
            print(f"Name: {classification.brainrot_name!r}  Tier: {classification.tier_text!r}")
            print(f"Prestige: {classification.is_prestige} ({classification.match_type})")
        return

    print("\n--- Session stats ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    sys.exit(main())

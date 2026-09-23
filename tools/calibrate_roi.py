"""Manual ROI calibration utility.

Loads one frame from a video, lets you drag a rectangle around a Brainrot
nameplate with the mouse, and saves the box as resolution-independent
normalized coordinates (0-1 fractions of frame width/height) into
config/settings.json under detector.manual_roi.

This is an OPTIONAL override for a fixed camera angle. The default pipeline
detects the nameplate dynamically every frame (see src/detector.py) because
analysis of the supplied recordings showed the label moves with the
Brainrot's position rather than staying in one place. Only use this if you
specifically want to lock detection to one fixed screen region.

Usage:
    python tools/calibrate_roi.py --video recordings/video1.mov --frame 100
    python tools/calibrate_roi.py --video recordings/video1.mov --frame 100 --clear
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as cfg  # noqa: E402

WINDOW_NAME = "Calibrate ROI - drag a box, ENTER to save, ESC to cancel"

_drag_state = {"start": None, "end": None, "dragging": False}


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _drag_state["start"] = (x, y)
        _drag_state["end"] = (x, y)
        _drag_state["dragging"] = True
    elif event == cv2.EVENT_MOUSEMOVE and _drag_state["dragging"]:
        _drag_state["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drag_state["end"] = (x, y)
        _drag_state["dragging"] = False


def load_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def run(video_path, frame_index):
    frame = load_frame(video_path, frame_index)
    h, w = frame.shape[:2]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

    print("Drag a rectangle around the Brainrot nameplate (name+tier).")
    print("Press ENTER to save, ESC to cancel, 'r' to reset the box.")

    while True:
        display = frame.copy()
        if _drag_state["start"] and _drag_state["end"]:
            cv2.rectangle(display, _drag_state["start"], _drag_state["end"], (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Cancelled.")
            cv2.destroyAllWindows()
            return
        if key == ord("r"):
            _drag_state["start"] = None
            _drag_state["end"] = None
        if key == 13:  # ENTER
            if not (_drag_state["start"] and _drag_state["end"]):
                print("No box drawn yet.")
                continue
            break

    cv2.destroyAllWindows()

    x1, y1 = _drag_state["start"]
    x2, y2 = _drag_state["end"]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    roi = {
        "left": round(left / w, 4),
        "top": round(top / h, 4),
        "right": round(right / w, 4),
        "bottom": round(bottom / h, 4),
    }

    settings = cfg.load_settings()
    settings["detector"]["manual_roi"] = roi
    cfg.save_settings(settings)
    print(f"Saved manual_roi to config/settings.json: {roi}")


def clear(video_path=None, frame_index=None):
    settings = cfg.load_settings()
    settings["detector"]["manual_roi"] = None
    cfg.save_settings(settings)
    print("Cleared manual_roi (dynamic detection is now active again).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate a manual ROI for the Brainrot nameplate")
    parser.add_argument("--video", required=True, help="Path to a video file to pick a frame from")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to load")
    parser.add_argument("--clear", action="store_true", help="Clear the manual ROI override instead")
    args = parser.parse_args()

    if args.clear:
        clear()
    else:
        run(args.video, args.frame)

"""Finds the Brainrot nameplate region in a frame.

The nameplate (name / tier / rate / health-bar) is anchored to the Brainrot's
in-world position, so it can appear anywhere in the upper-middle portion of
the screen and is not always present. Fixed-ROI detection was ruled out by
inspecting the supplied recordings (see README "How the recording was
analyzed").

Two independent anchor strategies run and their results are merged:

1. Health-bar anchor: the bright-green health-bar / rate-text blob, then the
   band of text directly above it. This is the primary strategy and works
   even when the name text itself is dim or partly obscured.
2. Name-text anchor: a bright white text band (the name line), validated by
   requiring a saturated/colored line of text directly below it (the tier
   line). This exists because the `prestige.MP4` recording showed several
   Prestige Brainrots (large robot-type models) whose health bar was
   permanently hidden behind their own 3D model, so strategy 1 alone missed
   them even though the name+tier text was in plain view.

All the size/geometry values in settings.json (widths, heights, margins,
band sizes) were tuned against the supplied recordings, captured at the
iPhone's native 2796x1290. A live capture at a different resolution has a
proportionally different nameplate in pixel terms, so every threshold is
rescaled per-frame: horizontal values by frame_width/REFERENCE_WIDTH,
vertical values by frame_height/REFERENCE_HEIGHT, independently. Two axes
on purpose: a screen-mirrored capture window is not guaranteed to preserve
the source aspect ratio (confirmed with a real screenshot from an AirServer
mirror window at 1366x768 — aspect ~1.78:1 versus the recordings' ~2.17:1 —
where a single width-only scale factor mis-scaled every y-bound and
rejected a real nameplate that was otherwise a clean match).

A manual fixed ROI (from tools/calibrate_roi.py) can still be supplied via
settings["detector"]["manual_roi"] for a fixed camera angle, but it is not
the default.
"""
import cv2
import numpy as np

from .models import LabelRegion


class LabelDetector:
    REFERENCE_WIDTH = 2796   # frame size settings.json's detector.* values were tuned against
    REFERENCE_HEIGHT = 1290

    def __init__(self, settings):
        d = settings["detector"]
        # spatial (pixel) values -- rescaled per-frame against REFERENCE_WIDTH/HEIGHT
        self._min_region_y = d["min_region_y"]
        self._max_region_y = d["max_region_y"]
        self._min_region_x = d["min_region_x"]
        self._max_region_x_margin = d["max_region_x_margin"]
        self._anchor_min_width = d["anchor_min_width"]
        self._anchor_min_height = d["anchor_min_height"]
        self._anchor_max_height = d["anchor_max_height"]
        self._band_height = d["label_band_height_above_anchor"]
        self._side_margin_min = d["label_band_side_margin_min"]
        self._side_margin_max = d["label_band_side_margin_max"]
        self._name_anchor_min_width = d["name_anchor_min_width"]
        self._name_anchor_min_height = d["name_anchor_min_height"]
        self._name_anchor_max_height = d["name_anchor_max_height"]
        self._name_label_band_below = d["name_label_band_below"]

        # ratios / colors -- resolution-independent, used as-is
        self.anchor_min_aspect = d["anchor_min_aspect"]
        self.side_margin_ratio = d["label_band_side_margin_ratio"]
        self.name_anchor_min_aspect = d["name_anchor_min_aspect"]
        self.name_text_brightness = d["name_text_brightness"]
        self.manual_roi = d.get("manual_roi")

        self._green_lower = np.array([35, 110, 110])
        self._green_upper = np.array([70, 255, 255])

        self._scale_x = 1.0  # recomputed per-frame in find_regions()
        self._scale_y = 1.0
        self._close_kernel_health = None
        self._close_kernel_name = None

    def _sx(self, value):
        """Scales a reference-resolution horizontal pixel value to the current frame."""
        return max(1, int(round(value * self._scale_x)))

    def _sy(self, value):
        """Scales a reference-resolution vertical pixel value to the current frame."""
        return max(1, int(round(value * self._scale_y)))

    def find_regions(self, frame):
        if self.manual_roi is not None:
            return [self._region_from_manual_roi(frame)]
        h, w = frame.shape[:2]
        new_scale_x = w / float(self.REFERENCE_WIDTH)
        new_scale_y = h / float(self.REFERENCE_HEIGHT)
        if new_scale_x != self._scale_x or new_scale_y != self._scale_y or self._close_kernel_health is None:
            self._scale_x = new_scale_x
            self._scale_y = new_scale_y
            self._close_kernel_health = np.ones((self._sy(3), self._sx(15)), np.uint8)
            self._close_kernel_name = np.ones((self._sy(3), self._sx(25)), np.uint8)
        return self._find_dynamic_regions(frame)

    def _region_from_manual_roi(self, frame):
        h, w = frame.shape[:2]
        r = self.manual_roi
        x1 = int(r["left"] * w)
        y1 = int(r["top"] * h)
        x2 = int(r["right"] * w)
        y2 = int(r["bottom"] * h)
        box = (x1, y1, x2 - x1, y2 - y1)
        return LabelRegion(anchor_box=box, label_box=box)

    def _find_dynamic_regions(self, frame):
        regions = self._find_healthbar_regions(frame)
        for text_region in self._find_name_text_regions(frame):
            if not self._overlaps_any(text_region.label_box, regions):
                regions.append(text_region)
        return regions

    def _find_healthbar_regions(self, frame):
        h, w = frame.shape[:2]
        min_region_x = self._sx(self._min_region_x)
        max_region_x = w - self._sx(self._max_region_x_margin)
        min_region_y = self._sy(self._min_region_y)
        max_region_y = self._sy(self._max_region_y)
        anchor_min_width = self._sx(self._anchor_min_width)
        anchor_min_height = self._sy(self._anchor_min_height)
        anchor_max_height = self._sy(self._anchor_max_height)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._green_lower, self._green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel_health)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < anchor_min_width:
                continue
            if not (anchor_min_height <= bh <= anchor_max_height):
                continue
            if bw / float(bh) < self.anchor_min_aspect:
                continue
            if not (min_region_x <= x <= max_region_x):
                continue
            if not (min_region_y <= y <= max_region_y):
                continue
            candidates.append((x, y, bw, bh))

        regions = []
        for box in candidates:
            label_box = self._label_band_above(box, w, h)
            if label_box is None:
                continue
            if self._has_text_like_content(frame, label_box):
                regions.append(LabelRegion(anchor_box=box, label_box=label_box))
        return regions

    def _find_name_text_regions(self, frame):
        """Fallback anchor: a bright white text band (the name line),
        validated by requiring a colored/saturated line of text directly
        below it (the tier line) so plain white HUD text elsewhere on
        screen isn't picked up."""
        h, w = frame.shape[:2]
        min_region_x = self._sx(self._min_region_x)
        max_region_x = w - self._sx(self._max_region_x_margin)
        min_region_y = self._sy(self._min_region_y)
        max_region_y = self._sy(self._max_region_y)
        name_min_width = self._sx(self._name_anchor_min_width)
        name_min_height = self._sy(self._name_anchor_min_height)
        name_max_height = self._sy(self._name_anchor_max_height)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, self.name_text_brightness, 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel_name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < name_min_width:
                continue
            if not (name_min_height <= bh <= name_max_height):
                continue
            if bw / float(bh) < self.name_anchor_min_aspect:
                continue
            if not (min_region_x <= x <= max_region_x):
                continue
            if not (min_region_y <= y <= max_region_y):
                continue
            candidates.append((x, y, bw, bh))

        regions = []
        for box in candidates:
            if not self._has_colored_line_below(frame, box):
                continue
            label_box = self._label_band_below_name(box, w, h)
            if label_box is None:
                continue
            regions.append(LabelRegion(anchor_box=box, label_box=label_box))
        return regions

    def _has_colored_line_below(self, frame, name_box, min_sat_ratio=0.015):
        x, y, bw, bh = name_box
        gap = self._sy(4)
        band_height = self._sy(55)
        side_pad = self._sx(20)
        y1 = y + bh + gap
        y2 = min(frame.shape[0], y1 + band_height)
        x1 = max(0, x - side_pad)
        x2 = min(frame.shape[1], x + bw + side_pad)
        if y2 <= y1 or x2 <= x1:
            return False
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat_mask = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 110)
        ratio = np.count_nonzero(sat_mask) / float(sat_mask.size)
        return ratio >= min_sat_ratio

    def _label_band_below_name(self, name_box, frame_w, frame_h):
        x, y, bw, bh = name_box
        cx = x + bw // 2
        side_margin_min = self._sx(self._side_margin_min)
        side_margin_max = self._sx(self._side_margin_max)
        side_margin = min(side_margin_max, max(side_margin_min, int(bw * 0.6)))
        x1 = max(0, cx - side_margin)
        x2 = min(frame_w, cx + side_margin)
        y1 = max(0, y - self._sy(10))
        y2 = min(frame_h, y + bh + self._sy(self._name_label_band_below))
        if y2 - y1 < self._sy(20) or x2 - x1 < self._sx(40):
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    def _overlaps_any(self, box, regions, iou_thresh=0.3):
        for r in regions:
            if self._iou(box, r.label_box) > iou_thresh:
                return True
        return False

    @staticmethod
    def _iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _label_band_above(self, anchor_box, frame_w, frame_h):
        x, y, bw, bh = anchor_box
        cx = x + bw // 2
        side_margin_min = self._sx(self._side_margin_min)
        side_margin_max = self._sx(self._side_margin_max)
        side_margin = min(side_margin_max, max(side_margin_min, int(bw * self.side_margin_ratio)))
        x1 = max(0, cx - side_margin)
        x2 = min(frame_w, cx + side_margin)
        y2 = max(0, y - self._sy(2))
        y1 = max(0, y2 - self._sy(self._band_height))
        if y2 - y1 < self._sy(20) or x2 - x1 < self._sx(40):
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    def _has_text_like_content(self, frame, label_box, min_edge_ratio=0.010):
        x, y, w, h = label_box
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edge_ratio = np.count_nonzero(edges) / float(edges.size)
        return edge_ratio >= min_edge_ratio

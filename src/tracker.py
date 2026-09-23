"""Target tracking across frames using bounding-box IoU and spatial proximity.

Maintains persistent identity for detected Brainrot nameplates so that when
multiple Brainrots are visible simultaneously, each maintains its own independent
temporal-confirmation history and alarm cooldown rather than competing for a single
global slot.
"""
from typing import Dict, List, Optional, Tuple

from .classifier import TemporalConfirmer
from .models import ClassificationResult, LabelRegion, OCRResult


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes (x, y, w, h)."""
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


def box_center_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Normalized Euclidean distance between box centers, relative to average diagonal."""
    ax_c, ay_c = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
    bx_c, by_c = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
    diag = max(1.0, 0.5 * ((a[2]**2 + a[3]**2)**0.5 + (b[2]**2 + b[3]**2)**0.5))
    dist = ((ax_c - bx_c)**2 + (ay_c - by_c)**2)**0.5
    return dist / diag


class TrackedTarget:
    """An individual Brainrot nameplate tracked across consecutive frames."""

    def __init__(self, track_id: int, region: LabelRegion, timestamp: float, settings: dict):
        self.track_id = track_id
        self.region = region
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.last_ocr_time = -1e9
        self.confirmer = TemporalConfirmer(settings)
        self.latest_ocr: Optional[OCRResult] = None
        self.latest_classification: Optional[ClassificationResult] = None
        self.is_confirmed_prestige: bool = False

    def update(self, region: LabelRegion, timestamp: float):
        self.region = region
        self.last_seen = timestamp


class TargetTracker:
    """Associates detected nameplate regions across frames using IoU and center proximity."""

    def __init__(self, settings: dict, iou_threshold: float = 0.25, max_age_seconds: float = 1.5):
        self.settings = settings
        self.iou_threshold = iou_threshold
        self.max_age_seconds = max_age_seconds
        self._next_id = 1
        self.tracks: Dict[int, TrackedTarget] = {}

    def update(self, regions: List[LabelRegion], timestamp: float) -> List[TrackedTarget]:
        # 1. Prune stale tracks that haven't been seen recently
        stale_cutoff = timestamp - self.max_age_seconds
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.last_seen >= stale_cutoff}

        if not regions:
            return list(self.tracks.values())

        # 2. Match regions to existing tracks
        matched_regions = set()
        matched_tracks = set()

        candidates = []
        for r_idx, region in enumerate(regions):
            for tid, track in self.tracks.items():
                iou = box_iou(region.label_box, track.region.label_box)
                anchor_iou = box_iou(region.anchor_box, track.region.anchor_box)
                best_iou = max(iou, anchor_iou)
                dist = box_center_distance(region.label_box, track.region.label_box)

                # Match if significant IoU overlap or very close center distance
                if best_iou >= self.iou_threshold or dist < 0.6:
                    score = best_iou + (1.0 - min(1.0, dist))
                    candidates.append((score, r_idx, tid))

        candidates.sort(key=lambda c: c[0], reverse=True)

        for score, r_idx, tid in candidates:
            if r_idx in matched_regions or tid in matched_tracks:
                continue
            self.tracks[tid].update(regions[r_idx], timestamp)
            matched_regions.add(r_idx)
            matched_tracks.add(tid)

        # 3. Create new tracks for unmatched regions
        for r_idx, region in enumerate(regions):
            if r_idx not in matched_regions:
                new_track = TrackedTarget(self._next_id, region, timestamp, self.settings)
                self.tracks[self._next_id] = new_track
                self._next_id += 1

        return list(self.tracks.values())

    def select_target_for_ocr(self, active_targets: List[TrackedTarget]) -> Optional[TrackedTarget]:
        """Picks the next tracked target to OCR, prioritizing targets waiting longest for OCR."""
        if not active_targets:
            return None
        # Sort by last_ocr_time (oldest first), breaking ties by anchor area (largest first)
        return min(
            active_targets,
            key=lambda t: (t.last_ocr_time, -(t.region.anchor_box[2] * t.region.anchor_box[3])),
        )

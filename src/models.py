"""Shared data structures passed between pipeline stages."""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class LabelRegion:
    """A candidate screen region believed to contain a Brainrot nameplate."""
    anchor_box: Tuple[int, int, int, int]  # x, y, w, h of the green anchor (rate/healthbar)
    label_box: Tuple[int, int, int, int]   # x, y, w, h of the text band above the anchor


@dataclass
class OCRResult:
    raw_text: str
    lines: list
    confidence: float  # 0-100
    processing_ms: float


@dataclass
class ClassificationResult:
    brainrot_name: str
    tier_text: str
    normalized_tier: str
    is_prestige: bool
    match_type: Optional[str]  # "exact" | "normalized" | "fuzzy" | "keyword" | None
    matched_term: Optional[str]
    match_score: float


@dataclass
class Detection:
    frame_index: int
    timestamp: float
    ocr: OCRResult
    classification: ClassificationResult
    label_box: Tuple[int, int, int, int] = field(default=None)

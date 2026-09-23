"""Turns OCR'd nameplate lines into a Prestige/not-Prestige classification.

Nameplate layout (confirmed from the supplied recordings): line 1 is the
Brainrot's name, line 2 is its tier/rank (this is the field that carries
"Six-Seven"), line 3 is the $/s rate, line 4 is the health bar. Only lines
1-2 matter for classification; later lines are numeric/currency noise.

config/prestige_brainrots.json["prestige"] may hold either kind of string:
a tier label (e.g. "Six-Seven") or a specific Brainrot name (e.g.
"UChik Chik Chik Chun"). Every OCR'd line is checked against the full list
rather than assuming a fixed line index, because a noisy frame can insert a
spurious extra line (misread background texture) and shift the real name/
tier text down by one position.
"""
import difflib
import re
from collections import deque
from typing import Optional

from .models import ClassificationResult


def normalize_text(text):
    text = text.lower()
    # Normalize unicode hyphens/dashes to standard hyphen
    text = re.sub(r"[—–−\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class PrestigeClassifier:
    def __init__(self, settings, prestige_data):
        c = settings["classification"]
        self.fuzzy_threshold = c["fuzzy_match_threshold"]
        self.prestige_keyword = normalize_text(c["prestige_keyword"])
        self.prestige_terms = list(prestige_data.get("prestige", []))
        self._normalized_terms = {normalize_text(t): t for t in self.prestige_terms}

    def _is_rate_or_health_line(self, line: str) -> bool:
        """Checks if an OCR line is currency/rate or numeric health bar rather than text."""
        norm = line.strip().lower()
        if re.search(r"/\s*s\b", norm):
            return True
        if re.match(r"^[\$€£¥₹]\s*[\d,.]+", norm):
            return True
        if re.match(r"^\d+\s*/\s*\d+$", norm) or re.match(r"^\d+%$", norm):
            return True
        return False

    def _extract_name_and_category(self, ocr_lines: list):
        """Isolates (brainrot_name, category_tier) from OCR lines.
        Line 1 is the Brainrot Name, Line 2 is the Category (Tier).
        Rate/currency/health lines are filtered out."""
        text_lines = [
            ln.strip() for ln in ocr_lines
            if ln.strip() and not self._is_rate_or_health_line(ln)
        ]
        if not text_lines:
            return "", ""

        if len(text_lines) >= 2:
            return text_lines[0], text_lines[1]

        # Single line case: check if it contains a known category merged with the name
        single = text_lines[0]
        # Target categories: prestige, shining, shinning, crystal
        cat_match = re.search(r"\b(prestige|shining|shinning|crystal)\b", single, re.IGNORECASE)
        if cat_match:
            cat = cat_match.group(1)
            name = re.sub(r"\b" + re.escape(cat) + r"\b", "", single, flags=re.IGNORECASE).strip()
            return name or single, cat

        # Known non-target categories: six-seven, elite, nebula
        cat_match_non = re.search(r"\b(six[\s-]?seven|elite|nebula)\b", single, re.IGNORECASE)
        if cat_match_non:
            cat = cat_match_non.group(1)
            name = re.sub(r"\b" + re.escape(cat) + r"\b", "", single, flags=re.IGNORECASE).strip()
            return name or single, cat

        return single, ""

    def classify(self, ocr_lines):
        name_line, tier_line = self._extract_name_and_category(ocr_lines)

        # Target ONLY the category line against the Prestige / Shining / Crystal list
        is_prestige, match_type, matched_term, score = self._match_against_list(tier_line)

        return ClassificationResult(
            brainrot_name=name_line,
            tier_text=tier_line,
            normalized_tier=normalize_text(tier_line),
            is_prestige=is_prestige,
            match_type=match_type,
            matched_term=matched_term,
            match_score=score,
        )

    def _match_against_list(self, line):
        if not line:
            return False, None, None, 0.0

        raw = line.strip()
        normalized = normalize_text(line)

        # 1. exact match (case-sensitive, as configured)
        if raw in self.prestige_terms:
            return True, "exact", raw, 1.0

        # 2. normalized match
        if normalized in self._normalized_terms:
            return True, "normalized", self._normalized_terms[normalized], 1.0

        # 3. explicit "Prestige" keyword anywhere in the category line
        if self.prestige_keyword and self.prestige_keyword in normalized:
            return True, "keyword", line.strip(), 1.0

        # 3b. known prestige term contained in the category line
        for norm_term, original_term in self._normalized_terms.items():
            if len(norm_term) >= 3 and re.search(r"\b" + re.escape(norm_term) + r"\b", normalized):
                return True, "contains", original_term, 1.0

        # 4. fuzzy match against known prestige terms
        best_term, best_score = None, 0.0
        for norm_term, original_term in self._normalized_terms.items():
            score = difflib.SequenceMatcher(None, normalized, norm_term).ratio()
            if score > best_score:
                best_score, best_term = score, original_term
        if best_term is not None and best_score >= self.fuzzy_threshold:
            return True, "fuzzy", best_term, best_score

        return False, None, None, best_score


class TemporalConfirmer:
    """Requires the same tier reading to be seen `required_matches` times
    within `window_seconds` before treating a detection as confirmed. This
    prevents a single bad OCR frame (e.g. "Six-Seve?") from triggering the
    alarm.

    Grouped on the tier text alone, not name+tier: the tier line is short
    and reads far more consistently than the (often long) name line, and
    Prestige status is entirely determined by the tier. Requiring the name
    to also match byte-for-byte across frames turned out to silently drop
    real confirmations whenever a single letter in a long name was
    misread on one frame (observed with "Cappuccina" -> "Gappuccina" in
    testing against the supplied recording) even though the tier itself
    read correctly on every one of those frames.
    """

    def __init__(self, settings):
        t = settings["temporal_confirmation"]
        self.required_matches = t["required_matches"]
        self.window_seconds = t["window_seconds"]
        self._history = deque()  # (timestamp, normalized_tier, ClassificationResult)

    def observe(self, timestamp, classification) -> Optional["ClassificationResult"]:
        norm_tier = classification.normalized_tier
        if not norm_tier:
            return None

        self._history.append((timestamp, norm_tier, classification))
        cutoff = timestamp - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        matches = [h for h in self._history if h[1] == norm_tier]
        if len(matches) >= self.required_matches:
            return matches[-1][2]
        return None

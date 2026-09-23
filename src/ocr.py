"""OCR over a nameplate ROI, with tiered preprocessing.

Tries a single fast preprocessing pass first (grayscale + upscale + a fixed
high brightness threshold), which is enough for the white name line and
brightly-colored tier text (confirmed against the supplied recordings — see
README "OCR troubleshooting"). Some tier colors (e.g. a dim purple) fall
below that cutoff and disappear, so whenever fewer than 2 lines come back —
meaning the tier line is likely missing — a second pass at a lower
brightness threshold runs to try to recover it. The most expensive variants
(adaptive threshold, CLAHE contrast+sharpen) only run if text is still
empty or low-confidence after that, to avoid burning CPU on every frame.
"""
import time

import cv2
import numpy as np
import pytesseract

from .models import OCRResult

_TESSERACT_CONFIG = "--oem 3 --psm 6"


class OCREngine:
    def __init__(self, settings):
        tesseract_cmd = settings.get("tesseract_cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.upscale_factor = settings["ocr"]["upscale_factor"]
        self.min_confidence = settings["ocr"]["min_confidence"]
        self.fixed_threshold = settings["ocr"]["fixed_brightness_threshold"]
        self.dim_threshold = settings["ocr"]["dim_brightness_threshold"]

    def read(self, roi_bgr):
        start = time.time()

        variant = self._preprocess_fast(roi_bgr)
        text, conf = self._ocr_variant(variant)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        if len(lines) < 2:
            dim_variant = self._preprocess_dim(roi_bgr)
            dim_text, dim_conf = self._ocr_variant(dim_variant)
            dim_lines = [ln.strip() for ln in dim_text.splitlines() if ln.strip()]
            if len(dim_lines) > len(lines):
                text, conf, lines = dim_text, dim_conf, dim_lines

        if not lines or conf < self.min_confidence:
            for preprocess in (self._preprocess_adaptive, self._preprocess_sharpened):
                variant = preprocess(roi_bgr)
                candidate_text, candidate_conf = self._ocr_variant(variant)
                candidate_lines = [ln.strip() for ln in candidate_text.splitlines() if ln.strip()]
                if candidate_conf > conf:
                    text, conf, lines = candidate_text, candidate_conf, candidate_lines
                if lines and conf >= self.min_confidence:
                    break

        elapsed_ms = (time.time() - start) * 1000
        return OCRResult(raw_text=text, lines=lines, confidence=conf, processing_ms=elapsed_ms)

    def _ocr_variant(self, image):
        data = pytesseract.image_to_data(
            image, config=_TESSERACT_CONFIG, output_type=pytesseract.Output.DICT
        )
        confs = []
        lines_dict = {}
        for i, word in enumerate(data["text"]):
            w = word.strip()
            if not w:
                continue
            c = data["conf"][i]
            if isinstance(c, str):
                c = float(c) if c not in ("-1", "") else -1.0
            if c >= 0:
                confs.append(c)
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines_dict.setdefault(key, []).append(w)

        lines = [" ".join(words) for words in lines_dict.values()]
        text = "\n".join(lines)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return text, avg_conf

    def _upscale(self, gray):
        if self.upscale_factor and self.upscale_factor != 1.0:
            return cv2.resize(
                gray, None, fx=self.upscale_factor, fy=self.upscale_factor,
                interpolation=cv2.INTER_CUBIC,
            )
        return gray

    def _preprocess_fast(self, roi_bgr):
        # Nameplate text (white name + bright colored tier) is consistently much
        # brighter than the game-world background behind it, so a fixed high
        # brightness cutoff isolates it far more cleanly than Otsu, which gets
        # pulled around by busy/detailed backgrounds (see README "OCR troubleshooting").
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._upscale(gray)
        _, thresh = cv2.threshold(gray, self.fixed_threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def _preprocess_dim(self, roi_bgr):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._upscale(gray)
        _, thresh = cv2.threshold(gray, self.dim_threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def _preprocess_adaptive(self, roi_bgr):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._upscale(gray)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        return thresh

    def _preprocess_sharpened(self, roi_bgr):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._upscale(gray)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

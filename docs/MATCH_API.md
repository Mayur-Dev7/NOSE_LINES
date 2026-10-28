# Spatial muzzle matcher API

This document is the contract for the spatial matcher branch. It deliberately preserves the fields already sent by `frontend/src/TwoVideoCosineMatcher.jsx` to `POST /api/video/extract-five-muzzle-parameters` and the corresponding filter/threshold controls.

## POST /api/video/match

Request JSON:
```json
{
  "video_a_path": "/path/to/video-a.mp4",
  "video_b_path": "/path/to/video-b.mp4",
  "filter_method": "frangi",
  "threshold_method": "adaptive",
  "adaptive_block_size": 15,
  "adaptive_c": 2,
  "exclude_nostrils": true,
  "min_spur_length": 8,
  "min_component_size": 30,
  "device": "cpu",
  "cpu_cores": 6
}
```

The video identifiers are the two existing `video_a_path` / `video_b_path` fields used by the current match UI. Filter/threshold names match the values sent to the extraction route. `frames_a`, `frames_b`, cache IDs, and any newly invented request fields are not part of this contract; the backend owns its frame cache.

Success response JSON:
```json
{
  "decision_score": 0.0,
  "best_score": 0.0,
  "median_score": 0.0,
  "method_a_sift": {
    "detector": "sift",
    "inlier_count": 0,
    "inlier_ratio": 0.0,
    "score": 0.0
  },
  "method_b_minutiae": {
    "matched_count": 0,
    "matched_fraction": 0.0,
    "score": 0.0
  },
  "verdict": "UNCALIBRATED",
  "dev_hint": "DEV_LIKELY_NO_MATCH",
  "calibrated": false,
  "threshold": null,
  "dev_provisional_threshold": 0.3,
  "match_viz_base64": ""
}
```

The scores above illustrate field types and shapes; real values are data-dependent. `verdict` is `MATCH` or `NO_MATCH` only when a trusted calibration threshold is available. Otherwise it must be `UNCALIBRATED`; `dev_hint` communicates a provisional development-only threshold result and is not a biometric decision.

Errors use FastAPI's standard JSON error object: `{"detail":"..."}`.

## GET /api/video/match-config

Success response JSON:
```json
{
  "calibrated": false,
  "status": "UNCALIBRATED",
  "threshold": null,
  "dev_provisional_threshold": 0.3,
  "min_calibration_genuine_pairs": 30,
  "min_calibration_impostor_pairs": 30
}
```

A threshold is trusted only when `config/matching_threshold.json` exists and records a calibrated threshold plus at least 30 genuine and 30 impostor calibration pairs. The repository's legacy `matcher_config.json` is not used to establish calibration for this matcher.

## Internal frame contract

Extractor-generated frame dictionaries retain the existing UI-facing image URLs and the 10 display parameters. Backend-only arrays used by the matcher are `clahe_gray`, `mask`, and `skeleton`, stored in the backend LRU cache and stripped from every JSON response. The request/response API never includes NumPy arrays.

## Development status

All initial weights and gates in `config/matching_params.json` are provisional pending held-out dataset evaluation. The `dev_hint` is provided only to exercise the frontend during development; it is not a calibrated identity verdict.

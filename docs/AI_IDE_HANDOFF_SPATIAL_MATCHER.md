# AI IDE Handoff — Spatial Cattle Muzzle Matcher

**Repository:** \`Mayur-Dev7/NOSE_LINES\`  
**Working branch:** \`feature/spatial-muzzle-matcher\`  
**Base branch:** \`main\`  
**Mode:** Local development only until the acceptance checks pass. Do not merge or push changes into \`main\`.

## Goal

Finish the spatial matcher so the ten global muzzle parameters are **Display Only (Not Used for Verdict)**. The matcher should compare local image features with SIFT (ORB fallback) and skeleton minutiae with spatial/orientation/patch gates and RANSAC. During development, the result must explicitly report \`verdict: "UNCALIBRATED"\` and a provisional \`dev_hint\`. A provisional score is not a biometric identity claim.

## Work rules

1. Start by reading this entire file and \`docs/MATCH_API.md\`.
2. Run \`git status --short --branch\`. Verify the branch is \`feature/spatial-muzzle-matcher\`. Never switch to or commit work directly on \`main\`.
3. Read the source before changing it. Keep existing API/UI keys and the legacy matcher reachable where necessary. Do not perform broad rewrites based on assumptions.
4. Treat \`config/matching_params.json\` as the authoritative provisional parameter file. **Do not rename or add keys** without an explicit reason and approval.
5. Do not mark the matcher calibrated unless the evaluation set has at least 30 genuine and 30 impostor pairs in the calibration split and reports held-out metrics. The development flag must never write \`config/matching_threshold.json\`.
6. Do not put NumPy arrays into HTTP JSON. Keep \`clahe_gray\`, \`mask\`, and \`skeleton\` in the backend LRU cache only; strip arrays from extraction responses.
7. Do not claim any test passed unless its command actually completed successfully. Capture full logs and report failing test names.
8. Keep edits in small commits on this feature branch. Do not merge or rebase onto \`main\` without explicit approval.

## Priority A — restore backend correctness

Inspect \`server.py\` first. Confirm these symbols exist exactly once and are in module scope before use:
- \`ExtractVideoFiveFramesRequest\`
- \`VideoMatchRequest\`
- \`_EXTRACTION_LOCK\`
- \`_json_safe_frame\`
- \`_spatial_cache_key\`
- \`_put_extraction_cache\` / \`_get_extraction_cache\`
- \`POST /api/video/extract-five-muzzle-parameters\`
- \`POST /api/video/match\`
- \`GET /api/video/match-config\`

The extraction route must exist and remain compatible with the original request names:
\`video_path\`, \`method\`, \`threshold_method\`, \`adaptive_block_size\`, \`adaptive_c\`, \`exclude_nostrils\`, \`device\`, \`cpu_cores\`, and the existing pruning options \`min_spur_length\`, \`min_component_size\`.

The match request should preserve the current frontend video identifiers and filter options:
\`video_a_path\`, \`video_b_path\`, \`filter_method\`, \`threshold_method\`, \`adaptive_block_size\`, \`adaptive_c\`, \`exclude_nostrils\`, \`min_spur_length\`, \`min_component_size\`, \`device\`, \`cpu_cores\`.
The backend owns the frame cache; the UI should not send NumPy arrays or precomputed frame dictionaries to the match endpoint.

The eight-entry LRU key must include both video identity/content metadata and all extraction/filter settings. Hold \`_EXTRACTION_LOCK\` only during extraction. Run CPU-heavy matching via \`asyncio.to_thread\` so the FastAPI event loop remains responsive.

The request/response contract in \`docs/MATCH_API.md\` is authoritative. Preserve JSON-safe extraction metadata and all existing stage image URLs/10 display parameters.

## Priority B — verify matcher and locked interfaces

Check:
- \`groove_graph.py\`: \`prune_skeleton(skel, min_spur_length=8, min_component_size=30)\` returns uint8 values \`{0,255}\`; removes connected components below size; a spur walk stops at any junction and leaves no short nub; a longer branch survives.
- \`groove_graph.py\`: \`extract_minutiae_points(skel, max_points=250)\` returns \`[{x, y, type: "endpoint"|"junction", angle, weight}]\`, ranked by incident branch length rather than arbitrary pixel order.
- \`video_groove_extractor.py\`: \`canonical_align_muzzle(crop_bgr, mask, target_size=448) -> (aligned_bgr, aligned_mask, meta)\`; use local-background contrast for small nostrils, not a global percentile; fall back to the mask principal axis; metadata includes \`alignment_method\` and \`rotation_deg\`.
- Existing \`align_muzzle_canonical\` and the extractor's five existing stage image URLs must remain compatible.
- Matcher entry points keep these signatures:
  - \`match_frame_pair(frame_a, frame_b, params_path="config/matching_params.json") -> dict\`
  - \`match_video_sets(frames_a, frames_b, params_path="config/matching_params.json", threshold_path="config/matching_threshold.json") -> dict\`
- Method A: CLAHE gray, mask erosion, glare masking, Lowe ratio, affine-partial RANSAC, reject degenerate transforms, score from inlier ratio and full-weight inlier gate.
- Method B: spatial radius, orientation delta, normalized 16×16 patch correlation, RANSAC; do not hard-reject endpoint-vs-junction pairs.
- Compare every selected frame pair; return best and median score plus base64 PNG visualization.

## Priority C — finish evaluation/calibration

Dataset format: \`dataset/<cow_id>/<video>.mp4\`.

Update \`tools/eval_matching.py\` to:
- Produce within-video sanity scores, genuine/impostor distributions, ROC, EER, plots and a JSON report.
- Print pair counts per cow and warn that same-session genuine pairs overstate accuracy.
- Use a deterministic held-out split grouped by \`cow_id\`; don't put one cow's recordings in both calibration and held-out sets.
- Write \`config/matching_threshold.json\` only when the calibration split has at least 30 genuine and 30 impostor pairs.
- \`--allow-dev-small-dataset\` may print small-folder metrics and plots but must never write/replace the threshold configuration.
- An insufficient dataset must remain \`UNCALIBRATED\`.

Audit the split math carefully. If the dataset is too small to support a meaningful held-out split, report that explicitly and don't produce a calibrated file.

## Priority D — frontend contract

File: \`frontend/src/TwoVideoCosineMatcher.jsx\`.

- POST only the exact video identifiers and existing filter/threshold controls to \`/api/video/match\`.
- Use \`GET /api/video/match-config\`.
- Show \`UNCALIBRATED (DEV MODE)\` with the provisional \`dev_hint\` when \`calibrated === false\`.
- When calibrated, use only the spatial \`verdict\` returned from the spatial matcher.
- Show decision, best and median scores; SIFT inlier count/ratio; matched-minutiae count/fraction; match visualization using \`match_viz_base64\`.
- Label ten parameters **Display Only (Not Used for Verdict)**.
- Keep the cosine utility, if retained, clearly marked as diagnostic only and completely disconnected from verdict state.
- Confirm Vite dev server compatibility at \`localhost:5173\` and backend API compatibility at \`localhost:8000\`.

## Acceptance tests

### 1. Static checks

\`\`\`bash
git status --short --branch
python -m compileall -q server.py muzzle_matcher.py groove_graph.py video_groove_extractor.py tools/eval_matching.py
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("config/matching_params.json").read_text())
expected = {
    "num_candidate_frames", "top_k_frames", "min_detection_confidence",
    "min_mask_area_ratio", "border_margin_px", "min_laplacian_var",
    "canonical_size", "min_spur_length", "min_component_size",
    "max_minutiae_points", "mask_erosion_px", "glare_threshold",
    "sift_lowe_ratio", "sift_ransac_reproj_thresh",
    "sift_min_inliers_full_weight", "minutiae_max_radius_px",
    "minutiae_max_angle_deg", "minutiae_min_patch_corr",
    "minutiae_ransac_reproj_thresh", "weight_sift", "weight_minutiae",
    "weight_best_pair", "weight_median_pair",
    "min_calibration_genuine_pairs", "min_calibration_impostor_pairs",
    "dev_provisional_threshold",
}
assert set(p) == expected, (set(p) - expected, expected - set(p))
print("matching_params.json key set OK")
PY
\`\`\`

### 2. Backend tests

\`\`\`bash
pytest -q
\`\`\`

At minimum, the focused tests must cover:
1. A 5-pixel spur is removed with no nub pixels; a 12-pixel branch survives; isolated components under 30 pixels are dropped.
2. Synthetic nostrils tilted about 15 degrees, occupying only about 3–5% of the mask, are detected using local-background contrast.
3. A shifted/rotated copy scores higher than an unrelated muzzle.
4. Missing/insufficient threshold config returns \`UNCALIBRATED\` and a \`dev_hint\`.
5. Valid calibration with 30+ genuine and 30+ impostor pairs results in \`MATCH\` or \`NO_MATCH\`.
6. Both routes return valid JSON and no ndarray serialization error.
7. Existing test suite remains green; identify any failures that existed on base separately from regressions.

### 3. Frontend build

\`\`\`bash
cd frontend
npm ci
npm run build
\`\`\`

If lockfile/dependency installation is already satisfied, \`npm run build\` is enough for the initial pass. Fix every new build/type/lint issue caused by these changes.

### 4. API smoke test

Run backend using the repo's documented environment, then:
\`\`\`bash
curl -sS http://localhost:8000/api/video/match-config
\`\`\`
Verify status is \`UNCALIBRATED\`, threshold is \`null\`, and provisional threshold is present before enough calibration pairs exist. Use a real development video path in the documented POST request to test extraction, match, visualization base64, and valid JSON response. Confirm \`http://localhost:5173\` loads the component and displays dev-only status.

## Suggested timeline (hands-on execution, not a promise of elapsed wall-clock time)

- **0–20 min — backend restoration:** check module order/imports, restore extraction route, run compileall, correct ndarray response stripping.
- **20–45 min — core algorithm checks:** run pruning and synthetic alignment tests; adjust short-spur termination and nostril detector until tests pass.
- **45–70 min — matcher/cache/API:** test all-pairs score ordering, RANSAC edge cases, cache hit/miss behavior, endpoint JSON.
- **70–95 min — evaluation tool:** deterministic cow-ID split, counts, plots/report, calibration gate and dev-mode no-write guarantee.
- **95–115 min — frontend:** match response field rendering, badge, base64 image and display-only parameter table; run Vite build.
- **115–135 min — final regression:** full pytest, build, API smoke test, inspect diff, summarize actual commands/results.

These are planning slots. Model initialization, large video datasets, and missing dependencies can take longer. Do not skip checks to fit the time window.

## Final report required from the AI IDE

Return:
1. Branch name and commit hash(es); state clearly whether any files were changed on \`main\` (expected: no).
2. Files changed and a one-line reason for each.
3. Exact commands executed with pass/fail counts and logs for failures.
4. API smoke-test request/response status; confirm no arrays leaked into JSON.
5. Whether calibration was withheld or generated; include genuine/impostor calibration counts and held-out FAR/FRR.
6. Known limitations and any remaining blockers.
7. Do not merge to \`main\`; stop after making the feature branch correct and reviewable.

# Level-2 Consolidation Report
**Date:** 2026-05-06  
**Scope:** Code consolidation after Hybrid Sampling CNN v2 upgrade  
**Overall Result:** PASS WITH NOTES

---

## Active Run Selected

| Field | Value |
|-------|-------|
| `active_run_id` | `run_20260506_113713` |
| `model_name` | `hybrid_sampling_cnn_v2` |
| `training_run_id` | `run_20260506_113713` |
| `evaluation_run_id` | `run_20260506_115143` |
| Pointer file | `outputs/reports/wafer/active_run.json` ✓ created |

**Note on split between runs:**  
Training happened in `run_20260506_113713` (produced `wafer_cnn_best.pth`, val metrics).  
Test-split evaluation and threshold calibration were re-run separately as `run_20260506_115143` using the same checkpoint. The active_run.json records both IDs and the canonical test-split metrics.

---

## Metrics (active, test split)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9430 |
| Macro F1 | 0.7312 |
| None Recall | 0.9711 |
| False Alarm Rate | 0.0289 |
| Defect Recall | 0.9067 |
| Scratch Precision | 0.3494 |
| Scratch Recall | 0.5251 |

---

## Files Verified

### `outputs/reports/wafer/runs/run_20260506_113713/` (training run)

| File | Present? | Notes |
|------|----------|-------|
| `evaluation_metrics.json` | ✓ | val split metrics (accuracy=0.9446, macro_f1=0.7399) |
| `training_metrics.json` | ✓ | 30 epochs, best_epoch=27 |
| `classification_report.csv` | ✓ | |
| `classification_report.json` | ✓ | |
| `confusion_matrix_val.png` | ✓ | val split (no test confusion matrix in this run) |
| `confusion_matrix_val_normalized.png` | ✓ | |
| `prediction_distribution.json` | ✓ | |
| `calibration_summary.json` | **✗ MISSING** | see run_20260506_115143 |
| `calibration_report.csv` | **✗ MISSING** | see run_20260506_115143 |
| `classification_report_calibrated.*` | **✗ MISSING** | see run_20260506_115143 |
| `confusion_matrix_calibrated.png` | **✗ MISSING** | see run_20260506_115143 |

### `outputs/reports/wafer/runs/run_20260506_115143/` (evaluation run, same checkpoint)

| File | Present? |
|------|----------|
| `evaluation_metrics.json` | ✓ (test split, accuracy=0.9430) |
| `confusion_matrix_test.png` | ✓ |
| `confusion_matrix_test_normalized.png` | ✓ |
| `classification_report.csv` | ✓ |
| `classification_report.json` | ✓ |
| `calibration_summary.json` | ✓ |
| `calibration_report.csv` | ✓ |
| `classification_report_calibrated.csv` | ✓ |
| `classification_report_calibrated.json` | ✓ |
| `confusion_matrix_calibrated.png` | ✓ |
| `misclassified.png` | ✓ |

**Decision:** Calibration files are NOT duplicated into run_20260506_113713 to avoid confusion. The `active_run.json` records `evaluation_run_id = run_20260506_115143` so any consumer can locate calibration data unambiguously.

---

## Checkpoint Handling

| File | Status |
|------|--------|
| `outputs/models/wafer_cnn_best.pth` | Unchanged (live, may be overwritten on next train) |
| `outputs/models/wafer_cnn_v2_hybrid_best.pth` | ✓ Created as stable copy |

`active_run.json` records both paths:
- `checkpoint_live`: `outputs/models/wafer_cnn_best.pth`
- `checkpoint_stable`: `outputs/models/wafer_cnn_v2_hybrid_best.pth`

---

## Dashboard Update Summary

**File:** `src/semiconductor_yield/dashboard/app.py`

| Change | Status |
|--------|--------|
| Added `import json` | ✓ |
| Added `WAFER_REPORTS_DIR` to config imports | ✓ |
| Added `_ACTIVE_RUN_JSON` path constant | ✓ |
| Added `_active_run()` cached loader | ✓ |
| `page_wafer()` — new Model Performance expander | ✓ shows accuracy / macro-F1 / none_recall / false_alarm_rate / defect_recall |
| `page_wafer()` — calibration section | ✓ shows threshold, calibrated metrics, and note that threshold 0.9 is a low-false-alarm operating point, not the overall best model |
| `page_home()` Module A description | ✓ updated from "WeightedRandomSampler" to "Hybrid class-group-aware sampling"; shows baseline metrics |
| Real Fab Considerations — Class Imbalance text | ✓ updated to reflect v2 hybrid sampling |
| SPC / Anomaly / RCA pages | Not touched |

---

## Scripts Checked

| Script | Overwrite risk? | Notes |
|--------|-----------------|-------|
| `scripts/train_wafer_cnn.py` | None | `--run-id` arg; auto-generates timestamp ID if unset |
| `scripts/evaluate_wafer_cnn.py` | None | Same pattern; outputs to new run dir each time |
| `scripts/compare_wafer_runs.py` | None | Read-only; writes `run_comparison_<cand>.json/csv` to reports root |

No overwrite risk found for existing runs.

---

## Text Safety Check

Scanned all `.py`, `.md`, `.json`, `.txt` files for:
- "production-ready", "production ready", "high accuracy production model"
- "confirmed root cause" (positive assertion)
- "improve yield XX%"
- "threshold 0.9 is the best model"
- "real fab performance" (positive assertion)

**Result: PASS.** All occurrences of potentially unsafe phrases are already wrapped in explicit negative qualifiers:
- "do not represent real fab performance"
- "not a confirmed root cause"
- "SPC signals are process control warnings, not confirmed root causes"
- "portfolio demo / public dataset"

---

## Risks Fixed

| Risk | Fix |
|------|-----|
| `wafer_cnn_best.pth` overwritten on next train run | Added `wafer_cnn_v2_hybrid_best.pth` stable copy |
| Dashboard showed no CNN training metrics | Added Model Performance expander with v2 metrics |
| Dashboard described v1 approach (WeightedRandomSampler) | Updated to hybrid sampling description |
| No canonical pointer to active run | Created `active_run.json` |

---

## Remaining Risks / Notes

1. **Calibration files are split across two run dirs.** The `active_run.json` records both IDs; any future script that needs calibration should read `evaluation_run_id` from `active_run.json`.

2. **README still describes v1 baseline.** Intentionally deferred — no README changes made per project constraints. README update is the explicit next step.

3. **`outputs/reports/wafer/` top-level files** (e.g., `evaluation_metrics.json`, `confusion_matrix_val.png`, `classification_report.csv`) are leftover from early v1 runs. They are not deleted (per constraint "do not overwrite old files") and are not referenced by the active dashboard or active_run.json.

4. **`wafer_cnn_best.pth` is still at risk** from any future `train_wafer_cnn.py` run. Mitigation: always pass `--run-id` and immediately copy checkpoint after training.

---

## Tests

```
pytest tests/test_wafer_cnn.py tests/test_spc.py
121 passed in 10.77s
```

All 121 tests pass.

---

## Next Step: Update README

README should be updated to:
- Replace v1 baseline metrics with v2 (accuracy=0.9430, macro_f1=0.7312, none_recall=0.9711, false_alarm_rate=0.0289)
- Replace "balanced-subset" / "WeightedRandomSampler" description with hybrid sampling
- Add calibration note (threshold 0.9 = low-false-alarm operating point)
- Add `compare_wafer_runs.py` to scripts section
- Update run commands to use `--sampling-mode hybrid`

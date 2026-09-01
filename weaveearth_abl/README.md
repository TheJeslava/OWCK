# WeaveEarth XLRS 3080 Ablation

This folder archives the lightweight pipeline files and summary metadata for the two WeaveEarth/Qwen3-VL XLRS-Bench-lite 3080 runs whose total accuracy is above 42%.

No dataset shards, raw prediction JSONL files, or model weights are included.

## Experiments

| Experiment | Samples | Correct | Total accuracy | Notes |
|---|---:|---:|---:|---|
| Final-stage text options | 3080 | 1311 | 42.56% | Uses the WeaveEarth final-stage text prompt style with XLRS choices; excludes SigLIP retrieval, routing image/router prompt, grid regions, neighbor expansion, minimal support evidence set, evidence board, and structured region metadata. |
| Full pipeline | 3080 | 1316 | 42.73% | Uses the WeaveEarth full XLRS Qwen3-VL pipeline with semantic retrieval and 6 support regions per sample. |

The full pipeline is +0.16 percentage points over the final-stage text-only run.

## Accuracy by Category

| Category | Samples | Final-stage text options | Full pipeline | Delta |
|---|---:|---:|---:|---:|
| Complex reasoning/Anomaly Detection and Interpretation | 100 | 72.00% | 74.00% | +2.00 pp |
| Complex reasoning/Environmental condition reasoning | 100 | 81.00% | 82.00% | +1.00 pp |
| Complex reasoning/Route planning | 100 | 29.00% | 30.00% | +1.00 pp |
| Counting/Counting with changing detection | 60 | 51.67% | 46.67% | -5.00 pp |
| Counting/Counting with complex reasoning | 100 | 52.00% | 52.00% | +0.00 pp |
| Counting/Overall counting | 60 | 15.00% | 21.67% | +6.67 pp |
| Counting/Regional counting | 100 | 23.00% | 50.00% | +27.00 pp |
| Land use classification/Overall Land use classification | 100 | 30.00% | 29.00% | -1.00 pp |
| Land use classification/Regional Land use classification | 200 | 77.50% | 81.00% | +3.50 pp |
| Object properties/Object classification | 800 | 41.88% | 39.62% | -2.25 pp |
| Object properties/Object color | 800 | 39.75% | 37.88% | -1.88 pp |
| Object properties/Object motion state | 60 | 66.67% | 68.33% | +1.67 pp |
| Object spatial relationship/Object spatial relationship | 500 | 27.20% | 27.00% | -0.20 pp |

## Archived Files

### Final-stage text options

- `pipeline/q3vlbase/inference_scripts/run_xlrs_qwen3vl_global_only.py`
- `artifacts/finalstage_text_options_3080/xlrs-qwen3vl-weaveearth-finalstage-text-options-3080.summary.json`
- `artifacts/finalstage_text_options_3080/xlrs-qwen3vl-weaveearth-finalstage-text-options-650.manifest.json`
- `artifacts/finalstage_text_options_3080/xlrs-qwen3vl-weaveearth-finalstage-text-options-remaining.manifest.json`
- `artifacts/finalstage_text_options_3080/xlrs-qwen3vl-weaveearth-finalstage-text-options-650.selection.json`
- `artifacts/finalstage_text_options_3080/xlrs-qwen3vl-weaveearth-finalstage-text-options-remaining.selection.json`

### Full pipeline

- `pipeline/weaveearth/inference_scripts/eval_lrs_vqa_qwen3vl.py`
- `pipeline/weaveearth/inference_scripts/run_xlrs_qwen3vl.py`
- `pipeline/weaveearth/inference_scripts/verify_xlrs_qwen3vl_full.py`
- `pipeline/weaveearth/requirements.txt`
- `artifacts/full_pipeline_3080/results.summary.json`
- `artifacts/full_pipeline_3080/650.manifest.json`
- `artifacts/full_pipeline_3080/remaining.manifest.json`
- `artifacts/full_pipeline_3080/650.selection.json`
- `artifacts/full_pipeline_3080/remaining.selection.json`

## Source Paths on Original Machine

- Final-stage text summary: `/root/autodl-tmp/otws/q3vlbase/output/xlrs-qwen3vl-weaveearth-finalstage-text-options-3080.summary.json`
- Full pipeline summary: `/root/autodl-tmp/otws/weaveearth/output/xlrs-qwen3vl-full/results.summary.json`
- Model path referenced by manifests, not included here: `/root/autodl-tmp/otws/weaveearth/ckpts/Qwen3-VL-8B-Instruct`

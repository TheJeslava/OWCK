# GE_baseline_obl

This directory contains a reproducible **no-tools prompt ablation** of the
GeoEyes XLRS-Bench evaluation pipeline. It intentionally excludes model
weights, XLRS-Bench data/images, and raw inference outputs. Obtain those
artifacts separately and pass their local paths through the environment
variables documented below.

## Base identity

The evaluated checkpoint is the published `initiacms/GeoEyes` checkpoint,
revision `7a0642355565684c9e4d40b2a88685c980f4103c`. Its local configuration
identifies the `Qwen2_5_VLForConditionalGeneration` / Qwen2.5-VL architecture.
`GE_baseline_obl` is therefore a prompt/inference-protocol ablation of
GeoEyes, not a newly trained model and not the `baseline2` model.

## What differs from GeoEyes

- The system prompt contains only `You are a helpful assistant.`
- The tool schema and all `image_zoom_in_tool` instructions are removed.
- The evaluator makes one model request per sample. It never crops an image,
  appends a tool response, or executes an unsolicited `<tool_call>` block.
- If an unsolicited tool block is emitted, it is recorded as raw output,
  ignored for execution, stripped before answer extraction, and the answer is
  extracted from that same response.
- The model checkpoint, image preprocessing, answer scoring, and balanced
  XLRS-650 sample selection remain aligned with the local GeoEyes run.

The original GeoEyes pipeline allows iterative `image_zoom_in_tool` calls and
tool-response follow-up turns. See `metadata/provenance.md` and
`metadata/accuracy_comparison.md` for the exact comparison and measurements.

## Reproduce

Tested environment: Python 3.12, vLLM 0.7.3, CUDA GPU with sufficient memory.
Install the versions in `requirements.txt` (choose a compatible PyTorch/CUDA
build for the host).

1. Download `initiacms/GeoEyes` to a local directory; do not place it in this
   repository. Set `MODEL_DIR` to that directory.
2. Prepare the 650 converted XLRS annotations and images outside Git, for
   example under `GE_baseline_obl/data/xlrsbench650`, or set `DATA_DIR` to any
   equivalent directory. The directory must contain `categories.json`, one
   annotation JSON per sample, and the referenced `images/` files.
3. Run a smoke test:

   ```bash
   MODEL_DIR=/path/to/GeoEyes \
   DATA_DIR=/path/to/xlrsbench_smoke1 \
   RESULT_DIR=/tmp/ge_baseline_obl_smoke \
   bash pipeline/run_smoke1.sh
   ```

4. Run the full 650 evaluation:

   ```bash
   MODEL_DIR=/path/to/GeoEyes \
   DATA_DIR=/path/to/xlrsbench650 \
   RESULT_DIR=/tmp/ge_baseline_obl_xlrs650 \
   MODEL_NAME=GeoEyes-no-tools-prompt-650 \
   bash pipeline/run_xlrs650.sh
   ```

The scorer requires exactly 650 unique results and writes
`xlrs650_metrics.json` plus a compact per-sample score file. The published
experiment summary is in `results/xlrs650_metrics.json`.

## Files intentionally absent

No model weights, tokenizer shards, XLRS images/annotations, vLLM logs, or
raw 650-sample conversation JSONL is tracked in this branch.

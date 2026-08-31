# Provenance

- Pipeline origin: local copy of the GeoEyes evaluation pipeline.
- Original repository: `nanocm/GeoEyes`.
- Evaluated checkpoint: `initiacms/GeoEyes`.
- Checkpoint revision: `7a0642355565684c9e4d40b2a88685c980f4103c`.
- Architecture recorded by the checkpoint config:
  `Qwen2_5_VLForConditionalGeneration` (`qwen2_5_vl`).
- Dataset source: `initiacms/XLRS-Bench-lite`.
- Evaluation subset: 650 samples, 50 per each of 13 categories, selected by
  the existing balanced selection manifest. The data and images are omitted.
- Evaluation date: 2026-08-31.
- Runtime versions used for the reported run: Python 3.12, vLLM 0.7.3,
  Transformers 4.57.6, OpenAI client 1.109.1, Pillow 11.0.0, NumPy 1.26.4,
  requests 2.34.2, tqdm 4.69.1, datasets 3.6.0.
- Scoring: deterministic GeoEyes option extraction; no independent LLM judge.

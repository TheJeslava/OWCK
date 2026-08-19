# GeoEyes XLRS Reproduction Report (650 + 530)

This branch archives two disjoint local GeoEyes inference batches on XLRS-Bench:

- Batch 1: 650 samples, with 50 samples from each of the 13 categories.
- Batch 2: 530 additional samples. Categories containing no more than 100 full-dataset samples contribute all remaining samples; the other categories contribute the next 50 unselected samples.
- Combined: 1,180 unique samples. The selection manifests are included in [`shared/sample_selection`](deepeyes_reproduction_650_650_530/shared/sample_selection/).

The report below covers the **original GeoEyes outputs only**. The `geo1` reference-bbox overlay is archived in this branch but is not included in these scores.

## Summary

| Evaluation protocol | Batch | Correct / samples | Micro accuracy | XLRS-3080 category-weighted accuracy |
|---|---:|---:|---:|---:|
| Original GeoEyes protocol | 650 | 350 / 650 | **53.85%** | **57.99%** |
| Original GeoEyes protocol | 530 | 282 / 530 | **53.21%** | **56.10%** |
| Original GeoEyes protocol | 650 + 530 | 632 / 1,180 | **53.56%** | **57.24%** |
| ZEABL deterministic protocol | 650 | 343 / 650 | **52.77%** | **57.53%** |
| ZEABL deterministic protocol | 530 | 271 / 530 | **51.13%** | **55.39%** |
| ZEABL deterministic protocol | 650 + 530 | 614 / 1,180 | **52.03%** | **56.66%** |

`Micro accuracy` weights every sampled record equally. `XLRS-3080 category-weighted accuracy` first computes accuracy within each category and then weights the 13 category accuracies by the official full XLRS-Bench category sizes, totaling 3,080 samples. For the combined rows, the category accuracy is computed from the pooled 650 + 530 records before applying the XLRS-3080 weights.

## Per-category results

| Category | XLRS count | Sampled 650 / 530 | GeoEyes 650 | GeoEyes 530 | GeoEyes combined | ZEABL 650 | ZEABL 530 | ZEABL combined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Complex reasoning / Anomaly Detection and Interpretation | 100 | 50 / 50 | 76.00% | 76.00% | 76.00% | 76.00% | 76.00% | 76.00% |
| Complex reasoning / Environmental condition reasoning | 100 | 50 / 50 | 84.00% | 78.00% | 81.00% | 84.00% | 78.00% | 81.00% |
| Complex reasoning / Route planning | 100 | 50 / 50 | 62.00% | 48.00% | 55.00% | 62.00% | 48.00% | 55.00% |
| Counting / Counting with changing detection | 60 | 50 / 10 | 50.00% | 40.00% | 48.33% | 50.00% | 40.00% | 48.33% |
| Counting / Counting with complex reasoning | 100 | 50 / 50 | 52.00% | 44.00% | 48.00% | 52.00% | 44.00% | 48.00% |
| Counting / Overall counting | 60 | 50 / 10 | 26.00% | 10.00% | 23.33% | 26.00% | 10.00% | 23.33% |
| Counting / Regional counting | 100 | 50 / 50 | 36.00% | 36.00% | 36.00% | 36.00% | 36.00% | 36.00% |
| Land use classification / Overall Land use classification | 100 | 50 / 50 | **20.00%** | **22.00%** | **21.00%** | **6.00%** | **0.00%** | **3.00%** |
| Land use classification / Regional Land use classification | 200 | 50 / 50 | 60.00% | 74.00% | 67.00% | 60.00% | 74.00% | 67.00% |
| Object properties / Object classification | 800 | 50 / 50 | 66.00% | 66.00% | 66.00% | 66.00% | 66.00% | 66.00% |
| Object properties / Object color | 800 | 50 / 50 | 68.00% | 64.00% | 66.00% | 68.00% | 64.00% | 66.00% |
| Object properties / Object motion state | 60 | 50 / 10 | 64.00% | 60.00% | 63.33% | 64.00% | 60.00% | 63.33% |
| Object spatial relationship / Object spatial relationship | 500 | 50 / 50 | 36.00% | 34.00% | 35.00% | 36.00% | 34.00% | 35.00% |

## Protocols

### Original GeoEyes protocol

The archived official GeoEyes evaluator applies rule-based option extraction first and uses an LLM judge for unresolved cases. In this local reproduction, the locally served GeoEyes checkpoint was used as that judge:

- Batch 650: 628 rule decisions and 22 LLM-judge decisions; 0 unjudged/error decisions.
- Batch 530: 510 rule decisions and 20 LLM-judge decisions; 0 unjudged/error decisions.
- Combined: 1,138 rule decisions and 42 LLM-judge decisions.

### ZEABL deterministic protocol

The ZEABL evaluator uses its deterministic Text-Before-Vision answer rules and makes zero LLM-judge calls:

- Two-pass answer-letter extraction.
- Single-choice questions: correct when the predicted and reference option sets have a non-empty intersection.
- Multi-choice questions: correct only when the predicted and reference option sets are exactly equal.
- Inference error rows remain in the denominator and are scored by the same deterministic logic (4 recorded error rows in the 650 batch and 1 in the 530 batch).

The 530 results were additionally checked by adapting the original GeoEyes JSONL records at the field boundary and running ZEABL's official `eval_xlrs.py`; the reproduced result is exactly 271 / 530 with 55.39% XLRS-3080 category-weighted accuracy.

## Interpretation

The two protocols produce identical correct counts in 12 of 13 categories. All score differences come from the multi-select **Overall Land use classification** category:

| Batch | Original GeoEyes protocol | ZEABL deterministic protocol | Difference |
|---|---:|---:|---:|
| 650 | 10 / 50 | 3 / 50 | +7 GeoEyes |
| 530 | 11 / 50 | 0 / 50 | +11 GeoEyes |
| Combined | 21 / 100 | 3 / 100 | +18 GeoEyes |

Consequently, the original GeoEyes protocol is 1.53 percentage points higher in combined micro accuracy and 0.58 percentage points higher after XLRS-3080 category weighting. These numbers are protocol-specific; they should not be compared without stating whether LLM judging was enabled.

## Source artifacts

- [Batch 650 original GeoEyes-protocol summary](deepeyes_reproduction_650_650_530/batch_01_original_650/results/original_geoeyes_protocol_local_geoeyes_judge/xlrsbench_evaluation_summary.json)
- [Batch 650 original inference output](deepeyes_reproduction_650_650_530/batch_01_original_650/results/xlrsbench_results.jsonl)
- [Batch 650 ZEABL result and per-sample decisions](deepeyes_reproduction_650_650_530/shared/scoring/zeabl_tbv/zeabl_tbv_scores.json)
- [Batch 530 original GeoEyes-protocol summary](deepeyes_reproduction_650_650_530/batch_03_next_530/results/original_geoeyes_protocol_local_geoeyes_judge/xlrsbench_evaluation_summary.json)
- [Batch 530 original inference output](deepeyes_reproduction_650_650_530/batch_03_next_530/results/xlrsbench_results.jsonl)
- [Batch 530 ZEABL-compatible deterministic metrics](deepeyes_reproduction_650_650_530/batch_03_next_530/results/deterministic/metrics.json)
- [Archived ZEABL evaluator and local adapter](deepeyes_reproduction_650_650_530/shared/scoring/zeabl_tbv/)

No model weights or XLRS image/data files are included in this branch.

# GE_baseline_obl vs. GeoEyes

Both measurements use the same balanced XLRS-650 selection: 50 samples from
each of 13 categories. The weighted figure extrapolates each category's
accuracy to its official XLRS-Bench-3080 category size. Model weights and the
dataset are not included in this repository.

## Overall

| Pipeline | Correct / 650 | Accuracy | Weighted correct equivalent / 3080 | Weighted accuracy |
|---|---:|---:|---:|---:|
| GeoEyes original tool pipeline | 350 / 650 | 53.85% | 1770.0 / 3080 | 57.47% |
| GE_baseline_obl no-tools prompt | 318 / 650 | 48.92% | 1306.4 / 3080 | 42.42% |
| Difference (ablation minus GeoEyes) | -32 | -4.92 pp | -463.6 | -15.05 pp |

## Per-category comparison

| Category | XLRS weight | GeoEyes | GE_baseline_obl | Delta |
|---|---:|---:|---:|---:|
| Complex reasoning / Anomaly Detection and Interpretation | 100 | 70% | 64% | -6 pp |
| Complex reasoning / Environmental condition reasoning | 100 | 86% | 80% | -6 pp |
| Complex reasoning / Route planning | 100 | 52% | 54% | +2 pp |
| Counting / Counting with changing detection | 60 | 62% | 62% | 0 pp |
| Counting / Counting with complex reasoning | 100 | 62% | 56% | -6 pp |
| Counting / Overall counting | 60 | 34% | 36% | +2 pp |
| Counting / Regional counting | 100 | 40% | 48% | +8 pp |
| Land use classification / Overall Land use classification | 100 | 2% | 4% | +2 pp |
| Land use classification / Regional Land use classification | 200 | 60% | 52% | -8 pp |
| Object properties / Object classification | 800 | 66% | 46% | -20 pp |
| Object properties / Object color | 800 | 68% | 30% | -38 pp |
| Object properties / Object motion state | 60 | 64% | 66% | +2 pp |
| Object spatial relationship / Object spatial relationship | 500 | 34% | 38% | +4 pp |

## Interpretation

The no-tools pipeline is the same GeoEyes checkpoint evaluated under a
different interaction protocol. It removes the tool affordance and prevents
all crop/tool-response follow-ups, so the measured difference is attributable
to prompt and inference behavior rather than a new model identity. The largest
weighted drops are in Object color (-38 pp) and Object classification (-20 pp),
which together carry 1,600 of the 3,080 full-dataset weight units.

The GE_baseline_obl run completed 650/650 unique samples with `success` status,
zero executed tool calls, and zero tool-response follow-ups. One sample
returned an empty `<answer></answer>` and was counted as incorrect.

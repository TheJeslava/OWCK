# Text-Before-Vision 实验汇报

本报告采用 ZEABL deterministic TBV 评分协议：两阶段答案提取，不调用 LLM judge。530 数据为 OWCK 选择的 `next_530` 样本；650 与 530 样本 ID 不重叠。

## 总结

| 实验 | 正确 / 样本 | Micro accuracy | XLRS-3080 类别加权准确率 | 加权等效正确数 |
|---|---:|---:|---:|---:|
| 650 | 379 / 650 | 58.3077% | 50.8831% | 1567.2 / 3080 |
| 530 | 330 / 530 | 62.2642% | 56.7532% | 1748.0 / 3080 |
| 650 + 530 | 709 / 1180 | 60.0847% | **53.7662%** | 1656.0 / 3080 |

合并加权准确率先在每个类别内合并 650 和 530 的正确数与样本数，再按官方 XLRS-Bench 3080 类别规模加权；不是两个加权准确率的简单平均。

## 分类别结果

| 类别 | XLRS 数量 | 650 | 530 | 合并 |
|---|---:|---:|---:|---:|
| Complex reasoning / Anomaly Detection and Interpretation | 100 | 46/50 (92.00%) | 42/50 (84.00%) | 88/100 (88.00%) |
| Complex reasoning / Environmental condition reasoning | 100 | 42/50 (84.00%) | 48/50 (96.00%) | 90/100 (90.00%) |
| Complex reasoning / Route planning | 100 | 40/50 (80.00%) | 37/50 (74.00%) | 77/100 (77.00%) |
| Counting / Counting with changing detection | 60 | 22/50 (44.00%) | 6/10 (60.00%) | 28/60 (46.67%) |
| Counting / Counting with complex reasoning | 100 | 28/50 (56.00%) | 26/50 (52.00%) | 54/100 (54.00%) |
| Counting / Overall counting | 60 | 14/50 (28.00%) | 1/10 (10.00%) | 15/60 (25.00%) |
| Counting / Regional counting | 100 | 25/50 (50.00%) | 21/50 (42.00%) | 46/100 (46.00%) |
| Land use classification / Overall Land use classification | 100 | 20/50 (40.00%) | 24/50 (48.00%) | 44/100 (44.00%) |
| Land use classification / Regional Land use classification | 200 | 39/50 (78.00%) | 44/50 (88.00%) | 83/100 (83.00%) |
| Object properties / Object classification | 800 | 26/50 (52.00%) | 33/50 (66.00%) | 59/100 (59.00%) |
| Object properties / Object color | 800 | 22/50 (44.00%) | 27/50 (54.00%) | 49/100 (49.00%) |
| Object properties / Object motion state | 60 | 40/50 (80.00%) | 9/10 (90.00%) | 49/60 (81.67%) |
| Object spatial relationship / Object spatial relationship | 500 | 15/50 (30.00%) | 12/50 (24.00%) | 27/100 (27.00%) |

## 结果文件

- 650: `results/current650-official/tbv_official650/tbv_score_zeabl.json`
- 530: `results/current530-owck/tbv_owck530/tbv_score_zeabl.json`
- 530 完整输出: `results/current530-owck/tbv_owck530/xlrsbench_results.jsonl`
- 运行配置: `results/current530-owck/experiment-config.json`
- 运行 provenance: `results/current530-owck/tbv_owck530/provenance.json`

模型权重、XLRS 图像和原始数据未上传；本分支保存管线代码、评测输出和评分产物。

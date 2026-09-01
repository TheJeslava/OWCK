# GeoEyes 基座模型无工具调用消融：XLRS-1680

本目录保存本次 XLRS-1680 复现所需的最小管线、精确样本选择清单和聚合指标。
包含三批原始推理 JSONL（逐样本问题、答案、模型输出和 ID，不含图片字节）。不包含模型
权重、数据集、图片、运行日志、临时文件、派生 scored JSONL 或按类别重复结果。

## 三次实验设置

| 实验 | 模型权重 | Prompt 与处理管线 | 评测规模 |
| --- | --- | --- | ---: |
| GeoEyes 原始工具管线 | `initiacms/GeoEyes`，revision `7a0642355565684c9e4d40b2a88685c980f4103c` | GeoEyes 原始工具 prompt、`image_zoom_in_tool`、迭代裁剪、工具响应回填和后续模型调用 | 650 |
| GE 无工具调用消融 | 同一份 `initiacms/GeoEyes` 权重 | 删除工具 schema、工具调用指令和裁剪/工具响应处理模块；每条样本只请求模型一次 | 650 |
| 本次基座模型消融 | `Qwen/Qwen2.5-VL-7B-Instruct`，即 GeoEyes 训练前的基座模型 | 使用与 GE 无工具消融相同的无工具 prompt 和单次请求处理方式，不加载 GeoEyes 训练后权重 | 1680 |

三次实验均采用 GeoEyes 确定性选项提取规则评分，不运行独立 LLM judge。
前两次实验的代码与结果位于本分支相邻目录：
[`deepeyes_reproduction_650_650_530`](../deepeyes_reproduction_650_650_530/) 和
[`GE_baseline_obl`](../GE_baseline_obl/)。

## 1680 样本组成

数据源为 `initiacms/XLRS-Bench-lite`，revision
`e540ee2aa745ce9a83784ae76541ddb7f79f03ac`。三批选择在官方 3080 条数据中的交集为零：

| 批次 | 样本数 | 说明 |
| --- | ---: | --- |
| `xlrs760_selection.json` | 760 | 原平衡 650，加 OCL 50、OCC 50、OMS 10 |
| `xlrs520_selection.json` | 520 | 排除前 760；十类各 50，RCCD 和 OC 各 10，不选 OMS |
| `xlrs400_remaining_selection.json` | 400 | 排除前 1280；RLUC、OCL、OCC、OSR 各 100 |

合并后共有 1680 个唯一身份样本，推理状态全部为 `success`，工具调用数为 0。

## 各类别准确率

GeoEyes 工具列读取远端归档中的确定性 `343/650` 结果；GE 无工具列读取
`GE_baseline_obl/results/xlrs650_metrics.json`；本次列读取本目录的 1680 汇总。

| 类别 | 官方权重 | GeoEyes 原始工具 | GE 无工具 | 本次基座 1680 |
| --- | ---: | ---: | ---: | ---: |
| ADI | 100 | 76.00% | 64.00% | 72.00% |
| ECR | 100 | 84.00% | 80.00% | 84.00% |
| RP | 100 | 62.00% | 54.00% | 42.00% |
| RCCD | 60 | 50.00% | 62.00% | 43.33% |
| CCR | 100 | 52.00% | 56.00% | 46.00% |
| OC | 60 | 26.00% | 36.00% | 25.00% |
| RC | 100 | 36.00% | 48.00% | 40.00% |
| OLUC | 100 | 6.00% | 4.00% | 2.00% |
| RLUC | 200 | 60.00% | 52.00% | 56.00% |
| OCL | 800 | 66.00% | 46.00% | 32.40% |
| OCC | 800 | 68.00% | 30.00% | 37.60% |
| OMS | 60 | 64.00% | 66.00% | 65.00% |
| OSR | 500 | 36.00% | 38.00% | 36.50% |

## 总体结果

类别加权准确率按 `sum(类别准确率 * 官方类别总量) / sum(官方类别总量)` 计算。

| 口径 | 权重总量 | GeoEyes 原始工具 | GE 无工具 | 本次基座 1680 |
| --- | ---: | ---: | ---: | ---: |
| 13 类 | 3080 | **57.53%** | 42.42% | 39.63% |
| 排除 OCC、OCL、OMS 的 10 类 | 1420 | **46.59%** | 46.39% | 43.77% |

本次基座模型原始 Micro 为 `726/1680 = 43.21%`。其 13 类加权准确率比
GeoEyes 原始工具管线低 `17.91 pp`，比 GE 无工具管线低 `2.79 pp`；10 类加权准确率
分别低 `2.82 pp` 和 `2.63 pp`。GeoEyes 原始工具和 GE 无工具在排除
OCC/OCL/OMS 后仅差 `0.20 pp`，完整 13 类的主要差距来自高权重 OCL/OCC。

## 复现

测试环境为 Python 3.12、vLLM 0.7.3、Transformers 4.57.6 和单张 24 GB GPU。
安装 `requirements.txt`，另行安装与宿主 CUDA 匹配的 PyTorch，然后执行：

```bash
MODEL_DIR=/path/to/Qwen2.5-VL-7B-Instruct \
DATASET_PATH=/path/to/XLRS-Bench-lite-3080 \
bash run_xlrs1680.sh
```

服务参数固定为 `temperature=0`、`max_tokens=1024`、`max_model_len=21248`、
`max_num_seqs=1`、`gpu_memory_utilization=0.90`、单请求最多 13 张图并启用 eager 模式。
系统 prompt 为 `You are a helpful assistant.`，不提供工具 schema，也不执行模型自行输出的工具调用。

脚本直接按三个 manifest 中的官方 Arrow 行号逐条读取数据，只在单条请求期间生成临时 JPEG，
请求完成即删除；不会复制或导出完整数据集。每批支持按成功 `sample_id` 断点续跑。

## 文件说明

- `pipeline/`：无工具调用评估逻辑和原规则评分函数。
- `interface/`：Arrow 行号适配、批次评分和三批指标合并。
- `metadata/`：构成本次 1680 的三个精确选择清单。
- `results/`：760、520、400 三批原始推理 JSONL 与指标，以及 1680 汇总；原始结果中的
  `data:image/jpeg;base64,` 仅为空占位符，不含图片内容。

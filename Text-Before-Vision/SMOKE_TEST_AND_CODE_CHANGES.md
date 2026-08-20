# Text-Before-Vision 冒烟测试与代码改动报告

报告日期：2026-08-20

## 1. 结论

**不能确认当前工作树已经通过端到端冒烟测试。更准确的结论是：静态语法检查通过，但已有冒烟运行没有形成“功能和结果都通过”的证据。**

当前 Git 基线为官方代码提交 `323199c`（`origin/main`），工作树只有以下一个文件存在未提交修改：

```text
evaluation/acc/eval_multi_xlrsbench2.py
```

验证结果如下：

| 检查项 | 结果 | 证据 |
|---|---|---|
| Python 语法编译 | 通过 | `python -m py_compile evaluation/acc/eval_multi_xlrsbench2.py` 输出 `PY_COMPILE_OK` |
| Shell 脚本语法 | 通过 | `bash -n /root/autodl-tmp/tbv-xlrs-lite/run_smoke.sh` 输出 `RUN_SMOKE_BASH_OK` |
| Git diff 空白检查 | 通过 | `git diff --check` 无报错 |
| 冒烟运行一（2026-08-20 17:55） | 失败 | vLLM 返回 HTTP 400；结果 `status=error`，评分 `errors=1`、`accuracy=0.0` |
| 冒烟运行二（2026-08-20 18:00） | 请求链路成功，但结果未通过 | 结果 `status=success`，预测为 `AB`，真值为 `D`，评分 `accuracy=0.0` |

相关产物：

- `/root/autodl-tmp/tbv-xlrs-lite/results/smoke-official-20260820/tbv_smoke/xlrsbench_results.jsonl`
- `/root/autodl-tmp/tbv-xlrs-lite/results/smoke-official-20260820/tbv_smoke/tbv_score_smoke.json`
- `/root/autodl-tmp/tbv-xlrs-lite/results/smoke-official-20260820/vllm-server.log`
- `/root/autodl-tmp/tbv-xlrs-lite/results/smoke-official-20260820-32768/tbv_smoke/xlrsbench_results.jsonl`
- `/root/autodl-tmp/tbv-xlrs-lite/results/smoke-official-20260820-32768/tbv_smoke/tbv_score_smoke.json`

因此，当前最稳妥的状态汇报是：**代码级检查通过；端到端冒烟测试尚未通过，且当前工作树的修改版本没有被重新做过一次完整的模型服务加评测验证。**

## 2. 对比基线

本报告以官方基线提交 `323199c` 中的
`evaluation/acc/eval_multi_xlrsbench2.py` 为对比对象。

Git 差异统计：

```text
47 insertions(+), 4 deletions(-)
```

除上述评测脚本外，没有发现其他官方代码文件被修改。

## 3. 每一处代码改动及目的

### 3.1 新增 `--max-samples`

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:30`

```diff
- 无此参数
+ parser.add_argument("--max-samples", type=int, default=None,
+                     help="Optional global sample cap for smoke runs")
```

**改成了什么：** 从没有全局样本数量限制，改为支持通过 `--max-samples N` 限制本次评测最多处理 `N` 个尚未处理样本。

**目的：** 支持快速冒烟运行，避免为了验证服务链路而启动完整数据集评测。该限制是跨类别的全局预算，不是每个类别各处理 `N` 个样本。

### 3.2 新增 `--sample-id`

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:31`

```diff
- 无此参数
+ parser.add_argument("--sample-id", action="append", default=None,
+                     help="Optional unique_id to run; can be repeated")
```

**改成了什么：** 从只能遍历整个数据目录，改为可以指定一个或多个样本的 `unique_id`。

**目的：** 支持精确复现单个样本或少量样本。实现还支持逗号分隔输入，例如：

```bash
--sample-id id_a,id_b --sample-id id_c
```

这些 ID 会先被合并为集合，再参与数据筛选。

### 3.3 新增 `--max-tokens`

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:32`

```diff
- 无此参数
+ parser.add_argument("--max-tokens", type=int, default=1024,
+                     help="Maximum new tokens per model call")
```

**改成了什么：** 从模型调用的输出 token 上限固定为 1024，改为可通过命令行配置，默认仍为 1024。

**目的：** 冒烟测试可以使用更小的输出上限以减少资源消耗；正式评测仍可保持原默认行为。

### 3.4 放宽已处理样本 ID 的读取

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:150-151`

```diff
- processed_sample_ids.add(result["sample_id"])
+ if result.get("sample_id"):
+     processed_sample_ids.add(result["sample_id"])
```

**改成了什么：** 从强制要求每条 JSON 结果都存在 `sample_id`，改为只在字段存在且非空时加入断点续跑集合。

**目的：** 读取历史结果时，避免因缺少或空的 `sample_id` 直接抛出 `KeyError`，提高断点续跑的容错性。

### 3.5 新增请求样本 ID 解析函数

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:157-163`

```python
def requested_sample_ids():
    if not args.sample_id:
        return None
    ids = set()
    for value in args.sample_id:
        ids.update(part.strip() for part in value.split(",") if part.strip())
    return ids or None
```

**改成了什么：** 新增对重复出现的 `--sample-id` 参数和逗号分隔 ID 的统一解析。

**目的：** 将命令行输入规范化为集合，便于后续筛选并自动去重；没有指定时返回 `None`，保持原有全量遍历逻辑。

### 3.6 新增按数据集位置排序的函数

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:166-173`

```python
def dataset_sort_key(item):
    sample_path, _category = item
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            return (json.load(f).get("dataset_position", 0), sample_path)
    except Exception:
        return (0, sample_path)
```

**改成了什么：** 新增读取样本 JSON 中 `dataset_position` 的排序键，并以文件路径作为并列排序依据；读取失败时回退为 `(0, sample_path)`。

**目的：** 让 `--max-samples` 截取的是稳定、可复现的数据集顺序，而不是依赖文件系统返回顺序。

### 3.7 新增从样本路径获取真实 ID 的函数

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:175-180`

```python
def sample_id_for_path(sample_path):
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f).get(
                "unique_id", os.path.basename(sample_path).replace(".json", "")
            )
    except Exception:
        return os.path.basename(sample_path).replace(".json", "")
```

**改成了什么：** 从默认使用 JSON 文件名作为样本 ID，改为优先使用样本内容中的 `unique_id`，缺失或读取失败时才回退到文件名。

**目的：** 使指定样本筛选、断点续跑和最终保存的 `sample_id` 使用同一套 ID 规则，避免“文件名 ID”和“数据内 unique_id”不一致造成重复评测或漏评。

### 3.8 将模型调用的输出上限改为可配置值

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:396-397`

```diff
- "max_tokens": 1024,
+ "max_tokens": args.max_tokens,
```

**改成了什么：** 将 API 请求中的固定值 `1024` 改为命令行参数 `args.max_tokens`。

**目的：** 让新增的 `--max-tokens` 参数真正作用于每次模型调用，并保留默认值 1024 的兼容行为。

### 3.9 原 `TBV_FAIL_FAST` 改动已还原

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:507-509`

当前逻辑已恢复为官方版本：

```python
except Exception as e:
    print(f"Error!!!!", e)
    status = "error"
```

**改成了什么：** 删除了本地新增的 `TBV_FAIL_FAST` 判断和 `raise`，恢复为无条件捕获异常、记录错误并继续处理后续样本。

**目的：** 与官方异常处理行为保持一致。注意：本地 `/root/autodl-tmp/tbv-xlrs-lite/run_pass1.sh` 仍保留 `export TBV_FAIL_FAST=1`，但当前评测 Python 脚本已不读取该变量，因此该环境变量现在没有实际效果。

### 3.10 将样本目录遍历改为排序遍历

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:565`

```diff
- for filename in os.listdir(xlrsbench_path):
+ for filename in sorted(os.listdir(xlrsbench_path)):
```

**改成了什么：** 从依赖 `os.listdir()` 的返回顺序，改为按文件名排序后遍历。

**目的：** 提升样本发现顺序的确定性，配合 `dataset_position` 排序，保证冒烟截样和结果复现更加稳定。

### 3.11 在收集样本时按 `unique_id` 过滤

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:571-574`

```diff
                 category = data.get("category", "unknown")
+                unique_id = data.get("unique_id", os.path.basename(file_path).replace(".json", ""))
+                if requested_ids is not None and unique_id not in requested_ids:
+                    continue
```

**改成了什么：** 从收集目录下所有符合类别的 JSON，改为在提供 `--sample-id` 时只收集匹配的样本。

**目的：** 让 `--sample-id` 在进入进程池前完成筛选，减少无关文件读取和模型调用。

### 3.12 对每个类别的样本按 `dataset_position` 排序

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:586-587`

```python
for files in json_files_by_category.values():
    files.sort(key=dataset_sort_key)
```

**改成了什么：** 新增类别内排序步骤。

**目的：** 保证跨类别处理时，每个类别内部按照数据集位置稳定排列，使全局 `--max-samples` 截取结果可重复。

### 3.13 增加全局样本预算并在类别间传递

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:598-618`

```diff
     with open(results_file, "a", encoding="utf-8") as f_results:
+        remaining_samples = args.max_samples
         # 按类别评测
         for test_type in test_types:
+            if remaining_samples is not None and remaining_samples <= 0:
+                break
...
+            if remaining_samples is not None:
+                image_args = image_args[:remaining_samples]
+                remaining_samples -= len(image_args)
```

**改成了什么：** 从每个类别不受限制地构造评测任务，改为维护一个跨类别共享的剩余样本数；预算耗尽后停止后续类别。

**目的：** 使 `--max-samples N` 的语义成为整个评测任务最多处理 `N` 个样本，而不是每个类别分别处理 `N` 个样本。

### 3.14 断点续跑过滤改用样本内容中的 ID

**位置：** `evaluation/acc/eval_multi_xlrsbench2.py:611-615`

```diff
                 (file_path, category)
                 for file_path, category in json_files_by_category[test_type]
-                if os.path.basename(file_path).replace(".json", "") not in processed_sample_ids
+                if sample_id_for_path(file_path) not in processed_sample_ids
```

**改成了什么：** 从按文件名判断样本是否已处理，改为按 `sample_id_for_path()` 得到的真实样本 ID 判断。

**目的：** 与结果文件里的 `sample_id` 保持一致，避免数据文件名和 `unique_id` 不同导致断点续跑失效。

## 4. 影响与注意事项

1. 默认不传新增参数时，样本上限仍为不限，输出 token 上限仍为 1024；原有批量评测路径基本保持不变。
2. `--max-samples` 只限制本次新建的评测任务，不会删除或覆盖已有总结果文件；结果仍以追加方式写入。
3. 断点续跑仍按 `sample_id` 去重，不区分同一 `sample_id` 的不同运行轮次。
4. `TBV_FAIL_FAST` 的 Python 处理逻辑已按官方版本还原；本地启动脚本中的同名环境变量目前不会改变评测行为。
5. 当前文件中还有一些更早就存在于官方基线的图像处理、多图提示词和工具调用逻辑；本报告没有把这些基线代码误列为本次改动。

## 5. 建议的最终验收标准

要把当前工作树标记为“冒烟测试通过”，至少应重新运行当前版本并同时满足：

- 模型服务成功启动并通过 `/v1/models` 健康检查；
- 评测脚本正常完成，结果中样本 `status` 不为 `error`；
- 结果文件包含预期样本数；
- 预测格式能够被评分器提取；
- 对选定冒烟样本，评分器不报告请求错误，且准确率是否合格应按项目约定单独判断。

## 6. 像素上限核对

这里有三个容易混淆的“像素上限”：

| 环节 | 本地设置 | 官方设置 | 说明 |
|---|---:|---:|---|
| PIL 读取图片的 `Image.MAX_IMAGE_PIXELS` | `1,000,000,000` | `1,000,000,000` | 防止 PIL 对超大图像触发 decompression-bomb 限制；不是模型输入尺寸。 |
| 评测脚本 crop 后 `smart_resize` 的 `MAX_PIXELS` | `4096 * 4096 = 16,777,216` | `4096 * 4096 = 16,777,216` | 限制裁剪图经过 resize 后的目标面积；本地与官方评测脚本一致。 |
| vLLM/视觉处理器 `max_pixels` | 模型 `preprocessor_config.json` 为 `16,777,216` | 官方 `s21.sh` 没有显式设置该 CLI 参数 | 本地 `run_pass1.sh` 的 `--max-pixels` 只传给 provenance 记录，不传给 vLLM；实际视觉处理器值来自模型配置。官方模型配置不在该 Git 仓库内，不能仅凭 `s21.sh` 推断其运行时配置。 |

另外，官方 `s21.sh` 的
`--limit-mm-per-prompt '{"image": 32}'` 限制的是**每个请求最多多少张图片**，不是图片像素数。当前本地 `run_pass1.sh`/`run_smoke.sh` 使用的是 `{"image":2}`，这同样是图片数量限制，不是 `MAX_PIXELS`。

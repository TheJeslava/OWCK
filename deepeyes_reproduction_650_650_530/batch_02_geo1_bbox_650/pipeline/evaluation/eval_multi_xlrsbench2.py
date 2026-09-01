import os
import json
import numpy as np
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
import argparse
import torch
from tqdm import tqdm
import math
import re
from io import BytesIO
from PIL import Image
import base64
import io
from openai import OpenAI
import requests

from question_bbox_monitor import (
    extract_question_bbox,
    has_complete_tool_call,
    is_question_bbox_category,
    normalize_bbox,
    parse_tool_call,
    synthetic_tool_call,
)

Image.MAX_IMAGE_PIXELS = 10_0000_0000


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="qwen", help="Model name for result save")
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1", help="API URL")
parser.add_argument("--xlrsbench_path", type=str, default=None, help="Path to the XLRS benchmark")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the results")
parser.add_argument("--eval_model_name", type=str, default=None, help="Model name for evaluation")
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()


openai_api_key = args.api_key
openai_api_base = args.api_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models["data"][0]["id"]
else:
    eval_model_name = args.eval_model_name

xlrsbench_path = args.xlrsbench_path
save_path = args.save_path
save_path = os.path.join(save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)
abc_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}

# 适配XLRSBench的超大图像 - 调整常量但保留原始处理逻辑
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
# 增加MAX_PIXELS以支持更大的图像
# Keep visual-token memory bounded on a single 24 GB GPU while preserving the
# official resize and iterative crop flow.
INITIAL_MAX_PIXELS = int(os.environ.get("GEOEYES_INITIAL_MAX_PIXELS", "2000000"))
MAX_PIXELS = int(os.environ.get("GEOEYES_CROP_MAX_PIXELS", "900000"))

# 多选题类别列表（如果有更多类别需要添加）
MULTI_ANSWER_CATEGORIES = ["Land use classification/Overall Land use classification"]

# 系统提示词 - 更新以支持多图和选择图像的工具
instruction_prompt_system = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"image_index":{"type":"integer","description":"The index of the image to crop (0 for the first image, 1 for the second image, etc.). For single image samples, use 0."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "image_index": 0, "label": "the apple on the desk"}}  
</tool_call>

For questions with multiple images, you can specify which image to crop using the image_index parameter (0 for the first image, 1 for the second image)."""

USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

# 区分单选和多选的提示词模板
instruction_prompt_single = (
    """Question: {question}
Options: {options}
Select the best answer for the multiple-choice question based on the image. Only respond with the letter corresponding to the correct answer (A, B, C, D).
"""
    + USER_PROMPT_V2
)

instruction_prompt_multi = (
    """Question: {question}
Options: {options}
Select the best answer(s) for the multiple-choice question based on the image. There may be more than one correct option. Only respond with the letter(s) corresponding to the correct answer(s) (A, B, C, D), with multiple choices separated by spaces.
"""
    + USER_PROMPT_V2
)

# 多图样本的提示词模板 - 通知模型有多张图像
instruction_prompt_multi_image = (
    """Question: {question}
Options: {options}
This question involves analyzing MULTIPLE IMAGES. Examine all images carefully before answering.
Select the best answer for the multiple-choice question based on the images. Only respond with the letter corresponding to the correct answer (A, B, C, D).
"""
    + USER_PROMPT_V2
)

# 多选+多图的提示词模板
instruction_prompt_multi_answer_multi_image = (
    """Question: {question}
Options: {options}
This question involves analyzing MULTIPLE IMAGES. Examine all images carefully before answering.
Select the best answer(s) for the multiple-choice question based on the images. There may be more than one correct option. Only respond with the letter(s) corresponding to the correct answer(s) (A, B, C, D), with multiple choices separated by spaces.
"""
    + USER_PROMPT_V2
)

user_prompt = USER_PROMPT_V2

start_token = "<tool_call>"
end_token = "</tool_call>"
MAX_GENERATION_TOKENS = int(os.environ.get("GEOEYES_MAX_TOKENS", "1024"))


# ==================== 新增功能：断点续跑 ====================


def load_processed_sample_ids(results_file):
    """
    从已存在的结果文件中读取已处理的样本 ID
    参数:
        results_file: 评测结果文件路径
    返回:
        已处理样本的 ID 集合
    """
    processed_sample_ids = set()
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed_sample_ids.add(result["sample_id"])
                except json.JSONDecodeError:
                    print("跳过无效的 JSON 行")
    return processed_sample_ids


# ==================== 原有功能 ====================


# 函数：提取答案中的字母
def extract_answer_letters(text):
    """提取答案中的字母，支持多个字母"""
    if not text:
        return ""

    # 清理一些常见的前缀
    prefixes = [
        "The answer is",
        "The best answer is",
        "The correct answer is",
        "The answers are",
        "The best answers are",
        "The correct answers are",
        "The answer",
        "Answer",
        "答案",
        "答案是",
    ]
    for prefix in prefixes:
        text = text.replace(prefix, "")

    # 尝试多种匹配模式
    # 1. 尝试匹配括号中的字母: (A) (B)
    matches = re.findall(r"\(([A-Ea-e])\)", text)

    # 2. 如果没有找到，尝试匹配单独的字母: A B C
    if not matches:
        matches = re.findall(r"(?:^|\s)([A-Ea-e])(?:$|[\s,.])", text)

    # 3. 如果仍然没有找到，匹配任何字母
    if not matches:
        matches = re.findall(r"[A-Ea-e]", text)

    # 转换为大写并去重
    if matches:
        return "".join(sorted(set(m.upper() for m in matches)))
    return ""


# # 保留原始图像处理函数
# def encode_image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')


# def encode_pil_image_to_base64(pil_image):
#     buffered = BytesIO()
#     pil_image.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
#     return img_str
def encode_image_to_base64(image_path):
    # with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode("utf-8")
    format = "JPEG"
    image = Image.open(image_path)
    new_h, new_w = smart_resize(
        image.height,
        image.width,
        factor=IMAGE_FACTOR,
        max_pixels=INITIAL_MAX_PIXELS,
    )
    if image.size != (new_w, new_h):
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# 保留原始图像缩放函数
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


# 判断是否为多选题
def is_multi_answer_question(category):
    # 当前仅支持Land use classification/Overall Land use classification类别的多选
    if category.startswith("Land use classification/Overall Land use classification"):
        return True
    return False


# 修改后的处理函数，支持多图样本
def process(img_arg):
    sample_path, category = img_arg

    # 加载样本数据
    with open(sample_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    # 检测是否为多图样本
    is_multi_image = anno.get("is_multi_image", False)

    # 获取问题和选项
    question = anno["question"]
    options = anno["options"]
    question_bbox_mode = is_question_bbox_category(category)
    question_bbox = extract_question_bbox(question) if question_bbox_mode else None
    option_str = "\n"
    for i in range(len(options)):
        option_str += abc_map[i + 1] + ". " + options[i] + "\n"

    # 判断是否为多选题
    is_multi = is_multi_answer_question(category)

    # 根据题型选择提示词模板
    if is_multi_image:
        if is_multi:
            prompt = instruction_prompt_multi_answer_multi_image.format(question=question, options=option_str)
        else:
            prompt = instruction_prompt_multi_image.format(question=question, options=option_str)
    else:
        if is_multi:
            prompt = instruction_prompt_multi.format(question=question, options=option_str)
        else:
            prompt = instruction_prompt_single.format(question=question, options=option_str)

    # 处理不同类型的图像输入
    if is_multi_image:
        # 多图样本处理
        image_paths = [os.path.join(os.path.dirname(sample_path), path) for path in anno["image_paths"]]
        pil_images = [Image.open(img_path) for img_path in image_paths]
        base64_images = [encode_image_to_base64(img_path) for img_path in image_paths]

        # 构建包含所有图像的初始消息
        content = []
        for base64_img in base64_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "system", "content": instruction_prompt_system}, {"role": "user", "content": content}]

        # 为打印准备消息
        print_content = []
        for _ in range(len(base64_images)):
            print_content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}})
        print_content.append({"type": "text", "text": prompt})

        print_messages = [{"role": "system", "content": instruction_prompt_system}, {"role": "user", "content": print_content}]
    else:
        # 单图样本处理
        img_path = os.path.join(os.path.dirname(sample_path), anno["image_path"])
        pil_img = Image.open(img_path)
        base64_image = encode_image_to_base64(img_path)

        messages = [
            {"role": "system", "content": instruction_prompt_system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        print_messages = [
            {"role": "system", "content": instruction_prompt_system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    chat_message = messages
    response_message = ""
    status = "success"
    try_count = 0
    turn_idx = 0
    tool_events = []

    # 修改工具调用逻辑以支持多图样本并修复重复回答问题
    try:
        while True:
            if try_count > int(os.environ.get("MAX_TRY", 10)):
                break

            params = {
                "model": eval_model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": MAX_GENERATION_TOKENS,
                "stop": ["<|im_end|>\n".strip()],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content

            model_requested_tool = has_complete_tool_call(response_message)
            synthetic_initial_call = question_bbox_mode and turn_idx == 0 and not model_requested_tool
            execute_tool = model_requested_tool or synthetic_initial_call

            # For question-bbox categories, the first turn always gets the
            # annotated crop before an answer is accepted. Other categories
            # retain the official answer-first behavior.
            if (
                "</answer>" in response_message
                and "<answer>" in response_message
                and not (question_bbox_mode and execute_tool)
            ):
                # 直接添加最终响应到历史记录并跳出
                p_message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)
                break

            if execute_tool:
                action_list = None
                parse_error = None
                if model_requested_tool:
                    try:
                        action_list = parse_tool_call(response_message)
                    except Exception as exc:
                        if not question_bbox_mode:
                            raise
                        parse_error = str(exc)

                bbox_list = []
                cropped_pil_image_content_list = []

                requested_bbox = None
                if action_list is not None and "bbox_2d" in action_list["arguments"]:
                    try:
                        requested_bbox = normalize_bbox(action_list["arguments"]["bbox_2d"])
                    except Exception as exc:
                        if not question_bbox_mode:
                            raise
                        parse_error = str(exc) if parse_error is None else f"{parse_error}; {exc}"

                # The three question-bbox categories always use the annotated
                # region, regardless of the coordinates emitted by the model.
                bbox = list(question_bbox) if question_bbox_mode else requested_bbox
                if bbox is None:
                    raise ValueError("tool call has no bbox_2d argument")

                # 处理可选的图像索引参数
                image_index = action_list["arguments"].get("image_index", 0) if action_list else 0

                # 根据是否为多图样本处理裁剪
                if is_multi_image:
                    # 确保图像索引在有效范围内
                    if 0 <= image_index < len(pil_images):
                        target_img = pil_images[image_index]
                    else:
                        # 索引超出范围，默认使用第一张图
                        image_index = 0
                        target_img = pil_images[0]
                else:
                    # 单图样本
                    target_img = pil_img

                # 裁剪图像
                left, top, right, bottom = bbox
                if question_bbox_mode and not (
                    0 <= left < right <= target_img.width and 0 <= top < bottom <= target_img.height
                ):
                    raise ValueError(
                        f"question bbox {bbox} is outside source image {target_img.size}"
                    )
                cropped_image = target_img.crop((left, top, right, bottom))
                source_crop_size = [cropped_image.width, cropped_image.height]
                new_h, new_w = smart_resize(cropped_image.height, cropped_image.width, factor=IMAGE_FACTOR)
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                cropped_pil_image = encode_pil_image_to_base64(cropped_image)

                tool_events.append(
                    {
                        "turn": turn_idx,
                        "model_requested_tool": model_requested_tool,
                        "synthetic_initial_call": synthetic_initial_call,
                        "requested_bbox": requested_bbox,
                        "applied_bbox": bbox,
                        "question_bbox_override": question_bbox_mode,
                        "image_index": image_index,
                        "source_image_size": [target_img.width, target_img.height],
                        "source_crop_size": source_crop_size,
                        "returned_image_size": [new_w, new_h],
                        "parse_error": parse_error,
                        "intercepted_response": response_message if synthetic_initial_call else None,
                    }
                )

                # 记录裁剪信息
                bbox_list.append({"bbox": bbox, "image_index": image_index})

                # 准备裁剪图像内容
                cropped_pil_image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"},
                }
                cropped_pil_image_content_list.append(cropped_pil_image_content)

                if len(bbox_list) == 1:
                    bbox_list = bbox_list[0]
                user_msg = user_prompt

                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                for cropped_pil_image_content in cropped_pil_image_content_list:
                    content_f.append(cropped_pil_image_content)
                content_f.append({"type": "text", "text": user_msg})
                content_f.append({"type": "text", "text": "</tool_response>"})

                assistant_history_content = response_message
                if synthetic_initial_call:
                    assistant_history_content = synthetic_tool_call(bbox)

                _message = [
                    {
                        "role": "assistant",
                        "content": assistant_history_content,
                    },
                    {
                        "role": "user",
                        "content": content_f,
                    },
                ]

                chat_message.extend(_message)

                p_message = [
                    {
                        "role": "assistant",
                        "content": assistant_history_content,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                            {"type": "text", "text": user_msg},
                        ],
                    },
                ]
                print_messages.extend(p_message)
                turn_idx += 1
            else:
                # 如果不是工具调用也不是最终答案，只记录助手回复
                p_message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)

            try_count += 1
    except Exception as e:
        print(f"Error!!!!", e)
        status = "error"

    # 结果提取逻辑 - 修改以支持多选
    if "</answer>" in response_message and "<answer>" in response_message:
        output_text = response_message.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        output_text = response_message

    # 提取答案字母
    extracted_answer = extract_answer_letters(output_text)

    # 适应XLRSBench数据格式的保存信息 - 支持多图
    save_info = {}
    save_info["question"] = question
    save_info["answer"] = anno["answer"]
    save_info["extracted_answer"] = extracted_answer  # 添加提取后的答案字母
    save_info["is_multi"] = is_multi  # 记录是否为多选题
    save_info["category"] = category
    # save_info['subcategory'] = anno.get('subcategory', 'default')
    save_info["pred_ans"] = output_text
    save_info["pred_output"] = print_messages
    save_info["status"] = status
    model_tool_calls = sum(event["model_requested_tool"] for event in tool_events)
    synthetic_tool_calls = sum(event["synthetic_initial_call"] for event in tool_events)
    save_info["tool_usage"] = {
        "model_calls": model_tool_calls,
        "synthetic_calls": synthetic_tool_calls,
        "effective_calls": len(tool_events),
    }
    save_info["question_bbox_monitor"] = {
        "enabled": question_bbox_mode,
        "question_bbox": question_bbox,
        "events": tool_events,
    }

    # 根据样本类型保存图像信息
    if is_multi_image:
        save_info["is_multi_image"] = True
        save_info["images"] = [os.path.basename(path) for path in image_paths]
        sample_id = anno.get("unique_id", os.path.basename(sample_path).replace(".json", ""))
    else:
        save_info["image"] = os.path.basename(img_path)
        sample_id = anno.get("unique_id", os.path.basename(sample_path).replace(".json", ""))

    save_info["sample_id"] = sample_id

    return save_info


if __name__ == "__main__":
    # 加载类别信息
    try:
        with open(os.path.join(xlrsbench_path, "categories.json"), "r") as f:
            category_mapping = json.load(f)
        test_types = list(category_mapping.keys())
    except:
        # 如果没有类别文件，则将所有样本视为一个类别
        test_types = ["all"]

    # 获取所有json文件并按类别组织
    json_files_by_category = {}
    for test_type in test_types:
        json_files_by_category[test_type] = []

    # 遍历所有JSON文件
    for filename in os.listdir(xlrsbench_path):
        if filename.endswith(".json") and filename != "categories.json" and filename != "processing_log.json":
            file_path = os.path.join(xlrsbench_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                category = data.get("category", "unknown")

                # 添加到对应类别
                if category in json_files_by_category:
                    json_files_by_category[category].append((file_path, category))
                else:
                    # 如果是未知类别但存在test_types["all"]
                    if "all" in json_files_by_category:
                        json_files_by_category["all"].append((file_path, category))
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")

    # 创建保存评测结果的JSONL文件
    results_file = os.path.join(save_path, "xlrsbench_results.jsonl")

    # 加载已处理的样本 ID
    processed_sample_ids = load_processed_sample_ids(results_file)
    print(f"已处理样本数: {len(processed_sample_ids)}")

    # 打开结果文件用于附加
    with open(results_file, "a", encoding="utf-8") as f_results:
        # 按类别评测
        for test_type in test_types:
            if not json_files_by_category[test_type]:
                print(f"类别 {test_type} 没有样本，跳过评测")
                continue

            save_name = f"result_{test_type.replace('/', '_')}_{args.model_name}.jsonl"
            save_json = []

            pool = multiprocessing.Pool(processes=args.num_workers)
            image_args = [
                (file_path, category)
                for file_path, category in json_files_by_category[test_type]
                if os.path.basename(file_path).replace(".json", "") not in processed_sample_ids
            ]

            with tqdm(total=len(image_args), desc=f"Processing XLRSBench {test_type}") as pbar:
                for result in pool.imap(process, image_args):
                    if result is not None:
                        save_json.append(result)
                        # 将结果写入总结果文件
                        f_results.write(json.dumps(result) + "\n")
                        f_results.flush()  # 确保立即写入磁盘
                        pbar.update(1)

            pool.close()
            pool.join()

            # 按原格式保存分类别结果
            with open(os.path.join(save_path, save_name), "w") as f:
                for item in save_json:
                    f.write(json.dumps(item) + "\n")

            print(f"完成类别 {test_type} 评测，结果保存至 {save_name}")

    print(f"所有评测结果已合并保存至 {results_file}")

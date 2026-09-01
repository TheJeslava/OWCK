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

# Tool descriptions and tool-call directions are deliberately omitted for the
# no-tools-prompt ablation. Learned, unsolicited tool calls are recorded but
# never executed.
instruction_prompt_system = "You are a helpful assistant."

USER_PROMPT_V2 = "\nThink first, then answer. Format strictly as: <think>...</think> <answer>...</answer> "

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
                    # A partially processed sample must remain eligible for a
                    # later retry.  Treat only completed inferences as durable
                    # checkpoints.
                    if result.get("status") == "success" and result.get("sample_id"):
                        processed_sample_ids.add(result["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    print("跳过无效的 JSON 行")
    return processed_sample_ids


def remove_ignored_tool_call(text):
    """Remove an unsolicited tool block before extracting an answer."""
    return re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.DOTALL).strip()


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

    # This ablation has no tool follow-up; record and extract the first response.
    try:
        params = {
            "model": eval_model_name,
            "messages": chat_message,
            "temperature": 0.0,
            "max_tokens": MAX_GENERATION_TOKENS,
            "stop": ["<|im_end|>\n".strip()],
        }
        response = client.chat.completions.create(**params)
        response_message = response.choices[0].message.content or ""
        print_messages.append({"role": "assistant", "content": response_message})
    except Exception as e:
        print(f"Error!!!!", e)
        status = "error"

    # 结果提取逻辑 - 修改以支持多选
    extraction_text = remove_ignored_tool_call(response_message) if start_token in response_message else response_message
    if "</answer>" in extraction_text and "<answer>" in extraction_text:
        output_text = extraction_text.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        output_text = extraction_text

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
        if (
            filename.endswith(".json")
            and filename not in {"categories.json", "processing_log.json", "selection_manifest.json"}
        ):
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

            # Rebuild the category file from the authoritative merged result.
            # On a resumed run save_json contains only newly processed samples;
            # writing it directly would erase the already completed category.
            category_results = []
            with open(results_file, "r", encoding="utf-8") as merged_results:
                for line in merged_results:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("category") == test_type:
                        category_results.append(item)

            with open(os.path.join(save_path, save_name), "w", encoding="utf-8") as f:
                for item in category_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"完成类别 {test_type} 评测，结果保存至 {save_name}")

    print(f"所有评测结果已合并保存至 {results_file}")

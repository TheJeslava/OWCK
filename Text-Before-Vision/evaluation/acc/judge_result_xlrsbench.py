import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
import json
import re
import argparse
import time
from tqdm import tqdm
from openai import OpenAI
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://localhost:8000/v1', help='API URL')
parser.add_argument('--xlrsbench_path', type=str, default=None, help='Path to the XLRS benchmark')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

# OpenAI客户端设置
openai_api_key = args.api_key
openai_api_base = args.api_url
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 获取模型名称
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models['data'][0]['id']
else:
    eval_model_name = args.eval_model_name

# 路径设置
xlrsbench_path = args.xlrsbench_path
result_root_path = args.save_path
result_root_path = os.path.join(result_root_path, args.model_name)

# 加载类别信息
try:
    with open(os.path.join(xlrsbench_path, "categories.json"), "r") as f:
        categories = json.load(f)
    category_list = list(categories.keys())
except:
    # 默认类别
    category_list = [
        'Complex reasoning/Route planning',
        'Complex reasoning/Terrain analysis',
        'Complex reasoning/Environmental impact',
        'Counting/Regional counting',
        'Counting/Counting with complex reasoning',
        'Land use classification/Overall Land use classification',
        'Land use classification/Land use classification with reasoning',
        'Object properties/Positional properties',
        'Object properties/Physical properties',
        'Object properties/Change detection',
        'Object spatial relationship/Distance relationship',
        'Object spatial relationship/Geographical relationship',
        'Object spatial relationship/Orientation relationship'
    ]

# 多选题类别
MULTI_ANSWER_CATEGORIES = ["Land use classification/Overall Land use classification"]

def is_multi_answer_question(category):
    """判断是否为多选题"""
    if category in MULTI_ANSWER_CATEGORIES or category.startswith("Land use classification/Overall Land use classification"):
        return True
    return False

def get_chat_template():
    system = "You are an AI assistant helping to evaluate model predictions for multiple-choice questions about remote sensing and satellite imagery."
    return system

def get_single_choice_ICE():
    ICE = """Example 1:
Question: What type of land use is most prevalent in this satellite image?
Options:
(A) Urban development
(B) Agricultural land
(C) Forest
(D) Water body
Predicted answer: B
Ground truth: B
Correct? Yes

Example 2:
Question: How many large buildings are visible in the image?
Options:
(A) 5-10
(B) 11-15
(C) 16-20
(D) More than 20
Predicted answer: D
Ground truth: C
Correct? No"""
    return ICE

def get_multi_choice_ICE():
    ICE = """Example 1:
Question: Which of the following features are visible in the satellite image?
Options:
(A) River
(B) Highway
(C) Forest
(D) Airport
Predicted answer: A B C
Ground truth: A C
Correct? No

Example 2:
Question: Which types of land use are present in this region?
Options:
(A) Agricultural
(B) Residential
(C) Industrial
(D) Commercial
Predicted answer: A B
Ground truth: A B
Correct? Yes"""
    return ICE

def get_prompt(predict_str, ground_truth, question, options, is_multi=False):
    # 选择合适的示例
    ICE = get_multi_choice_ICE() if is_multi else get_single_choice_ICE()
    formatted_options = "\n".join([f"({abc_map[i+1]}) {opt}" for i, opt in enumerate(options)])
    
    prompt = f"""Please evaluate if the predicted answer matches the ground truth for the following {'multi-choice' if is_multi else 'multiple-choice'} question about satellite imagery:

{ICE}

Question: {question}
Options:
{formatted_options}
Predicted answer: {predict_str}
Ground truth: {ground_truth}
Correct?"""
    return prompt

def extract_answer_letters_improved(text):
    """增强版答案提取函数，处理多种格式"""
    if not text:
        return ""
    
    # 1. 处理空格分隔的单个字母 (如 "A B C" -> "ABC")
    space_separated = re.findall(r"(?:^|\s)([A-Ea-e])(?:$|[\s,.])", text)
    if space_separated and len(space_separated) >= 1:
        return "".join(sorted(set(m.upper() for m in space_separated)))
    
    # 2. "A and B and C" 格式
    if " and " in text.lower():
        and_parts = re.findall(r"(?:^|\s|and\s+)([A-Ea-e])(?:$|\s|,|\sand)", text.lower())
        if and_parts and len(and_parts) >= 1:
            return "".join(sorted(set(m.upper() for m in and_parts)))
    
    # 3. 逗号分隔格式 "A, B, C"
    if "," in text:
        comma_parts = re.findall(r"(?:^|\s|,\s*)([A-Ea-e])(?:$|\s|,)", text)
        if comma_parts and len(comma_parts) >= 1:
            return "".join(sorted(set(m.upper() for m in comma_parts)))
    
    # 4. 括号中的字母: (A) (B)
    bracket_matches = re.findall(r"\(([A-Ea-e])\)", text)
    if bracket_matches and len(bracket_matches) >= 1:
        return "".join(sorted(set(m.upper() for m in bracket_matches)))
    
    # 5. 尝试直接提取所有A-E字母作为最后手段
    all_letters = re.findall(r"[A-Ea-e]", text)
    if all_letters and len(all_letters) >= 1:
        return "".join(sorted(set(m.upper() for m in all_letters)))
    
    return ""

def compare_answers(predicted, ground_truth, is_multi=False):
    """比较预测答案和真实答案，考虑多选情况"""
    # 标准化答案格式
    pred_set = set(predicted.upper())
    truth_set = set(ground_truth.upper())
    
    if is_multi:
        # 多选题：所有选项都必须匹配
        return pred_set == truth_set
    else:
        # 单选题：只要有一个字母匹配即可
        return len(pred_set.intersection(truth_set)) > 0

# 答案映射
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
abc_re_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

def process_result(result):
    """处理单个评测结果，使用三层判断逻辑"""
    try:
        # 提取基本信息
        pred_ans = result.get('pred_ans', '')
        extracted_answer = result.get('extracted_answer', '')
        answer = result.get('answer', '')
        question = result.get('question', '')
        options = result.get('options', [])
        category = result.get('category', 'unknown')
        
        # 判断是否为多选题
        is_multi = result.get('is_multi', False) or is_multi_answer_question(category)
        
        # 第一层: 使用eval阶段提取的答案进行判断
        is_correct = False
        method_used = "rule"
        predicted_answer = extracted_answer
        
        if extracted_answer:
            is_correct = compare_answers(extracted_answer, answer, is_multi)
        
        # 第二层: 如果第一层不正确，尝试从原始回答重新提取
        if not is_correct:
            # 从原始答案中提取
            re_extracted = extract_answer_letters_improved(pred_ans)
            
            # 只有当新提取结果与原提取结果不同时才尝试比较
            if re_extracted and re_extracted != extracted_answer:
                predicted_answer = re_extracted
                is_correct = compare_answers(re_extracted, answer, is_multi)
        
        # 第三层: 只有在特定情况下才使用LLM判断
        need_llm = False
        
        # 对于多选题，如果仍然不正确且有特殊格式，考虑使用LLM
        if not is_correct and is_multi:
            # 检查是否是格式问题
            # 1. 原始答案包含"and"、逗号等多选标志
            # 2. 原始答案与标准答案长度接近(可能是顺序或格式问题)
            has_multi_format = ("and" in pred_ans.lower()) or ("," in pred_ans) or (len(pred_ans.strip().split()) > 2)
            length_similar = predicted_answer and abs(len(predicted_answer) - len(answer)) <= 1
            
            if has_multi_format or length_similar:
                need_llm = True
        
        # 如果不需要LLM或预测答案为空，返回规则判断结果
        if not need_llm or not predicted_answer:
            return {
                'sample_id': result.get('sample_id', 'unknown'),
                'category': category,
                'is_multi_image': result.get('is_multi_image', False),
                'is_multi_choice': is_multi,
                'predicted': predicted_answer,
                'original_extracted': extracted_answer,
                'original_pred': pred_ans,
                'answer': answer,
                'correct': is_correct,
                'score': 1 if is_correct else 0,
                'method': 'rule'
            }
        
        # 使用LLM进行最终判断
        prompt = get_prompt(pred_ans, answer, question, options, is_multi)
        system = get_chat_template()
        
        try:
            # 调用API
            chat_response = client.chat.completions.create(
                model=eval_model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10,
            )
            
            # 解析结果
            llm_result = chat_response.choices[0].message.content.strip().lower()
            is_correct = "yes" in llm_result and "no" not in llm_result
            method_used = "llm"
        except Exception as e:
            print(f"LLM调用失败: {e}, 使用规则判断结果")
            # LLM调用失败，回退到规则判断结果
        
        return {
            'sample_id': result.get('sample_id', 'unknown'),
            'category': category,
            'is_multi_image': result.get('is_multi_image', False),
            'is_multi_choice': is_multi,
            'predicted': predicted_answer,
            'original_extracted': extracted_answer,
            'original_pred': pred_ans,
            'answer': answer,
            'correct': is_correct,
            'score': 1 if is_correct else 0,
            'method': method_used
        }
        
    except Exception as e:
        print(f"评估结果时出错: {e}")
        return {
            'sample_id': result.get('sample_id', 'unknown'),
            'error': str(e),
            'score': 0,
            'method': 'error'
        }

def process_worker(result):
    return process_result(result)

def sanitize_filename(category):
    """将类别名转换为有效的文件名"""
    return category.replace('/', '_').replace(' ', '_')

if __name__ == '__main__':
    # 加载评测结果
    results_file = os.path.join(result_root_path, "xlrsbench_results.jsonl")
    if not os.path.exists(results_file):
        print(f"找不到结果文件: {results_file}")
        exit(1)
    
    # 读取结果
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    
    print(f"加载了 {len(results)} 个评测结果")
    
    # 多进程评估
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        evaluations = list(tqdm(pool.imap(process_worker, results), total=len(results), desc="评估答案"))
    
    # 统计结果
    total_correct = sum(1 for e in evaluations if e.get('correct', False))
    overall_accuracy = total_correct / len(evaluations) if evaluations else 0
    
    # 按类别分组评估结果
    category_results = {}
    category_stats = {}
    
    for eval_result in evaluations:
        category = eval_result.get('category', 'unknown')
        
        # 分组结果
        if category not in category_results:
            category_results[category] = []
        category_results[category].append(eval_result)
        
        # 统计数据
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        category_stats[category]['total'] += 1
        if eval_result.get('correct', False):
            category_stats[category]['correct'] += 1
    
    # 计算每个类别的准确率
    for key, data in category_stats.items():
        data['accuracy'] = data['correct'] / data['total'] if data['total'] > 0 else 0
    
    # 统计多选题和多图样本的性能
    multi_choice_stats = {'correct': 0, 'total': 0}
    multi_image_stats = {'correct': 0, 'total': 0}
    method_stats = {'rule': 0, 'llm': 0, 'error': 0}
    
    for eval_result in evaluations:
        # 统计多选和多图
        if eval_result.get('is_multi_choice', False):
            multi_choice_stats['total'] += 1
            if eval_result.get('correct', False):
                multi_choice_stats['correct'] += 1
        
        if eval_result.get('is_multi_image', False):
            multi_image_stats['total'] += 1
            if eval_result.get('correct', False):
                multi_image_stats['correct'] += 1
        
        # 统计使用的方法
        method = eval_result.get('method', 'unknown')
        if method in method_stats:
            method_stats[method] += 1
    
    # 计算多选和多图准确率
    multi_choice_stats['accuracy'] = multi_choice_stats['correct'] / multi_choice_stats['total'] if multi_choice_stats['total'] > 0 else 0
    multi_image_stats['accuracy'] = multi_image_stats['correct'] / multi_image_stats['total'] if multi_image_stats['total'] > 0 else 0
    
    # 保存评估结果
    eval_results = {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_samples': len(evaluations),
        'categories': category_stats,
        'multi_choice_stats': multi_choice_stats,
        'multi_image_stats': multi_image_stats,
        'method_stats': method_stats
    }
    
    # 创建结果目录
    evaluation_dir = os.path.join(result_root_path, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # 保存总体详细结果
    total_details_file = os.path.join(evaluation_dir, "xlrsbench_all_evaluation_details.jsonl")
    with open(total_details_file, 'w', encoding='utf-8') as f:
        for eval_result in evaluations:
            f.write(json.dumps(eval_result, ensure_ascii=False) + '\n')
    
    # 按类别保存详细结果
    for category, results in category_results.items():
        category_file = os.path.join(evaluation_dir, f"xlrsbench_{sanitize_filename(category)}_evaluation.jsonl")
        with open(category_file, 'w', encoding='utf-8') as f:
            for eval_result in results:
                f.write(json.dumps(eval_result, ensure_ascii=False) + '\n')
    
    # 保存汇总结果
    eval_summary_file = os.path.join(evaluation_dir, "xlrsbench_evaluation_summary.json")
    with open(eval_summary_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print(f"\n评估完成, 总体准确率: {overall_accuracy:.4f} ({total_correct}/{len(evaluations)})")
    print(f"结果保存至: {evaluation_dir}")
    
    # 打印方法使用统计
    print(f"\n评估方法使用情况:")
    print(f"规则匹配: {method_stats['rule']} ({method_stats['rule']/len(evaluations):.2%})")
    print(f"LLM判断: {method_stats['llm']} ({method_stats['llm']/len(evaluations):.2%})")
    if method_stats['error'] > 0:
        print(f"错误: {method_stats['error']} ({method_stats['error']/len(evaluations):.2%})")
    
    # 打印每个类别的结果
    print("\n按类别准确率:")
    for category, stats in sorted(category_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{category}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    # 打印多选题和多图样本的性能
    if multi_choice_stats['total'] > 0:
        print(f"\n多选题准确率: {multi_choice_stats['accuracy']:.4f} ({multi_choice_stats['correct']}/{multi_choice_stats['total']})")
    
    if multi_image_stats['total'] > 0:
        print(f"多图样本准确率: {multi_image_stats['accuracy']:.4f} ({multi_image_stats['correct']}/{multi_image_stats['total']})")

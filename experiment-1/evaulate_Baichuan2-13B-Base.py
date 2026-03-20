import argparse
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import json
import time

# 支持最多 8 个选项
choices_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

def parse_choices(s):
    """解析 choices 字符串，返回列表"""
    s = s.strip().strip('[]')
    pattern = r"'(.*?)'"
    items = re.findall(pattern, s)
    return items

def format_example(df, idx, include_answer=True):
    row = df.iloc[idx]
    question = row['question']
    choice_list = parse_choices(row['choices'])
    answer_idx = int(row['answer'])
    prompt = f"问题：{question}\n"
    for i, choice in enumerate(choice_list):
        label = choices_labels[i] if i < len(choices_labels) else str(i)
        prompt += f"{label}. {choice}  "
    prompt += "\n答案："
    if include_answer:
        if answer_idx >= len(choice_list):
            answer_idx = len(choice_list) - 1
        prompt += f" {choices_labels[answer_idx]}\n\n"
    else:
        prompt += "\n"
    return prompt

def extract_answer(response, num_choices):
    """提取模型输出的答案"""
    patterns = [
        r"答案：\s*([A-Z0-9]+)",
        r"答案\s*([A-Z0-9]+)",
        r"选择\s*([A-Z0-9]+)",
        r"\b([A-Z0-9])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            pred = match.group(1)
            if pred in choices_labels[:num_choices]:
                return choices_labels.index(pred)
            try:
                idx = int(pred)
                if idx < num_choices:
                    return idx
            except:
                continue
    return num_choices - 1  # 默认最后一个选项

def eval_subject(model, tokenizer, subject, test_df, dev_df, args):
    correct = 0
    total = len(test_df)
    results = []

    for i in tqdm(range(total), desc=f"评估 {subject}"):
        # few-shot
        few_shot_prompt = ""
        for j in range(min(args.ntrain, len(dev_df))):
            few_shot_prompt += format_example(dev_df, j, include_answer=True)

        # 测试题
        test_prompt = format_example(test_df, i, include_answer=False)
        prompt = few_shot_prompt + test_prompt

        # 编码
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        # 推理
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.5,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        choice_list = parse_choices(test_df.iloc[i]['choices'])
        num_choices = len(choice_list)
        pred_idx = extract_answer(response, num_choices)

        # 获取正确答案
        gold_idx = int(test_df.iloc[i]['answer'])
        if gold_idx >= num_choices:
            gold_idx = num_choices - 1
        if pred_idx >= num_choices:
            pred_idx = num_choices - 1

        # 打印题目和结果
        print("=" * 30)
        print(f"题目 {i + 1}: {test_df.iloc[i]['question']}")
        for idx, choice in enumerate(choice_list):
            label = choices_labels[idx] if idx < len(choices_labels) else str(idx)
            print(f"{label}. {choice}")
        print(f"模型输出: {response}")
        print(f"预测答案: {choices_labels[pred_idx]} (索引 {pred_idx}) | 正确答案: {choices_labels[gold_idx]} (索引 {gold_idx})")
        print("=" * 30)

        if pred_idx == gold_idx:
            correct += 1
        results.append((gold_idx, pred_idx))

    acc = correct / total if total > 0 else 0.0
    print(f"✅ {subject:<30} acc = {acc:.3f}")
    return results, acc

def main(args):
    start_time = time.time()

    print("🚀 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    subjects = sorted([d.replace("_dev.csv", "").replace("_test.csv", "")
                       for d in os.listdir(os.path.join(args.data_dir, "dev"))
                       if d.endswith("_dev.csv")])

    if args.subjects != "all":
        subjects = [s for s in args.subjects.split(",") if s in subjects]

    os.makedirs(args.save_dir, exist_ok=True)
    result_json_path = os.path.join(args.save_dir, "result.json")

    # 加载已有 result.json 或初始化
    if os.path.exists(result_json_path):
        with open(result_json_path, "r") as f:
            result_data = json.load(f)
    else:
        result_data = {"subjects": {}, "total_correct": 0, "total_count": 0, "overall": 0.0}

    for subject in subjects:
        print(f"\n🔄 正在评估: {subject}")
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", f"{subject}_dev.csv"))
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", f"{subject}_test.csv"))

        cors, acc = eval_subject(model, tokenizer, subject, test_df, dev_df, args)

        # 保存单学科 CSV
        result_df = pd.DataFrame(cors, columns=["gold", "pred"])
        result_df.to_csv(os.path.join(args.save_dir, f"{subject}.csv"), index=False)

        num_questions = len(test_df)
        correct_answers = acc * num_questions

        # 如果该学科已存在，先减去旧数据
        if subject in result_data["subjects"]:
            old_acc = result_data["subjects"][subject]
            old_count = num_questions
            result_data["total_correct"] -= old_acc * old_count
            result_data["total_count"] -= old_count

        # 更新 JSON 数据
        result_data["subjects"][subject] = acc
        result_data["total_correct"] += correct_answers
        result_data["total_count"] += num_questions
        result_data["overall"] = result_data["total_correct"] / result_data["total_count"]

        # 写入 JSON
        with open(result_json_path, "w") as f:
            json.dump(result_data, f, indent=4)

        print(f"✅ {subject} 评估完成，当前总平均准确率: {result_data['overall']:.3f}")

    print(f"\n🎯 所有学科完成，总平均准确率: {result_data['overall']:.3f}")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5, help="few-shot 样本数")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据集根目录")
    parser.add_argument("--model_dir", type=str, required=True, help="模型路径")
    parser.add_argument("--save_dir", type=str, required=True, help="结果保存目录")
    parser.add_argument("--subjects", type=str, default="all", help="以逗号分隔的subject名称或'all'")
    args = parser.parse_args()
    main(args)

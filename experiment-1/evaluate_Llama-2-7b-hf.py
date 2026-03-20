import argparse
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import json
import time

choices = ["A", "B", "C", "D"]


def parse_choices(s):
    """解析 choices 字符串，提取单引号内字符串"""
    s = s.strip().strip('[]')
    pattern = r"'(.*?)'"
    items = re.findall(pattern, s)
    return items


def format_example(df, idx, include_answer=True):
    row = df.iloc[idx]
    question = row['question']
    choice_list = parse_choices(row['choices'])
    if len(choice_list) < 4:
        choice_list += [''] * (4 - len(choice_list))
    answer_idx = int(row['answer'])
    prompt = f"问题：{question}\n选项：A. {choice_list[0]} B. {choice_list[1]} C. {choice_list[2]} D. {choice_list[3]}\n答案："
    if include_answer:
        prompt += f" {choices[answer_idx]}\n\n"
    else:
        prompt += "\n"
    return prompt


def extract_answer(response):
    """提取模型输出中的答案"""
    patterns = [
        r"答案：\s*([A-D])",
        r"答案：\s*([0-3])",
        r"答案\s*([A-D])",
        r"答案\s*([0-3])",
        r"\b([A-D])\b",
        r"\b([0-3])\b",
        r"选择\s*([A-D])",
        r"选择\s*([0-3])"
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            pred = match.group(1)
            if pred in choices:
                return choices.index(pred)
            try:
                return int(pred)
            except:
                continue

    first_char = response.strip()[0] if response.strip() else None
    if first_char in choices:
        return choices.index(first_char)
    elif first_char in ['0', '1', '2', '3']:
        return int(first_char)

    print(f"❌ 无法提取答案，模型输出: {response}")
    return 0


def eval_subject(model, tokenizer, subject, test_df, dev_df, args):
    correct = 0
    total = len(test_df)
    results = []

    for i in tqdm(range(total), desc=f"评估 {subject}"):
        k = args.ntrain
        few_shot_prompt = ""

        for j in range(k):
            few_shot_prompt += format_example(dev_df, j, include_answer=True)

        test_prompt = format_example(test_df, i, include_answer=False)
        prompt = few_shot_prompt + test_prompt

        # LLaMA 推理
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.5,
                top_p=0.95,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        print(f"Prompt: {prompt}")
        print(f"模型输出: {response}")

        pred = extract_answer(response)
        gold_idx = int(test_df.iloc[i]['answer'])
        results.append((gold_idx, pred))

        if pred == gold_idx:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"✅ {subject:<30} acc = {acc:.3f}")
    return results, acc


def main(args):
    start_time = time.time()

    print("🚀 加载 Llama-2-7b-hf 模型中...")
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

    total_correct = 0
    total_count = 0
    subject_accuracies = {}
    os.makedirs(args.save_dir, exist_ok=True)

    result_json_path = os.path.join(args.save_dir, "result.json")
    if os.path.exists(result_json_path):
        with open(result_json_path, "r") as f:
            result_data = json.load(f)
        subject_accuracies = result_data.get("subjects", {})
        total_correct = result_data.get("total_correct", 0)
        total_count = result_data.get("total_count", 0)
    else:
        result_data = {"subjects": {}, "total_correct": 0, "total_count": 0}

    for subject in subjects:
        print(f"\n🔄 正在评估: {subject}")
        dev_path = os.path.join(args.data_dir, "dev", f"{subject}_dev.csv")
        test_path = os.path.join(args.data_dir, "test", f"{subject}_test.csv")

        dev_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)

        cors, acc = eval_subject(model, tokenizer, subject, test_df, dev_df, args)

        total_correct += acc * len(test_df)
        total_count += len(test_df)
        subject_accuracies[subject] = acc

        result_df = pd.DataFrame(cors, columns=["gold", "pred"])
        result_df.to_csv(os.path.join(args.save_dir, f"{subject}.csv"), index=False)

        result_data["subjects"] = subject_accuracies
        result_data["total_correct"] = total_correct
        result_data["total_count"] = total_count
        result_data["overall"] = total_correct / total_count if total_count > 0 else 0.0

        with open(result_json_path, "w") as f:
            json.dump(result_data, f, indent=4)

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    print(f"\n🎯 总平均准确率: {overall_acc:.3f}")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5, help="few-shot 样本数")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据集根目录")
    parser.add_argument("--model_dir", type=str, default="/data/codehouse/model_path/meta-llama/Llama-2-7b-hf", help="模型路径")
    parser.add_argument("--save_dir", type=str, required=True, help="结果保存目录")
    parser.add_argument("--subjects", type=str, default="all", help="以逗号分隔的subject名称或'all'")
    args = parser.parse_args()

    main(args)

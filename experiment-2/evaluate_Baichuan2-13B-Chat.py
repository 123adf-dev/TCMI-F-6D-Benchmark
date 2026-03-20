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
    """
    解析 choices 字符串，适配 ['opt1' 'opt2' 'opt3' 'opt4'] 格式，
    提取单引号内字符串，返回列表
    """
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
    """
    改进的答案提取函数，处理多种可能的输出格式
    """
    patterns = [
        r"答案：\s*([A-D])",  # 匹配 "答案：A"
        r"答案：\s*([0-3])",  # 匹配 "答案：0"
        r"答案\s*([A-D])",  # 匹配 "答案A"
        r"答案\s*([0-3])",  # 匹配 "答案0"
        r"\b([A-D])\b",  # 匹配单独的"A"
        r"\b([0-3])\b",  # 匹配单独的"0"
        r"选择\s*([A-D])",  # 匹配 "选择A"
        r"选择\s*([0-3])"  # 匹配 "选择0"
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

    # 如果没有匹配到任何模式，尝试提取第一个字符
    first_char = response.strip()[0] if response.strip() else None
    if first_char in choices:
        return choices.index(first_char)
    elif first_char in ['0', '1', '2', '3']:
        return int(first_char)

    # 如果仍然无法提取答案，返回默认值
    print(f"无法提取答案，模型输出: {response}")
    return 0  # 默认值


def eval_subject(model, tokenizer, subject, test_df, dev_df, args):
    correct = 0
    total = len(test_df)
    results = []

    for i in tqdm(range(total), desc=f"评估 {subject}"):
        k = args.ntrain
        few_shot_prompt = ""

        # 生成 few-shot 示例
        for j in range(k):
            few_shot_prompt += format_example(dev_df, j, include_answer=True)

        # 构建测试问题的 prompt
        test_prompt = format_example(test_df, i, include_answer=False)
        prompt = few_shot_prompt + test_prompt

        # 打印 prompt 用于调试
        print(f"Few-shot Prompt: {few_shot_prompt}")
        print(f"Test Prompt: {test_prompt}")

        # 将输入 token 化并送入模型
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        # 使用 no_grad 进行无梯度推理
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,  # 增加生成的token数量
                temperature=0.5,  # 调整温度
                do_sample=True,  # 开启随机采样
                top_p=0.95,  # 调整top_p
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )

        # 解码输出并去除特殊符号
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # 只保留prompt之后的部分
        response = response[len(prompt):].strip()
        print(f"Prompt: {prompt}")
        print(f"模型输出: {response}")

        # 提取答案
        pred = extract_answer(response)
        if pred is None:
            # 如果无法提取答案，尝试获取第一个token
            try:
                pred_token = output[0][-1].item()
                if pred_token in [tokenizer.encode(c)[0] for c in choices]:
                    pred = choices.index(tokenizer.decode(pred_token))
                else:
                    pred = 0  # 默认值
            except:
                pred = 0  # 默认值
            print(f"无法提取答案，使用默认答案: {pred}")

        # 获取正确答案
        gold_idx = int(test_df.iloc[i]['answer'])
        results.append((gold_idx, pred))

        # 计算准确率
        if pred == gold_idx:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"✅ {subject:<30} acc = {acc:.3f}")

    return results, acc


def main(args):
    start_time = time.time()

    print("🚀 加载模型中...")
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

    # 如果结果文件已经存在，加载现有结果
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

        # 更新结果文件
        result_data["subjects"] = subject_accuracies
        result_data["total_correct"] = total_correct
        result_data["total_count"] = total_count
        result_data["overall"] = total_correct / total_count if total_count > 0 else 0.0

        with open(result_json_path, "w") as f:
            json.dump(result_data, f, indent=4)

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    print(f"\n🎯 总平均准确率: {overall_acc:.3f}")

    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5, help="few-shot 样本数")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据集根目录")
    parser.add_argument("--model_dir", type=str, required=True, help="模型路径")
    parser.add_argument("--save_dir", type=str, required=True, help="结果保存目录")
    parser.add_argument("--subjects", type=str, default="all", help="以逗号分隔的subject名称或'all'")
    args = parser.parse_args()

    main(args)
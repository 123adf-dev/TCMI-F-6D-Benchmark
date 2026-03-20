import argparse
import json
import os
import re
import time
from typing import List, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

choices = ["A", "B", "C", "D"]

client = OpenAI(
    api_key="",   # 你的真实 key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def parse_choices(s: str) -> List[str]:
    return re.findall(r"'(.*?)'", s.strip().strip("[]"))

def format_example(df, idx: int, include_answer=True) -> str:
    row = df.iloc[idx]
    q = row["question"]
    opts = parse_choices(row["choices"]) + [""] * 4
    ans_idx = int(row["answer"])
    prompt = f"问题：{q}\n选项：A. {opts[0]} B. {opts[1]} C. {opts[2]} D. {opts[3]}\n答案："
    if include_answer:
        prompt += f" {choices[ans_idx]}\n\n"
    else:
        prompt += "\n"
    return prompt

def extract_answer(text: str) -> int:
    for p in [
        r"答案：\s*([A-D])", r"答案：\s*([0-3])", r"\b([A-D])\b", r"\b([0-3])\b"
    ]:
        m = re.search(p, text)
        if m:
            g = m.group(1)
            return choices.index(g) if g in choices else int(g)
    first = text.strip()[0] if text.strip() else ""
    if first in choices:
        return choices.index(first)
    if first in "0123":
        return int(first)
    print(f"无法提取答案，原文：{text}")
    return 0

def call_deepseek(prompt: str) -> str:
    """使用百炼 OpenAI-Compatible 接口"""
    resp = client.chat.completions.create(
        model="deepseek-v3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=100,
        stream=False,
    )
    return resp.choices[0].message.content.strip()

def eval_subject(subject: str, test_df, dev_df, ntrain: int = 5) -> Tuple[List[Tuple[int, int]], float]:
    correct = 0
    records = []
    for i in tqdm(range(len(test_df)), desc=f"评估 {subject}"):
        few_shot = "".join(format_example(dev_df, j, include_answer=True) for j in range(ntrain))
        prompt = few_shot + format_example(test_df, i, include_answer=False)
        ans_text = call_deepseek(prompt)
        pred = extract_answer(ans_text)
        gold = int(test_df.iloc[i]["answer"])
        records.append((gold, pred))
        if pred == gold:
            correct += 1
    acc = correct / len(test_df) if len(test_df) else 0.0
    print(f"✅ {subject:<30} acc = {acc:.3f}")
    return records, acc


def main(args):
    global client
    client = OpenAI(
        api_key=args.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    os.makedirs(args.save_dir, exist_ok=True)
    dev_dir = os.path.join(args.data_dir, "dev")
    subjects = sorted([f.replace("_dev.csv", "") for f in os.listdir(dev_dir) if f.endswith("_dev.csv")])
    if args.subjects != "all":
        subjects = [s for s in args.subjects.split(",") if s in subjects]

    total_correct = 0
    total_count = 0
    subject_acc = {}

    result_json = os.path.join(args.save_dir, "result.json")
    data = {"subjects": {}, "total_correct": 0, "total_count": 0}
    if os.path.exists(result_json):
        with open(result_json, encoding="utf-8") as f:
            data = json.load(f)
        subject_acc = data.get("subjects", {})
        total_correct = data.get("total_correct", 0)
        total_count = data.get("total_count", 0)

    for sub in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", f"{sub}_dev.csv"))
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", f"{sub}_test.csv"))
        cors, acc = eval_subject(sub, test_df, dev_df, args.ntrain)

        subject_acc[sub] = acc
        total_correct += acc * len(test_df)
        total_count += len(test_df)

        pd.DataFrame(cors, columns=["gold", "pred"]).to_csv(
            os.path.join(args.save_dir, f"{sub}.csv"), index=False
        )

        data.update(
            {
                "subjects": subject_acc,
                "total_correct": total_correct,
                "total_count": total_count,
                "overall": total_correct / total_count if total_count else 0.0,
            }
        )
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n🎯 总平均准确率: {total_correct / total_count:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain",   type=int, default=5, help="few-shot 样本数")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据集根目录")
    parser.add_argument("--save_dir", type=str, required=True, help="结果保存目录")
    parser.add_argument("--subjects", type=str, default="all", help="以逗号分隔的subject名称或'all'")
    parser.add_argument("--api_key",  type=str, required=True, help="阿里云百炼 DashScope API-Key")
    # 兼容旧命令
    parser.add_argument("--model_dir", type=str, default="", help="已弃用，可留空")
    args = parser.parse_args()
    main(args)
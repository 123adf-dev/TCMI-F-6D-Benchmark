# generate_examples.py
import os
import pandas as pd
import random
import json
import re

def parse_choices(s):
    """解析 choices 列表字段"""
    if isinstance(s, list):
        return s
    s = str(s).strip().strip('[]')
    pattern = r"'(.*?)'|\"(.*?)\""
    items = re.findall(pattern, s)
    items = [x[0] or x[1] for x in items]
    return items

def generate_examples(data_dir, subjects, save_path, n=5):
    examples = {}

    for subject in subjects:
        dev_path = os.path.join(data_dir, "dev", f"{subject}_dev.csv")
        if not os.path.exists(dev_path):
            print(f"⚠️ 找不到 {dev_path}")
            continue

        df = pd.read_csv(dev_path)

        # 随机抽取 n 个样本
        sampled = df.sample(n=min(n, len(df)), random_state=42)

        examples[subject] = []
        for _, row in sampled.iterrows():
            q = row["question"]
            chs = parse_choices(row["choices"])
            a = int(row["answer"])
            examples[subject].append({
                "question": q,
                "choices": chs,
                "answer": a
            })

    # 保存成 JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)

    print(f"✅ 已生成 {save_path}")


if __name__ == "__main__":
    data_dir = r"G:\file\yjs\Education AI\data"
    subjects = [
        "clinical_knowledge",
        "college_medicine",
        "medical_genetics",
        "nutrition",
        "machine_learning",
        "college_computer_science"
    ]
    save_path = r"G:\file\yjs\Education AI\examples\examples.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    generate_examples(data_dir, subjects, save_path, n=5)

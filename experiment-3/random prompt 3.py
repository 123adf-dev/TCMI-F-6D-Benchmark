import os
import pandas as pd
import json
import re

def parse_choices(s):
    """把 choices 字符串解析成列表"""
    if isinstance(s, list):
        return s
    s = str(s).strip().strip("[]")
    items = re.findall(r"'(.*?)'", s)
    if items:
        return items
    items = re.findall(r'"(.*?)"', s)
    if items:
        return items
    return []

def generate_multiple_example_files(merged_dir, subjects, save_dir, n=5, num_files=10):
    os.makedirs(save_dir, exist_ok=True)

    for ex_num in range(1, num_files + 1):
        examples = {}

        for subject in subjects:
            file_path = os.path.join(merged_dir, f"{subject}_dev_val_merged.csv")

            if not os.path.exists(file_path):
                print(f"⚠️ 找不到文件: {file_path}")
                continue

            df = pd.read_csv(file_path)

            if len(df) == 0:
                print(f"⚠️ {subject} 文件为空，跳过")
                continue

            sample_size = min(n, len(df))
            sampled_df = df.sample(n=sample_size).reset_index(drop=True)

            examples[subject] = []
            for _, row in sampled_df.iterrows():
                examples[subject].append({
                    "question": row["question"],
                    "choices": parse_choices(row["choices"]),
                    "answer": int(row["answer"])
                })

        save_path = os.path.join(save_dir, f"ex{ex_num}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=4, ensure_ascii=False)

        print(f"✅ 已生成: {save_path}")

if __name__ == "__main__":
    merged_dir = r"G:\file\yjs\Education AI\data3"
    save_dir = r"G:\file\yjs\Education AI\examples"

    subjects = [
        "clinical_knowledge",
        "college_medicine",
        "medical_genetics",
        "nutrition",
        "machine_learning",
        "college_computer_science"
    ]

    generate_multiple_example_files(
        merged_dir=merged_dir,
        subjects=subjects,
        save_dir=save_dir,
        n=5,
        num_files=10
    )
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


def load_examples(example_path):
    with open(example_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_example_from_csv(df, idx, include_answer=True):
    row = df.iloc[idx]
    question = row["question"]
    choice_list = parse_choices(row["choices"])
    if len(choice_list) < 4:
        choice_list += [""] * (4 - len(choice_list))
    answer_idx = int(row["answer"])

    prompt = (
        "下面是一个单项选择题，请只回答一个选项字母（A/B/C/D）。\n"
        f"问题：{question}\n"
        f"选项：\n"
        f"A. {choice_list[0]}\n"
        f"B. {choice_list[1]}\n"
        f"C. {choice_list[2]}\n"
        f"D. {choice_list[3]}\n"
        f"答案："
    )

    if include_answer:
        prompt += f"{choices[answer_idx]}\n\n"
    return prompt


def format_example_from_json(example, include_answer=True):
    question = example["question"]
    choice_list = example["choices"]

    if len(choice_list) < 4:
        choice_list += [""] * (4 - len(choice_list))

    answer_idx = int(example["answer"])

    prompt = (
        "下面是一个单项选择题，请只回答一个选项字母（A/B/C/D）。\n"
        f"问题：{question}\n"
        f"选项：\n"
        f"A. {choice_list[0]}\n"
        f"B. {choice_list[1]}\n"
        f"C. {choice_list[2]}\n"
        f"D. {choice_list[3]}\n"
        f"答案："
    )

    if include_answer:
        prompt += f"{choices[answer_idx]}\n\n"
    return prompt


def get_choice_token_ids(tokenizer):
    token_ids = {}
    for ch in choices:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) == 1:
            token_ids[ch] = ids[0]
        else:
            ids = tokenizer.encode(" " + ch, add_special_tokens=False)
            if len(ids) == 1:
                token_ids[ch] = ids[0]
            else:
                raise ValueError(f"无法为选项 {ch} 找到单token id，请检查 tokenizer。")
    return token_ids


def predict_answer_by_logits(model, tokenizer, prompt, choice_token_ids, max_input_length):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_logits = outputs.logits[0, -1, :]

    scores = []
    for ch in choices:
        token_id = choice_token_ids[ch]
        scores.append(last_logits[token_id].item())

    pred = int(torch.tensor(scores).argmax().item())
    return pred, scores


def eval_subject(model, tokenizer, subject, test_df, example_data, args, choice_token_ids):
    correct = 0
    total = len(test_df)
    results = []

    if subject not in example_data:
        raise ValueError(f"示例文件中缺少学科: {subject}")

    subject_examples = example_data[subject]
    k = min(args.ntrain, len(subject_examples))

    for i in tqdm(range(total), desc=f"评估 {subject}"):
        few_shot_prompt = ""

        for j in range(k):
            few_shot_prompt += format_example_from_json(subject_examples[j], include_answer=True)

        test_prompt = format_example_from_csv(test_df, i, include_answer=False)
        prompt = few_shot_prompt + test_prompt

        try:
            pred, scores = predict_answer_by_logits(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                choice_token_ids=choice_token_ids,
                max_input_length=args.max_input_length
            )
        except Exception as e:
            print(f"\n❌ logits预测失败 | subject={subject} | question_idx={i}")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            raise

        gold_idx = int(test_df.iloc[i]["answer"])

        results.append({
            "question_idx": i,
            "gold": gold_idx,
            "pred": pred,
            "correct": int(pred == gold_idx),
            "score_A": scores[0],
            "score_B": scores[1],
            "score_C": scores[2],
            "score_D": scores[3]
        })

        if pred == gold_idx:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"✅ {subject:<30} acc = {acc:.3f}")
    return results, acc


def main(args):
    start_time = time.time()

    print("🚀 加载模型中...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    choice_token_ids = get_choice_token_ids(tokenizer)
    print("选项token ids:", choice_token_ids)

    print(f"📘 加载示例文件: {args.example_file}")
    example_data = load_examples(args.example_file)

    test_dir = os.path.join(args.data_dir, "test")
    subjects = sorted([
        f.replace("_test.csv", "")
        for f in os.listdir(test_dir)
        if f.endswith("_test.csv")
    ])

    if args.subjects != "all":
        requested_subjects = [s.strip() for s in args.subjects.split(",")]
        subjects = [s for s in requested_subjects if s in subjects]

    os.makedirs(args.save_dir, exist_ok=True)

    result_json_path = os.path.join(args.save_dir, "result.json")
    subject_accuracies = {}
    total_correct = 0
    total_count = 0

    for subject in subjects:
        print(f"\n🔄 正在评估: {subject}")
        test_path = os.path.join(args.data_dir, "test", f"{subject}_test.csv")
        test_df = pd.read_csv(test_path)

        cors, acc = eval_subject(
            model=model,
            tokenizer=tokenizer,
            subject=subject,
            test_df=test_df,
            example_data=example_data,
            args=args,
            choice_token_ids=choice_token_ids
        )

        subject_correct = sum(item["correct"] for item in cors)
        total_correct += subject_correct
        total_count += len(test_df)
        subject_accuracies[subject] = acc

        result_df = pd.DataFrame(cors)
        result_df.to_csv(
            os.path.join(args.save_dir, f"{subject}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        result_data = {
            "model_dir": args.model_dir,
            "example_file": args.example_file,
            "subjects": subject_accuracies,
            "total_correct": total_correct,
            "total_count": total_count,
            "overall": total_correct / total_count if total_count > 0 else 0.0
        }

        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    print(f"\n🎯 总平均准确率: {overall_acc:.3f}")

    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ntrain", type=int, default=5, help="few-shot 样本数")
    parser.add_argument("--data_dir", type=str, default="/home/tengzhaohang/EA/data", help="数据集根目录")
    parser.add_argument("--example_file", type=str, required=True, help="few-shot 示例文件，例如 /home/tengzhaohang/EA/examples/ex1.json")
    parser.add_argument("--model_dir", type=str, required=True, help="模型路径")
    parser.add_argument("--save_dir", type=str, required=True, help="结果保存目录")
    parser.add_argument("--subjects", type=str, default="all", help="逗号分隔的学科名，或 all")
    parser.add_argument("--max_input_length", type=int, default=1024, help="输入最大长度")

    args = parser.parse_args()
    main(args)
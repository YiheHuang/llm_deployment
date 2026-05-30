#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量测试 2-3 个大模型，并将结果保存为 Markdown/JSON。

用法示例：
python run_eval.py \
  --models /mnt/data/Qwen-7B-Chat /mnt/data/chatglm3-6b /mnt/data/Baichuan2-7B-Base \
  --max-new-tokens 200
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_QUESTIONS = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上。",
    "他知道我知道你知道他不知道吗？这句话里，到底谁不知道？",
    "明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思，小意思。领导：你这人真有意思。小明：其实也没有别的意思。领导：那我就不好意思了。小明：是我不好意思。请问：以上“意思”分别是什么意思？",
    "请用三句话解释什么是大语言模型，并举一个实际应用场景。",
]


def parse_args():
    parser = argparse.ArgumentParser(description="大模型横向对比测试脚本")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="模型本地目录列表，例如 /mnt/data/Qwen-7B-Chat /mnt/data/chatglm3-6b",
    )
    parser.add_argument(
        "--questions-file",
        default="",
        help="可选：题目文件（UTF-8，每行一题）。不传则使用内置题目。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200, help="每题最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top-p 采样")
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="输出目录（会写入 results.md 和 results.json）",
    )
    return parser.parse_args()


def load_questions(path: str):
    if not path:
        return DEFAULT_QUESTIONS
    if not os.path.exists(path):
        raise FileNotFoundError(f"questions file not found: {path}")
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q:
                questions.append(q)
    if not questions:
        raise ValueError("questions file is empty")
    return questions


def generate_answer(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][input_ids.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    questions = load_questions(args.questions_file)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "run_id": run_id,
        "time": now_str(),
        "models": args.models,
        "params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "questions": questions,
        "results": {},
    }

    print(f"[{now_str()}] 测试开始，模型数量: {len(args.models)}，题目数量: {len(questions)}")

    for model_path in args.models:
        model_name = os.path.basename(model_path.rstrip("/")) or model_path
        print(f"\n[{now_str()}] 正在加载模型: {model_name} ({model_path})")
        if not os.path.exists(model_path):
            print(f"[WARN] 模型路径不存在，跳过: {model_path}")
            all_results["results"][model_name] = {"error": f"path not found: {model_path}", "answers": []}
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
            ).eval()
        except Exception as e:
            print(f"[ERROR] 加载失败: {e}")
            all_results["results"][model_name] = {"error": str(e), "answers": []}
            continue

        model_answers = []
        for i, q in enumerate(questions, start=1):
            print(f"[{now_str()}] [{model_name}] 第 {i}/{len(questions)} 题生成中...")
            start = time.time()
            try:
                ans = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=q,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                elapsed = round(time.time() - start, 2)
                print(f"[{now_str()}] [{model_name}] 第 {i} 题完成，耗时 {elapsed}s")
                model_answers.append(
                    {
                        "question_id": i,
                        "question": q,
                        "answer": ans,
                        "elapsed_sec": elapsed,
                    }
                )
            except Exception as e:
                elapsed = round(time.time() - start, 2)
                print(f"[ERROR] [{model_name}] 第 {i} 题失败，耗时 {elapsed}s, 错误: {e}")
                model_answers.append(
                    {
                        "question_id": i,
                        "question": q,
                        "answer": "",
                        "elapsed_sec": elapsed,
                        "error": str(e),
                    }
                )

        all_results["results"][model_name] = {"error": "", "answers": model_answers}

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = os.path.join(args.output_dir, f"results_{run_id}.json")
    md_path = os.path.join(args.output_dir, f"results_{run_id}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"# 大语言模型对比测试结果（{run_id}）")
    lines.append("")
    lines.append(f"- 测试时间：{all_results['time']}")
    lines.append(f"- 模型列表：{', '.join(args.models)}")
    lines.append(
        f"- 参数：max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}"
    )
    lines.append("")

    for model_name, data in all_results["results"].items():
        lines.append(f"## 模型：{model_name}")
        if data.get("error"):
            lines.append(f"- 加载错误：{data['error']}")
            lines.append("")
            continue
        for item in data.get("answers", []):
            lines.append(f"### Q{item['question_id']}")
            lines.append(f"**问题**：{item['question']}")
            if item.get("error"):
                lines.append(f"**错误**：{item['error']}")
            else:
                lines.append(f"**回答**：{item['answer']}")
            lines.append(f"**耗时**：{item['elapsed_sec']} 秒")
            lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[{now_str()}] 测试完成")
    print(f"JSON 结果: {json_path}")
    print(f"Markdown 结果: {md_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。")
        sys.exit(130)


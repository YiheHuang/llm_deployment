# llm_deployment

人工智能导论 HW3：大语言模型部署体验项目（黄一和，2351269）。

## 项目简介
本项目基于 ModelScope CPU 环境，完成了两个中文大语言模型的部署与对比实验：
1. `Qwen-7B-Chat`
2. `ChatGLM3-6B`

统一采用同一组测试问题与同一推理参数，输出结果保存为 Markdown 与 JSON，最终整理为课程报告。

## 项目结构
```text
.
├─ docs/
│  ├─ hw3_报告.tex          # 报告 LaTeX 源文件
│  └─ hw3_报告.pdf          # 报告 PDF
├─ images/                  # 实验过程截图
├─ results/
│  ├─ results_20260530_212226.json
│  └─ results_20260530_212226.md
├─ scripts/
│  └─ run_eval.py           # 统一评测脚本
└─ reqs/                    # 作业原始要求与参考材料
```

## 复现实验
在 ModelScope 终端中运行：

```bash
python scripts/run_eval.py \
  --models /mnt/data/Qwen-7B-Chat /mnt/data/chatglm3-6b \
  --max-new-tokens 80 \
  --temperature 0.2 \
  --top-p 0.8 \
  --output-dir ./results
```

## 得分点对应说明
根据作业评分细则，本仓库中的对应证据如下：

1. 项目公开可访问链接（8分）
1. 当前仓库即公开链接：`https://github.com/YiheHuang/llm_deployment`

2. 报告（12分）
1. 部署截图（3分）：
   - `images/账户首页+绑定阿里云.png`
   - `images/cpu环境启动.png`
   - `images/pipinstall所有依赖.png`
   - `images/gitclone完成.png`
2. 问答测试结果截图（3分）：
   - `images/运行脚本做对比实验.png`
   - 详细原始输出：`results/results_20260530_212226.md`
3. 大语言模型横向对比分析（6分）：
   - 报告章节：`docs/hw3_报告.tex` 的“实验结果”部分
   - 原始数据：`results/results_20260530_212226.json`

## 备注
1. 本实验在 CPU 环境下完成，推理耗时较长属正常现象。
2. 报告附录已包含评测脚本与原始结果文件内容，便于审阅与复现。


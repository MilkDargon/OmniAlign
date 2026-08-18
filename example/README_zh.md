<p align="left">
  中文&nbsp;｜&nbsp;<a href="./README.md">English</a>
</p>

# OmniAlign 推理示例

本 `example/` 目录是 OmniAlign 模型的**独立推理示例**。

> **多语言能力，并非仅限中英。**  
> OmniAlign 是一个统一的多语言 checkpoint，同时支持跨语言**词对齐**与**句对齐**。已在多种语言对上评测（如 zh–en、de–en、fr–en、es–en、pt–en、ru–en、it–en、ja–en、ro–en、de–fr 等），并对未见语言对表现出 zero-shot 能力。  
> 下方命令仅以**中英**作为简短示例。使用其他语言对时，请替换 `--src` / `--tgt`，并设置对应的 `--src_lang` / `--tgt_lang`。对其他语言，请换成该语言合适的分词器（必要时也包括分句器）。

## 快速开始

```bash
# 1. 安装仅推理所需依赖（在模型仓库根目录执行）
pip install -r example/requirements.txt

# 2. 运行端到端示例
bash example/example.sh
```

### 词对齐（示例：zh–en）

```bash
python example/example.py \
  --src "他没有遵守承诺" \
  --tgt "He broke promise" \
  --src_lang zh --tgt_lang en \
  --task word_align \
  --model_path .
```

输出：

```
===Word Alignment Result ===
他  <==>  He
没有  <==>  broke
遵守  <==>  broke
承诺  <==>  promise
```

### 句对齐（示例：en–zh）

```bash
python example/example.py \
  --src "Large language models (LLMs) represent one of the most transformative breakthroughs in artificial intelligence over the past decade. Trained on terabytes of text data spanning books, websites, and academic papers, these models excel at capturing complex linguistic patterns, contextual nuances, and even logical reasoning." \
  --tgt "大语言模型（LLMs）是过去十年人工智能领域最具变革性的突破之一。这些模型基于涵盖书籍、网站与学术论文在内的万亿字节级文本数据进行训练，擅长捕捉复杂的语言模式、语境细微差异乃至逻辑推理能力。" \
  --src_lang en --tgt_lang zh \
  --task sent_align \
  --model_path .
```

输出：

```
===Sentence Alignment Result ===
Large language models (LLMs) represent one of the most transformative breakthroughs in artificial intelligence over the past decade.  <==>  大语言模型（LLMs）是过去十年人工智能领域最具变革性的突破之一。
Trained on terabytes of text data spanning books, websites, and academic papers, these models excel at capturing complex linguistic patterns, contextual nuances, and even logical reasoning.  <==>  这些模型基于涵盖书籍、网站与学术论文在内的万亿字节级文本数据进行训练，擅长捕捉复杂的语言模式、语境细微差异乃至逻辑推理能力。
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--src` | （必填） | 源文本 |
| `--tgt` | （必填） | 目标文本 |
| `--src_lang` / `--tgt_lang` | `zh` / `en` | 两侧语言代码（默认值仅服务于短示例，**不代表**模型只支持中英） |
| `--task` | `word_align` | `word_align` 或 `sent_align` |
| `--model_path` | （必填） | OmniAlign 模型目录路径（在模型仓库根目录运行时可用 `.`） |
| `--align_layer` | `7` | 词对齐使用的 Transformer 层 |
| `--threshold` | `1e-3` | 词对齐 softmax 阈值 |
| `--device` | `cuda:0` | 推理设备 |

各语言对的评测结果见模型卡（[`../README_CN.md`](../README_CN.md)）。

---

*Read this in [English](./README.md).*

<p align="left">
  <a href="./README_zh.md">中文</a>&nbsp;｜&nbsp;English
</p>

# OmniAlign Inference Example

This `example/` folder is a self-contained inference example for the OmniAlign model.

> **Multilingual, not zh–en only.**  
> OmniAlign is a single multilingual checkpoint for cross-lingual word alignment and sentence alignment. It is evaluated on many pairs (e.g. zh–en, de–en, fr–en, es–en, pt–en, ru–en, it–en, ja–en, ro–en, de–fr, …) and also shows zero-shot behavior on unseen pairs.  
> The commands below use Chinese–English only as short illustrative examples. Replace `--src` / `--tgt` and set `--src_lang` / `--tgt_lang` to any supported language codes for other pairs. For other languages, please swap in an appropriate tokenizer (and sentence splitter if needed) for that language.

## Quick Start

```bash
# 1. Install inference-only dependencies (from the model repo root)
pip install -r example/requirements.txt

# 2. Run the bundled end-to-end example
bash example/example.sh
```

### Word alignment (example: zh–en)

```bash
python example/example.py \
  --src "他没有遵守承诺" \
  --tgt "He broke promise" \
  --src_lang zh --tgt_lang en \
  --task word_align \
  --model_path .
```

Output:

```
===Word Alignment Result ===
他  <==>  He
没有  <==>  broke
遵守  <==>  broke
承诺  <==>  promise
```

### Sentence alignment (example: en–zh)

```bash
python example/example.py \
  --src "Large language models (LLMs) represent one of the most transformative breakthroughs in artificial intelligence over the past decade. Trained on terabytes of text data spanning books, websites, and academic papers, these models excel at capturing complex linguistic patterns, contextual nuances, and even logical reasoning." \
  --tgt "大语言模型（LLMs）是过去十年人工智能领域最具变革性的突破之一。这些模型基于涵盖书籍、网站与学术论文在内的万亿字节级文本数据进行训练，擅长捕捉复杂的语言模式、语境细微差异乃至逻辑推理能力。" \
  --src_lang en --tgt_lang zh \
  --task sent_align \
  --model_path .
```

Output:

```
===Sentence Alignment Result ===
Large language models (LLMs) represent one of the most transformative breakthroughs in artificial intelligence over the past decade.  <==>  大语言模型（LLMs）是过去十年人工智能领域最具变革性的突破之一。
Trained on terabytes of text data spanning books, websites, and academic papers, these models excel at capturing complex linguistic patterns, contextual nuances, and even logical reasoning.  <==>  这些模型基于涵盖书籍、网站与学术论文在内的万亿字节级文本数据进行训练，擅长捕捉复杂的语言模式、语境细微差异乃至逻辑推理能力。
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--src` | (required) | Source text |
| `--tgt` | (required) | Target text |
| `--src_lang` / `--tgt_lang` | `zh` / `en` | Language code of each side (defaults are only for the short examples; **not** a model limitation) |
| `--task` | `word_align` | `word_align` or `sent_align` |
| `--model_path` | (required) | Path to the OmniAlign model folder (use `.` when running from the model repo root) |
| `--align_layer` | `7` | Transformer layer used for word alignment |
| `--threshold` | `1e-3` | Softmax threshold for word alignment |
| `--device` | `cuda:0` | Device to run inference on |

For benchmark numbers across language pairs, see the model card ([`../README.md`](../README.md)).

---

*Read this in [中文](./README_zh.md).*

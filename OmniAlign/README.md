---
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction
inference: false
language:
- multilingual
- zh
- en
- de
- fr
- es
- it
- ru
- ja
- pt
- ro
base_model: Alibaba-NLP/gte-multilingual-mlm-base
tags:
- word-alignment
- sentence-alignment
- multilingual
---

<p align="center">
  <a href="https://huggingface.co/WPS-Qingqiu/OmniAlign/blob/main/README_CN.md">中文</a>&nbsp;｜&nbsp;English
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper (arXiv)</a>&nbsp;|&nbsp;
  <a href="https://github.com/MilkDargon/OmniAlign">💻 GitHub</a>
</p>

# OmniAlign

> Universal multilingual sequence alignment model for **sentence alignment** and **word alignment**.

**OmniAlign** is a universal multilingual alignment model designed to deliver efficient **sentence alignment** and **word alignment** through multi-stage training. It combines contextual word embeddings for token-level alignment with sentence embeddings for cross-lingual semantic matching, and progressively strengthens both through a four-stage training pipeline (pretrain → unsupervised → supervised → distillation).

- **Word alignment**: infers token similarity from contextual word embeddings and derives subword-level alignments.
- **Sentence alignment**: encodes source and target sentences into a shared vector space, then matches them with a dynamic-programming two-step approach — first pinning down approximate one-to-one anchors, then searching for all valid alignments under anchor constraints.
- **Four-stage training**: progressively strengthens both word and sentence embeddings through four training stages.

This model is the final checkpoint produced by the four-stage distillation pipeline, built on top of [`Alibaba-NLP/gte-multilingual-mlm-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-mlm-base).

> **This repository hosts model weights and configs only.**  
> Inference code, alignment scripts, and runnable examples are in the GitHub repo: [**MilkDargon/OmniAlign**](https://github.com/MilkDargon/OmniAlign).  
> Loading the weights here gives you the encoder; word/sentence alignment is implemented in the GitHub `example/` code.

---

## Model Details

| Attribute | Value |
|---|---|
| **Model type** | Transformer encoder (GTE / NEW architecture) |
| **Base model** | [`Alibaba-NLP/gte-multilingual-mlm-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-mlm-base) |
| **Language** | Multilingual (optimized for zh–en, en–es, en–it, en–de, en–fr, en–ru, de–fr, and more) |
| **Params** | ~305M |
| **Max sequence length** | 8192 tokens |
| **Training** | Four-stage: pretrain → unsupervised → supervised → distillation |
| **License** | Apache 2.0 |

---

## Installation

Clone the **GitHub** repo (code), then install inference dependencies. Python 3.12 is recommended.

```bash
git clone https://github.com/MilkDargon/OmniAlign.git
cd OmniAlign
hf download WPS-Qingqiu/OmniAlign --local-dir OmniAlign
pip install -r example/requirements.txt
```

The example needs only inference dependencies (`torch`, `transformers`, `sentence_transformers`, `jieba`, `nltk`, `sentence_splitter`, `numpy`, `numba`, `faiss-cpu`).

Weights are downloaded from this Hub repo into a local `OmniAlign/` folder. If the `hf` command is missing, run `pip install -U huggingface_hub` first.

---

## Quick Start

From the cloned GitHub repo, set `--model_path` to the downloaded weights directory:

### Word Alignment

```bash
python example/example.py \
  --src "他没有遵守承诺" \
  --tgt "He broke promise" \
  --src_lang zh --tgt_lang en \
  --task word_align \
  --model_path OmniAlign
```

Output:

```
===Word Alignment Result ===
他  <==>  He
没有  <==>  broke
遵守  <==>  broke
承诺  <==>  promise
```

### Sentence Alignment

```bash
python example/example.py \
  --src "Large language models (LLMs) represent one of the most transformative breakthroughs in artificial intelligence over the past decade. Trained on terabytes of text data spanning books, websites, and academic papers, these models excel at capturing complex linguistic patterns, contextual nuances, and even logical reasoning." \
  --tgt "大语言模型（LLMs）是过去十年人工智能领域最具变革性的突破之一。这些模型基于涵盖书籍、网站与学术论文在内的万亿字节级文本数据进行训练，擅长捕捉复杂的语言模式、语境细微差异乃至逻辑推理能力。" \
  --src_lang en --tgt_lang zh \
  --task sent_align \
  --model_path OmniAlign
```

Output:

```
===Sentence Alignment Result ===
Large language models (LLMs) represent one of the most transformative breakthroughs in artificial intelligence over the past decade.  <==>  大语言模型（LLMs）是过去十年人工智能领域最具变革性的突破之一。
Trained on terabytes of text data spanning books, websites, and academic papers, these models excel at capturing complex linguistic patterns, contextual nuances, and even logical reasoning.  <==>  这些模型基于涵盖书籍、网站与学术论文在内的万亿字节级文本数据进行训练，擅长捕捉复杂的语言模式、语境细微差异乃至逻辑推理能力。
```

See [`example/README.md`](https://github.com/MilkDargon/OmniAlign/blob/main/example/README.md) in the GitHub repo for the full list of arguments.

---

## Evaluation

### Word Alignment (AER, lower is better)

AER (Alignment Error Rate) on standard word alignment test sets. For each language pair, the best result is <span style="color:red"><b>red and bold</b></span>, and the second-best is <span style="color:blue"><u>blue and underlined</u></span>.

<div translate="no">

| Methods | zh-en | de-en | fr-en | ro-en | ja-en | es-en | pt-en | ru-en | it-en |
|---|---|---|---|---|---|---|---|---|---|
| FastAlign | 38.1 | 27.0 | 10.5 | 27.0 | 51.1 | - | - | - | - |
| GIZA++ | 35.1 | 20.6 | 5.9 | 26.4 | 48.0 | - | - | - | - |
| SimAlign | 21.6 | 16.6 | 7.5 | 22.3 | 46.6 | 14.2 | 14.1 | 15.4 | 17.7 |
| AwesomeAlign | 13.3 | 13.3 | 3.8 | 18.7 | 37.4 | 12.0 | 12.7 | <span style="color:blue"><u>13.5</u></span> | 15.7 |
| AccAlign | 11.5 | 12.1 | 2.8 | 16.9 | 36.8 | <span style="color:blue"><u>11.1</u></span> | <span style="color:blue"><u>12.1</u></span> | 12.5 | <span style="color:blue"><u>14.3</u></span> |
| WSPAlign (Bilingual) | 13.1 | 11.1 | 2.8 | <span style="color:blue"><u>10.1</u></span> | <span style="color:blue"><u>19.3</u></span> | - | - | - | - |
| WSPAlign (Multilingual) | 22.3 | 20.0 | 12.8 | 26.4 | 45.8 | 13.4 | 12.3 | 13.1 | 17.1 |
| BinaryAlign (Bilingual) | <span style="color:red"><b>4.8</b></span> | <span style="color:red"><b>7.8</b></span> | <span style="color:red"><b>1.9</b></span> | <span style="color:red"><b>7.4</b></span> | <span style="color:red"><b>14.3</b></span> | - | - | - | - |
| **OmniAlign (ours)** | <span style="color:blue"><u>8.5</u></span> | <span style="color:blue"><u>11.0</u></span> | <span style="color:blue"><u>2.7</u></span> | 16.7 | 29.6 | <span style="color:red"><b>10.7</b></span> | <span style="color:red"><b>11.9</b></span> | <span style="color:red"><b>12.1</b></span> | <span style="color:red"><b>14.1</b></span> |

</div>

### Sentence Alignment (F1, higher is better)

F1 scores on standard sentence alignment test sets. For each language pair, the best result is <span style="color:red"><b>red and bold</b></span>, and the second-best is <span style="color:blue"><u>blue and underlined</u></span>.

<div translate="no">

| Algorithm | en-zh | en-es | en-it | en-de | en-fr | en-ru | de-fr |
|---|---|---|---|---|---|---|---|
| Gale–Church | 0.682 | 0.900 | 0.977 | 0.897 | 0.838 | 0.911 | 0.680 |
| BleuAlign | 0.782 | 0.819 | 0.901 | 0.806 | 0.757 | 0.791 | 0.770 |
| VecAlign | 0.957 | 0.892 | 0.956 | 0.869 | 0.880 | 0.921 | 0.902 |
| BertAlign | <span style="color:blue"><u>0.969</u></span> | <span style="color:blue"><u>0.897</u></span> | <span style="color:red"><b>0.984</b></span> | <span style="color:blue"><u>0.900</u></span> | <span style="color:blue"><u>0.909</u></span> | <span style="color:red"><b>0.938</b></span> | <span style="color:red"><b>0.939</b></span> |
| SentAlign | 0.968 | 0.872 | 0.978 | 0.892 | 0.903 | 0.920 | <span style="color:blue"><u>0.932</u></span> |
| CrocoAlign | 0.660 | 0.696 | 0.864 | 0.804 | 0.788 | 0.783 | 0.714 |
| **OmniAlign (ours)** | <span style="color:red"><b>0.970</b></span> | <span style="color:red"><b>0.906</b></span> | <span style="color:blue"><u>0.978</u></span> | <span style="color:red"><b>0.913</b></span> | <span style="color:red"><b>0.912</b></span> | <span style="color:blue"><u>0.935</u></span> | 0.922 |

</div>

---

## Citation

<!-- TODO: fill in the paper citation once available -->

```bibtex
@misc{omni-align,
  title        = {OmniAlign: Universal Multilingual Sequence Alignment},
  author       = {yangmengpeng},
  year         = {2026},
  publisher    = {Hugging Face},
  journal      = {Hugging Face repository},
  howpublished = {https://huggingface.co/WPS-Qingqiu/OmniAlign}
}
```

---

## License

This model is released under the [Apache License, Version 2.0](LICENSE).

*Read this in [中文](https://huggingface.co/WPS-Qingqiu/OmniAlign/blob/main/README_CN.md).*

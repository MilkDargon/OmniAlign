<p align="center">
  <a href="README_CN.md">中文</a>&nbsp;｜&nbsp;English
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper (arXiv)</a>&nbsp;|&nbsp;
  <a href="https://github.com/YOUR_ORG/Omni-align">💻 GitHub</a>&nbsp;|&nbsp;
  <a href="https://YOUR_EXAMPLE_URL">🕹️ Example</a>
</p>

# OmniAlign

> Universal multilingual sequence alignment model for **sentence alignment** and **word alignment**.

**OmniAlign** is a universal multilingual alignment model designed to deliver efficient **sentence alignment** and **word alignment** through multi-stage training. It combines contextual word embeddings for token-level alignment with sentence embeddings for cross-lingual semantic matching, and progressively strengthens both through a four-stage training pipeline (pretrain → unsupervised → supervised → distillation).

- **Word alignment**: infers token similarity from contextual word embeddings and derives subword-level alignments.
- **Sentence alignment**: encodes source and target sentences into a shared vector space, then matches them with a dynamic-programming two-step approach — first pinning down approximate one-to-one anchors, then searching for all valid alignments under anchor constraints.
- **Four-stage training**: progressively strengthens both word and sentence embeddings through four training stages.

This model is the final checkpoint produced by the four-stage distillation pipeline, built on top of [`Alibaba-NLP/gte-multilingual-mlm-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-mlm-base).

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

## Evaluation

### Word Alignment (AER, lower is better)

AER (Alignment Error Rate) on standard word alignment test sets. For each language pair, the best result is **bolded** and the second-best is underlined.

| Methods | zh-en | de-en | fr-en | ro-en | ja-en | es-en | pt-en | ru-en | it-en |
|---|---|---|---|---|---|---|---|---|---|
| FastAlign | 38.1 | 27.0 | 10.5 | 27.0 | 51.1 | - | - | - | - |
| GIZA++ | 35.1 | 20.6 | 5.9 | 26.4 | 48.0 | - | - | - | - |
| SimAlign | 21.6 | 16.6 | 7.5 | 22.3 | 46.6 | 14.2 | 14.1 | 15.4 | 17.7 |
| AwesomeAlign | 13.3 | 13.3 | 3.8 | 18.7 | 37.4 | 12.0 | 12.7 | 13.5 | 15.7 |
| AccAlign | 11.5 | 12.1 | 2.8 | 16.9 | 36.8 | 11.1 | 12.1 | 12.5 | 14.3 |
| WSPAlign (Bilingual) | 13.1 | 11.1 | 2.8 | 10.1 | 19.3 | - | - | - | - |
| WSPAlign (Multilingual) | 22.3 | 20.0 | 12.8 | 26.4 | 45.8 | 13.4 | 12.3 | 13.1 | 17.1 |
| BinaryAlign (Bilingual) | **4.8** | **7.8** | **1.9** | **7.4** | **14.3** | - | - | - | - |
| **OmniAlign (ours)** | 8.5 | 11.0 | 2.7 | 16.7 | 29.6 | **10.7** | **11.9** | **12.1** | **14.1** |

### Sentence Alignment (F1, higher is better)

F1 scores on standard sentence alignment test sets. For each language pair, the best result is **bolded** and the second-best is underlined.

| Algorithm | en-zh | en-es | en-it | en-de | en-fr | en-ru | de-fr |
|---|---|---|---|---|---|---|---|
| Gale–Church | 0.682 | 0.900 | 0.977 | 0.897 | 0.838 | 0.911 | 0.680 |
| BleuAlign | 0.782 | 0.819 | 0.901 | 0.806 | 0.757 | 0.791 | 0.770 |
| VecAlign | 0.957 | 0.892 | 0.956 | 0.869 | 0.880 | 0.921 | 0.902 |
| BertAlign | 0.969 | 0.897 | **0.984** | 0.900 | 0.909 | **0.938** | **0.939** |
| SentAlign | 0.968 | 0.872 | 0.978 | 0.892 | 0.903 | 0.920 | 0.932 |
| CrocoAlign | 0.660 | 0.696 | 0.864 | 0.804 | 0.788 | 0.783 | 0.714 |
| **OmniAlign (ours)** | **0.970** | **0.906** | 0.978 | **0.913** | **0.912** | 0.935 | 0.922 |

---

## Installation

```bash
# Python 3.12 recommended
pip install -r example/requirements.txt
```

The example needs only inference dependencies (`torch`, `transformers`, `sentence_transformers`, `jieba`, `nltk`, `sentence_splitter`, `numpy`, `numba`, `faiss-cpu`).

---

## Quick Start

Run the end-to-end example from the model repo root:

```bash
bash example/example.sh
```

Or run each task directly:

### Word Alignment

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

### Sentence Alignment

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

See [`example/README.md`](example/README.md) for the full list of arguments.

---

## Loading with the HuggingFace `transformers` library

The model uses remote code (`trust_remote_code=True`). You can load it directly from the Hub:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("YOUR_ORG/OmniAlign", trust_remote_code=True)
model = AutoModel.from_pretrained("YOUR_ORG/OmniAlign", trust_remote_code=True)
```

> Replace `YOUR_ORG/OmniAlign` with the actual model repository id once published.

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
  howpublished = {https://huggingface.co/YOUR_ORG/OmniAlign}
}
```

---

## License

This model is released under the [Apache License, Version 2.0](LICENSE).

*Read this in [中文](README_CN.md).*

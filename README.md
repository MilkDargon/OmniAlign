<p align="center">
  <a href="README_CN.md">中文</a>&nbsp;｜&nbsp;English
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper (arXiv)</a>&nbsp;|&nbsp;
  <a href="https://huggingface.co/WPS-Qingqiu/OmniAlign"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="18" height="18" style="vertical-align: text-bottom;"> Hugging Face</a>
</p>

# OmniAlign

> Universal multilingual sequence alignment model for **sentence alignment** and **word alignment**.

**OmniAlign** is a universal multilingual alignment model designed to deliver efficient **sentence alignment** and **word alignment** through multi-stage training. It combines contextual word embeddings for token-level alignment with sentence embeddings for cross-lingual semantic matching, and progressively strengthens both through a four-stage training pipeline (pretrain → unsupervised → supervised → distillation).

- **Word alignment**: infers token similarity from contextual word embeddings and derives subword-level alignments.
- **Sentence alignment**: encodes source and target sentences into a shared vector space, then matches them with a dynamic-programming two-step approach — first pinning down approximate one-to-one anchors, then searching for all valid alignments under anchor constraints.
- **Four-stage training**: progressively strengthens both word and sentence embeddings through four training stages.

This model is the final checkpoint produced by the four-stage distillation pipeline, built on top of [`Alibaba-NLP/gte-multilingual-mlm-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-mlm-base).

> **This GitHub repository hosts inference code only.**  
> Model weights live on Hugging Face: [`WPS-Qingqiu/OmniAlign`](https://huggingface.co/WPS-Qingqiu/OmniAlign).  
> Download them into the local `OmniAlign/` directory before running the examples.

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

Clone this repo, download the weights from Hugging Face into `OmniAlign/`, then install inference dependencies. Python 3.12 is recommended.

```bash
git clone https://github.com/MilkDargon/OmniAlign.git
cd OmniAlign
hf download WPS-Qingqiu/OmniAlign --local-dir OmniAlign
pip install -r example/requirements.txt
```

The example needs only inference dependencies (`torch`, `transformers`, `sentence_transformers`, `jieba`, `nltk`, `sentence_splitter`, `numpy`, `numba`, `faiss-cpu`).

If the `hf` command is missing, run `pip install -U huggingface_hub` first.

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

See [`example/README.md`](example/README.md) for the full list of arguments.

---

## Evaluation

### Word Alignment (AER, lower is better)

AER (Alignment Error Rate) on standard word alignment test sets. For each language pair, the best result is <span style="color:#d1242f"><strong>red and bold</strong></span>, and the second-best is <span style="color:#0969da"><u>blue and underlined</u></span>.

<table>
  <thead>
    <tr>
      <th>Methods</th>
      <th>zh-en</th>
      <th>de-en</th>
      <th>fr-en</th>
      <th>ro-en</th>
      <th>ja-en</th>
      <th>es-en</th>
      <th>pt-en</th>
      <th>ru-en</th>
      <th>it-en</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FastAlign</td>
      <td>38.1</td><td>27.0</td><td>10.5</td><td>27.0</td><td>51.1</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>GIZA++</td>
      <td>35.1</td><td>20.6</td><td>5.9</td><td>26.4</td><td>48.0</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>SimAlign</td>
      <td>21.6</td><td>16.6</td><td>7.5</td><td>22.3</td><td>46.6</td><td>14.2</td><td>14.1</td><td>15.4</td><td>17.7</td>
    </tr>
    <tr>
      <td>AwesomeAlign</td>
      <td>13.3</td><td>13.3</td><td>3.8</td><td>18.7</td><td>37.4</td><td>12.0</td><td>12.7</td><td><span style="color:#0969da"><u>13.5</u></span></td><td>15.7</td>
    </tr>
    <tr>
      <td>AccAlign</td>
      <td>11.5</td><td>12.1</td><td>2.8</td><td>16.9</td><td>36.8</td><td><span style="color:#0969da"><u>11.1</u></span></td><td><span style="color:#0969da"><u>12.1</u></span></td><td>12.5</td><td><span style="color:#0969da"><u>14.3</u></span></td>
    </tr>
    <tr>
      <td>WSPAlign (Bilingual)</td>
      <td>13.1</td><td>11.1</td><td>2.8</td><td><span style="color:#0969da"><u>10.1</u></span></td><td><span style="color:#0969da"><u>19.3</u></span></td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>WSPAlign (Multilingual)</td>
      <td>22.3</td><td>20.0</td><td>12.8</td><td>26.4</td><td>45.8</td><td>13.4</td><td>12.3</td><td>13.1</td><td>17.1</td>
    </tr>
    <tr>
      <td>BinaryAlign (Bilingual)</td>
      <td><span style="color:#d1242f"><strong>4.8</strong></span></td><td><span style="color:#d1242f"><strong>7.8</strong></span></td><td><span style="color:#d1242f"><strong>1.9</strong></span></td><td><span style="color:#d1242f"><strong>7.4</strong></span></td><td><span style="color:#d1242f"><strong>14.3</strong></span></td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td><strong>OmniAlign (ours)</strong></td>
      <td><span style="color:#0969da"><u>8.5</u></span></td><td><span style="color:#0969da"><u>11.0</u></span></td><td><span style="color:#0969da"><u>2.7</u></span></td><td>16.7</td><td>29.6</td><td><span style="color:#d1242f"><strong>10.7</strong></span></td><td><span style="color:#d1242f"><strong>11.9</strong></span></td><td><span style="color:#d1242f"><strong>12.1</strong></span></td><td><span style="color:#d1242f"><strong>14.1</strong></span></td>
    </tr>
  </tbody>
</table>

### Sentence Alignment (F1, higher is better)

F1 scores on standard sentence alignment test sets. For each language pair, the best result is <span style="color:#d1242f"><strong>red and bold</strong></span>, and the second-best is <span style="color:#0969da"><u>blue and underlined</u></span>.

<table>
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>en-zh</th>
      <th>en-es</th>
      <th>en-it</th>
      <th>en-de</th>
      <th>en-fr</th>
      <th>en-ru</th>
      <th>de-fr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gale–Church</td>
      <td>0.682</td><td>0.900</td><td>0.977</td><td>0.897</td><td>0.838</td><td>0.911</td><td>0.680</td>
    </tr>
    <tr>
      <td>BleuAlign</td>
      <td>0.782</td><td>0.819</td><td>0.901</td><td>0.806</td><td>0.757</td><td>0.791</td><td>0.770</td>
    </tr>
    <tr>
      <td>VecAlign</td>
      <td>0.957</td><td>0.892</td><td>0.956</td><td>0.869</td><td>0.880</td><td>0.921</td><td>0.902</td>
    </tr>
    <tr>
      <td>BertAlign</td>
      <td><span style="color:#0969da"><u>0.969</u></span></td><td><span style="color:#0969da"><u>0.897</u></span></td><td><span style="color:#d1242f"><strong>0.984</strong></span></td><td><span style="color:#0969da"><u>0.900</u></span></td><td><span style="color:#0969da"><u>0.909</u></span></td><td><span style="color:#d1242f"><strong>0.938</strong></span></td><td><span style="color:#d1242f"><strong>0.939</strong></span></td>
    </tr>
    <tr>
      <td>SentAlign</td>
      <td>0.968</td><td>0.872</td><td>0.978</td><td>0.892</td><td>0.903</td><td>0.920</td><td><span style="color:#0969da"><u>0.932</u></span></td>
    </tr>
    <tr>
      <td>CrocoAlign</td>
      <td>0.660</td><td>0.696</td><td>0.864</td><td>0.804</td><td>0.788</td><td>0.783</td><td>0.714</td>
    </tr>
    <tr>
      <td><strong>OmniAlign (ours)</strong></td>
      <td><span style="color:#d1242f"><strong>0.970</strong></span></td><td><span style="color:#d1242f"><strong>0.906</strong></span></td><td><span style="color:#0969da"><u>0.978</u></span></td><td><span style="color:#d1242f"><strong>0.913</strong></span></td><td><span style="color:#d1242f"><strong>0.912</strong></span></td><td><span style="color:#0969da"><u>0.935</u></span></td><td>0.922</td>
    </tr>
  </tbody>
</table>

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

*Read this in [中文](README_CN.md).*

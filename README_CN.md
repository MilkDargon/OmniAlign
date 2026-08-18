<p align="center">
  <a href="README.md">English</a>&nbsp;｜&nbsp;中文
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper (arXiv)</a>&nbsp;|&nbsp;
  <a href="https://github.com/WPS-Qingqiu/Omni-align">💻 GitHub</a>&nbsp;|&nbsp;
  <a href="https://YOUR_EXAMPLE_URL">🕹️ 示例</a>
</p>

# OmniAlign

> 面向**句对齐**与**词对齐**的通用多语言序列对齐模型。

**OmniAlign** 是一个通用多语言对齐模型，通过多阶段训练实现高效的**句对齐**与**词对齐**。它将用于 token 级对齐的上下文词嵌入与用于跨语言语义匹配的句嵌入相结合，并通过四阶段训练流程（预训练 → 无监督 → 有监督 → 蒸馏）逐步增强两者。

- **词对齐**：通过上下文词嵌入推断 token 相似度，并转换为子词级别的对齐。
- **句对齐**：将源句与目标句编码到共享向量空间，再通过动态规划两步法进行匹配——先确定近似的一对一锚点，再在锚点约束下搜索所有有效对齐。
- **四阶段训练**：通过四个训练阶段逐步增强词嵌入与句嵌入。

本模型是四阶段蒸馏流程产出的最终 checkpoint，基于 [`Alibaba-NLP/gte-multilingual-mlm-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-mlm-base) 构建。

---

## 模型信息

| 属性 | 值 |
|---|---|
| **模型类型** | Transformer 编码器（GTE / NEW 架构） |
| **基础模型** | [`Alibaba-NLP/gte-multilingual-mlm-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-mlm-base) |
| **语言** | 多语言（针对 zh–en、en–es、en–it、en–de、en–fr、en–ru、de–fr 等优化） |
| **参数量** | 约 305M |
| **最大序列长度** | 8192 tokens |
| **训练方式** | 四阶段：预训练 → 无监督 → 有监督 → 蒸馏 |

---

## 评测结果

### 词对齐（AER，越低越好）

在标准词对齐测试集上的 AER（对齐错误率）。每个语言对中，<span style="color:red">最优结果红色加粗</span>，<span style="color:blue"><u>次优结果蓝色下划线</u></span>。

| 方法 | zh-en | de-en | fr-en | ro-en | ja-en | es-en | pt-en | ru-en | it-en |
|---|---|---|---|---|---|---|---|---|---|
| FastAlign | 38.1 | 27.0 | 10.5 | 27.0 | 51.1 | - | - | - | - |
| GIZA++ | 35.1 | 20.6 | 5.9 | 26.4 | 48.0 | - | - | - | - |
| SimAlign | 21.6 | 16.6 | 7.5 | 22.3 | 46.6 | 14.2 | 14.1 | 15.4 | 17.7 |
| AwesomeAlign | 13.3 | 13.3 | 3.8 | 18.7 | 37.4 | 12.0 | 12.7 | <span style="color:blue"><u>13.5</u></span> | 15.7 |
| AccAlign | 11.5 | 12.1 | 2.8 | 16.9 | 36.8 | <span style="color:blue"><u>11.1</u></span> | <span style="color:blue"><u>12.1</u></span> | 12.5 | <span style="color:blue"><u>14.3</u></span> |
| WSPAlign (双语) | 13.1 | 11.1 | 2.8 | <span style="color:blue"><u>10.1</u></span> | <span style="color:blue"><u>19.3</u></span> | - | - | - | - |
| WSPAlign (多语) | 22.3 | 20.0 | 12.8 | 26.4 | 45.8 | 13.4 | 12.3 | 13.1 | 17.1 |
| BinaryAlign (双语) | <span style="color:red">**4.8**</span> | <span style="color:red">**7.8**</span> | <span style="color:red">**1.9**</span> | <span style="color:red">**7.4**</span> | <span style="color:red">**14.3**</span> | - | - | - | - |
| **OmniAlign（本工作）** | <span style="color:blue"><u>8.5</u></span> | <span style="color:blue"><u>11.0</u></span> | <span style="color:blue"><u>2.7</u></span> | 16.7 | 29.6 | <span style="color:red">**10.7**</span> | <span style="color:red">**11.9**</span> | <span style="color:red">**12.1**</span> | <span style="color:red">**14.1**</span> |

### 句对齐（F1，越高越好）

在标准句对齐测试集上的 F1 分数。每个语言对中，<span style="color:red">最优结果红色加粗</span>，<span style="color:blue"><u>次优结果蓝色下划线</u></span>。

| 算法 | en-zh | en-es | en-it | en-de | en-fr | en-ru | de-fr |
|---|---|---|---|---|---|---|---|
| Gale–Church | 0.682 | 0.900 | 0.977 | 0.897 | 0.838 | 0.911 | 0.680 |
| BleuAlign | 0.782 | 0.819 | 0.901 | 0.806 | 0.757 | 0.791 | 0.770 |
| VecAlign | 0.957 | 0.892 | 0.956 | 0.869 | 0.880 | 0.921 | 0.902 |
| BertAlign | <span style="color:blue"><u>0.969</u></span> | <span style="color:blue"><u>0.897</u></span> | <span style="color:red">**0.984**</span> | <span style="color:blue"><u>0.900</u></span> | <span style="color:blue"><u>0.909</u></span> | <span style="color:red">**0.938**</span> | <span style="color:red">**0.939**</span> |
| SentAlign | 0.968 | 0.872 | 0.978 | 0.892 | 0.903 | 0.920 | <span style="color:blue"><u>0.932</u></span> |
| CrocoAlign | 0.660 | 0.696 | 0.864 | 0.804 | 0.788 | 0.783 | 0.714 |
| **OmniAlign（本工作）** | <span style="color:red">**0.970**</span> | <span style="color:red">**0.906**</span> | <span style="color:blue"><u>0.978</u></span> | <span style="color:red">**0.913**</span> | <span style="color:red">**0.912**</span> | <span style="color:blue"><u>0.935</u></span> | 0.922 |

---

## 安装

```bash
# 推荐 Python 3.12
pip install -r example/requirements.txt
```

example 仅需推理依赖（`torch`、`transformers`、`sentence_transformers`、`jieba`、`nltk`、`sentence_splitter`、`numpy`、`numba`、`faiss-cpu`）。

---

## 快速开始

在模型仓库根目录运行端到端示例：

```bash
bash example/example.sh
```

或分别运行每个任务：

### 词对齐

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

### 句对齐

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

完整参数列表见 [`example/README_zh.md`](example/README_zh.md)。

---

## 通过 HuggingFace `transformers` 加载

本模型使用远程代码（需 `trust_remote_code=True`），可直接从 Hub 加载：

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("WPS-Qingqiu/OmniAlign", trust_remote_code=True)
model = AutoModel.from_pretrained("WPS-Qingqiu/OmniAlign", trust_remote_code=True)
```

> 发布后请将 `WPS-Qingqiu/OmniAlign` 替换为实际的模型仓库 id。

---

## 引用

<!-- TODO: 论文正式发布后补充引用信息 -->

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

## 许可协议

本模型基于 [Apache License, Version 2.0](LICENSE) 发布。

*English version: [README.md](README.md).*

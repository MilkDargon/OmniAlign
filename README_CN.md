<p align="center">
  <a href="README.md">English</a>&nbsp;｜&nbsp;中文
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper (arXiv)</a>&nbsp;|&nbsp;
  <a href="https://huggingface.co/WPS-Qingqiu/OmniAlign"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="18" height="18" style="vertical-align: text-bottom;"> Hugging Face</a>
</p>

# OmniAlign

> 面向**句对齐**与**词对齐**的通用多语言序列对齐模型。

**OmniAlign** 是一个通用多语言对齐模型，通过多阶段训练实现高效的**句对齐**与**词对齐**。它将用于 token 级对齐的上下文词嵌入与用于跨语言语义匹配的句嵌入相结合，并通过四阶段训练流程（预训练 → 无监督 → 有监督 → 蒸馏）逐步增强两者。

- **词对齐**：通过上下文词嵌入推断 token 相似度，并转换为子词级别的对齐。
- **句对齐**：将源句与目标句编码到共享向量空间，再通过动态规划两步法进行匹配——先确定近似的一对一锚点，再在锚点约束下搜索所有有效对齐。
- **四阶段训练**：通过四个训练阶段逐步增强词嵌入与句嵌入。

本模型是四阶段蒸馏流程产出的最终 checkpoint，基于 [`Alibaba-NLP/gte-multilingual-mlm-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-mlm-base) 构建。

> **本 GitHub 仓库只存放推理代码。**  
> 模型权重在 Hugging Face：[**WPS-Qingqiu/OmniAlign**](https://huggingface.co/WPS-Qingqiu/OmniAlign)。  
> 运行示例前，请先把权重下载到本地的 `OmniAlign/` 目录。

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

## 安装

克隆本仓库，从 Hugging Face 将权重下载到 `OmniAlign/`，再安装推理依赖。推荐 Python 3.12。

```bash
git clone https://github.com/MilkDargon/OmniAlign.git
cd OmniAlign
hf download WPS-Qingqiu/OmniAlign --local-dir OmniAlign
pip install -r example/requirements.txt
```

example 仅需推理依赖（`torch`、`transformers`、`sentence_transformers`、`jieba`、`nltk`、`sentence_splitter`、`numpy`、`numba`、`faiss-cpu`）。

若没有 `hf` 命令，先执行 `pip install -U huggingface_hub`。

---

## 快速开始

在克隆下来的仓库中运行，并将 `--model_path` 指向刚下载的权重目录：

### 词对齐

```bash
python example/example.py \
  --src "他没有遵守承诺" \
  --tgt "He broke promise" \
  --src_lang zh --tgt_lang en \
  --task word_align \
  --model_path OmniAlign
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
  --model_path OmniAlign
```

输出：

```
===Sentence Alignment Result ===
Large language models (LLMs) represent one of the most transformative breakthroughs in artificial intelligence over the past decade.  <==>  大语言模型（LLMs）是过去十年人工智能领域最具变革性的突破之一。
Trained on terabytes of text data spanning books, websites, and academic papers, these models excel at capturing complex linguistic patterns, contextual nuances, and even logical reasoning.  <==>  这些模型基于涵盖书籍、网站与学术论文在内的万亿字节级文本数据进行训练，擅长捕捉复杂的语言模式、语境细微差异乃至逻辑推理能力。
```

完整参数列表见 [`example/README_zh.md`](example/README_zh.md)。

---

## 评测结果

### 词对齐（AER，越低越好）

在标准词对齐测试集上的 AER（对齐错误率）。每个语言对中，<span style="color:#d1242f"><strong>最优结果红色加粗</strong></span>，<span style="color:#0969da"><u>次优结果蓝色下划线</u></span>。

<table>
  <thead>
    <tr>
      <th>方法</th>
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
      <td>WSPAlign (双语)</td>
      <td>13.1</td><td>11.1</td><td>2.8</td><td><span style="color:#0969da"><u>10.1</u></span></td><td><span style="color:#0969da"><u>19.3</u></span></td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>WSPAlign (多语)</td>
      <td>22.3</td><td>20.0</td><td>12.8</td><td>26.4</td><td>45.8</td><td>13.4</td><td>12.3</td><td>13.1</td><td>17.1</td>
    </tr>
    <tr>
      <td>BinaryAlign (双语)</td>
      <td><span style="color:#d1242f"><strong>4.8</strong></span></td><td><span style="color:#d1242f"><strong>7.8</strong></span></td><td><span style="color:#d1242f"><strong>1.9</strong></span></td><td><span style="color:#d1242f"><strong>7.4</strong></span></td><td><span style="color:#d1242f"><strong>14.3</strong></span></td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td><strong>OmniAlign（本工作）</strong></td>
      <td><span style="color:#0969da"><u>8.5</u></span></td><td><span style="color:#0969da"><u>11.0</u></span></td><td><span style="color:#0969da"><u>2.7</u></span></td><td>16.7</td><td>29.6</td><td><span style="color:#d1242f"><strong>10.7</strong></span></td><td><span style="color:#d1242f"><strong>11.9</strong></span></td><td><span style="color:#d1242f"><strong>12.1</strong></span></td><td><span style="color:#d1242f"><strong>14.1</strong></span></td>
    </tr>
  </tbody>
</table>

### 句对齐（F1，越高越好）

在标准句对齐测试集上的 F1 分数。每个语言对中，<span style="color:#d1242f"><strong>最优结果红色加粗</strong></span>，<span style="color:#0969da"><u>次优结果蓝色下划线</u></span>。

<table>
  <thead>
    <tr>
      <th>算法</th>
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
      <td><strong>OmniAlign（本工作）</strong></td>
      <td><span style="color:#d1242f"><strong>0.970</strong></span></td><td><span style="color:#d1242f"><strong>0.906</strong></span></td><td><span style="color:#0969da"><u>0.978</u></span></td><td><span style="color:#d1242f"><strong>0.913</strong></span></td><td><span style="color:#d1242f"><strong>0.912</strong></span></td><td><span style="color:#0969da"><u>0.935</u></span></td><td>0.922</td>
    </tr>
  </tbody>
</table>

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

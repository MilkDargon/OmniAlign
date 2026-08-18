#!/usr/bin/env bash
# OmniAlign inference example
#
# Run from anywhere:
#   bash example/example.sh
# (the script locates the model repo root relative to its own position)

# Model repo root = parent of this script's directory (example/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/../OmniAlign"

# ---- Word alignment example ----
src="他没有遵守承诺"
tgt="He broke promise"

python "$SCRIPT_DIR/example.py" \
  --src "$src" \
  --tgt "$tgt" \
  --src_lang zh \
  --tgt_lang en \
  --task word_align \
  --model_path "$MODEL_PATH"

echo
echo "----------------------------------------"
echo

# ---- Sentence alignment example ----
src="Large language models (LLMs) represent one of the most transformative breakthroughs in artificial intelligence over the past decade. Trained on terabytes of text data spanning books, websites, and academic papers, these models excel at capturing complex linguistic patterns, contextual nuances, and even logical reasoning. Beyond basic text generation, LLMs have found practical applications in machine translation, code debugging, scientific literature summarization, and personalized tutoring systems."
tgt="大语言模型（LLMs）是过去十年人工智能领域最具变革性的突破之一。这些模型基于涵盖书籍、网站与学术论文在内的万亿字节级文本数据进行训练，擅长捕捉复杂的语言模式、语境细微差异乃至逻辑推理能力。除基础文本生成外，大语言模型已在机器翻译、代码调试、科技文献摘要以及个性化辅导系统等领域实现了落地应用。"

python "$SCRIPT_DIR/example.py" \
  --src "$src" \
  --tgt "$tgt" \
  --src_lang en \
  --tgt_lang zh \
  --task sent_align \
  --model_path "$MODEL_PATH"

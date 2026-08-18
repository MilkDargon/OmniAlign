#!/usr/bin/env python
# Copyright (C) 2025-2026 yangmengpeng (Kingsoft Office) <yangmengpeng@wps.cn>
# -*- coding: utf-8 -*-

"""
Word alignment via hidden-state intersection

Usage:
python word_align.py \
    --src "他没有遵守承诺" \
    --tgt "He broke promise" \
    --src_lang zh \
    --tgt_lang en \
    --model_path checkpoint/four_stage_distill \
    --align_layer 7 \
    --threshold 1e-3 \
    --device cuda:0
"""

import argparse
import itertools
import re
import jieba
import torch
import transformers
from nltk import word_tokenize
from corelib import *
from sentence_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer
# ----------------------------
# word split
# ----------------------------
def word_split(text, lang):
    if lang == "zh":
        return list(jieba.cut(text))
    elif lang == "en":
        return word_tokenize(text)
    return text.split()
# ----------------------------
# sent split
# ----------------------------
def split_sents(text, lang):
    def _split_zh(text, limit=1000):
        sent_list = []
        text = re.sub('(?P<quotation_mark>([。？！](?![”’"\'）])))', r'\g<quotation_mark>\n', text)
        text = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'）])', r'\g<quotation_mark>\n', text)

        sent_list_ori = text.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)

        return sent_list
    if lang == 'zh':
        sents = _split_zh(text)
    else:
        splitter = SentenceSplitter(language=lang)
        sents = splitter.split(text=text) 
        sents = [sent.strip() for sent in sents]
    return sents

def clean_text(text):
    clean_lines = []
    text = text.strip()
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line:
            line = re.sub(r"\s+", " ", line)
            clean_lines.append(line)
    return "\n".join(clean_lines)


# ----------------------------
# word alignment function
# ----------------------------
def word_align(
    src,
    tgt,
    model,
    tokenizer,
    src_lang,
    tgt_lang,
    align_layer,
    threshold,
    device,
):
    # 1. 分词
    src_words = word_split(src, src_lang)
    tgt_words = word_split(tgt, tgt_lang)

    # 2. word -> subword
    token_src = [tokenizer.tokenize(w) for w in src_words]
    token_tgt = [tokenizer.tokenize(w) for w in tgt_words]

    wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
    wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

    flat_src = list(itertools.chain(*wid_src))
    flat_tgt = list(itertools.chain(*wid_tgt))

    ids_src = tokenizer.prepare_for_model(
        flat_src,
        return_tensors="pt",
        truncation=True,
        model_max_length=tokenizer.model_max_length,
    )["input_ids"].to(device)

    ids_tgt = tokenizer.prepare_for_model(
        flat_tgt,
        return_tensors="pt",
        truncation=True,
        model_max_length=tokenizer.model_max_length,
    )["input_ids"].to(device)

    # 3. subword -> word 映射
    sub2word_src = []
    sub2word_tgt = []

    for i, w in enumerate(token_src):
        sub2word_src.extend([i] * len(w))

    for i, w in enumerate(token_tgt):
        sub2word_tgt.extend([i] * len(w))

    # 4. forward + hidden states
    model.eval()
    with torch.no_grad():
        out_src = model(
            ids_src.unsqueeze(0),
            output_hidden_states=True,
        ).hidden_states[align_layer][0, 1:-1]

        out_tgt = model(
            ids_tgt.unsqueeze(0),
            output_hidden_states=True,
        ).hidden_states[align_layer][0, 1:-1]

        sim = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.softmax(sim, dim=-1)
        softmax_tgtsrc = torch.softmax(sim, dim=-2)

        inter = (softmax_srctgt > threshold) & (
            softmax_tgtsrc > threshold
        )

    # 5. subword -> word 对齐
    align_sub = torch.nonzero(inter, as_tuple=False)

    align_words = set()
    for i, j in align_sub:
        align_words.add(
            (sub2word_src[i.item()], sub2word_tgt[j.item()])
        )

    align_words = sorted(list(align_words))
    align_results = [
        (src_words[i], tgt_words[j]) for i, j in align_words
    ]

    return align_results


# ----------------------------
# sentence alignment
# ----------------------------
def sent_align(src,tgt,src_lang,tgt_lang, model, tokenizer, device, max_align=8, top_k=3,win=5,skip=-0.1, margin=False,is_split=False,):
    src = clean_text(src)
    tgt = clean_text(tgt)

    if is_split:
        src_sents = src.splitlines()
        tgt_sents = tgt.splitlines()
    else:
        src_sents = split_sents(src, src_lang)
        tgt_sents = split_sents(tgt, tgt_lang)

    src_num = len(src_sents)
    tgt_num = len(tgt_sents)

    print(
        f"Source language: {src_lang}, "
        f"Number of sentences: {src_num}"
    )
    print(
        f"Target language: {tgt_lang}, "
        f"Number of sentences: {tgt_num}"
    )

    src_vecs, src_lens = encode_layers_to_vecs(
        src_sents, model, tokenizer, max_align - 1
    )
    tgt_vecs, tgt_lens = encode_layers_to_vecs(
        tgt_sents, model, tokenizer, max_align - 1
    )

    char_ratio = (
       src_lens[0].sum() / tgt_lens[0].sum()
    )

    def align_sents():
        print("Performing first-step alignment ...")
        D, I = find_top_k_sents(
            src_vecs[0, :],
            tgt_vecs[0, :],
            k=top_k,
        )

        first_alignment_types = get_alignment_types(2)
        first_w, first_path = find_first_search_path(
            src_num, tgt_num
        )

        first_pointers = first_pass_align(
            src_num,
            tgt_num,
            first_w,
            first_path,
            first_alignment_types,
            D,
            I,
        )

        first_alignment = first_back_track(
            src_num,
            tgt_num,
            first_pointers,
            first_path,
            first_alignment_types,
        )

        print("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(
            max_align
        )

        second_w, second_path = find_second_path(
            first_alignment,
            win,
            src_num,
            tgt_num,
        )

        second_pointers, second_cost = second_pass_align(
            src_vecs,
            tgt_vecs,
            src_lens,
            tgt_lens,
            second_w,
            second_path,
            second_alignment_types,
            char_ratio,
            skip,
            margin=margin,
        )

        second_alignment = second_back_track(
            src_num,
            tgt_num,
            second_pointers,
            second_cost,
            second_path,
            second_alignment_types,
        )

        blocks, scores = second_alignment

        alignments = []

        for (src_ids, tgt_ids), score in zip(blocks, scores):
            src_text = " ".join(src_sents[i] for i in src_ids)
            tgt_text = " ".join(tgt_sents[j] for j in tgt_ids)

            alignments.append(
                (src_text, tgt_text)
            )

        return alignments

    return align_sents()


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        "Word/Sentence alignment script"
    )

    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source sentence",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        required=True,
        help="Target sentence",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="zh",
        help="Source language code (e.g. zh, en, de, fr, es, pt, ru, it, ja, ro, nl, hu). "
        "Defaults are for the short example sentences only; the model is multilingual.",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="en",
        help="Target language code (same as --src_lang). Not limited to zh/en.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="word_align",
        choices=["word_align", "sent_align"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--align_layer",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    args = parser.parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if args.task == "word_align":
    
        model = transformers.AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            device_map={"": args.device},
            dtype=torch.bfloat16,
        )
        alignments = word_align(
            src=args.src,
            tgt=args.tgt,
            model=model,
            tokenizer=tokenizer,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            align_layer=args.align_layer,
            threshold=args.threshold,
            device=args.device,
        )
        print("\n===Word Alignment Result ===")
        for src_word, tgt_word in alignments:
            print(f"{src_word}  <==>  {tgt_word}")
    else:
        model = SentenceTransformer(args.model_path, trust_remote_code=True, device=args.device, model_kwargs={"dtype": torch.bfloat16})
        alignments= sent_align(
            src=args.src,
            tgt=args.tgt,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            model=model,
            tokenizer=tokenizer,
            device=args.device
        )
        print("\n===Sentence Alignment Result ===")
        # print(alignments)
        for src_sent, tgt_sent in alignments:
            print(f"{src_sent}  <==>  {tgt_sent}")



if __name__ == "__main__":
    main()

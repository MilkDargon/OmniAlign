# Copyright (C) 2025-2026 yangmengpeng (Kingsoft Office) <yangmengpeng@wps.cn>
import os
import re
import shutil
import numpy as np
import numba as nb
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# ----------------------------
# Basic text helpers
# ----------------------------
def preprocess_line(line: str) -> str:
    line = line.strip()
    if len(line) == 0:
        return "BLANK_LINE"
    return line

def layer(lines, num_overlaps, comb=" "):
    if num_overlaps < 1:
        raise Exception("num_overlaps must be >= 1")
    out = ["PAD"] * min(num_overlaps - 1, len(lines))
    for i in range(len(lines) - num_overlaps + 1):
        out.append(comb.join(lines[i:i + num_overlaps]))
    return out


# ----------------------------
# Embedding (in-memory)
# ----------------------------
def encode_layers_to_vecs(lines, model: SentenceTransformer, tokenizer: AutoTokenizer,
                          num_overlaps: int, batch_size: int = 256, max_length: int = 256):
    """
    Build:
      vecs0: (num_overlaps, num_lines, dim) float32 normalized
      vecs1: (num_overlaps, num_lines) int32 lengths (tokenizer.encode)
    where num_overlaps == max_align-1
    """
    lines = [preprocess_line(x) for x in lines]
    n_lines = len(lines)
    dim = model.get_sentence_embedding_dimension()

    vecs0 = np.empty((num_overlaps, n_lines, dim), dtype=np.float32)
    vecs1 = np.empty((num_overlaps, n_lines), dtype=np.int32)

    # For each overlap size, build layer strings and embed directly
    for oi, overlap in enumerate(range(1, num_overlaps + 1)):
        layer_lines = layer(lines, overlap)

        # Safety: very long strings will slow tokenization/encoding; keep consistent with your old 10000 cut
        layer_lines = [x[:10000] for x in layer_lines]

        # SentenceTransformer encode -> np.ndarray float32
        emb = model.encode(
            layer_lines,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        if emb.shape[1] != dim:
            raise RuntimeError(f"Embedding dim mismatch: got {emb.shape[1]} vs {dim}")

        vecs0[oi, :, :] = emb

        # length feature: follow your bert_align.py choice (tokenizer length)
        # NOTE: encode per line is slower but matches old behavior; you can optimize later with batch tokenization if needed
        for jj, s in enumerate(layer_lines):
            vecs1[oi, jj] = len(tokenizer.encode(s, truncation=True, max_length=max_length))

    return vecs0, vecs1


# ----------------------------
# FAISS top-k
# ----------------------------
def find_top_k_sents(src_vecs, tgt_vecs, k=3):
    embedding_size = src_vecs.shape[1]
    index = faiss.IndexFlatIP(embedding_size)
    index.add(tgt_vecs.astype(np.float32))
    D, I = index.search(src_vecs.astype(np.float32), k)
    return D, I


# ----------------------------
# DP utilities (same logic as your code)
# ----------------------------
def get_alignment_types(max_alignment_size):
    alignment_types = [[0, 1], [1, 0]]
    for x in range(1, max_alignment_size):
        for y in range(1, max_alignment_size):
            if x + y <= max_alignment_size:
                alignment_types.append([x, y])
    return np.array(alignment_types)

def find_first_search_path(src_len, tgt_len, min_win_size=250, percent=0.06):
    win_size = max(min_win_size, int(max(src_len, tgt_len) * percent))
    search_path = []
    yx_ratio = tgt_len / src_len if src_len > 0 else 1.0
    for i in range(0, src_len + 1):
        center = int(yx_ratio * i)
        win_start = max(0, center - win_size)
        win_end = min(center + win_size, tgt_len)
        search_path.append([win_start, win_end])
    return win_size, np.array(search_path)

@nb.jit(nopython=True, fastmath=True, cache=True)
def first_pass_align(src_len, tgt_len, w, search_path, align_types, dist, index):
    cost = np.zeros((src_len + 1, 2 * w + 1))
    pointers = np.zeros((src_len + 1, 2 * w + 1), dtype=nb.int64)
    top_k = index.shape[1]

    for i in range(1, src_len + 1):
        i_start = search_path[i][0]
        i_end = search_path[i][1]
        for j in range(i_start, i_end + 1):
            if i + j == 0:
                continue
            best_score = -np.inf
            best_a = -1

            for a in range(align_types.shape[0]):
                a_1 = align_types[a][0]
                a_2 = align_types[a][1]
                prev_i = i - a_1
                prev_j = j - a_2
                if prev_i < 0 or prev_j < 0:
                    continue

                prev_i_start = search_path[prev_i][0]
                prev_i_end = search_path[prev_i][1]
                if prev_j < prev_i_start or prev_j > prev_i_end:
                    continue

                prev_j_offset = prev_j - prev_i_start
                score = cost[prev_i][prev_j_offset]
                if score == -np.inf:
                    continue

                if a_1 > 0 and a_2 > 0:
                    for k in range(top_k):
                        if index[i - 1][k] == j - 1:
                            score += dist[i - 1][k]

                if score > best_score:
                    best_score = score
                    best_a = a

            j_offset = j - i_start
            cost[i][j_offset] = best_score
            pointers[i][j_offset] = best_a

    return pointers

def first_back_track(i, j, pointers, search_path, a_types):
    alignment = []
    while True:
        j_offset = j - search_path[i][0]
        a = pointers[i][j_offset]
        s = a_types[a][0]
        t = a_types[a][1]
        if a == 2:  # best 1-1 alignment
            alignment.append((i, j))
        i = i - s
        j = j - t
        if i == 0 and j == 0:
            return alignment[::-1]

def find_second_path(align, w, src_len, tgt_len):
    last_bead_src = align[-1][0]
    last_bead_tgt = align[-1][1]

    if last_bead_src != src_len:
        if last_bead_tgt == tgt_len:
            align.pop()
        align.append((src_len, tgt_len))
    else:
        if last_bead_tgt != tgt_len:
            align.pop()
            align.append((src_len, tgt_len))

    prev_src, prev_tgt = 0, 0
    path = []
    max_w = -np.inf
    for src, tgt in align:
        lower_bound = max(0, prev_tgt - w)
        upper_bound = min(tgt_len, tgt + w)
        path.extend([(lower_bound, upper_bound) for _ in range(prev_src + 1, src + 1)])
        prev_src, prev_tgt = src, tgt
        width = upper_bound - lower_bound
        if width > max_w:
            max_w = width
    path = [path[0]] + path
    return max_w + 1, np.array(path)

@nb.jit(nopython=True, fastmath=True, cache=True)
def nb_dot(x, y):
    return np.dot(x, y)

@nb.jit(nopython=True, fastmath=True, cache=True)
def calculate_neighbor_similarity(vec, overlap, sent_idx, sent_len, db):
    left_idx = sent_idx - overlap
    right_idx = sent_idx + 1

    if right_idx <= sent_len:
        right_embed = db[0, right_idx - 1, :]
        neighbor_right_sim = nb_dot(vec, right_embed)
    else:
        neighbor_right_sim = 0.0

    if left_idx > 0:
        left_embed = db[0, left_idx - 1, :]
        neighbor_left_sim = nb_dot(vec, left_embed)
    else:
        neighbor_left_sim = 0.0

    neighbor_ave_sim = neighbor_left_sim + neighbor_right_sim
    if neighbor_right_sim != 0.0 and neighbor_left_sim != 0.0:
        neighbor_ave_sim /= 2.0
    return neighbor_ave_sim

@nb.jit(nopython=True, fastmath=True, cache=True)
def calculate_similarity_score(src_vecs, tgt_vecs,
                               src_idx, tgt_idx,
                               src_overlap, tgt_overlap,
                               src_len, tgt_len,
                               margin=False):
    src_v = src_vecs[src_overlap - 1, src_idx - 1, :]
    tgt_v = tgt_vecs[tgt_overlap - 1, tgt_idx - 1, :]
    similarity = nb_dot(src_v, tgt_v)

    if margin:
        tgt_neighbor_ave_sim = calculate_neighbor_similarity(
            src_v, tgt_overlap, tgt_idx, tgt_len, tgt_vecs
        )
        src_neighbor_ave_sim = calculate_neighbor_similarity(
            tgt_v, src_overlap, src_idx, src_len, src_vecs
        )
        neighbor_ave_sim = (tgt_neighbor_ave_sim + src_neighbor_ave_sim) / 2.0
        similarity -= neighbor_ave_sim

    return similarity

@nb.jit(nopython=True, fastmath=True, cache=True)
def calculate_length_penalty(src_lens, tgt_lens,
                             src_idx, tgt_idx,
                             src_overlap, tgt_overlap,
                             char_ratio):
    src_l = src_lens[src_overlap - 1, src_idx - 1]
    tgt_l = tgt_lens[tgt_overlap - 1, tgt_idx - 1]
    tgt_l = tgt_l * char_ratio
    min_len = min(src_l, tgt_l)
    max_len = max(src_l, tgt_l)
    length_penalty = np.log2(1.0 + min_len / max_len)
    return length_penalty

@nb.jit(nopython=True, fastmath=True, cache=True)
def second_pass_align(src_vecs, tgt_vecs, src_lens, tgt_lens,
                      w, search_path, align_types,
                      char_ratio, skip, margin=False):
    src_len = src_vecs.shape[1]
    tgt_len = tgt_vecs.shape[1]
    cost = np.zeros((src_len + 1, w))
    pointers = np.zeros((src_len + 1, w), dtype=nb.int64)

    for i in range(1, src_len + 1):
        i_start = search_path[i][0]
        i_end = search_path[i][1]
        for j in range(i_start, i_end + 1):
            if i + j == 0:
                continue
            best_score = -np.inf
            best_a = -1
            for a in range(align_types.shape[0]):
                a_1 = align_types[a][0]
                a_2 = align_types[a][1]
                prev_i = i - a_1
                prev_j = j - a_2

                if prev_i < 0 or prev_j < 0:
                    continue
                prev_i_start = search_path[prev_i][0]
                prev_i_end = search_path[prev_i][1]
                if prev_j < prev_i_start or prev_j > prev_i_end:
                    continue
                prev_j_offset = prev_j - prev_i_start
                score = cost[prev_i][prev_j_offset]
                if score == -np.inf:
                    continue

                if a_1 == 0 or a_2 == 0:
                    cur_score = skip
                else:
                    cur_score = calculate_similarity_score(
                        src_vecs, tgt_vecs,
                        i, j, a_1, a_2,
                        src_len, tgt_len,
                        margin=margin
                    )
                    len_penalty = calculate_length_penalty(
                        src_lens, tgt_lens,
                        i, j, a_1, a_2,
                        char_ratio
                    )
                    cur_score *= len_penalty

                score += cur_score
                if score > best_score:
                    best_score = score
                    best_a = a

            j_offset = j - i_start
            cost[i][j_offset] = best_score
            pointers[i][j_offset] = best_a

    return pointers, cost

def second_back_track(i, j, pointers, cost, search_path, a_types):
    alignment = []
    scores = []
    while not (i == 0 and j == 0):
        j_offset = j - search_path[i][0]
        a = pointers[i][j_offset]
        s = a_types[a][0]
        t = a_types[a][1]
        src_range = [i - offset - 1 for offset in range(s)][::-1]
        tgt_range = [j - offset - 1 for offset in range(t)][::-1]
        alignment.append((src_range, tgt_range))

        prev_i = i - s
        prev_j = j - t
        prev_j_offset = prev_j - search_path[prev_i][0]
        prev_score = cost[prev_i][prev_j_offset]
        cur_score = cost[i][j_offset]
        score = cur_score - prev_score
        scores.append(score)

        i = prev_i
        j = prev_j

    return alignment[::-1], scores[::-1]


# ----------------------------
# I/O and job control
# ----------------------------
def make_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def create_jobs(src_dir, tgt_dir, out_dir):
    jobs = []
    for file in os.listdir(src_dir):
        if re.match(r"^\d+$", file):
            src_path = os.path.abspath(os.path.join(src_dir, file))
            tgt_path = os.path.abspath(os.path.join(tgt_dir, file))
            out_path = os.path.abspath(os.path.join(out_dir, file + ".align"))
            jobs.append((src_path, tgt_path, out_path))
    jobs.sort(key=lambda x: os.path.basename(x[0]))
    return jobs

def print_alignments(alignments, out_file):
    with open(out_file, "wt", encoding="utf-8") as f:
        for x, y in alignments:
            new_x = [int(v) for v in x]
            new_y = [int(v) for v in y]
            f.write(f"{new_x}:{new_y}\n")
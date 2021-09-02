import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from textblob import TextBlob
from tqdm import tqdm
from transformers import BertTokenizer

seg2id = {
    "CLS": 0,
    "SEP": 1,
    "IMG": 2,
    "LANG": 3,
    "POS_V": 4,
    "POS_N": 5,
    "POS_M": 6,
    "POS_W": 7,
    "POS_P": 8,
    "POS_F": 9,
    "ALL": 10,
}


def _merge_bpe_tokens(input_ids, tokenizer):
    # join the bpe's, split words have ## in front except for the first portion
    input_ids = input_ids.cpu().tolist()
    input_ids = [
        id_ if id_ < tokenizer.vocab_size else 1825 for id_ in input_ids if id_ > 0
    ]
    txt_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    merged_txt_tokens = []
    id_map = (
        {}
    )  # maps original txt sequence indices to new indices with merged bpe and removed CLS, SEP,'?'

    for j in range(len(txt_tokens)):
        if len(txt_tokens[j]) > 2 and txt_tokens[j][:2] == "##":
            merged_txt_tokens[-1] += txt_tokens[j][2:]
        else:
            merged_txt_tokens.append(txt_tokens[j])
        id_map[j] = len(merged_txt_tokens) - 1

    # reverse, so that old merged idx maps to original idx
    inv_id_map = {}
    for k, v in id_map.items():
        inv_id_map.setdefault(v, []).append(k)
    return merged_txt_tokens, inv_id_map


def segment_sequence(text_ids, text_mask, image_mask, tokenizer):
    seq2seg = {}  # { "CLS", SEP", "LANG", "IMG", "POS_*"}
    special_mask = torch.tensor(
        tokenizer.get_special_tokens_mask(text_ids, already_has_special_tokens=True)
    )
    cls_idxs = (text_ids == 101).nonzero(as_tuple=False).flatten()
    sep_idxs = (text_ids == 102).nonzero(as_tuple=False).flatten()
    special_idxs = special_mask.nonzero(as_tuple=False).flatten()

    start_idx = 0
    # Segment Language Subsequences
    lang_idxs = []
    img_idxs = [
        text_mask.shape[-1] + idx
        for idx in image_mask.nonzero(as_tuple=False).flatten()
    ]
    pos_v, pos_n, pos_m, pos_w, pos_p, pos_f = [], [], [], [], [], []

    # Assume first idx is CLS
    for idx in special_idxs[1:]:
        subseq = text_ids[start_idx + 1 : idx]
        merged_subseq, id_map = _merge_bpe_tokens(subseq, tokenizer)

        pos_tags = TextBlob(" ".join(merged_subseq))
        pos_tags = [tag[1] for tag in pos_tags.tags]

        # Convert merged BPE to PoS tags
        for merged_idx, tag in enumerate(pos_tags):
            real_idx = [idx + start_idx + 1 for idx in id_map[merged_idx]]
            if tag[0] == "V":  # Verbs
                pos_v.extend(real_idx)
            elif tag[0] == "N" or tag[:2] == "PP":  # Nouns and Pronouns
                pos_n.extend(real_idx)
            elif tag[0] == "J" or tag[0] == "R":  # Modifiers
                pos_m.extend(real_idx)
            elif tag[0] == "W":  # Question words
                pos_w.extend(real_idx)
            elif tag == "IN":
                pos_p.extend(real_idx)
            else:  # The rest of it
                pos_f.extend(real_idx)

        lang_idxs.extend(i for i in range(start_idx + 1, idx))
        start_idx = idx

    # seq2seg["ALL"] = text_ids
    seq2seg["CLS"] = cls_idxs
    seq2seg["SEP"] = sep_idxs
    seq2seg["IMG"] = torch.tensor(img_idxs)
    seq2seg["LANG"] = torch.tensor(lang_idxs)
    seq2seg["POS_V"] = torch.tensor(pos_v)
    seq2seg["POS_N"] = torch.tensor(pos_n)
    seq2seg["POS_M"] = torch.tensor(pos_m)
    seq2seg["POS_W"] = torch.tensor(pos_w)
    seq2seg["POS_P"] = torch.tensor(pos_p)
    seq2seg["POS_F"] = torch.tensor(pos_f)
    return seq2seg


def get_attn_confusion(attn_weights, seq2seg):
    # attn_weights: layer x head x seq x seq
    # { segmentName: NumSegments x Layers x Heads }
    # Returns:
    seg2attn = defaultdict(dict)
    for target_segment, target_idxs in seq2seg.items():
        if not target_idxs.numel():
            continue
        for src_segment, src_idxs in seq2seg.items():
            if not src_idxs.numel():
                continue
            if target_segment.startswith("POS"):
                sliced_attns = attn_weights[..., src_idxs, :][..., target_idxs]
                seg2attn[target_segment][src_segment] = torch.sum(
                    sliced_attns, dim=[2, 3]
                ) / (src_idxs.numel() * target_idxs.numel())
            else:
                sliced_attns = attn_weights[..., src_idxs, :][..., target_idxs]
                seg2attn[target_segment][src_segment] = torch.sum(
                    sliced_attns, dim=[2, 3]
                ) / (src_idxs.numel())
    return seg2attn

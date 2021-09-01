import sys
from collections import defaultdict

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer
from scipy.stats import pearsonr, spearmanr
from textblob import TextBlob
from tqdm import tqdm

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
    input_ids = [id_ if id_ < 28896 else 1825 for id_ in input_ids if id_ > 0]
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


def _segment_sequence(input_ids, seqlen, tokenizer):
    seq2seg = {}  # { "CLS", SEP", "LANG", "IMG", "POS_*"}
    cls_idxs = (input_ids == 101).nonzero(as_tuple=False).flatten()
    sep_idxs = (input_ids == 102).nonzero(as_tuple=False).flatten()
    img_idxs = [idx for idx in range(max(sep_idxs) + 1, seqlen)]

    start_idx = 0
    # Segment Language Subsequences
    lang_idxs = []
    pos_v, pos_n, pos_m, pos_w, pos_p, pos_f = [], [], [], [], [], []
    for idx in sep_idxs:
        subseq = input_ids[start_idx + 1 : idx]
        merged_subseq, id_map = _merge_bpe_tokens(subseq, tokenizer)

        pos_tags = TextBlob(" ".join(merged_subseq))
        pos_tags = [tag[1] for tag in pos_tags.tags]
        # print(merged_subseq)
        # print(pos_tags)
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

    seq2seg["ALL"] = torch.tensor([_ for _ in range(seqlen)])
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


def _get_attn_confusion(attn_weights, seq2seg):
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
                # if target_segment.startswith("POS_V"): import pdb; pdb.set_trace()
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


def _get_cumulative_sum(attn_weights, seq2seg):
    # Slice a given modality if provided
    # attn_weights: layer x head x seq x seq
    seg2cumsum = defaultdict(list)
    for target_segment, target_idxs in seq2seg.items():
        if not target_idxs.numel():
            continue
        # import pdb; pdb.set_trace()
        attended_weights = attn_weights[..., target_idxs].sum(dim=2)
        sorted_weights, _ = attended_weights.sort(dim=-1, descending=True)
        seg2cumsum[target_segment] = (
            sorted_weights.cumsum(dim=-1) / attn_weights.shape[-1]
        )
    return seg2cumsum


def _get_visual_confs(attn_weights, img_idxs, soft_labels):
    img_attns = attn_weights[..., img_idxs].sum(dim=2).cpu()
    # Get highest prob class
    img_confs, _ = torch.max(soft_labels.cpu(), dim=-1)
    img_corr = [
        [list(pearsonr(img_confs, img_attns[layer][head])) for head in range(12)]
        for layer in range(12)
    ]
    return torch.tensor(img_corr)


def bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)


def collect_batch_metrics(batch, encoded_batch, attn_batch, tokenizer=None):
    """
    Parameters:
    batch: A batch example from the dataloader
    attn_batch: Attention weights for the given batch of shape
        Shape { batch_size x layers x heads x seqlen x seqlen }

    Returns:
    seg2attn: Dict of Attentions to each segment type from every type of segment
        { segment_name: batch_size x tensor([num_segments x head x layers]) }

    seg2cumsum: Dict of Attentions to each segment type from every type of segment
        { segment_name: batch_size x tensor([num_segments x head x layers]) }

    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    seg2attn = defaultdict(lambda: defaultdict(list))
    seg2cumsum = defaultdict(lambda: defaultdict(list))
    # pos_counts =
    img_corr = []
    # 11 keys (all_sim, lang_sim, img_sim ... ) values: 11 (layers)
    enc_sim = {
        "ALL": torch.zeros(11).cuda(),
        "LANG": torch.zeros(11).cuda(),
        "IMG": torch.zeros(11).cuda(),
        "CLS": torch.zeros(11).cuda(),
        "SEP": torch.zeros(11).cuda(),
        "POS_V": torch.zeros(11).cuda(),
        "POS_N": torch.zeros(11).cuda(),
        "POS_M": torch.zeros(11).cuda(),
        "POS_W": torch.zeros(11).cuda(),
        "POS_P": torch.zeros(11).cuda(),
        "POS_F": torch.zeros(11).cuda(),
    }
    counter = np.zeros(11)  # counter for bum examples with specific tag

    for idx, (encoded_layers, attn_weights) in enumerate(
        zip(encoded_batch, attn_batch)
    ):
        seqlen = batch["attn_masks"][idx].sum()
        input_ids = batch["input_ids"][idx].cpu()
        img_cnt = seqlen - torch.count_nonzero(input_ids)
        img_soft_labels = batch["img_soft_label"][idx][:img_cnt].cpu()

        # Collect which indices in the sequence map to each modality
        # Ex. seq2seg["LANG"] --> 1, ..., n-1
        seq2seg = _segment_sequence(input_ids, seqlen, tokenizer)

        # Get corresponding encoded layers
        all_enc = encoded_layers[:, seq2seg["ALL"], :]
        lang_enc = encoded_layers[:, seq2seg["LANG"], :]
        img_enc = encoded_layers[:, seq2seg["IMG"], :]
        cls_enc = encoded_layers[:, seq2seg["CLS"], :]
        sep_enc = encoded_layers[:, seq2seg["SEP"], :]
        if len(seq2seg["POS_V"]) != 0:
            v_enc = encoded_layers[:, seq2seg["POS_V"], :]
        if len(seq2seg["POS_N"]) != 0:
            n_enc = encoded_layers[:, seq2seg["POS_N"], :]
        if len(seq2seg["POS_M"]) != 0:
            m_enc = encoded_layers[:, seq2seg["POS_M"], :]
        if len(seq2seg["POS_W"]) != 0:
            w_enc = encoded_layers[:, seq2seg["POS_W"], :]
        if len(seq2seg["POS_P"]) != 0:
            p_enc = encoded_layers[:, seq2seg["POS_P"], :]
        if len(seq2seg["POS_F"]) != 0:
            f_enc = encoded_layers[:, seq2seg["POS_F"], :]

        # calc similarity between consecutive layers
        all_sim = []
        lang_sim = []
        img_sim = []
        cls_sim = []
        sep_sim = []
        v_sim = []
        n_sim = []
        m_sim = []
        w_sim = []
        p_sim = []
        f_sim = []

        for layer in range(len(lang_enc) - 1):
            # print((lang_enc[layer]/torch.linalg.norm(lang_enc[layer], dim=1).view(-1,1)).shape)
            all_sim.append(
                bdot(
                    all_enc[layer]
                    / torch.linalg.norm(all_enc[layer], dim=1).view(-1, 1),
                    all_enc[layer + 1]
                    / torch.linalg.norm(all_enc[layer + 1], dim=1).view(-1, 1),
                )
            )
            lang_sim.append(
                bdot(
                    lang_enc[layer]
                    / torch.linalg.norm(lang_enc[layer], dim=1).view(-1, 1),
                    lang_enc[layer + 1]
                    / torch.linalg.norm(lang_enc[layer + 1], dim=1).view(-1, 1),
                )
            )
            img_sim.append(
                bdot(
                    img_enc[layer]
                    / torch.linalg.norm(img_enc[layer], dim=1).view(-1, 1),
                    img_enc[layer + 1]
                    / torch.linalg.norm(img_enc[layer + 1], dim=1).view(-1, 1),
                )
            )
            cls_sim.append(
                bdot(
                    cls_enc[layer]
                    / torch.linalg.norm(cls_enc[layer], dim=1).view(-1, 1),
                    cls_enc[layer + 1]
                    / torch.linalg.norm(cls_enc[layer + 1], dim=1).view(-1, 1),
                )
            )
            sep_sim.append(
                bdot(
                    sep_enc[layer]
                    / torch.linalg.norm(sep_enc[layer], dim=1).view(-1, 1),
                    sep_enc[layer + 1]
                    / torch.linalg.norm(sep_enc[layer + 1], dim=1).view(-1, 1),
                )
            )
            if len(seq2seg["POS_V"]) != 0:
                v_sim.append(
                    bdot(
                        v_enc[layer]
                        / torch.linalg.norm(v_enc[layer], dim=1).view(-1, 1),
                        v_enc[layer + 1]
                        / torch.linalg.norm(v_enc[layer + 1], dim=1).view(-1, 1),
                    )
                )
            if len(seq2seg["POS_N"]) != 0:
                n_sim.append(
                    bdot(
                        n_enc[layer]
                        / torch.linalg.norm(n_enc[layer], dim=1).view(-1, 1),
                        n_enc[layer + 1]
                        / torch.linalg.norm(n_enc[layer + 1], dim=1).view(-1, 1),
                    )
                )
            if len(seq2seg["POS_M"]) != 0:
                m_sim.append(
                    bdot(
                        m_enc[layer]
                        / torch.linalg.norm(m_enc[layer], dim=1).view(-1, 1),
                        m_enc[layer + 1]
                        / torch.linalg.norm(m_enc[layer + 1], dim=1).view(-1, 1),
                    )
                )
            if len(seq2seg["POS_W"]) != 0:
                w_sim.append(
                    bdot(
                        w_enc[layer]
                        / torch.linalg.norm(w_enc[layer], dim=1).view(-1, 1),
                        w_enc[layer + 1]
                        / torch.linalg.norm(w_enc[layer + 1], dim=1).view(-1, 1),
                    )
                )
            if len(seq2seg["POS_P"]) != 0:
                p_sim.append(
                    bdot(
                        p_enc[layer]
                        / torch.linalg.norm(p_enc[layer], dim=1).view(-1, 1),
                        p_enc[layer + 1]
                        / torch.linalg.norm(p_enc[layer + 1], dim=1).view(-1, 1),
                    )
                )
            if len(seq2seg["POS_F"]) != 0:
                f_sim.append(
                    bdot(
                        f_enc[layer]
                        / torch.linalg.norm(f_enc[layer], dim=1).view(-1, 1),
                        f_enc[layer + 1]
                        / torch.linalg.norm(f_enc[layer + 1], dim=1).view(-1, 1),
                    )
                )

        # Average across tokens in a example
        # all examples will have these
        counter[:5] += 1
        all_sim = torch.stack(all_sim, dim=0)
        all_sim = torch.mean(all_sim, dim=1)
        lang_sim = torch.stack(lang_sim, dim=0)
        lang_sim = torch.mean(lang_sim, dim=1)
        img_sim = torch.stack(img_sim, dim=0)
        img_sim = torch.mean(img_sim, dim=1)
        cls_sim = torch.stack(cls_sim, dim=0)
        cls_sim = torch.mean(cls_sim, dim=1)
        sep_sim = torch.stack(sep_sim, dim=0)
        sep_sim = torch.mean(sep_sim, dim=1)
        # sometimes these dont exist in sentence
        if len(seq2seg["POS_V"]) != 0:
            counter[5] += 1
            v_sim = torch.stack(v_sim, dim=0)
            v_sim = torch.mean(v_sim, dim=1)
        if len(seq2seg["POS_N"]) != 0:
            counter[6] += 1
            n_sim = torch.stack(n_sim, dim=0)
            n_sim = torch.mean(n_sim, dim=1)
        if len(seq2seg["POS_M"]) != 0:
            counter[7] += 1
            m_sim = torch.stack(m_sim, dim=0)
            m_sim = torch.mean(m_sim, dim=1)
        if len(seq2seg["POS_W"]) != 0:
            counter[8] += 1
            w_sim = torch.stack(w_sim, dim=0)
            w_sim = torch.mean(w_sim, dim=1)
        if len(seq2seg["POS_P"]) != 0:
            counter[9] += 1
            p_sim = torch.stack(p_sim, dim=0)
            p_sim = torch.mean(p_sim, dim=1)
        if len(seq2seg["POS_F"]) != 0:
            counter[10] += 1
            f_sim = torch.stack(f_sim, dim=0)
            f_sim = torch.mean(f_sim, dim=1)
        # print("after norm", img_sim.shape, img_enc.shape[1])

        # sum examples together
        enc_sim["ALL"] += all_sim
        enc_sim["LANG"] += lang_sim
        enc_sim["IMG"] += img_sim
        enc_sim["CLS"] += cls_sim
        enc_sim["SEP"] += sep_sim
        if len(seq2seg["POS_V"]) != 0:
            enc_sim["POS_V"] += v_sim
        if len(seq2seg["POS_N"]) != 0:
            enc_sim["POS_N"] += n_sim
        if len(seq2seg["POS_M"]) != 0:
            enc_sim["POS_M"] += m_sim
        if len(seq2seg["POS_W"]) != 0:
            enc_sim["POS_W"] += w_sim
        if len(seq2seg["POS_P"]) != 0:
            enc_sim["POS_P"] += p_sim
        if len(seq2seg["POS_F"]) != 0:
            enc_sim["POS_F"] += f_sim

        # Hack for only getting img_confs for the first occurence of duplicated visual features
        img_corr.append(
            _get_visual_confs(attn_weights, seq2seg["IMG"], img_soft_labels)
        )

        attn_cumsums = _get_cumulative_sum(attn_weights, seq2seg)
        attn_confusion = _get_attn_confusion(attn_weights, seq2seg)
        for target_seg in seg2id.keys():
            for src_seg in seg2id.keys():
                if src_seg not in attn_confusion[target_seg]:
                    seg2attn[target_seg][src_seg].append(torch.zeros(12, 12).cuda())
                else:
                    seg2attn[target_seg][src_seg].append(
                        attn_confusion[target_seg][src_seg]
                    )
                    seg2cumsum[target_seg][src_seg].append(attn_cumsums[target_seg])

        # TODO: Pad 1's on seg2cumsums
    img_corr = torch.stack(img_corr)
    for target_seg in seg2id.keys():
        if target_seg not in seg2attn:
            continue
        for src_seg in seg2id.keys():
            if src_seg not in seg2attn[target_seg]:
                continue
            seg2attn[target_seg][src_seg] = torch.stack(
                seg2attn[target_seg][src_seg]
            ).cpu()
        # seg2cumsum[target_seg][src_seg] = nnseg2cumsum[target_seg][src_seg],
        #                                                    batch_first=True, padding_value=1.0)
    # print("one batch", enc_sim)
    return seg2attn, seg2cumsum, img_corr, enc_sim, counter


@torch.no_grad()
def aggregate_metrics(model, eval_dataloader, debug=False):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

    seg2attn = defaultdict(lambda: defaultdict(list))
    seg2cumsum = defaultdict(lambda: defaultdict(list))
    img_corr = []
    # encoded_similarity
    enc_sim = {
        "ALL": torch.zeros(11).cuda(),
        "LANG": torch.zeros(11).cuda(),
        "IMG": torch.zeros(11).cuda(),
        "CLS": torch.zeros(11).cuda(),
        "SEP": torch.zeros(11).cuda(),
        "POS_V": torch.zeros(11).cuda(),
        "POS_N": torch.zeros(11).cuda(),
        "POS_M": torch.zeros(11).cuda(),
        "POS_W": torch.zeros(11).cuda(),
        "POS_P": torch.zeros(11).cuda(),
        "POS_F": torch.zeros(11).cuda(),
    }
    counter = np.zeros(11)

    for idx, batch in tqdm(enumerate(eval_dataloader)):
        encoded_layers, attn_weights = model(
            batch, compute_loss=False, output_attentions=True
        )
        attn_weights = torch.stack(attn_weights).permute(1, 0, 2, 3, 4)
        encoded_layers = torch.stack(encoded_layers).permute(1, 0, 2, 3)
        # print("encoded layers", len(encoded_layers), encoded_layers[0].shape)
        # print("attn weights", len(attn_weights), attn_weights[0].shape)
        batch_metrics = collect_batch_metrics(
            batch, encoded_layers, attn_weights, tokenizer
        )

        for target_seg in seg2id.keys():
            for src_seg in seg2id.keys():
                if len(batch_metrics[0][target_seg][src_seg]) == 0:
                    continue
                seg2attn[target_seg][src_seg].append(
                    batch_metrics[0][target_seg][src_seg]
                )
            # seg2cumsum[seg].append(batch_metrics[1][seg])\
        img_corr.append(batch_metrics[2])

        # add across batches
        counter += batch_metrics[4]
        print("counter for each PoS", counter)
        batch_sim = batch_metrics[3]

        enc_sim["ALL"] += batch_sim["ALL"]
        enc_sim["LANG"] += batch_sim["LANG"]
        enc_sim["IMG"] += batch_sim["IMG"]
        enc_sim["CLS"] += batch_sim["CLS"]
        enc_sim["SEP"] += batch_sim["SEP"]
        if batch_metrics[4][5] != 0:
            enc_sim["POS_V"] += batch_sim["POS_V"]
        if batch_metrics[4][6] != 0:
            enc_sim["POS_N"] += batch_sim["POS_N"]
        if batch_metrics[4][7] != 0:
            enc_sim["POS_M"] += batch_sim["POS_M"]
        if batch_metrics[4][8] != 0:
            enc_sim["POS_W"] += batch_sim["POS_W"]
        if batch_metrics[4][9] != 0:
            enc_sim["POS_P"] += batch_sim["POS_P"]
        if batch_metrics[4][10] != 0:
            enc_sim["POS_F"] += batch_sim["POS_F"]

        if debug and idx > 2:
            break  # <<<<<<<<<<<<<<<<<<<<<
        torch.cuda.empty_cache()
        # if task=="vcr":
        #     vcr_group_shape = (len(attn_weights)//4, 4,) + attn_weights.shape[1:]
        #
        #     vcr_weights = attn_weights.reshape(vcr_group_shape)
        #     qa_weights = vcr_weights[::2,:].reshape((-1,12,12,seq_len,seq_len))
        #     qar_weights = vcr_weights[1::2,:].reshape((-1,12,12,seq_len,seq_len))
        #
        #     qa_metrics = collect_batch_metrics(batch, attn_weights, tokenizer)
        #     qar_metrics = collect_batch_metrics(batch, attn_weights, tokenizer)
        #     for seg in seg2id.keys():
        #         seg2attn[seg].append(batch_metrics[0][seg])
        #         seg2cumsum[seg].append(batch_metrics[1][seg])
        #
        #     img_conf.append(batch_metrics[2])
        #     img_attn.append(batch_metrics[3])

    # average over all examples
    for n, (key, val) in enumerate(enc_sim.items()):
        enc_sim[key] = val / counter[n]

    img_corr = torch.cat(img_corr)
    for target_seg in seg2id.keys():
        for src_seg in seg2id.keys():
            seg2attn[target_seg][src_seg] = torch.cat(seg2attn[target_seg][src_seg])
        # seg2cumsum[seg].append(batch_metrics[1][seg])
    return seg2attn, seg2cumsum, img_corr, enc_sim

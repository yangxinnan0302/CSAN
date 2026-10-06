"""Stage-wise visualization and quantitative analysis for CSAN IDIN.

The script replays the trained ``EncoderSimilarity`` module without changing
its parameters.  It records the exact region-word attention used at every
IDIN stage and exports:

* reference-style image overlays for Initial and all refinement stages;
* per-token alignment trajectories and raw arrays;
* label-free stability/drift metrics;
* correction/regression metrics when word-to-region ground truth is supplied.

Python 3.7 and PyTorch 1.7 compatible.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}
EPS = 1e-12


def torch_l2norm(values, dim=-1, eps=1e-8):
    norm = torch.pow(values, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return torch.div(values, norm)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize and quantify IDIN alignment evolution."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--data_name", default=None)
    parser.add_argument("--bert_path", default=None)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--caption_index", type=int, nargs="+")
    selection.add_argument("--all", action="store_true",
                           help="Compute metrics for every caption in the split.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap used with --all, useful for a dry run.")
    parser.add_argument("--visualize_limit", type=int, default=12,
                        help="Maximum number of figures/NPZ files when --all is used.")
    parser.add_argument("--output_dir", default="./idin_stage_analysis")
    parser.add_argument(
        "--ground_truth_json", default=None,
        help=("Optional manual word-region labels. JSON maps caption index to token "
              "index -> acceptable zero-based region indices."),
    )
    parser.add_argument("--images_root", default=None,
                        help="Root containing original images for box overlays.")
    parser.add_argument("--id_mapping", default=None,
                        help="JSON mapping dataset image ids to image filenames.")
    parser.add_argument(
        "--image_ids_file", default=None,
        help=("Optional split ids file (for example test_ids.txt). It converts the "
              "precomputed array index into the dataset id used by id_mapping.json."),
    )
    parser.add_argument("--boxes_file", default=None,
                        help="NPY/NPZ boxes shaped [images, regions, 4].")
    parser.add_argument("--token_index", type=int, default=None,
                        help="Optional zero-based tokenizer position to visualize.")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def decode_caption(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _attention(query, context, matrix, smooth):
    """Return the exact attention distribution used by model.cross_attention."""
    matrix = matrix.to(query.device)
    scaled_query = torch.mul(query, matrix)
    logits = torch.bmm(context, scaled_query.transpose(1, 2))
    logits = F.leaky_relu(logits, negative_slope=0.1)
    logits = torch_l2norm(logits, dim=-1)
    return F.softmax(logits.transpose(1, 2).contiguous() * smooth, dim=2)


def _weighted_context(attention, context):
    result = torch.bmm(attention, context)
    return torch_l2norm(result, dim=-1)


def trace_idin(model, opt, image_features, caption_ids):
    """Replay one matched pair and return stage attention and state tensors.

    Returned attentions have shape [stage, word, region]. The correction
    protocol intentionally requires a t2i checkpoint because its softmax is a
    probability distribution over regions for each word.
    """
    if opt.attn_type != "t2i":
        raise ValueError(
            "IDIN word-to-region analysis requires opt.attn_type='t2i'. "
            "An i2t softmax is normalized over words for each region and cannot "
            "be transposed and interpreted as the same probability distribution."
        )
    image_tensor = image_features.unsqueeze(0)
    caption_tensor = torch.tensor(caption_ids, dtype=torch.long).unsqueeze(0)
    lengths = [len(caption_ids)]

    with torch.no_grad():
        image_emb, text_emb, _ = model.forward_emb(
            image_tensor, caption_tensor, lengths
        )
        sim_enc = model.sim_enc
        image_global = sim_enc.v_global_w(image_emb, torch.mean(image_emb, 1))
        caption = text_emb[:, :lengths[0], :]
        caption_guide = caption.mean(dim=1)
        adapted_image = sim_enc.adapt_txt(
            image_emb.permute(0, 2, 1), image_global, caption_guide, None
        ).permute(0, 2, 1)

        query, context = caption, adapted_image

        smooth = opt.t2i_smooth if opt.attn_type == "t2i" else opt.i2t_smooth
        matrix = torch.ones(sim_enc.embed_dim, device=query.device)
        attentions = []
        contexts = []
        matrices = []
        temperatures = []
        stage_scores = []
        sim_high = None

        for stage, aggregation_module in enumerate(sim_enc.rar_modules):
            attention = _attention(query, context, matrix, smooth)
            attended_context = _weighted_context(attention, context)
            sim_mid = sim_enc.alv_modules[stage](query, context, matrix, smooth)
            if stage == 0:
                sim_high = torch.mean(sim_mid, 1)

            matrices.append(matrix.detach().cpu().numpy())
            if torch.is_tensor(smooth):
                temperatures.append(smooth.detach().cpu().numpy())
            else:
                temperatures.append(np.asarray(smooth, dtype=np.float32))

            word_region = attention[0]
            word_context = attended_context[0]
            attentions.append(word_region.detach().cpu().numpy())
            contexts.append(word_context.detach().cpu().numpy())

            if stage < len(sim_enc.rcr_modules):
                matrix, smooth = sim_enc.rcr_modules[stage](sim_mid, matrix, smooth)
            sim_high = aggregation_module(sim_mid, sim_high)
            score = sim_enc.sigmoid(sim_enc.sim_eval_w(sim_high))
            stage_scores.append(float(score[0, 0].detach().cpu().item()))

    return {
        "attention": np.stack(attentions, axis=0),
        "context": np.stack(contexts, axis=0),
        "matrix": matrices,
        "temperature": temperatures,
        "score": np.asarray(stage_scores, dtype=np.float32),
        "adapted_image": adapted_image[0].detach().cpu().numpy(),
        "text": caption[0].detach().cpu().numpy(),
    }


def _normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), EPS)


def js_divergence(p, q):
    """Row-wise Jensen-Shannon divergence in bits, bounded by [0, 1]."""
    p = _normalize_rows(p)
    q = _normalize_rows(q)
    midpoint = 0.5 * (p + q)
    kl_p = np.sum(np.where(p > 0, p * np.log2((p + EPS) / (midpoint + EPS)), 0), axis=-1)
    kl_q = np.sum(np.where(q > 0, q * np.log2((q + EPS) / (midpoint + EPS)), 0), axis=-1)
    return 0.5 * (kl_p + kl_q)


def normalized_entropy(attention):
    attention = _normalize_rows(attention)
    region_count = attention.shape[-1]
    entropy = -np.sum(
        np.where(attention > 0, attention * np.log(attention + EPS), 0), axis=-1
    )
    return entropy / max(math.log(max(region_count, 2)), EPS)


def topk_overlap(previous, current, k):
    k = min(k, previous.shape[-1])
    previous_top = np.argpartition(previous, -k, axis=-1)[:, -k:]
    current_top = np.argpartition(current, -k, axis=-1)[:, -k:]
    values = []
    for left, right in zip(previous_top, current_top):
        values.append(len(set(left.tolist()).intersection(right.tolist())) / float(k))
    return np.asarray(values)


def context_cosine_distance(previous, current):
    numerator = np.sum(previous * current, axis=-1)
    denominator = np.linalg.norm(previous, axis=-1) * np.linalg.norm(current, axis=-1)
    return 1.0 - numerator / np.maximum(denominator, EPS)


def compute_label_free_metrics(attention, context, valid_tokens, top_k=3):
    """Compute per-transition and per-stage metrics without semantic labels."""
    valid_tokens = np.asarray(valid_tokens, dtype=np.int64)
    rows = []
    for stage in range(attention.shape[0]):
        stage_attention = attention[stage, valid_tokens]
        row = {
            "stage": stage,
            "mean_normalized_entropy": float(normalized_entropy(stage_attention).mean()),
            "mean_peak_confidence": float(stage_attention.max(axis=-1).mean()),
        }
        if stage == 0:
            row.update({
                "js_from_previous": None,
                "top1_switch_rate": None,
                "topk_overlap": None,
                "context_cosine_step": None,
            })
        else:
            previous = attention[stage - 1, valid_tokens]
            row.update({
                "js_from_previous": float(js_divergence(previous, stage_attention).mean()),
                "top1_switch_rate": float(
                    np.mean(previous.argmax(axis=-1) != stage_attention.argmax(axis=-1))
                ),
                "topk_overlap": float(topk_overlap(previous, stage_attention, top_k).mean()),
                "context_cosine_step": float(context_cosine_distance(
                    context[stage - 1, valid_tokens], context[stage, valid_tokens]
                ).mean()),
            })
        rows.append(row)
    return rows


def compute_ground_truth_metrics(attention, ground_truth):
    """Compute accuracy, correction, and regression from manual region labels."""
    labelled_tokens = sorted(int(key) for key in ground_truth.keys())
    if not labelled_tokens:
        return [], {}
    correct = np.zeros((attention.shape[0], len(labelled_tokens)), dtype=bool)
    predictions = attention[:, labelled_tokens].argmax(axis=-1)
    for column, token_index in enumerate(labelled_tokens):
        accepted = set(int(value) for value in ground_truth[str(token_index)])
        correct[:, column] = np.asarray(
            [int(prediction) in accepted for prediction in predictions[:, column]]
        )

    stage_rows = []
    for stage in range(attention.shape[0]):
        stage_rows.append({
            "stage": stage,
            "alignment_accuracy": float(correct[stage].mean()),
        })

    initially_wrong = ~correct[0]
    initially_correct = correct[0]
    corrected = initially_wrong & correct[-1]
    regressed = initially_correct & ~correct[-1]
    ever_correct = np.any(correct, axis=0)
    later_wrong = np.asarray([
        np.any(~correct[first_correct + 1:, column])
        if np.any(correct[:, column]) else False
        for column, first_correct in enumerate(np.argmax(correct, axis=0))
    ])
    summary = {
        "labelled_token_count": int(len(labelled_tokens)),
        "initially_wrong_count": int(initially_wrong.sum()),
        "initially_correct_count": int(initially_correct.sum()),
        "corrected_count": int(corrected.sum()),
        "regressed_count": int(regressed.sum()),
        "ever_correct_count": int(ever_correct.sum()),
        "drift_after_correct_count": int((ever_correct & later_wrong).sum()),
        "stage_correct_counts": [int(value) for value in correct.sum(axis=1)],
        "stage_label_counts": [int(len(labelled_tokens))] * attention.shape[0],
        "early_error_correction_rate": (
            float(corrected.sum() / float(initially_wrong.sum()))
            if initially_wrong.any() else None
        ),
        "regression_rate": (
            float(regressed.sum() / float(initially_correct.sum()))
            if initially_correct.any() else None
        ),
        "drift_after_correct_rate": (
            float((ever_correct & later_wrong).sum() / float(ever_correct.sum()))
            if ever_correct.any() else None
        ),
        "corrected_token_indices": [
            labelled_tokens[i] for i in np.where(corrected)[0].tolist()
        ],
        "regressed_token_indices": [
            labelled_tokens[i] for i in np.where(regressed)[0].tolist()
        ],
    }
    return stage_rows, summary


def load_ground_truth(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    return loaded


def load_boxes(path, image_id, num_regions):
    if path is None:
        return None
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        boxes = loaded["boxes"] if "boxes" in loaded.files else loaded[loaded.files[0]]
    else:
        boxes = loaded
    if boxes.ndim == 3 or (boxes.ndim == 1 and boxes.dtype == object):
        boxes = boxes[image_id]
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 4 or boxes.shape[0] < num_regions:
        raise ValueError("Boxes must contain at least [num_regions, 4] entries.")
    return boxes[:num_regions, :4]


def resolve_image(images_root, id_mapping_path, image_id, image_ids_file=None):
    if images_root is None or id_mapping_path is None:
        return None
    with open(id_mapping_path, "r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    lookup_id = image_id
    if image_ids_file is not None:
        with open(image_ids_file, "r", encoding="utf-8") as stream:
            split_ids = [line.strip() for line in stream if line.strip()]
        if image_id >= len(split_ids):
            raise IndexError("Image index {} is outside {}.".format(image_id, image_ids_file))
        lookup_id = split_ids[image_id]
    if isinstance(mapping, list):
        filename = mapping[int(lookup_id)]
    else:
        filename = mapping.get(str(lookup_id), mapping.get(lookup_id))
    if filename is None:
        raise KeyError("Image id {} is absent from {}.".format(lookup_id, id_mapping_path))
    if isinstance(filename, dict):
        filename = filename.get("file_name", filename.get("filename"))
    return str(Path(images_root) / str(filename))


def prepare_boxes(boxes, width, height):
    boxes = boxes.copy()
    if np.nanmax(np.abs(boxes)) <= 2.0:
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def choose_token(attention, valid_tokens, ground_truth, requested):
    if requested is not None:
        if requested not in valid_tokens:
            raise ValueError("--token_index selects a special/padded token.")
        return requested, "requested"
    if ground_truth:
        _, summary = compute_ground_truth_metrics(attention, ground_truth)
        if summary.get("corrected_token_indices"):
            return summary["corrected_token_indices"][0], "ground-truth corrected"
    valid = np.asarray(valid_tokens, dtype=np.int64)
    changed = attention[0, valid].argmax(axis=-1) != attention[-1, valid].argmax(axis=-1)
    if changed.any():
        candidates = valid[changed]
        gains = attention[-1, candidates].max(axis=-1) - attention[0, candidates].max(axis=-1)
        return int(candidates[np.argmax(gains)]), "unverified changed-alignment candidate"
    entropy_drop = normalized_entropy(attention[0, valid]) - normalized_entropy(attention[-1, valid])
    return int(valid[np.argmax(entropy_drop)]), "unverified confidence-refinement candidate"


def draw_stage_panel(ax, image_path, boxes, weights, title, color):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ax.imshow(image)
    ax.axis("off")
    boxes = prepare_boxes(boxes, width, height)
    top_regions = np.argsort(weights)[::-1][:3]
    for rank, region in enumerate(top_regions):
        x1, y1, x2, y2 = boxes[region]
        alpha = 1.0 if rank == 0 else 0.45
        linewidth = 2.5 if rank == 0 else 1.2
        rect = patches.Rectangle(
            (x1, y1), max(x2 - x1, 1), max(y2 - y1, 1), fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha,
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1, "r{} {:.3f}".format(region, weights[region]),
            fontsize=6, color="white",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=1),
        )
    ax.set_title(title, fontsize=9)


def save_visualization(path, caption, token, token_index, reason, trace,
                       image_path, boxes, dpi):
    stages = trace["attention"].shape[0]
    figure, axes = plt.subplots(2, stages + 1, figsize=(3.2 * (stages + 1), 6.1))
    if image_path and boxes is not None:
        image = Image.open(image_path).convert("RGB")
        axes[0, 0].imshow(image)
        axes[0, 0].axis("off")
        axes[0, 0].set_title("Input image", fontsize=9)
        for stage in range(stages):
            draw_stage_panel(
                axes[0, stage + 1], image_path, boxes,
                trace["attention"][stage, token_index],
                "{}\n{}({:.3f})".format(
                    "Initial" if stage == 0 else "Stage {}".format(stage),
                    token, trace["attention"][stage, token_index].max(),
                ), "#d62728",
            )
    else:
        axes[0, 0].axis("off")
        axes[0, 0].text(0.5, 0.5, "Original image/boxes\nnot supplied",
                        ha="center", va="center")
        for stage in range(stages):
            weights = trace["attention"][stage, token_index]
            axes[0, stage + 1].bar(np.arange(len(weights)), weights, color="#2a6fbb")
            axes[0, stage + 1].set_title(
                "{}: {}({:.3f})".format(
                    "Initial" if stage == 0 else "Stage {}".format(stage),
                    token, weights.max(),
                ), fontsize=9,
            )
            axes[0, stage + 1].set_xlabel("Region index")
            axes[0, stage + 1].set_ylim(0, max(0.05, trace["attention"][:, token_index].max() * 1.1))

    heat = trace["attention"][:, token_index, :]
    image_handle = axes[1, 0].imshow(heat, aspect="auto", cmap="viridis")
    axes[1, 0].set_yticks(np.arange(stages))
    axes[1, 0].set_yticklabels(["Initial"] + ["Stage {}".format(i) for i in range(1, stages)])
    axes[1, 0].set_xlabel("Region index")
    axes[1, 0].set_title("Alignment trajectory")
    figure.colorbar(image_handle, ax=axes[1, 0], fraction=0.046, pad=0.04)

    top1 = trace["attention"][:, token_index].argmax(axis=-1)
    peak = trace["attention"][:, token_index].max(axis=-1)
    axes[1, 1].plot(np.arange(stages), top1, marker="o", color="#d62728")
    axes[1, 1].set_xticks(np.arange(stages))
    axes[1, 1].set_xlabel("IDIN stage")
    axes[1, 1].set_ylabel("Top-1 region")
    axes[1, 1].set_title("Top-1 path")

    axes[1, 2].plot(np.arange(stages), peak, marker="o", label="peak attention")
    axes[1, 2].plot(np.arange(stages), trace["score"], marker="s", label="pair score")
    axes[1, 2].set_xticks(np.arange(stages))
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xlabel("IDIN stage")
    axes[1, 2].set_title("Confidence evolution")
    axes[1, 2].legend(fontsize=7)

    for column in range(3, stages + 1):
        axes[1, column].axis("off")
    figure.suptitle(
        "{}\nToken {}: '{}' - {}".format(caption, token_index, token, reason),
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(str(path), "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    # Keep metric utilities importable in lightweight environments that do not
    # have the legacy BERT package required by the trained CSAN model.
    from data import PrecompDataset, get_tokenizer
    from model import CSAN

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint)
    if "opt" not in checkpoint or "model" not in checkpoint:
        raise KeyError("Expected checkpoint keys 'opt' and 'model'.")
    opt = checkpoint["opt"]
    if args.data_path is not None:
        opt.data_path = args.data_path
    if args.data_name is not None:
        opt.data_name = args.data_name
    if args.bert_path is not None:
        opt.bert_path = args.bert_path

    tokenizer = get_tokenizer(opt.bert_path)
    dataset = PrecompDataset(os.path.join(opt.data_path, opt.data_name), args.split, tokenizer)
    model = CSAN(opt)
    model.load_state_dict(checkpoint["model"])
    model.val_start()
    ground_truth_all = load_ground_truth(args.ground_truth_json)

    if args.all:
        caption_indices = list(range(len(dataset)))
        if args.max_samples is not None:
            caption_indices = caption_indices[:args.max_samples]
    else:
        caption_indices = args.caption_index

    aggregate_rows = []
    correction_summaries = []
    for sample_number, caption_index in enumerate(caption_indices):
        image_features, caption_ids, _, image_id = dataset[caption_index]
        tokens = tokenizer.convert_ids_to_tokens(caption_ids)
        valid_tokens = [
            index for index, token in enumerate(tokens) if token not in SPECIAL_TOKENS
        ]
        trace = trace_idin(model, opt, image_features, caption_ids)
        ground_truth = ground_truth_all.get(str(caption_index), {})
        label_free = compute_label_free_metrics(
            trace["attention"], trace["context"], valid_tokens, args.top_k
        )
        gt_rows, correction = compute_ground_truth_metrics(trace["attention"], ground_truth)
        gt_by_stage = {row["stage"]: row for row in gt_rows}
        for row in label_free:
            row.update(gt_by_stage.get(row["stage"], {}))
            row["caption_index"] = caption_index
            row["image_id"] = image_id
            aggregate_rows.append(row)
        if correction:
            correction["caption_index"] = caption_index
            correction["image_id"] = image_id
            correction_summaries.append(correction)

        should_visualize = (not args.all) or sample_number < args.visualize_limit
        if should_visualize:
            token_index, reason = choose_token(
                trace["attention"], valid_tokens, ground_truth, args.token_index
            )
            image_path = resolve_image(
                args.images_root, args.id_mapping, image_id, args.image_ids_file
            )
            boxes = load_boxes(args.boxes_file, image_id, trace["attention"].shape[-1])
            stem = "caption_{:06d}".format(caption_index)
            save_visualization(
                output_dir / (stem + ".png"), decode_caption(dataset.captions[caption_index]),
                tokens[token_index], token_index, reason, trace, image_path, boxes, args.dpi,
            )
            np.savez_compressed(
                str(output_dir / (stem + ".npz")),
                attention=trace["attention"], context=trace["context"],
                score=trace["score"], tokens=np.asarray(tokens),
            )
            metadata = {
                "caption_index": caption_index,
                "image_id": image_id,
                "caption": decode_caption(dataset.captions[caption_index]),
                "tokens": tokens,
                "visualized_token_index": token_index,
                "selection_reason": reason,
                "has_ground_truth": bool(ground_truth),
                "ground_truth_summary": correction,
            }
            with (output_dir / (stem + ".json")).open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2, ensure_ascii=False)

    write_csv(output_dir / "stage_metrics.csv", aggregate_rows)
    stage_summary = []
    stage_ids = sorted(set(row["stage"] for row in aggregate_rows))
    excluded = {"stage", "caption_index", "image_id"}
    for stage in stage_ids:
        stage_rows = [row for row in aggregate_rows if row["stage"] == stage]
        summary_row = {"stage": stage, "sample_count": len(stage_rows)}
        metric_keys = sorted(set().union(*(row.keys() for row in stage_rows)) - excluded)
        for key in metric_keys:
            values = [row.get(key) for row in stage_rows if row.get(key) is not None]
            if values:
                summary_row[key] = float(np.mean(values))
        stage_summary.append(summary_row)

    correction_totals = {}
    if correction_summaries:
        count_keys = [
            "labelled_token_count", "initially_wrong_count", "initially_correct_count",
            "corrected_count", "regressed_count", "ever_correct_count",
            "drift_after_correct_count",
        ]
        correction_totals = {
            key: int(sum(item[key] for item in correction_summaries)) for key in count_keys
        }
        stage_count = len(correction_summaries[0]["stage_correct_counts"])
        stage_correct_counts = [
            int(sum(item["stage_correct_counts"][stage] for item in correction_summaries))
            for stage in range(stage_count)
        ]
        stage_label_counts = [
            int(sum(item["stage_label_counts"][stage] for item in correction_summaries))
            for stage in range(stage_count)
        ]
        correction_totals.update({
            "stage_correct_counts": stage_correct_counts,
            "stage_label_counts": stage_label_counts,
            "stage_alignment_accuracy": [
                correct_count / float(label_count) if label_count else None
                for correct_count, label_count in zip(
                    stage_correct_counts, stage_label_counts
                )
            ],
            "early_error_correction_rate": (
                correction_totals["corrected_count"] /
                float(correction_totals["initially_wrong_count"])
                if correction_totals["initially_wrong_count"] else None
            ),
            "regression_rate": (
                correction_totals["regressed_count"] /
                float(correction_totals["initially_correct_count"])
                if correction_totals["initially_correct_count"] else None
            ),
            "drift_after_correct_rate": (
                correction_totals["drift_after_correct_count"] /
                float(correction_totals["ever_correct_count"])
                if correction_totals["ever_correct_count"] else None
            ),
        })
    final_summary = {
        "evaluated_caption_count": len(caption_indices),
        "stage_metrics_macro_average": stage_summary,
        "ground_truth_micro_average": correction_totals,
        "per_caption_ground_truth": correction_summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(final_summary, stream, indent=2, ensure_ascii=False)
    print("Saved IDIN analysis to {}".format(output_dir.resolve()))
    if not ground_truth_all:
        print("No ground truth supplied: correction candidates are unverified and must not be reported as corrected errors.")


if __name__ == "__main__":
    main()

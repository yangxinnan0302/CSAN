"""Visualize how ITE and ASM change structural-semantic representations.

The script uses one matched image-caption pair at a time and produces a
publication-ready overview figure together with the underlying numerical
arrays.  It works with the precomputed SCAN-style region features used by
CSAN.  An original image and its Faster R-CNN boxes are optional.
"""

from __future__ import print_function

import argparse
import json
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image
import torch

from data import PrecompDataset, get_tokenizer
from model import CSAN


SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ITE/ASM interpretability visualizations for CSAN."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model_best.pth.tar (or another CSAN checkpoint).")
    parser.add_argument("--data_path", default=None,
                        help="Dataset root; overrides the value stored in the checkpoint.")
    parser.add_argument("--data_name", default=None,
                        help="Dataset folder, e.g. f30k_precomp or coco_precomp.")
    parser.add_argument("--bert_path", default=None,
                        help="BERT folder containing vocab.txt; overrides the checkpoint path.")
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--caption_index", type=int, nargs="+", default=[0],
                        help="One or more caption indices from the selected split.")
    parser.add_argument("--output_dir", default="./gafm_visualizations")
    parser.add_argument("--image", default=None,
                        help="Optional original image (supported for one caption index).")
    parser.add_argument("--boxes_file", default=None,
                        help="Optional .npy/.npz boxes in [x1,y1,x2,y2] format.")
    parser.add_argument("--top_edges", type=int, default=18,
                        help="Number of strongest ITE region-dependency edges to draw.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_checkpoint(path):
    """Load legacy checkpoints on both old and recent PyTorch releases."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def decode_caption(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def cosine_matrix(regions, words, eps=1e-8):
    regions = regions / np.maximum(np.linalg.norm(regions, axis=1, keepdims=True), eps)
    words = words / np.maximum(np.linalg.norm(words, axis=1, keepdims=True), eps)
    return np.matmul(regions, words.T)


def load_boxes(path, image_id, num_regions):
    if path is None:
        return None

    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "boxes" in loaded.files:
            boxes = loaded["boxes"]
        elif len(loaded.files) == 1:
            boxes = loaded[loaded.files[0]]
        else:
            raise ValueError("The .npz file must contain a 'boxes' array.")
    else:
        boxes = loaded

    if boxes.ndim == 3:
        boxes = boxes[image_id]
    elif boxes.ndim == 1 and boxes.dtype == object:
        boxes = boxes[image_id]
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError("Boxes must have shape [num_regions, 4] or [num_images, num_regions, 4].")
    if boxes.shape[0] < num_regions:
        raise ValueError("The selected boxes array has fewer entries than the region features.")
    return boxes[:num_regions, :4]


def prepare_boxes(boxes, width, height):
    boxes = boxes.copy()
    if np.nanmax(np.abs(boxes)) <= 2.0:
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def draw_region_overlay(ax, image_path, boxes, attention, modulation, top_edges):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ax.imshow(image)
    ax.axis("off")

    if boxes is None:
        ax.set_title("Input image (boxes not provided)")
        return

    boxes = prepare_boxes(boxes, width, height)
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0,
                               (boxes[:, 1] + boxes[:, 3]) / 2.0))
    norm = plt.Normalize(vmin=float(np.min(modulation)),
                         vmax=float(np.max(modulation)) + 1e-8)
    cmap = plt.get_cmap("magma")

    relation = (attention + attention.T) / 2.0
    upper_i, upper_j = np.triu_indices(relation.shape[0], k=1)
    order = np.argsort(relation[upper_i, upper_j])[::-1][:top_edges]
    edge_values = relation[upper_i[order], upper_j[order]]
    edge_min = float(edge_values.min()) if edge_values.size else 0.0
    edge_span = float(edge_values.max() - edge_min) if edge_values.size else 1.0
    for i, j, value in zip(upper_i[order], upper_j[order], edge_values):
        strength = (float(value) - edge_min) / (edge_span + 1e-8)
        ax.plot([centers[i, 0], centers[j, 0]],
                [centers[i, 1], centers[j, 1]],
                color="cyan", linewidth=0.6 + 2.0 * strength,
                alpha=0.25 + 0.55 * strength, zorder=2)

    for region_id, (box, score) in enumerate(zip(boxes, modulation), start=1):
        x1, y1, x2, y2 = box
        color = cmap(norm(float(score)))
        rect = patches.Rectangle((x1, y1), max(x2 - x1, 1), max(y2 - y1, 1),
                                 fill=False, edgecolor=color, linewidth=1.4, zorder=3)
        ax.add_patch(rect)
        ax.text(x1, y1, str(region_id), color="white", fontsize=6,
                bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=0.8),
                zorder=4)
    ax.set_title("ITE structure edges + ASM region strength")


def draw_region_bar(ax, modulation):
    region_ids = np.arange(1, len(modulation) + 1)
    colors = plt.get_cmap("magma")(
        plt.Normalize(float(np.min(modulation)), float(np.max(modulation)) + 1e-8)(modulation)
    )
    ax.bar(region_ids, modulation, color=colors, width=0.85)
    ax.set_xlabel("Region index")
    ax.set_ylabel(r"$\|\Delta v_i\|_2 / \|v_i\|_2$")
    ax.set_title("ASM relative modulation by region")
    ax.grid(axis="y", alpha=0.2)


def draw_heatmap(ax, matrix, title, xlabels=None, ylabels=None,
                 cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)

    if xlabels is not None:
        step = max(1, int(np.ceil(len(xlabels) / 18.0)))
        ticks = np.arange(0, len(xlabels), step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([xlabels[i] for i in ticks], rotation=55,
                           ha="right", fontsize=7)
    else:
        ax.set_xlabel("Key region")

    if ylabels is not None:
        step = max(1, int(np.ceil(len(ylabels) / 18.0)))
        ticks = np.arange(0, len(ylabels), step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([ylabels[i] for i in ticks], fontsize=7)
    else:
        step = max(1, matrix.shape[0] // 9)
        ticks = np.arange(0, matrix.shape[0], step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(i + 1) for i in ticks], fontsize=7)
        ax.set_ylabel("Query region")

    return im


def export_visualization(model, opt, dataset, tokenizer, caption_index, args):
    if caption_index < 0 or caption_index >= len(dataset):
        raise IndexError("caption_index {} is outside [0, {}).".format(
            caption_index, len(dataset)))

    image_features, caption_ids, _, image_id = dataset[caption_index]
    caption_tensor = torch.tensor(caption_ids, dtype=torch.long).unsqueeze(0)
    image_tensor = image_features.unsqueeze(0)
    caption_length = len(caption_ids)

    model.enable_visualization(True)
    with torch.no_grad():
        image_after_ite, text_after_ite, lengths = model.forward_emb(
            image_tensor, caption_tensor, [caption_length]
        )
        model.forward_sim(opt, image_after_ite, text_after_ite, lengths)
    cache = model.get_visualization_cache(cpu=True)

    required = {
        "image_before_ite", "text_before_ite", "image_after_ite",
        "text_after_ite", "ite_image_attention", "ite_text_attention",
        "asm_input", "asm_output", "asm_delta", "asm_gamma", "asm_beta",
        "similarity_score",
    }
    missing = sorted(required.difference(cache))
    if missing:
        raise RuntimeError("Visualization hooks did not record: {}".format(
            ", ".join(missing)))

    all_tokens = tokenizer.convert_ids_to_tokens(caption_ids)
    keep = [i for i, token in enumerate(all_tokens) if token not in SPECIAL_TOKENS]
    if not keep:
        keep = list(range(len(all_tokens)))
    tokens = [all_tokens[i] for i in keep]

    image_before = cache["image_before_ite"][0].numpy()
    text_before = cache["text_before_ite"][0].numpy()[keep]
    image_ite = cache["image_after_ite"][0].numpy()
    text_ite = cache["text_after_ite"][0].numpy()[keep]
    image_asm = cache["asm_output"][0].numpy().T

    image_attention_heads = cache["ite_image_attention"][0].numpy()
    text_attention_heads_full = cache["ite_text_attention"][0].numpy()
    image_attention = image_attention_heads.mean(axis=0)
    text_attention_full = text_attention_heads_full.mean(axis=0)
    text_attention = text_attention_full[np.ix_(keep, keep)]
    text_attention_heads = text_attention_heads_full[:, keep][:, :, keep]

    asm_input = cache["asm_input"][0].numpy().T
    asm_delta = cache["asm_delta"][0].numpy().T
    modulation_absolute = np.linalg.norm(asm_delta, axis=1)
    modulation = modulation_absolute / np.maximum(
        np.linalg.norm(asm_input, axis=1), 1e-8
    )
    gamma = cache["asm_gamma"][0].numpy().reshape(-1)
    beta = cache["asm_beta"][0].numpy().reshape(-1)

    similarity_before = cosine_matrix(image_before, text_before)
    similarity_ite = cosine_matrix(image_ite, text_ite)
    similarity_gafm = cosine_matrix(image_asm, text_ite)
    similarity_delta = similarity_gafm - similarity_before

    boxes = load_boxes(args.boxes_file, image_id, image_before.shape[0])
    raw_caption = decode_caption(dataset.captions[caption_index])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "gafm_caption_{:05d}".format(caption_index)

    figure, axes = plt.subplots(2, 4, figsize=(21, 10.5), constrained_layout=True)
    if args.image is not None:
        draw_region_overlay(axes[0, 0], args.image, boxes, image_attention,
                            modulation, args.top_edges)
    else:
        draw_region_bar(axes[0, 0], modulation)

    im = draw_heatmap(axes[0, 1], image_attention,
                      "ITE visual dependency attention")
    figure.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.03)

    im = draw_heatmap(axes[0, 2], text_attention,
                      "ITE textual dependency attention", tokens, tokens)
    figure.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.03)

    axes[0, 3].hist(gamma, bins=35, alpha=0.70, label=r"$\gamma$", color="#2878b5")
    axes[0, 3].hist(beta, bins=35, alpha=0.60, label=r"$\beta$", color="#d95f02")
    axes[0, 3].set_title("ASM channel-wise modulation parameters")
    axes[0, 3].set_xlabel("Parameter value")
    axes[0, 3].set_ylabel("Frequency")
    axes[0, 3].legend(frameon=False)

    sim_min = min(float(similarity_before.min()), float(similarity_ite.min()),
                  float(similarity_gafm.min()))
    sim_max = max(float(similarity_before.max()), float(similarity_ite.max()),
                  float(similarity_gafm.max()))
    sim_titles = ("Before ITE", "After ITE", "After ITE + ASM (GAFM)")
    sim_values = (similarity_before, similarity_ite, similarity_gafm)
    for column, (title, values) in enumerate(zip(sim_titles, sim_values)):
        im = draw_heatmap(axes[1, column], values,
                          title + "\nregion-word cosine similarity",
                          tokens, None, cmap="coolwarm", vmin=sim_min, vmax=sim_max)
        axes[1, column].set_xlabel("Caption token")
        figure.colorbar(im, ax=axes[1, column], fraction=0.046, pad=0.03)

    delta_limit = max(abs(float(similarity_delta.min())),
                      abs(float(similarity_delta.max())), 1e-8)
    im = draw_heatmap(axes[1, 3], similarity_delta,
                      "GAFM-induced similarity change\n(after ITE + ASM) - before ITE",
                      tokens, None, cmap="RdBu_r", vmin=-delta_limit, vmax=delta_limit)
    axes[1, 3].set_xlabel("Caption token")
    figure.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.03)

    score = float(cache["similarity_score"].reshape(-1)[0])
    title = "Caption {} | image {} | score {:.4f}\n{}".format(
        caption_index, image_id, score, textwrap.fill(raw_caption, width=130)
    )
    figure.suptitle(title, fontsize=12)
    figure_path = output_dir / (stem + ".png")
    figure.savefig(str(figure_path), dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    arrays_path = output_dir / (stem + ".npz")
    np.savez_compressed(
        str(arrays_path),
        ite_image_attention=image_attention,
        ite_image_attention_heads=image_attention_heads,
        ite_text_attention=text_attention,
        ite_text_attention_heads=text_attention_heads,
        asm_region_relative_modulation=modulation,
        asm_region_absolute_modulation=modulation_absolute,
        asm_gamma=gamma,
        asm_beta=beta,
        similarity_before_ite=similarity_before,
        similarity_after_ite=similarity_ite,
        similarity_after_gafm=similarity_gafm,
        similarity_change=similarity_delta,
        tokens=np.asarray(tokens),
    )

    metadata = {
        "caption_index": int(caption_index),
        "image_id": int(image_id),
        "caption": raw_caption,
        "tokens": tokens,
        "similarity_score": score,
        "checkpoint": os.path.abspath(args.checkpoint),
        "figure": str(figure_path.resolve()),
        "arrays": str(arrays_path.resolve()),
    }
    metadata_path = output_dir / (stem + ".json")
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    print("Saved {}".format(figure_path))
    print("Saved {}".format(arrays_path))
    print("Saved {}".format(metadata_path))


def main():
    args = parse_args()
    if args.image is not None and len(args.caption_index) != 1:
        raise ValueError("--image can only be used with one --caption_index value.")

    checkpoint = load_checkpoint(args.checkpoint)
    if "opt" not in checkpoint or "model" not in checkpoint:
        raise KeyError("Expected a training checkpoint containing 'opt' and 'model'.")
    opt = checkpoint["opt"]
    if args.data_path is not None:
        opt.data_path = args.data_path
    if args.data_name is not None:
        opt.data_name = args.data_name
    if args.bert_path is not None:
        opt.bert_path = args.bert_path

    tokenizer = get_tokenizer(opt.bert_path)
    dataset_path = os.path.join(opt.data_path, opt.data_name)
    dataset = PrecompDataset(dataset_path, args.split, tokenizer)

    model = CSAN(opt)
    model.load_state_dict(checkpoint["model"])
    model.val_start()
    print("Loaded checkpoint from epoch {}".format(checkpoint.get("epoch", "unknown")))

    for caption_index in args.caption_index:
        export_visualization(model, opt, dataset, tokenizer, caption_index, args)


if __name__ == "__main__":
    main()

"""Fixed-image positive/hard-negative text visualization for CSAN GAFM.

Example pair used in this script:
  positive caption index: 2140
  negative caption index: 2194

The image belonging to the positive caption is fixed. The negative caption's
original image is deliberately ignored. Positive and negative captions are
evaluated in separate forward passes because ASM is text-guided and the
visualization cache records the most recent ASM output.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import re
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
from model_copy import CSAN
from visualize_gafm import (
    SPECIAL_TOKENS,
    decode_caption,
    load_boxes,
    load_checkpoint,
    prepare_boxes,
    resolve_original_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a positive caption and a hard-negative caption against "
            "the same fixed image."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--data_name", default=None)
    parser.add_argument("--bert_path", default=None)
    parser.add_argument(
        "--split", default="test", choices=("train", "dev", "test")
    )
    parser.add_argument(
        "--caption_index",
        type=int,
        required=True,
        help="Positive caption index. Its image is used as the fixed image.",
    )
    parser.add_argument(
        "--negative_caption_index",
        type=int,
        required=True,
        help=(
            "Hard-negative caption index. Its original image is ignored; "
            "the image selected by --caption_index remains fixed."
        ),
    )
    parser.add_argument(
        "--positive_color",
        default="orange",
        help="Color word in the positive caption.",
    )
    parser.add_argument(
        "--negative_color",
        default="white",
        help="Color word in the hard-negative caption.",
    )
    parser.add_argument(
        "--boxes_file",
        default=None,
        help=(
            "Optional .npy/.npz boxes with shape "
            "[num_images, num_regions, 4]."
        ),
    )
    parser.add_argument(
        "--annotations_csv",
        default=None,
        help=(
            "Optional flickr_annotations_30k.csv used to locate the original "
            "image from the positive caption."
        ),
    )
    parser.add_argument(
        "--positive_image",
        default=None,
        help="Optional explicit path to the fixed original image.",
    )
    parser.add_argument("--top_regions", type=int, default=8)
    parser.add_argument("--output_dir", default="./gafm_text_pair_results")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def cosine_matrix(regions, words, eps=1e-8):
    regions = regions / np.maximum(
        np.linalg.norm(regions, axis=1, keepdims=True), eps
    )
    words = words / np.maximum(
        np.linalg.norm(words, axis=1, keepdims=True), eps
    )
    return np.matmul(regions, words.T)


def normalize_caption(value):
    """Build a punctuation-insensitive caption key for CSV lookup."""
    value = str(value).strip().lower()
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def detect_csv_columns(fieldnames):
    normalized = {str(name).strip().lower(): name for name in fieldnames or []}
    image_candidates = (
        "image_name",
        "filename",
        "file_name",
        "image",
        "image_id",
    )
    caption_candidates = (
        "comment",
        "caption",
        "sentence",
        "raw",
        "description",
    )
    image_column = next(
        (normalized[name] for name in image_candidates if name in normalized),
        None,
    )
    caption_column = next(
        (normalized[name] for name in caption_candidates if name in normalized),
        None,
    )
    if image_column is None or caption_column is None:
        raise ValueError(
            "Could not identify image/caption columns in annotations CSV. "
            "Columns found: {}".format(fieldnames)
        )
    return image_column, caption_column


def expand_caption_cell(value):
    """Return one or more captions stored in a CSV cell."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    value = str(value).strip()
    if not value:
        return []

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]

    return [value]


def find_raw_image(dataset_path, filename):
    dataset_path = Path(dataset_path)
    filename = Path(str(filename).strip()).name
    names = [filename]
    if Path(filename).suffix == "":
        names.extend((filename + ".jpg", filename + ".jpeg", filename + ".png"))

    image_roots = (
        dataset_path / "flickr30k-images",
        dataset_path / "flickr30k-images1",
        dataset_path / "images",
        dataset_path.parent / "flickr30k-images",
        dataset_path.parent / "flickr30k-images1",
        dataset_path.parent / "images",
    )
    for root in image_roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return str(candidate.resolve())

    for root in image_roots:
        if not root.is_dir():
            continue
        for name in names:
            match = next(root.rglob(name), None)
            if match is not None:
                return str(match.resolve())
    return None


def resolve_image_from_annotations(csv_path, dataset_path, caption):
    """Resolve the fixed original image by normalized caption matching."""
    target = normalize_caption(caption)
    matches = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        image_column, caption_column = detect_csv_columns(reader.fieldnames)
        for row in reader:
            captions = expand_caption_cell(row.get(caption_column, ""))
            if any(normalize_caption(item) == target for item in captions):
                filename = row.get(image_column, "")
                if filename and filename not in matches:
                    matches.append(filename)

    if not matches:
        raise KeyError(
            "Caption was not found in {}: {}".format(csv_path, caption)
        )
    if len(matches) > 1:
        raise ValueError(
            "Caption matched multiple images in {}: {}".format(
                csv_path, matches
            )
        )

    image_path = find_raw_image(dataset_path, matches[0])
    if image_path is None:
        raise FileNotFoundError(
            "CSV resolved image filename '{}', but it was not found under {}. "
            "Use --positive_image to provide it explicitly.".format(
                matches[0], dataset_path
            )
        )
    return image_path


def visible_token_positions(all_tokens):
    """Remove special tokens and legacy bytes-wrapper artifacts."""
    keep = [
        index
        for index, token in enumerate(all_tokens)
        if token not in SPECIAL_TOKENS
    ]
    visible = [all_tokens[index] for index in keep]
    quote_tokens = {"'", '"', "##'", '##"'}

    if (
        len(visible) >= 2
        and visible[0].lower() == "b"
        and visible[1] in quote_tokens
    ):
        keep = keep[2:]
        visible = visible[2:]
    if visible and visible[-1] in quote_tokens:
        keep = keep[:-1]
    return keep


def relative_modulation(cache):
    asm_input = cache["asm_input"].numpy().transpose(0, 2, 1)
    asm_delta = cache["asm_delta"].numpy().transpose(0, 2, 1)
    numerator = np.linalg.norm(asm_delta, axis=2)
    denominator = np.maximum(np.linalg.norm(asm_input, axis=2), 1e-8)
    return numerator / denominator


def evaluate_fixed_image_text(
    model, opt, tokenizer, image_feature, caption_ids
):
    """Evaluate one caption against the fixed image and copy cached outputs."""
    image_tensor = image_feature.float().unsqueeze(0)
    caption_tensor = torch.tensor(caption_ids, dtype=torch.long).unsqueeze(0)
    lengths = [len(caption_ids)]

    model.enable_visualization(True)
    with torch.no_grad():
        image_after, text_after, output_lengths = model.forward_emb(
            image_tensor, caption_tensor, lengths
        )
        model.forward_sim(opt, image_after, text_after, output_lengths)
    cache = model.get_visualization_cache(cpu=True)

    required = {
        "image_before_ite",
        "text_before_ite",
        "image_after_ite",
        "text_after_ite",
        "asm_input",
        "asm_output",
        "asm_delta",
        "similarity_score",
    }
    missing = sorted(required.difference(cache))
    if missing:
        raise RuntimeError(
            "Visualization cache is missing: {}".format(", ".join(missing))
        )

    all_tokens = tokenizer.convert_ids_to_tokens(caption_ids)
    keep = visible_token_positions(all_tokens)
    tokens = [all_tokens[index] for index in keep]
    if not tokens:
        raise RuntimeError("No displayable caption tokens were found.")

    image_before = cache["image_before_ite"][0].numpy()
    text_before = cache["text_before_ite"][0].numpy()[keep]
    image_ite = cache["image_after_ite"][0].numpy()
    text_ite = cache["text_after_ite"][0].numpy()[keep]
    image_gafm = cache["asm_output"][0].numpy().T

    matrices = {
        "Before ITE": cosine_matrix(image_before, text_before),
        "After ITE": cosine_matrix(image_ite, text_ite),
        "After ITE + ASM": cosine_matrix(image_gafm, text_ite),
    }

    return {
        "tokens": tokens,
        "matrices": matrices,
        "modulation": relative_modulation(cache)[0].copy(),
        "global_score": float(
            cache["similarity_score"].numpy().reshape(-1)[0]
        ),
    }


def find_token_index(tokens, target):
    target = target.lower()
    for index, token in enumerate(tokens):
        if token.lower().replace("##", "") == target:
            return index
    raise ValueError(
        "Token {!r} was not found. Available tokens: {}".format(
            target, tokens
        )
    )


def extract_structural_scores(matrix, tokens, color_word):
    """Return Dog, Playing, Ball and same-region Color-to-Ball scores."""
    dog_index = find_token_index(tokens, "dog")
    playing_index = find_token_index(tokens, "playing")
    ball_index = find_token_index(tokens, "ball")
    color_index = find_token_index(tokens, color_word)

    dog_score = matrix[:, dog_index].max()
    playing_score = matrix[:, playing_index].max()
    ball_score = matrix[:, ball_index].max()

    # Both color and ball must be supported by the same visual region.
    color_ball_score = np.max(
        (matrix[:, color_index] + matrix[:, ball_index]) / 2.0
    )

    return np.asarray(
        [dog_score, playing_score, ball_score, color_ball_score],
        dtype=np.float32,
    )


def draw_condition_image(
    ax,
    image_path,
    boxes,
    modulation,
    title,
    border_color,
    top_regions,
):
    if image_path is None:
        region_ids = np.arange(1, len(modulation) + 1)
        ax.bar(region_ids, modulation, color=border_color, width=0.85)
        ax.set_xlabel("Region index")
        ax.set_ylabel(r"$\|\Delta v_i\|_2 / \|v_i\|_2$")
        ax.set_title(title + "\n(raw image unavailable)",
                     fontsize=12, fontweight="bold", color=border_color)
        ax.grid(axis="y", alpha=0.20)
        return

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ax.imshow(image)
    ax.axis("off")

    if boxes is not None:
        boxes = prepare_boxes(boxes, width, height)
        top_regions = min(top_regions, len(boxes))
        selected = np.argsort(modulation)[::-1][:top_regions]
        selected_values = modulation[selected]
        norm = plt.Normalize(
            float(selected_values.min()),
            float(selected_values.max()) + 1e-8,
        )
        cmap = plt.get_cmap("magma")

        for region_id in selected:
            x1, y1, x2, y2 = boxes[region_id]
            color = cmap(norm(float(modulation[region_id])))
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    max(x2 - x1, 1),
                    max(y2 - y1, 1),
                    fill=False,
                    edgecolor=color,
                    linewidth=2.0,
                )
            )
            ax.text(
                x1,
                y1,
                "R{}".format(region_id + 1),
                color="white",
                fontsize=7,
                bbox=dict(
                    facecolor=color,
                    alpha=0.90,
                    edgecolor="none",
                    pad=1.0,
                ),
            )

    ax.add_patch(
        patches.Rectangle(
            (1, 1),
            max(width - 2, 1),
            max(height - 2, 1),
            fill=False,
            edgecolor=border_color,
            linewidth=4.0,
        )
    )
    ax.set_title(title, fontsize=12, fontweight="bold", color=border_color)


def draw_word_bars(
    ax, labels, positive, negative, title, ylim, show_legend=False
):
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(
        x - width / 2.0,
        positive,
        width,
        color="#4285F4",
        label="Positive text",
    )
    ax.bar(
        x + width / 2.0,
        negative,
        width,
        color="#16C79A",
        label="Hard-negative text",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Max region-word cosine similarity")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.20)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="best")


def resolve_fixed_image(args, dataset_path, dataset, caption, image_id):
    if args.positive_image is not None:
        return resolve_original_image(
            args.positive_image,
            dataset_path,
            args.split,
            args.caption_index,
            image_id,
            len(dataset),
            int(dataset.images.shape[0]),
        )
    if args.annotations_csv is not None:
        try:
            return resolve_image_from_annotations(
                args.annotations_csv, dataset_path, caption
            )
        except FileNotFoundError as error:
            print("Warning: {}".format(error))
            print("Continuing without the raw image; ASM region bars will be shown.")
            return None
    return resolve_original_image(
        None,
        dataset_path,
        args.split,
        args.caption_index,
        image_id,
        len(dataset),
        int(dataset.images.shape[0]),
    )


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    if "opt" not in checkpoint or "model" not in checkpoint:
        raise KeyError("Checkpoint must contain 'opt' and 'model'.")

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

    for name, index in (
        ("caption_index", args.caption_index),
        ("negative_caption_index", args.negative_caption_index),
    ):
        if index < 0 or index >= len(dataset):
            raise IndexError("{} is outside the selected split.".format(name))

    fixed_feature, positive_ids, _, fixed_image_id = dataset[
        args.caption_index
    ]
    _, negative_ids, _, negative_source_image_id = dataset[
        args.negative_caption_index
    ]
    fixed_image_id = int(fixed_image_id)
    negative_source_image_id = int(negative_source_image_id)

    positive_caption = decode_caption(
        dataset.captions[args.caption_index]
    )
    negative_caption = decode_caption(
        dataset.captions[args.negative_caption_index]
    )

    print("Fixed image index: {}".format(fixed_image_id))
    print("Positive caption index: {}".format(args.caption_index))
    print("Positive caption: {}".format(positive_caption))
    print(
        "Hard-negative caption index: {}".format(
            args.negative_caption_index
        )
    )
    print("Hard-negative caption: {}".format(negative_caption))
    print(
        "Hard-negative caption source image {} is ignored.".format(
            negative_source_image_id
        )
    )

    model = CSAN(opt)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.val_start()

    positive_result = evaluate_fixed_image_text(
        model, opt, tokenizer, fixed_feature, positive_ids
    )
    negative_result = evaluate_fixed_image_text(
        model, opt, tokenizer, fixed_feature, negative_ids
    )

    print("Positive tokens: {}".format(positive_result["tokens"]))
    print("Hard-negative tokens: {}".format(negative_result["tokens"]))

    labels = ["Dog", "Playing", "Ball", "Color→Ball"]
    stage_names = ["Before ITE", "After ITE", "After ITE + ASM"]
    stages = []

    for stage_name in stage_names:
        positive_values = extract_structural_scores(
            positive_result["matrices"][stage_name],
            positive_result["tokens"],
            args.positive_color,
        )
        negative_values = extract_structural_scores(
            negative_result["matrices"][stage_name],
            negative_result["tokens"],
            args.negative_color,
        )
        stages.append((stage_name, positive_values, negative_values))

        print("\n{}".format(stage_name))
        for label, pos_value, neg_value in zip(
            labels, positive_values, negative_values
        ):
            print(
                "{:12s} positive={:.4f}, hard-negative={:.4f}".format(
                    label, pos_value, neg_value
                )
            )

    positive_global = positive_result["global_score"]
    negative_global = negative_result["global_score"]
    print("\nPositive global score: {:.4f}".format(positive_global))
    print("Hard-negative global score: {:.4f}".format(negative_global))

    all_values = np.concatenate(
        [
            values
            for _, positive_values, negative_values in stages
            for values in (positive_values, negative_values)
        ]
    )
    value_min = float(all_values.min())
    value_max = float(all_values.max())
    padding = max(0.03, 0.08 * (value_max - value_min))
    ylim = (value_min - padding, value_max + padding)

    fixed_image_path = resolve_fixed_image(
        args,
        dataset_path,
        dataset,
        positive_caption,
        fixed_image_id,
    )
    if fixed_image_path is None:
        print(
            "Warning: fixed raw image was not found. Continuing with ASM "
            "region bars. Use --positive_image to include the photograph."
        )

    boxes = None
    if args.boxes_file is not None:
        boxes = load_boxes(
            args.boxes_file,
            fixed_image_id,
            fixed_feature.shape[0],
        )

    figure, axes = plt.subplots(
        2, 3, figsize=(18, 9), constrained_layout=True
    )

    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.02,
        0.94,
        "Fixed image: image_id={}".format(fixed_image_id),
        transform=axes[0, 0].transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    axes[0, 0].text(
        0.02,
        0.75,
        "Positive text:\n{}".format(
            textwrap.fill(positive_caption, width=48)
        ),
        transform=axes[0, 0].transAxes,
        fontsize=11,
        color="#1565c0",
        va="top",
    )
    axes[0, 0].text(
        0.02,
        0.42,
        "Hard-negative text:\n{}".format(
            textwrap.fill(negative_caption, width=48)
        ),
        transform=axes[0, 0].transAxes,
        fontsize=11,
        color="#2e7d32",
        va="top",
    )
    axes[0, 0].add_patch(
        patches.Rectangle(
            (0.005, 0.05),
            0.98,
            0.90,
            transform=axes[0, 0].transAxes,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            edgecolor="#555555",
        )
    )

    draw_condition_image(
        axes[0, 1],
        fixed_image_path,
        boxes,
        positive_result["modulation"],
        "Fixed image + positive text\nGlobal score {:.4f}".format(
            positive_global
        ),
        "#1565c0",
        args.top_regions,
    )
    draw_condition_image(
        axes[0, 2],
        fixed_image_path,
        boxes,
        negative_result["modulation"],
        "Fixed image + hard-negative text\nGlobal score {:.4f}".format(
            negative_global
        ),
        "#2e7d32",
        args.top_regions,
    )

    for index, (stage_name, positive_values, negative_values) in enumerate(
        stages
    ):
        draw_word_bars(
            axes[1, index],
            labels,
            positive_values,
            negative_values,
            stage_name,
            ylim,
            show_legend=(index == 0),
        )

    figure.suptitle(
        "Fixed-image comparison: {} ball versus {} ball".format(
            args.positive_color, args.negative_color
        ),
        fontsize=16,
        fontweight="bold",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "gafm_text_pair_pos_{:05d}_neg_{:05d}".format(
        args.caption_index, args.negative_caption_index
    )

    figure_path = output_dir / (stem + ".png")
    figure.savefig(str(figure_path), dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "fixed_image_index": fixed_image_id,
        "positive_caption_index": int(args.caption_index),
        "negative_caption_index": int(args.negative_caption_index),
        "negative_caption_source_image_ignored": negative_source_image_id,
        "positive_caption": positive_caption,
        "negative_caption": negative_caption,
        "positive_color": args.positive_color,
        "negative_color": args.negative_color,
        "labels": labels,
        "positive_global_score": positive_global,
        "negative_global_score": negative_global,
        "stages": {
            stage_name: {
                "positive": positive_values.tolist(),
                "hard_negative": negative_values.tolist(),
            }
            for stage_name, positive_values, negative_values in stages
        },
    }
    metadata_path = output_dir / (stem + ".json")
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    arrays_path = output_dir / (stem + ".npz")
    np.savez_compressed(
        str(arrays_path),
        labels=np.asarray(labels),
        before_ite_positive=stages[0][1],
        before_ite_hard_negative=stages[0][2],
        after_ite_positive=stages[1][1],
        after_ite_hard_negative=stages[1][2],
        after_gafm_positive=stages[2][1],
        after_gafm_hard_negative=stages[2][2],
        positive_global_score=np.asarray([positive_global]),
        negative_global_score=np.asarray([negative_global]),
        positive_asm_modulation=positive_result["modulation"],
        negative_asm_modulation=negative_result["modulation"],
        positive_tokens=np.asarray(positive_result["tokens"]),
        negative_tokens=np.asarray(negative_result["tokens"]),
    )

    print("\nSaved {}".format(figure_path))
    print("Saved {}".format(metadata_path))
    print("Saved {}".format(arrays_path))


if __name__ == "__main__":
    main()

"""Paper-style positive/hard-negative visualization for CSAN GAFM.

The figure follows a query/positive/negative layout and compares word-level
region similarities before ITE, after ITE, and after ITE+ASM.  Faster R-CNN
boxes must have the same image/region order as the precomputed region features.
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
        description="Visualize positive versus hard-negative GAFM responses."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--data_name", default=None)
    parser.add_argument("--bert_path", default=None)
    parser.add_argument("--split", default="test",
                        choices=("train", "dev", "test"))
    parser.add_argument("--caption_index", type=int, required=True,
                        help="Query caption index.")
    parser.add_argument(
        "--negative_image_index", type=int, default=None,
        help=("Optional image-array index of a mismatched image. If omitted, "
              "the highest-scoring non-matching image is selected automatically.")
    )
    parser.add_argument(
        "--candidate_batch_size", type=int, default=64,
        help="Image batch size used during automatic hard-negative mining."
    )
    parser.add_argument("--boxes_file", required=True,
                        help="Boxes with shape [num_images, num_regions, 4].")
    parser.add_argument("--annotations_csv", default=None,
                        help="flickr_annotations_30k.csv used to resolve raw images from captions.")
    parser.add_argument("--positive_image", default=None,
                        help="Optional override for the matched raw image.")
    parser.add_argument("--negative_image", default=None,
                        help="Optional override for the negative raw image.")
    parser.add_argument("--top_regions", type=int, default=8)
    parser.add_argument("--top_words", type=int, default=4,
                        help="Number of largest final-stage margins to list.")
    parser.add_argument("--output_dir", default="./gafm_pair_visualizations")
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
    # SCAN captions often tokenize punctuation as ``hat .`` whereas the
    # original Flickr30K CSV stores ``hat.``.  Punctuation is irrelevant for
    # image lookup, so replace it with spaces before collapsing whitespace.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def detect_csv_columns(fieldnames):
    normalized = {str(name).strip().lower(): name for name in fieldnames or []}
    image_candidates = (
        "image_name", "filename", "file_name", "image", "image_id"
    )
    caption_candidates = (
        "comment", "caption", "sentence", "raw", "description"
    )
    image_column = next(
        (normalized[name] for name in image_candidates if name in normalized), None
    )
    caption_column = next(
        (normalized[name] for name in caption_candidates if name in normalized), None
    )
    if image_column is None or caption_column is None:
        raise ValueError(
            "Could not identify image/caption columns in annotations CSV. "
            "Columns found: {}".format(fieldnames)
        )
    return image_column, caption_column


def expand_caption_cell(value):
    """Return one or more captions stored in a CSV cell.

    Flickr30K annotation tables may store five captions as a JSON list in one
    cell instead of storing one caption per row.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]

    return [text]


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
    """Resolve an image filename by exact normalized caption matching."""
    target = normalize_caption(caption)
    matches = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        image_column, caption_column = detect_csv_columns(reader.fieldnames)
        for row in reader:
            row_captions = expand_caption_cell(row.get(caption_column, ""))
            if any(normalize_caption(item) == target for item in row_captions):
                filename = row.get(image_column, "")
                if filename and filename not in matches:
                    matches.append(filename)

    if not matches:
        raise KeyError(
            "Caption was not found in {}: {}".format(csv_path, caption)
        )
    if len(matches) > 1:
        raise ValueError(
            "Caption matched multiple images in {}: {}".format(csv_path, matches)
        )

    image_path = find_raw_image(dataset_path, matches[0])
    if image_path is None:
        raise FileNotFoundError(
            "CSV resolved image filename '{}', but it was not found under {}."
            .format(matches[0], dataset_path)
        )
    return image_path


def word_region_scores(regions, words):
    """Return max-over-region score for every word and the full matrix."""
    matrix = cosine_matrix(regions, words)
    return matrix.max(axis=0), matrix


def visible_token_positions(all_tokens):
    """Return positions used in the plot, excluding special/wrapper tokens.

    Some legacy SCAN-style data loaders read a caption as bytes and then call
    ``str(value)``, which produces a sequence such as ``b'The ...'``.  The
    leading ``b`` and surrounding quotes are serialization artifacts rather
    than caption words.  They are excluded from the plot, while the checkpoint
    itself is evaluated without changing its input sequence.
    """
    keep = [i for i, token in enumerate(all_tokens)
            if token not in SPECIAL_TOKENS]
    visible = [all_tokens[i] for i in keep]
    quote_tokens = {"'", '"', "##'", '##"'}

    if (len(visible) >= 2 and visible[0].lower() == "b" and
            visible[1] in quote_tokens):
        keep = keep[2:]
        visible = visible[2:]
    if visible and visible[-1] in quote_tokens:
        keep = keep[:-1]
    return keep


def mine_hard_negative(model, opt, dataset, caption_ids, positive_image_id,
                       batch_size):
    """Select the highest-scoring image that is not the ground-truth image."""
    if batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive.")

    caption_tensor = torch.tensor(
        caption_ids, dtype=torch.long
    ).unsqueeze(0)
    caption_length = [len(caption_ids)]
    candidate_scores = []
    device = next(model.parameters()).device

    model.enable_visualization(False)
    with torch.no_grad():
        # Encode the query only once. Re-running BERT for every image chunk is
        # unnecessary because ITE processes the image and text streams
        # independently.
        reference_image = torch.as_tensor(
            np.asarray(dataset.images[int(positive_image_id)]),
            dtype=torch.float32
        ).unsqueeze(0)
        _, caption_emb, _ = model.forward_emb(
            reference_image, caption_tensor, caption_length
        )

        for start in range(0, int(dataset.images.shape[0]), batch_size):
            end = min(start + batch_size, int(dataset.images.shape[0]))
            image_batch = torch.as_tensor(
                np.asarray(dataset.images[start:end]), dtype=torch.float32
            ).to(device)
            image_before_ite = model.img_enc(image_batch)
            image_emb, _ = model.GAT_model(image_before_ite, caption_emb)
            scores = model.forward_sim(
                opt, image_emb, caption_emb, caption_length
            )[:, 0]
            candidate_scores.append(scores.detach().cpu().numpy())

    candidate_scores = np.concatenate(candidate_scores, axis=0)
    candidate_scores[int(positive_image_id)] = -np.inf
    negative_image_id = int(np.argmax(candidate_scores))
    return negative_image_id, float(candidate_scores[negative_image_id])


def relative_modulation(cache):
    asm_input = cache["asm_input"].numpy().transpose(0, 2, 1)
    asm_delta = cache["asm_delta"].numpy().transpose(0, 2, 1)
    numerator = np.linalg.norm(asm_delta, axis=2)
    denominator = np.maximum(np.linalg.norm(asm_input, axis=2), 1e-8)
    return numerator / denominator


def draw_query(ax, caption, tokens, margins, top_words):
    ax.axis("off")
    ax.text(0.02, 0.94, "Query sentence", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top")
    ax.text(0.02, 0.78, textwrap.fill(caption, width=43),
            transform=ax.transAxes, fontsize=13, va="top", linespacing=1.45)

    order = np.argsort(margins)[::-1]
    positive_order = [i for i in order if margins[i] > 0]
    if not positive_order:
        positive_order = order.tolist()
    strongest = [tokens[i] for i in positive_order[:min(top_words, len(tokens))]]
    ax.text(0.02, 0.32, "Largest final-stage positive-negative margins:",
            transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.text(0.02, 0.22, ", ".join(strongest), transform=ax.transAxes,
            fontsize=11, color="#b2182b")
    ax.add_patch(patches.Rectangle(
        (0.005, 0.05), 0.98, 0.90, transform=ax.transAxes,
        fill=False, linestyle="--", linewidth=1.5, edgecolor="#555555"
    ))


def draw_box_image(ax, image_path, boxes, modulation, title,
                   border_color, top_regions):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ax.imshow(image)
    ax.axis("off")

    boxes = prepare_boxes(boxes, width, height)
    top_regions = min(top_regions, len(boxes))
    selected = np.argsort(modulation)[::-1][:top_regions]
    selected_values = modulation[selected]
    norm = plt.Normalize(float(selected_values.min()),
                         float(selected_values.max()) + 1e-8)
    cmap = plt.get_cmap("magma")

    for region_id in selected:
        x1, y1, x2, y2 = boxes[region_id]
        color = cmap(norm(float(modulation[region_id])))
        ax.add_patch(patches.Rectangle(
            (x1, y1), max(x2 - x1, 1), max(y2 - y1, 1),
            fill=False, edgecolor=color, linewidth=2.0
        ))
        ax.text(x1, y1, "R{}".format(region_id + 1), color="white",
                fontsize=7, bbox=dict(facecolor=color, alpha=0.90,
                                      edgecolor="none", pad=1.0))

    ax.add_patch(patches.Rectangle(
        (1, 1), max(width - 2, 1), max(height - 2, 1),
        fill=False, edgecolor=border_color, linewidth=4.0
    ))
    ax.set_title(title, fontsize=13, fontweight="bold", color=border_color)


def draw_word_bars(ax, tokens, positive, negative, title, ylim,
                   negative_label, show_legend=False):
    x = np.arange(len(tokens))
    width = 0.38
    ax.bar(x - width / 2.0, positive, width,
           color="#4285F4", label="Positive image")
    ax.bar(x + width / 2.0, negative, width,
           color="#16C79A", label=negative_label)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=48, ha="right", fontsize=8)
    ax.set_ylabel("Max region-word cosine similarity")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.20)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="best")


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

    if args.caption_index < 0 or args.caption_index >= len(dataset):
        raise IndexError("caption_index is outside the selected split.")

    positive_feature, caption_ids, _, positive_image_id = dataset[args.caption_index]
    positive_image_id = int(positive_image_id)

    model = CSAN(opt)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.val_start()

    automatic_negative = args.negative_image_index is None
    if automatic_negative:
        negative_image_index, mined_score = mine_hard_negative(
            model, opt, dataset, caption_ids, positive_image_id,
            args.candidate_batch_size
        )
        print(
            "Automatically selected hard negative image {} (score {:.4f})."
            .format(negative_image_index, mined_score)
        )
    else:
        negative_image_index = int(args.negative_image_index)
        if (negative_image_index < 0 or
                negative_image_index >= dataset.images.shape[0]):
            raise IndexError("negative_image_index is outside the image array.")
        if negative_image_index == positive_image_id:
            raise ValueError("The negative image must differ from the positive image.")

    negative_feature = torch.tensor(
        dataset.images[negative_image_index], dtype=torch.float32
    )
    images = torch.stack((positive_feature.float(), negative_feature), dim=0)
    caption_tensor = torch.tensor(caption_ids, dtype=torch.long).unsqueeze(0)
    captions = caption_tensor.repeat(2, 1)
    lengths = [len(caption_ids), len(caption_ids)]

    model.enable_visualization(True)

    with torch.no_grad():
        image_after, text_after, output_lengths = model.forward_emb(
            images, captions, lengths
        )
        model.forward_sim(opt, image_after, text_after, output_lengths)
    cache = model.get_visualization_cache(cpu=True)

    required = {
        "image_before_ite", "text_before_ite", "image_after_ite",
        "text_after_ite", "asm_input", "asm_output", "asm_delta",
        "similarity_score",
    }
    missing = sorted(required.difference(cache))
    if missing:
        raise RuntimeError("Visualization cache is missing: {}".format(
            ", ".join(missing)))

    all_tokens = tokenizer.convert_ids_to_tokens(caption_ids)
    keep = visible_token_positions(all_tokens)
    tokens = [all_tokens[i] for i in keep]
    if not tokens:
        raise RuntimeError("No displayable caption tokens were found.")

    image_before = cache["image_before_ite"].numpy()
    text_before = cache["text_before_ite"].numpy()[:, keep, :]
    image_ite = cache["image_after_ite"].numpy()
    text_ite = cache["text_after_ite"].numpy()[:, keep, :]
    image_asm = cache["asm_output"].numpy().transpose(0, 2, 1)

    stages = []
    stage_inputs = (
        ("Before ITE", image_before, text_before),
        ("After ITE", image_ite, text_ite),
        ("After ITE + ASM", image_asm, text_ite),
    )
    for title, image_stage, text_stage in stage_inputs:
        positive_scores, _ = word_region_scores(image_stage[0], text_stage[0])
        negative_scores, _ = word_region_scores(image_stage[1], text_stage[1])
        stages.append((title, positive_scores, negative_scores))

    all_values = np.concatenate([
        value for _, positive, negative in stages
        for value in (positive, negative)
    ])
    value_min = float(all_values.min())
    value_max = float(all_values.max())
    padding = max(0.03, 0.08 * (value_max - value_min))
    ylim = (value_min - padding, value_max + padding)

    final_positive = stages[-1][1]
    final_negative = stages[-1][2]
    gains = final_positive - final_negative

    modulation = relative_modulation(cache)
    positive_boxes = load_boxes(
        args.boxes_file, positive_image_id, image_before.shape[1]
    )
    negative_boxes = load_boxes(
        args.boxes_file, negative_image_index, image_before.shape[1]
    )

    negative_caption_index = int(negative_image_index * dataset.im_div)
    positive_caption = decode_caption(dataset.captions[args.caption_index])
    negative_caption = decode_caption(dataset.captions[negative_caption_index])

    if args.positive_image is not None:
        positive_image_path = resolve_original_image(
            args.positive_image, dataset_path, args.split,
            args.caption_index, positive_image_id,
            len(dataset), int(dataset.images.shape[0])
        )
    elif args.annotations_csv is not None:
        positive_image_path = resolve_image_from_annotations(
            args.annotations_csv, dataset_path, positive_caption
        )
    else:
        positive_image_path = resolve_original_image(
            None, dataset_path, args.split,
            args.caption_index, positive_image_id,
            len(dataset), int(dataset.images.shape[0])
        )

    if args.negative_image is not None:
        negative_image_path = resolve_original_image(
            args.negative_image, dataset_path, args.split,
            negative_caption_index, negative_image_index,
            len(dataset), int(dataset.images.shape[0])
        )
    elif args.annotations_csv is not None:
        negative_image_path = resolve_image_from_annotations(
            args.annotations_csv, dataset_path, negative_caption
        )
    else:
        negative_image_path = resolve_original_image(
            None, dataset_path, args.split,
            negative_caption_index, negative_image_index,
            len(dataset), int(dataset.images.shape[0])
        )
    if positive_image_path is None or negative_image_path is None:
        raise FileNotFoundError(
            "Could not resolve both raw images; use --positive_image and "
            "--negative_image to provide them explicitly."
        )

    score_matrix = cache["similarity_score"].numpy()
    positive_global = float(score_matrix[0, 0])
    negative_global = float(score_matrix[1, 0])
    caption = positive_caption

    figure, axes = plt.subplots(2, 3, figsize=(19, 10),
                                constrained_layout=True)
    draw_query(axes[0, 0], caption, tokens, gains, args.top_words)
    draw_box_image(
        axes[0, 1], positive_image_path, positive_boxes, modulation[0],
        "Positive image | score {:.4f}".format(positive_global),
        "#159447", args.top_regions
    )
    negative_kind = "Hard negative" if automatic_negative else "Selected negative"
    draw_box_image(
        axes[0, 2], negative_image_path, negative_boxes, modulation[1],
        "{} | score {:.4f}".format(negative_kind, negative_global),
        "#d62728", args.top_regions
    )

    for index, (title, positive, negative) in enumerate(stages):
        draw_word_bars(
            axes[1, index], tokens, positive, negative, title, ylim,
            negative_kind,
            show_legend=(index == 0)
        )

    figure.suptitle(
        ("GAFM responses for the matched image and the highest-scoring "
         "mismatched image" if automatic_negative else
         "GAFM responses for the matched image and a selected negative"),
        fontsize=16, fontweight="bold"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "gafm_pair_caption_{:05d}_negative_{:05d}".format(
        args.caption_index, negative_image_index
    )
    figure_path = output_dir / (stem + ".png")
    figure.savefig(str(figure_path), dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "caption_index": int(args.caption_index),
        "caption": caption,
        "tokens": tokens,
        "positive_image_index": positive_image_id,
        "negative_image_index": int(negative_image_index),
        "negative_selection": (
            "highest_scoring_non_matching" if automatic_negative else "manual"
        ),
        "positive_image": positive_image_path,
        "negative_image": negative_image_path,
        "positive_score": positive_global,
        "negative_score": negative_global,
        "word_discrimination_gain": gains.tolist(),
    }
    metadata_path = output_dir / (stem + ".json")
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    print("Saved {}".format(figure_path))
    print("Saved {}".format(metadata_path))


if __name__ == "__main__":
    main()

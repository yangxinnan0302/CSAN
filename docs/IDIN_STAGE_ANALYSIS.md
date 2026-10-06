# IDIN stage-wise alignment analysis

This protocol addresses the reviewer request for evidence that IDIN preserves
stable correspondences and corrects early alignment errors. It records the
actual attention used by `cross_attention` at every recursive stage; final
pair confidence is not used as a substitute for region-word alignment.

The current correction protocol requires the paper's text-to-image (`t2i`)
attention direction, whose softmax is a probability distribution over regions
for every word. An image-to-text distribution is normalized over words for
each region; merely transposing it would not produce the same conditional
distribution, so the script rejects such checkpoints instead of silently
reporting invalid word-to-region metrics.

## Important notation check

The implementation updates a word-conditioned feature gate
`matrix` with shape `[word, embedding_dim]` and a word-conditioned attention
temperature `smooth`. In `cross_attention`, the gate multiplies each query
embedding before region-word affinities are computed. This is materially
different from adding one scalar bias to every regional logit: a common scalar
bias would cancel exactly under softmax. The revised method section should
therefore describe the control state as a vector (or feature-wise gate), not as
a region-independent scalar bias in Eq. (15).

## What to report

For stage `n`, let `A_j^(n)` be the probability distribution over detected
regions for word `j`.

1. **Adjacent-stage JS drift (lower is more stable)**

   `D_JS(A_j^(n-1), A_j^(n))`, averaged over non-special words and samples.
   Jensen-Shannon divergence is symmetric and bounded in `[0, 1]` when base-2
   logarithms are used.

2. **Top-1 switch rate (lower is more stable)**

   The fraction of words whose maximum-attention region changes between two
   adjacent stages.

3. **Top-k overlap (higher is more stable)**

   `|TopK(A_j^(n-1)) intersection TopK(A_j^(n))| / k`.

4. **Normalized entropy (interpret together with accuracy)**

   `H(A_j^(n)) / log(K)`, where `K` is the number of regions. A decrease means
   a sharper alignment, but does not by itself prove a more correct alignment.

5. **Visual-context trajectory step (lower means smaller updates)**

   `1 - cosine(vhat_j^(n-1), vhat_j^(n))`. Plotting this across stages shows
   whether refinement converges instead of oscillating.

6. **Alignment accuracy and error correction (requires ground truth)**

   A prediction is correct if its Top-1 region belongs to the acceptable
   region set annotated for that word. Report stage-wise accuracy, plus:

   `correction rate = #(wrong initially, correct finally) / #wrong initially`

   `regression rate = #(correct initially, wrong finally) / #correct initially`

   `drift-after-correct rate = #(correct at a stage, wrong later) / #ever correct`

Do not call a changed attention map an "error correction" without word-region
ground truth or an explicit human annotation. Without labels, the script marks
examples as **unverified candidates** only.

## Ground-truth JSON

Token indices are zero-based BERT/WordPiece positions, including `[CLS]` in
position 0. Region indices are zero-based positions in the 36-region BUTD
feature array. Multiple boxes may be acceptable for one word.

```json
{
  "125": {
    "3": [7],
    "6": [11, 19]
  },
  "402": {
    "2": [4]
  }
}
```

The safest annotation procedure is to show all 36 indexed detector boxes,
select concrete nouns/adjectives with visible referents, and have two people
annotate independently. Resolve disagreements before computing correction
rates. Report the number of image-caption pairs, labelled tokens, and the
agreement protocol.

## Run

Index-based visualization and label-free metrics:

```bash
python analyze_idin_stages.py \
  --checkpoint ./runs/model_best.pth.tar \
  --data_path ./data \
  --data_name f30k_precomp \
  --bert_path ./uncased_L-12_H-768_A-12/ \
  --split test \
  --all \
  --visualize_limit 12 \
  --output_dir ./idin_stage_analysis
```

Reference-style overlays and verified correction metrics:

```bash
python analyze_idin_stages.py \
  --checkpoint ./runs/model_best.pth.tar \
  --data_path ./data \
  --data_name f30k_precomp \
  --bert_path ./uncased_L-12_H-768_A-12/ \
  --split test \
  --caption_index 125 402 \
  --ground_truth_json ./annotations/idin_word_region_gt.json \
  --images_root /path/to/flickr30k-images \
  --id_mapping ./data/f30k/id_mapping.json \
  --image_ids_file ./data/f30k/precomp/test_ids.txt \
  --boxes_file ./data/f30k/precomp/test_boxes.npy \
  --output_dir ./idin_stage_analysis
```

`test_boxes.npy` must use exactly the same region order as `test_ims.npy`.
The standard SCAN feature package commonly lacks box coordinates; boxes from a
different detector/order make overlays and correction labels invalid.

Outputs include a PNG, NPZ, and JSON for each selected qualitative sample,
per-caption `stage_metrics.csv`, and aggregate `summary.json`. With `--all`,
metrics cover the split while `--visualize_limit` prevents thousands of
figures from being created. Use `--max_samples 20` for a dry run.

## Suggested experiment design

- Use the fixed released checkpoint and evaluation split; do not cherry-pick
  only successful examples for aggregate metrics.
- Compute aggregate metrics on at least the full Flickr30K test split if
  feasible. Bootstrap image-caption pairs (for example 1,000 resamples) to
  report 95% confidence intervals.
- For verified correction rates, annotate a predeclared random subset and
  disclose its size. Include both successful corrections and failure cases.
- Add three qualitative groups: stable-correct, corrected, and regressed/drift.
  Each panel should keep the same queried word and region colors across stages.
- Compare IDIN with a no-history control that recomputes each stage without
  carrying the RCR state. This isolates state propagation from merely adding
  depth.

## Draft reviewer response (fill only measured values)

> Thank you for pointing out that the original Fig. 4 and final retrieval
> results did not directly demonstrate the evolution of correspondences. We
> have added a stage-wise analysis of IDIN in Sec. [X], Fig. [Y], and Table [Z].
> For each non-special word, we record the region-attention distribution used
> at every IDIN stage. We visualize the same word and detector regions across
> stages and plot its alignment trajectory. On [dataset/subset, N pairs and M
> annotated tokens], adjacent-stage JS drift changes from [ ] to [ ], Top-1
> switch rate from [ ] to [ ], Top-3 overlap from [ ] to [ ], and normalized
> entropy from [ ] to [ ]. Using manually verified word-region annotations,
> stage-wise Top-1 alignment accuracy is [stage values]; among initially
> incorrect alignments, [ ]% are corrected at the final stage, while [ ]% of
> initially correct alignments regress. We also include corrected, stable, and
> failure cases to avoid selective presentation. These results directly show
> [the measured conclusion; do not claim monotonic improvement unless the
> values support it]. We additionally clarified the feature-wise recursive
> control state in Eqs. [ ]-[ ] so that the notation matches the implementation.

Replace every bracket with measured results and exact manuscript locations.
If the measured stability is non-monotonic, say so and discuss the failure
cases instead of claiming that all correspondences improve at every stage.

# CSAN
CSAN: Cascaded Structure-Aware Network for Image-Text Matching
# Introduction
The framework of CSAN:


<img width="5918" height="2099" alt="fig2" src="https://github.com/user-attachments/assets/0a7fe1a4-4a4a-4375-8cec-b64cba2a1edc" />

# Requirements
Utilize pip install -r requirements.txt for the following dependencies.
- Python 3.7.11
- PyTorch 1.7.1
- NumPy 1.21.5
- Punkt Sentence Tokenizer:

```
import nltk
nltk.download()
> d punkt
```
# Download data and vocab
We follow SCAN to obtain image features and vocabularies, which can be downloaded by using:
```
https://www.kaggle.com/datasets/kuanghueilee/scan-features
```
```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   └── id_mapping.json  # mapping from coco-id to image's file name
│   
│
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   └── id_mapping.json  # mapping from f30k index to image's file name
│   
│
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)
```
# Training
```
python train.py
```
# Evaluation
```
python evaluation.py
```

# Efficiency analysis

`efficiency.py` reports the quantities needed for a reproducible efficiency
comparison: total/trainable parameters, component-level parameters, FLOPs,
inference latency, throughput, and peak GPU memory. It also optionally measures
the peak memory of one forward/backward training step.

Install the FLOP-counting dependency:

```
pip install -r requirements-efficiency.txt
```

Measure single-pair FP32 inference and a training step with batch size 64:

```
python efficiency.py \
  --bert_path ./uncased_L-12_H-768_A-12/ \
  --checkpoint ./runs/model_best.pth.tar \
  --device cuda \
  --precision fp32 \
  --batch_size 1 \
  --train_batch_size 64 \
  --num_regions 36 \
  --seq_len 32 \
  --warmup 20 \
  --iterations 100 \
  --output efficiency_results.json \
  --csv_output efficiency_results.csv
```

For throughput at the evaluation shard size, repeat the measurement with
`--batch_size 32`. The model produces a `B x B` similarity matrix, so the JSON
distinguishes aligned inputs per second from candidate-pair scores per second.

Important reporting conventions:

- Inputs are 36 precomputed 2048-dimensional BUTD region features and 32 BERT
  token IDs. The offline Faster R-CNN detector is not included.
- `fvcore` treats one fused multiply-add (FMA) as one operation. The script also
  reports the converted value when a multiply and an add are counted as two
  FLOPs.
- Review `flops.unsupported_ops` in the generated JSON before quoting FLOPs. An
  empty mapping means all encountered operators were supported by the profiler.
- Latency includes synchronization before and after every timed iteration. Use
  the same GPU, precision, software environment, input dimensions, warm-up, and
  iteration count for every compared method.
- The checkpoint is optional for complexity measurement because weights do not
  affect parameter counts or FLOPs, but using the evaluated checkpoint is
  recommended for a fully traceable experiment.

# GAFM interpretability visualization

`visualize_gafm.py` exposes the intermediate evidence produced by the two GAFM
components without changing checkpoint parameters or retrieval scores. For each
matched image-caption pair, it exports:

- ITE image-region and word-word dependency attention;
- ASM region-wise modulation strength and the distributions of its learned
  gamma/beta parameters;
- region-word cosine-similarity maps before ITE, after ITE, and after the full
  ITE+ASM pipeline;
- a PNG figure, an NPZ file containing the plotted values, and JSON metadata.

Run it with the same checkpoint, precomputed features, and BERT vocabulary used
for evaluation:

```
python visualize_gafm.py \
  --checkpoint ./runs/model_best.pth.tar \
  --data_path ./data \
  --data_name f30k_precomp \
  --bert_path ./uncased_L-12_H-768_A-12/ \
  --split test \
  --caption_index 0 25 50 \
  --output_dir ./gafm_visualizations
```

The SCAN precomputed package contains region features but not the corresponding
box coordinates, so the command above directly produces index-based region
visualizations. To overlay ASM strengths and the strongest ITE relations on an
original image, provide a single image and its Faster R-CNN boxes:

```
python visualize_gafm.py \
  --checkpoint ./runs/model_best.pth.tar \
  --data_path ./data \
  --bert_path ./uncased_L-12_H-768_A-12/ \
  --split test \
  --caption_index 0 \
  --image ./images/example.jpg \
  --boxes_file ./test_boxes.npy
```

`test_boxes.npy` may have shape `[36, 4]` for the selected image or
`[num_images, 36, 4]` for the full split. Coordinates must follow
`[x1, y1, x2, y2]`; normalized coordinates are also accepted.





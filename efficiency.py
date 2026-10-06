"""Reproducible efficiency measurements for CSAN.

The image detector is intentionally excluded: CSAN consumes the 36 precomputed
BUTD region features distributed with SCAN.  Run this script in the same
environment used for training so the latency and memory numbers are comparable.

FLOPs are measured with fvcore.  fvcore counts one fused multiply-add (FMA) as
one operation.  We report both that value and the common two-operations-per-FMA
conversion, and record unsupported operators in the JSON output.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import platform
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn as nn


DEFAULTS = {
    "img_dim": 2048,
    "bert_size": 768,
    "embed_size": 1024,
    "sim_dim": 256,
    "no_imgnorm": False,
    "no_txtnorm": False,
    "ft_bert": True,
    "module_name": "SGR",
    "sgr_step": 3,
    "focal_type": "glo",
    "self_regulator": "coop_rcar",
    "rcar_step": 2,
    "rcr_step": 2,
    "rar_step": 2,
    "attn_type": "t2i",
    "t2i_smooth": 10.0,
    "i2t_smooth": 3.0,
    "margin": 0.2,
    "max_violation": True,
    "grad_clip": 2.0,
    "bert_lr": 2e-5,
    "other_lr": 2e-4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure CSAN parameters, FLOPs, latency, throughput, and GPU memory."
    )
    parser.add_argument("--bert_path", required=True,
                        help="Local pytorch_pretrained_bert BERT directory.")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional CSAN checkpoint; its saved options are reused.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--precision", default="fp32", choices=("fp32", "fp16"))
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Inference batch size. Use 1 for single-pair latency.")
    parser.add_argument("--train_batch_size", default=0, type=int,
                        help="If >0, also measure one forward/backward training step.")
    parser.add_argument("--num_regions", default=36, type=int)
    parser.add_argument("--seq_len", default=32, type=int)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--output", default="efficiency_results.json")
    parser.add_argument("--csv_output", default="efficiency_results.csv")
    parser.add_argument("--skip_flops", action="store_true")
    return parser.parse_args()


def load_options(args):
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        opt = checkpoint.get("opt")
        if opt is None:
            raise KeyError("The checkpoint does not contain an 'opt' entry.")
    else:
        opt = SimpleNamespace()

    for name, value in DEFAULTS.items():
        if not hasattr(opt, name):
            setattr(opt, name, value)
    opt.bert_path = args.bert_path
    return opt, checkpoint


class EfficiencyForward(nn.Module):
    """Expose the repository's split embedding/similarity API as one forward."""

    def __init__(self, model, opt, seq_len):
        super(EfficiencyForward, self).__init__()
        # Register the four standard submodules directly. CSAN overrides
        # state_dict() with a legacy list-based checkpoint format, which is not
        # compatible with torch.jit/fvcore tracing when CSAN itself is nested.
        self.img_enc = model.img_enc
        self.txt_enc = model.txt_enc
        self.gat_model = model.GAT_model
        self.sim_enc = model.sim_enc
        self.opt = opt
        self.seq_len = seq_len

    def forward(self, images, captions):
        lengths = [self.seq_len] * captions.size(0)
        # Call the submodules directly. The repository's forward_emb() moves
        # inputs to CUDA whenever any GPU is visible, which prevents a valid
        # --device cpu measurement on GPU-equipped machines.
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        img_emb, cap_emb = self.gat_model(img_emb, cap_emb)
        return self.sim_enc(self.opt, img_emb, cap_emb, lengths)


def count_parameters(model):
    components = {}
    for name in ("img_enc", "txt_enc", "GAT_model", "sim_enc"):
        module = getattr(model, name)
        components[name] = {
            "total": sum(p.numel() for p in module.parameters()),
            "trainable": sum(p.numel() for p in module.parameters()
                             if p.requires_grad),
        }
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters()
                         if p.requires_grad),
        "components": components,
    }


def make_inputs(batch_size, args, opt, device):
    images = torch.randn(
        batch_size, args.num_regions, opt.img_dim, device=device
    )
    # BERT's standard uncased vocabulary contains 30,522 entries. Restricting
    # synthetic IDs to 100 avoids depending on tokenizer files while remaining
    # valid for standard checkpoints.
    captions = torch.randint(
        low=1, high=100, size=(batch_size, args.seq_len),
        dtype=torch.long, device=device
    )
    return images, captions


@contextmanager
def autocast_context(args):
    if args.device == "cuda" and args.precision == "fp16":
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def synchronize(args):
    if args.device == "cuda":
        torch.cuda.synchronize()


def percentile(values, q):
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def measure_inference(wrapper, inputs, args):
    wrapper.eval()
    with torch.no_grad():
        for _ in range(args.warmup):
            with autocast_context(args):
                wrapper(*inputs)
        synchronize(args)

        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated = torch.cuda.memory_allocated()

        times_ms = []
        for _ in range(args.iterations):
            synchronize(args)
            start = time.perf_counter()
            with autocast_context(args):
                wrapper(*inputs)
            synchronize(args)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    batch_size = inputs[0].size(0)
    result = {
        "batch_size": batch_size,
        "mean_ms_per_batch": mean_ms,
        "median_ms_per_batch": statistics.median(times_ms),
        "p95_ms_per_batch": percentile(times_ms, 0.95),
        "aligned_inputs_per_second": batch_size * 1000.0 / mean_ms,
        "pair_scores_per_second": batch_size * batch_size * 1000.0 / mean_ms,
    }
    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated()
        result["peak_allocated_gib"] = peak / (1024.0 ** 3)
        result["incremental_peak_allocated_gib"] = (
            peak - baseline_allocated
        ) / (1024.0 ** 3)
    return result


def measure_training(wrapper, args, opt, device):
    if args.train_batch_size <= 0:
        return None
    wrapper.train()
    inputs = make_inputs(args.train_batch_size, args, opt, device)
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()

    for parameter in wrapper.parameters():
        parameter.grad = None
    synchronize(args)
    start = time.perf_counter()
    with autocast_context(args):
        scores = wrapper(*inputs)
        # A simple differentiable scalar exercises the same forward/backward
        # graph without changing checkpoint weights through an optimizer step.
        loss = scores.mean()
    loss.backward()
    synchronize(args)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    result = {
        "batch_size": args.train_batch_size,
        "forward_backward_ms": elapsed_ms,
    }
    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated()
        result["peak_allocated_gib"] = peak / (1024.0 ** 3)
        result["incremental_peak_allocated_gib"] = (
            peak - baseline_allocated
        ) / (1024.0 ** 3)
    for parameter in wrapper.parameters():
        parameter.grad = None
    return result


def measure_flops(wrapper, args, opt, device):
    if args.skip_flops:
        return {"status": "skipped"}
    try:
        from fvcore.nn import FlopCountAnalysis
        from fvcore.nn.jit_handles import get_shape
    except ImportError:
        return {
            "status": "unavailable",
            "error": "Install fvcore with: pip install 'fvcore>=0.1.5'",
        }

    wrapper.eval()
    inputs = make_inputs(1, args, opt, device)
    try:
        analysis = FlopCountAnalysis(wrapper, inputs)

        def value_numel(value):
            shape = get_shape(value)
            count = 1
            for dimension in shape or []:
                count *= int(dimension)
            return count

        def output_elements(op_inputs, op_outputs):
            return value_numel(op_outputs[0])

        def reduction_ops(op_inputs, op_outputs):
            return max(0, value_numel(op_inputs[0]) - value_numel(op_outputs[0]))

        def mean_ops(op_inputs, op_outputs):
            # Reduction additions plus one division per output element.
            return value_numel(op_inputs[0])

        def softmax_ops(op_inputs, op_outputs):
            # Common approximation: max, subtraction, exp, sum, division.
            return 5 * value_numel(op_outputs[0])

        def zero_ops(op_inputs, op_outputs):
            return 0

        # fvcore focuses on tensor contractions and leaves most elementwise
        # operations unsupported. Add explicit, documented handles so the
        # reported total includes CSAN's normalization and activation work.
        for op_name in (
            "aten::add", "aten::div", "aten::leaky_relu", "aten::mul",
            "aten::pow", "aten::sigmoid", "aten::sqrt", "aten::sub",
            "aten::tanh",
        ):
            analysis.set_op_handle(op_name, output_elements)
        analysis.set_op_handle("aten::sum", reduction_ops)
        analysis.set_op_handle("aten::mean", mean_ops)
        analysis.set_op_handle("aten::softmax", softmax_ops)
        analysis.set_op_handle("aten::embedding", zero_ops)
        analysis.set_op_handle("aten::repeat", zero_ops)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)

        # fvcore rejects numpy.int32 results produced by some handlers with the
        # NumPy/PyTorch versions used by the original CSAN environment. Convert
        # NumPy scalar counts to Python int/float before fvcore aggregates them.
        def wide_count_handle(handle):
            def wrapped(op_inputs, op_outputs):
                counts = handle(op_inputs, op_outputs)
                if hasattr(counts, "item"):
                    return counts.item()
                if hasattr(counts, "items"):
                    return Counter({
                        key: (value.item() if hasattr(value, "item") else value)
                        for key, value in counts.items()
                    })
                return counts
            return wrapped

        analysis._op_handles = {
            name: wide_count_handle(handle)
            for name, handle in analysis._op_handles.items()
        }
        fma_one = float(analysis.total())
        unsupported = {str(k): int(v)
                       for k, v in analysis.unsupported_ops().items()}
        uncalled = sorted(analysis.uncalled_modules())
        return {
            "status": "ok",
            "batch_size": 1,
            "fma_one_count": fma_one,
            "gflops_fma_equals_one": fma_one / 1e9,
            "gflops_multiply_add_equals_two": 2.0 * fma_one / 1e9,
            "unsupported_ops": unsupported,
            "uncalled_modules": uncalled,
            "note": (
                "Input is one image-caption pair. The offline Faster R-CNN "
                "feature extractor is excluded. Review unsupported_ops before "
                "using the value in a paper."
            ),
        }
    except Exception as error:
        return {"status": "failed", "error": repr(error)}


def write_csv(path, results):
    params = results["parameters"]
    inference = results["inference"]
    flops = results["flops"]
    row = {
        "total_parameters_m": params["total"] / 1e6,
        "trainable_parameters_m": params["trainable"] / 1e6,
        "gflops_fma_equals_one": flops.get("gflops_fma_equals_one", ""),
        "gflops_multiply_add_equals_two": flops.get(
            "gflops_multiply_add_equals_two", ""
        ),
        "batch_size": inference["batch_size"],
        "mean_ms_per_batch": inference["mean_ms_per_batch"],
        "median_ms_per_batch": inference["median_ms_per_batch"],
        "p95_ms_per_batch": inference["p95_ms_per_batch"],
        "aligned_inputs_per_second": inference["aligned_inputs_per_second"],
        "pair_scores_per_second": inference["pair_scores_per_second"],
        "peak_inference_memory_gib": inference.get("peak_allocated_gib", ""),
    }
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if args.precision == "fp16" and args.device != "cuda":
        raise ValueError("fp16 measurement is supported only with --device cuda.")
    if args.batch_size < 1 or args.seq_len < 1 or args.num_regions < 1:
        raise ValueError("batch_size, seq_len, and num_regions must be positive.")
    if args.num_regions != 36:
        raise ValueError(
            "The current CSAN VisualSA module is constructed for exactly 36 "
            "BUTD regions; use --num_regions 36."
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    # Import lazily so `python efficiency.py --help` works before the original
    # training dependencies have been installed.
    from model_copy import CSAN

    opt, checkpoint = load_options(args)
    model = CSAN(opt).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    wrapper = EfficiencyForward(model, opt, args.seq_len).to(device)

    inference_inputs = make_inputs(args.batch_size, args, opt, device)
    results = {
        "scope": {
            "included": "CSAN image projection, BERT, GAFM, and similarity module",
            "excluded": "offline Faster R-CNN/BUTD feature extraction",
            "num_regions": args.num_regions,
            "sequence_length": args.seq_len,
            "precision": args.precision,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": (torch.cuda.get_device_name(device)
                    if args.device == "cuda" else None),
        },
        "parameters": count_parameters(model),
        "flops": measure_flops(wrapper, args, opt, device),
        "inference": measure_inference(wrapper, inference_inputs, args),
        "training": measure_training(wrapper, args, opt, device),
    }

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
    write_csv(args.csv_output, results)

    print(json.dumps(results, indent=2, sort_keys=True))
    print("Saved JSON to {}".format(args.output))
    print("Saved CSV to {}".format(args.csv_output))


if __name__ == "__main__":
    main()


---
title: "CUDA, TensorRT, and the FP16 Softmax Overflow"
date: 2026-05-17
draft: false
tags:
  - deep-learning
  - computer-vision
  - tensorrt
  - optimization
  - onnx
  - cuda
description: "Taking ONNX models from 80 FPS (PyTorch CUDA) to 97 FPS (TensorRT mixed-precision), including a deep dive into a known TensorRT fusion bug that produces garbage in FP16."
aliases:
  - "optimizing-samurai-part-2"
series: "Optimizing SAMURAI"
series_order: 2
---

[Part 1](optimizing-samurai-part-1) left us with 4 ONNX modules (80 MB total, EfficientTAM-Ti @ 512) verified against PyTorch to `max_abs < 1e-3`. On CPU the pipeline runs at 187 ms per frame. The ONNX format was the prerequisite, not the destination.

This post covers what happens when you plug those same `.onnx` files into NVIDIA's execution stack: CUDA EP, TensorRT engine building, FP16 precision, and a bug in TensorRT's fused attention kernel that took days to diagnose and produced eight failed workarounds before I found the fix.

## CUDA Execution Provider: the silent fallback

ONNX Runtime's [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) runs ONNX graphs on NVIDIA GPUs. Setup looks trivial:

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "image_encoder.onnx",
    providers=["CUDAExecutionProvider"]
)
```

Except my first run was suspiciously slow. 34 ms for image_encoder, same as CPU. The issue: `onnxruntime-gpu==1.26.0` (stable) is built against **CUDA 12** (`libcublasLt.so.12`), but my workstation has CUDA 13. ORT can't find the CUDA 12 libraries, so it **silently falls back to `CPUExecutionProvider`** without any error.

You can detect this:

```python
# Always check which provider is actually running
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
actual = session.get_providers()
print(actual)  # ['CPUExecutionProvider'] ← bad! Should list CUDA first
```

My initial workaround was installing CUDA 12 runtime alongside CUDA 13 (`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`) and prepending those lib dirs to `LD_LIBRARY_PATH`. That works, but you end up maintaining two CUDA toolchains.

The actual fix: switch to [ORT nightly](https://onnxruntime.ai/docs/install/) builds which ship with CUDA 13 support:

```bash
pip install ort-nightly-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/
```

> [!warning]
> The nightly build bit us later: an engine cache invalidation bug caused TRT to silently load stale `.engine` files after ONNX changes. The fix was a newer nightly. The tradeoff between nightly (CUDA 13, latest features) and stable (CUDA 12, proven) is real. I chose nightly because the CUDA 12 lib juggling was worse than the occasional regression.

The broader lesson: **ORT-GPU and PyTorch-CUDA are independent CUDA toolchains.** They don't share runtime libraries. Check `session.get_providers()` before assuming anything.

### The numbers

Once CUDA EP is actually running (RTX 4090, FP32 ONNX, 50-iteration steady state):

| Sub-module | CPU EP | CUDA EP | Speedup |
|---|---|---|---|
| memory_encoder | 7.06 ms | **0.55 ms** | 12.8x |
| image_encoder | 34.52 ms | **2.85 ms** | 12.1x |
| mask_decoder | 3.53 ms | **0.86 ms** | 4.1x |
| memory_attention | 46.82 ms | **2.36 ms** | 19.8x |
| **Frame total** | **91.93 ms** | **6.59 ms** | **14x** |

### The surprise

Here's the cross-stack comparison that made me double-take:

| Stack | Frame total | FPS |
|---|---|---|
| ORT CPU EP, FP32 | 91.9 ms | 11 |
| PyTorch CUDA, FP32 | 12.4 ms | 80 |
| PyTorch CUDA, FP16 (autocast) | 6.1 ms | 163 |
| **ORT CUDA EP, FP32** | **6.59 ms** | **152** |

ORT CUDA EP running **FP32 weights** (6.59 ms) essentially matches **PyTorch with FP16 tensor cores** (6.1 ms). The static graph runtime closes the entire FP16 gap without changing precision. How?

- **Zero Python dispatch**: no GIL, no `torch.Tensor` wrapper allocation, no autograd bookkeeping
- **Operator fusion**: Conv+BatchNorm, Linear+GELU, and other patterns fused into single kernels
- **Persistent launch parameters**: kernel configurations computed once at session init, reused every frame
- **Memory planning**: the runtime pre-allocates all intermediate buffers at session creation

Another interesting shift: under ORT CUDA, `memory_attention` drops from 61% of frame time to 36%. `image_encoder` becomes the new bottleneck at 43%. The relative costs change once you remove enough overhead, which means the optimization target shifts.

## TensorRT: where the real 2x lives

ORT CUDA EP only gains ~9% from switching to FP16 ONNX (6.59 → 6.02 ms). It doesn't select tensor-core-friendly kernel implementations. For that, you need [TensorRT](https://developer.nvidia.com/tensorrt).

TRT goes further than ORT's CUDA EP: it fuses entire subgraphs (not just adjacent ops), selects tactics per-layer from a library of kernel implementations, performs memory layout optimization, and auto-tunes for the specific GPU at build time. The cost: engine builds take 20-80 seconds per module, and the resulting `.engine` files are GPU-specific (not portable).

```python
# Using TRT through ORT's TensorRT Execution Provider
session = ort.InferenceSession(
    "image_encoder.onnx",
    providers=[("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_cache/",
    })]
)
# First call is slow (engine build), subsequent calls use cached .engine
```

Install: `pip install tensorrt` pulls `tensorrt-cu13==10.16.1.11` (~700 MB). Add the TRT library path to `LD_LIBRARY_PATH` for ORT to find it.

### The numbers

| Stack | Frame total | FPS | vs CUDA EP |
|---|---|---|---|
| ORT CUDA EP, FP32 | 6.59 ms | 152 | baseline |
| ORT CUDA EP, FP16 | 6.02 ms | 165 | 1.09x |
| TRT EP, FP16 (from FP32 .onnx) | 2.86 ms | 349 | 2.30x |
| **TRT EP, FP16 (from native FP16 .onnx)** | **2.39 ms** | **418** | **2.76x** |

Per-module breakdown with TRT FP16 (from native FP16 ONNX):

| Sub-module | TRT FP16 | vs ORT CUDA FP16 |
|---|---|---|
| memory_encoder | 0.20 ms | 2.1x |
| image_encoder | 0.71 ms | 3.5x |
| mask_decoder | 0.35 ms | 2.4x |
| memory_attention | 1.13 ms | 2.0x |
| **Total** | **2.39 ms** | **2.53x** |

`image_encoder` gets the biggest TRT speedup (3.5x) because its Conv→LayerNorm→Linear→GELU stack is the ideal fusion target. `memory_attention` gets the smallest (2.0x) because attention softmax+matmul is already a single fused cuBLAS-LT kernel under CUDA EP.

### Why native FP16 ONNX is faster

When TRT receives FP32 ONNX and converts to FP16 internally, it inserts Cast nodes at subgraph partition boundaries. These break some fusion patterns. Exporting a native FP16 ONNX (via `model.half()` before `torch.onnx.export`) gives TRT a clean FP16 graph with no boundary Casts, enabling ~17% more aggressive fusion.

The FP16 export requires patching two hardcoded FP32 assumptions in EfficientTAM:

```python
# Patch 1: PositionEmbeddingRandom hardcodes float32
# Original: pe = torch.randn(..., dtype=torch.float32)
# Fix: follow the buffer's dtype
pe = torch.randn(..., dtype=self.positional_encoding_gaussian_matrix.dtype)

# Patch 2: PromptEncoder initializes sparse_embeddings as FP32
# Original: sparse_embeddings = torch.empty(...)
# Fix: match the weight dtype
sparse_embeddings = torch.empty(..., dtype=self.point_embeddings[0].weight.dtype)
```

## From isolated modules to a real pipeline

418 FPS sounds incredible. It's also misleading. That's four isolated module calls with pre-allocated tensors and no state management. A real tracking pipeline adds: frame preprocessing (resize + normalize), memory bank state updates between frames, Python orchestration logic, and the overhead of multiple TRT engines sharing a single CUDA stream.

Real-video chained inference (150 frames, RTX 4090):

| Config | Preprocess | Inference modules | Total | FPS |
|---|---|---|---|---|
| CPU EP | 8.1 ms | 113.4 ms | 121.5 ms | 8.2 |
| CUDA EP FP32 | 4.0 ms | 11.4 ms | 15.4 ms | 65 |
| TRT FP16 (initial) | 3.8 ms | 7.8 ms | 11.6 ms | 86 |
| **TRT FP16 (optimized)** | **0.5 ms** | **8.7 ms** | **9.15 ms** | **109** |

### Closing the preprocessing gap

The biggest non-model cost was NumPy preprocessing: normalize + transpose was taking 3.9 ms per frame, 40%+ of the total pipeline time. The fix was straightforward: in-place operations instead of allocating new arrays:

```python
# Before: 3.91 ms — allocates 3 new arrays
image = (image / 255.0 - mean) / std
image = image.transpose(2, 0, 1)  # HWC → CHW

# After: 0.48 ms — in-place ops, single allocation
image = image.astype(np.float32)
np.subtract(image, mean * 255, out=image)
np.multiply(image, 1.0 / (std * 255), out=image)
image = np.ascontiguousarray(image.transpose(2, 0, 1))
```

### Baking operations into the ONNX graph

Two more pipeline wins came from moving Python-level tensor operations *into* the ONNX graph. The idea: any operation you do in Python between model calls (reshape, upsample, transpose) could instead be an ONNX node. Once inside the graph, TRT can fuse it with adjacent layers or eliminate it entirely through layout optimization.

Here's what that looks like concretely using the [ONNX helper API](https://onnx.ai/onnx/api/helper.html):

**1. Upsample baked into mask_decoder**

The mask decoder outputs 128x128 logits. The pipeline upsamples to 512x512 in Python before computing the mask. Moving that into the graph:

```python
import onnx
from onnx import helper, TensorProto

model = onnx.load("mask_decoder.onnx")
graph = model.graph

# Add a Resize node after the final output
resize_node = helper.make_node(
    "Resize",
    inputs=[graph.output[0].name, "", "", "scales"],
    outputs=["mask_upsampled"],
    mode="linear",  # bilinear
)
graph.node.append(resize_node)

# Add scales as initializer: [1, 1, 4, 4] for 128→512
scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 4, 4])
graph.initializer.append(scales)

# Update graph output to the new node
# ...
```

TRT fuses this Resize with the preceding transposed convolution into a single kernel.

**2. Transpose baked into memory_attention**

The memory module expected `(T, B, C)` layout but the pipeline stored features as `(B, T, C)`. Rather than a Python `permute()` call every frame:

```python
# Add Transpose at the input
transpose_node = helper.make_node(
    "Transpose",
    inputs=["memory_input"],
    outputs=["memory_input_transposed"],
    perm=[1, 0, 2],  # (B, T, C) → (T, B, C)
)
graph.node.insert(0, transpose_node)
```

TRT typically eliminates these transposes entirely by choosing a compatible memory layout for the preceding layer's output.

**3. Low-resolution bounding box**: computing bboxes from the 128x128 mask directly (scaling coordinates instead of pixels) avoids the upsample entirely when you only need bounding box coordinates, not the full-resolution mask.

Each individually small (5-15%), but they compound.

### The memory_attention gap

One number stood out: `memory_attention` took 1.13 ms in isolation but 3.31 ms in the chained pipeline. The cause: four TRT engines share a single CUDA stream (the default). Each engine call forces a context switch. With async execution, the GPU pipeline bubbles between engines. This is addressable with multi-stream execution, but I left it on the table for now since the pipeline already exceeds real-time.

## The FP16 Softmax catastrophe

Here's where the story gets interesting. Everything above assumed FP16 "just works" on TRT. For three of four modules, it does. For image_encoder, it catastrophically doesn't.

### The symptom

Enabling TRT FP16 on the full pipeline:

| Module precision | mIoU | FPS |
|---|---|---|
| All FP32 (TRT) | 0.988 | 88 |
| **All FP16 (TRT)** | **0.119** | **112** |
| image_encoder FP32, rest FP16 | 0.982 | 103 |

mIoU collapses from 0.98 to 0.12. The model produces random-looking masks. Per-module bisection pinpoints `image_encoder` as the sole regressor: swapping just that one module to TRT FP16 destroys accuracy, while the other three pass cleanly.

### The TRT build warning

During engine building, TRT emits:

```
[TRT] [W] Detected layernorm nodes in FP16. Running layernorm after 
self-attention with FP16 Reduce or Pow may cause overflow.
```

This looks like the answer. It isn't.

### The actual root cause

I enabled verbose tactic logging in the TRT builder:

```python
import tensorrt as trt

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
# Enable verbose logging to see kernel selection
logger = trt.Logger(trt.Logger.VERBOSE)
```

The log reveals: in FP16 mode, **all 12 attention blocks** select the `_gemm_mha_v2` kernel. In FP32 mode: zero selections (uses unfused matmul/softmax tactics instead).

`_gemm_mha_v2` is TRT's fused Multi-Head Attention kernel. It computes `Q·K^T → Softmax → ·V` as a single kernel with FP16 I/O. The problem: **it materializes the raw attention scores `Q·K^T` as FP16 *before* applying Softmax.** For ViT attention logits where values exceed ~11, `exp(z) >= 65504 = FP16_MAX`, and the Softmax overflows to Inf.

This is a [known issue in TRT 10.x](https://github.com/NVIDIA/TensorRT/issues/4723). The standard mitigation in attention implementations is to compute `softmax(x - max(x))` which shifts values into a safe range. But TRT's fused kernel skips this stabilization step.

### Proving it: the probe model experiment

To confirm the fusion is the culprit, I modified the ONNX graph to materialize attention intermediate outputs (adding Identity nodes after each block's attention scores). With 12 extra outputs added, TRT can no longer form the `_gemm_mha_v2` fusion. Result: **correct output with the extra nodes, garbage without them.** Breaking the fusion fixes the accuracy.

The error also compounds: forcing just one block to materialize drops rel_err from 1.05 to 1.03. All 12 blocks: drops to 0.007. Each block adds a small FP16 overflow error that accumulates through the residual stream.

### Eight workarounds that failed

I tried every ORT TRT EP knob available:

| Strategy | Result | Why it failed |
|---|---|---|
| `trt_layer_norm_fp32_fallback=True` | No effect | LayerNorm isn't the issue |
| Force Pow/Sqrt/ReduceMean to FP32 | No effect | Same reason |
| `trt_op_types_to_exclude="Softmax"` | Accuracy fixed, no speedup | Graph fragments into 13 partitions |
| Insert Cast(fp16→fp32) around Softmax in ONNX | No effect | TRT constant-folds the Casts away |
| Cast-wrap + `trt_strict_type_constraints` | 0.16% error, 2.58 ms | Strict kills ALL fusion |
| Manual `softmax(x - max(x))` in ONNX graph | No effect | TRT pattern-matches and re-fuses |
| `trt_builder_optimization_level` 0 vs 5 | Identical | Bug is in tactic selection, not opt level |
| BF16 flag | Same error | Same `_gemm_mha_v2` kernel selected |

The BF16 result is particularly instructive: BF16 has FP32's dynamic range, which should avoid the overflow. But TRT selects the same broken kernel regardless of the precision flag. The kernel implementation has hardcoded FP16 intermediate storage.

### The fix: raw TensorRT API

ORT's TensorRT EP doesn't expose per-layer precision control at the granularity needed. The fix: bypass ORT entirely for engine building and use the [TRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) directly.

```python
import tensorrt as trt

def build_mixed_precision_engine(onnx_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    # Mark specific layers as FP32
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        # Force the score MatMul and Softmax to FP32
        # This prevents _gemm_mha_v2 selection
        if layer.type == trt.LayerType.MATRIX_MULTIPLY:
            if "attn" in layer.name and "score" in layer.name:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
        elif layer.type == trt.LayerType.SOFTMAX:
            layer.precision = trt.float32
            layer.set_output_type(0, trt.float32)

    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    engine = builder.build_serialized_network(network, config)
    return engine
```

The key difference from ORT's approach: setting `precision` and `output_type` on individual layers, combined with `OBEY_PRECISION_CONSTRAINTS`, forces TRT to respect the per-layer precision requests. ORT's `trt_op_types_to_exclude` takes a different approach (pulling ops out of TRT entirely), which fragments the graph.

Results:

| Engine | Latency | Rel error vs FP32 |
|---|---|---|
| FP16 (broken) | 0.81 ms | 69.2% |
| **Mixed (Softmax + score MatMul FP32)** | **0.84 ms** | **0.50%** |
| FP32 | 1.40 ms | 0.04% |

The mixed engine is 3.7% slower than broken-FP16 but **40% faster than pure FP32**. Accuracy is gold.

### ORT vs raw TRT: when to use which

ORT's TensorRT EP is excellent for the common case: drop in your ONNX, set `trt_fp16_enable=True`, and get 2-3x speedup. For three of my four modules, it worked perfectly with zero configuration.

The raw TRT API is needed when:
- You need per-layer precision control (this bug)
- ORT's EP knobs can't express your constraint
- You want to inspect tactic selections and fusion patterns

The cost: you own the memory management, stream synchronization, and I/O binding. That's manageable for 4 modules; it would be painful for 40.

## The CUDA stream race

A brief war story from the raw-TRT integration. After building the mixed-precision engine and wiring it into the pipeline, all modes (including FP32) produced mIoU 0.28. Not an FP16 issue since even the FP32 engine was wrong.

The bug: host-to-device copies used `non_blocking=True` on the default CUDA stream, but `execute_async_v3` ran on a custom stream. The TRT engine read stale (uninitialized) GPU memory.

```python
# Bug: copy on default stream, execute on custom stream
input_tensor.copy_(data, non_blocking=True)  # default stream
context.execute_async_v3(stream_handle=self._stream)  # custom stream
# TRT may execute before the copy finishes!

# Fix: ensure copy and execute share the same stream
with torch.cuda.stream(self._stream):
    input_tensor.copy_(data, non_blocking=True)
    context.execute_async_v3(stream_handle=self._stream.cuda_stream)
```

This is a classic async GPU bug: operations on different streams have no ordering guarantee unless you explicitly synchronize. It manifested as a "precision issue" (because the stale memory contained semi-plausible floating point values), which sent me down the wrong debugging path for an hour.

## Dropping PyTorch: cuda-python

The raw TRT session needs exactly four CUDA operations: allocate device memory, copy host→device, execute, copy device→host. Using `torch.cuda` for this works, but it pulls ~2 GB of transitive dependencies (PyTorch + all its CUDA libs). For a deployment runtime, that's excessive.

Replacement: [cuda-python](https://nvidia.github.io/cuda-python/) (~5 MB). Direct CUDA runtime bindings.

```python
from cuda.bindings import runtime as cudart

# Allocate device memory
err, d_ptr = cudart.cudaMalloc(nbytes)

# Create a stream
err, stream = cudart.cudaStreamCreate()

# Copy host → device (async)
(err,) = cudart.cudaMemcpyAsync(
    d_ptr, h_ptr, nbytes,
    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    stream
)

# Execute TRT engine
context.execute_async_v3(stream_handle=int(stream))

# Copy device → host
(err,) = cudart.cudaMemcpyAsync(
    h_ptr, d_ptr, nbytes,
    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    stream
)

# Synchronize
(err,) = cudart.cudaStreamSynchronize(stream)
```

Three gotchas for anyone adopting cuda-python 13.x:

1. **Import path changed**: `from cuda.bindings import runtime as cudart` (not the old `from cuda import cudart`)
2. **Void functions return 1-tuples**: `cudaMemcpyAsync` returns `(err,)` not bare `err`. Miss the comma and you get a confusing tuple-is-not-an-error TypeError.
3. **Stream handles need `int()`**: TRT's `execute_async_v3` expects a raw integer handle. The `cudaStream_t` wrapper from cuda-python needs explicit `int()` conversion.

Result: 91 → **97 FPS** (+7%) just from eliminating torch's import and dispatch overhead. Same model, same precision, same GPU. The runtime overhead of having PyTorch loaded (even unused) was measurable.

## Where we landed

Final validation (RTX 4090, 4 LaSOT person clips, 250 frames each):

| Clip | mIoU | FPS (p50) |
|---|---|---|
| person-1 | 0.907 | 97.6 |
| person-5 | 0.639 | 96.7 |
| person-10 | 0.873 | 97.4 |
| person-15 | 0.781 | 96.8 |
| **Mean** | **0.800** | **97.1** |

The full pipeline configuration:
- **Precision**: mixed (image_encoder attention Softmax/MatMul in FP32, everything else FP16)
- **Runtime**: raw TRT engines via cuda-python, no PyTorch at inference time
- **Model**: EfficientTAM-Ti @ 512, 80 MB ONNX bundle, ~200 MB TRT engines (GPU-specific, cached)

### The performance arc through this post

Two views of progress. First, the isolated module benchmark (same 4 modules, same RTX 4090, no pipeline overhead, apples-to-apples):

| Stack | Frame total | FPS | mIoU |
|---|---|---|---|
| PyTorch CUDA FP32 (Part 1 baseline) | 12.4 ms | 80 | 0.988 |
| ORT CUDA EP FP32 | 6.59 ms | 152 | 0.988 |
| ORT CUDA EP FP16 | 6.02 ms | 165 | 0.986 |
| TRT FP16 (all modules) | 2.39 ms | 418 | 0.119 (broken) |
| **TRT mixed precision (Softmax fix)** | **2.58 ms** | **388** | **0.982** |

Second, the real end-to-end pipeline (150+ frames, video input, all overhead included):

| Configuration | FPS | mIoU | Note |
|---|---|---|---|
| PyTorch CUDA FP32 | 80 | 0.988 | Part 1 baseline |
| ORT CUDA EP FP32 pipeline | 65 | 0.988 | Python orchestration overhead |
| TRT FP16 pipeline (naive) | 86 | 0.119 | Broken accuracy |
| TRT mixed precision pipeline | 91 | 0.982 | FP16 Softmax fixed |
| + Preprocessing optimized | 109 | 0.982 | In-place numpy |
| + Baked ONNX ops | 109 | 0.982 | Removes Python tensor ops |
| **+ cuda-python (drop PyTorch)** | **97** | **0.800** | **Final (different eval clips)** |

> [!note]
> The final 97 FPS number comes from the full LaSOT validation set (4 clips, 250 frames each), which is a harder evaluation than the 150-frame single-clip benchmarks above. The accuracy drop from 0.982 to 0.800 mIoU is the evaluation set difference, not a regression.

The gap between isolated modules (388 FPS) and real pipeline (97 FPS) is the engineering reality: preprocessing, state management, multi-engine stream serialization, and evaluation on harder clips. Each component is individually addressable, but together they define the current ceiling.

97 FPS at 0.800 mIoU exceeds the 30 FPS real-time bar by 3x on an RTX 4090. For the drone cinematography use case, this is more than sufficient on desktop-class hardware.

**Next up**: [Part 3](optimizing-samurai-part-3) takes the same ONNX models to Apple Silicon with CoreML, where the constraints are completely different: no CUDA, no TensorRT, unified memory architecture, and a Neural Engine that only speaks certain graph patterns.

---

## References

- [ONNX Runtime CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [NVIDIA/TensorRT#4723](https://github.com/NVIDIA/TensorRT/issues/4723), FP16 MHA overflow bug
- [microsoft/onnxruntime#22713](https://github.com/microsoft/onnxruntime/issues/22713), TRT cache + exclude interaction
- [cuda-python](https://nvidia.github.io/cuda-python/), NVIDIA CUDA Python bindings
- [EfficientTAM](https://github.com/yformer/EfficientTAM)

# Precision — the dtype map for π-family training & inference

What runs in bf16, what deliberately doesn't, and why. This is the config the
π0 / π0.5 / π0-FAST wrappers actually implement (matching how lerobot/openpi
ship these models) — bf16 is the backbone, but the correct setup is **not**
uniform: precision is spent where averaging happens and saved everywhere bulk.

Related: [`quantization.md`](quantization.md) (NF4/int8 — storage, not compute),
[`runpod_training.md`](runpod_training.md) (the knobs that select all of this).

---

## 1. Training dtype map (default: `QUANTIZE="none"`, LoRA)

| tensor | dtype | why |
|---|---|---|
| base VLM weights (frozen) | **bf16** | storage + fast matmuls; frozen, so update precision is irrelevant |
| action expert weights (trained) | **bf16** | lerobot casts the whole policy via `dtype: "bfloat16"` |
| **LoRA adapters (trained)** | **fp32** | deliberate — `_inject_vlm_lora` calls `p.data.float()` on every `lora_*` param. They are tiny (~10–20 M), and AdamW on fp32 params keeps the small updates that bf16's 8-bit mantissa would round to zero |
| activations | **bf16** | follow the weights (peft casts in/out of the fp32 adapters internally) |
| gradients | match their param: fp32 (adapters) / bf16 (expert) | PyTorch grads inherit param dtype |
| AdamW moments | same split as gradients | fp32 exactly where it counts |
| **the loss** | **fp32** | lerobot upcasts the expert output before the flow-matching loss (`suffix_out.to(torch.float32)`) — the reduction is the numerically fragile part |

With `QUANTIZE="nf4"` only one row changes: the frozen VLM's **storage** becomes
4-bit (dequantized to bf16 at matmul time), and the wrapper additionally forces
the forward under `torch.autocast("cuda", bf16)` (`PiPolicy._amp()`) so the fp32
adapters can't upcast activations into the bf16 expert
(`mat1 float != mat2 BFloat16` — see [`quantization.md`](quantization.md) §2).

---

## 2. Inference dtype map

| mode | weights | compute | footprint (π0) |
|---|---|---|---|
| standard (`deploy.py`) | bf16 | bf16 | ~7 GB |
| low-VRAM (`deploy.py --quantize nf4`) | VLM 4-bit **storage**, expert bf16 | **bf16** | ~3–4 GB |
| off-CUDA (CPU/MPS) | fp32 | fp32 | not real-time — CUDA is required at 30 Hz anyway |

Inference in bf16 is the *expected* config, not a downgrade: it is the precision
the checkpoint trained in. 4-bit never changes compute dtype — storage only.

---

## 3. The three rules behind the map

1. **bf16, never fp16.** Both are 16-bit, but bf16 keeps fp32's exponent range
   and gives up mantissa. That is why there is **no GradScaler anywhere** in the
   training loop — bf16 does not overflow the way fp16 does. PaliGemma-class
   models are known to be unstable in fp16; if fp16 is ever suggested for these,
   decline.
2. **fp32 exactly where averaging happens:** trainable-adapter params and their
   optimizer moments, and the loss reduction. Everything bulk — the billions of
   frozen/large params and all activations — is bf16.
3. **4-bit is storage, never compute.** NF4 changes what sits in VRAM, not the
   dtype any matmul runs in.

---

## 4. Known caveat (and the escape hatch)

The fully-trained **action expert** runs bf16 params with bf16 AdamW moments —
how lerobot ships it, and fine at our LR (peak 2.5e-5) and step budget (30k).
It is, however, less conservative than classical mixed precision (fp32 master
weights + bf16 autocast compute): with pure-bf16 optimizer state, updates
smaller than ~1/256 of a weight's magnitude can round away.

**Symptom to watch for:** loss plateaus suspiciously early while grad norms stay
healthy. **Escape hatch:** set `dtype: "float32"` (model config) and rely on the
autocast for bf16 compute — fp32 master weights + fp32 moments at ~2× the
expert's memory. With current headroom that is affordable; don't reach for it
without the symptom.

---

## 5. Where each piece lives in code

| behavior | location |
|---|---|
| whole-policy bf16 cast | lerobot `PI0Config/PI05Config(dtype="bfloat16")`, threaded from our `dtype` knob |
| LoRA params forced fp32 | `_inject_vlm_lora` (`common/vla_pretrained.py` + inline notebook copy) — `p.data.float()` |
| loss upcast to fp32 | lerobot `modeling_pi05.py` / `modeling_pi0.py` — `suffix_out.to(torch.float32)` before the loss |
| autocast under quantization | `PiPolicy._amp()` in the wrappers — active only when the VLM is k-bit |
| NF4 storage / bf16 compute | `quantize_vlm` (`compute_dtype=torch.bfloat16`) |
| inference PTQ | `deploy.py --quantize` → `quantize_for_inference` |

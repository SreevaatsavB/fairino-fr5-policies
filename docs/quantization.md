# Quantization — QLoRA training & post-training inference (π-family only)

Quantization only applies to the **π0 / π0.5 / π0-FAST** VLAs — they carry a
2–3 B-parameter PaliGemma vision-language model, which is what makes them heavy.
ACT, Diffusion Policy, and DiT are small and run fine in bf16/fp32 unquantized.

There are **two distinct uses**, in opposite orders:

1. **QLoRA training** — quantize the frozen VLM *before* training so a big model
   fits on a modest GPU. (`QUANTIZE="nf4"` in the notebooks.)
2. **Post-training inference quantization** — quantize a *trained* checkpoint at
   load time so it runs on a low-VRAM robot box. (`deploy.py --quantize nf4`.)

Getting the order right is the whole trick, and it differs between the two — see §4.

---

## 1. What NF4 / int8 actually do

Nothing here is plain half-precision. `dtype: bfloat16` is 16-bit *floating point*
— that is **not** quantization. Quantization stores weights in **4 or 8 bits**:

- **NF4** (4-bit NormalFloat, via bitsandbytes): weights stored as 4-bit codes on a
  distribution tuned for neural-net weights, with **double quantization** (the
  quantization constants are themselves quantized). Dequantized to bf16 **at matmul
  time**, so compute stays bf16-fast; only storage shrinks. The 2 B VLM goes from
  **~4.6 GB → ~1.4 GB**.
- **int8** (LLM.int8()): 8-bit weights with an fp16 outlier path. ~2× shrink, a
  touch higher quality than NF4, less savings.

The cost is a small, well-documented accuracy hit that LoRA adapters largely absorb.

### What is and isn't quantized

`quantize_vlm` (in `common/vla_pretrained.py`, mirrored inline in the notebooks)
swaps every `nn.Linear` **inside the PaliGemma tower** for a bitsandbytes 4-/8-bit
layer, moving each to the GPU immediately so the full-precision copy is freed
layer-by-layer (peak stays at one layer, not a second whole model).

**Deliberately left unquantized:**

| kept in bf16 | why |
|---|---|
| the **action expert** (Gemma-300M) | it is **fully trained**; 4-bit weights take no gradient, so quantizing it would silently freeze the one part that must learn |
| `lm_head` / token embeddings | vocabulary-sized and weight-tied; π0-FAST extends them with action tokens that must stay trainable |
| LayerNorms | 1-D, negligible memory, and quantization hurts them most |

Because the base is frozen once it's 4-bit, quantized training **requires** LoRA
(or an explicitly-trained expert) — NF4 with neither would train nothing, and the
code refuses it.

---

## 2. QLoRA training (the notebooks)

**When to use it:** only when VRAM forces you. The official openpi finetune recipe
is plain **bf16 LoRA — no quantization** (30k steps, batch 32, LoRA rank 16 on the
VLM's attention *and* MLP). NF4 matches 16-bit quality (the QLoRA result) but pays
~20-30% per-step slowdown for dequantization — so on a 40 GB+ card, bf16 LoRA is
both faster and the recipe the base model was tuned by; NF4 earns its keep on
<=24 GB cards where bf16 LoRA doesn't fit. The notebooks default to
`QUANTIZE="none"` accordingly.

Set in the parameters cell:

```python
QUANTIZE     = "nf4"     # none | nf4 | int8
FINETUNE_MODE = "auto"   # nf4 forces this to "lora"
LORA_RANK    = 16
```

What you get on a 46 GB A40 (measured): base VLM NF4 ~1.4 GB, action expert bf16,
~6.7 M LoRA adapter params, **~700 M trainable / ~2.25 B frozen**, and a training
step that sits around ~13–30 GB depending on batch — leaving lots of headroom (see
[`runpod_training.md`](runpod_training.md) §3 on using it).

### The forward runs under autocast

QLoRA keeps the LoRA adapters in **fp32** (for stable AdamW). lerobot's π-family
forward has no autocast, so those fp32 adapters would upcast activations to fp32,
which then collide with the **bf16 action expert** →
`RuntimeError: mat1 float != mat2 BFloat16`. The wrappers fix this by autocasting
the policy forward/`select_action` to bf16 whenever the VLM is k-bit quantized
(`PiPolicy._amp()`), so every matmul is bf16 regardless of the fp32 adapter dtype.
No-op when not quantized.

### Requirements

- **bitsandbytes** (installed by the notebook), **CUDA only**.
- NF4/int8 kernels need a **Turing-or-newer GPU** (compute capability ≥ 7.5): T4,
  RTX 20xx/30xx/40xx, A-series, L-series. Pre-Turing (GTX 10xx, P100) will not work.

---

## 3. Post-training inference quantization (deploy.py)

You do **not** need to have trained quantized. Quantizing trained bf16 weights is
standard post-training quantization (PTQ) — it's exactly what QLoRA does at its
start, just applied to your finetuned weights.

```bash
# a checkpoint trained in bf16, quantized for a small robot-box GPU:
python common/deploy.py --checkpoint best.pt --quantize nf4
python common/deploy.py --hf-repo <you>/fr5-pi0-lora --quantize nf4   # straight from the Hub
```

π0 lands at **~3–4 GB total** in NF4 (VLM ~1.4 GB + bf16 expert ~0.6 GB +
activations), so it fits an 8 GB card, easily a T4/16 GB. Same Turing+ requirement
as above. Expect a small accuracy dip — measure it (see §5) before trusting it on
the arm.

`--quantize` only applies to `pi0`/`pi05`/`pi0_fast`; it errors on other policies.

---

## 4. Why the order matters (and differs)

Quantization is **destructive**: once a `Linear` is 4-bit `Params4bit`, you cannot
`load_state_dict` float weights into it. So the two paths build in opposite orders:

**Training** (weights already loaded from the base, then adapt):
```
build → load pretrained base → quantize VLM → inject LoRA
```
Quantize *after* the real base weights are in (NF4-ing random weights then loading
over them does not round-trip). LoRA *after* quantize, because peft must see the
4-bit `Linear4bit` to build `lora.Linear4bit` wrappers.

**Inference from a checkpoint** (`deploy.py --quantize`):
```
build UNQUANTIZED → load_state_dict(checkpoint) → merge LoRA → quantize VLM
```
Here the trained weights come from the checkpoint, so we must build unquantized,
load, and **only then** quantize. deploy forces the build to `quantize="none"`
regardless of what the checkpoint's config says, then calls
`quantize_for_inference`.

**The LoRA merge** is essential and easy to get wrong: injected LoRA layers are
*not* `nn.Linear` (they're peft wrappers), so quantizing without merging would
4-bit the base and the adapters separately and corrupt the learned update.
`quantize_for_inference` first folds every adapter into its base
(`merge()`, function-preserving) and replaces the wrapper with the plain updated
`nn.Linear`, then quantizes the clean tree. Works for both LoRA and full-finetune
checkpoints.

---

## 5. Is NF4 good enough? Measure it

Quantization trades a little accuracy for a lot of memory. Whether that trade is
acceptable on *your* task is an empirical question — don't guess:

```bash
# offline GT-vs-prediction MAE, bf16 vs NF4, on the same held-out episodes:
python experiments/eval_on_test.py --ckpt best.pt
python experiments/eval_on_test.py --ckpt best.pt --quantize nf4
```

Or in the notebook, run the full-eval cell (12c) once unquantized and once with the
model quantized, and compare `summary.csv` MAE per episode. If the dip is small,
NF4 buys you a much cheaper deployment; if a specific joint degrades, keep bf16 for
the robot and use NF4 only where memory forces it.

---

## 6. Cheat sheet

| I want to… | do this |
|---|---|
| train on a 40 GB+ pod | `QUANTIZE="none"` — bf16 LoRA, the official openpi recipe (faster too) |
| train on a ≤24 GB pod | `QUANTIZE="nf4"`, `FINETUNE_MODE="auto"` (→ lora) |
| full finetune on 80 GB | `QUANTIZE="none"`, `FINETUNE_MODE="full"` |
| deploy a bf16 checkpoint on a big GPU | `deploy.py --checkpoint best.pt` (no `--quantize`) |
| deploy on a small robot GPU | `deploy.py --hf-repo <repo> --quantize nf4` |
| check the accuracy cost of NF4 | eval bf16 vs NF4 MAE (§5) |

Requirements recap: bitsandbytes, CUDA, **compute capability ≥ 7.5**. Action expert,
lm_head, embeddings and norms are never quantized. A quantized model is frozen —
QLoRA adapters (training) or the merged update (inference) carry everything learned.

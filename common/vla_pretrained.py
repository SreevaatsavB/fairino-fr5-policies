"""
common/vla_pretrained.py — shared helpers for the π0-family VLA policies
(pi0, pi05, pi0_fast): construction, verified pretrained-weight loading,
optional k-bit quantization, and VLM LoRA injection.

Why this exists: lerobot's PI0Policy/PI05Policy constructors build PaliGemma
FROM CONFIG (random init, no download), and their from_pretrained loads with
strict=False, printing — not raising — on missing keys. Under transformers
>= 5.4 (which dropped the `.vision_model` nesting inside SigLIP) that silently
leaves the ENTIRE vision tower random-init. These helpers load with a
version-proof remap and refuse to continue on a partial match.

The build order the wrappers follow, and why it is that order:

    with build_context(device, dtype):      # 1. construct ON the GPU, in bf16
        policy = PI05Policy(lerobot_cfg)
    warn_or_load_pretrained(policy, cfg)    # 2. real weights (fp) — must precede (3)
    quantize_vlm(policy, cfg.quantize, ...) # 3. NF4 the frozen VLM
    _inject_vlm_lora(policy, ...)           # 4. adapters on top -> QLoRA

(2) before (3) because quantization is destructive: NF4-ing random weights and
then load_state_dict-ing over Params4bit does not round-trip. (3) before (4)
because peft must see Linear4bit to build lora.Linear4bit wrappers.
"""

from contextlib import contextmanager

_TORCH_DTYPES = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16":  "float16",  "fp16": "float16", "half": "float16",
    "float32":  "float32",  "fp32": "float32", "float": "float32",
}


@contextmanager
def build_context(device, dtype: str = None):
    """Construct a VLA directly on `device` — never staged through host RAM.

    lerobot builds the policy wherever torch's default device points (CPU), then
    `.to(config.device)` at the end. For pi0/pi05 that means ~3.5B params
    materialise as fp32 in HOST RAM first: ~14 GB of allocation churn that
    OOM-kills a container with a modest cgroup memory limit ("the kernel appears
    to have died") long before the GPU is ever touched. Pointing torch's default
    device at the GPU for the duration of __init__ makes every nn.Linear
    allocate straight into VRAM, so peak host RAM stays near zero and the
    trailing `.to(device)` becomes a no-op.

    `dtype` additionally redirects torch's default dtype, halving the build's
    peak VRAM (~14 GB fp32 -> ~7 GB bf16). It is OFF by default because it is
    not numerically free: anything computed at construction time lands in bf16
    rather than fp32. Parameters do not care (the pretrained load overwrites
    every one of them), but a buffer derived at __init__ — RoPE inverse
    frequencies being the classic case — would keep the reduced precision.
    transformers guards inv_freq with an explicit `.float()`, so bf16 is
    believed safe here; pass it only when the fp32 build genuinely will not fit,
    which on a 40 GB+ card it does.

    No-op on CPU/MPS, where there is nowhere better to build than the default.
    """
    import torch

    want_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not want_cuda:
        yield
        return

    prev = torch.get_default_dtype()
    if dtype is not None:
        # default dtype only accepts floating types; bf16/fp16/fp32 all qualify.
        torch.set_default_dtype(getattr(torch, _TORCH_DTYPES.get(str(dtype).lower(), "float32")))
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(prev)


def quantize_vlm(policy, mode: str, tag: str, *, lora_rank: int = 0,
                 expert_only: bool = False, compute_dtype=None):
    """QLoRA-style k-bit quantization of the FROZEN VLM, in place. No-op if off.

    Swaps every nn.Linear inside the PaliGemma tower for a bitsandbytes
    Linear4bit (NF4 + double quantization) or Linear8bitLt, then moves each one
    to the GPU IMMEDIATELY — bitsandbytes quantizes lazily on `.cuda()`, so
    doing it layer-by-layer frees each full-precision weight as we go and keeps
    the peak footprint at one layer rather than a second copy of the model.

    NF4 stores weights as 4-bit normal-float with fp16/bf16 dequantization at
    matmul time: the 2B VLM drops ~4.6 GB (bf16) -> ~1.4 GB, which is what buys
    back the headroom for a larger batch size. Matmuls still run in
    `compute_dtype`, so throughput is close to bf16; the cost is a small,
    well-documented quality hit that LoRA adapters largely absorb.

    Deliberately NOT quantized:
      • the action expert — it is FULLY trained; 4-bit weights cannot take a
        gradient, so quantizing it would silently freeze the one part that must
        learn. Only `.paligemma` is touched.
      • lm_head / embeddings — vocabulary-sized and weight-tied, and pi0_fast
        extends them with FAST action tokens that must stay trainable.
      • norms — 1-D, negligible memory, and quantization hurts them most.

    Requires adapters (or an explicitly trained action expert), because a 4-bit
    base is frozen by construction: NF4 + no LoRA + no expert would train
    nothing at all.
    """
    import torch
    import torch.nn as nn

    mode = (mode or "none").lower()
    if mode in ("none", "off", "false", ""):
        return
    if mode not in ("nf4", "int8"):
        raise ValueError(f"[{tag}] model.quantize must be one of "
                         f"'none' | 'nf4' | 'int8', got {mode!r}")
    if lora_rank <= 0 and not expert_only:
        raise ValueError(
            f"[{tag}] quantize={mode!r} freezes the VLM (4-bit weights take no "
            f"gradient) but vlm_lora_rank=0 and train_expert_only=False — nothing "
            f"would train. Set model.vlm_lora_rank > 0 for QLoRA.")
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise RuntimeError(
            f"[{tag}] model.quantize={mode!r} needs bitsandbytes "
            f"(pip install bitsandbytes)") from e
    if not torch.cuda.is_available():
        raise RuntimeError(f"[{tag}] bitsandbytes k-bit quantization is CUDA-only")

    compute_dtype = compute_dtype or torch.bfloat16
    vlm = policy.model.paligemma_with_expert.paligemma
    skip = ("lm_head", "embed_tokens", "embed_out")

    n_swapped, saved = 0, 0

    def _swap(module, prefix=""):
        nonlocal n_swapped, saved
        for name, child in list(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(s in path for s in skip):
                w, b = child.weight.data, (child.bias.data if child.bias is not None else None)
                if mode == "nf4":
                    new = bnb.nn.Linear4bit(
                        child.in_features, child.out_features,
                        bias=b is not None, compute_dtype=compute_dtype,
                        quant_type="nf4", compress_statistics=True,  # double-quant
                    )
                    new.weight = bnb.nn.Params4bit(
                        w.to(compute_dtype), requires_grad=False,
                        quant_type="nf4", compress_statistics=True)
                else:
                    new = bnb.nn.Linear8bitLt(
                        child.in_features, child.out_features,
                        bias=b is not None, has_fp16_weights=False, threshold=6.0)
                    new.weight = bnb.nn.Int8Params(
                        w.to(torch.float16), requires_grad=False, has_fp16_weights=False)
                if b is not None:
                    new.bias = nn.Parameter(b.to(compute_dtype), requires_grad=False)
                # move NOW: this is where bnb actually quantizes, and it lets the
                # full-precision weight above be freed before the next layer is built.
                setattr(module, name, new.to("cuda"))
                n_swapped += 1
                saved += w.numel() * (w.element_size() - (0.5 if mode == "nf4" else 1))
                del w, b, child
            else:
                _swap(child, path)

    _swap(vlm)
    torch.cuda.empty_cache()

    # QLoRA + gradient checkpointing: the checkpointed blocks sit behind a fully
    # frozen base, so without an input that requires grad the recomputed graph is
    # detached and the adapters receive no gradient at all (silent no-op training).
    if getattr(policy.config, "gradient_checkpointing", False):
        try:
            vlm.enable_input_require_grads()
        except AttributeError:
            print(f"[{tag}] WARNING: could not enable input grads on the VLM; if "
                  f"LoRA grads come back None, set gradient_checkpointing: false")

    print(f"[{tag}] quantized VLM to {mode.upper()}: {n_swapped} Linear layers, "
          f"~{saved/1e9:.1f} GB saved (action expert + lm_head left in "
          f"{str(compute_dtype).split('.')[-1]})")

def _load_pretrained_weights(policy, repo_id: str, tag: str):
    """Load openpi-ported weights with version-proof key remapping + a HARD check.

    lerobot's own from_pretrained loads with strict=False and only prints missing
    keys — under transformers >= 5.4 (which dropped the `.vision_model` nesting
    inside SigLIP) that silently leaves the ENTIRE vision tower random-init
    (437 tensors, verified against lerobot/pi0_base). We remap and refuse to
    continue unless (nearly) everything matched."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download(repo_id, "model.safetensors"))
    sd = policy._fix_pytorch_state_dict_keys(sd, policy.config)
    sd = {(k if k.startswith("model.") else f"model.{k}"): v for k, v in sd.items()}

    model_keys = set(policy.state_dict().keys())
    if (any(".vision_tower.vision_model." in k for k in sd)
            and not any(".vision_tower.vision_model." in k for k in model_keys)):
        sd = {k.replace(".vision_tower.vision_model.", ".vision_tower."): v
              for k, v in sd.items()}

    missing, unexpected = policy.load_state_dict(sd, strict=False)
    n_loaded = len(model_keys) - len(missing)
    print(f"[{tag}] pretrained load: {n_loaded}/{len(model_keys)} tensors from "
          f"{repo_id} ({len(unexpected)} unexpected ignored)")
    if n_loaded < 0.99 * len(model_keys):
        raise RuntimeError(
            f"[{tag}] only {n_loaded}/{len(model_keys)} tensors matched {repo_id} — "
            f"a partial load silently finetunes random weights. First missing keys: "
            f"{sorted(missing)[:5]}")


def _inject_vlm_lora(policy, rank: int, alpha: int, dropout: float, tag: str):
    """LoRA adapters on the (frozen) VLM's attention projections, in place.

    peft's inject_adapter_in_model mutates the Linear layers without wrapping or
    renaming the module tree, so lerobot's internal attribute paths keep working.
    Adapter params are kept fp32 for stable AdamW on top of bf16 base weights."""
    from peft import LoraConfig, inject_adapter_in_model

    vlm = policy.model.paligemma_with_expert.paligemma
    inject_adapter_in_model(
        LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                   bias="none"),
        vlm)
    n_lora = 0
    for n, p in policy.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
            p.requires_grad_(True)
            n_lora += p.numel()
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[{tag}] LoRA r={rank} on VLM q/k/v/o: {n_lora/1e6:.1f}M adapter params; "
          f"total trainable {n_train/1e6:.0f}M (frozen base VLM + full action expert)")


# per-policy base checkpoint — used only to make the random-init warning concrete.
_BASE_HINT = {
    "pi0":      "lerobot/pi0_base",
    "pi05":     "lerobot/pi05_base",
    "pi0_fast": "lerobot/pi0fast_base",
}


def warn_or_load_pretrained(policy, cfg, tag: str):
    """Wrapper entry point: load the pretrained base, or LOUDLY flag random init.

    An empty cfg.pretrained is only correct when a full checkpoint is loaded right
    after build_model (deploy / eval / probe — see strip_pretrained_for_checkpoint).
    For a TRAINING run it means the VLA starts from random weights and learns
    nothing, so the message is loud AND names the exact base repo to set. Single
    source of truth so the three wrappers can't drift."""
    if cfg.pretrained:
        _load_pretrained_weights(policy, cfg.pretrained, tag)
    else:
        base = _BASE_HINT.get(tag, "the base checkpoint")
        print(f"[{tag}] pretrained='' — RANDOM-INIT base. Expected ONLY when a full "
              f"checkpoint is loaded right after (deploy / eval / probe). If you are "
              f"TRAINING, set model.pretrained={base} or the run learns nothing.")


def strip_pretrained_for_checkpoint(cfg_dict: dict):
    """For checkpoint loaders (deploy / eval / probe): force model.pretrained empty
    before build_model so it never re-downloads the gated ~6 GB base — the
    checkpoint's model_state carries every weight (base VLM + LoRA + expert +
    buffers), which load_state_dict restores immediately after. Mutates cfg_dict.

    Set UNCONDITIONALLY on purpose: every VLA build_model falls back to its default
    base id when the key is absent (m.get("pretrained", "lerobot/..") or ""), so a
    checkpoint whose saved config omitted the key would still resurrect the download
    if we only nulled a present-and-truthy value. Forcing "" also covers the case of
    an explicit `pretrained` override, which would be pointless here anyway since
    load_state_dict overwrites the base a line later. No-op for non-VLA policies,
    whose build_model never reads model.pretrained."""
    m = cfg_dict.setdefault("model", {})
    prev = m.get("pretrained")
    m["pretrained"] = ""
    if prev:
        print(f"[checkpoint] skip pretrained download ({prev}) — weights load from the checkpoint")
    return cfg_dict

"""
common/vla_pretrained.py — shared helpers for the π0-family VLA policies
(pi0, pi05): verified pretrained-weight loading and VLM LoRA injection.

Why this exists: lerobot's PI0Policy/PI05Policy constructors build PaliGemma
FROM CONFIG (random init, no download), and their from_pretrained loads with
strict=False, printing — not raising — on missing keys. Under transformers
>= 5.4 (which dropped the `.vision_model` nesting inside SigLIP) that silently
leaves the ENTIRE vision tower random-init. These helpers load with a
version-proof remap and refuse to continue on a partial match.
"""

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


def warn_or_load_pretrained(policy, cfg, tag: str):
    """Wrapper entry point: load the pretrained base, or LOUDLY flag random init.

    An empty cfg.pretrained is only correct when a full checkpoint is loaded right
    after build_model (deploy / eval / probe — see strip_pretrained_for_checkpoint).
    For a TRAINING run it means the VLA starts from random weights and learns
    nothing, so the message is loud enough to catch that case. Single source of
    truth so the three wrappers can't drift."""
    if cfg.pretrained:
        _load_pretrained_weights(policy, cfg.pretrained, tag)
    else:
        print(f"[{tag}] pretrained='' — RANDOM-INIT base. Expected ONLY when a full "
              f"checkpoint is loaded right after (deploy / eval / probe). If you are "
              f"TRAINING, set model.pretrained to a base checkpoint or the run learns "
              f"nothing.")


def strip_pretrained_for_checkpoint(cfg_dict: dict, model_overrides: dict | None = None):
    """For checkpoint loaders (deploy / eval / probe): null model.pretrained before
    build_model so it does NOT re-download the gated ~6 GB base only to overwrite it —
    the checkpoint's model_state carries every weight (base VLM + LoRA + expert +
    buffers). Mutates and returns cfg_dict.

    Respects an explicit `pretrained` in model_overrides: if the caller deliberately
    asked to (re)build from a hub base, that wins and nothing is stripped."""
    m = cfg_dict.get("model", {})
    if "pretrained" in (model_overrides or {}):
        return cfg_dict                      # caller explicitly chose a base — honour it
    if m.get("pretrained"):
        print(f"[checkpoint] skip pretrained download ({m['pretrained']}) — "
              f"weights load from the checkpoint")
        m["pretrained"] = ""
    return cfg_dict

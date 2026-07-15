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

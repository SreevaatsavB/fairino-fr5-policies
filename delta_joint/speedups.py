"""
The four things the pi0 benchmark run left on the table, as testable helpers so
the notebook stays thin. Each one is independently useful; none touches common/.

  enable_compile     lerobot exposes compile_model (configuration_pi0.py:82) and
                     really calls torch.compile (modeling_pi0.py:590-594), but our
                     _lerobot_config() never passes it. Typically 1.3-1.8x.
  full_lora_targets  our target list uses Gemma's layer names throughout, so
                     SigLIP only gets q/k/v — verified: 18*7 + 27*3 = 207, exactly
                     the "merged 207 LoRA pairs" in the deploy gate log. SigLIP's
                     out_proj and its whole MLP were never adapted.
  warmup_cosine      common/train.py builds a bare AdamW with NO scheduler. The
                     warmup+cosine the 30k run used lives only in the notebook.
  init_from          warm-start weights WITHOUT the stats buffers. model.py
                     registers action_mean/action_std as buffers, so a plain
                     load_state_dict silently restores the old ABSOLUTE stats over
                     fresh delta ones and cancels the entire SNR gain.
"""

import math

import torch

# openpi's reference pairing: lr 2.5e-5 at batch 32 (config.py). LeRobot does no
# LR auto-scaling, so scale it here when the batch changes.
OPENPI_LR, OPENPI_BATCH = 2.5e-5, 32

# stats live in the state_dict as buffers; these must never be warm-started
STATS_BUFFERS = ("action_mean", "action_std")

# SigLIP names its projections out_proj/fc1/fc2; Gemma uses o_proj/gate/up/down.
# A list written for one leaves the other partly unadapted.
FULL_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",        # attention (both)
                     "gate_proj", "up_proj", "down_proj",           # Gemma MLP
                     "out_proj", "fc1", "fc2")                      # SigLIP attn-out + MLP


def scaled_lr(batch_size, base_lr=OPENPI_LR, base_batch=OPENPI_BATCH):
    """sqrt scaling off openpi's reference. sqrt (not linear) is what the probe
    docs and LeRobot's multi-GPU guidance recommend when the schedule is unchanged."""
    return base_lr * math.sqrt(batch_size / base_batch)


def enable_compile(policy_mod, mode="default"):
    """Turn on torch.compile for a policies/<name>/model.py module.

    Wraps _lerobot_config so the flag lands on the config object before
    PI0Policy() reads it in __init__ — setting it afterwards is too late.
    mode='default' not 'max-autotune' (lerobot's default): max-autotune's Triton
    GEMM autotune is the one that had to be backed out on sm_100. Safe to try
    max-autotune on A100/sm_80 once a default-mode run is known good.
    """
    original = policy_mod._lerobot_config

    def with_compile(cfg):
        lr_cfg = original(cfg)
        lr_cfg.compile_model, lr_cfg.compile_mode = True, mode
        return lr_cfg

    policy_mod._lerobot_config = with_compile
    return policy_mod


def warmup_cosine(optimizer, total_steps, warmup_steps=1000, floor_ratio=0.1):
    """openpi / lerobot pi0 schedule: linear warmup then cosine decay.

    floor_ratio keeps a non-zero LR at the end (cosine to exactly 0 wastes the
    tail). Returns a LambdaLR — call .step() once per OPTIMIZER step, not epoch.
    """
    warmup_steps = max(1, min(warmup_steps, total_steps))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def init_from(model, ckpt_path, device="cpu", drop=STATS_BUFFERS):
    """Warm-start WEIGHTS from a checkpoint, keeping the model's own stats buffers.

    Why this exists: policies/pi0/model.py registers action_mean / action_std with
    register_buffer, and load_state_dict copies buffers. Warm-starting an absolute
    checkpoint into a delta run would restore the absolute stats (std ~13 deg) over
    the delta ones (std ~2.6 deg) — targets collapse to 0.12 of a unit and the 5x
    gain is exactly cancelled, silently, with the loss still going down.

    Optimizer state is deliberately NOT restored: Adam's moments are calibrated to
    the old target scale.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k: v for k, v in ckpt["model_state"].items() if k not in drop}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    kept = [k for k in drop if k in ckpt["model_state"]]
    missing = [k for k in missing if k not in drop]      # ours on purpose
    print(f"[init_from] {ckpt_path}  epoch={ckpt.get('epoch')} "
          f"action_space={ckpt.get('action_space')!r}")
    print(f"[init_from] kept THIS run's stats, dropped {kept} from the checkpoint")
    if missing or unexpected:
        print(f"[init_from] missing={len(missing)} unexpected={len(unexpected)}")
        for k in (missing + unexpected)[:5]:
            print(f"    {k}")
    return model


def use_file_system_sharing():
    """Route DataLoader worker tensors through files instead of /dev/shm.

    RunPod (and Docker generally) often caps /dev/shm at 64 MB. Eight workers at
    prefetch 4, batch 32, two 224x224 cameras is ~1.2 GB of in-flight tensors, so
    the default "file_descriptor" strategy exhausts shm and every worker dies at
    once — surfacing as

        RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly

    typically at an epoch boundary, where persistent_workers re-enters _reset.
    "file_system" is safe at any shm size. Process-global and idempotent.
    """
    import torch
    torch.multiprocessing.set_sharing_strategy("file_system")


def shm_size_mb():
    """Size of /dev/shm in MB, or None off Linux. Purely informational."""
    import shutil
    try:
        return shutil.disk_usage("/dev/shm").total / 1e6
    except (FileNotFoundError, OSError):
        return None


def loader_kwargs(num_workers=8, prefetch_factor=4):
    """DataLoader settings for a network-volume pod. common/train.py hardcodes
    num_workers=2 with no prefetch — the notebook measured 9.2 s/step starved vs
    ~4.7 s/step compute-bound, i.e. the GPU spends half its life waiting on JPEGs.

    Sets file_system sharing as a side effect when workers are used: it is a
    process-global that every caller here wants, and forgetting it costs a dead run
    hours in. NUM_WORKERS=0 is the bulletproof fallback if workers still die.
    """
    kw = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        use_file_system_sharing()
        kw.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    return kw


def probe_input_bound(loader, n=20):
    """Is the run data-bound or compute-bound? Decides whether a bigger batch will
    help at all. Returns seconds per batch spent purely fetching data."""
    import time
    it = iter(loader)
    next(it)                                    # let the workers spin up
    t0 = time.perf_counter()
    for _ in range(n):
        next(it)
    per_batch = (time.perf_counter() - t0) / n
    print(f"[probe] dataloader alone: {per_batch:.3f} s/batch over {n} batches")
    print("[probe] compare against your s/step: if it is a large fraction, raise "
          "num_workers before touching batch size")
    return per_batch


# ── finetune modes ────────────────────────────────────────────────────────────
# openpi's primary recipe is FULL fine-tune: README lists it as ">70 GB, A100
# (80GB)/H100" against LoRA's ">22.5 GB, RTX 4090", and the LoRA configs are named
# *_low_mem_finetune. On an A100-80 there is no reason to run the 4090 recipe.
#
# The trap in the LoRA path: policies/pi0/model.py:181 sets
#     train_expert_only = cfg.train_expert_only or cfg.vlm_lora_rank > 0
# and lerobot's train_expert_only (modeling_pi0.py:428) freezes ALL of paligemma,
# vision tower included. openpi's get_freeze_filter (pi0_config.py:88) freezes only
# PathRegex(".*llm.*"), and the model is nnx.Dict(llm=llm, img=img) (pi0.py:91) —
# so SigLIP is never frozen there. Freezing it is our deviation, not their recipe,
# and it is the worst thing to freeze on a camera rig the base model has never seen.

FINETUNE_MODES = ("full", "lora", "expert_only")


def finetune_flags(mode, lora_rank=16):
    """Model-config flags for a finetune mode. 'full' is openpi's primary recipe.

    'lora' deliberately leaves train_expert_only False so lerobot does not freeze
    the vision tower; call freeze_llm_keep_vision() after build to get openpi's
    actual shape (Gemma frozen except adapters, SigLIP trainable).
    """
    if mode not in FINETUNE_MODES:
        raise ValueError(f"mode must be one of {FINETUNE_MODES}, got {mode!r}")
    return {
        "full":        dict(vlm_lora_rank=0, freeze_vision_encoder=False,
                            train_expert_only=False),
        "lora":        dict(vlm_lora_rank=lora_rank, freeze_vision_encoder=False,
                            train_expert_only=False),
        "expert_only": dict(vlm_lora_rank=0, freeze_vision_encoder=False,
                            train_expert_only=True),
    }[mode]


def freeze_llm_keep_vision(model):
    """ENFORCE openpi's LoRA shape: Gemma LLM frozen; SigLIP, expert and adapters
    trainable — regardless of what lerobot froze before this call.

    It must un-freeze, not just freeze: with vlm_lora_rank > 0, model.py:181 sets
    train_expert_only, and lerobot's _set_requires_grad then freezes ALL of
    paligemma INCLUDING the vision tower. openpi never freezes the vision tower
    (freeze filter is PathRegex(\".*llm.*\"); the model is nnx.Dict(llm=..., img=...)),
    so the tower must be switched back on here.

    Matches on parameter NAME substrings rather than a hardcoded module path, so it
    survives lerobot renaming its internals; the asserts fail loudly if the naming
    ever stops matching instead of silently training the wrong subset.
    """
    frozen = vision = expert = adapters = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            adapters += p.numel()
        elif "vision_tower" in name:
            p.requires_grad = True          # lerobot froze it; openpi trains it
            vision += p.numel()
        elif "gemma_expert" in name:
            p.requires_grad = True
            expert += p.numel()
        elif "paligemma" in name:
            p.requires_grad = False         # the Gemma LLM (+ projector): frozen
            frozen += p.numel()

    assert frozen, "no paligemma LLM params matched — lerobot naming changed"
    assert vision, "no vision_tower params found — refusing to guess"
    print(f"[finetune] openpi LoRA shape: froze {frozen/1e6:.0f}M Gemma | trainable: "
          f"vision {vision/1e6:.0f}M + expert {expert/1e6:.0f}M + adapters "
          f"{adapters/1e6:.1f}M")
    return model

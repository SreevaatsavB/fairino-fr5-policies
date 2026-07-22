"""
policies/pi0_fast/model.py — π0-FAST autoregressive VLA for the FR5.

Wraps lerobot's PI0FastPolicy (lerobot==0.5.1).

⚠️  REQUIRES PALIGEMMA WEIGHTS (gated HuggingFace download ~5 GB):
    huggingface-cli login
    # accept licence: https://huggingface.co/google/paligemma-3b-pt-224
    Also downloads: lerobot/fast-action-tokenizer (public, small)

Architecture — key differences from pi0 / pi05
──────────────────────────────────────────────
• NO flow-matching — instead uses FAST (Frequency-space Action Sequence Tokens):
  actions are VQ-tokenised into discrete tokens and generated autoregressively
  like text, using PaliGemma's language modelling head.
• Inference is a single forward pass through the language model (no ODE steps),
  making it much faster at runtime (~3-5ms vs ~80ms for pi0 with 10 ODE steps).
• Uses KV cache for fast autoregressive decoding.
• forward() signature has no `reduction` parameter (returns scalar loss directly).

Normalisation
• Same as pi0: undo ImageNet norm → [0,1] → policy handles [-1,1] for SigLIP
• State: mean-std then padded to max_state_dim=32
• Actions: the policy tokenises them internally via lerobot/fast-action-tokenizer;
  we still mean-std normalise before passing so the tokeniser sees consistent range
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig as _LRConfig
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

# common/ is on sys.path when run via train.py / deploy.py; fall back to an
# explicit path so the import works no matter how model.py gets loaded.
try:
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio
    from vla_pretrained import (warn_or_load_pretrained, _inject_vlm_lora,
                                build_context, quantize_vlm,
                                pick_build_dtype)
except ImportError:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio
    from vla_pretrained import (warn_or_load_pretrained, _inject_vlm_lora,
                                build_context, quantize_vlm,
                                pick_build_dtype)

try:
    from transformers import AutoTokenizer
    _HAS_TOKENIZER = True
except ImportError:
    _HAS_TOKENIZER = False

STATE_KEY      = "observation.state"
IMAGE_KEY      = "observation.images.wrist_cam"
ACTION_KEY     = "action"
LANG_TOKENS    = "observation.language.tokens"
LANG_ATTN_MASK = "observation.language.attention_mask"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_TEXT_TOKENIZER = "google/paligemma-3b-pt-224"   # text side (gated)

def _image_keys(camera_names) -> list:
    """('wrist_cam','scene_cam') -> ['observation.images.wrist_cam', 'observation.images.scene_cam'].
    lerobot's PI0/PI05/PI0Fast encode each camera with the SHARED SigLIP and prepend its
    tokens — no extra params, so multi-camera is pretrained-weight-compatible."""
    return [f"observation.images.{c}" for c in camera_names]


@dataclass
class Pi0FastConfig:
    state_dim:  int  = 7
    action_dim: int  = 7
    chunk_size: int  = 50
    # inference-only: execute the first k of each chunk then re-plan (receding horizon).
    # None = execute the whole chunk (most open-loop). Overridable via deploy.py
    # --n-action-steps; NOT temporal ensembling, which is too slow for a 2.3B VLA at 30 Hz.
    n_action_steps: int = None
    use_image:  bool = True
    max_state_dim:  int = 32
    max_action_dim: int = 32
    tokenizer_max_length: int = 200
    camera_names: tuple = ("wrist_cam", "scene_cam")  # wrist + scene, shared SigLIP

    # pretrained π0-FAST weights (openpi port, gated on HF). CRITICAL: constructing
    # PI0FastPolicy(config) alone RANDOM-INITS the whole model — same trap as pi0/pi05.
    # Finetuning requires loading this; set "" only for architecture smoke tests.
    pretrained: str = "lerobot/pi0fast_base"

    # memory / VRAM. NOTE: pi0_fast is FULL-finetuned (no LoRA here) — unlike pi0/pi05
    # it has no separate action expert; FAST extends PaliGemma's token vocabulary, so
    # the action-token embeddings + LM head must train, which attention-only LoRA would
    # freeze. bf16 + gradient checkpointing keep the full finetune feasible.
    dtype:                  str  = "bfloat16"
    gradient_checkpointing: bool = True

    # k-bit quantization of the FROZEN VLM: "none" | "nf4" | "int8". Defaults OFF
    # because the full finetune above is pi0_fast's native recipe. Turning it on
    # necessarily converts the run to QLoRA — a 4-bit base takes no gradient — so
    # it REQUIRES vlm_lora_rank > 0. lm_head and the token embeddings are left
    # unquantized either way, so the FAST action tokens can still be learned.
    quantize:         str   = "none"
    vlm_lora_rank:    int   = 0        # >0 -> LoRA on the VLM q/k/v/o (needed for quantize)
    vlm_lora_alpha:   int   = 32
    vlm_lora_dropout: float = 0.05

    # proprioception handling (see common/proprio.py): full | dropout | none
    proprio_mode:         str   = "full"
    proprio_dropout_rate: float = 0.3


def _lerobot_config(cfg: Pi0FastConfig) -> _LRConfig:
    input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(cfg.state_dim,)),
    }
    norm_map = {
        "STATE":  NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    if cfg.use_image:
        for key in _image_keys(cfg.camera_names):
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        norm_map["VISUAL"] = NormalizationMode.IDENTITY

    return _LRConfig(
        n_obs_steps=1,
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps or cfg.chunk_size,
        input_features=input_features,
        output_features={ACTION_KEY: PolicyFeature(type=FeatureType.ACTION,
                                                    shape=(cfg.action_dim,))},
        normalization_mapping=norm_map,
        max_state_dim=cfg.max_state_dim,
        max_action_dim=cfg.max_action_dim,
        tokenizer_max_length=cfg.tokenizer_max_length,
        dtype=cfg.dtype,
        gradient_checkpointing=cfg.gradient_checkpointing,
        use_kv_cache=True,
    )


class Pi0Fast(nn.Module):
    def __init__(self, cfg: Pi0FastConfig, stats: dict, device=None):
        super().__init__()
        self.cfg    = cfg
        self._quantized = str(getattr(cfg, 'quantize', 'none')).lower() in ('nf4', 'int8')
        # Build order matters — see the module docstring of common/vla_pretrained.py.
        # build_context puts the params straight into VRAM instead of materialising
        # them in host RAM first (which OOM-kills a memory-capped container before
        # the GPU is ever touched).
        with build_context(device, pick_build_dtype(cfg.dtype, device, "pi0_fast")):
            self.policy = PI0FastPolicy(_lerobot_config(cfg))
        warn_or_load_pretrained(self.policy, cfg, "pi0_fast")
        quantize_vlm(self.policy, cfg.quantize, "pi0_fast", lora_rank=cfg.vlm_lora_rank)
        if cfg.vlm_lora_rank > 0:
            # pi0/pi05 freeze the base VLM via lerobot's train_expert_only, but
            # PI0FastConfig has no such flag — without freezing here the adapters
            # would train alongside a fully-unfrozen 3B backbone, which is not LoRA
            # in any meaningful sense (and blows up optimizer state). lm_head and
            # the token embeddings stay trainable on purpose: FAST extends the
            # vocabulary with action tokens that the model must still learn.
            for n, p in self.policy.model.paligemma_with_expert.paligemma.named_parameters():
                if "lm_head" not in n and "embed_tokens" not in n:
                    p.requires_grad_(False)
            _inject_vlm_lora(self.policy, cfg.vlm_lora_rank, cfg.vlm_lora_alpha,
                             cfg.vlm_lora_dropout, "pi0_fast")

        # proprioception mode (full | dropout | none) — applied in _make_batch.
        self.proprio = ProprioConfig(cfg.proprio_mode, cfg.proprio_dropout_rate)
        if self.proprio.active:
            print(f"[pi0_fast] {_describe_proprio(self.proprio)}")

        if _HAS_TOKENIZER:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(_TEXT_TOKENIZER)
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

        self.register_buffer("state_mean",  torch.as_tensor(stats["state_mean"]).float())
        self.register_buffer("state_std",   torch.as_tensor(stats["state_std"]).float())
        self.register_buffer("action_mean", torch.as_tensor(stats["action_mean"]).float())
        self.register_buffer("action_std",  torch.as_tensor(stats["action_std"]).float())
        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN.clone())
        self.register_buffer("_imagenet_std",  _IMAGENET_STD.clone())

    def _norm_state(self, s):    return (s - self.state_mean) / self.state_std
    def _norm_action(self, a):   return (a - self.action_mean) / self.action_std
    def _unnorm_action(self, a): return a * self.action_std + self.action_mean

    def _to_raw(self, img):
        return (img * self._imagenet_std + self._imagenet_mean).clamp(0, 1)

    def _tokenize(self, task, device):
        B = len(task) if isinstance(task, list) else 1
        if self.tokenizer is None:
            ids  = torch.zeros(B, self.cfg.tokenizer_max_length, dtype=torch.long, device=device)
            mask = torch.zeros(B, self.cfg.tokenizer_max_length, dtype=torch.long, device=device)
            return ids, mask
        if isinstance(task, str):
            task = [task]
        enc = self.tokenizer(task, return_tensors="pt", padding="max_length",
                             truncation=True, max_length=self.cfg.tokenizer_max_length)
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def _make_batch(self, obs_state, actions=None, action_is_pad=None,
                    obs_image=None, task=None, training=None):
        # State: always (B, state_dim) — pi0_fast adds seq dim internally.
        if training is None:
            training = self.training
        dev  = obs_state.device
        B    = obs_state.shape[0]
        task = task or [""] * B
        batch = {STATE_KEY: mask_state(self._norm_state(obs_state), self.proprio, training)}
        if self.cfg.use_image and obs_image is not None:
            if torch.is_tensor(obs_image):                 # 1 camera -> wrap as dict
                obs_image = {self.image_keys[0]: obs_image}
            for key in self.image_keys:                    # multi-cam: feed each camera
                if key not in obs_image:
                    raise ValueError(f"missing camera {key!r}; got {list(obs_image)}")
                batch[key] = self._to_raw(obs_image[key])
        ids, mask = self._tokenize(task, dev)
        batch[LANG_TOKENS]    = ids
        batch[LANG_ATTN_MASK] = mask
        if actions is not None:
            batch[ACTION_KEY]      = self._norm_action(actions)
            batch["action_is_pad"] = action_is_pad
        return batch

    def _amp(self):
        """Autocast the policy forward to bf16 when the VLM is k-bit quantized.
        QLoRA keeps the LoRA adapters in fp32, which would otherwise upcast
        activations and collide with the bf16 (unquantized) action expert —
        `mat1 float != mat2 BFloat16`. No-op when not quantized / off-CUDA."""
        from contextlib import nullcontext
        if self._quantized and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def forward(self, obs_state, actions, action_is_pad, obs_image=None, task=None):
        """pi0_fast.forward has no `reduction` param — returns (loss, loss_item, 0.0)."""
        with self._amp():
            loss, _ = self.policy.forward(
                self._make_batch(obs_state, actions, action_is_pad, obs_image, task)
            )
        return loss, loss.item(), 0.0

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs_state, obs_image=None, task=None):
        with self._amp():
            action_norm = self.policy.select_action(
                self._make_batch(obs_state, obs_image=obs_image, task=task, training=False)
            )
        return self._unnorm_action(action_norm)


def build_model(cfg: dict, stats: dict, device) -> Pi0Fast:
    m, d = cfg["model"], cfg["dataset"]
    model_cfg = Pi0FastConfig(
        state_dim=m["state_dim"], action_dim=m["action_dim"],
        chunk_size=d["chunk_size"], use_image=d["use_image"],
        n_action_steps=m.get("n_action_steps"),  # inference receding-horizon knob
        max_state_dim=m.get("max_state_dim", 32),
        max_action_dim=m.get("max_action_dim", 32),
        tokenizer_max_length=m.get("tokenizer_max_length", 200),
        camera_names=tuple(d.get("camera_names", ("wrist_cam", "scene_cam"))),  # dataset-level (matches ACT)
        pretrained=m.get("pretrained", "lerobot/pi0fast_base") or "",
        dtype=m.get("dtype", "bfloat16"),
        gradient_checkpointing=m.get("gradient_checkpointing", True),
        quantize=m.get("quantize", "none") or "none",
        vlm_lora_rank=m.get("vlm_lora_rank", 0),
        vlm_lora_alpha=m.get("vlm_lora_alpha", 32),
        vlm_lora_dropout=m.get("vlm_lora_dropout", 0.05),
        proprio_mode=m.get("proprio_mode", "full"),
        proprio_dropout_rate=m.get("proprio_dropout_rate", 0.3),
    )
    # device is threaded in (not just used for the trailing .to) so the policy
    # can be constructed directly on the GPU — see build_context.
    return Pi0Fast(model_cfg, stats, device=device).to(device)

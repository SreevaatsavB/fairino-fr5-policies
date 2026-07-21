"""
policies/pi05/model.py — π0.5 (pi05) flow-matching VLA for the FR5.

Wraps lerobot's PI05Policy (lerobot==0.5.1).

⚠️  REQUIRES PALIGEMMA WEIGHTS (gated HuggingFace download ~5 GB):
    huggingface-cli login
    # accept licence at https://huggingface.co/google/paligemma-3b-pt-224

Differences from pi0
────────────────────
• tokenizer_max_length = 200  (longer context window for richer language)
• Uses QUANTILE normalization convention internally (we still do mean-std
  in the wrapper and set IDENTITY so the policy doesn't double-normalise)
• Otherwise identical architecture and inference path

Everything else (PaliGemma backbone, Gemma 300M action expert, flow-matching
objective, image [0,1]→[-1,1] normalization) is the same as pi0.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config as _LRConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

# common/ is on sys.path when run via train.py / deploy.py; fall back to an
# explicit path so the import works no matter how model.py gets loaded.
try:
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio
    from vla_pretrained import (_inject_vlm_lora, warn_or_load_pretrained,
                                build_context, quantize_vlm,
                                pick_build_dtype)
except ImportError:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio
    from vla_pretrained import (_inject_vlm_lora, warn_or_load_pretrained,
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
_PALIGEMMA_TOKENIZER = "google/paligemma-3b-pt-224"

def _image_keys(camera_names) -> list:
    """('wrist_cam','scene_cam') -> ['observation.images.wrist_cam', 'observation.images.scene_cam'].
    lerobot's PI0/PI05/PI0Fast encode each camera with the SHARED SigLIP and prepend its
    tokens — no extra params, so multi-camera is pretrained-weight-compatible."""
    return [f"observation.images.{c}" for c in camera_names]


@dataclass
class Pi05Config:
    state_dim:  int  = 7
    action_dim: int  = 7
    chunk_size: int  = 50
    use_image:  bool = True
    num_inference_steps:   int = 10
    max_state_dim:         int = 32
    max_action_dim:        int = 32
    paligemma_variant:     str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    tokenizer_max_length:  int = 200
    camera_names: tuple = ("wrist_cam", "scene_cam")  # wrist + scene, shared SigLIP    # longer than pi0's 48

    # pretrained π0.5 weights (openpi port, ~6 GB from HF). CRITICAL: constructing
    # PI05Policy(config) alone RANDOM-INITS the whole model — lerobot builds
    # PaliGemma from config, it does not download weights. Finetuning requires
    # loading this checkpoint; set to "" only for architecture smoke tests.
    pretrained: str = "lerobot/pi05_base"

    # finetuning recipe — LoRA adapters on the VLM + FULL training of the action
    # expert & projections (the base 2B VLM stays frozen). rank 0 disables LoRA,
    # in which case freeze_vision_encoder / train_expert_only below apply instead.
    vlm_lora_rank:    int   = 16
    vlm_lora_alpha:   int   = 32
    vlm_lora_dropout: float = 0.05

    # memory / VRAM knobs (threaded to lerobot's PI05Config; see notebooks/):
    #   dtype                   "bfloat16" halves weights+activations vs "float32"
    #   gradient_checkpointing  recompute activations in backward — big VRAM save, ~20% slower
    #   freeze_vision_encoder   freeze SigLIP only
    #   train_expert_only       freeze the whole VLM, train action expert + projections
    #                           (the 24 GB-GPU option; also the least-overfitting one on 150 eps)
    dtype:                  str  = "bfloat16"
    gradient_checkpointing: bool = True
    freeze_vision_encoder:  bool = False
    train_expert_only:      bool = False

    # k-bit quantization of the FROZEN VLM (QLoRA): "none" | "nf4" | "int8".
    # "nf4" takes the 2B VLM from ~4.6 GB to ~1.4 GB, which is what makes a
    # larger batch fit on a 24-48 GB card. Requires vlm_lora_rank > 0 (a 4-bit
    # base cannot take a gradient, so the adapters carry the entire VLM update).
    # The action expert is never quantized — it is fully trained.
    quantize: str = "none"

    # proprioception handling (see common/proprio.py): full | dropout | none
    proprio_mode:         str   = "full"
    proprio_dropout_rate: float = 0.3


def _lerobot_config(cfg: Pi05Config) -> _LRConfig:
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
        n_action_steps=cfg.chunk_size,
        input_features=input_features,
        output_features={ACTION_KEY: PolicyFeature(type=FeatureType.ACTION,
                                                    shape=(cfg.action_dim,))},
        normalization_mapping=norm_map,
        paligemma_variant=cfg.paligemma_variant,
        action_expert_variant=cfg.action_expert_variant,
        max_state_dim=cfg.max_state_dim,
        max_action_dim=cfg.max_action_dim,
        num_inference_steps=cfg.num_inference_steps,
        tokenizer_max_length=cfg.tokenizer_max_length,
        dtype=cfg.dtype,
        gradient_checkpointing=cfg.gradient_checkpointing,
        freeze_vision_encoder=cfg.freeze_vision_encoder,
        # with LoRA the base VLM must be frozen — adapters carry the VLM update
        train_expert_only=cfg.train_expert_only or cfg.vlm_lora_rank > 0,
    )


class Pi05(nn.Module):
    def __init__(self, cfg: Pi05Config, stats: dict, device=None):
        super().__init__()
        self.cfg = cfg
        self.image_keys = _image_keys(cfg.camera_names)
        # Build order matters — see the module docstring of common/vla_pretrained.py.
        # build_context puts the 3.5B params straight into VRAM instead of
        # materialising them in host RAM first (which OOM-kills a memory-capped
        # container before the GPU is ever touched). Pass cfg.dtype as the second
        # arg to halve the build's peak VRAM if the fp32 build does not fit.
        with build_context(device, pick_build_dtype(cfg.dtype, device, "pi05")):
            self.policy = PI05Policy(_lerobot_config(cfg))
        warn_or_load_pretrained(self.policy, cfg, "pi05")
        quantize_vlm(self.policy, cfg.quantize, "pi05",
                     lora_rank=cfg.vlm_lora_rank, expert_only=cfg.train_expert_only)
        if cfg.vlm_lora_rank > 0:
            _inject_vlm_lora(self.policy, cfg.vlm_lora_rank, cfg.vlm_lora_alpha,
                             cfg.vlm_lora_dropout, "pi05")

        # proprioception mode (full | dropout | none) — applied in _make_batch.
        self.proprio = ProprioConfig(cfg.proprio_mode, cfg.proprio_dropout_rate)
        if self.proprio.active:
            print(f"[pi05] {_describe_proprio(self.proprio)}")

        if _HAS_TOKENIZER:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(_PALIGEMMA_TOKENIZER)
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
        # State: always (B, state_dim) — pi05 adds seq dim internally in embed_suffix.
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

    def forward(self, obs_state, actions, action_is_pad, obs_image=None, task=None):
        loss, _ = self.policy.forward(
            self._make_batch(obs_state, actions, action_is_pad, obs_image, task)
        )
        return loss, loss.item(), 0.0

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs_state, obs_image=None, task=None):
        action_norm = self.policy.select_action(
            self._make_batch(obs_state, obs_image=obs_image, task=task, training=False)
        )
        return self._unnorm_action(action_norm)


def build_model(cfg: dict, stats: dict, device) -> Pi05:
    m, d = cfg["model"], cfg["dataset"]
    model_cfg = Pi05Config(
        state_dim=m["state_dim"], action_dim=m["action_dim"],
        chunk_size=d["chunk_size"], use_image=d["use_image"],
        num_inference_steps=m.get("num_inference_steps", 10),
        max_state_dim=m.get("max_state_dim", 32),
        max_action_dim=m.get("max_action_dim", 32),
        paligemma_variant=m.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=m.get("action_expert_variant", "gemma_300m"),
        tokenizer_max_length=m.get("tokenizer_max_length", 200),
        camera_names=tuple(d.get("camera_names", ("wrist_cam", "scene_cam"))),  # dataset-level (matches ACT)
        pretrained=m.get("pretrained", "lerobot/pi05_base") or "",
        vlm_lora_rank=m.get("vlm_lora_rank", 16),
        vlm_lora_alpha=m.get("vlm_lora_alpha", 32),
        vlm_lora_dropout=m.get("vlm_lora_dropout", 0.05),
        dtype=m.get("dtype", "bfloat16"),
        gradient_checkpointing=m.get("gradient_checkpointing", True),
        freeze_vision_encoder=m.get("freeze_vision_encoder", False),
        train_expert_only=m.get("train_expert_only", False),
        quantize=m.get("quantize", "none") or "none",
        proprio_mode=m.get("proprio_mode", "full"),
        proprio_dropout_rate=m.get("proprio_dropout_rate", 0.3),
    )
    # device is threaded in (not just used for the trailing .to) so the policy can
    # be constructed directly on the GPU — see build_context.
    return Pi05(model_cfg, stats, device=device).to(device)

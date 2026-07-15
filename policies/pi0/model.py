"""
policies/pi0/model.py — π0 (pi0) flow-matching VLA for the FR5.

Wraps lerobot's PI0Policy (lerobot==0.5.1).

⚠️  REQUIRES PALIGEMMA WEIGHTS (gated HuggingFace download ~5 GB):
    huggingface-cli login
    # accept licence at https://huggingface.co/google/paligemma-3b-pt-224
    python common/train.py --policy pi0 ...

Architecture
────────────
• PaliGemma (2B) vision-language backbone — encodes image + task text
• Gemma 300M action expert — denoises the action chunk
• Flow-matching objective (same as dit_flow but conditioned on a full VLM)
• 10-step Euler ODE at inference

Normalization (done in wrapper; pi0 policy sees IDENTITY)
• State / action : mean-std normalised then padded to max_state_dim=32
• Images : undo ImageNet norm (from dataset.py) → [0,1];
           PI0Policy._preprocess_images then maps [0,1] → [-1,1] for SigLIP
• Language : tokenised via the PaliGemma tokenizer (AutoTokenizer)

Differences from dit_flow
• Backbone is a full 2B-param VLM, not just CLIP
• State/action are padded to max_state_dim/max_action_dim (architecture is fixed-width)
• tokenizer_max_length = 48 (shorter than pi05's 200)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from lerobot.policies.pi0.configuration_pi0 import PI0Config as _LRConfig
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

# common/ is on sys.path when run via train.py / deploy.py; fall back to an
# explicit path so the import works no matter how model.py gets loaded.
try:
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio
except ImportError:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio

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

# PaliGemma tokenizer name — requires HF auth + licence acceptance
_PALIGEMMA_TOKENIZER = "google/paligemma-3b-pt-224"


@dataclass
class Pi0Config:
    state_dim:  int  = 7
    action_dim: int  = 7
    chunk_size: int  = 50      # pi0 default; controls prediction horizon
    use_image:  bool = True

    # flow-matching inference
    num_inference_steps: int = 10

    # pi0 architecture constants (must match PaliGemma variant)
    max_state_dim:  int = 32
    max_action_dim: int = 32
    paligemma_variant:    str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    tokenizer_max_length:  int = 48

    # pretrained π0 weights (openpi port, ~6 GB from HF). CRITICAL: constructing
    # PI0Policy(config) alone RANDOM-INITS the whole 2.3B model — lerobot builds
    # PaliGemma from config, it does not download weights. Finetuning requires
    # loading this checkpoint; set to "" only for architecture smoke tests.
    pretrained: str = "lerobot/pi0_base"

    # memory / VRAM knobs (threaded to lerobot's PI0Config; see notebooks/):
    #   dtype                   "bfloat16" halves weights+activations vs "float32"
    #   gradient_checkpointing  recompute activations in backward — big VRAM save, ~20% slower
    #   freeze_vision_encoder   freeze SigLIP only
    #   train_expert_only       freeze the whole VLM, train action expert + projections
    #                           (the 24 GB-GPU option; also the least-overfitting one on 150 eps)
    dtype:                  str  = "bfloat16"
    gradient_checkpointing: bool = True
    freeze_vision_encoder:  bool = False
    train_expert_only:      bool = False

    # proprioception handling (see common/proprio.py): full | dropout | none
    proprio_mode:         str   = "full"
    proprio_dropout_rate: float = 0.3


def _lerobot_config(cfg: Pi0Config) -> _LRConfig:
    input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(cfg.state_dim,)),
    }
    norm_map = {
        "STATE":  NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    if cfg.use_image:
        input_features[IMAGE_KEY] = PolicyFeature(type=FeatureType.VISUAL,
                                                   shape=(3, 224, 224))
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
        train_expert_only=cfg.train_expert_only,
    )


class Pi0(nn.Module):
    def __init__(self, cfg: Pi0Config, stats: dict):
        super().__init__()
        self.cfg = cfg
        if cfg.pretrained:
            # loads the openpi-ported π0 weights, keeping OUR config (FR5 features).
            # strict=False: normalization/stat buffers differ (we use IDENTITY norms).
            self.policy = PI0Policy.from_pretrained(
                cfg.pretrained, config=_lerobot_config(cfg), strict=False)
            print(f"[pi0] loaded pretrained weights: {cfg.pretrained}")
        else:
            self.policy = PI0Policy(_lerobot_config(cfg))
            print("[pi0] WARNING: pretrained='' — RANDOM-INIT weights (smoke tests only; "
                  "finetuning a from-scratch 2.3B VLA on this dataset will not work)")

        # proprioception mode (full | dropout | none) — applied in _make_batch.
        # pi0 always has language conditioning, so 'none' is valid with or without image.
        self.proprio = ProprioConfig(cfg.proprio_mode, cfg.proprio_dropout_rate)
        if self.proprio.active:
            print(f"[pi0] {_describe_proprio(self.proprio)}")

        # load tokenizer (requires HF auth + PaliGemma licence)
        if _HAS_TOKENIZER:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(_PALIGEMMA_TOKENIZER)
            except Exception:
                self.tokenizer = None   # will use zero tokens — smoke test only
        else:
            self.tokenizer = None

        # mean-std normalisation buffers
        self.register_buffer("state_mean",  torch.as_tensor(stats["state_mean"]).float())
        self.register_buffer("state_std",   torch.as_tensor(stats["state_std"]).float())
        self.register_buffer("action_mean", torch.as_tensor(stats["action_mean"]).float())
        self.register_buffer("action_std",  torch.as_tensor(stats["action_std"]).float())

        # image renorm buffers
        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN.clone())
        self.register_buffer("_imagenet_std",  _IMAGENET_STD.clone())

    def _norm_state(self, s):    return (s - self.state_mean) / self.state_std
    def _norm_action(self, a):   return (a - self.action_mean) / self.action_std
    def _unnorm_action(self, a): return a * self.action_std + self.action_mean

    def _to_raw(self, img):
        """Undo ImageNet norm → [0,1]; PI0Policy will convert to [-1,1] internally."""
        return (img * self._imagenet_std + self._imagenet_mean).clamp(0, 1)

    def _tokenize(self, task, device):
        B = len(task) if isinstance(task, list) else 1
        if self.tokenizer is None:
            # fallback: zero tokens (smoke-test / no PaliGemma access)
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
        """Build batch for PI0Policy.

        State shape: always (B, state_dim) — NO n_obs_steps unsqueeze.
        PI0Pytorch.embed_suffix calls state_proj(state) expecting (B, max_state_dim)
        then adds the sequence dim itself via state_emb[:, None, :]. Adding unsqueeze
        here would give (B, 1, max_state_dim) → state_emb[:, None, :] → (B,1,1,width).

        Images: (B, C, H, W) in [0,1]; _preprocess_images converts to [-1,1].
        Actions: (B, chunk_size, action_dim); prepare_action pads to max_action_dim.
        """
        if training is None:
            training = self.training
        dev  = obs_state.device
        B    = obs_state.shape[0]
        task = task or [""] * B

        state = mask_state(self._norm_state(obs_state), self.proprio, training)
        batch = {STATE_KEY: state}                         # (B, state_dim), full/dropout/none

        if self.cfg.use_image and obs_image is not None:
            batch[IMAGE_KEY] = self._to_raw(obs_image)     # (B, C, H, W) in [0,1]

        ids, mask = self._tokenize(task, dev)
        batch[LANG_TOKENS]    = ids
        batch[LANG_ATTN_MASK] = mask

        if actions is not None:
            batch[ACTION_KEY]      = self._norm_action(actions)
            batch["action_is_pad"] = action_is_pad

        return batch

    def forward(self, obs_state, actions, action_is_pad, obs_image=None, task=None):
        """Returns (loss, loss_item, 0.0) — flow-matching MSE, no KL."""
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


def build_model(cfg: dict, stats: dict, device) -> Pi0:
    m, d = cfg["model"], cfg["dataset"]
    model_cfg = Pi0Config(
        state_dim=m["state_dim"],
        action_dim=m["action_dim"],
        chunk_size=d["chunk_size"],
        use_image=d["use_image"],
        num_inference_steps=m.get("num_inference_steps", 10),
        max_state_dim=m.get("max_state_dim", 32),
        max_action_dim=m.get("max_action_dim", 32),
        paligemma_variant=m.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=m.get("action_expert_variant", "gemma_300m"),
        tokenizer_max_length=m.get("tokenizer_max_length", 48),
        pretrained=m.get("pretrained", "lerobot/pi0_base") or "",
        dtype=m.get("dtype", "bfloat16"),
        gradient_checkpointing=m.get("gradient_checkpointing", True),
        freeze_vision_encoder=m.get("freeze_vision_encoder", False),
        train_expert_only=m.get("train_expert_only", False),
        proprio_mode=m.get("proprio_mode", "full"),
        proprio_dropout_rate=m.get("proprio_dropout_rate", 0.3),
    )
    return Pi0(model_cfg, stats).to(device)

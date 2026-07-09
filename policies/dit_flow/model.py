"""
policies/dit_flow/model.py — Diffusion Transformer + flow-matching policy for the FR5.

Wraps lerobot's MultiTaskDiTPolicy (lerobot==0.5.1) with objective="flow_matching".

Architecture overview
─────────────────────
Observation encoder
  • Vision encoder — wrist-cam frames. Either:
      - CLIP ViT-B/16 CLS token (lerobot stock; dino_backbone unset), or
      - a FROZEN self-supervised ViT (DINOv2 / DINOv3 / I-JEPA) via
        common/vision_backbone.py, mean-pooled + projected to 768 so it is a
        drop-in for the CLIP CLS vector (see DiTVisionAdapter; dino_backbone set)
  • CLIP text encoder             — language task string (independent of vision choice)
  • Linear projection             — joint state (6 DOF)
  → flat conditioning vector per timestep

Action denoiser — Diffusion Transformer (DiT)
  • Transformer with rotary PE (RoPE)
  • Conditioning injected via cross-attention / AdaLN
  • Predicts the velocity field  v(x_t, t, cond)

Training objective — conditional flow matching
  Path:   x_t  = t·a + (1 − (1−σ)·t)·ε,   ε ∼ 𝒩(0, I)
  Target: v    = a − (1−σ)·ε
  Loss:   ‖v_θ(x_t, t, cond) − v‖²

Inference — Euler ODE integration, t: 0 → 1
  x_{t+Δt} = x_t + Δt · v_θ(x_t, t, cond)
  (num_integration_steps steps, configurable)

Normalization (all done in this wrapper; lerobot policy sees IDENTITY norms)
  • State  / action : min-max  → [-1, 1]
  • Images : dataset.py emits ImageNet-normed 224x224. CLIP encoder → re-norm to
    CLIP stats; DINO/I-JEPA backbones expect ImageNet stats → passed through as-is.
  • Language       : CLIPTokenizerFast via AutoTokenizer
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from lerobot.policies.multi_task_dit.configuration_multi_task_dit import (
    MultiTaskDiTConfig as _LRConfig,
)
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

# common/ is on sys.path when run via train.py / deploy.py; fall back to an
# explicit path so the import works no matter how model.py gets loaded.
try:
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio
    from vision_backbone import VisionBackbone
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio
    from vision_backbone import VisionBackbone


# ── batch key constants ────────────────────────────────────────────────────────
STATE_KEY      = "observation.state"
IMAGE_KEY      = "observation.images.wrist_cam"
ACTION_KEY     = "action"
LANG_TOKENS    = "observation.language.tokens"
LANG_ATTN_MASK = "observation.language.attention_mask"

# ImageNet statistics applied by dataset.py — we undo these before CLIP
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# CLIP ViT normalization expected by HuggingFace's CLIPModel pixel_values
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


@dataclass
class DiTFlowConfig:
    state_dim:  int  = 7
    action_dim: int  = 7
    chunk_size: int  = 32         # horizon == n_action_steps
    use_image:  bool = True

    # flow-matching / diffusion
    objective:             str   = "flow_matching"   # "flow_matching" | "diffusion"
    num_integration_steps: int   = 10                # Euler/RK4 ODE steps at inference
    integration_method:    str   = "euler"           # "euler" | "rk4"

    # inference smoothing (no effect on training or weights — safe to change per deploy):
    #   temporal_ensemble_coeff  ACT-style temporal ensembling: re-predict the full chunk
    #                            every step and exponentially blend overlapping predictions
    #                            (w = exp(-coeff*age)). Costs one ODE sample per control
    #                            step (~5-10 ms CUDA — fine at 30 Hz; ~80 ms CPU — too slow).
    #                            None disables it.
    #   n_action_steps           only used when ensembling is off: execute the first k of
    #                            chunk_size actions, then re-plan (receding horizon).
    #                            None -> full chunk open-loop (the old, jerky behaviour).
    temporal_ensemble_coeff: float | None = 0.01
    n_action_steps:          int   | None = None

    # DiT transformer
    hidden_dim:     int   = 512
    num_layers:     int   = 6
    num_heads:      int   = 8
    dropout:        float = 0.1

    # CLIP vision + language encoders. vision_encoder_name must stay a CLIP id even
    # when dino_backbone is set (lerobot's config validates the name; the CLIP vision
    # tower it builds is then replaced by DiTVisionAdapter). Text always stays CLIP.
    vision_encoder_name:  str = "openai/clip-vit-base-patch16"
    text_encoder_name:    str = "openai/clip-vit-base-patch16"
    tokenizer_max_length: int = 77

    # vision backbone override: "" -> lerobot's stock CLIP CLS encoder; or a FROZEN
    # self-supervised ViT: "dinov2_vits14" (torch.hub), "dinov3_vits16" (HF, gated),
    # "ijepa_vith14" (HF, GPU-only), or any "owner/model" HF id. See common/vision_backbone.py.
    dino_backbone: str = ""
    # number of LAST ViT blocks to fine-tune (0 = fully frozen); >0 trains at 0.1x LR (train.py)
    dino_trainable_blocks: int = 0

    # proprioception handling (see common/proprio.py): full | dropout | none
    proprio_mode:         str   = "full"
    proprio_dropout_rate: float = 0.3


def _lerobot_config(cfg: DiTFlowConfig) -> _LRConfig:
    input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(cfg.state_dim,)),
    }
    norm_map = {
        "STATE":  NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    if cfg.use_image:
        input_features[IMAGE_KEY] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        norm_map["VISUAL"] = NormalizationMode.IDENTITY

    # With temporal ensembling the wrapper re-predicts every step and needs the FULL
    # horizon chunk (lerobot slices _generate_actions output to n_action_steps).
    # Without it, n_action_steps < horizon gives lerobot's receding-horizon queue.
    if cfg.temporal_ensemble_coeff is not None:
        n_action_steps = cfg.chunk_size
    else:
        n_action_steps = cfg.n_action_steps or cfg.chunk_size
    return _LRConfig(
        n_obs_steps=1,
        horizon=cfg.chunk_size,
        n_action_steps=n_action_steps,
        input_features=input_features,
        output_features={ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(cfg.action_dim,))},
        normalization_mapping=norm_map,
        objective=cfg.objective,
        num_integration_steps=cfg.num_integration_steps,
        integration_method=cfg.integration_method,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        vision_encoder_name=cfg.vision_encoder_name if cfg.use_image else None,
        text_encoder_name=cfg.text_encoder_name,
        tokenizer_max_length=cfg.tokenizer_max_length,
        image_resize_shape=None,   # dataset already outputs 224×224
        image_crop_shape=None,     # no additional crop needed
        image_crop_is_random=False,
        use_rope=True,
        do_mask_loss_for_padding=False,
    )


class DiTVisionAdapter(nn.Module):
    """Drop-in replacement for lerobot's CLIPVisionEncoder backed by a frozen
    self-supervised ViT (common/vision_backbone.py).

    lerobot's ObservationEncoder expects `forward(x) -> (B, 768, 1, 1)` (CLIP CLS
    vector) and sizes the DiT's conditioning_dim from `get_output_shape()` at
    construction. VisionBackbone emits a patch-token grid (B, D, h, w); we mean-pool
    it to a global vector and linearly project D -> 768 so conditioning_dim (and the
    already-built noise_predictor) are unchanged.

    Attribute naming matters: `self.backbone` holds the VisionBackbone whose inner
    net is `.dino`, so param names contain "backbone.dino" and train.py's 0.1x-LR
    fine-tune group applies when dino_trainable_blocks > 0.
    """

    CLIP_EMBED_DIM = 768   # CLIP ViT-B/16 CLS dim the DiT conditioning was sized for

    def __init__(self, name: str, trainable_blocks: int = 0):
        super().__init__()
        self.backbone = VisionBackbone(name, trainable_blocks=trainable_blocks)
        self.embed_dim = self.CLIP_EMBED_DIM
        self.proj = nn.Linear(self.backbone.embed_dim, self.CLIP_EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)["feature_map"]        # (B, D, h, w)
        pooled = fmap.flatten(2).mean(dim=2)          # (B, D) — mean over patches
        out = self.proj(pooled)                       # (B, 768)
        return out.reshape(out.shape[0], self.CLIP_EMBED_DIM, 1, 1)

    def get_output_shape(self) -> tuple:
        return (self.CLIP_EMBED_DIM, 1, 1)


class DiTFlow(nn.Module):
    def __init__(self, cfg: DiTFlowConfig, stats: dict):
        super().__init__()
        self.cfg = cfg
        self.policy = MultiTaskDiTPolicy(_lerobot_config(cfg))

        # optional: replace the stock CLIP vision tower with a FROZEN self-supervised
        # ViT (DINOv2 / DINOv3 / I-JEPA). Same post-construction surgery as ACT's
        # backbone swap; the adapter keeps the output at (B, 768, 1, 1) so the DiT's
        # conditioning_dim — fixed when the policy was built — still matches.
        self.uses_dino = bool(cfg.dino_backbone) and cfg.use_image
        if self.uses_dino:
            adapter = DiTVisionAdapter(cfg.dino_backbone,
                                       trainable_blocks=cfg.dino_trainable_blocks)
            enc = self.policy.observation_encoder
            enc.vision_encoder = adapter
            enc.vision_encoders = None   # single shared encoder path
            n_train = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
            print(f"[DiT-Flow] vision backbone -> FROZEN {cfg.dino_backbone} "
                  f"(embed_dim={adapter.backbone.embed_dim}, patch={adapter.backbone.patch}, "
                  f"projected to {adapter.CLIP_EMBED_DIM}); "
                  f"trainable params now {n_train/1e6:.1f}M")

        # proprioception mode (full | dropout | none) — applied in _make_batch
        self.proprio = ProprioConfig(cfg.proprio_mode, cfg.proprio_dropout_rate)
        if self.proprio.mode == "none" and not cfg.use_image:
            raise ValueError(
                "proprio_mode='none' (state-free) needs use_image=True — with no "
                "camera and no state the DiT has only language to condition on.")
        if self.proprio.active:
            print(f"[DiT-Flow] {_describe_proprio(self.proprio)}")

        # inference smoothing — temporal ensembling (ACT-style) or receding horizon.
        # Ensembling re-predicts the full chunk every step and blends overlapping
        # predictions; it operates in NORMALIZED action space (unnorm happens after).
        if cfg.temporal_ensemble_coeff is not None:
            self.ensembler = ACTTemporalEnsembler(cfg.temporal_ensemble_coeff, cfg.chunk_size)
            print(f"[DiT-Flow] inference: temporal ensembling "
                  f"(coeff={cfg.temporal_ensemble_coeff}, chunk={cfg.chunk_size}) — "
                  f"one ODE sample per control step")
        else:
            self.ensembler = None
            k = cfg.n_action_steps or cfg.chunk_size
            mode = ("open-loop full chunk" if k == cfg.chunk_size
                    else f"receding horizon (execute {k}/{cfg.chunk_size}, then re-plan)")
            print(f"[DiT-Flow] inference: {mode}")

        # language tokenizer (same CLIP checkpoint as the text encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_name)

        # min-max normalization buffers — saved into the checkpoint
        eps = 1e-6
        self.register_buffer("state_min",   torch.as_tensor(stats["state_min"]).float())
        self.register_buffer("state_max",   torch.as_tensor(stats["state_max"]).float() + eps)
        self.register_buffer("action_min",  torch.as_tensor(stats["action_min"]).float())
        self.register_buffer("action_max",  torch.as_tensor(stats["action_max"]).float() + eps)

        # image renormalization constants (stored as buffers for device movement)
        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN.clone())
        self.register_buffer("_imagenet_std",  _IMAGENET_STD.clone())
        self.register_buffer("_clip_mean",     _CLIP_MEAN.clone())
        self.register_buffer("_clip_std",      _CLIP_STD.clone())

    # ── normalization helpers ──────────────────────────────────────────────────

    def _norm_state(self, s):
        return 2.0 * (s - self.state_min) / (self.state_max - self.state_min) - 1.0

    def _norm_action(self, a):
        return 2.0 * (a - self.action_min) / (self.action_max - self.action_min) - 1.0

    def _unnorm_action(self, a):
        return (a + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min

    def _renorm_image(self, img):
        """Undo ImageNet normalization (applied by dataset.py) → apply CLIP normalization."""
        raw = img * self._imagenet_std + self._imagenet_mean   # → [0, 1]
        return (raw - self._clip_mean) / self._clip_std

    def _tokenize(self, task, device):
        """Tokenize a list of language strings → input_ids / attention_mask tensors."""
        if isinstance(task, str):
            task = [task]
        enc = self.tokenizer(
            task,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.tokenizer_max_length,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # ── batch builder ──────────────────────────────────────────────────────────

    def _make_batch(self, obs_state, actions=None, action_is_pad=None,
                    obs_image=None, task=None, for_training=True, training=None):
        """Build a batch dict for lerobot's MultiTaskDiTPolicy.

        Two distinct modes:

        for_training=True  (→ policy.forward)
          State: (B, 1, state_dim) — encode() reads batch[OBS_STATE].shape[:2]
          Image: (B, C, H, W)     — _prepare_batch adds camera dim, encode adds n_obs_steps

        for_training=False  (→ policy.select_action via queue system)
          State: (B, state_dim)   — queue stacks to (B, 1, state_dim) via dim=1
          Image: (B, C, H, W)     — _prepare_batch adds camera dim → (B, 1, C, H, W),
                                     queue stacks to (B, 1, 1, C, H, W) — 6D for encode

        Language tokens are always (B, max_length); encode handles the n_obs_steps
        expansion internally. They are NOT queued (not in policy._queues).
        """
        if training is None:
            training = self.training
        dev = obs_state.device
        B   = obs_state.shape[0]

        state = self._norm_state(obs_state)
        state = mask_state(state, self.proprio, training)   # full/dropout/none
        if for_training:
            state = state.unsqueeze(1)    # → (B, 1, state_dim) for encode()
        batch = {STATE_KEY: state}

        if self.cfg.use_image and obs_image is not None:
            # Always (B, C, H, W); _prepare_batch and/or the queue adds n_obs_steps.
            # dataset.py emits ImageNet-normed images: the DINO/I-JEPA backbones expect
            # exactly that (pass through), the stock CLIP tower expects CLIP stats.
            batch[IMAGE_KEY] = obs_image if self.uses_dino else self._renorm_image(obs_image)

        # Language tokens — always include so conditioning_dim stays constant.
        # Default to empty strings; CLIP text encoder handles them gracefully.
        if task is None:
            task = [""] * B
        ids, mask = self._tokenize(task, dev)
        batch[LANG_TOKENS]    = ids
        batch[LANG_ATTN_MASK] = mask

        if actions is not None:
            batch[ACTION_KEY]      = self._norm_action(actions)   # (B, chunk, action_dim)
            batch["action_is_pad"] = action_is_pad

        return batch

    # ── public interface (matches common/train.py + common/deploy.py) ──────────

    def forward(self, obs_state, actions, action_is_pad, obs_image=None, task=None):
        """Returns (loss, loss_item, 0.0) — flow-matching MSE, no KL term."""
        batch = self._make_batch(obs_state, actions, action_is_pad, obs_image, task,
                                 for_training=True)
        loss, _ = self.policy.forward(batch)
        return loss, loss.item(), 0.0

    def reset(self):
        """Call once before each episode during deployment."""
        self.policy.reset()
        if self.ensembler is not None:
            self.ensembler.reset()

    @torch.no_grad()
    def predict(self, obs_state, obs_image=None, task=None):
        """Next action → (action_dim,) tensor in original units.

        With temporal ensembling (default): every call samples a FULL chunk_size chunk
        via the ODE and blends it with previous overlapping predictions
        (w = exp(-coeff*age)) — smooth, reactive, one ODE sample per control step.

        Without it: select_action's queue pops pre-computed actions and re-plans every
        n_action_steps (receding horizon) or chunk_size (open-loop) steps.

        Call reset() at the start of each episode.
        """
        if self.ensembler is not None:
            # Bypass the action queue: training-style batch shapes (state (B,1,D),
            # image (B,C,H,W)) → _prepare_batch stacks cameras → _generate_actions
            # returns the full normalized chunk (B, chunk_size, action_dim).
            self.policy.eval()
            batch = self._make_batch(obs_state, obs_image=obs_image, task=task,
                                     for_training=True, training=False)
            batch = self.policy._prepare_batch(batch)
            chunk = self.policy._generate_actions(batch)
            action_norm = self.ensembler.update(chunk)
        else:
            action_norm = self.policy.select_action(
                self._make_batch(obs_state, obs_image=obs_image, task=task,
                                 for_training=False, training=False)
            )
        return self._unnorm_action(action_norm)


# ── policy entry point for common/train.py and common/deploy.py ───────────────

def build_model(cfg: dict, stats: dict, device) -> DiTFlow:
    m, d = cfg["model"], cfg["dataset"]
    model_cfg = DiTFlowConfig(
        state_dim=m["state_dim"],
        action_dim=m["action_dim"],
        chunk_size=d["chunk_size"],
        use_image=d["use_image"],
        objective=m.get("objective", "flow_matching"),
        num_integration_steps=m.get("num_integration_steps", 10),
        integration_method=m.get("integration_method", "euler"),
        temporal_ensemble_coeff=m.get("temporal_ensemble_coeff", 0.01),
        n_action_steps=m.get("n_action_steps"),
        hidden_dim=m.get("hidden_dim", 512),
        num_layers=m.get("num_layers", 6),
        num_heads=m.get("num_heads", 8),
        dropout=m.get("dropout", 0.1),
        vision_encoder_name=m.get("vision_encoder_name", "openai/clip-vit-base-patch16"),
        text_encoder_name=m.get("text_encoder_name", "openai/clip-vit-base-patch16"),
        tokenizer_max_length=m.get("tokenizer_max_length", 77),
        dino_backbone=m.get("dino_backbone", "") or "",
        dino_trainable_blocks=m.get("dino_trainable_blocks", 0),
        proprio_mode=m.get("proprio_mode", "full"),
        proprio_dropout_rate=m.get("proprio_dropout_rate", 0.3),
    )
    return DiTFlow(model_cfg, stats).to(device)

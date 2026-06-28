"""
Wraps lerobot's ACTPolicy so train.py / deploy.py stay simple. Tested vs lerobot==0.5.1.

This branch (fix/dino-act-cvae-probe) adds, on top of the base ACT wrapper:
  * a FROZEN DINOv2 ViT backbone option (drop-in for lerobot's ResNet18) + multi-camera,
  * a CVAE KL-collapse fix — KL annealing + free-bits + a lower kl_weight — because the
    default `kl_weight=10` baked into lerobot's loss drove the latent to ~0 (posterior
    collapse), turning ACT into a plain mean-regressor (see docs/failure_analysis/).
  * `predict_chunk()` for the offline vision-vs-state probe (experiments/shortcut_probe.py).

Notes on the 0.5.x API: config takes input_features/output_features + normalization_mapping;
we normalise state/action ourselves and tell lerobot every feature is IDENTITY; images arrive
ImageNet-normalised from dataset.py. lerobot's ACTPolicy.forward computes l1 + kl*kl_weight
internally — for the KL fix we replicate its forward (the OBS_IMAGES stack + self.model call)
so we can re-weight / free-bit the KL ourselves.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.act.configuration_act import ACTConfig as _LRConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
from lerobot.utils.constants import OBS_IMAGES, ACTION

# common/ is on sys.path when run via train.py / deploy.py; fall back to an explicit path.
try:
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from proprio import ProprioConfig, mask_state, describe as _describe_proprio


STATE_KEY = "observation.state"
ENV_KEY = "observation.environment_state"
IMAGE_KEY = "observation.images.wrist_cam"   # default single-cam key (back-compat)
ACTION_KEY = "action"


def _image_keys(camera_names) -> list:
    """['wrist_cam','scene_cam'] -> ['observation.images.wrist_cam', 'observation.images.scene_cam']."""
    return [f"observation.images.{c}" for c in camera_names]


@dataclass
class ACTConfig:
    state_dim:          int   = 7
    action_dim:         int   = 7
    latent_dim:         int   = 32
    d_model:            int   = 512
    nhead:              int   = 8
    num_encoder_layers: int   = 4
    num_decoder_layers: int   = 7
    dim_feedforward:    int   = 3200
    dropout:            float = 0.1
    chunk_size:              int          = 100
    use_image:               bool         = True
    kl_weight:               float        = 0.5    # lowered from 10 (was a collapse trigger)
    temporal_ensemble_coeff: float | None = 0.01

    # CVAE KL-collapse fix (see docs/failure_analysis/06_cvae_kl_deep_dive.md)
    kl_warmup_steps: int   = 2000   # linearly ramp the KL weight 0 -> kl_weight over these steps
    kl_free_bits:    float = 0.03   # per-dim KL floor (lambda); gentle (~1 nat / 32 dims) so the latent
                                    # stays nonzero WITHOUT being forced strong (z=0 at inference)

    # cameras fed as ACT image inputs (lerobot shares one backbone across cameras)
    camera_names: tuple = ("wrist_cam",)

    # proprioception handling (see common/proprio.py): full | dropout | none
    proprio_mode:         str   = "full"
    proprio_dropout_rate: float = 0.3

    # vision backbone: "" -> lerobot's trainable ResNet18; or a FROZEN self-supervised ViT:
    #   "dinov2_vits14" (torch.hub), "dinov3_vits16" (HF, gated, the upgrade), "ijepa_vith14"
    #   (HF, GPU-only), or any "owner/model" HF id. See VisionBackbone.
    dino_backbone: str = ""
    # number of LAST ViT blocks to fine-tune (0 = fully frozen); >0 trains at 0.1x LR (train.py)
    dino_trainable_blocks: int = 0


def _lerobot_config(cfg: ACTConfig) -> _LRConfig:
    input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(cfg.state_dim,)),
    }
    norm_map = {
        FeatureType.STATE:  NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    if cfg.use_image:
        for key in _image_keys(cfg.camera_names):
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        norm_map[FeatureType.VISUAL] = NormalizationMode.IDENTITY
    else:
        input_features[ENV_KEY] = PolicyFeature(type=FeatureType.ENV, shape=(cfg.state_dim,))
        norm_map[FeatureType.ENV] = NormalizationMode.IDENTITY

    output_features = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(cfg.action_dim,))}
    n_action_steps = 1 if cfg.temporal_ensemble_coeff is not None else cfg.chunk_size

    return _LRConfig(
        chunk_size=cfg.chunk_size,
        n_action_steps=n_action_steps,
        n_obs_steps=1,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping=norm_map,
        dim_model=cfg.d_model,
        n_heads=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        n_encoder_layers=cfg.num_encoder_layers,
        n_decoder_layers=cfg.num_decoder_layers,
        n_vae_encoder_layers=cfg.num_encoder_layers,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        kl_weight=cfg.kl_weight,   # informational only — we re-compute the KL term ourselves
        use_vae=True,
        temporal_ensemble_coeff=cfg.temporal_ensemble_coeff,
        vision_backbone="resnet18",
        pretrained_backbone_weights=(
            "ResNet18_Weights.IMAGENET1K_V1" if cfg.use_image else None
        ),
    )


# Short backbone names -> HuggingFace ids (or pass a full "facebook/..." id directly).
# DINOv3 weights are GATED: `huggingface-cli login` with a token that has accepted the
# license at huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m. I-JEPA is open but
# its smallest public model is ViT-H (~630M) — GPU only.
_HF_BACKBONES = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",   # 21M — upgrade for dinov2_vits14
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",   # 86M
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",   # 300M
    "ijepa_vith14":  "facebook/ijepa_vith14_1k",                   # 630M — GPU only
    "ijepa_vith16":  "facebook/ijepa_vith16_1k",
}


class VisionBackbone(nn.Module):
    """Frozen self-supervised ViT (DINOv2 / DINOv3 / I-JEPA) as a drop-in for lerobot
    ACT's ResNet backbone.

    lerobot calls `backbone(img)["feature_map"]` and expects (B, C, h, w). Every ViT here
    emits patch tokens (B, N, D); we reshape them to (B, D, H/patch, W/patch). The ViT is
    frozen (no grad, always eval) — only the 1x1 projection + transformer + heads train —
    unless `trainable_blocks > 0`, which unfreezes the last N transformer blocks at 0.1x LR
    (handled in train.py via the `backbone.dino` name match; the inner net is kept as
    `self.dino` for that reason regardless of which model it is).

    Backbone selection by `name`:
      * "dinov2_*"  -> torch.hub facebookresearch/dinov2 (patch 14), the original.
      * "dinov3_*"  -> HuggingFace AutoModel (patch 16, +CLS +4 register tokens). GATED.
      * "ijepa_*"   -> HuggingFace AutoModel (patch 14/16). Big (ViT-H).
      * any "owner/model" string -> loaded verbatim via HuggingFace AutoModel.
    """

    def __init__(self, name: str = "dinov2_vits14", trainable_blocks: int = 0):
        super().__init__()
        self.trainable_blocks = int(trainable_blocks)
        self.kind = ("dinov2" if name.startswith("dinov2")
                     else "ijepa" if name.startswith("ijepa")
                     else "dinov3" if name.startswith("dinov3")
                     else "hf")
        if self.kind == "dinov2":
            self.dino = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
            self.embed_dim = int(self.dino.embed_dim)
            self.patch = int(getattr(self.dino, "patch_size", 14))
        else:                                              # dinov3 / ijepa / explicit HF id
            from transformers import AutoModel
            hf_id = _HF_BACKBONES.get(name, name)
            self.dino = AutoModel.from_pretrained(hf_id)
            self.embed_dim = int(self.dino.config.hidden_size)
            self.patch = int(self.dino.config.patch_size)

        for p in self.dino.parameters():
            p.requires_grad_(False)
        if self.trainable_blocks > 0:
            blocks = self._transformer_blocks()
            if blocks is not None:
                for blk in blocks[-self.trainable_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
            else:
                print(f"[VisionBackbone] WARN: couldn't find transformer blocks to unfreeze "
                      f"for {name!r}; keeping it fully frozen")
        self.dino.eval()

    def _transformer_blocks(self):
        """Best-effort handle on the ViT's transformer block list across model families."""
        for path in ("blocks", "layer", "layers", "encoder.layer", "encoder.layers"):
            obj = self.dino
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                return obj
        return None

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()   # keep the frozen backbone in eval even when the parent trains
        return self

    def _patch_tokens(self, img: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> patch tokens (B, N, D), dropping any leading CLS/register tokens."""
        if self.kind == "dinov2":
            return self.dino.forward_features(img)["x_norm_patchtokens"]
        hs = self.dino(pixel_values=img).last_hidden_state          # (B, T, D)
        n_patch = (img.shape[-2] // self.patch) * (img.shape[-1] // self.patch)
        return hs[:, hs.shape[1] - n_patch:, :]                     # patches are the trailing tokens

    def forward(self, img: torch.Tensor) -> dict:
        if self.trainable_blocks > 0:
            tok = self._patch_tokens(img)
        else:
            with torch.no_grad():
                tok = self._patch_tokens(img)
        B, N, D = tok.shape
        h = img.shape[-2] // self.patch
        w = img.shape[-1] // self.patch
        fmap = tok.transpose(1, 2).reshape(B, D, h, w).contiguous()
        return {"feature_map": fmap}


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig, stats: dict):
        super().__init__()
        self.cfg = cfg
        self.image_keys = _image_keys(cfg.camera_names)
        self.policy = ACTPolicy(_lerobot_config(cfg))

        # optional: replace lerobot's trainable ResNet with a FROZEN self-supervised ViT
        # (DINOv2 / DINOv3 / I-JEPA — see VisionBackbone).
        if getattr(cfg, "dino_backbone", "") and cfg.use_image:
            dino = VisionBackbone(cfg.dino_backbone,
                                  trainable_blocks=getattr(cfg, "dino_trainable_blocks", 0))
            self.policy.model.backbone = dino
            self.policy.model.encoder_img_feat_input_proj = nn.Conv2d(
                dino.embed_dim, cfg.d_model, kernel_size=1)
            n_train = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
            print(f"[ACT] vision backbone -> FROZEN {cfg.dino_backbone} "
                  f"(embed_dim={dino.embed_dim}, patch={dino.patch}); "
                  f"trainable params now {n_train/1e6:.1f}M")

        # proprioception mode (full | dropout | none) — applied in _make_batch
        self.proprio = ProprioConfig(cfg.proprio_mode, cfg.proprio_dropout_rate)
        if self.proprio.mode == "none" and not cfg.use_image:
            raise ValueError("proprio_mode='none' needs use_image=True — no camera and no "
                             "state leaves nothing to condition on.")
        if self.proprio.active:
            print(f"[ACT] {_describe_proprio(self.proprio)}")

        # CVAE KL fix: re-weighting schedule (annealing) + free-bits floor. _kl_step is a
        # saved buffer so resumed runs keep the schedule; it advances only on train forwards.
        self.kl_weight = float(cfg.kl_weight)
        self.kl_warmup_steps = max(1, int(cfg.kl_warmup_steps))
        self.kl_free_bits = float(cfg.kl_free_bits)
        self.register_buffer("_kl_step", torch.zeros((), dtype=torch.long))
        print(f"[ACT] CVAE: kl_weight={self.kl_weight} warmup={self.kl_warmup_steps} "
              f"free_bits={self.kl_free_bits}")

        # mean/std normalisation buffers (saved in state_dict, restored on load)
        self.register_buffer("state_mean",  torch.as_tensor(stats["state_mean"]).float())
        self.register_buffer("state_std",   torch.as_tensor(stats["state_std"]).float())
        self.register_buffer("action_mean", torch.as_tensor(stats["action_mean"]).float())
        self.register_buffer("action_std",  torch.as_tensor(stats["action_std"]).float())

    # ── normalisation helpers ──────────────────────────────────────────────
    def _norm_state(self, s):    return (s - self.state_mean) / self.state_std
    def _norm_action(self, a):   return (a - self.action_mean) / self.action_std
    def _unnorm_action(self, a): return a * self.action_std + self.action_mean

    def _make_batch(self, obs_state, actions=None, action_is_pad=None, obs_images=None,
                    training=None):
        # `training` gates proprio dropout (on only for the training forward); predict()
        # forces False so dropout never fires at inference.
        if training is None:
            training = self.training
        state = self._norm_state(obs_state)
        state = mask_state(state, self.proprio, training)   # full/dropout/none
        batch = {STATE_KEY: state}
        if self.cfg.use_image:
            if obs_images is None:
                raise ValueError("use_image=True but no images given")
            # accept a single (B,C,H,W) tensor (1 camera) or a {key: tensor} dict (multi-cam)
            if torch.is_tensor(obs_images):
                obs_images = {self.image_keys[0]: obs_images}
            for key in self.image_keys:
                if key not in obs_images:
                    raise ValueError(f"missing camera input {key!r}; got {list(obs_images)}")
                batch[key] = obs_images[key]   # already ImageNet-normed by dataset.py
        else:
            batch[ENV_KEY] = state
        if actions is not None:
            batch[ACTION_KEY] = self._norm_action(actions)
            batch["action_is_pad"] = action_is_pad
        return batch

    def forward(self, obs_state, actions, action_is_pad, obs_images=None, task=None):
        """Returns (loss, l1_item, kl_item). Replicates lerobot ACTPolicy.forward (the
        OBS_IMAGES stack + self.model call) but re-computes the KL term with annealing +
        free-bits and a lower weight, fixing the posterior collapse. The VAE encoder only
        produces (mu, logvar) in train mode, so we force train mode around the model call
        (train.py wraps validation in no_grad, so no gradients leak)."""
        batch = self._make_batch(obs_state, actions, action_is_pad, obs_images)
        if self.cfg.use_image:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[k] for k in self.policy.config.image_features]

        was_training = self.policy.training
        self.policy.train()
        try:
            actions_hat, (mu, log_sigma_x2) = self.policy.model(batch)
        finally:
            self.policy.train(was_training)

        # L1 reconstruction (masking padded steps), same as lerobot
        l1 = (F.l1_loss(batch[ACTION], actions_hat, reduction="none")
              * ~batch["action_is_pad"].unsqueeze(-1)).mean()

        # per-latent-dim KL, then FREE-BITS floor before summing, then mean over batch
        kld_dim = -0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())   # (B, latent)
        kld = kld_dim.clamp(min=self.kl_free_bits).sum(-1).mean()

        # ANNEAL the weight 0 -> kl_weight over kl_warmup_steps
        kl_w = self.kl_weight * min(1.0, float(self._kl_step.item()) / self.kl_warmup_steps)
        loss = l1 + kl_w * kld

        if self.training and torch.is_grad_enabled():
            self._kl_step += 1
        return loss, l1.item(), kld.item()

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def predict(self, obs_state, obs_images=None, task=None):
        action_norm = self.policy.select_action(
            self._make_batch(obs_state, obs_images=obs_images, training=False)
        )
        return self._unnorm_action(action_norm)

    @torch.no_grad()
    def predict_chunk(self, obs_state, obs_images=None):
        """Full unnormalised action chunk (B, chunk, action_dim), bypassing temporal
        ensembling — used by experiments/shortcut_probe.py for offline ablation."""
        was_training = self.policy.training
        self.policy.eval()
        try:
            chunk_norm = self.policy.predict_action_chunk(
                self._make_batch(obs_state, obs_images=obs_images, training=False))
        finally:
            self.policy.train(was_training)
        return self._unnorm_action(chunk_norm)


def build_model(cfg: dict, stats: dict, device) -> "ACT":
    """Policy entry point used by common/train.py and common/deploy.py."""
    m, d, t = cfg["model"], cfg["dataset"], cfg["training"]
    model_cfg = ACTConfig(
        state_dim=m["state_dim"],
        action_dim=m["action_dim"],
        latent_dim=m["latent_dim"],
        d_model=m["d_model"],
        nhead=m["nhead"],
        num_encoder_layers=m["num_encoder_layers"],
        num_decoder_layers=m["num_decoder_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
        chunk_size=d["chunk_size"],
        use_image=d["use_image"],
        kl_weight=t["kl_weight"],
        temporal_ensemble_coeff=m.get("temporal_ensemble_coeff"),
        kl_warmup_steps=t.get("kl_warmup_steps", 2000),
        kl_free_bits=t.get("kl_free_bits", 0.03),
        camera_names=tuple(d.get("camera_names", ["wrist_cam"])),
        proprio_mode=m.get("proprio_mode", "full"),
        proprio_dropout_rate=m.get("proprio_dropout_rate", 0.3),
        dino_backbone=m.get("dino_backbone", ""),
        dino_trainable_blocks=m.get("dino_trainable_blocks", 0),
    )
    return ACT(model_cfg, stats).to(device)

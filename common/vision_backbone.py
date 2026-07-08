"""
common/vision_backbone.py — frozen self-supervised ViT encoders shared across policies.

Moved out of policies/act/model.py so any policy (ACT, dit_flow, ...) can use the same
DINOv2 / DINOv3 / I-JEPA backbones. Import as:

    from vision_backbone import VisionBackbone          # common/ on sys.path

The `backbone.dino` parameter-name convention matters: common/train.py matches that
substring to put fine-tuned ViT blocks in a 0.1x-LR param group, so keep the inner
net named `.dino` (and, in wrappers, the VisionBackbone attribute named `.backbone`).
"""

import torch
import torch.nn as nn

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

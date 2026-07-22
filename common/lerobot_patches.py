"""
common/lerobot_patches.py — compatibility shims for lerobot 0.5.1 against the
transformers / torch versions it is actually run with here.

lerobot 0.5.1 pins loosely and was cut against an older transformers/torch than
the current stack, so two of its π-family code paths call APIs that have since
drifted. Neither shows up in training — both are on the INFERENCE path — so they
surface only at eval/deploy, after hours of training. Applied once, idempotently,
by apply_lerobot_compat_patches(); call it before building or running any
π-family policy (vla_pretrained does this on import).

1. create_causal_mask(cache_position=...) — pi_gemma.py passes cache_position,
   which transformers >= 5.x dropped from the signature (no **kwargs to absorb
   it). Hits pi0 AND pi05 (shared pi_gemma) during select_action.

2. _prepare_attention_masks_4d — the π-family builds a Long attention mask and
   hands it to torch.where, which torch >= 2.8 rejects (needs bool).

Both shims are version-proof: they inspect the installed API and no-op when it is
already compatible, so they stay correct if the stack is upgraded.
"""

import functools
import inspect

_APPLIED = False


def _patch_create_causal_mask():
    """Drop kwargs the installed create_causal_mask no longer accepts (e.g.
    cache_position). Returns a short status string for logging."""
    try:
        import lerobot.policies.pi_gemma as pg
    except Exception as e:  # pragma: no cover - lerobot import itself failing
        return f"create_causal_mask: skipped ({type(e).__name__})"

    fn = getattr(pg, "create_causal_mask", None)
    if fn is None:
        return "create_causal_mask: not present (nothing to patch)"
    if getattr(fn, "_kwarg_filtered", False):
        return "create_causal_mask: already patched"

    sig = inspect.signature(fn)
    names = {p.name for p in sig.parameters.values()}
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw or "cache_position" in names:
        return "create_causal_mask: signature already compatible"

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        return fn(*args, **{k: v for k, v in kwargs.items() if k in names})

    wrapped._kwarg_filtered = True
    pg.create_causal_mask = wrapped
    return f"create_causal_mask: patched (filters to {sorted(names)})"


def _patch_attention_mask_bool():
    """Cast the π-family's Long attention mask to bool before torch.where.

    torch >= 2.8 rejects a non-bool mask in torch.where; earlier torch accepted it.
    Patch each policy's *Pytorch._prepare_attention_masks_4d that exists."""
    out = []
    targets = [
        ("lerobot.policies.pi0.modeling_pi0",           "PI0Pytorch"),
        ("lerobot.policies.pi05.modeling_pi05",         "PI05Pytorch"),
        ("lerobot.policies.pi0_fast.modeling_pi0_fast", "PI0FastPytorch"),
    ]
    for mod_name, cls_name in targets:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            meth = cls._prepare_attention_masks_4d
        except Exception:
            continue  # this policy isn't available in this build; skip quietly
        if getattr(meth, "_bool_patched", False):
            out.append(f"{cls_name}: already patched")
            continue

        @functools.wraps(meth)
        def patched(self, att_2d_masks, *a, _orig=meth, **k):
            return _orig(self, att_2d_masks.bool(), *a, **k)

        patched._bool_patched = True
        cls._prepare_attention_masks_4d = patched
        out.append(f"{cls_name}: patched")
    return "attention_mask_bool: " + (", ".join(out) if out else "no targets found")


def apply_lerobot_compat_patches(verbose: bool = False):
    """Apply every lerobot compat shim once. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return
    status = [_patch_create_causal_mask(), _patch_attention_mask_bool()]
    _APPLIED = True
    if verbose:
        for s in status:
            print(f"[lerobot_patches] {s}")
    return status

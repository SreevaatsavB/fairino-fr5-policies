# 9. "Is the model too small?" — No. It's overfitting, not under-capacity.

A natural question, given the architecture *looks* modest (a 32-dim CVAE latent, `d_model=512`)
and the **train/val loss gap is large** (train ~0.05, val ~0.14–0.17 — see file 3): *do we just
need a bigger model?*

**Short answer: no.** A large train/val gap is the textbook signature of **overfitting**, which
means the model has **enough (if anything, too much) capacity for the data** — not too little.
Growing the model would most likely make the gap **worse**.

---

## 9.1 Under-capacity vs overfitting are opposite signatures

| | train loss | val loss | gap | what it means | fix |
|---|---|---|---|---|---|
| **Under**fitting (too small / under-trained) | **high** | high | **small** | model can't even fit the training data | more capacity / longer training |
| **Over**fitting (our case) | **low (~0.05)** | **high (~0.15)** | **large (~0.10)** | model *memorizes* the train set, doesn't generalize | **more / more-diverse data**, regularization, best-checkpoint selection |

Our numbers (train ~0.05, val ~0.15, gap ~0.10) are squarely the **overfitting** row. If the model
were too small we'd see *both* losses stuck high with a *small* gap — the opposite of what the logs
show.

---

## 9.2 The 32-dim latent is standard — and isn't the capacity knob anyway

`d_model=512`, `latent_dim=32`, 4 encoder / 7 decoder layers is the **canonical ACT configuration**
— exactly what the ACT paper (ALOHA) and lerobot use, and what people run on custom bots. This repo
is, if anything, slightly *larger* than lerobot's default (7 decoder layers).

Crucially, **`latent_dim=32` is the CVAE "style" latent** — it encodes *which demonstration mode*
this is, not the model's main representational capacity. Bumping it does **not** add the kind of
capacity that would close a generalization gap; it just gives the (already-collapsing, file 6)
latent more dimensions to ignore.

---

## 9.3 The decisive evidence: it's data-bound, not capacity-bound

From the training-log analysis (file 3): the **94M-parameter paper-scale ACT had the *same*
validation loss as the small one.** When *more capacity does not lower val loss*, the bottleneck is
**data, not parameters** — adding params just memorizes the same small dataset faster. With ~54–86
episodes, the model is data-limited.

This matches community practice: ACT overfitting is universally treated as a *data / checkpoint*
issue ("divergence between train and val curves — do **not** use the final checkpoint; pick the
lowest-val one"), and the tuning order when success is low is **chunk size → β/KL → learning rate**
— *never* "add parameters."

---

## 9.4 What people actually do for production ACT on custom bots

- **Keep the standard 512/32 arch.** Nobody scales the ACT backbone for a single custom task.
- **Scale data, not the model.** ~**50 demos → 80%+** on a simple pick-place is typical; the lever
  is **more *diverse* demos** (100–150, varied object positions), not more params.
- **Reduce *trainable* capacity to fight overfitting** — which is exactly what a **frozen
  DINOv2/DINOv3 backbone** does (it removes ~11M trainable ResNet params and replaces them with a
  strong, frozen visual prior the 54-episode set could never learn itself). Plus photometric
  augmentation and **best-checkpoint selection** (by the probe, file 8 / `experiments/shortcut_probe.py`,
  not val_l1).
- **Bigger model only if** you move to *multi-task / many-object generalization* **and** bring a lot
  more data — and even then the right move is a **pretrained VLA** (the **Octo** finetune in this
  repo, or π0), not a bumped `latent_dim`.

---

## 9.5 What actually closes the gap (in priority)

1. **More + more-diverse data** (fixed-home + varied-marker, 100–150 demos) — the real fix, and the
   one that also kills the proprioceptive shortcut (file 2 §2.2 / file 4 §4.2).
2. **Frozen pretrained backbone** (DINOv2/v3) — fewer trainable params + a strong visual prior. ✅ done.
3. **Augmentation + checkpoint selection by the vision-vs-state probe** — ✅ in place.
4. A **pretrained VLA** (Octo / π0) if you need broad generalization — not a bigger ACT.

> **One-line takeaway:** the 32-dim latent and 512-wide ACT are correct and standard; the large
> train/val gap is **overfitting on a small dataset**, so the cure is **more/diverse data + the
> frozen pretrained encoder + probe-based checkpoint selection** — *not* more parameters. Growing the
> model would likely widen the gap.

---

### References
- Zhao et al. — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT) — RSS 2023 —
  [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- lerobot ACT config —
  [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py)
- ACT practical guide (SVRC) — https://www.roboticscenter.ai/blog/act-policy-explained
- Shaka-Labs ACT (sub-30-demo training) — https://github.com/Shaka-Labs/ACT
- See also file 3 (`03_training_log_findings.md` — the val-plateau / capacity finding) and file 4
  (`04_root_cause_and_fixes.md` — the data fix).

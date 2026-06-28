# 6. The CVAE KL Term — a from-scratch deep dive

This file zooms all the way in on the single most important *training* finding in this folder:
the **KL term of ACT's CVAE collapsed to ~0** ("posterior collapse"), which silently turned ACT
into a plain regressor that mode-averages. To understand that, you need to understand what the KL
term *is*, why it must be there, and why it died. This file builds it up from zero.

Cross-refs: the consequence (mode averaging) is [file 2 §2.3](02_failure_modes_explained.md);
the measured collapse is [file 3 §3.2](03_training_log_findings.md); the fixes are
[file 4 §4.1](04_root_cause_and_fixes.md).

---

## 6.1 Why ACT has a latent `z` at all

ACT's core problem: **for one observation there are many valid action sequences** — a human
grasps slightly differently every time (different approach angle, speed, exact grasp point). A
plain network forced to output *one* answer for *many* valid ones outputs their **average**
(mode averaging — and the average of "reach left" and "reach right" is "reach to the middle,"
which grasps nothing).

The CVAE's fix is to add a **latent variable `z`** (32-dim in your config: `latent_dim: 32`) that
encodes *"which style/mode is this demonstration."*

```
                    TRAINING                                    INFERENCE
   ┌───────────────────────────────────────┐      ┌──────────────────────────────────┐
   │ encoder sees the REAL demo action chunk│      │ no demo to encode at deploy time │
   │   → infers a z for it                  │      │   → just set z = 0 (prior center)│
   │   ("this was a left-reach, slow")      │      │   ("give me the default mode")   │
   │ decoder reconstructs actions GIVEN     │      │ decoder produces actions GIVEN   │
   │   (observation, z)                     │      │   (observation, z=0)             │
   └───────────────────────────────────────┘      └──────────────────────────────────┘
```

Because the decoder is *told which mode* via `z`, it never has to average. **But this only works
if `z` actually carries the mode information.** The KL term is what forces `z` to do that — and
what makes `z = 0` a meaningful choice at inference.

---

## 6.2 The CVAE loss = reconstruction + KL

ACT is trained to maximise the **ELBO** (evidence lower bound), which as a loss is:

```
L  =  reconstruction_loss      +      β · KL_term
      └──── the L1 term ────┘          └── β = kl_weight = 10 in your config ──┘
```

- **Reconstruction loss** — "did the decoder reproduce the demonstrated actions?" (your L1).
- **KL term** — a regulariser on the latent. **This is the part you asked about.**

The `β` (= `kl_weight`) is the knob that sets the balance between the two. Your runs used
`kl_weight = 10`, which — as we'll see — is a big part of why it collapsed.

---

## 6.3 What the KL term actually *is* (the math, explained)

The encoder does **not** output a single `z`. It outputs a **distribution** over `z` — a Gaussian
with a mean `μ` and standard deviation `σ` predicted from the demo:

```
q(z | demo)  =  Normal(μ, σ²)          ← the "posterior": the z this specific demo implies
```

The **prior** is a fixed standard Gaussian:

```
p(z)  =  Normal(0, 1)                  ← the "prior": z before seeing any demo
```

The **KL term** is the **KL divergence** between those two — a single number measuring *how far
the encoder's predicted distribution is from the standard Gaussian*:

```
KL_term  =  KL( q(z|demo) ‖ p(z) )
```

For two Gaussians this has a clean closed form (per latent dimension, summed over all 32):

```
KL  =  ½ · Σ_i ( μ_i² + σ_i² − 1 − log σ_i² )
```

Read the formula intuitively:

| Situation | KL value | Meaning |
|---|---|---|
| `μ = 0`, `σ = 1` (posterior = prior) | **0** | the encoder output is identical to the prior → `z` says **nothing** about the demo |
| `μ` pushed away from 0 | **grows** | the encoder uses the latent *mean* to encode demo-specific info |
| `σ` shrunk below 1 | **grows** | the encoder is *confident/precise* about this demo's `z` |

So the KL term is a direct measure of **how much information the encoder is packing into `z`:**

- **KL large** → posterior far from prior → different demos get **different** `z` → `z` carries
  real mode information. **Good.**
- **KL ≈ 0** → posterior ≈ prior for *every* demo → the encoder outputs the same standard-Gaussian
  noise no matter what → `z` carries **nothing**. **This is the collapse.**

### The reparameterisation trick (one-line aside)
You can't backprop through "sample `z` from Normal(μ,σ)". So ACT samples noise `ε ~ Normal(0,1)`
and computes `z = μ + σ·ε`. The randomness lives in `ε` (no gradient needed); gradients flow
through `μ` and `σ`. That's just *how* the latent is made trainable — not the bug, but it's why
`μ`, `σ` are learnable quantities the KL can act on.

---

## 6.4 Why the KL term has to be there (the tug-of-war)

The two loss terms pull `z` in **opposite directions**:

| Term | Wants `z` to… | Why |
|---|---|---|
| **Reconstruction (L1)** | carry **lots** of info | the more `z` tells the decoder, the easier to reconstruct the exact demo → lower L1 |
| **KL** | carry **no** info (μ→0, σ→1) | the closer the posterior is to the prior, the lower the KL |

A **healthy** CVAE settles in the middle: `z` carries *just enough* information to capture the
demo's **mode/style** (not every pixel-level detail), with a smooth, prior-shaped latent space.
`β = kl_weight` sets where that balance lands — higher β pushes harder toward "z carries nothing."

There is a second, subtler reason the KL must be there: **it is what makes `z = 0` valid at
inference.** Training pulls every demo's posterior toward the `Normal(0,1)` prior, so the prior's
center `z = 0` becomes a meaningful "average/default mode" the decoder knows how to decode. Remove
the KL and the latent space becomes arbitrary — `z = 0` would point at garbage.

---

## 6.5 Posterior collapse — what went wrong in your runs

**Posterior collapse** = the KL side of the tug-of-war wins *completely*. The encoder learns to
output `μ ≈ 0, σ ≈ 1` for **every** demo — i.e. exactly the prior — so `KL → 0`. Your logs show
this precisely, in every ACT run:

```
KL over epochs (train_dino_frozen.out, representative of ALL runs):
  ep1   0.2107
  ep2   0.0509
  ep3   0.0171
  ep5   0.0054
  ep10  0.0012
  ep15  0.0004
  ep20  0.0002
  ep30+ 0.0001     ← dead (the partial-unfreeze run prints a literal 0.0000)
```

Halved every 1–2 epochs, effectively dead by ~epoch 10–15. With `kl_weight = 10`, the KL's
contribution to the total loss at that point is `10 × 0.0001 = 0.001` — negligible next to an L1
of ~0.08–0.15. The latent is gone.

### What that does to the model
```
z carries no info  →  decoder learns z is useless noise  →  decoder ignores z entirely
                                                                      │
                                  ACT now predicts actions from the observation ALONE
                                                                      │
                            = a plain regression network  →  outputs E[action | obs]
                                                                      │
                                  = the AVERAGE action  →  mode averaging (file 2 §2.3)
                                                                      │
                          object-blind "mean" reach + mid-air grasp (your deploy logs)
```

**In one sentence:** the one architectural feature meant to *prevent* averaging
(**the CVAE latent**) **erased itself during training**, in every run, and nobody noticed because
the overall loss still went down smoothly.

---

## 6.6 Why it collapsed (the cause, concretely)

Two ingredients, both present in your setup:

1. **High `kl_weight = 10`.** A large β over-weights "push the posterior to the prior," so the
   optimiser finds it *cheaper to zero the KL* than to learn to use `z`. This is the classic,
   well-documented trigger for posterior collapse (Bowman et al. 2016, who invented the standard
   fix while training text VAEs).
2. **A powerful decoder.** ACT's transformer decoder is strong enough to reconstruct the action
   chunk *reasonably well without `z`*. When the decoder can "do fine" ignoring the latent, the
   reconstruction term doesn't fight to keep `z` alive, so the KL pressure wins unopposed.

There's also a **chicken-and-egg** dynamic at the start of training: before the decoder has
learned to *use* `z`, the latent genuinely looks like useless noise — so the KL term happily
shrinks it toward zero in the first few epochs, and then the decoder never gets a reason to start
using it. Once collapsed, it stays collapsed. (That's why yours died by ~epoch 10, never to
recover, in every run.)

---

## 6.7 How to keep the KL healthy (the fixes)

All three break the "KL wins instantly" dynamic. Use one or combine them.

| Fix | What it does | Why it works |
|---|---|---|
| **Lower `kl_weight`** (10 → ~0.1–1.0) | reduce β so the latent isn't crushed | less pressure toward the prior → `z` survives and stays informative |
| **KL annealing / warm-up** | start β ≈ 0, ramp it up over the first few thousand steps | lets the decoder *first learn to use `z`* while it's free, *then* gently regularise — so it can't collapse before it's useful (Bowman et al.'s original fix for exactly this) |
| **Free bits** | floor the KL per dimension, e.g. `KL_used = max(KL, λ)` with λ≈0.1 | the optimiser gets no reward for pushing KL below the floor, so it physically cannot drive it to zero |

### How to confirm it worked
1. **Plot KL over training** — it should **stabilise at a non-trivial value** (roughly 0.5–5),
   not crater to ~0.0001.
2. **Sanity-check the latent is alive** — feed the *same* observation with two different `z`
   samples; the predicted action chunk should **visibly differ**. If `z` changes the output *and*
   KL stays > 0, the CVAE is back and ACT can commit to a mode instead of averaging.

---

## 6.8 One-paragraph summary

The KL term measures how much real information ACT's latent `z` carries — formally the KL
divergence between the encoder's per-demo Gaussian posterior `q(z|demo)=Normal(μ,σ²)` and the
standard-normal prior `p(z)=Normal(0,1)`, which equals `½Σ(μ²+σ²−1−logσ²)` and is zero only when
the posterior equals the prior (μ=0, σ=1). It must stay non-trivial for `z` to encode *which mode*
a demo is — and for `z=0` to be a meaningful default at inference — so the decoder can commit to
one mode instead of averaging. In your runs, `kl_weight=10` plus a strong transformer decoder
drove the KL to ~0 within ~10 epochs in every run (**posterior collapse**): the latent went dead,
the decoder ignored it, and ACT degenerated into a plain mean-predicting regressor — which is the
root training cause of the object-blind, mode-averaged behaviour you saw on the robot. Fix it with
a lower KL weight, KL annealing, and/or free bits, and verify the KL stabilises above zero and the
latent actually changes the output.

---

### References for this file
- Kingma & Welling — *Auto-Encoding Variational Bayes* (the VAE / ELBO / reparameterisation) — 2013 —
  [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Sohn, Lee, Yan — *Learning Structured Output Representation using Deep Conditional Generative
  Models* (the CVAE) — NeurIPS 2015
- Bowman et al. — *Generating Sentences from a Continuous Space* (posterior collapse + KL
  annealing) — CoNLL 2016 — [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)
- Zhao et al. — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT; the
  CVAE and the 35%→2% ablation) — RSS 2023 — [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- See also [`docs/act.md`](../act.md) §6 (the CVAE loss) in this repo.

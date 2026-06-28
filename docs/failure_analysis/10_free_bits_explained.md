# 10. Free-bits, explained in detail (the "tax-free allowance" for the latent)

This is a focused, slow walkthrough of **free-bits** — one of the three CVAE anti-collapse
mechanisms (file 8 §8.1). The name is unhelpful; the idea is simple once you see it as a *tax*.
For the broader KL/CVAE context read [`06_cvae_kl_deep_dive.md`](06_cvae_kl_deep_dive.md) first;
this file zooms all the way in on free-bits.

---

## 10.1 First: what the KL term does to the latent

The CVAE loss is `L = reconstruction(L1) + kl_weight · KL`. The **KL term is a penalty** in the
loss. It penalizes the latent `z` for carrying information. Because the optimizer *minimizes* the
loss, it tries to make the KL **small** — which means it pushes the information in `z` toward
**zero**.

The total KL is a sum over all 32 latent dimensions:

```
KL_total  =  kl_1 + kl_2 + ... + kl_32
```

where each `kl_i` measures *how much information dimension i carries* (in **nats**, a unit of
information). `kl_i = 0` means dimension *i* is **dead** — it carries nothing and the decoder can't
use it.

**The collapse:** because the optimizer wants the KL small, it drives **every** `kl_i → 0`. All 32
dims die → the latent carries nothing → the decoder learns to ignore `z` → ACT degenerates into a
plain mean-regressor (mode averaging → the object-blind grasp). That is exactly what the FR5
training logs showed (KL `0.21 → 0.0001`).

---

## 10.2 The mental model: KL is a *tax on information*

Imagine that every nat of information you store in the latent gets **taxed** by the KL penalty.
What does a rational tax-avoider do? Store **nothing**. Empty latent → collapse.

That is the whole problem in one image: **the KL taxes information, so the model avoids storing
any.**

---

## 10.3 Free-bits = a tax-free allowance per dimension

Free-bits changes the rule: **the first `λ` nats in *each* dimension are tax-free.** Only
information *above* `λ` gets taxed.

Formally, instead of penalizing `kl_i`, you penalize `max(kl_i, λ)`:

```
KL_used  =  max(kl_1, λ) + max(kl_2, λ) + ... + max(kl_32, λ)
```

- If a dimension carries **more** than `λ` → `max(kl_i, λ) = kl_i` → it is taxed, and the KL still
  pulls it down toward `λ`.
- If a dimension carries **less** than `λ` → `max(kl_i, λ) = λ`, a **constant** → its derivative is
  **zero** → there is **no gradient pushing it any lower**.

So the optimizer **gains nothing** by squeezing a dimension below `λ`. Below the floor the KL stops
fighting, and the **reconstruction** loss is then free to *use* that dimension (up to `λ` nats)
because it costs nothing. Net effect: each dimension keeps ~`λ` nats of usable information, and the
latent **stays alive**.

> One sentence: **free-bits gives each latent dimension a tax-free allowance of `λ` nats — the
> model can store that much "for free," so it stops emptying the latent to zero.**

---

## 10.4 The gradient view (why "no push below the floor")

The reason free-bits works is entirely about the **gradient** of `max(kl, λ)`. Plot the penalty a
dimension pays as a function of how much info it carries:

```
 penalty paid
 by the dim
   │
   │                                  ╱   slope = 1  (taxed: gradient pulls kl DOWN toward λ)
   │                                ╱
   │                              ╱
 λ ┤━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╱
   │        flat  (no gradient)
   │   ── the dim is NOT pushed down here ──
   0 └──────────────────────────┴────────────────►  kl_i  (info the dim carries, nats)
   0                            λ

  kl_i < λ  : penalty is flat at λ  → gradient 0  → optimizer leaves the dim alone
  kl_i > λ  : penalty rises with kl → gradient pulls it back down toward λ
```

- **Right of `λ`** (a dim carrying a lot): the KL tax has a downward gradient → it shrinks the dim
  back toward `λ`. (Normal regularization — keeps the latent from carrying *too* much, which would
  break `z = 0` inference.)
- **Left of `λ`** (a dim nearly dead): the penalty is a flat line → **gradient is zero** → the KL
  does **not** drag it to zero. The dimension is "parked" with up to `λ` nats it may use for free.

Without free-bits the curve is just `penalty = kl` (a straight line through the origin with slope 1
everywhere) → there is *always* a downward gradient → every dim is dragged to 0 → collapse. Free-
bits flattens the curve below `λ`, removing that downward pull exactly where collapse happens.

---

## 10.5 A tiny worked example (λ = 0.5, 4 dimensions)

| Dimension's `kl_i` | **Without** free-bits (penalty = `kl_i`) | **With** free-bits (penalty = `max(kl_i, 0.5)`) |
|---|---|---|
| 0.1 (nearly dead) | 0.1 — **gradient pulls it to 0** ❌ dead | **0.5 (flat) — gradient 0 → NOT pushed down** ✅ kept |
| 0.5 (at the floor) | 0.5 — pulled toward 0 | 0.5 — no downward pressure |
| 2.0 (carries a lot) | 2.0 — pulled down hard | 2.0 — taxed only on the part **above** 0.5 |

Without free-bits, that first dimension gets dragged to 0 and dies. With free-bits it keeps its
0.5 nats for free, so the model actually **uses** it instead of abandoning it.

---

## 10.6 Why it *specifically* stops the collapse (the chicken-and-egg)

Posterior collapse is a vicious cycle:

```
early in training: z is random noise  ─►  it doesn't help reconstruction yet
                                          ─►  reconstruction has no reason to "defend" it
                                          ─►  the KL tax (downward gradient) kills it: kl_i → 0
                                          ─►  now the dim is dead, the decoder never learns to use it
                                          ─►  permanently collapsed
```

Free-bits breaks the cycle at the third arrow: **below `λ` there is no downward KL gradient**, so the
dims are *not* forced to zero. They stay parked with a small free budget, which gives reconstruction
the chance to start using them once it learns how. (KL **annealing** — file 8 — helps the same cycle
from a different angle: it removes the KL pressure *entirely* early on so reconstruction can learn to
use `z` first; free-bits is the *steady-state* guarantee that no dim hits exactly zero.)

---

## 10.7 In our code (and the exact value we use)

`policies/act/model.py`, in `ACT.forward`:

```python
# per-latent-dim KL: kld_dim has shape (B, 32)
kld_dim = -0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())
# FREE-BITS: floor each dim at lambda, THEN sum over dims, THEN mean over batch
kld = kld_dim.clamp(min=self.kl_free_bits).sum(-1).mean()
#                   └─ clamp(min=λ) is exactly max(kl_i, λ) per dimension ─┘
```

- `clamp(min=0.03)` replaces any dimension whose KL is below **0.03** with a flat `0.03` → no
  gradient below it → that dimension cannot be squeezed to zero.
- With 32 dims, the latent keeps **at least ~`0.03 × 32 ≈ 1 nat`** of information, instead of
  collapsing to ~0.

### Why `λ = 0.03` (gentle), not `0.1`
We deliberately use a **small** floor. ACT *intends* the latent to be a **weak** "style hint"
(because `z = 0` at inference — see file 8 §8.1). A large floor (the original `0.1/dim ≈ 3.2 nats`)
would *force* the latent to carry a lot, then zero it at deploy → a train/test mismatch. `0.03/dim`
(~1 nat total, inside the healthy 0.5–5 KL band) is just enough to stop the latent dying, **not**
enough to over-force it. The free-bits floor is intentionally minimal; it is the cheap insurance,
not the main lever (the lower `kl_weight` is).

---

## 10.8 One-paragraph summary

The KL term taxes the latent for carrying information, so the optimizer empties the latent to zero
(posterior collapse). Free-bits gives each of the 32 latent dimensions a **tax-free allowance of
`λ` nats** — implemented as `max(kl_i, λ)` (in code, `clamp(min=λ)`): above `λ` the dimension is
taxed normally (gradient pulls it down toward `λ`), but **below `λ` the penalty is flat so its
gradient is zero**, meaning the optimizer never drags it to zero. This keeps every dimension carrying
at least `λ` nats of usable information, breaks the collapse, and — kept deliberately small
(`λ = 0.03/dim`) — does so without forcing the latent to be strong enough to hurt the `z = 0`
inference that ACT relies on.

See also [`06_cvae_kl_deep_dive.md`](06_cvae_kl_deep_dive.md) (KL/ELBO/reparameterisation from
scratch) and [`08_the_fix_implementation.md`](08_the_fix_implementation.md) §8.1 (free-bits +
annealing + lower-weight together, and the honesty note).

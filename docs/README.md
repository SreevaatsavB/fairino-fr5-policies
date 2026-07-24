# policy explainers

plain-English deep dives on each policy in this repo — what it is, how it works, the math, and how it differs from the others. read in this order; each builds on the last.

> **training the π-family VLAs** — the self-contained RunPod notebooks: every parameter,
> VRAM/speed tuning, checkpointing + resume, Hub push, and the evaluation dashboard, in
> [`runpod_training.md`](runpod_training.md).
>
> **quantization** — QLoRA training (fit a big model on a small pod) and post-training
> inference quantization (`deploy.py --quantize nf4`), what's/isn't quantized, and the
> Turing+ requirement, in [`quantization.md`](quantization.md).
>
> **precision** — the exact dtype map (what runs bf16, what deliberately stays fp32,
> why never fp16, and the NF4 storage-vs-compute distinction), in
> [`precision.md`](precision.md).
>
> **how they differ at deployment** — query cadence, temporal ensembling, action queues, the
> 30 Hz budget, per-model latency, and loading checkpoints (local / `--hf-repo` / `--quantize`)
> are compared in one place in [`inference.md`](inference.md).
>
> **evaluating a trained policy** — log predicted actions over time (robot) and compare
> prediction vs ground truth on the held-out test set (per-dim plots + MAE) with the shared
> action-logging layer: [`evaluation.md`](evaluation.md).
>
> **want to learn by coding?** [`exercises/`](exercises/) has from-scratch PyTorch exercises
> (attention, CVAE, DDPM/DDIM, flow matching, DiT/AdaLN, FAST tokenization, temporal ensembling)
> with stubs + self-checks + tested solutions. run `python docs/exercises/check_all.py`.

| # | doc | model | one-liner |
|---|---|---|---|
| — | [il_failure_modes.md](il_failure_modes.md) | **all** | why BC fails (causal confusion, covariate shift, data gap, open-loop blindness) — mapped to FR5 |
| — | [action_spaces.md](action_spaces.md) | **all** | action representation design: why delta EEF is the field standard, TE compatibility, FR5 migration path |
| — | [proprioception_modes.md](proprioception_modes.md) | **all** | benchmark state dependence: full / dropout / none modes, one-flag runs, what to measure |
| 1 | [act.md](act.md) | **ACT** | transformer directly predicts a 100-step action chunk; CVAE captures demo variability |
| 2 | [diffusion_policy.md](diffusion_policy.md) | **Diffusion Policy** | denoise a corrupted action chunk back to clean (DDPM + 1-D U-Net) |
| 3 | [dit_flow.md](dit_flow.md) | **DiT + Flow Matching** | same denoising idea, but transformer denoiser + straight-line flow matching + language |
| 4 | [pi0.md](pi0.md) | **π0** | flow matching, but the encoder is a full 2B vision-language model (PaliGemma) |
| 5 | [pi05.md](pi05.md) | **π0.5** | π0 tuned for open-world generalization (longer language, quantile norm, co-training) |
| 6 | [pi0_fast.md](pi0_fast.md) | **π0-FAST** | π0's backbone, but actions become discrete tokens generated like text (no ODE) |
| 7 | [octo.md](octo.md) | **Octo** | the definitive Octo doc — JAX generalist pretrained on 800k Open-X trajectories; pretraining background, architecture, all 3 FR5 workflows, the data adapter, action-space/normalization, 30 Hz deployment, and the isolated dependency stack |
| — | [octo_model.md](octo_model.md) | **Octo** | the Octo architecture from scratch — tokenizers, block-wise attention, readout tokens, the diffusion action head |
| — | [octo_finetuning.md](octo_finetuning.md) | **Octo** | how Octo is finetuned — full vs head-only vs LoRA, frozen_keys, the optimizer recipe |

---

## data & inference pipeline

```mermaid
flowchart LR
    subgraph Record
        A[Human Teleoperation\n100 Hz CSV + 30 fps video]
    end
    subgraph Convert
        B[convert_episodes.py\nresample → LeRobot v3 dataset]
    end
    subgraph Train
        C[FR5Dataset\naugmentation + chunking]
        D{policy}
        D -->|ACT| E[CVAE + Transformer]
        D -->|Diffusion| F[U-Net DDIM]
        D -->|DiT+Flow| G[DiT Euler ODE]
        D -->|π0 family| H[PaliGemma VLA]
    end
    subgraph Deploy
        I[deploy.py\n30 Hz control loop]
        J[FR5 Robot]
    end
    A --> B --> C --> D
    E & F & G & H --> I --> J
```

---

## the big picture — two families

```
DIRECT PREDICTION                 ITERATIVE GENERATION
─────────────────                 ────────────────────
ACT                               Diffusion Policy ─┐
  predict actions                   denoise (DDPM)  │
  in one forward pass               U-Net           │
  CVAE for multimodality                            │  same "corrupt the
                                  DiT + Flow Matching│  future, learn to
                                    denoise (flow)   │  correct it" DNA
                                    transformer      │
                                    + language       │
                                                     │
                                  π0 / π0.5 ─────────┤
                                    flow matching    │
                                    + 2B VLM brain   │
                                                     │
                                  π0-FAST ───────────┘
                                    actions as tokens
                                    (autoregressive)
```

**the shared paradigm** (everything except ACT): condition on the present (images + joints + maybe language), corrupt the future action chunk with noise, train the network to predict the correction. they differ in *how* they corrupt (DDPM vs flow matching vs tokenization), *what* denoiser they use (U-Net vs DiT vs full VLM), and *whether* they use language.

**ACT** sits apart: no denoising, just direct prediction with a CVAE latent for handling demo variability.

---

## how to pick one

| your situation | use |
|---|---|
| simplest baseline, CPU ok, single task | **ACT** |
| multimodal task (many valid solutions), have a GPU | **Diffusion Policy** |
| want language conditioning + modern flow recipe | **DiT + Flow Matching** |
| want a pretrained foundation model, multi-task, generalization | **π0 / π0.5** |
| same as π0 but inference speed is critical | **π0-FAST** |

> reality check for this repo: we have very few demonstrations. these are pipeline
> integrations, not trained-to-convergence policies. ACT and Diffusion Policy are the
> realistic ones to actually train here; the π0 family needs a GPU + PaliGemma weights
> and shines with large/diverse data.

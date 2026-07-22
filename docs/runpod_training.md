# Training the π-family VLAs on RunPod (self-contained notebooks)

The π0 / π0.5 / π0-FAST vision-language-action models are far too large to train on
the robot box, so they train on a rented GPU (RunPod). Each notebook in
`notebooks/` is **fully self-contained** — no `git clone`, no repo files needed on
the pod. Everything (dataset class, policy wrapper, training loop, evaluation) is
defined inline, and the checkpoints it writes stay **byte-compatible with the
repo's `common/deploy.py`** so you can pull a trained model straight onto the FR5.

| notebook | model | recipe |
|---|---|---|
| `notebooks/train_pi0_runpod.ipynb` | π0 | flow-matching VLA, QLoRA on the VLM + full action expert |
| `notebooks/train_pi05_runpod.ipynb` | π0.5 | π0 with a longer language context (tokenizer_max_length=200) |
| `notebooks/train_pi0_fast_runpod.ipynb` | π0-FAST | autoregressive FAST tokens; full-finetune by default |
| `notebooks/convert_and_push_dataset.ipynb` | — | raw HF episodes → LeRobot dataset → push to the Hub |

> **New to why these need quantization / LoRA?** Read
> [`quantization.md`](quantization.md) first — this doc references those knobs
> and assumes you know what NF4 / QLoRA mean.

---

## 0. What you need before you start

1. **A Python-3.12 GPU pod.** lerobot 0.5.1 requires Python ≥ 3.12, and RunPod's
   default PyTorch images ship 3.11. Pick an image on an **Ubuntu 24.04** base
   (that's what ships 3.12). Verify in the pod terminal: `python --version`.
   - VRAM: **24 GB** works (4090, with `expert_only` or NF4-QLoRA), **48 GB**
     (A40 / A6000 / L40S) is comfortable, **80 GB** (A100/H100) trains fastest.
   - If you want more than one dataloader worker, launch the pod with
     `--shm-size 16g` (the default `/dev/shm` is tiny and crashes workers).
2. **A HuggingFace token (read + write)** whose account has **accepted the
   PaliGemma license**: <https://huggingface.co/google/paligemma-3b-pt-224>.
   The tokenizer is gated there; the actual weights come from `lerobot/pi05_base`
   etc. Set it as a pod secret / env var `HF_TOKEN`, or the notebook prompts for
   it (masked). **Never hardcode it** — it is read from `os.environ` or `getpass`.
3. **The dataset on the Hub, in LeRobot v3 format.** If your recordings are raw
   `episode_XXXX/` folders, run `convert_and_push_dataset.ipynb` first, then set
   `HF_DATASET_REPO` to its output repo. The training notebook auto-converts if
   you point it at raw episodes, but pushing a converted dataset once is faster.

---

## 1. The parameters cell (cell 2) — every knob

```python
HF_DATASET_REPO = "<you>/fr5-pick-place-lerobot"   # LeRobot v3 dataset on the Hub
PRETRAINED      = "lerobot/pi05_base"  # the base VLA weights ("" = random init, smoke only)
TASK_TEXT       = "pick up the block and place it in the bin"  # fallback prompt; per-episode
                                                               # prompts from the dataset win

# ---- finetuning recipe ----
FINETUNE_MODE = "auto"   # auto | lora | full | freeze_vision | expert_only
QUANTIZE      = "none"   # none | nf4 | int8. The official recipe is plain bf16 LoRA; NF4
                         # (QLoRA) is the MEMORY fallback for <=24 GB cards — quality-neutral
                         # per the QLoRA paper, but ~20-30% slower per step (dequantization)
LORA_RANK     = 16       # LoRA rank (openpi uses 16 on the 2B VLM)
LORA_TARGETS  = [...]    # attention q/k/v/o AND MLP gate/up/down — openpi LoRAs attn+ffn,
                         # and QLoRA finds all-linear adapters match full-finetune quality
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05

# ---- optimization (openpi-style: budget in STEPS, warmup + cosine decay) ----
BATCH_SIZE   = None      # None -> auto from free VRAM; set an int to override
MAX_STEPS    = 30_000    # optimizer-step budget. This is how openpi finetunes the pi family
                         # (30k steps at batch 32) — NOT epochs. 100 epochs here would be
                         # ~330k steps, 11x the official budget, with no evidence of benefit.
                         # Set None to budget by EPOCHS instead (MAX_EPOCHS then drives both
                         # the stop and the cosine-decay horizon).
WARMUP_STEPS = 1_000     # linear warmup, then cosine decay to LR_MIN
MAX_EPOCHS   = 100       # hard cap when MAX_STEPS is set; THE budget when MAX_STEPS=None
LR           = 2.5e-5    # PEAK lr (openpi: 5e-5 at batch 32-64; scaled for our smaller batch)
LR_MIN       = 2.5e-6    # cosine floor
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
CHUNK_SIZE   = 50        # action horizon (1.67 s @ 30 Hz)

# ---- data / throughput ----
FRAME_STRIDE = 5         # sample every Kth frame as a chunk START (cuts redundant 30 fps
                         # samples; the chunk itself stays full-rate)
GRAD_CKPT    = True      # gradient checkpointing. With NF4 you have VRAM to spare -> set
                         # False for ~20-30% faster steps (uses more memory)
NUM_WORKERS  = 0         # DataLoader workers. 0 is safe on a small /dev/shm; set 4-8 only if
                         # you launched the pod with --shm-size 16g (biggest speedup if the
                         # GPU sits idle between steps)
VAL_FRAC     = 0.1
AUG_LEVEL    = "crops"   # none | crops
PROPRIO_MODE = "full"    # full | dropout | none  (benchmark axis, see proprioception_modes.md)

# ---- logging / checkpointing ----
LOG_EVERY  = 25          # optimizer STEPS between metric logs (wandb + metrics_steps.csv)
SAVE_EVERY = 10          # epochs between periodic checkpoints
PUSH_EVERY = 10          # epochs between Hub pushes of best.pt (0 = only the final cell)
RESUME     = "auto"      # "auto" -> continue from CKPT_DIR/last.pt if present; "" -> fresh
```

### FINETUNE_MODE — what actually trains

| mode | VLM | action expert | when |
|---|---|---|---|
| `lora` | frozen + LoRA adapters on q/k/v/o | fully trained | **default** on ≥40 GB (or any card with NF4) |
| `expert_only` | fully frozen | fully trained | the 24 GB option; least overfitting on few demos |
| `freeze_vision` | SigLIP frozen, rest trained | fully trained | middle ground |
| `full` | fully trained | fully trained | 80 GB, most capacity, most overfitting risk |
| `auto` | picks `lora` if VRAM ≥ 40 GB **or** QUANTIZE is nf4/int8, else `expert_only` | | |

`QUANTIZE="nf4"` **requires** LoRA (a 4-bit base takes no gradient), so it forces
the run to `lora` and the adapters carry the whole VLM update. `full` + quantize
is rejected. π0-FAST is `full` by default and uses `LORA_RANK` instead of
`FINETUNE_MODE` — see its notebook.

---

## 2. Cell-by-cell walkthrough

| cell | what it does |
|---|---|
| **2 · Parameters** | everything above. The HF token is read from env or a masked prompt. |
| **4 · Install** | pip-installs lerobot + a pinned DL stack (transformers, peft, bitsandbytes, cv2, …) **verbosely** (a silent wrong-version resolve costs hours later), then imports every dependency up front so a bad one fails in seconds, not after a multi-GB pull. |
| **6 · HF auth** | logs in, checks the PaliGemma license is accepted. |
| **8 · GPU check** | prints total **and free** VRAM, picks `FINETUNE_MODE`/`BATCH_SIZE`, warns if a dead process is squatting on VRAM (does **not** kill anything). |
| **5 · Dataset** | pulls `HF_DATASET_REPO`; auto-converts raw episodes → LeRobot if needed. |
| **6b · GIF preview** | replays one training episode (cameras + recorded joints) as a GIF and prints its language prompt — catch a dead camera / mislabeled episode before an 8-hour run. |
| **7 · Wrapper** | defines the inline `PiPolicy` (multi-camera, verified pretrained load, LoRA, NF4). |
| **9 · Build + time a step** | builds the model **directly on the GPU** (never staged through host RAM), loads the base, quantizes, injects LoRA, and times one forward+backward. Prints the **true** training-step peak VRAM and free headroom — tune `BATCH_SIZE` from this. |
| **10 · Train** | the loop: per-step logging, per-epoch val, `last.pt` (resume) + `best.pt` + periodic `epoch_*.pt`, and a background Hub push every `PUSH_EVERY`. |
| **12b · Prediction replay** | rolls the trained policy over a val episode (deployment-faithful) and renders a camera + trajectory video with a playhead. |
| **12c · Full eval** | rolls out **every** val episode, writes per-episode `pred.csv` + `traj.png` + replay MP4 + `summary.csv` under `eval_<policy>/`. Resolution knobs, full-res camera, optional Hub push. |
| **12d · Dashboard** | turns the eval folder into one interactive `dashboard_<policy>.html`: per-joint plots + error, playhead synced to the video. |
| **13 · Ship** | pushes `best.pt` + metrics to `<you>/fr5-<policy>-<mode>` on the Hub. |

---

## 3. Using the GPU fully / going faster

The step-timing cell (9) prints the real footprint, e.g.
`training-step peak VRAM: 31.2 / 46 GB (15 GB free -> room to raise BATCH_SIZE)`.

- **Fill the card:** raise `BATCH_SIZE` (re-run only the training cell — batch is not
  a build-time setting) until the peak sits around **42 / 46** (leave ~4 GB; a real
  step spikes above the average and 46/46 OOMs mid-epoch).
- **Go faster with the VRAM you have:** set `GRAD_CKPT = False` (rebuild required —
  re-run cells 7 → 9 → 10). Checkpointing recomputes activations to *save* memory;
  turning it off keeps them resident and runs ~20-30% faster.
- **If the GPU sits at 0 % util between steps** it is starved for data. Relaunch the
  pod with `--shm-size 16g` and set `NUM_WORKERS = 4`. This is usually the single
  biggest speedup and it does not touch quality.
- **The budget is steps, not epochs.** openpi's official finetune configs run
  **30k optimizer steps** (batch 32) with warmup + cosine decay — that's the whole
  recipe, and it lands around 9-10 epochs here. `MAX_STEPS` enforces it; watch val
  loss and stop earlier if it flattens. Training "100 epochs" (~330k steps) is 11x
  the recipe for no established benefit and is where multi-day ETAs come from.
- For a big batch (≥ ~5× the default) nudge `LR` up toward `5e-5`.

---

## 4. Checkpoints, resume, and the Hub

Three checkpoint files land in `CKPT_DIR`:

| file | contents | purpose |
|---|---|---|
| `best.pt` | model weights + config + stats | **deploy this** — lowest val L1 |
| `epoch_*.pt` | same, every `SAVE_EVERY` epochs | roll back to an earlier point |
| `last.pt` | weights **+ optimizer state + counters** | `RESUME="auto"` continues from here |

Optimizer state lives **only** in `last.pt` (AdamW's fp32 moments would double the
size of every deployable checkpoint otherwise). A crash costs one epoch, not the run.

`/workspace` dies when you release the pod, so pushing to the Hub is not optional:

- **During training:** `PUSH_EVERY=10` pushes `best.pt` + metrics to
  `<you>/fr5-<policy>-<mode>` every 10 epochs (in a background thread).
- **At the end:** cell 13 pushes the final `best.pt`.
- **The eval folder:** set `EVAL_PUSH=True` in cell 12c to push `eval_<policy>/`
  (videos + CSVs + dashboard) to `<you>/fr5-<policy>-eval`.

**Resuming a quantized run:** you cannot resume an NF4 run from a bf16 `last.pt`
(different param shapes). Starting a new quantized run, set `RESUME=""` and a fresh
`CKPT_DIR` so it doesn't collide with old checkpoints.

---

## 5. From the Hub onto the robot

Once `best.pt` is on the Hub, deploy pulls it directly (no manual download):

```bash
python common/deploy.py --hf-repo <you>/fr5-pi0-lora            # pulls best.pt
python common/deploy.py --hf-repo <you>/fr5-pi0-lora --hf-file epoch_0040.pt
```

For a low-VRAM robot box, quantize at load time — even though you trained in bf16:

```bash
python common/deploy.py --hf-repo <you>/fr5-pi0-lora --quantize nf4
```

See [`quantization.md`](quantization.md) for what `--quantize` does and its GPU
requirements, and [`inference.md`](inference.md) for the 30 Hz runtime details.

---

## 6. Common issues

| symptom | cause / fix |
|---|---|
| `Could not find a version ... lerobot==0.5.1` | pod is Python 3.11 — redeploy on an Ubuntu-24.04 (3.12) image |
| kernel dies during **build**, before any step | host-RAM OOM — the notebook builds on-GPU to avoid this; if it still dies the container RAM is tiny, size up the pod |
| `CUDA out of memory` citing "Process NNNN has X GB" | another job (or a crashed kernel) holds VRAM; sizing is from **free** VRAM, but a live job leaves less — do not blanket-kill PIDs, restart the pod to clear a dead one |
| `mat1 float != mat2 BFloat16` under NF4 | fixed — the wrapper autocasts the QLoRA forward to bf16 |
| `create_causal_mask() got an unexpected keyword 'cache_position'` | fixed — lerobot/transformers drift shim in the wrapper cell |
| eval cell "looks hung" at 0/N | it's ~18k model calls at frame_stride=1; use the inner progress bar, cap with `EVAL_MAX_FRAMES`, or do a `EVAL_VIDEO="none"` numbers pass first |

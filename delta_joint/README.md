# Why the pi0 policy wandered, and what this folder does about it

Written after the 2026-07-28 robot trials, where a pi0 checkpoint trained to 30k
steps drove cleanly for ~3 seconds, then wandered for 12 more, and never once
closed the gripper.

The short version: **the model was asked to predict the wrong quantity.** Not
undertrained, not a bad checkpoint, not the format-v3 work. Wrong units.

---

## 1. The problem, in plain language

In our demonstrations the arm moves **0.06 degrees between camera frames**. That
is tiny — the operator teleoperates slowly, and episodes run two minutes.

But we ask the model to output the arm's **full position number** — something
like `-99.26`. Across an episode that number ranges over about 13 degrees.

So the model learns to say *"about -99."* That looks excellent on a plot. But the
`0.06` on the end — the part that actually makes the arm move — is lost in the
rounding.

> It is like measuring the width of a hair with a meter stick. The meter stick
> is not broken. It is the wrong scale for the thing being measured.

Every symptom follows from that one fact.

---

## 2. The evidence

Measured on `experiment_data/training_episodes_processed/` and
`experiment_data/eval_pi0/` — no GPU, no pod, just the recorded data.

### The scale mismatch

```
per-step demo motion (the signal)     0.0585 deg
action spread per episode            13.099 deg
                              ratio      224 : 1
```

After mean-std normalisation, the number that matters occupies the bottom
**0.45%** of the model's output range.

### What the model actually learned

Three-way comparison on held-out episodes — model against two trivial baselines:

| predictor | error |
|---|---|
| "don't move" (repeat the current joint angles) | **0.075°** |
| **the trained pi0 model** | **1.656°** |
| "always predict the average action" | 9.775° |

Read this carefully, because it rules things out:

- The model is **6× better than predicting the average**, so it *did* learn the
  trajectory. It is not confused, not stuck, not collapsed to a mean.
- The model is **22× worse than not moving at all**. Its error is **9× larger
  than the per-step motion it is supposed to produce.**

So: right shape, useless precision. Exactly what "wrong units" produces.

### How that shows up on the robot

| what we saw | why |
|---|---|
| gate E: 1.219° step error vs 0.25° demo motion — *"NOISE DOMINATES"* | error is 9× the signal |
| the policy requested **2.44°** median joint steps — **13× the demo's 0.187°** | that is model noise, not intent |
| the 1°/step safety cap bound **65% of all commands** | inevitable at 13× demo speed |
| gripper never crossed 0.65 (max 0.154 over 450 steps) | the channel is drowned in the same noise |
| clean for ~100 steps, then wandering | error compounds from step one |

One mechanism, all five symptoms.

---

## 3. The two fixes

Both make the number the model must predict **bigger**, so it survives
normalisation. They are independent and they multiply.

### Fix 1 — predict the *change*, not the *position*

Instead of *"go to -99.26"*, the model says *"move by 0.06"*. The small number is
now the answer itself, so it cannot be rounded away.

```
action[k] = absolute_action[t+k] - state[t]      for the 6 joints
action[k] = absolute_action[t+k]                 for the gripper (a 0/1 command)
```

The gripper is deliberately left alone — it is a command, not a position.

```
ABS   action std (normaliser)  13.099 deg  ->  signal/std = 0.0045
DELTA action std (normaliser)   2.606 deg  ->  signal/std = 0.0225

                                        measured gain:  5.0x
```

Only 5×, not 200×, because the delta's spread is dominated by the *far end* of
the 50-step chunk (~2.6°), not the first step. Subtracting the state removes the
episode-to-episode position spread but not the within-chunk spread.

### Fix 2 — predict less often

Right now the chunk samples every frame, so one step is 1/30 s of motion =
0.06°. Sample every 5th frame and one step becomes 0.29°.

```
stride  1  ->  0.059 deg per step     1.0x
stride  5  ->  0.291 deg per step     5.0x
stride 10  ->  0.578 deg per step     9.9x
```

Bonus: the 50-step chunk now covers **8.3 seconds instead of 1.67**, which
independently helps on two-minute episodes.

**Measured together on real data: ~6.5× net, not 5×5=25×.** Striding widens the
chunk to 8.2 s, so the far entries move more and the normaliser std grows with
them (3.0° → 8.2° at stride 5). Signal grows 4.8×, std grows 2.7×, net 6.5×.
The two fixes help — they do not multiply cleanly.

### The catch that would have wrecked the arm

Striding changes the *time between chunk entries*. Training sees them 5 frames
apart, so at deploy each one must be **held for 5 control steps**. Skip that and
the arm replays the whole trajectory at 5× the demonstrated speed. `hold()`
handles it, wired up automatically from the checkpoint. This is not optional.

---

## 4. "But openpi trained pi0 on absolute joint angles"

Half right, and worth stating clearly because it was the sharpest objection
raised against this change.

Actually **no — openpi does not train pi0 on raw absolute joint positions
anywhere.** Verified against `src/openpi/training/config.py` on `main`:

| config | `delta_action_mask` | meaning |
|---|---|---|
| `LeRobotAlohaDataConfig` | `make_bool_mask(6, -1, 6, -1)` | both arms delta, both grippers absolute |
| `LeRobotLiberoDataConfig` | `make_bool_mask(6, -1)` | arm delta, gripper absolute |
| `RLDSDroidDataConfig` | `make_bool_mask(7, -1)` | delta, for the JOINT_POSITION action space |
| `LeRobotDROIDDataConfig` | *none* | *"We assume joint **velocity** actions, so we should not apply an additional delta transform"* |

Every config that uses joint **position** actions applies a delta mask. The one
that doesn't uses **velocity**, which is already a delta by construction.

`make_bool_mask(6, -1)` is `6 True, 1 False` — six delta dims and one absolute.
That is exactly joints-delta / gripper-absolute. And `transforms.DeltaActions` is
the same arithmetic implemented here:

```python
actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
```

So absolute joint targets were never the openpi recipe. **This repo's setup was
the outlier**, and the change in this folder moves it onto the standard path
rather than away from it.

(Demonstration speed still matters independently: ours run two minutes at
5.6 °/s, which is slow enough that even the delta targets are small. That is what
Fix 2 addresses.)

### The gripper convention (checked against openpi source, 2026-07-30)

Is 0 open and 1 close in the original pi0? **There is no single convention in the
base model** — it was pretrained on a mixture whose embodiments contradict each
other:

| embodiment | gripper channel | where verified |
|---|---|---|
| ALOHA | **0 = closed, 1 = open** | `aloha_policy.py:117-138` — monotonic chain from `POSITION_CLOSED=0.01844` → `POSITION_OPEN=0.05800`, normalized (0,1) |
| DROID | 0 = open, **1 = closed** | `droid_policy.py` passes the robot's `gripper_position` straight through (DROID convention) |
| Libero | −1 open, +1 close | no gripper transform in `libero_policy.py` — native ±1 goes in as-is |

Ours (0 = open, **1 = closed**, measured from the Slifold episodes) matches DROID
and opposes ALOHA. Not a bug: fine-tuning fully retrains the action expert, so
the channel's meaning is relearned from our data — at worst the base prior gives
the gripper no head start. Two consequences:

- What actually matters is **train/deploy consistency on our side**: dataset says
  1 = closed; deploy closes at ≥ 0.65 and opens at ≤ 0.35. (Before 2026-07 deploy
  had this inverted, and nothing could ever grasp regardless of the model.)
- Because the base prior is agnostic-to-contradictory here, the gripper earns
  everything from the fine-tune — consistent with grasp commitment being the last
  behaviour to emerge in the v2 run, and with openpi issue #912, where three
  independent platforms report arms that reach but never close. The delta
  transform deliberately leaves this channel absolute; gate criterion 2 is what
  watches it.

---

## 5. Using it

```bash
# train (defaults: delta on, stride 5)
python delta_joint/run.py train --policy pi0 --action-stride 5

# stride only, absolute targets
python delta_joint/run.py train --policy pi0 --no-delta

# deploy / eval take NO scale flags — they read the recipe from the checkpoint
python delta_joint/run.py deploy --checkpoint best.pt --task "..."
python delta_joint/run.py eval   --ckpt best.pt

# self-check (13 checks, no GPU, ~5 s)
python delta_joint/test_delta_joint.py
```

**Nothing in `common/`, `policies/` or `experiments/` is modified.** Training
swaps the dataset class; inference wraps `model.predict`, so `deploy.py`'s
existing `"joint"` branch never changed.

### How a checkpoint stays self-describing

`train.py` copies only `info["action_space"]` into the checkpoint, so the stride
rides along inside that string:

```
"joint"             stock — absolute, every frame
"delta_joint"       offsets, every frame
"delta_joint@5"     offsets, every 5th frame, held 5 control steps at deploy
```

`deploy.py` treats all of them as joint-space (its `else` branch), which is
correct once the wrappers have run. A checkpoint therefore cannot be run at the
wrong speed or with the state left off.

### Files

| file | what |
|---|---|
| `dataset_delta.py` | the dataset transform, matching stats, and the two inference wrappers |
| `run.py` | `train` / `deploy` / `eval` entry points |
| `gate.py` | offline pass/fail on eval npz — exit 0 = book robot time |
| `speedups.py` | compile / LoRA targets / schedule / warm-start / finetune modes |
| `test_delta_joint.py` | 13 checks: round-trip, stride correctness, stats-match-getitem, both wrappers, idempotence, aliasing, CLI edges |

---

## 6. What this does *not* fix

Honest limits, so nobody reads more into this than it earns.

- **Measured net gain is ~6.5×, not a silver bullet.** signal/std goes from
  0.005 to 0.032 on the real data (striding widens the normaliser too, so the
  5× and 5× do not multiply). Whether 6.5× clears the bar is exactly what the
  offline gate decides; it is not proven in advance.
- **This is untested on the robot.** Every number above is from recorded data and
  offline arithmetic. The claim "this fixes the wandering" is a hypothesis with
  good evidence, not a result.
- **The two-minute episodes are still long.** pi0 sees one frame with no history
  (`n_obs_steps=1`). Striding to an 8.3 s chunk helps; it does not eliminate the
  ambiguity of a multi-stage task seen through a single frame.
- **The slow teleop is the root cause.** The real fix is faster, shorter demos.
  This folder compensates for the data we have.

### Related findings from the same investigation, not addressed here

Separate issues, all verified, none of them the cause of the wandering:

1. **SigLIP is only one-third LoRA-adapted.** `vlm_lora_targets` uses Gemma's
   layer names; SigLIP calls them `out_proj`, `fc1`, `fc2`. The arithmetic checks
   out exactly: 18 layers × 7 (Gemma) + 27 layers × 3 (SigLIP q/k/v only) = 207,
   matching the "merged 207 LoRA pairs" in the gate log. SigLIP's attention output
   projection and its entire MLP get **zero** adaptation, on a camera rig pi0_base
   has never seen. Add `out_proj`, `fc1`, `fc2` to the target list.
2. **`deploy.py` declares `cmdt=0.033` on every `servo_j`** while actually
   delivering at 44 ms median / 150 ms at chunk boundaries (measured: 16.7 Hz
   effective, never 30). Starving the FR5's servo buffer is a plausible
   contributor to the ServoJ error 99 that ended the run.
3. **`deploy_*.npz`'s `executed` field is not what was executed** — it records the
   raw prediction, because the launcher caps joint targets *below* the level
   deploy records at. Measured `mean|executed − pred| = 0.00°` across 450 steps.
   The real commanded stream lives only in `servo.csv`.
4. **The instruction vocabulary was an episode-ID lookup key.**
   `convert_episodes.py` passes each raw episode's `language_instruction` through
   unchanged, giving 400 unique strings over 9 canonical tasks — a 1:1
   episode↔phrasing binding, so the instruction could be used to recall *which
   trajectory this was* rather than *what to do*. The v3 training patch swapped in
   the canonical string 50% of the time to weaken the binding; `tools/
   rebalance_instructions.py` fixes the data instead: it keeps a small SHARED
   vocabulary per canonical task (the canonical string plus a few of that group's
   real phrasings) and assigns it round-robin. At `--variants 5`: 400 strings → 45,
   ~8.9 episodes each. Nothing is invented — every surviving instruction is one the
   operator wrote. Language still separates the 9 tasks; it can no longer name an
   episode. Run it before converting/pushing the next dataset revision.
5. **Instruction granularity is ~60x coarser than pi0's pretraining labels.**
   The pi0 paper (arXiv 2410.24164) describes its language supervision as *"diverse
   language labels, combining task names and **segment annotations** (fine-grained
   labels for sub-trajectories, typically about **2 seconds** in length)"* — a
   bussing episode carries *"around 30 instructions in one episode"*, and a
   high-level VLM emits atomic commands like *"pick up the napkin"* / *"throw the
   napkin into the trash"*. Our label covers a whole **2-minute** episode: one
   string for ~3600 frames. So the base model's language pathway was trained to
   answer "what do I do in the next 2 s", and we ask it "what is the goal of the
   next 2 min". This is the same aliasing the single-frame observation suffers
   (§6), seen from the language side, and it is a **data-collection** fix
   (segment-level annotation), not a code fix. Style is a smaller matter but
   sourced: openpi's `PaligemmaTokenizer` does
   `prompt.strip().replace("_"," ").replace("\n"," ")` and then appends one `"\n"`
   — it does **not** lowercase (only the FAST tokenizers do), and DROID's
   crowd-sourced labels are short concrete imperatives ("Put the yellow block in
   the blue cup", 5-12 words). `tools/rebalance_instructions.py` now stores text
   already in that exact cleaned form and flags outliers.
6. **The predeploy gate's test C is mis-specified for this dataset.** It demands
   that different instructions produce different trajectories, but our dataset
   binds one unique phrasing per episode and deliberately swaps in a canonical
   string 50% of the time to *break* that binding. All nine strings describe one
   physical task, so they *should* produce the same trajectory. Gate C is valid on
   a genuinely multi-task dataset; on ours its failure is close to evidence the
   anti-shortcut mixing worked. Gates B, D and E involve no language and stand.

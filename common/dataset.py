import json
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision import transforms


VIDEO_KEY = "observation.images.wrist_cam"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── temporally-consistent transform wrapper ───────────────────────────────────

class _SeededTransform:
    """Apply a composed transform with an optional fixed random seed.

    Dominant practice (UMI, Columbia Diffusion Policy, robomimic): augmentation
    is applied INDEPENDENTLY per frame — (B, T, C, H, W) is reshaped to
    (B*T, C, H, W) and each frame gets its own random params. This is fine for
    wrist cameras where the moving viewpoint already provides natural temporal
    variation between frames.

    We expose seed-based determinism anyway because:
      1. It is useful for semantic/video-diffusion augmentation where consistency
         across the trajectory matters (e.g., RoboAug, SVD-based relighting).
      2. It enables exact reproducibility for debugging.
      3. It does no harm for standard photometric augmentation.

    When seed=None (default for n_obs_steps > 1), each frame gets its own
    random params — matching the industry standard.

    Usage:
        transform = _SeededTransform(...)
        # independent per frame (standard practice):
        aug_t0 = transform(img_t0)
        aug_t1 = transform(img_t1)
        # consistent across window (semantic augmentation):
        seed = random.randint(0, 2**31)
        aug_t0 = transform(img_t0, seed)
        aug_t1 = transform(img_t1, seed)
    """

    def __init__(self, transform):
        self._t = transform

    def __call__(self, img: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        if seed is None:
            return self._t(img)
        py_state    = random.getstate()
        torch_state = torch.get_rng_state()
        random.seed(seed)
        torch.manual_seed(seed)
        result = self._t(img)
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        return result


# ── Tier 2: background texture replacement ────────────────────────────────────

class _BackgroundAugment:
    """Replace low-texture (table/background) regions with random synthetic textures.

    Works on raw [0, 1] float tensors (C, H, W) BEFORE ImageNet normalisation.

    How it works:
      1. Estimate foreground via Laplacian edge magnitude — objects have high
         local edge density; flat table surface has low edge density.
      2. Dilate + soften the mask so object boundaries are well-covered.
      3. Blend the background region with a random procedural texture.

    Why this matters: tabletop texture change alone drops policy success from
    0.58 → 0.04 (confirmed 3-0, arXiv Nov 2025). This directly attacks that
    failure mode without needing depth data or a segmentation model.
    """

    def __init__(self, p: float = 0.8, blend: float = 0.85):
        self.p     = p
        self.blend = blend

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        _, H, W = img.shape

        gray = img.mean(0).numpy()
        lap  = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edge = np.abs(lap)
        edge = cv2.GaussianBlur(edge, (15, 15), 0)
        if edge.max() > 1e-6:
            edge /= edge.max()

        fg_mask = (edge > 0.08).astype(np.uint8)
        fg_mask = cv2.dilate(fg_mask, np.ones((13, 13), np.uint8), iterations=3)
        fg_soft = cv2.GaussianBlur(fg_mask.astype(np.float32), (21, 21), 0)
        fg      = torch.from_numpy(fg_soft).unsqueeze(0)   # (1, H, W)

        tex    = self._random_texture(H, W)
        bg_out = (self.blend * tex + (1 - self.blend) * img).clamp(0, 1)
        return (fg * img + (1 - fg) * bg_out).clamp(0, 1)

    @staticmethod
    def _random_texture(H: int, W: int) -> torch.Tensor:
        choice = random.randint(0, 4)
        if choice == 0:
            base = torch.rand(3, 1, 1).expand(3, H, W).clone()
            return (base + torch.randn(3, H, W) * 0.03).clamp(0, 1)
        elif choice == 1:
            lo, hi = sorted([random.random(), random.random()])
            g = torch.linspace(lo, hi, W).unsqueeze(0).expand(H, W)
            return (g.unsqueeze(0) * torch.rand(3, 1, 1)).clamp(0, 1)
        elif choice == 2:
            base = torch.rand(3, 1, 1).expand(3, H, W).clone()
            return (base + torch.randn(3, H, W) * 0.18).clamp(0, 1)
        elif choice == 3:
            sz   = random.randint(10, 40)
            xs   = (torch.arange(W) // sz) % 2
            ys   = (torch.arange(H) // sz) % 2
            grid = (xs.unsqueeze(0) ^ ys.unsqueeze(1)).float()
            c1, c2 = torch.rand(3, 1, 1), torch.rand(3, 1, 1)
            return (grid.unsqueeze(0) * c1 + (1 - grid.unsqueeze(0)) * c2).clamp(0, 1)
        else:
            fx = random.uniform(1, 5)
            fy = random.uniform(1, 5)
            x  = torch.linspace(0, fx * 2 * 3.14159, W)
            y  = torch.linspace(0, fy * 2 * 3.14159, H)
            gx, gy = torch.meshgrid(y, x, indexing='ij')
            n  = (torch.sin(gx + random.random() * 6) *
                  torch.cos(gy + random.random() * 6) * 0.5 + 0.5)
            return (n.unsqueeze(0) * torch.rand(3, 1, 1)).clamp(0, 1)


# ── transform factory ─────────────────────────────────────────────────────────

class ResizeWithPad:
    """Aspect-preserving resize + centred letterbox — openpi/lerobot resize_with_pad.

    Format v3: pi0_base was pretrained on letterboxed images, so squashing 640x480
    into a square is an input-format deviation. Formula-identical to lerobot's
    resize_with_pad_torch (itself an exact copy of openpi's): max-ratio, int
    truncation of the resized dims, bilinear, clamp, centred pad with the extra
    pixel bottom/right. common/deploy.py carries the same class for the robot path
    (it cannot import this module without the teleop SDK on the PYTHONPATH);
    test_pad_resize.py asserts the two stay identical.

    Black bars are applied in [0,1] space, BEFORE ImageNet normalisation.
    """

    def __init__(self, h, w):
        self.h, self.w = h, w

    def __call__(self, img):                          # (C,H,W) float in [0,1]
        import torch.nn.functional as F
        c, ih, iw = img.shape
        ratio = max(iw / self.w, ih / self.h)
        nh, nw = int(ih / ratio), int(iw / ratio)
        img = F.interpolate(img.unsqueeze(0), size=(nh, nw), mode="bilinear",
                            align_corners=False).squeeze(0).clamp(0.0, 1.0)
        out = img.new_zeros(c, self.h, self.w)
        t, l = (self.h - nh) // 2, (self.w - nw) // 2
        out[:, t:t + nh, l:l + nw] = img
        return out


def _build_transform(image_size: tuple, aug_level: str,
                     pad_resize: bool = False,
                     aug_prob: float = 1.0) -> _SeededTransform:
    """
    Returns a _SeededTransform so that multiple frames in the same observation
    window can be augmented with identical parameters (same crop position,
    same brightness, same background texture) by passing the same seed.

    aug_level:
      "none"   — deterministic resize + ImageNet norm (always used for val)
      "crops"  — random resized crop + mild brightness jitter + ImageNet norm
      "full"   — crops + background texture replacement + brightness jitter + norm

    aug_prob: fraction of samples that get the photometric jitter at all; the rest
      pass through completely clean. 1.0 reproduces openpi, whose ColorJitter is
      gated on nothing (models/model.py:181). Lower it when the deployment scene
      varies far less than the jitter range does — augmenting every sample then
      spends model capacity on variation that never occurs. Geometry (resize) is
      never gated: it is framing, not noise.

    pad_resize: letterbox instead of squashing (format v3, what pi0_base was
      pretrained on). MUST match the pad_resize flag recorded in the checkpoint,
      which is what common/deploy.py reads to pick its own transform — squashing
      here while deploy letterboxes is the train/inference mismatch that
      invalidated every pre-2026-07-28 robot trial.
    """
    h, w = image_size
    aug_prob = min(max(float(aug_prob), 0.0), 1.0)
    # wrap the photometric ops so only aug_prob of samples get them. At 1.0 this
    # collapses to a plain Compose, so the openpi-equivalent path carries no extra
    # RandomApply.
    _maybe = (lambda *t: transforms.Compose(list(t))) if aug_prob >= 1.0 else \
             (lambda *t: transforms.RandomApply(list(t), p=aug_prob))
    # swap the plain squash-resize for the letterbox at both scales the aug levels use
    _rs = (lambda th, tw: ResizeWithPad(th, tw)) if pad_resize else \
          (lambda th, tw: transforms.Resize((th, tw)))

    if aug_level == "none":
        base = transforms.Compose([
            _rs(h, w),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    elif aug_level == "photometric":
        # PHOTOMETRIC ONLY — brightness, contrast, and blur. No crop, flip,
        # rotation, saturation, hue, or geometric transform of any kind, so the
        # marker/bowl positions in pixel space are never moved (required for the
        # single-marker benchmark). Resize is deterministic (not a random crop).
        base = transforms.Compose([
            _rs(h, w),
            _maybe(transforms.ColorJitter(brightness=0.3, contrast=0.3),
                   transforms.RandomApply(
                       [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                       p=0.5)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    elif aug_level == "crops":
        # Parameters calibrated to UMI (real-stanford/universal_manipulation_interface):
        #   brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08
        # RandomGrayscale from UMI — cheap, helps with lighting-change robustness.
        # Applied independently per frame (standard practice, same as UMI/Columbia).
        base = transforms.Compose([
            _rs(int(h * 1.12), int(w * 1.12)),
            transforms.RandomCrop(image_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.4,
                                   saturation=0.5, hue=0.08),
            transforms.RandomGrayscale(p=0.05),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    elif aug_level == "full":
        # Same jitter params + background texture replacement (Tier 2).
        # Background aug runs on raw [0, 1] tensor before norm.
        base = transforms.Compose([
            _rs(int(h * 1.12), int(w * 1.12)),
            transforms.RandomCrop(image_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.4,
                                   saturation=0.5, hue=0.08),
            transforms.RandomGrayscale(p=0.05),
            _BackgroundAugment(p=0.8, blend=0.85),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    else:
        raise ValueError(f"unknown aug_level {aug_level!r}; use 'none', "
                         "'photometric', 'crops', or 'full'")

    return _SeededTransform(base)


# ── dataset ───────────────────────────────────────────────────────────────────

class FR5Dataset(Dataset):
    """PyTorch dataset for the FR5 LeRobot-format episodes.

    aug_level controls training-time visual augmentation (val always uses "none"):
      "none"  — resize + ImageNet norm only
      "crops" — random crop + mild colour jitter (Tier 1)
      "full"  — crops + jitter + background texture replacement (Tier 1 + Tier 2)

    n_obs_steps: number of consecutive observation frames to return.
      = 1 (default) : observation.image is (C, H, W)
      > 1           : observation.image is (n_obs_steps, C, H, W), all frames
                      augmented with IDENTICAL random parameters (temporally
                      consistent — same crop position, same brightness shift).
    """

    def __init__(self, root, chunk_size=100, use_image=True,
                 image_size=(224, 224), episode_indices=None,
                 aug_level="none", n_obs_steps=1, frame_stride=1,
                 pad_resize=False, aug_prob=1.0):
        self.root        = Path(root)
        self.chunk_size  = chunk_size
        self.use_image   = use_image
        self.image_size  = image_size
        self.aug_level   = aug_level
        self.pad_resize  = bool(pad_resize)
        self.aug_prob    = float(aug_prob)
        self.n_obs_steps = n_obs_steps
        # take every Kth frame as a training START sample (the 30 Hz capture is
        # heavily oversampled; chunks are still predicted at the full rate). Must
        # match convert_episodes --extract-stride so the start-frame JPEGs exist.
        self.frame_stride = max(1, int(frame_stride))

        with open(self.root / "meta/info.json") as f:
            self.info = json.load(f)

        # camera keys present in the dataset (one or many). Derived from the
        # feature list so adding the scene cam needs no code change here.
        self.camera_keys = [k for k in self.info.get("features", {})
                            if k.startswith("observation.images.")]
        if not self.camera_keys:
            self.camera_keys = [VIDEO_KEY]

        self.df = pq.read_table(
            self.root / "data/chunk-000/file-000.parquet"
        ).to_pandas()

        self.episodes = pq.read_table(
            self.root / "meta/episodes/chunk-000/file-000.parquet"
        ).to_pandas()

        if episode_indices is not None:
            self.episodes = self.episodes[
                self.episodes["episode_index"].isin(episode_indices)
            ].reset_index(drop=True)

        self._samples = self._build_index()

        tasks_path = self.root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            tasks_df       = pq.read_table(tasks_path).to_pandas()
            self._task_map = dict(zip(tasks_df["task_index"].tolist(),
                                      tasks_df["task"].tolist()))
        else:
            self._task_map = {}

        self._img_transform = _build_transform(image_size, aug_level,
                                               self.pad_resize, self.aug_prob)

    def _build_index(self):
        samples = []
        for _, ep in self.episodes.iterrows():
            ep_idx = int(ep["episode_index"])
            from_i = int(ep["dataset_from_index"])
            to_i   = int(ep["dataset_to_index"])
            for t in range(from_i, to_i, self.frame_stride):
                samples.append((ep_idx, t, from_i, to_i))
        return samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        ep_idx, frame_abs, ep_from, ep_to = self._samples[idx]

        row   = self.df.iloc[frame_abs]
        state = torch.tensor(row["observation.state"], dtype=torch.float32)

        chunk_end  = min(frame_abs + self.chunk_size, ep_to)
        chunk_rows = self.df.iloc[frame_abs:chunk_end]
        actions    = torch.tensor(
            np.array(chunk_rows["action"].tolist()), dtype=torch.float32
        )

        pad_len = self.chunk_size - len(actions)
        is_pad  = torch.zeros(self.chunk_size, dtype=torch.bool)
        if pad_len > 0:
            padding = actions[-1:].expand(pad_len, -1)
            actions = torch.cat([actions, padding], dim=0)
            is_pad[self.chunk_size - pad_len:] = True

        task_idx = int(row["task_index"]) if "task_index" in row.index else 0
        sample = {
            "observation.state": state,
            "action":            actions,
            "action_is_pad":     is_pad,
            "task":              self._task_map.get(task_idx, ""),
        }

        if self.use_image:
            # one seed per sample — all frames in the obs window share it so
            # crop position, brightness, and background texture are identical
            # across the temporal sequence (temporally-consistent augmentation).
            aug_seed = random.randint(0, 2**31) if self.aug_level != "none" else None

            if self.n_obs_steps == 1:
                frame_in_ep = int(row["frame_index"])
                # load EVERY camera under its own dataset key (wrist + scene).
                # Each camera gets an independent augmentation draw.
                for cam_key in self.camera_keys:
                    cam_seed = (random.randint(0, 2**31)
                                if self.aug_level != "none" else None)
                    img = self._load_frame(cam_key, ep_idx, frame_in_ep, cam_seed)
                    if img is None:
                        print(f"[dataset] WARN: missing {cam_key} ep{ep_idx} "
                              f"idx{frame_in_ep}, using zeros")
                        img = torch.zeros(3, *self.image_size)
                    sample[cam_key] = img
            else:
                # Industry standard (UMI, Columbia DP, robomimic): independent
                # augmentation per frame. Each frame gets its own seed so crop
                # position, jitter, and background vary across the temporal window.
                # Pass the same seed to _load_frame only if you want temporally
                # consistent semantic augmentation (e.g., SVD-based relighting).
                # multi-step history (single-camera path; ACT uses n_obs_steps=1)
                cam_key = self.camera_keys[0]
                frames = []
                for step in range(self.n_obs_steps - 1, -1, -1):
                    hist_abs  = max(frame_abs - step, ep_from)
                    hist_row  = self.df.iloc[hist_abs]
                    frame_idx = int(hist_row["frame_index"])
                    frame_seed = (random.randint(0, 2**31)
                                  if self.aug_level != "none" else None)
                    img = self._load_frame(cam_key, ep_idx, frame_idx, frame_seed)
                    frames.append(img if img is not None else torch.zeros(
                        3, *self.image_size))
                # (n_obs_steps, C, H, W)
                sample[cam_key] = torch.stack(frames, dim=0)

        return sample

    def _load_frame(self, cam_key: str, ep_idx: int, frame_idx: int,
                    aug_seed: int | None = None) -> torch.Tensor | None:
        jpg = (self.root / "frames" / cam_key /
               f"ep-{ep_idx:03d}" / f"{frame_idx:06d}.jpg")
        frame = cv2.imread(str(jpg)) if jpg.exists() else \
                self._read_video_frame(cam_key, ep_idx, frame_idx)

        if frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return self._img_transform(img, aug_seed)   # seed ensures temporal consistency

    def _read_video_frame(self, cam_key: str, ep_idx: int,
                          frame_idx: int) -> np.ndarray | None:
        vid = self.root / "videos" / cam_key / "chunk-000" / f"file-{ep_idx:03d}.mp4"
        if not vid.exists():
            return None
        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def get_stats(self):
        ep_mask = self.df["episode_index"].isin(
            self.episodes["episode_index"].tolist()
        )
        sub    = self.df[ep_mask]
        state  = np.array(sub["observation.state"].tolist())
        action = np.array(sub["action"].tolist())
        return {
            "state_mean":  state.mean(0).astype(np.float32),
            "state_std":   state.std(0).clip(1e-6).astype(np.float32),
            "state_min":   state.min(0).astype(np.float32),
            "state_max":   state.max(0).astype(np.float32),
            "action_mean": action.mean(0).astype(np.float32),
            "action_std":  action.std(0).clip(1e-6).astype(np.float32),
            "action_min":  action.min(0).astype(np.float32),
            "action_max":  action.max(0).astype(np.float32),
        }

    @staticmethod
    def episode_split(n_episodes, val_frac=0.1, seed=42):
        indices = list(range(n_episodes))
        random.seed(seed)
        random.shuffle(indices)
        n_val = max(1, int(val_frac * n_episodes))
        return indices[n_val:], indices[:n_val]

# Modded-NanoGPT-JAX

A pure-JAX port of [Keller Jordan's modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun, targeting Google Cloud TPUs.

> **Based on prior work.** All credit for the speedrun benchmark, the model architecture choices, the Muon optimizer, and the training recipe goes to **Keller Jordan and the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) contributors**, building on **Andrej Karpathy's [NanoGPT](https://github.com/karpathy/nanogpt)**. The JAX port and the original blog write-up are by **Nor** ([blog post](https://nor-blog.pages.dev/posts/2025-08-21-modded-nanogpt-jax/)). This repository extends that work with multi-host TPU support and a detailed debugging guide.

## What this is

A community **speedrun competition** for training a small GPT-2-style language model (~125M parameters, 12 layers) with modern tweaks: the Muon optimizer for matrix parameters, RoPE, QK-norm, U-Net-style skip connections between encoder and decoder halves, logit softcap, and sequence-length warmup from 1024 → 2048 tokens.

### The rules

The competition has **two tracks**, both targeting the same validation loss but optimizing different things.

#### Main track — minimize wall-clock time

> **Reach mean validation loss ≤ 3.28 on FineWeb10B in the shortest possible wall-clock time.**

- **Measured in wall-clock minutes**, not training steps. Optimizing for fewer steps doesn't help if your steps are slower.
- **Records are NOT normalized across hardware.** To beat a record, you must run on the same machine as the prior record-holder. There is no cross-hardware leaderboard.
- The original main-track runs on **8× NVIDIA H100 GPUs**; the current upstream record is ~1.4 minutes.
- This JAX port targets **8-chip Cloud TPUs** instead — the H100 wall-clock records don't apply.

#### Optimization track — minimize steps

From the upstream repo:

> *"Besides the main track, there is also an optimization track where we try to minimize steps subject to fixed arch/data/bsz and with unlimited wallclock budget."*

- **Minimizes the number of training steps** to reach val_loss ≤ 3.28.
- **Architecture, data, and batch size are fixed** — you can only change the optimizer and training schedule (LR schedule, momentum, etc.).
- **Unlimited wall-clock budget** — slow steps are fine if there are fewer of them.
- **Hardware-independent** — since you're counting steps with a fixed batch size, the score is comparable across machines.

This repo's `train.py` is set up for the main track on TPU, not the optimization track. Either could be pursued by tweaking the script.

### Reference numbers (TPU)

| Hardware | Hosts | Wall clock | val_loss | Notes |
|---|---|---|---|---|
| TPU v6e-8 (single-host) | 1 | ~10 min | ≤ 3.28 | The blog's reference run; what `train.py` was originally tuned for |
| TPU v5p-16 (multi-host) | 2 | ~13 min | 3.276 | This work; 8 chips total split across 2 hosts |

These two numbers are **not directly comparable** in the speedrun sense — different chip generations, plus the v5p-16 run pays multi-host coordination overhead the v6e-8 run does not. They're listed as data points, not leaderboard entries.

For the original write-up on the porting process, see the blog post: [The modded nanogpt speedrun, but in JAX and on TPUs](https://nor-blog.pages.dev/posts/2025-08-21-modded-nanogpt-jax/).

## How to run (single-host: TPU v6e-8)

### 1. Setup

```bash
git clone https://github.com/<your-fork>/modded-nanogpt-jax.git
cd modded-nanogpt-jax

pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install numpy huggingface_hub einops
```

### 2. Download data

```bash
python cached_fineweb10B.py 10        # ~2 GB, sufficient for the default 1675-step run
# or for a quick smoke test:
python cached_fineweb10B.py 1
```

Data lands in `fineweb10B/` next to the script.

### 3. Train

```bash
python train.py
```

AOT compilation runs first (1-3 minutes per unique input shape, cached to `/tmp/jax_cache`), then the training loop. Logs, metrics, and a final checkpoint are written under `logs/<run_id>/`.

## How to run (multi-host: TPU v5p-16, v5p-64, v6e-64, etc.)

**Multi-host requires a different setup than the single-host instructions above.** The upstream `train.py` was originally written for v6e-8 (single host) and several issues need to be addressed: the data loader, sharding of model state, the compile path, and the libtpu version. The fixes are already in this repo's `train.py`.

**Required: pin the JAX version.** The default `jax[tpu]` resolves to a libtpu version with a deterministic segfault on multi-host. Use:

```
--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.5.3
numpy
einops
huggingface_hub
wandb
```

For the full multi-host recipe, including [jobman](https://github.com/Zephyr271828/jobman) orchestration, the GCS-bucket pattern for data and code, the multi-host JAX fundamentals, and a complete diagnostic playbook, see:

📖 **[`MULTIHOST_TPU_GUIDE.md`](MULTIHOST_TPU_GUIDE.md)** — extensively detailed lab notebook covering:

- Final working setup recipe
- Multi-host JAX fundamentals (process/device distinction, sharding, compile determinism)
- Full debugging journey with every dead end documented
- Things that did not work and why (table of 14 attempts)
- The fix that actually worked
- Operational gotchas with jobman + existing TPUs
- Diagnostic commands and a minimal multi-host sanity test
- Hard rules for working with TPUs without breaking them

If you've never set up multi-host JAX training before, read that guide before doing anything.

## Logging

`train.py` writes to `logs/<run_id>/` and `logs/<run_id>.txt` on the master process (`jax.process_index() == 0`). Per-step metrics include: `step`, `loss`, `step_secs`, `tokens_per_sec`, `seq_len`, `batch_size`. Validation runs every 125 steps and adds a `val_loss` line.

If `WANDB_API_KEY` is set in the environment, the script also logs to Weights & Biases. Configure the project/run with `WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_RUN_ID` env vars.

## Code overview

- **`train.py`** — the entire training script:
  - Model architecture (GPT-2 with modern enhancements)
  - Data loading with sequence-length warmup, multi-host-aware via `jax.make_array_from_process_local_data`
  - Multi-optimizer (Adam variants + Muon) with per-parameter routing
  - Training and evaluation loops
  - Logger with file + wandb integration
  - Background heartbeat thread (keeps SSH alive during silent compiles)
- **`cached_fineweb10B.py`** — utility to download pre-tokenized FineWeb10B shards from the Hugging Face Hub (`kjj0/fineweb10B-gpt2`)
- **`MULTIHOST_TPU_GUIDE.md`** — comprehensive guide for multi-host TPU runs (read before running on multi-host)
- **`records/sub10m.txt`** — log from a successful single-host v6e-8 run

## Acknowledgments

A direct port and adaptation of the work done by the open-source community on the original NanoGPT speedrun.

- **Andrej Karpathy** for [NanoGPT](https://github.com/karpathy/nanogpt), which started it all
- **Keller Jordan and all contributors** to the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repository for pioneering architecture and optimizer improvements
- **Nor (the original blog author)** for the JAX port and detailed write-up

# Modded-NanoGPT-JAX

A pure-JAX port of [Keller Jordan's modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun, optimized for Google TPUs. Trains a GPT-2-style language model (12 layers, ~125M parameters) on the FineWeb10B dataset to a held-out validation loss of **≤ 3.28** in the shortest wall-clock time possible.

The training script is self-contained in `train.py` and is written using core JAX APIs — no Flax, Optax, or Orbax — in the spirit of the original speedrun.

For the original write-up on the porting process, optimizations, and performance analysis, see the blog post: [The modded nanogpt speedrun, but in JAX and on TPUs](https://nor-blog.pages.dev/posts/2025-08-21-modded-nanogpt-jax/).

## What this is

A speedrun benchmark for training technique. The model itself is a small GPT-2-style transformer with several modern tweaks:

- **Muon optimizer** for matrix-valued parameters, **Adam** for embeddings, LM head, and scalar parameters
- Rotary position embeddings (RoPE)
- QK-norm
- U-Net-style skip connections between encoder and decoder halves
- Logit softcap
- Sequence-length warmup from 1024 → 2048 tokens over training
- Ahead-of-time XLA compilation for each unique input shape
- Pure data-parallel sharding via `jax.NamedSharding`

Success is binary: hit val_loss ≤ 3.28 by the end of the run. The exercise is in the optimization technique (architecture, optimizer choice, sharding, compilation strategy), not raw model quality.

## Performance

| Hardware | Hosts | Wall clock | val_loss | Tokens/sec |
|---|---|---|---|---|
| TPU v6e-8 (single host)  | 1 | ~10 min | ≤ 3.28 | ~1.4M |
| TPU v5p-16 (multi-host)  | 2 | ~13 min | 3.276  | ~1.18M |

Default config: 1675 training steps, batch 512 at seq_len 1024 / batch 256 at seq_len 2048, 16 micro-batches, ~880M training tokens consumed.

## How to run (single-host: v6e-8)

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

AOT compilation runs first (1-3 minutes per unique shape, cached to `/tmp/jax_cache`), then the training loop. Logs, metrics, and a final checkpoint are written under `logs/<run_id>/`.

## How to run (multi-host: v5p-16, v5p-64, v6e-64, etc.)

**Multi-host requires a different setup than the single-host instructions above.** The upstream `train.py` was originally written for v6e-8 and several issues need to be addressed (data loader, sharding, compile path, libtpu version). The fixes are already in this repo's `train.py`.

**Required: pin the JAX version.** The default `jax[tpu]` resolves to a libtpu version with a deterministic segfault on multi-host. Use:

```
--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.5.3
numpy
einops
huggingface_hub
wandb
```

For the full multi-host recipe, including jobman orchestration, the GCS-bucket pattern for data and code, the multi-host JAX fundamentals you need to know, and a complete diagnostic playbook, see:

📖 **[`MULTIHOST_TPU_GUIDE.md`](MULTIHOST_TPU_GUIDE.md)** — extensively detailed lab notebook covering:

- The final working setup recipe
- Multi-host JAX fundamentals (process/device distinction, sharding, compile determinism)
- The full debugging journey with every dead end documented
- Things that did not work and why (table of 14 attempts)
- The fix that actually worked
- Operational gotchas with [jobman](https://github.com/Zephyr271828/jobman) + existing TPUs
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

A direct port and adaptation of the incredible work done by the open-source community on the original NanoGPT speedrun.

- **Andrej Karpathy** for [NanoGPT](https://github.com/karpathy/nanogpt), which started it all
- **Keller Jordan and all contributors** to the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repository for pioneering architecture and optimizer improvements
- **Nor (the original blog author)** for the JAX port and detailed write-up

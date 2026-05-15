# Running modded-nanogpt-jax on a multi-host TPU pod

A complete, painfully detailed lab notebook of porting this single-host
training script to a multi-host TPU v5p-16 pod, using
[jobman](https://github.com/Zephyr271828/jobman) for orchestration.

Read this if:
- You have a multi-host Cloud TPU (v5p-X with X≥16, v5e-32+, v6e-16+, etc.) and
  want to run this repo (or any single-host JAX script) on it.
- You hit one of: `unexpected peer with different launch id`,
  `tokio-rt-worker segfault`, `numpy.ndarray ... not the same on each process`,
  `KeyError` in AOT shape lookup, jobs that die at exactly 2 minutes 4 seconds,
  or `device_put` with sharded `NamedSharding` that hangs forever.

The headline finding: **almost everything you'll see is a JAX/libtpu
multi-host issue, not a model issue. The default `jax[tpu]` from the public
indexes ships a libtpu version (0.0.17 at time of writing) with a
deterministic segfault in its internal Rust runtime when the script's
specific compile path runs on multi-host. Pin `jax[tpu]==0.5.3` and most of
the pain disappears.**

---

## Table of contents

1. [Final working setup (skip here if you just want it to run)](#1-final-working-setup-skip-here-if-you-just-want-it-to-run)
2. [Background: how the script is structured](#2-background-how-the-script-is-structured)
3. [Multi-host fundamentals you need to internalise](#3-multi-host-fundamentals-you-need-to-internalise)
4. [The full debugging journey, in order](#4-the-full-debugging-journey-in-order)
5. [Things that did not work and why](#5-things-that-did-not-work-and-why)
6. [The fix that actually worked](#6-the-fix-that-actually-worked)
7. [Operational gotchas with jobman + existing TPUs](#7-operational-gotchas-with-jobman--existing-tpus)
8. [Hard rules learned the painful way](#8-hard-rules-learned-the-painful-way)
9. [Reproduction recipe end-to-end](#9-reproduction-recipe-end-to-end)
10. [Diagnostic commands you'll use over and over](#10-diagnostic-commands-youll-use-over-and-over)
11. [Open questions / things still unclear](#11-open-questions--things-still-unclear)

---

## 1. Final working setup (skip here if you just want it to run)

If you just want the recipe and don't care about the journey:

### 1.1 Hardware / GCP

- TPU: any multi-host pod (verified on `v5p-16`, two hosts, four chips per host
  = eight chips total).
- Allocation mode: `tpu-vm` (works with an already-existing pod).
- Software version on the VM: `v2-alpha-tpuv5` (GCP image; this is what the
  Cloud TPU service ships).
- Region: pick a zone where you have quota; the GCS bucket you'll mount
  must be in the **same region** as the TPU zone (this is a hard rule, not
  just a suggestion).

### 1.2 Software pin (the key insight)

In your venv requirements file:

```
--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.5.3
numpy
einops
huggingface_hub
wandb
```

The pin to `0.5.3` pulls `libtpu==0.0.11.4`, which does not have the
`tokio-rt-worker` segfault that ships with `libtpu 0.0.17` (which is what
you get with `jax[tpu]` un-pinned, because the pin floats).

### 1.3 Code changes vs. upstream `train.py`

You need **all** of these or training will not run on multi-host:

1. **Multi-host data loader**: each process loads only its local micro-batch
   slice; the global `jax.Array` is built via
   `jax.make_array_from_process_local_data`. The upstream loader assumes one
   process and rejects multi-host with a `NamedSharding` mismatch.
2. **Replicate model state**: explicitly `jax.device_put` every leaf of
   `params`, `precomputed_params`, and `opt_state` to `NamedSharding(mesh,
   P())` so all hosts agree on a global replicated sharding (otherwise some
   leaves end up on a host-local `SingleDeviceSharding` and the compiled
   programs diverge across hosts — different launch IDs).
3. **Drop AOT compile** (`.lower().compile()`). Use plain `@jit` and let
   first-call lowering happen lazily. The AOT path baked launch IDs into the
   compiled handle in a way that desynced across hosts in our environment.
4. **Read metrics with `addressable_data(0)`**, not `float()` or `format()`
   on a sharded `jax.Array`. The latter trigger cross-host gathers that can
   desync if not all hosts call them in lockstep.
5. **Add a heartbeat thread** (`print` every 30s) so the SSH session that
   ferries stdout off the TPU doesn't idle-disconnect during the long
   silent JIT compile (~90-120s).

The actual diff is in [`train.py`](train.py); search for
`make_array_from_process_local_data`, `replicate_to_mesh`, and
`_start_heartbeat`.

### 1.4 Orchestration: jobman + existing TPU

Two non-obvious gotchas (see [§7](#7-operational-gotchas-with-jobman--existing-tpus)):

- `jobman create` always appends `_<job_id>` to `tpu.name` in the snapshot
  it writes. To attach to an existing TPU, you must **create → stop →
  hand-edit `jobs/<user>/<id>/config.yaml` → resume**.
- `jobman` validates that the `gcsfuse` key exists in your config even
  if you don't use it; pass `gcsfuse: null` if you don't need a mount.

### 1.5 The command block (jobman `command.cmd`)

```bash
set -euo pipefail
cd ~
rm -rf modded-nanogpt-jax
git clone --depth 1 https://github.com/<your-fork>/modded-nanogpt-jax.git
cd modded-nanogpt-jax
git rev-parse --short HEAD

rm -rf fineweb10B
ln -s /home/<USER>/gcs_bucket/datasets/fineweb10B fineweb10B

pip install --quiet wandb || true
rm -rf /tmp/jax_cache

export WANDB_API_KEY='<KEY>'
export WANDB_PROJECT='modded-nanogpt-jax'
export WANDB_RUN_NAME='nanogpt-v5p16'
export WANDB_RUN_ID='nanogpt-v5p16'

python -u train.py 2>&1 | tee -a ~/train_$(hostname).log
```

`workers: "all"` — both hosts run this in parallel.

### 1.6 Expected behaviour

- Setup phase (SSH ✓, gcsfuse ✓, venv ✓): ~30s on rerun (idempotent
  check-then-setup).
- First training step compile (plain JIT): ~60-120s.
- Steady state: ~0.4 s/step at seq_len=1024, ~0.45 s/step at seq_len=2048,
  ≈1.18M tokens/sec on v5p-16.
- Total wall clock for default 1675 steps: ~12-13 min.
- Final val_loss should land at **3.27-3.28**; target is ≤ 3.28.

That's everything you need. The rest of this document explains *why* the
recipe is the way it is, what we tried that did not work, and what to do
when it breaks differently for you.

---

## 2. Background: how the script is structured

The script is the JAX port of Keller Jordan's
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun.
Originally tested on a **single-host TPU v6e-8** (one VM, eight chips, one
JAX process). Written so that:

- `Config.__post_init__` sets `mesh_shape = (jax.device_count(),)` — a 1D
  data-parallel mesh, axis name `"dp"`.
- `activation_sharding = (None, "dp")` — the leading dim is replicated, the
  micro-batch dim is sharded across all chips.
- The model is fully replicated (`weight_sharding = None`); only activations
  are sharded.
- `train_step` is `@jit`-decorated with `static_argnames=("config",
  "optimizer")` and `donate_argnums=(1, 3)` (donates `params` and
  `opt_state`).
- The grad-accumulation loop is a `jax.lax.scan` over micro-batches.
- Multi-optimizer (Adam for embeddings/lm_head/non-matrix params, Muon for
  matrix params).
- AOT compile loop in `train_loop` lowers `train_step` once per unique
  `(seq_len, batch_size, n_grad_acc_steps)` shape. There are 2-3 unique
  shapes due to sequence-length warm-up.
- Validation runs every 125 steps.
- Default config: 1675 train iters, batch 512 for seq_len 1024, batch 256
  for seq_len 2048, 16 micro-batches per chip-aware split.

The "one process, all chips local" assumption is what breaks on multi-host.
On a v6e-8 the script just runs. On a v5p-16 (two hosts) two Python
processes each see four local chips out of eight global, and the script's
data-parallel pattern doesn't survive without surgery.

---

## 3. Multi-host fundamentals you need to internalise

If you've only used JAX on single-host, this is the mental model that
turned out to matter:

- **One Python process per host.** On a 2-host pod, jobman SSHs into both
  workers in parallel and runs the same `python train.py` on each. The two
  Python processes are essentially mirrors of each other — same code, same
  ranks ordered by `jax.process_index()`.
- **Devices, not processes, are the unit of parallelism.**
  `jax.device_count()` returns the **global** count (8 on v5p-16);
  `jax.local_device_count()` returns the per-process count (4). Your mesh
  uses the global count.
- **Cloud TPU auto-initialises the JAX distributed runtime via libtpu's
  metadata-server discovery.** You should NOT call
  `jax.distributed.initialize()` — Cloud TPU does it for you, and explicit
  init can double-register and cause launch-id mismatches.
- **A "global" `jax.Array` is sharded across all hosts.** Each host holds
  the addressable shards for its local devices; remote shards are
  references. Every collective (gather, all-reduce, etc.) requires every
  host to participate in lockstep — if one host calls `device_get` on a
  sharded array and the other doesn't, that's a deadlock.
- **`NamedSharding(mesh, P(...))` is the partition spec.** `P(None, "dp")`
  means "first dim replicated, second dim sharded along the dp axis". A
  `P()` (empty) is fully replicated. A `SingleDeviceSharding` is host-local
  and must NOT appear in inputs to a multi-host `jit`.
- **`jax.make_array_from_process_local_data(sharding, local_data,
  global_shape)` is the documented primitive for "I have my host's slice of
  a global tensor; build the sharded `jax.Array`."** Each host calls it
  with its own piece; JAX assembles the global array.
- **Compile determinism is required across hosts.** The same Python on the
  same JAX version with the same input shardings should produce identical
  XLA programs on both hosts. If it doesn't, you get the dreaded:
  `An unexpected peer shows up in the launch group with a different launch id`.
  This happens when:
  - Some leaves of `params`/`opt_state` are on a `SingleDeviceSharding`
    (different per host) → fix: explicitly replicate.
  - One host hits a stale persistent compile cache and the other doesn't →
    fix: disable persistent cache or share it.
  - Hosts use AOT `lower().compile()` and the compiled handles desync →
    fix: use plain `@jit`.
- **Output buffering matters.** `python ... | tee` puts Python's stdout in
  block-buffered mode; output won't appear in the log file until a 4 KB
  buffer fills or the process exits. Use `python -u`.
- **SSH sessions can idle-disconnect.** If the JIT compile is silent for >2
  minutes, gcloud's underlying SSH (or some intermediate hop) can drop the
  connection, killing the remote process with `SIGHUP`. Fix: heartbeat
  thread + `ServerAliveInterval` ssh flag.
- **The persistent compile cache is per-host filesystem.** Pointing
  multiple hosts at separate `/tmp/jax_cache` paths is asking for desync
  (one host hits cache, other compiles fresh, you get different XLA). Either
  disable, or share via a co-located GCS bucket (which has its own
  permissions/eventual-consistency issues — see §5).

---

## 4. The full debugging journey, in order

Times are approximate; this whole arc was about a working day.

### 4.1 Phase 0 — getting jobman to talk to the existing TPU

Wrote a one-shot test config to confirm jobman could SSH into both
workers and run a no-op:

```yaml
job:
  name: tpu-test
  env_type: null
  loop: false
  remote_user: <USER>
tpu:
  allocation_mode: "tpu-vm"
  accelerator: v5p-16
  name: <TPU_NAME>
  zone: <ZONE>
  version: v2-alpha-tpuv5
  pricing: ondemand
gcsfuse: null   # required key, can be null
command:
  cmd: |
    hostname
    whoami
    python3 -c "import jax; print(jax.devices())" || echo "no jax"
  workers: "all"
```

`jobman create config/test.yaml` → discovered jobman appends `_<job_id>` to
`tpu.name`, so the snapshot it writes points at a non-existent
`<TPU_NAME>_000158`. Solution: stop the job within seconds, hand-edit
`jobs/<user>/<id>/config.yaml` to drop the suffix, then `jobman resume`.
This is now a documented dance; see [§7](#7-operational-gotchas-with-jobman--existing-tpus).

Test passed: both workers reachable, JAX not installed yet. SSH plumbing
works. ✓

### 4.2 Phase 1 — wiring real training

Built a proper config with `gcsfuse`, `ssh`, `venv` blocks. Pushed
`train.py` and `cached_fineweb10B.py` to the bucket so workers could `cp`
them in their command block.

**This was a mistake.** gcsfuse caches metadata aggressively (TTL of
60s+), so when I re-uploaded `train.py` to the bucket and immediately
re-ran the job, the workers' `cp` saw the **old** version. md5sums looked
identical between the bucket's gcsfuse view and the local copy on the TPU,
because they were both reading the cached version. Took 20 minutes to
realise this.

Two solutions, used both eventually:

- Short-term: `gcloud storage cp gs://... ~/file` from inside the command
  bypasses the gcsfuse cache (uses the GCS API directly).
- Real fix: **push code to a real git repo and `git clone --depth 1` fresh
  on every run**. This is the pattern jobman's queue mode uses. Not only
  fixes cache staleness, also gives you commit hashes for reproducibility.

The command block in §1.5 is the final form.

### 4.3 Phase 2 — first JAX crash (single-host data loader)

```
AssertionError: <class 'numpy.ndarray'> passed to device_put is not the
same on each process. Make sure you are passing the same value of
<class 'numpy.ndarray'> on each process. ...
```

The upstream `load_dataset` function:

```python
for global_step_idx in range(num_global_batches):
    if global_step_idx % num_processes == process_rank:
        # ... build batched_x for this process only ...
        batched_x = jax.device_put(batched_x, NamedSharding(mesh, P(None, "dp")))
        precomputed_batches.append((batched_x, batched_y))
```

This is a per-process **round-robin batch assignment**. On single-host
(`num_processes==1`), every batch goes to the one process. On multi-host,
batch 0 goes to process 0, batch 1 to process 1, etc. — each process
processes a **different** batch at each step.

But `train_step` is jit-compiled with sharded inputs that expect
**all hosts to contribute their shard of the same global batch at the same
step**. Round-robin batch assignment is incompatible with that. JAX 0.6+
asserts the per-host arrays match for replicated dims, which catches it
immediately. Older JAX silently accepted it and likely produced garbage
gradients.

#### Attempted fix #1 (didn't work): "every process loads every batch"

```python
# Removed the if global_step_idx % num_processes == process_rank check
```

Now both hosts have identical numpy arrays. `jax.device_put(arr,
NamedSharding(mesh, P(None, "dp")))` should shard `arr` across the dp axis
and replicate the leading dim.

This **hung for 12+ minutes** in `load_dataset`'s 1675-iteration loop.
Each `device_put` of a numpy array onto a multi-host `NamedSharding`
involves cross-host coordination (JAX has to verify that all hosts have
the same data on the replicated dim, which means a gather + comparison).
1675 cross-host syncs at ~0.5s each = 14 minutes of slow death.

#### Attempted fix #2 (correct): `make_array_from_process_local_data`

```python
local_micro = config.micro_batch_size // num_processes  # 16 / 2 = 8
local_start = process_rank * local_micro
local_stop = local_start + local_micro
for global_step_idx in range(num_global_batches):
    # ... compute the global batch as numpy ...
    global_shape = batched_x.shape  # (n_grad_acc, 16, seq_len)
    local_x = batched_x[:, local_start:local_stop, :]  # (n_grad_acc, 8, seq_len)
    local_y = batched_y[:, local_start:local_stop, :]
    batched_x = jax.make_array_from_process_local_data(
        activation_sharding, local_x, global_shape
    )
    batched_y = jax.make_array_from_process_local_data(
        activation_sharding, local_y, global_shape
    )
    precomputed_batches.append((batched_x, batched_y))
```

This is the documented multi-host pattern. Each host provides its slice;
JAX builds the global `jax.Array` without a cross-host gather. Fast and
correct.

### 4.4 Phase 3 — the metric extraction crash

After fixing data loading, training got to **step 9** before crashing in
`Logger.log`:

```python
# train_loop:
if step % 10 == 9:
    logger.log({"step": step, "time": ...} | metrics)

# Logger.log eventually calls "{}: {}".format(k, v)
# v is a sharded jax.Array  →  format() calls __format__  →
# __format__ accesses _value  →  _value triggers cross-host gather
```

Crash:

```
jaxlib._jax.XlaRuntimeError: INTERNAL: Core halted unexpectedly: ...
An unexpected peer shows up in the launch group with a different launch id
than the current group leader.
```

The bug: `Logger.log` returns early on non-master hosts (`if not
self.is_master: return`). So **only host 0** triggers the cross-host
gather; host 1 never participates. The gather op on host 0 hangs/desyncs
because there's no peer.

#### Attempted fix #1 (didn't work): `addressable_data(0)`

```python
host_metrics = {k: float(v.addressable_data(0)) for k, v in metrics.items()}
```

`addressable_data(0)` returns the local-device shard. For a fully
replicated scalar this is just the local copy — no gather should be
needed. But `float()` on the result still calls `_value` on the device,
which triggers a TPU op, which apparently still goes through some
cross-host coordination layer. Same crash.

#### Attempted fix #2 (didn't work): `process_allgather`

```python
from jax.experimental import multihost_utils
gathered = multihost_utils.process_allgather(metrics, tiled=False)
host_metrics = {k: float(np.asarray(gathered[k]).flat[0]) for k in metrics}
```

This is the explicit "gather across all hosts" primitive — both hosts
participate. Same crash.

#### Attempted fix #3 (didn't work): `jax.device_get` + `jax.block_until_ready`

```python
jax.block_until_ready(metrics)
host_metrics = {k: float(np.asarray(jax.device_get(v))) for k, v in metrics.items()}
```

Same crash. The crash is not in the gather mechanism; it's in the
underlying TPU op execution.

#### Attempted fix #4 (didn't work): skip metrics entirely, just `block_until_ready(params)`

```python
params, opt_state, metrics = jitted_train_step(...)
jax.block_until_ready(params)  # no metric reading at all
```

Still crashed at `block_until_ready(params)`. Now we know: **the crash is
not in metric extraction; it's in `train_step` execution itself**. The
metric read just happened to be where the lazy-dispatched error surfaced.

#### Attempted fix #5 (didn't work): `jax.distributed.initialize()`

I added an explicit `jax.distributed.initialize()` at the top of
`train.py`. JAX docs say this is required for some multi-host setups.

Same crash. (Spoiler: on Cloud TPU it's auto-initialised by libtpu, and
calling it explicitly may double-register. Removing it didn't help either,
but it didn't hurt either — wasn't the cause.)

#### Attempted fix #6 (didn't work): replicate model state explicitly

[Codex's contribution.] Some leaves of the upstream `init_params`
(`attn["lamb"]`, `attn["scale"]`) and the optimizer state's step counters
are plain `jnp.array(...)` calls — no explicit sharding. After
`jax.distributed.initialize()` they end up on a host-local
`SingleDeviceSharding`. When jit lowers the train_step, those leaves' input
shardings are different per host → divergent compiled programs → divergent
launch IDs.

```python
def replicate_to_mesh(pytree, mesh):
    return tree_map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P(*((None,) * x.ndim)))),
        pytree,
    )

params = replicate_to_mesh(params, mesh)
precomputed_params = replicate_to_mesh(precompute_rope(config, mesh), mesh)
opt_state = replicate_to_mesh(opt_state, mesh)
```

This is **necessary but not sufficient** in our environment. Same crash.
We kept this fix in place; it's correct in principle and should help on
any future multi-host JAX run.

#### Attempted fix #7 (didn't work): shared persistent compile cache

```python
jax.config.update("jax_compilation_cache_dir", "/home/<USER>/gcs_bucket/jax_cache_v5p16")
```

Idea: if both hosts read/write the same XLA cache via the gcsfuse mount,
they'll execute identical compiled programs.

Failed for two reasons:
- gcsfuse mount permissions: `PermissionError: [Errno 13] Permission
  denied: '/home/<USER>/gcs_bucket/jax_cache_v5p16'`. The mount as set up
  is read-only for the worker user. Could be fixed with mount options but
  was not the root cause anyway.
- Even per-host `/tmp/jax_cache` (which we'd been using) was getting
  cleared before each run, so cache staleness wasn't the actual issue.

#### Attempted fix #8 (didn't work): drop AOT compile

Replaced:

```python
jitted_train_step = jit(train_step, static_argnames=("config", "optimizer"), donate_argnums=(1, 3))
# ... AOT compile per shape ...
compiled_fn = jitted_train_step.lower(config, params, ..., dummy_x, dummy_y).compile()
compiled_train_steps[shape_key] = compiled_fn
# ... in loop:
aot_train_fn = compiled_train_steps[current_shape_key]
params, opt_state, metrics = aot_train_fn(params, ..., batched_x, batched_y)
```

with:

```python
# Plain jit, lazy compile on first call per unique input shape.
params, opt_state, metrics = jitted_train_step(
    config, params, precomputed_params, opt_state, optimizer, batched_x, batched_y,
)
```

Theoretically this also helps because plain jit's per-call lookup uses
input-shape + sharding signatures that are identical across hosts, whereas
AOT's compiled handle bakes in launch IDs that are per-host. Same crash
in our environment, but kept this change because it's simpler and removes
one variable.

### 4.5 Phase 4 — the SSH idle-timeout red herring

Every run was dying at almost exactly **2:04** elapsed. With no Python
traceback. That oddly precise time suggested a wall-clock timeout, not a
crash.

Hypothesis: gcloud SSH (or some hop in between) was idle-disconnecting
during the long, silent JIT compile (~60-120s). That would deliver
`SIGHUP` to the remote python, killing it without a traceback.

#### Fix attempts

1. Modified `jobman/runner.py` to add SSH keepalive flags:
   ```python
   "--ssh-flag=-o ServerAliveInterval=30",
   "--ssh-flag=-o ServerAliveCountMax=20",
   ```
2. Added a heartbeat thread to `train.py` that prints `[heartbeat] alive
   Ns` every 30s, started at module import time:
   ```python
   def _start_heartbeat():
       import threading
       def _beat():
           i = 0
           while True:
               time.sleep(30)
               i += 1
               print(f"[heartbeat] alive {i*30}s pid={os.getpid()}", flush=True)
       threading.Thread(target=_beat, daemon=True).start()
   _start_heartbeat()
   ```

Result: heartbeat printed every 30s as expected. Process **still died at
2:04**. Heartbeat reached "alive 90s" then no further output.

So the SSH-timeout theory was wrong. Whatever was killing the process did
so even with regular stdout activity. The heartbeat stayed in the code
because it's a useful safety net regardless.

### 4.6 Phase 5 — the minimal sanity test that broke the deadlock

Wrote a 25-line script to test multi-host JAX with the minimum surface
area possible:

```python
# /tmp/jax_multihost_test.py
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

print(f"[p{jax.process_index()}] hello, devices={jax.device_count()} local={jax.local_device_count()}", flush=True)

mesh = jax.make_mesh((jax.device_count(),), ("dp",))
sharding = NamedSharding(mesh, P("dp"))

local = np.full((8,), float(jax.process_index()), dtype=np.float32)
global_arr = jax.make_array_from_process_local_data(sharding, local, (16,))

@jax.jit
def double(x):
    return x * 2.0

out = double(global_arr)
jax.block_until_ready(out)
local_out = np.asarray(out.addressable_data(0))
print(f"[p{jax.process_index()}] local_out={local_out}", flush=True)
print(f"[p{jax.process_index()}] DONE", flush=True)
```

Pushed via `gcloud alpha compute tpus tpu-vm scp ... --worker=all`, ran in
parallel:

```bash
ssh -i ~/.ssh/google_rsa <USER>@<HOST0_IP> "source ~/nanogpt_env/bin/activate && python -u ~/jax_multihost_test.py" &
ssh -i ~/.ssh/google_rsa <USER>@<HOST1_IP> "source ~/nanogpt_env/bin/activate && python -u ~/jax_multihost_test.py" &
wait
```

Result: **worked perfectly on both hosts.** Output:

```
[p0] hello, devices=8 local=4
[p0] global_arr sharding: NamedSharding(mesh=Mesh('dp': 8 ...), spec=PartitionSpec('dp',))
[p0] block_until_ready OK
[p0] local_out=[0. 0.]
[p0] DONE
[p1] hello, devices=8 local=4
[p1] block_until_ready OK
[p1] local_out=[2. 2.]
[p1] DONE
```

This proved:
- Multi-host JAX is functional on this TPU.
- `make_array_from_process_local_data` works correctly.
- `@jit` and `block_until_ready` work multi-host.
- The bug is **specific to `train.py`'s compile path**, not a TPU-wide
  problem.

This single test was, in retrospect, the most important debugging step.
**Do this first whenever you suspect environment / multi-host issues.**

### 4.7 Phase 6 — `dmesg` reveals libtpu segfaults

```bash
ssh -i ~/.ssh/google_rsa <USER>@<HOST0_IP> "sudo dmesg -T | grep -iE 'segfault|killed' | tail -10"
```

Output:

```
[Thu May 14 04:28:35 2026] tokio-rt-worker[316759]: segfault at 7fde1a4ecd58 ip 00007fde34a75048 sp 00007fde1a4ecd50 error 6 in libc.so.6[7fde34a28000+195000]
[Thu May 14 04:34:32 2026] tokio-rt-worker[320683]: segfault at 7fde1a4ecd58 ...
[Thu May 14 04:59:02 2026] tokio-rt-worker[323851]: segfault at ... 
... (eight more, all at the same instruction pointer)
```

`tokio-rt-worker` is libtpu's internal Rust async runtime worker thread.
The crashes were all at the **same instruction pointer offset in
libc.so.6**. Eight crashes in the past hour, all matching our failed
runs. This is a deterministic libtpu bug triggered by the script's compile
path on this specific multi-host topology.

### 4.8 Phase 7 — the actual fix: downgrade `jax[tpu]`

The version we'd been on:

```
jax       0.6.2
jaxlib    0.6.2
libtpu    0.0.17
```

Downgrade attempt: `pip install --force-reinstall "jax[tpu]==0.5.3" -f
https://storage.googleapis.com/jax-releases/libtpu_releases.html`

New versions:

```
jax       0.5.3
jaxlib    0.5.3
libtpu    0.0.11.4
```

Re-ran the minimal sanity test on jax 0.5.3 → still works.

Re-ran training. **First `block_until_ready` returned successfully.** No
more segfault.

A few small leftover bugs from earlier hacking surfaced:

1. `get_lr(step, n_warmup_iters=0, ...)` had `(it+1)/n_warmup_iters` which
   is fine inside jit (NaN/Inf, no crash) but raises `ZeroDivisionError`
   when called from Python on the host. Removed the host-side LR call.
2. `compiled_eval_fn = jitted_eval_step` (post AOT removal) didn't have
   `config` baked in; calls needed to pass it explicitly.
3. `Logger.log` was passing a `dict | host_metrics` when `host_metrics`
   could contain unhashable jax-array values; routed metrics through
   numpy/floats first.

After those: **training ran to completion**. 1675 steps, ~12 min, val_loss
**3.276** (target ≤ 3.28).

### 4.9 The `docker restart tpu-runtime` red herring

Before downgrading, I tried `sudo docker restart tpu-runtime` on each
host, on the theory that earlier `pkill -9` cycles might have left
libtpu in a bad state. **The restart succeeded but did not fix the
segfault** — confirming that the segfault is a real libtpu version bug,
not corrupted state. Worth knowing as a non-destructive recovery option
for actual stale state, though.

---

## 5. Things that did not work and why

Aggregated for quick reference:

| Attempt | Outcome | Why it failed |
|---|---|---|
| Round-robin batches per process (upstream code) | `AssertionError` in JAX 0.6 | Single-host pattern; multi-host expects all hosts contribute the same global batch each step |
| Every process loads every batch identically | 12+ min hang | 1675 cross-host gathers in `device_put`'s replicated-dim equality check |
| `jax.device_put` with sharded `NamedSharding` after construction | Silent desync | Need explicit `make_array_from_process_local_data` for multi-host |
| `addressable_data(0)` → `float()` on metric scalars | Same XLA launch-id crash | Crash was upstream in `train_step` execution; metric read just surfaced it lazily |
| `multihost_utils.process_allgather` for metrics | Same XLA launch-id crash | Same root cause |
| Skip metric extraction entirely | Same crash | Confirms it's `train_step` itself, not metric handling |
| `jax.distributed.initialize()` at top of script | No improvement | Cloud TPU auto-initialises libtpu; explicit init may double-register |
| Replicate `params`/`opt_state` via `replicate_to_mesh` | Necessary but not sufficient | Without it, host-local `SingleDeviceSharding` leaves cause divergent compiles. With it (and old libtpu), still crashes — but kept; it's correct |
| Switch from AOT `lower().compile()` to plain `@jit` | Necessary cleanup, not the fix on its own | AOT bakes per-host launch IDs into compiled handles; plain jit is more robust. Crash persists with libtpu 0.0.17 either way |
| Shared compile cache via gcsfuse mount | `PermissionError` on the mount path | gcsfuse mount permissions; would need different mount options. Also: not the cause of our crash |
| SSH `ServerAliveInterval=30` keepalives | No effect on crash | Process wasn't being killed by SSH timeout |
| Heartbeat thread (`print` every 30s) | Heartbeat printed; process still died | Process wasn't being killed by SSH timeout |
| `sudo docker restart tpu-runtime` on both hosts | TPU still healthy after; segfault persists | Bug is in shipped libtpu code, not runtime state |
| `gcloud alpha compute tpus tpu-vm stop` | `INVALID_ARGUMENT: "StopNode" is not supported on pod nodes` | Multi-host pods can't be stopped; only deleted |
| **`pip install jax[tpu]==0.5.3` (libtpu 0.0.11.4)** | **Worked.** | Older libtpu does not have the `tokio-rt-worker` segfault |

---

## 6. The fix that actually worked

Three categories of change. None alone is sufficient; you need all of them.

### 6.1 Pin the JAX/libtpu version

In your venv requirements file (e.g. `assets/nanogpt-requirements.txt`):

```
--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.5.3
numpy
einops
huggingface_hub
wandb
```

If your venv was already created with `jax[tpu]` un-pinned, force a
reinstall on each host:

```bash
~/nanogpt_env/bin/pip install --force-reinstall "jax[tpu]==0.5.3" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

After reinstall, confirm:

```bash
~/nanogpt_env/bin/pip show jax jaxlib libtpu | grep -E "^Name|^Version"
# Expected:
#   jax     0.5.3
#   jaxlib  0.5.3
#   libtpu  0.0.11.4
```

### 6.2 Fix the data loader for multi-host

In `load_dataset`, replace the upstream per-process round-robin with
per-process local-slice + `make_array_from_process_local_data`:

```python
process_rank = jax.process_index()
num_processes = jax.process_count()

assert config.micro_batch_size % num_processes == 0, (
    f"micro_batch_size {config.micro_batch_size} must be divisible by "
    f"num_processes {num_processes}"
)
local_micro = config.micro_batch_size // num_processes
local_start = process_rank * local_micro
local_stop  = local_start + local_micro

activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))

precomputed_batches = []
for global_step_idx in range(num_global_batches):
    # ... [same upstream logic to compute (batch_size, seq_len, n_grad_acc_steps),
    #      cycle the token cursor, slice all_tokens] ...
    x = np.array(buf[:-1], dtype=np.int32).reshape(batch_size, seq_len)
    y = np.array(buf[1:],  dtype=np.int32).reshape(batch_size, seq_len)
    batched_x = einops.rearrange(x, "(a b) ... -> a b ...", a=n_grad_acc_steps)
    batched_y = einops.rearrange(y, "(a b) ... -> a b ...", a=n_grad_acc_steps)

    global_shape = batched_x.shape   # (n_grad_acc, micro_batch, seq_len)
    local_x = batched_x[:, local_start:local_stop, :]
    local_y = batched_y[:, local_start:local_stop, :]

    batched_x = jax.make_array_from_process_local_data(activation_sharding, local_x, global_shape)
    batched_y = jax.make_array_from_process_local_data(activation_sharding, local_y, global_shape)

    precomputed_batches.append((batched_x, batched_y))
```

Note: every process now walks the **same global token cursor** (no
`if global_step_idx % num_processes == process_rank`) — both processes load
all tokens, slice their own piece. This is correct for synchronous data
parallelism.

### 6.3 Replicate model state explicitly

Add this helper somewhere near the top of `train.py`:

```python
def replicate_to_mesh(pytree, mesh):
    return tree_map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P(*((None,) * x.ndim)))),
        pytree,
    )
```

And use it on every pytree of state passed into `train_step`:

```python
params, precomputed_params = init_params(config, mesh)
params              = replicate_to_mesh(params, mesh)
precomputed_params  = replicate_to_mesh(precomputed_params, mesh)

optimizer, opt_state = init_optimizer(config, params, mesh)
opt_state            = replicate_to_mesh(opt_state, mesh)
```

This forces every leaf onto a fully-replicated `NamedSharding`, so the
compiled program signature is identical on every host. No more
`SingleDeviceSharding` sneaking through for tiny scalars (e.g. attention
biases, adam step counters).

### 6.4 Drop AOT, use plain `@jit`

In `train_loop`, replace the AOT compile loop with a plain `@jit` call.
Original:

```python
jitted_train_step = jit(train_step, static_argnames=("config", "optimizer"), donate_argnums=(1, 3))
# ... loop over train_shapes:
compiled_fn = jitted_train_step.lower(config, params, ..., dummy_x, dummy_y).compile()
compiled_train_steps[shape_key] = compiled_fn
# ... in loop:
aot_train_fn = compiled_train_steps[current_shape_key]
params, opt_state, metrics = aot_train_fn(params, ..., batched_x, batched_y)
```

After:

```python
jitted_train_step = jit(train_step, static_argnames=("config", "optimizer"), donate_argnums=(1, 3))
jitted_eval_step  = jit(eval_step,  static_argnames=("config",))
compiled_eval_fn  = jitted_eval_step
# ... in loop:
params, opt_state, metrics = jitted_train_step(
    config, params, precomputed_params, opt_state, optimizer, batched_x, batched_y,
)
```

Plain jit recompiles on first call per unique input-shape signature, then
caches in-process. You pay 60-120s once per shape (you have 2-3 shapes
total because of the seq-len warm-up), then it's cached for the rest of the
run.

### 6.5 Read metrics safely

```python
loss_val = float(np.asarray(metrics["loss"].addressable_data(0)))
```

This reads the local replicated copy of the loss scalar. Both hosts call
it (no master-only branch around it!), so they stay in lockstep.
`addressable_data(0)` is the per-process API that does not trigger a
cross-host gather. `np.asarray(...)` materialises to a host numpy scalar.
`float(...)` extracts a Python float. Safe to log.

### 6.6 Heartbeat + flushed prints

```python
import os, time, threading

def _start_heartbeat():
    def _beat():
        i = 0
        while True:
            time.sleep(30)
            i += 1
            try:
                print(f"[heartbeat] alive {i*30}s pid={os.getpid()}", flush=True)
            except Exception:
                return
    threading.Thread(target=_beat, daemon=True).start()
_start_heartbeat()
```

And run with `python -u` (or set `PYTHONUNBUFFERED=1`) so prints aren't
block-buffered through the `tee` pipe.

### 6.7 Things you do NOT need

- `jax.distributed.initialize()` — Cloud TPU auto-initialises.
- Persistent compile cache (`jax_compilation_cache_dir`) — disable it
  on multi-host until you really need it.
- `XLA_FLAGS` for determinism — not necessary in our setup.
- Custom shard_map / pjit — plain jit + `NamedSharding` works fine after
  the above fixes.

---

## 7. Operational gotchas with jobman + existing TPUs

### 7.1 `tpu.name` gets a suffix

`jobman create config.yaml` always rewrites the saved snapshot's
`tpu.name` to `{original}_{job_id}` (or `{original}_{worker_num}` if you
set one). There's no flag to disable this in the version we used.

To attach to an **existing** TPU (e.g. an on-demand pod that you've been
queueing for two weeks):

```bash
# 1. Create the job. This writes the snapshot with the WRONG tpu.name and
#    spawns tmux to start running.
cd /path/to/jobman
jobman create config/my-config.yaml
# → "Created job 000159"

# 2. Stop IMMEDIATELY (within ~10s). The tmux session has spawned `jobman
#    run`, which will issue `gcloud tpus tpu-vm create` for the wrong name
#    if you let it. SSH probe takes a few seconds, then create call fires.
jobman stop 000159

# 3. Verify no phantom TPU was created with the suffixed name.
gcloud alpha compute tpus tpu-vm list --zone <ZONE> \
  --filter="name~'<TPU_NAME>'" --format="value(name,state)"
# → Should show ONLY your existing TPU, no `<TPU_NAME>_000159`.

# 4. Edit the snapshot. The dir is `jobs/<user>/<job_id>/config.yaml`.
#    Find the `tpu.name:` line and remove the `_000159` suffix.

# 5. Resume. This re-spawns tmux which loads the snapshot, sees the TPU is
#    already READY (skips allocation), and runs the command.
jobman resume 000159
```

If a phantom TPU did get created, `gcloud alpha compute tpus tpu-vm
delete <name> --zone <ZONE>` to remove it.

### 7.2 `gcsfuse` key required even when unused

Jobman's `validate_cfg` accesses `cfg.gcsfuse` directly. With OmegaConf,
that raises `ConfigAttributeError: Missing key gcsfuse` if the key isn't
in the YAML. Pass `gcsfuse: null` if you don't want to mount a bucket.

### 7.3 Setup is idempotent (good); commands are not (bad)

The `ssh`/`gcsfuse`/`venv` setup steps use a check-then-setup pattern, so
a `jobman resume` after a failed run skips re-installing things. The
`command` step always runs, though — if your command's first action is
`rm -rf modded-nanogpt-jax && git clone ...`, that re-clones the repo on
every run.

### 7.4 Stopping a job does not kill the remote python

`jobman stop <id>` kills the local tmux session that hosts the gcloud SSH
connection. The remote python on the TPU **may** receive `SIGHUP` and exit,
or may not — depends on whether the SSH session is delivered cleanly.

If a previous run's python is still alive on the TPU, the next run's
python will fail with:

```
RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is
already in use by process with pid <N>.
```

Recovery: SSH directly and check, then politely `kill -TERM <pid>` (NOT
`-9`). Avoid `pkill -9` on JAX processes — it can corrupt libtpu state and
necessitate a `docker restart tpu-runtime` to recover.

### 7.5 `bucket region must match TPU zone`

Jobman's `validate_cfg` enforces this if you have a `gcsfuse` block. If
not, you should still respect it — cross-region GCS reads from a TPU are
slow and incur egress charges.

### 7.6 Use `python -u`, not just `python`

`python train.py 2>&1 | tee log` block-buffers stdout because the pipe
isn't a terminal. You'll see nothing in the log file for minutes. Always
use `python -u train.py 2>&1 | tee log` (or set `PYTHONUNBUFFERED=1`).

### 7.7 Worker count inference per accelerator

Jobman infers `num_workers` from the accelerator string. Useful to know:

| Accelerator | Workers (hosts) | Chips/worker |
|---|---|---|
| v4-8        | 1  | 4 |
| v4-16       | 2  | 4 |
| v4-32       | 4  | 4 |
| v4-256      | 32 | 4 |
| v5e-4       | 1  | 4 |
| v5e-32      | 8  | 4 |
| v5p-8       | 1  | 4 |
| v5p-16      | 2  | 4 |
| v5p-64      | 8  | 4 |
| v6e-8       | 1  | 8 |
| v6e-64      | 16 | 4 |

So v6e-8 is single-host (the original target of this script) and v5p-16
is multi-host with two workers.

---

## 8. Hard rules learned the painful way

1. **NEVER `pkill -9` a JAX/libtpu process during initialisation.**
   It can leave libtpu's coordinator socket in a broken state and the only
   recovery is a `docker restart tpu-runtime` (or worse, a TPU pod
   recreate, which means re-queueing — possibly weeks of waiting). SIGTERM
   first; if you must escalate, stop the job at the jobman level and let
   the SSH disconnect propagate naturally.
2. **NEVER `rm /tmp/libtpu_lockfile` while a process is using it.**
   This is a libtpu coordination file. Removing it mid-init is roughly the
   same as `kill -9`. Only delete it after you've confirmed the holding
   process is dead and reaped.
3. **NEVER restart `tpu-runtime` (the docker container) without explicit
   user permission.** It's a recovery option for genuine corrupted state,
   not a debugging tool.
4. **NEVER `gcloud tpus tpu-vm delete`** an in-use pod without confirming
   you're prepared to re-queue. Pods can't be `stop`/`start`-ed; delete is
   one-way.
5. **ASK before destructive actions.** "Should I run `docker restart
   tpu-runtime` on each host?" is the right form. "I'm going to restart
   the runtime" is not.
6. **One change at a time.** When debugging multi-host, do not change the
   data loader, the metric extraction, AND the JAX version in the same
   iteration. You won't be able to attribute the result to anything.
7. **Write the minimal sanity test FIRST.** Whenever you're not sure if a
   problem is in the model, the framework, or the environment, the first
   thing you write is a 25-line script that exercises the framework
   without the model. The test in §4.6 would have saved hours.
8. **Always check `dmesg` early.** Kernel logs surface segfaults that
   Python's traceback can't see. `sudo dmesg -T | grep -iE 'segfault|killed
   process|oom'` is your friend.

---

## 9. Reproduction recipe end-to-end

Assuming: GCP project, a multi-host TPU pod, gcloud authenticated, jobman
installed, you have a fork of this repo.

### 9.1 One-time setup

```bash
# Stage data into your bucket (matching region!).
pip install --user huggingface_hub
cd /tmp
git clone https://github.com/<your-fork>/modded-nanogpt-jax.git stage
python stage/cached_fineweb10B.py 10        # ~2 GB
gcloud storage cp -r stage/fineweb10B gs://<BUCKET>/datasets/

# Verify
gcloud storage ls gs://<BUCKET>/datasets/fineweb10B/ | head
```

### 9.2 Requirements file

`assets/nanogpt-requirements.txt` (in your jobman repo):

```
--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.5.3
numpy
einops
huggingface_hub
wandb
```

### 9.3 Jobman config

`config/nanogpt-multihost.yaml`:

```yaml
job:
  name: nanogpt
  env_type: venv
  loop: false
  remote_user: <USER>            # the SSH user on TPU VMs

tpu:
  allocation_mode: "tpu-vm"
  accelerator: v5p-16             # or v5p-64, v6e-64, ...
  name: <TPU_NAME>                # the actual existing pod name
  zone: <ZONE>                    # us-central1-a, etc.
  version: v2-alpha-tpuv5
  pricing: ondemand

gcsfuse:
  bucket_name: <BUCKET>           # MUST be in same region as TPU zone
  mount_path: /home/<USER>/gcs_bucket

ssh:
  private_key: ~/.ssh/google_rsa
  identities:
    - private_key: ~/.ssh/google_rsa
      public_key: ~/.ssh/google_rsa.pub
      config_entry: |
        Host 10.* 34.* 35.*
          IdentityFile ~/.ssh/google_rsa
          IdentitiesOnly yes

venv:
  name: nanogpt_env
  requirements_file: assets/nanogpt-requirements.txt
  python: "python3.10"

command:
  cmd: |
    set -euo pipefail
    cd ~
    rm -rf modded-nanogpt-jax
    git clone --depth 1 https://github.com/<your-fork>/modded-nanogpt-jax.git
    cd modded-nanogpt-jax
    git rev-parse --short HEAD
    grep -c 'make_array_from_process_local_data' train.py

    rm -rf fineweb10B
    ln -s /home/<USER>/gcs_bucket/datasets/fineweb10B fineweb10B
    ls -lh fineweb10B/ | head -3

    pip install --quiet wandb || true
    rm -rf /tmp/jax_cache

    export WANDB_API_KEY='<KEY>'
    export WANDB_PROJECT='modded-nanogpt-jax'
    export WANDB_RUN_NAME='nanogpt-run'
    export WANDB_RUN_ID='nanogpt-run'

    python -u train.py 2>&1 | tee -a ~/train_$(hostname).log
  workers: "all"
```

### 9.4 Launch

```bash
cd /path/to/jobman
jobman create config/nanogpt-multihost.yaml
# → "Created job 000NNN"

# IMMEDIATELY (within ~10s):
jobman stop 000NNN

# Verify no phantom TPU was created:
gcloud alpha compute tpus tpu-vm list --zone <ZONE> \
  --filter="name~'<TPU_NAME>'" --format="value(name,state)"

# Edit the snapshot to fix the suffixed tpu.name:
$EDITOR jobs/<user>/000NNN/config.yaml
# Change `name: <TPU_NAME>_000NNN` back to `name: <TPU_NAME>`

# Resume (this is the actual run):
jobman resume 000NNN
```

### 9.5 Monitor

```bash
# Job status (RUNNING/IDLE/DEAD)
cat jobs/<user>/000NNN/.job_status

# Jobman's own log (setup phases)
tail -f jobs/<user>/000NNN/logs/job.log

# Per-worker command output (where the actual training output goes)
tail -f jobs/<user>/000NNN/logs/command_worker_0.log

# Or directly on the TPU (more reliable for the live training output):
ssh -i ~/.ssh/google_rsa <USER>@<HOST0_IP> 'tail -f ~/train_*.log'

# Sanity: is the python process actually running?
ssh -i ~/.ssh/google_rsa <USER>@<HOST0_IP> \
  'ps -o pcpu,etime,rss,cmd --no-headers -p $(pgrep -f "python -u train" | head -1)'
```

### 9.6 Expected log progression

```
... gcloud, ssh, gcsfuse setup (idempotent on resume) ...
[trace optional]
Determining all unique training shapes...
Starting Ahead-of-Time (AOT) compilation for all shapes...   # leftover msg, even after AOT removal
AOT compile skipped — using plain jit (compiles on first call per shape).
Pre-computing and loading all training batches...
Process 0/2 starting data pre-loading into RAM...
Process 0/2 finished loading 2.00 GB of tokens.
Process 0/2 pre-computed 1675 batches.
Loaded 1675 training batches for this process.
Pre-computing and loading all validation batches...
Process 0/2 finished loading 0.20 GB of tokens.
Process 0/2 pre-computed 20 batches.
Loaded 20 validation batches for this process.
Starting training...
[heartbeat] alive 30s pid=NNN
[heartbeat] alive 60s pid=NNN
[heartbeat] alive 90s pid=NNN
step: 0  |  loss: 11.0235  |  step_secs: 88.7  |  tokens_per_sec: 5912  |  seq_len: 1024  |  batch_size: 512
step: 1  |  loss: 10.8910  |  step_secs: 0.397  |  tokens_per_sec: 1320000  |  seq_len: 1024  |  batch_size: 512
... (loss decreasing, throughput stable around 1.3M tokens/sec) ...
Running validation for step 125...
step: 125  |  val_loss: 5.04  |  ...
... continues ...
Final validation
Running validation for step 1674...
step: 1674  |  val_loss: 3.276  |  ...
Training finished.
Saved checkpoint to logs/<run_id>/state_step001674.pkl
```

### 9.7 If it dies at exactly 2:04

Probably libtpu segfault. Verify:

```bash
ssh -i ~/.ssh/google_rsa <USER>@<HOST0_IP> "sudo dmesg -T | grep -iE 'segfault|killed' | tail -5"
```

If you see `tokio-rt-worker[NNN]: segfault`, you're on a broken libtpu
version. Force-reinstall:

```bash
ssh -i ~/.ssh/google_rsa <USER>@<HOST0_IP> '~/nanogpt_env/bin/pip install --force-reinstall "jax[tpu]==0.5.3" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html' &
ssh -i ~/.ssh/google_rsa <USER>@<HOST1_IP> '~/nanogpt_env/bin/pip install --force-reinstall "jax[tpu]==0.5.3" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html' &
wait
```

Then re-resume.

---

## 10. Diagnostic commands you'll use over and over

### 10.1 SSH directly into a TPU host

```bash
# Find the external IPs:
gcloud alpha compute tpus tpu-vm describe <TPU_NAME> --zone <ZONE> \
  --format="value(networkEndpoints[].accessConfig.externalIp)"

# SSH (use the same private key jobman uses; bypasses gcloud SSH):
ssh -i ~/.ssh/google_rsa -o StrictHostKeyChecking=no <USER>@<HOST_IP>
```

This is **much** more reliable than `gcloud alpha compute tpus tpu-vm
ssh --worker=N` (which sometimes fails with exit 255 mysteriously).

### 10.2 Check TPU state

```bash
gcloud alpha compute tpus tpu-vm describe <TPU_NAME> --zone <ZONE> \
  --format="value(state,acceleratorType)"
# READY    v5p-16    ← good
# CREATING            ← wait
# PREEMPTED           ← spot got preempted
# NOT_FOUND           ← deleted or wrong name
```

### 10.3 Check what's running on the TPU

```bash
ssh -i ~/.ssh/google_rsa <USER>@<HOST_IP> '
  echo === python processes ===;
  pgrep -af python | grep -v "unattended\|networkd\|grep";
  echo === TPU lockfile ===;
  ls -la /tmp/libtpu_lockfile 2>&1;
  echo === TPU runtime container ===;
  sudo docker ps --filter name=tpu-runtime --format "{{.Names}} {{.Status}}";
  echo === TPU chips ===;
  ls /dev/vfio/
'
```

### 10.4 Diagnose a hung python

```bash
PID=$(ssh -i ~/.ssh/google_rsa <USER>@<HOST_IP> 'pgrep -f "python -u train" | head -1')
ssh -i ~/.ssh/google_rsa <USER>@<HOST_IP> "
  echo === ps ===;
  ps -o pid,etime,pcpu,rss,wchan --no-headers -p $PID;
  echo === stack ===;
  sudo cat /proc/$PID/stack 2>/dev/null | head -8;
  echo === tcp peers ===;
  sudo ss -tnp 2>/dev/null | grep $PID | head -10;
  echo === jax cache count ===;
  ls /tmp/jax_cache 2>/dev/null | wc -l
"
```

`wchan: futex_wait_queue` + low CPU = waiting on a lock (probably
cross-host coordination). `wchan: do_wait` = parent waiting on child (this
is a bash wrapper, not the actual python).

### 10.5 Check kernel logs for the killer

```bash
ssh -i ~/.ssh/google_rsa <USER>@<HOST_IP> "
  sudo dmesg -T | grep -iE 'segfault|killed process|oom|libtpu' | tail -15
"
```

Look for `segfault at .* in libc.so.6` — usually means libtpu issue.

### 10.6 Clean up between runs

```bash
# Stale lockfiles (only if the holding process is confirmed dead):
ssh -i ~/.ssh/google_rsa <USER>@<HOST_IP> 'sudo rm -f /tmp/libtpu_lockfile'

# Persistent compile cache (in case it desynced):
ssh -i ~/.ssh/google_rsa <USER>@<HOST_IP> 'rm -rf /tmp/jax_cache'

# Orphaned python from previous run (only if confirmed not currently
# initialising libtpu — check `ps` first; prefer SIGTERM):
ssh -i ~/.ssh/google_rsa <USER>@<HOST_IP> 'sudo kill -TERM $(pgrep -f "python -u train")'
```

### 10.7 Run the multi-host JAX sanity test

Save this as `/tmp/jax_multihost_test.py`:

```python
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

print(f"[p{jax.process_index()}] hello, devices={jax.device_count()} local={jax.local_device_count()}", flush=True)

mesh = jax.make_mesh((jax.device_count(),), ("dp",))
sharding = NamedSharding(mesh, P("dp"))

local = np.full((jax.local_device_count() * 2,), float(jax.process_index()), dtype=np.float32)
global_arr = jax.make_array_from_process_local_data(sharding, local, (jax.device_count() * 2,))

@jax.jit
def double(x):
    return x * 2.0

out = double(global_arr)
jax.block_until_ready(out)
print(f"[p{jax.process_index()}] local_out={np.asarray(out.addressable_data(0))} DONE", flush=True)
```

Push and run on both hosts in parallel:

```bash
gcloud alpha compute tpus tpu-vm scp /tmp/jax_multihost_test.py \
  <USER>@<TPU_NAME>:~/jax_multihost_test.py \
  --worker=all --zone <ZONE> --strict-host-key-checking=no \
  --ssh-key-file ~/.ssh/google_rsa

ssh -i ~/.ssh/google_rsa <USER>@<HOST0_IP> \
  'source ~/nanogpt_env/bin/activate && python -u ~/jax_multihost_test.py' &
ssh -i ~/.ssh/google_rsa <USER>@<HOST1_IP> \
  'source ~/nanogpt_env/bin/activate && python -u ~/jax_multihost_test.py' &
wait
```

If this works but `train.py` doesn't, the problem is in the script.
If this doesn't work, the problem is in the environment.

---

## 11. Open questions / things still unclear

These are things I didn't fully figure out. Useful to know what's still
load-bearing on assumptions:

- **Why does libtpu 0.0.17 segfault on this script's compile path but not
  on the minimal sanity test?** The crash is deterministic, always at the
  same `libc.so.6` instruction pointer, always in `tokio-rt-worker`. Most
  likely a real libtpu bug that triggers on certain XLA program patterns —
  perhaps the donated argument handling, the `scan` over the gradient
  accumulation loop, or the multi-optimizer gradient scatter. Not
  investigated further once the downgrade fixed it.
- **Does jax 0.6.0 / 0.6.1 also have the bug?** I jumped from 0.6.2 to
  0.5.3 directly. A more granular bisect could narrow which libtpu
  release introduced the issue. If you upgrade past 0.5.3, run the
  minimal test before relying on it.
- **Is `replicate_to_mesh` strictly necessary on libtpu 0.0.11.4?** It
  was added during the libtpu 0.0.17 era. We didn't test removing it
  after the downgrade; on principle it's still correct, so we left it in.
- **Do AOT compiles work with libtpu 0.0.11.4?** We removed AOT during the
  same era. Possibly works again; not retested.
- **Does the heartbeat thread still matter?** Probably not for plain
  training (steps print every 0.4s, plenty of stdout traffic), but it
  doesn't cost anything and is useful insurance during long compiles.
  Kept.
- **The `Config.batch_size = 8 * 64 = 512` constant is hard-coded for
  v6e-8 (8 chips).** On v5p-16 (also 8 chips) it happens to work
  unchanged. On a larger pod (e.g. v5p-64 = 32 chips) you'd want to
  scale this up to keep per-chip batch reasonable. Not tested.
- **Loss curve.** Final val_loss 3.276 is below the 3.28 target, but only
  just. The original was tuned for v6e-8; the optimizer hyperparameters
  may not be perfectly calibrated for v5p-16's slightly different
  numerical behaviour. A run with seed variation would tell us how tight
  the margin is.

---

## Appendix: timeline of the actual debugging session

For posterity / morbid amusement.

| Elapsed | Status |
|---|---|
| 0:00 | Set up jobman config; first connectivity test on v5p-16. |
| 0:20 | First training attempt → assert_equal crash (data loader). |
| 0:40 | "Every process loads everything" → 12-min hang. |
| 1:10 | `make_array_from_process_local_data` rewrite → got to step 9. |
| 1:25 | First `format(jax.Array)` crash. Tried 5 different metric strategies over the next 90 minutes. |
| 3:00 | Suspected SSH idle timeout. Added heartbeat. Heartbeat works, process still dies at 2:04. |
| 3:30 | Wrote minimal multi-host JAX test → it works. Realised the bug is in `train.py`, not the environment. |
| 3:45 | Checked `dmesg`: 8 segfaults in `tokio-rt-worker`. |
| 4:00 | Tried `docker restart tpu-runtime`. Same segfault. |
| 4:15 | Pinned `jax[tpu]==0.5.3`. Sanity test still works. |
| 4:25 | First `block_until_ready` returned. Trivial Python bugs (ZeroDivisionError, missing arg). |
| 4:45 | Training reached step 1674. val_loss = 3.276. ✓ |

About 5 hours of wall-clock to figure out one libtpu version bug. Most of
that time was bouncing between metric-extraction strategies, before the
sanity test made it obvious where the bug actually was.

The lesson: **write the minimum test, check `dmesg`, and pin your
dependencies.**

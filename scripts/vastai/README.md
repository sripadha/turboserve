# `scripts/vastai/` — rent an H100, run the suite, bring the numbers home

Five scripts and an on-start hook. Together they are `make bench-h100`: rent one H100 80GB,
prepare it, push this working tree, run the benchmark suite, pull `results/` back, and
destroy the instance.

```
provision.sh ──► onstart.sh (runs on the instance)
     │
     ├─► sync.sh           laptop working tree ──rsync──► /workspace/turboserve
     ├─► run_remote.sh     make bench PROFILE=h100
     ├─► pull_results.sh   results/ ◄──rsync── instance
     └─► destroy.sh        stop the meter
```

The long version, with the reasoning, is in [`docs/vastai.md`](../../docs/vastai.md). This
file is the reference.

## Prerequisites

```bash
uv tool install vastai          # or: pip install --user vastai
vastai set api-key <your key>   # from https://cloud.vast.ai/account/
ssh-keygen -t ed25519           # if you have no key
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
```

`rsync`, `ssh` and `python3` must be on PATH locally. Nothing else is installed on the
laptop; torch, vLLM and the models are installed on the instance.

## The scripts

| Script | What it does |
| --- | --- |
| `provision.sh` | Searches offers, picks the cheapest that passes the filters, creates the instance with `onstart.sh`, waits for ssh and for on-start to finish, writes the state file |
| `onstart.sh` | Runs on the instance: apt packages, uv, clone, `uv sync --extra vllm`, download the profile's models, write the ready marker |
| `sync.sh` | rsync this working tree over the clone (this is how uncommitted code gets measured) |
| `run_remote.sh` | Run `make bench PROFILE=h100` (or any command) under `nohup`, streaming the log |
| `pull_results.sh` | rsync `results/` back; never deletes local files |
| `destroy.sh` | List result files still on the instance, confirm, destroy |

## Typical session

```bash
export HF_TOKEN=hf_...                   # only needed for gated checkpoints
scripts/vastai/provision.sh              # ~10-30 min, mostly the model downloads
scripts/vastai/sync.sh
scripts/vastai/run_remote.sh             # make bench PROFILE=h100
scripts/vastai/pull_results.sh
scripts/vastai/destroy.sh
make results                             # render tables and plots from results/*.json
```

## Configuration

Every script reads a state file written by `provision.sh`, and every value in it can be
overridden from the environment.

```
${XDG_STATE_HOME:-$HOME/.local/state}/turboserve/vastai.env
```

It lives outside the repository on purpose: a rented instance's id and ssh endpoint are
not something to commit, and a `git clean` must not lose track of a machine that is
billing.

| Variable | Default | Used by |
| --- | --- | --- |
| `VAST_STATE_FILE` | `$XDG_STATE_HOME/turboserve/vastai.env` | all |
| `GPU_NAME` | `H100_SXM` | `provision.sh` |
| `NUM_GPUS` | `1` | `provision.sh` |
| `DISK_GB` | `200` | `provision.sh` |
| `CUDA_MIN` | `12.4` | `provision.sh` |
| `MIN_RELIABILITY` | `0.98` | `provision.sh` |
| `MIN_DOWNLOAD_MBPS` | `300` | `provision.sh` |
| `MAX_PRICE` | `4.00` | `provision.sh` — refuses to rent above this $/hr |
| `IMAGE` | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` | `provision.sh` |
| `DRY_RUN` | `0` | `provision.sh` — print the offers and stop |
| `WAIT_ONSTART` | `1` | `provision.sh` |
| `HF_TOKEN` | unset | `provision.sh` → the instance's environment |
| `TURBOSERVE_REPO_URL` / `TURBOSERVE_REF` | the public repo / `main` | `onstart.sh` |
| `TURBOSERVE_MODELS` | the h100 profile's four Qwen2.5 checkpoints | `onstart.sh` |
| `DELETE` | `1` | `sync.sh` — mirror deletions to the instance |
| `PROFILE` | `h100` | `run_remote.sh` |
| `FORCE` | `0` | `destroy.sh` — skip the confirmation |

## Things about vast.ai that shaped these scripts

**An instance is a container.** You choose the image at creation time and there is no
nested Docker inside it. So the Helm chart and `docker-compose.yml` cannot be used on
vast.ai: vLLM is pip-installed into the instance (`uv sync --extra vllm`,
`deploy/vllm/launch.sh`) and the gateway runs as a process beside it. Running each engine
in its own official image means renting one instance per engine, not one instance with
several containers.

**The price is per instance, and it belongs in the results.** `provision.sh` records the
instance's `dph_total` in the state file; `run_remote.sh` exports it as
`TURBOSERVE_GPU_PRICE_PER_HOUR` with `TURBOSERVE_GPU_PRICE_SOURCE="vast.ai instance price
at run time"`, and the benchmark writes both into every result file. Cost per million
tokens is derived from that, never from a price typed into a document.

**Disk is per instance, and models are downloaded there.** Nothing large is ever rsynced
from the laptop; `sync.sh` excludes `.venv/`, `results/`, `adapters/` and every cache.
`onstart.sh` downloads the checkpoints straight from Hugging Face on the instance's link,
which is an order of magnitude faster than a home connection.

**ssh is the only interface, and it is on a non-standard port.** Every script takes
`-p $VAST_SSH_PORT`; `provision.sh` asks for a direct endpoint (`--ssh --direct`) rather
than the vast.ai proxy, because rsync over the proxy is slow enough to notice.

**Runs outlive connections.** `run_remote.sh` starts the command under `nohup` and tails
its log, so a dropped laptop connection does not kill a run that is being billed by the
hour. Re-running the script reattaches to the same log.

**`ncu` may not work.** Nsight Systems (`nsys`) works inside containers, but Nsight
Compute's hardware counters usually need a host-level setting
(`NVreg_RestrictProfilingToAdminUsers=0`) that a rented instance cannot change. If `ncu`
fails, capture its exact error into the run's notes rather than working around it.

## What has been run here

These scripts have been checked for syntax (`bash -n`) and their embedded Python helpers
have been exercised against recorded `vastai` JSON shapes. They have not been run against
the vast.ai API from this checkout: doing so would rent a GPU and, per the project's rules,
no measurement runs happen on the development machine. The first real run is the one that
produces the measured results.

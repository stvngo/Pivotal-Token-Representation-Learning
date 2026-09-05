# Driving Colab GPUs from the terminal

The Colab CLI replaces the notebook workflow for this project. Instead of
opening a browser, pasting a config cell and babysitting a session, GPU work
is a shell command:

```bash
scripts/colab new -s ptrl-g4 --gpu G4      # provision
scripts/colab exec -s ptrl-g4 -f job.py    # run a LOCAL script remotely
scripts/colab download -s ptrl-g4 /content/out.json ./out.json
scripts/colab stop -s ptrl-g4              # release (IMPORTANT: bills otherwise)
```

`colab run script.py` does provision → run → tear down in one shot, and
propagates the script's exit code, which makes it usable in a pipeline.

Use `scripts/colab`, not the bare `colab` binary — the wrapper fixes two
environment problems described below.

---

## Hardware actually available on this account

| `--gpu` | What you get | VRAM |
| --- | --- | --- |
| `G4` | **NVIDIA RTX PRO 6000 Blackwell Server Edition**, sm_120, 188 SMs | **96 GB** |
| `A100` | A100-SXM4 | 40 GB |
| `H100`, `L4`, `T4` | as named | — |
| *(omitted)* | CPU | — |

Both G4 and A100 provision successfully on this Pro+ account. **G4 is the
one to use**: 96 GB against the A100's 40 GB means a far larger KV cache,
which is exactly what the wave scheduler needs — more concurrent sequences
means deeper batches, and rollout generation is ~95% of PTS's cost. It also
comfortably holds Qwen3-4B (8 GB of weights in bf16) with ~85 GB left for
cache.

Two cautions, both from the CLI's own skill file:

- **An unrecognized `--gpu` value silently falls back to A100.** So a typo
  gets you the wrong (billed) hardware rather than an error. The accepted
  set is exactly `T4, L4, G4, H100, A100`.
- **Nothing reclaims an idle session except a 24h keep-alive cap.** An
  unstopped VM burns compute units indefinitely. Always `colab stop`.

---

## Setup, and the three things that went wrong

### 1. Install it isolated, and pin `jupyter-kernel-client`

```bash
uv tool install google-colab-cli --with "jupyter-kernel-client==0.15.0"
```

colab-cli 0.6.0 calls `jupyter_kernel_client.KernelClient`, which was
renamed in that package's 1.0.0. Installing into this project's venv
resolves it to 1.0.2 (pulled in by `ipykernel`) and every `colab exec` dies
with `AttributeError: module 'jupyter_kernel_client' has no attribute
'KernelClient'`. 0.15.0 is the last release that still exposes it.

### 2. Certificates

The python.org build of Python ships no CA bundle, so the CLI's `urllib`
calls fail with `CERTIFICATE_VERIFY_FAILED` before reaching Google. The
wrapper sets `SSL_CERT_FILE` from `certifi`.

### 3. Auth: ADC, with the scope everyone misses

`--auth=adc` is the headless path; `oauth2` opens a browser and cannot be
driven from a script. The wrapper always passes `--auth=adc`.

ADC must be minted with four scopes. The one that is easy to omit is
`colaboratory`, which the keep-alive RPC needs:

```bash
gcloud auth application-default login \
  --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```

Check with `scripts/colab whoami`, which prints the active identity, scopes
and token expiry. Note the token expires hourly; a long run needs a fresh
one, and the keep-alive daemon inherits whatever `--auth` was used.

> Observed: provisioning and `exec` both work even without the
> `colaboratory` scope, and the keep-alive daemon started cleanly. The
> skill warns it *should* 403. Add the scope anyway before a multi-hour
> run — a keep-alive death mid-run loses the VM, though not the work,
> since PTS checkpoints per wave.

---

## Getting vLLM running (the fiddly part)

On a fresh G4 session, in this order:

```bash
scripts/colab install -s <name> vllm
```

then, once, on the VM:

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/cu130
```

Three separate breakages, each with a non-obvious cause:

1. **`ImportError: libcudart.so.13`.** vLLM 0.28 is built for CUDA 13;
   Colab's toolkit is 12.8. The library it needs ships *inside torch* at
   `nvidia/cu13/lib/libcudart.so.13`, so importing `torch` before `vllm`
   resolves it and importing `vllm` first does not. Our backend does this
   deliberately. Note this also means a stale kernel can mislead you: an
   `import vllm` that succeeded in a session where torch was already
   imported will fail in a fresh process. `colab restart-kernel` before
   concluding anything.
2. **`Detected that PyTorch and TorchAudio were compiled with different
   CUDA versions`.** Installing vllm upgrades torch to cu130, orphaning
   Colab's preinstalled cu128 torchaudio/torchvision. Uninstall
   torchaudio; *reinstall* torchvision from the cu130 index, because vLLM
   imports it at engine init and fails with
   `ModuleNotFoundError: No module named 'torchvision'` otherwise.
3. **`FlashInfer requires GPUs with sm75 or higher`** on an sm_120 GPU,
   which plainly satisfies that. A detection bug. The backend defaults
   `VLLM_ATTENTION_BACKEND=TRITON_ATTN` and
   `VLLM_USE_FLASHINFER_SAMPLER=0`, both overridable.

Measured once working, Qwen3-0.6B on the G4:

| workload | time | throughput |
| --- | --- | --- |
| engine init | 29 s | — |
| 1 node x 40 rollouts | 3.0 s | 4,280 tok/s |
| 64 nodes x 40 rollouts | 21.5 s | **38,074 tok/s** |

## Long jobs

`colab exec` streams over a websocket and gives up after a few minutes --
vLLM spends longer than that compiling for a new architecture before it
emits a token. Use the job runner, which starts the script under `nohup`
and polls its log:

```bash
scripts/colab_job.py -s ptrl-g4 --script scripts/pts_run.py -- --model Qwen/Qwen3-0.6B ...
scripts/colab_job.py -s ptrl-g4 --status
scripts/colab_job.py -s ptrl-g4 --tail -n 50
```

It syncs the repo on the VM from the working branch before starting,
unless `--no-sync`.

## Agent notes

The CLI ships its own skill file: `colab skill`. Worth reading; the points
that matter most here are:

- **Kernel state persists across `colab exec` calls in one session.** Each
  call reattaches to the same kernel, so imports and variables survive.
  Build state up incrementally instead of re-importing every time.
- **Never run `repl`, `console`, `auth`, or `drivemount` from an agent** —
  they want a TTY and will hang. `exec` and `run` are the scriptable ones.
- `--config <path>` isolates session state, so two parallel runs do not
  fight over `~/.config/colab-cli/sessions.json`.
- `colab log -s <name> -o run.ipynb` exports a session as a notebook, which
  is a convenient artifact to attach to a result.

## MCP

`.mcp.json` registers `colab-mcp`, which bridges to a Colab session in the
browser and is aimed at *notebook* development. It is complementary to the
CLI rather than a replacement: the CLI is what runs jobs. Requires a
restart of the client to load.

## How this changes the plan

The per-model Colab notebooks (`notebooks/pts_generate_*.ipynb`) are no
longer the primary path — they remain as a fallback if the CLI is
unavailable. The generation phase is now a shell command against a G4,
which means it can be driven, monitored and resumed from here rather than
handed over.

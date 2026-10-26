---
name: nori-rel-setup
description: Install Nori-Rel with the right extras for the user's database and inference mode (local Nori in the process, or hosted Nori with an API key), then verify the interpreter, the model runtime, the weights or key, and the accelerator. Use when someone starts a Nori-Rel project, asks how to install or configure nori-rel, or an import, weight-download or API-key error blocks a prediction.
---

# Set up Nori-Rel

Nori-Rel is the `nori-rel` package on PyPI. Two choices decide the install:
which database it reads, and where Nori runs. Make them with the user, install,
verify, and report the versions.

## 1. Two questions

Ask only what changes the install:

- **Database:** PostgreSQL (also AWS RDS and Aurora), Snowflake, Databricks,
  or files. Files go through [nori-rel-prepare-data](../nori-rel-prepare-data/SKILL.md),
  which loads them into PostgreSQL, so they need the PostgreSQL extra too.
- **Where Nori runs:**

| | Local (default) | Hosted |
|---|---|---|
| Account | none | a Synthefy Nori API key in `SYNTHEFY_NORI_API_KEY` |
| What leaves the process | nothing | the feature matrices and labels; never credentials or raw tables |
| Targets | numeric: counts, sums, averages, static columns; point or quantile output | numeric, and yes/no targets such as `COUNT(orders.*,0,30,days)=0` |
| Hardware | CPU works; CUDA or Apple MPS is faster | none needed |
| Extra | `local` (installs the pinned `synthefy-nori` runtime and PyTorch) | none |

A yes/no question ("will they churn", "will this lead convert") needs hosted
mode. Everything else runs locally unless the user prefers hosted.

## 2. Install

Use Python 3.11, 3.12 or 3.13 in a fresh virtual environment. Install the
latest release; resolve it from PyPI rather than assuming:

```bash
python -m pip index versions nori-rel      # the first line is the latest
```

| Database | Local Nori | Hosted Nori |
|---|---|---|
| PostgreSQL, RDS, Aurora, files | `pip install "nori-rel[database,local]"` | `pip install "nori-rel[database]"` |
| Snowflake | `pip install "nori-rel[snowflake,local]"` | `pip install "nori-rel[snowflake]"` |
| Databricks | `pip install "nori-rel[databricks,local]"` | `pip install "nori-rel[databricks]"` |

Add `agentic` to the extras only when the user wants the agent feature search
([nori-rel-features](../nori-rel-features/SKILL.md)); it adds the Anthropic and
OpenAI SDKs and DuckDB.

The `local` extra pins the `synthefy-nori` runtime to the one version this
release of Nori-Rel supports. Do not install or upgrade `synthefy-nori`
separately. On a Linux machine without a GPU, install the CPU build of
PyTorch first to avoid a multi-gigabyte CUDA download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "nori-rel[database,local]"
```

## 3. Verify

Run this in the environment that will run the predictions:

```python
import sys
from importlib.metadata import version

from nori_rel import NoriRelClient

assert (3, 11) <= sys.version_info[:2] <= (3, 13), "needs Python 3.11 to 3.13"
print("nori-rel", version("nori-rel"))
try:
    print("synthefy-nori", version("synthefy-nori"))  # local extra
except Exception:
    print("synthefy-nori not installed: hosted mode only")
try:
    import torch

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("torch", torch.__version__, "device", device)
except ImportError:
    pass
client = NoriRelClient()  # or NoriRelClient(mode="hosted")
print(client)  # mode and default model
client.close()
```

For hosted mode, confirm `SYNTHEFY_NORI_API_KEY` is set in the shell
(`test -n "$SYNTHEFY_NORI_API_KEY" && echo set`); never print or paste the key.

The first local prediction downloads the default model's weights into the
Hugging Face cache (`~/.cache/huggingface` unless `HF_HOME` is set). Run that
first prediction with network access; afterwards `HF_HUB_OFFLINE=1` keeps every
run offline. Nori-Rel never falls back to hosted Nori when weights are missing;
it raises an error that says how to recover.

## 4. Report

Tell the user the `nori-rel` version, the runtime version in local mode, the
mode, the default model the client printed, the device, and the extras
installed. Keep these in the project notes: a saved pipeline records the model
and runtime, so a later upgrade must be a deliberate step
([nori-rel-schedule](../nori-rel-schedule/SKILL.md)).

Then continue with [nori-rel-connect](../nori-rel-connect/SKILL.md), or
[nori-rel-prepare-data](../nori-rel-prepare-data/SKILL.md) for files.

## Problems

| Message | Cause and fix |
|---|---|
| `ImportError: Install a connector extra` | No connector installed. Install `nori-rel[database]`, `[snowflake]` or `[databricks]`. |
| `RuntimeError` naming the runtime version and the models it has | The installed `synthefy-nori` is older than the model the client asks for. Reinstall with the `local` extra of the current `nori-rel`; do not pin `synthefy-nori` yourself. |
| `ValueError: mode='hosted' requires api_key or SYNTHEFY_NORI_API_KEY` | Export the key in the shell that runs Python. |
| `ValueError: mode='local' does not use an API key` | Drop `api_key=` or pass `mode="hosted"`. |
| Weight download fails or `HF_HUB_OFFLINE` set with an empty cache | Run once with network access, then go offline. |
| `OutOfMemoryError` during a prediction | Lower `max_context_rows` in the task, or pass `device="cpu"`. |
| Python 3.10 or 3.14 | Unsupported. Create the environment with 3.11, 3.12 or 3.13. |

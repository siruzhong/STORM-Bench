# STORM THOR Pipeline

AI2-THOR construction code for simulated videos, rule-generated QA, and an optional VLM wording pass. Run the scripts from this checkout; inputs and outputs are external paths. This directory contains no dataset annotations, videos, or evaluation results.

## Installation

For simulation, use Linux x86_64, an NVIDIA GPU with Vulkan support, and Python 3.11. CPU-only QA processing and tests also run on macOS. From the repository root:

```bash
cd storm-sim/thor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
# Required only for simulation:
python -m pip install ./vendor/ai2thor
python scripts/check_environment.py
```

The vendored Python SDK retains its upstream Apache-2.0 license. Unity binaries and scene assets are downloaded separately on first use, or supplied with `STORM_THOR_EXECUTABLE`. See [NOTICE.md](NOTICE.md). The default Unity build is `4d2e1f1d04051fafcd9794b810f227551121253a`; platform and executable overrides are documented in [.env.example](.env.example).

## Run

Set an absolute output root outside the repository:

```bash
export STORM_DATA_ROOT=/absolute/path/outside/STORM-Bench
python scripts/rollout.py --output "$STORM_DATA_ROOT/thor/run01"
```

The default recipe creates 28 videos across four room categories and 300 four-choice questions. It exports `raw/`, `qa_reference/`, and `qa_rule/` under the selected output directory. The v2.5 recipe preserves this fixed size. For a small reference-profile run:

```bash
python scripts/rollout.py --profile reference \
  --episodes-per-room 1 --question-count 44 \
  --output "$STORM_DATA_ROOT/thor/small"
```

Existing accepted rollouts can be reused with repeated `--reuse-run ROOM=PATH` arguments. To rebuild QA from an existing reference dataset:

```bash
python scripts/export_qa.py \
  --source-dataset "$STORM_DATA_ROOT/thor/run01/qa_reference" \
  --output "$STORM_DATA_ROOT/thor/rebuilt_qa"
```

`scripts/polish_qa.py` connects to a user-configured vision-capable Chat Completions endpoint. Export `VLM_BASE_URL`, `VLM_API_KEY`, and `VLM_MODEL` using [.env.example](.env.example). The scripts do not load `.env` automatically. Run `python scripts/polish_qa.py --help` for input/output and sampling options. The API pass edits question/evidence wording while preserving options and labels; it does not replace the rule-based answer generator.

The script loads [the wording prompt](src/tools/storm/prompts/qa_polish.txt) from the package. It asks for a natural question with the same object reference, evidence scope, and answer meaning, and retains the original text when an edit would require an unsupported inference. The response schema allows only the two text fields and a review decision. The prompt is included in the run settings, so a changed prompt cannot silently reuse an earlier wording checkpoint.

```bash
python scripts/polish_qa.py \
  --input "$STORM_DATA_ROOT/thor/run01/qa_rule" \
  --output "$STORM_DATA_ROOT/thor/wording_preview" \
  --dry-run --limit 2
```

For an API run, use a new output directory and omit `--dry-run` after exporting the provider settings.

For the detailed capture, reuse, polish, and validation commands, see [USAGE.md](docs/USAGE.md). [PIPELINE.md](docs/PIPELINE.md) describes the generation stages.

## Repository layout

```text
src/tools/storm/             # Planning, simulation adapters, QA, and export
scripts/                    # Rollout, QA export, VLM polish, and validation
configs/                    # Simulation and four-room QA recipes
tests/                      # Offline QA and integration checks
vendor/ai2thor/              # Python SDK source and upstream license
docs/                       # Backend documentation
pyproject.toml              # Editable install and offline test configuration
```

The `tools.storm` import namespace is retained for compatibility with the tested generator. Source code lives under `src/`, and regression tests live under `tests/`, matching the construction-package organization of `storm-real`.

## Offline checks

```bash
python -m pytest -q
```

These tests do not render a complete dataset. To verify a simulator installation separately, use `python scripts/check_environment.py --render --output "$STORM_DATA_ROOT/thor/environment.png"`.

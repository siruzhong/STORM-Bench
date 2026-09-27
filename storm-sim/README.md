# STORM Simulation Pipelines

Construction code for the two STORM-Sim backends: **THOR** (AI2-THOR) and **VHome** (VirtualHome). Each backend generates videos, records event evidence, builds QA, and exports annotations for evaluation. The two implementations have separate simulator dependencies and can be installed independently.

This directory follows the construction-code organization of [`storm-real`](../storm-real/README.md): backend packages contain `src/`, `tests/`, `pyproject.toml`, and usage documentation. Simulation-specific scripts, configurations, and camera patch sources remain beside their corresponding package.

**Datasets are external inputs and outputs.** Do not put videos, generated QA, capture records, reviewed evidence, model checkpoints, or evaluation results in this directory. All example paths below refer to storage outside the repository. The ignore rules also exclude common data, output, runtime, media, and annotation paths.

```text
Simulator and scene configuration
    -> Planned events and observer motion
    -> Capture with synchronized event and visibility records
    -> Evidence-grounded QA candidates and selection
    -> Wording revision and export validation
    -> External video and annotation directories
```

## Backends

| Backend | Code directory | Default release recipe | QA workflow |
| --- | --- | --- | --- |
| THOR | [`thor/`](thor/README.md) | Four room categories, 28 videos, 300 questions | Rule-generated answers and evidence; optional user-configured VLM wording pass |
| VHome | [`vhome/`](vhome/README.md) | 29 room instances, 29 videos, 290 questions | Visibility evidence plus reviewed physical-state endpoints; quota selection and deterministic wording revision |

These are backend-specific recipes, not a claim that every capture attempt succeeds. The VHome QA profile expects ten questions per episode and is calibrated for 29 episodes. Both current release recipes use four answer options; earlier experimental VHome mixed-arity recipes are separate tools.

## Repository layout

```text
storm-sim/
  README.md
  .gitignore
  thor/
    README.md
    pyproject.toml
    .env.example
    src/tools/storm/             # THOR planning, capture, QA, and export
      prompts/qa_polish.txt      # VLM wording prompt
    scripts/                    # Backend command-line entry points
    configs/                    # Simulator and QA recipes
    tests/                      # Offline tests
    vendor/ai2thor/              # SDK source; no Unity binaries
    docs/
  vhome/
    README.md
    pyproject.toml
    src/storm_virtualhome/      # Scene planning, capture, and validation
      prompts/qa_polish.txt     # VLM wording prompt
    src/storm_virtualhome_qa/   # Current evidence-to-QA pipeline
    scripts/                   # Capture, preprocessing, and QA entry points
    configs/                   # Scene and observer recipes
    unity_debug/               # Camera and navigation patch source
    tests/                     # Offline tests
    docs/
```

`src/` contains reusable implementation, `tests/` contains offline checks, and each backend's `pyproject.toml` declares its dependencies. Unlike the real-video pipeline, simulated capture also needs a separately installed Unity runtime. Backend scripts should be run from their checkout directory.

## Installation

Use Python 3.10 or newer; Python 3.11 is recommended for the THOR simulator SDK. Linux with the relevant NVIDIA rendering drivers is needed for the documented simulation setup. QA-only processing runs on the CPU and does not need Unity or a GPU.

Install only the backend you need, using a separate environment for each:

```bash
# From the STORM-Bench repository root:
cd storm-sim/thor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip install ./vendor/ai2thor  # Needed for THOR simulation
```

```bash
# From the STORM-Bench repository root, in a separate shell:
cd storm-sim/vhome
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[qa,test]'
```

For VHome rendering, follow the runtime, Xorg, and camera-patch setup in [vhome/README.md](vhome/README.md). The current multi-room capture workflow is documented in [ROOM_DATASET.md](vhome/docs/ROOM_DATASET.md). The package contains source code only; simulator binaries are obtained separately.

## THOR: generate videos and QA

From `storm-sim/thor`, after installing the simulator dependencies:

```bash
export STORM_DATA_ROOT=/absolute/path/outside/STORM-Bench
python scripts/rollout.py --output "$STORM_DATA_ROOT/thor/run01"
```

The output directory must be new. The default run produces raw captures, reference QA, and rule-generated QA under that directory. Existing accepted captures can be reused without rendering them again. See [thor/README.md](thor/README.md) and its linked usage guide for reuse and optional API polishing.

## VHome: generate QA from captured evidence

From `storm-sim/vhome`, supply the evidence and review files from your own rollout:

```bash
export STORM_DATA_ROOT=/absolute/path/outside/STORM-Bench
storm-virtualhome-qa run \
  --evidence "$STORM_DATA_ROOT/vhome/inputs/retimed_evidence_bundle.json" \
  --visibility "$STORM_DATA_ROOT/vhome/inputs/visibility_audit.json" \
  --review "$STORM_DATA_ROOT/vhome/inputs/physical_endpoint_review.json" \
  --reference "$STORM_DATA_ROOT/thor/reference/questions.jsonl" \
  --work "$STORM_DATA_ROOT/vhome/qa_work" \
  --output "$STORM_DATA_ROOT/vhome/qa_dataset"

storm-virtualhome-qa validate "$STORM_DATA_ROOT/vhome/qa_dataset"
```

`python -m storm_virtualhome_qa` is equivalent to `storm-virtualhome-qa`. `run` builds candidates, selects the quota-constrained set, balances answer positions, revises wording, and validates the exported contract. It does not render videos or call a model API. The work and output directories must be new and separate.

The endpoint review file is an external input: known physical-state answers require visually reviewed evidence. Graph state is not a substitute for visible evidence. See [QA_GENERATION.md](vhome/docs/QA_GENERATION.md) for `collect`, `visibility`, `retime`, `generate`, `polish`, and `validate`, and [INPUTS.md](vhome/docs/INPUTS.md) for their input schemas. An optional user-configured VLM script remains available for raw episode QA; it is separate from the current deterministic release-wording stage.

## Wording prompts

The API scripts load their English prompts from packaged text files:

| Backend | Prompt | Editable response fields |
| --- | --- | --- |
| THOR | [`qa_polish.txt`](thor/src/tools/storm/prompts/qa_polish.txt) | `question` and the private `video_evidence` description; `decision` records whether the text was revised, retained, or flagged |
| VHome | [`qa_polish.txt`](vhome/src/storm_virtualhome/prompts/qa_polish.txt) | `question` only |

Each prompt asks for natural phrasing while preserving the original observation task, object references, temporal scope, and answer meaning. Sparse prefix frames provide context for the edit. When a wording change would require resolving an unsupported fact, the original text is retained. Questions about visible outcomes remain distinct from questions about actions that happened out of view.

The prompts use short section headings with paragraph-based guidance on the input, evidence, temporal references, natural phrasing, and preservation of the observation task. Editing examples show both a suitable revision and a nearby alternative that would change the meaning. A final check and an explicit JSON response structure connect these language instructions to the fields accepted by each script.

Candidate selection precedes this step. The wording scripts retain answer choices, their order, answer indices, query times, and evidence labels. THOR also records review decisions; VHome can optionally run the meaning-preservation review described in its backend guide. The prompts are included in source and wheel distributions, so the wording instructions are available alongside the code. `--dry-run` exports the actual requests to an external output directory without calling an API.

## External dataset layout

Generated release directories use the following shared organization. The exact supporting metadata differs between backends:

```text
/external/storage/dataset/
  questions.jsonl
  model_inputs.jsonl
  evaluation_labels.json
  evaluation_splits.json
  evaluation_protocol.json
  meta_data/
    qa_results/<scene>/<episode>.json
    gene_videos/<scene>/<episode>.mp4
```

The current VHome exporter also writes `evaluation_gt.jsonl`, `video_manifest.json`, distribution reports, and a `private/` provenance directory. It references videos through relative paths and does not copy video files. Materialize the matching videos in the external dataset directory before evaluation. The top-level benchmark evaluator lives under [`scripts/eval/`](../scripts/eval/README.md); its runs and outputs are separate from dataset construction.

## QA fields and evidence

Both backends use the six STORM semantic categories: `factual_retrieval`, `current_state`, `state_change`, `object_tracking`, `temporal_reasoning`, and `history_aggregation`.

The core annotations include query time, question type/subtype, question text, answer options, zero-based answer index, evidence spans, video-evidence text, epistemic status, uncertainty sources, diagnostic rationale, and change intensity. Known and uncertain labels describe the evidence available for an answer; they are independent of question type.

Ground-truth annotations contain answers and private evidence. Answering models should receive only the approved model-input fields and the appropriate clean video context. Event logs, simulator graphs, debug overlays, and answer labels are not model inputs.

## Offline checks

Run the suites separately because each backend has its own package and test configuration:

```bash
# From storm-sim/thor, in its environment:
python -m pytest -q

# From storm-sim/vhome, in its environment:
python -m pytest -q
```

The offline checks use synthetic fixtures and mocked services. They test code behavior without generating a real dataset or calling paid model APIs. A successful test suite does not establish that a new simulator setup or an unreviewed capture is ready for benchmark release.

# VirtualHome QA generator

The `storm_virtualhome_qa` package contains code for generating QA from recorded VirtualHome rollouts. It contains candidate generation, evidence-based selection, answer-position balancing, wording revision, export, and contract validation. It does not run the simulator or call a language-model API. All processing uses the CPU.

The default profile produces 290 questions from 29 episodes, with ten questions per episode. This package preserves the current evaluated release's four-choice policy: every question has three concrete alternatives and one uncertainty option. It does not restore the older mixed-arity policy.

## Install

Use Python 3.10 or newer. From the `storm-sim/vhome` directory:

```bash
python3 -m venv /tmp/virtualhome-qa-env
source /tmp/virtualhome-qa-env/bin/activate
python -m pip install -e '.[qa,test]'
storm-virtualhome-qa --help
```

Dependencies are NumPy, SciPy, and Pillow. No CUDA, Unity installation, model checkpoints, credentials, or network access are needed after installation. Keep virtual environments and generated data outside this code directory.

## Generate a complete release

Supply existing rollout evidence, visibility counts, reviewed physical-state endpoints, and reference AI2THOR annotations. Paths below are placeholders for your own files. Both output directories must be new and must not contain one another.

```bash
storm-virtualhome-qa run \
  --evidence /data/inputs/retimed_evidence_bundle.json \
  --visibility /data/inputs/visibility_audit.json \
  --review /data/inputs/physical_endpoint_review.json \
  --reference /data/inputs/ai2thor_questions.jsonl \
  --work /data/outputs/qa_work \
  --output /data/outputs/qa_dataset \
  --time-limit 90

storm-virtualhome-qa validate /data/outputs/qa_dataset
```

The equivalent Bash entry point is:

```bash
bash scripts/generate_qa.sh \
  /data/inputs/retimed_evidence_bundle.json \
  /data/inputs/visibility_audit.json \
  /data/inputs/physical_endpoint_review.json \
  /data/inputs/ai2thor_questions.jsonl \
  /data/outputs/qa_work \
  /data/outputs/qa_dataset
```

`PYTHON` selects the interpreter for the Bash script. `QA_SOLVER_SECONDS` changes the selection time limit. Option-position balancing has a separate 30-second limit. A solver time limit is not a total pipeline deadline. If no feasible selection is found, the command fails without substituting unsupported questions; use a new work/output directory for a retry.

The intermediate directory retains `selection/candidates.jsonl`, `selection/availability.json`, `selection/selected.jsonl`, and `raw/`. These are generated data, not part of this source distribution.

## Build evidence from captures

Skip these commands if you already have the matching evidence bundle and visibility audit. See [INPUTS.md](INPUTS.md) for the capture contract.

```bash
storm-virtualhome-qa collect \
  --source /data/rollout \
  --reference /data/inputs/ai2thor_questions.jsonl \
  --output /data/evidence/evidence_bundle.json

storm-virtualhome-qa visibility \
  --evidence /data/evidence/evidence_bundle.json \
  --output /data/evidence/visibility_audit.json \
  --workers 8

storm-virtualhome-qa retime /data/evidence
```

The collector reads capture paths from the rollout manifest. Relative paths resolve against `--source`. For relocated captures, `--capture-root /data/captures` maps each episode to `/data/captures/episode_id`. Visibility auditing reads every native-frame instance mask; worker count controls CPU and memory use. It does not decode RGB videos.

`retime` preserves the historical event-discovery correction and writes `retimed_evidence_bundle.json` and `timeline_corrections.json`. Its 12/40-pixel thresholds adjust endpoint timestamps; QA generation independently requires 128 pixels and uses a 1 FPS evidence clock. Mask visibility alone cannot establish a physical state: known physical questions require the external endpoint review.

## Run stages separately

`generate` accepts the same seven file/directory flags as `run`, but requires an explicit `--quotas` file and exports raw wording. The bundled profile is `src/storm_virtualhome_qa/qa290.json`.

To revise a previously exported raw dataset without selecting new questions:

```bash
storm-virtualhome-qa polish \
  --source /data/outputs/raw_dataset \
  --output /data/outputs/revised_dataset
storm-virtualhome-qa validate /data/outputs/revised_dataset
```

This stage preserves IDs, answer positions, evidence, query times, and question categories. It applies the existing English wording rules and adds the shared evidence scope to evaluator prompts. It uses no API. `validate` checks the final revised contract; raw exports are validated internally by `generate`.

## Output layout

```text
qa_dataset/
  questions.jsonl
  evaluation_gt.jsonl
  model_inputs.jsonl
  evaluation_labels.json
  evaluation_splits.json
  video_manifest.json
  distribution_alignment.json
  qa_revision_manifest.json
  evaluation_protocol.json
  validation.json
  summary.json
  release_status.json
  ALL_QA.md
  EPISODE_SAMPLE.json
  QA_REBUILD_REPORT.md
  meta_data/qa_results/SceneN_RoomM/episode_id.json
  private/qa_provenance.jsonl
  private/physical_endpoint_review.json
  private/qa_wording_changes.jsonl
```

Video paths follow `meta_data/gene_videos/SceneN_RoomM/episode_id.mp4`, relative to the dataset root. Video files are not copied or generated. Place or link the matching videos there before evaluation and verify them against `video_manifest.json`.

`questions.jsonl` and `evaluation_gt.jsonl` contain ground-truth answers. Only `model_inputs.jsonl` is designed to be passed to a model without labels or provenance. A custom evaluator must construct prompts from its allowlisted fields rather than stringify an entire GT record.

## Selection rules and scope

The bundled profile contains type and uncertainty counts, not questions or answers. It matches the current 290-question release: 237 known answers and 53 uncertain answers. Its uncertainty option appears in every question, so option presence does not directly identify an uncertain label.

Candidates use two consecutive one-second samples to confirm a visibility change. A target is visible at 128 or more pixels, absent at zero pixels, and otherwise ambiguous. A fully hidden interval must have zero target pixels in every native frame, including frames between one-second samples. Physical state answers come from approved RGB endpoint reviews. Simulation graph state alone is not sufficient.

Selection chooses ten existing candidates per episode, with at most three of one question type and one or two uncertain questions. Type/status counts, repeated-fact limits, and physical/visibility/count diversity bounds determine which candidates are included; each selected item retains its evidence and semantic answer. The settings in `generation.select` were calibrated for 29 episodes. For a different dataset size, configure the quota file and the selection bounds together.

A fresh optimization can select a different valid set because solver versions and time limits affect the solution. Archive the evidence inputs, selected rows, output annotations, and software versions with an experiment. To preserve a frozen release exactly, start from its raw annotation export and run `polish`.

The validator checks IDs, mirrors, labels, model-input separation, causal timestamps, option uniqueness, and answer-position balance. It does not certify RGB semantic correctness or verify actual video files. Existing wording and QA methodology are retained rather than replaced with a new dataset design.

## Tests

```bash
python -m pytest -q
```

Tests cover future-frame isolation, native-frame hidden intervals, answer balancing, wording/label preservation, overwrite protection, and rejection of label fields in model inputs. Test fixtures are synthesized in Python; no real dataset, videos, review images, evaluation outputs, or API keys are included.

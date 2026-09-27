# Align QA with a four-choice reference

For the current sampled-visibility, known/uncertain export and AI2THOR directory
contract, use [Reference-aligned QA export](REFERENCE_ALIGNMENT.md). The workflow
below is the retained physical-state, known-only version used in earlier audits.

This pass rebuilds questions from captured events. It leaves the videos and raw QA intact and does not use evaluation predictions. It supports references whose questions all have four choices. It is not a wording-only operation: question IDs, types, answers, and observation timestamps can change.

Install the optional allocation dependency:

```bash
pip install -e '.[qa]'
```

Use a capture snapshot containing `manifest.json` and the original episode evidence. The reference is a JSONL file with `question_type` and `options` fields.

```bash
python scripts/collect_qa_evidence.py CAPTURE_SNAPSHOT REFERENCE.jsonl RUN/evidence_bundle.json
python scripts/audit_qa_visibility.py RUN/evidence_bundle.json RUN/visibility_audit.json
python scripts/retime_qa_evidence.py RUN
python scripts/align_qa_distribution.py RUN/retimed_evidence_bundle.json RUN/polish
```

The collector verifies state labels against saved scene graphs. The visibility audit measures every event target in all recorded segmentation frames, with one CPU worker per room, up to 29 workers. Retiming finds visible frames within stable state intervals, excluding the operation animation. This workflow currently follows the capture configuration used by the 29-room batch: 40 pixels for general objects and 12 for light switches. Check these thresholds against a new capture configuration before using it for another batch.

Each room supplies ten questions. Six question-type quotas follow the reference proportions using largest remainders. Selection enforces one or two questions of each type per room, limits duplicate facts, and balances the four semantic answer groups within each type to within one question. If those constraints cannot be met, the allocation fails rather than padding binary questions with invalid choices.

Four-choice domains cover paired observed states, paired earlier states, object-specific transition descriptions, identity and state after an offscreen change, ordered object pairs, and confirmed-change counts. Unknown graph states are excluded. This version is a known-answer subset and does not measure uncertainty recognition.

Outputs include `polished_questions.jsonl`, `alignment_validation.json`, candidate availability, and per-question provenance. The generator refuses to replace a dataset once `RUN/data/qa_results/questions.jsonl` exists. Freeze the questions before evaluation and keep model inputs separate from private answers, graphs, segmentation masks, and provenance.

An instance mask establishes object visibility, not whether a state is visually readable. Review RGB evidence before treating the dataset as a fully validated benchmark. Fixed-rate evaluation can also miss brief evidence; report sampling coverage and do not equate a 25% uniform-guess baseline with an expected 25% text-only model score.

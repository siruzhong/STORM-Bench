# Natural-choice QA

The `natural_structure_v2` schema supports two, three, or four choices. Uniform
random accuracy is `1 / num_choices` for each question. Report the mean of these
values for the evaluated subset, together with accuracy and the accuracy-minus-
chance gap. A mixed-choice dataset does not have a universal 25% baseline.

## Rebuild an existing capture

```bash
python scripts/rebuild_batch_qa.py \
  --input outputs/batch_30_v1 \
  --output outputs/batch_30_v1/structured_qa \
  --render-prefixes
```

The source videos, original QA, and previously measured results are retained.
The output contains `qa_private.jsonl`, `model_inputs.jsonl`,
`evaluation_labels.jsonl`, and `audit.json`. Resolve video paths against the
original batch directory. Only the public model-input fields belong in a model
request; do not include answers, evidence intervals, generation keys, or labels.
Question IDs are opaque with respect to the answer and observation status.

New batch runs produce the same QA directory after capture completion and expose
it through `recommended_qa` in `status.json`, including the required prefixes.
Capture completion does not imply
that QA has passed semantic review or is a release-ready benchmark.

## Question rules

- Binary states use their two physical values. Unsupported destruction or
  disappearance options are not added to make four choices.
- Current-state questions always include the same uncertainty option for the
  same physical domain, including when the state is known. Hidden queries and
  post-discovery queries are separate candidates with separate causal prefixes.
- Change-direction and ordering questions use the two directions or states
  specified by the task. They do not pretend to test whether an event occurred.
- Object tracking uses unique descriptions of objects with matching operation
  capabilities. A single eligible object does not yield a multiple-choice item.
- Historical aggregation compares the cumulative confirmed changes of two objects,
  including a tie option. Temporal reasoning instead compares the first confirmed
  changes of two distinct objects; equal discovery times are excluded. Neither task
  is a reformulation of recalling the previous state of one object.
- Public questions use observable changes and the end of the prefix as anchors.
  Absolute seconds remain in private evidence metadata, not in question text.
- Answer domains without examples of every answer are excluded from batch
  selection. Within each question-type/option-set cell, selected answer counts
  differ by at most one. This defines a balanced development distribution rather
  than the natural frequency of household events.
- Selection uses annotations only, never a target model's predictions. A bounded
  deterministic allocation search changes episode order when a greedy allocation
  cannot satisfy answer balance; it never weakens the balance constraint. There are
  ten selected questions per video. Six types have equal totals for batches whose
  size is a multiple of three. Questions need not correspond one-to-one to events.

## Evaluation contract

Use a generic instruction such as: "Answer the multiple-choice question using
the supplied evidence. Output only the letter of one of the supplied options."
Do not tell the model that every question has four choices. Restrict parsing to
that row's actual option count. Shuffle options reproducibly and preserve the
permutation when scoring. Keep invalid predictions in the denominator.

Report accuracy and uniform chance by arity and QA type, the overall weighted
chance, and a paired video-minus-text comparison. Bootstrap at the video level.
The in-sample option-set-majority score in `audit.json` is a shortcut diagnostic,
not a held-out baseline. A text-only model can exploit residual priors and need
not score exactly at chance.

Do not select questions to force a model to a requested accuracy. Freeze the
candidate metadata before evaluation. Further changes prompted by model results
belong to development; final testing requires unseen scenes/program families,
independent model checks, and visual evidence review.

## Remaining limits of the existing 30 captures

Rebuilding QA does not change the original five-object paired-event programs,
fixed opening action, or action/visibility correlations. The existing batch is a
development candidate. Timestamp and silhouette checks do not prove that a state
change is legible after sampling. Review semantic evidence, use wrong-video and
temporal controls where appropriate, and reroll more varied programs before
claiming a final visual-reasoning benchmark.

The measured `natural_choices_v1` candidate and its scores are retained separately.
Scores from that schema do not apply to `natural_structure_v2`.
